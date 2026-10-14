"""Versioned HTTP boundary: strict JSON, documented schemas and safe errors."""

import asyncio
import hmac
import json
import logging
import math
import time
from contextlib import asynccontextmanager
from traceback import walk_tb
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.routing import APIRoute
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.requests import ClientDisconnect

from . import __version__
from .backend import ModelBackend
from .config import Settings
from .documentation import render_guide
from .errors import ErrorDetail, ErrorResponse, ServiceError
from .schemas import DecisionResponse, request_model
from .service import DecisionService

logger = logging.getLogger("llm_decision_api")


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate_json_key")
        result[key] = value
    return result


def _constant(_):
    raise ValueError("non_finite_json_number")


def _float(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("non_finite_json_number")
    return number


class LimitedJSONRequest(Request):
    async def body(self):
        if not hasattr(self, "_body"):
            settings = self.app.state.settings

            async def receive_body():
                chunks = []
                size = 0
                async for chunk in self.stream():
                    size += len(chunk)
                    if size > settings.max_body_bytes:
                        raise HTTPException(413)
                    chunks.append(chunk)
                return b"".join(chunks)

            try:
                # One budget for the whole upload, including waiting for its first chunk.
                self._body = await asyncio.wait_for(receive_body(), settings.body_timeout_seconds)
            except asyncio.TimeoutError:
                # FastAPI preserves HTTPException during body parsing; other errors become 400.
                raise HTTPException(408) from None
        return self._body

    async def json(self):
        if not hasattr(self, "_json"):
            try:
                self._json = json.loads(
                    await self.body(),
                    object_pairs_hook=_object,
                    parse_constant=_constant,
                    parse_float=_float,
                )
            except (ValueError, UnicodeError, RecursionError):
                raise HTTPException(400) from None
        return self._json


class JSONRoute(APIRoute):
    def get_route_handler(self):
        original = super().get_route_handler()

        async def handle(request):
            if self.body_field is not None:
                media_type = request.headers.get("content-type", "").partition(";")[0].strip().lower()
                if not (
                    media_type == "application/json"
                    or (media_type.startswith("application/") and media_type.endswith("+json"))
                ):
                    raise HTTPException(415)
            return await original(LimitedJSONRequest(request.scope, request.receive))

        return handle


class RequestBoundaryMiddleware:
    def __init__(self, app, on_error):
        self.app = app
        self.on_error = on_error

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        request_id = uuid4().hex
        scope.setdefault("state", {})["request_id"] = request_id
        started = time.perf_counter()
        status = 500
        response_started = False

        async def respond(message):
            nonlocal status, response_started
            if message["type"] == "http.response.start":
                response_started = True
                status = message["status"]
                headers = [
                    (key, value)
                    for key, value in message.get("headers", [])
                    if key.lower() != b"x-request-id"
                ]
                message["headers"] = [*headers, (b"x-request-id", request_id.encode())]
            await send(message)

        try:
            await self.app(scope, receive, respond)
        except Exception as exc:
            # A framework Exception handler still re-raises the original to the
            # server. Handle it here so exception text cannot reach server logs.
            response = await self.on_error(Request(scope, receive=receive), exc)
            if response_started:
                # Headers cannot be replaced; let the server close the connection
                # with a fixed error that suppresses the original exception chain.
                raise RuntimeError("HTTP response failed after it started.") from None
            await response(scope, receive, respond)
        finally:
            route = scope.get("route")
            logger.info(
                json.dumps(
                    {
                        "event": "http_request",
                        "request_id": request_id,
                        "method": scope["method"],
                        "route": getattr(route, "path", "unmatched"),
                        "status": status,
                        "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                    }
                )
            )


async def evaluate_connected(request, service, body):
    async def disconnected():
        while True:
            if (await request.receive())["type"] == "http.disconnect":
                return

    evaluation = asyncio.create_task(service.evaluate(body))
    connection = asyncio.create_task(disconnected())
    try:
        done, _ = await asyncio.wait((evaluation, connection), return_when=asyncio.FIRST_COMPLETED)
        if evaluation in done:
            return await evaluation
        raise ServiceError(499, "client_disconnected", "客户端连接已断开。")
    finally:
        for task in (evaluation, connection):
            if not task.done():
                task.cancel()
        await asyncio.gather(evaluation, connection, return_exceptions=True)


def create_app(settings: Settings | None = None, *, backend_factory=ModelBackend) -> FastAPI:
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app):
        service = DecisionService(settings, backend_factory(settings))
        app.state.service = service
        try:
            if settings.verify_upstream_on_startup:
                try:
                    await service.verify_backend()
                except ServiceError as exc:
                    raise RuntimeError(f"Model startup check failed: {exc.code}: {exc.message}") from None
            logger.info(
                json.dumps(
                    {
                        "event": "service_ready",
                        "model": settings.model,
                        "max_options": settings.max_options,
                        "upstream_verified": service.verified,
                    }
                )
            )
            yield
        finally:
            await service.close()

    app = FastAPI(
        title="LLM Decision API",
        version=__version__,
        lifespan=lifespan,
        description=(
            "基于大模型候选概率的语义判断 API。请求体仅包含 state 和 questions。"
            "候选概率为当前候选集合内的相对分布，未经正确率校准。"
            "Choice/Score 返回按 TypeSafe 官方适配器公式计算的 confidence；"
            "Noul 仅返回成立概率。置信度不代表实际判断正确率。"
        ),
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.state.settings = settings
    app.router.route_class = JSONRoute

    def error_response(request, exc):
        envelope = ErrorResponse(
            error=ErrorDetail(
                code=exc.code,
                message=exc.message,
                request_id=getattr(request.state, "request_id", uuid4().hex),
                question_id=exc.question_id,
                details=exc.details,
            )
        )
        headers = {"X-Request-ID": envelope.error.request_id}
        if exc.status == 503:
            headers["Retry-After"] = "1"
        if exc.status == 401:
            headers["WWW-Authenticate"] = "Bearer"
        if exc.status == 408 and request.scope.get("http_version") in ("1.0", "1.1"):
            headers["Connection"] = "close"
        return JSONResponse(envelope.model_dump(exclude_none=True), status_code=exc.status, headers=headers)

    @app.exception_handler(ServiceError)
    async def service_error(request, exc):
        return error_response(request, exc)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        def safe_text(value):
            return (
                value.encode("utf-8", errors="replace").decode("utf-8") if isinstance(value, str) else value
            )

        details = [
            {
                "path": [safe_text(part) for part in item["loc"]],
                "type": item["type"],
                "message": safe_text(item["msg"]),
            }
            for item in exc.errors()
        ]
        return error_response(
            request, ServiceError(422, "invalid_request", "请求字段校验失败。", details=details)
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request, exc):
        known = {
            400: ("invalid_json", "请求体必须是有效 JSON，且不能包含重复键或非有限数值。"),
            408: ("request_body_timeout", "请求体接收超过允许的时限。"),
            413: ("body_too_large", "请求体超过服务允许的大小。"),
            404: ("not_found", "请求的接口不存在。"),
            405: ("method_not_allowed", "该接口不支持此 HTTP 方法。"),
            415: ("unsupported_media_type", "请使用 application/json 请求体。"),
        }
        code, message = known.get(exc.status_code, ("http_error", "HTTP 请求失败。"))
        response = error_response(request, ServiceError(exc.status_code, code, message))
        if exc.status_code == 405:
            for name, value in (exc.headers or {}).items():
                if name.lower() == "allow":
                    response.headers["Allow"] = value
        return response

    @app.exception_handler(ClientDisconnect)
    async def client_disconnect(request, exc):
        return error_response(request, ServiceError(499, "client_disconnected", "客户端连接已断开。"))

    async def unexpected_error(request, exc):
        logger.error(
            json.dumps(
                {
                    "event": "internal_error",
                    "request_id": getattr(request.state, "request_id", None),
                    "exception_type": type(exc).__name__,
                    "stack": [
                        {
                            "file": frame.f_code.co_filename,
                            "line": line,
                            "function": frame.f_code.co_name,
                        }
                        for frame, line in walk_tb(exc.__traceback__)
                    ],
                }
            )
        )
        return error_response(request, ServiceError(500, "internal_error", "服务内部错误。"))

    app.add_middleware(RequestBoundaryMiddleware, on_error=unexpected_error)

    bearer = HTTPBearer(auto_error=False)

    async def authenticate(credentials: HTTPAuthorizationCredentials | None = Depends(bearer)):
        if settings.service_api_key is None:
            return
        supplied = credentials.credentials if credentials else ""
        if not hmac.compare_digest(supplied.encode(), settings.service_api_key.get_secret_value().encode()):
            raise ServiceError(401, "unauthorized", "需要有效的服务访问密钥。")

    body_type = request_model(settings.max_options)
    responses = {
        status: {"model": ErrorResponse, "description": description}
        for status, description in {
            400: "无效 JSON",
            401: "认证失败",
            408: "请求体接收超时",
            413: "请求体过大",
            415: "不支持的媒体类型",
            422: "请求校验失败",
            500: "内部错误",
            502: "模型响应无效或调用失败",
            503: "服务暂不可用",
            504: "决策超时",
        }.items()
    }

    @app.post(
        "/v1/decide",
        response_model=DecisionResponse,
        responses=responses,
        dependencies=[Depends(authenticate)],
        operation_id="decide",
        summary="针对同一份资料独立回答一至三十二道问题",
    )
    async def decide(request: Request, body: body_type):
        return await evaluate_connected(request, request.app.state.service, body)

    @app.get("/docs", response_class=HTMLResponse, include_in_schema=False)
    async def usage_guide(request: Request):
        return HTMLResponse(render_guide(settings, str(request.base_url)))

    if settings.service_api_key is None:
        # An optional deployment key must not appear mandatory in OpenAPI.
        original_openapi = app.openapi

        def openapi():
            schema = original_openapi()
            schema["paths"]["/v1/decide"]["post"].pop("security", None)
            return schema

        app.openapi = openapi
    return app
