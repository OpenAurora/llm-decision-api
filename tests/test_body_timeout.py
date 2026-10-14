import asyncio
import json

import httpx
import pytest
from conftest import model_response

from llm_decision_api.api import create_app
from llm_decision_api.backend import ModelBackend
from llm_decision_api.config import Settings


@pytest.fixture
def app_factory(settings):
    def build(handler, **overrides):
        config = Settings(_env_file=None, **(settings.model_dump() | overrides))
        return create_app(
            config,
            backend_factory=lambda cfg: ModelBackend(cfg, transport=httpx.MockTransport(handler)),
        )

    return build


@pytest.mark.parametrize("upload_mode", ["no_data", "stalled", "trickle"])
def test_body_timeout_stops_upload_before_authentication_or_inference(app_factory, upload_mode):
    async def run():
        calls = []
        upload_stopped = asyncio.Event()

        def handler(request):
            calls.append(request)
            return httpx.Response(200, json=model_response())

        async def upload():
            try:
                if upload_mode == "no_data":
                    await asyncio.Event().wait()
                yield b"{"
                if upload_mode == "trickle":
                    # Each gap is shorter than the deadline, but the whole upload is not.
                    while True:
                        await asyncio.sleep(0.01)
                        yield b" "
                await asyncio.Event().wait()
            finally:
                upload_stopped.set()

        app = app_factory(handler, body_timeout_seconds=0.15, service_api_key="inbound-only")
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test", trust_env=False
            ) as client:
                # Deliberately omit authorization: body reception still has a deadline.
                response = await asyncio.wait_for(
                    client.post("/v1/decide", content=upload(), headers={"content-type": "application/json"}),
                    2,
                )
                assert response.status_code == 408, response.text
                error = response.json()["error"]
                assert error["code"] == "request_body_timeout"
                assert error["request_id"] == response.headers["x-request-id"]
                assert "question_id" not in error
                assert response.headers["connection"] == "close"
                assert upload_stopped.is_set()
                assert not calls
                assert (await client.get("/docs")).status_code == 200

    asyncio.run(run())


@pytest.mark.parametrize(
    "upload_delay, model_delay, body_timeout, decision_timeout",
    [(0.2, 0, 1, 0.1), (0, 0.2, 0.1, 1)],
)
def test_normal_upload_and_inference_have_independent_deadlines(
    app_factory, body, upload_delay, model_delay, body_timeout, decision_timeout
):
    async def run():
        captured = []

        async def handler(request):
            captured.append(json.loads(json.loads(request.content)["messages"][1]["content"]))
            await asyncio.sleep(model_delay)
            return httpx.Response(200, json=model_response())

        async def upload():
            raw = json.dumps(body).encode()
            yield raw[:10]
            await asyncio.sleep(upload_delay)
            yield raw[10:]

        app = app_factory(
            handler, body_timeout_seconds=body_timeout, timeout_seconds=decision_timeout, service_api_key=None
        )
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test", trust_env=False
            ) as client:
                response = await asyncio.wait_for(
                    client.post("/v1/decide", content=upload(), headers={"content-type": "application/json"}),
                    3,
                )
                assert response.status_code == 200, response.text
                assert response.json()["answers"]["department"]["choice"] == "refund"
                assert "connection" not in response.headers
                assert len(captured) == 1
                assert captured[0]["state"] == body["state"]

    asyncio.run(run())


def test_streamed_body_still_enforces_total_size_limit(app_factory):
    async def run():
        calls = []
        chunks_sent = 0

        def handler(request):
            calls.append(request)
            return httpx.Response(200, json=model_response())

        async def upload():
            nonlocal chunks_sent
            for _ in range(3):
                chunks_sent += 1
                yield b" " * 600

        app = app_factory(handler, max_body_bytes=1024, body_timeout_seconds=1)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test", trust_env=False
            ) as client:
                response = await asyncio.wait_for(
                    client.post("/v1/decide", content=upload(), headers={"content-type": "application/json"}),
                    2,
                )
                assert response.status_code == 413, response.text
                assert response.json()["error"]["code"] == "body_too_large"
                assert chunks_sent == 2
                assert not calls

    asyncio.run(run())


def test_body_timeout_environment_setting_is_independent(monkeypatch):
    monkeypatch.delenv("LLM_DECISION_BODY_TIMEOUT_SECONDS", raising=False)
    config = Settings(_env_file=None, api_key="synthetic-key", model="test-model", timeout_seconds=2)
    assert config.body_timeout_seconds == 10
    monkeypatch.setenv("LLM_DECISION_BODY_TIMEOUT_SECONDS", "0.25")
    config = Settings(_env_file=None, api_key="synthetic-key", model="test-model", timeout_seconds=2)
    assert config.body_timeout_seconds == 0.25
    assert config.timeout_seconds == 2


def test_openapi_documents_body_timeout_error(client_factory):
    with client_factory() as client:
        schema = client.app.openapi()
    response = schema["paths"]["/v1/decide"]["post"]["responses"]["408"]
    assert response["content"]["application/json"]["schema"]["$ref"] == "#/components/schemas/ErrorResponse"
