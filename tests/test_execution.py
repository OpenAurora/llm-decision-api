import asyncio
import json
import time

import httpx
import pytest
from conftest import model_response

from llm_decision_api import service as service_module
from llm_decision_api.api import create_app
from llm_decision_api.backend import ModelBackend
from llm_decision_api.config import Settings
from llm_decision_api.errors import ServiceError
from llm_decision_api.schemas import request_model
from llm_decision_api.service import DecisionService


def config(**overrides):
    return Settings(
        _env_file=None,
        **(
            {
                "api_key": "secret",
                "base_url": "http://model.invalid/v1",
                "model": "test-model",
                "verify_upstream_on_startup": False,
            }
            | overrides
        ),
    )


def request(count=1, instructions=None):
    return request_model(20).model_validate(
        {
            "state": "evidence",
            "questions": {
                f"q{i}": {
                    "type": "choice",
                    "instructions": instructions[i] if instructions else "question",
                    "criteria": {"yes": None, "no": None},
                }
                for i in range(count)
            },
        }
    )


def test_concurrent_batches_have_no_application_or_connection_pool_cap():
    async def run():
        batch_count = 6
        questions_per_batch = 32
        total = batch_count * questions_per_batch
        active = 0
        peak = 0
        entered = asyncio.Event()
        release = asyncio.Event()
        handlers = []
        response_body = json.dumps(model_response()).encode()

        async def handler(reader, writer):
            nonlocal active, peak
            handlers.append(asyncio.current_task())
            try:
                headers = await reader.readuntil(b"\r\n\r\n")
                length = next(
                    int(line.partition(b":")[2])
                    for line in headers.split(b"\r\n")
                    if line.lower().startswith(b"content-length:")
                )
                await reader.readexactly(length)
                active += 1
                peak = max(peak, active)
                if active == total:
                    entered.set()
                try:
                    # No upstream call finishes until every batch has arrived.
                    await release.wait()
                    writer.write(
                        b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                        + f"Content-Length: {len(response_body)}\r\n".encode()
                        + b"Connection: close\r\n\r\n"
                        + response_body
                    )
                    await writer.drain()
                finally:
                    active -= 1
            finally:
                writer.close()
                await writer.wait_closed()

        # Real loopback HTTP exercises the model client's connection pool as well
        # as API admission; MockTransport would bypass the connection pool.
        server = await asyncio.start_server(handler, "127.0.0.1", 0, backlog=256)
        try:
            port = server.sockets[0].getsockname()[1]
            app = create_app(config(base_url=f"http://127.0.0.1:{port}/v1"))
            async with app.router.lifespan_context(app):
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app), base_url="http://test", trust_env=False
                ) as client:
                    body = request(questions_per_batch).model_dump()
                    pending = [
                        asyncio.create_task(client.post("/v1/decide", json=body)) for _ in range(batch_count)
                    ]
                    try:
                        await asyncio.wait_for(entered.wait(), 5)
                        assert active == total
                        release.set()
                        results = await asyncio.gather(*pending)
                        assert all(result.status_code == 200 for result in results)
                        assert all(len(result.json()["answers"]) == questions_per_batch for result in results)
                        assert peak == total
                    finally:
                        release.set()
                        await asyncio.gather(*pending, return_exceptions=True)
        finally:
            release.set()
            server.close()
            await server.wait_closed()
            completed = await asyncio.gather(*handlers, return_exceptions=True)
        assert all(result is None for result in completed)

    asyncio.run(run())


def test_total_deadline_cancels_all_unfinished_model_calls():
    async def run():
        calls = 0
        cancelled = 0

        async def handler(_):
            nonlocal calls, cancelled
            calls += 1
            try:
                await asyncio.Event().wait()
            finally:
                cancelled += 1

        settings = config(timeout_seconds=0.02)
        service = DecisionService(settings, ModelBackend(settings, transport=httpx.MockTransport(handler)))
        try:
            with pytest.raises(ServiceError) as error:
                await service.evaluate(request(2))
            assert error.value.status == 504 and error.value.code == "decision_timeout"
            assert calls == 2 and cancelled == 2
        finally:
            await service.close()

    asyncio.run(run())


@pytest.mark.parametrize("stage, expected_calls", [("prepare_question", 0), ("to_answer", 1)])
def test_deadline_covers_input_preparation_and_answer_assembly(monkeypatch, stage, expected_calls):
    elapsed = 0.0
    original = getattr(service_module, stage)

    def overrun(*args, **kwargs):
        nonlocal elapsed
        result = original(*args, **kwargs)
        elapsed += 2
        return result

    monkeypatch.setattr(service_module, "monotonic", lambda: elapsed)
    monkeypatch.setattr(service_module, stage, overrun)

    async def run():
        calls = 0

        async def handler(_):
            nonlocal calls
            calls += 1
            return httpx.Response(200, json=model_response())

        settings = config(timeout_seconds=1)
        service = DecisionService(settings, ModelBackend(settings, transport=httpx.MockTransport(handler)))
        try:
            with pytest.raises(ServiceError) as error:
                await service.evaluate(request())
            assert error.value.status == 504 and error.value.code == "decision_timeout"
            assert calls == expected_calls
        finally:
            await service.close()

    asyncio.run(run())


def test_input_preparation_reduces_remaining_model_time(monkeypatch):
    preparation_time = 0.0
    original = service_module.prepare_question

    def prepare(*args, **kwargs):
        nonlocal preparation_time
        result = original(*args, **kwargs)
        preparation_time += 0.75
        return result

    monkeypatch.setattr(service_module, "monotonic", lambda: time.monotonic() + preparation_time)
    monkeypatch.setattr(service_module, "prepare_question", prepare)

    async def run():
        cancelled = False

        async def handler(_):
            nonlocal cancelled
            try:
                # Fits the full budget, but not the time left after preparation.
                await asyncio.sleep(0.5)
            except asyncio.CancelledError:
                cancelled = True
                raise
            return httpx.Response(200, json=model_response())

        settings = config(timeout_seconds=1)
        service = DecisionService(settings, ModelBackend(settings, transport=httpx.MockTransport(handler)))
        try:
            with pytest.raises(ServiceError) as error:
                await service.evaluate(request())
            assert error.value.status == 504 and error.value.code == "decision_timeout"
            assert cancelled
        finally:
            await service.close()

    asyncio.run(run())


def test_first_upstream_failure_cancels_other_questions():
    async def run():
        entered = asyncio.Event()
        cancelled = asyncio.Event()

        async def handler(req):
            content = json.loads(json.loads(req.content)["messages"][1]["content"])
            if content["instructions"] == "slow":
                entered.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    cancelled.set()
            await entered.wait()
            return httpx.Response(500)

        settings = config()
        service = DecisionService(settings, ModelBackend(settings, transport=httpx.MockTransport(handler)))
        try:
            with pytest.raises(ServiceError) as error:
                await service.evaluate(request(2, ["slow", "fail"]))
            assert error.value.question_id == "q1"
            assert cancelled.is_set()
        finally:
            await service.close()

    asyncio.run(run())


def test_disconnected_http_request_cancels_model_call():
    async def run():
        started = asyncio.Event()
        cancelled = asyncio.Event()

        async def handler(_):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        settings = config()
        app = create_app(
            settings, backend_factory=lambda cfg: ModelBackend(cfg, transport=httpx.MockTransport(handler))
        )
        sent_body = False
        responses = []

        async def receive():
            nonlocal sent_body
            if not sent_body:
                sent_body = True
                return {
                    "type": "http.request",
                    "body": request().model_dump_json().encode(),
                    "more_body": False,
                }
            await started.wait()
            return {"type": "http.disconnect"}

        async def send(message):
            responses.append(message)

        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/v1/decide",
            "raw_path": b"/v1/decide",
            "query_string": b"",
            "root_path": "",
            "headers": [(b"content-type", b"application/json")],
            "server": ("test", 80),
            "client": ("test", 1),
        }
        async with app.router.lifespan_context(app):
            await asyncio.wait_for(app(scope, receive, send), 2)
        assert cancelled.is_set()
        assert responses[0]["status"] == 499

    asyncio.run(run())
