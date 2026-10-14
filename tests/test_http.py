import json
import logging
import traceback

import httpx
import pytest
from conftest import model_response
from fastapi.testclient import TestClient
from starlette.responses import Response

from llm_decision_api.api import create_app
from llm_decision_api.backend import ModelBackend


def test_validation_precedes_all_inference_and_openapi_matches_config(client_factory, body):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json=model_response())

    with client_factory(handler) as client:
        overflow = {"type": "choice", "instructions": "判断", "criteria": {str(i): None for i in range(21)}}
        response = client.post(
            "/v1/decide",
            json={"state": "PRIVATE-STATE", "questions": body["questions"] | {"overflow": overflow}},
        )
        assert response.status_code == 422
        assert "PRIVATE-STATE" not in response.text and not calls
        response = client.post("/v1/decide", json=body | {"model": "caller-override"})
        assert response.status_code == 422 and not calls
        response = client.post(
            "/v1/decide",
            json={"state": "x", "questions": {str(i): body["questions"]["department"] for i in range(33)}},
        )
        assert response.status_code == 422 and not calls
        schema = client.app.openapi()
        assert (
            schema["components"]["schemas"]["ChoiceQuestion"]["properties"]["criteria"]["maxProperties"] == 20
        )
        assert schema["components"]["schemas"]["ScoreQuestion"]["properties"]["criteria"]["maxItems"] == 20
        assert set(schema["components"]["schemas"]["DecisionRequest"]["properties"]) == {"state", "questions"}
    with client_factory(max_options=62) as client:
        schema = client.app.openapi()
        assert (
            schema["components"]["schemas"]["ChoiceQuestion"]["properties"]["criteria"]["maxProperties"] == 62
        )
        assert schema["components"]["schemas"]["ScoreQuestion"]["properties"]["criteria"]["maxItems"] == 62


@pytest.mark.parametrize(
    "raw",
    [
        b"{",
        b'{"state":"x","state":"y","questions":{}}',
        b'{"state":{"v":NaN},"questions":{}}',
        b'{"state":{"v":1e999},"questions":{}}',
    ],
)
def test_nonstandard_or_duplicate_json_rejected(client_factory, raw):
    with client_factory() as client:
        response = client.post("/v1/decide", content=raw, headers={"content-type": "application/json"})
    assert response.status_code == 400, response.text
    assert response.json()["error"]["code"] == "invalid_json"


def test_openapi_describes_confidence_and_structured_legends(client_factory):
    with client_factory() as client:
        schema = client.app.openapi()
    schemas = schema["components"]["schemas"]
    for name in ("ChoiceAnswer", "ScoreAnswer"):
        answer = schemas[name]
        assert "confidence" in answer["required"]
        assert answer["properties"]["confidence"]["type"] == "number"
        assert answer["properties"]["confidence"]["minimum"] == 0
        assert answer["properties"]["confidence"]["maximum"] == 1
    assert set(schemas["NoulAnswer"]["properties"]) == {"type", "noul"}
    assert schemas["NoulAnswer"]["properties"]["noul"]["type"] == "number"
    legend = schemas["ScoreAnswer"]["properties"]["legend"]["additionalProperties"]
    assert {variant["type"] for variant in legend["anyOf"]} == {"string", "object", "array"}


def test_body_limit_authentication_and_request_tracing(client_factory, body):
    with client_factory(service_api_key="inbound-only", max_body_bytes=1024) as client:
        response = client.post("/v1/decide", json=body)
        assert response.status_code == 401
        assert response.headers["WWW-Authenticate"] == "Bearer"
        response = client.post("/v1/decide", json=body, headers={"Authorization": "Bearer inbound-only"})
        assert response.status_code == 200
        assert len(response.headers["X-Request-ID"]) == 32
        response = client.post(
            "/v1/decide", content=b" " * 1025, headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 413
        assert response.json()["error"]["request_id"] == response.headers["X-Request-ID"]
        assert client.get("/docs").status_code == 200
        assert client.app.state.service.verified is False


@pytest.mark.parametrize(
    "status, expected_status, expected_code",
    [
        (401, 502, "upstream_rejected"),
        (429, 503, "upstream_busy"),
        (503, 503, "upstream_busy"),
        (504, 504, "upstream_timeout"),
    ],
)
def test_upstream_errors_do_not_expose_secrets(client_factory, body, status, expected_status, expected_code):
    with client_factory(lambda _: httpx.Response(status, text="PRIVATE-UPSTREAM-RESPONSE")) as client:
        response = client.post("/v1/decide", json=body)
    assert response.status_code == expected_status
    assert response.json()["error"]["code"] == expected_code
    assert response.json()["error"]["question_id"] == "department"
    assert response.json()["error"]["request_id"] == response.headers["x-request-id"]
    assert "PRIVATE-UPSTREAM-RESPONSE" not in response.text
    assert "answers" not in response.json()


def test_upstream_option_limit_error_is_actionable(client_factory, body):
    with client_factory(
        lambda _: httpx.Response(400, text="Requested sample logprobs of 62, greater than max allowed: 20"),
        max_options=62,
    ) as client:
        response = client.post("/v1/decide", json=body)
    assert response.status_code == 502
    assert response.json()["error"]["code"] == "upstream_logprobs_limit"


def test_startup_check_verifies_full_configured_capacity(client_factory):
    sizes = []

    def handler(request):
        payload = json.loads(request.content)
        labels = payload["structured_outputs"]["choice"]
        sizes.append(len(labels))
        return httpx.Response(200, json=model_response(labels))

    with client_factory(handler, verify_upstream_on_startup=True) as client:
        assert client.app.state.service.verified is True
        assert client.get("/docs").status_code == 200
    assert sizes == [20]
    with pytest.raises(RuntimeError, match="incomplete_candidate_scores"):
        with client_factory(
            lambda _: httpx.Response(200, json=model_response()), verify_upstream_on_startup=True
        ):
            pass


def test_unknown_routes_methods_and_unicode_validation_use_error_envelope(client_factory, body):
    with client_factory() as client:
        assert client.get("/missing").json()["error"]["code"] == "not_found"
        assert client.get("/v1/decide").json()["error"]["code"] == "method_not_allowed"
        body["state"] = "\ud800"
        response = client.post(
            "/v1/decide", content=json.dumps(body), headers={"content-type": "application/json"}
        )
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "invalid_request"
        body["state"] = "x"
        body["questions"]["\ud800"] = body["questions"].pop("department")
        response = client.post(
            "/v1/decide", content=json.dumps(body), headers={"content-type": "application/json"}
        )
        assert response.status_code == 422


@pytest.mark.parametrize(
    "method, path, expected_allow",
    [("GET", "/v1/decide", "POST"), ("POST", "/docs", "GET")],
)
def test_method_not_allowed_preserves_route_methods_and_error_envelope(
    client_factory, method, path, expected_allow
):
    with client_factory() as client:
        response = client.request(method, path)
    assert response.status_code == 405
    assert response.headers["allow"] == expected_allow
    assert response.json()["error"]["code"] == "method_not_allowed"
    assert response.json()["error"]["request_id"] == response.headers["x-request-id"]


def test_non_json_media_type_returns_415(client_factory, body):
    with client_factory() as client:
        response = client.post("/v1/decide", content=json.dumps(body), headers={"content-type": "text/plain"})
    assert response.status_code == 415
    assert response.json()["error"]["code"] == "unsupported_media_type"


def test_internal_errors_keep_request_id_and_hide_exception_text(settings, body, caplog):
    def broken(_):
        try:
            raise ValueError("private-root-cause")
        except ValueError as exc:
            raise RuntimeError("private-exception-text") from exc

    app = create_app(
        settings, backend_factory=lambda cfg: ModelBackend(cfg, transport=httpx.MockTransport(broken))
    )
    # The default TestClient propagates exceptions that would reach Uvicorn.
    with caplog.at_level(logging.INFO, logger="llm_decision_api"), TestClient(app) as client:
        response = client.post("/v1/decide", json=body)
    assert response.status_code == 500
    assert response.json()["error"]["request_id"] == response.headers["x-request-id"]
    for private in ("private-exception-text", "private-root-cause"):
        assert private not in response.text and private not in caplog.text
    records = [record for record in caplog.records if record.name == "llm_decision_api"]
    assert all(record.exc_info is None for record in records)
    events = [json.loads(record.getMessage()) for record in records]
    errors = [event for event in events if event["event"] == "internal_error"]
    assert len(errors) == 1
    assert errors[0]["exception_type"] == "RuntimeError"
    assert errors[0]["request_id"] == response.headers["x-request-id"]
    assert any(frame["function"] == "broken" and frame["line"] > 0 for frame in errors[0]["stack"])
    assert events[-1]["event"] == "http_request" and events[-1]["status"] == 500


def test_error_after_response_starts_only_exposes_fixed_server_error(settings, caplog):
    class BrokenResponse(Response):
        async def __call__(self, scope, receive, send):
            await send({"type": "http.response.start", "status": 200, "headers": []})
            raise RuntimeError("private-response-exception")

    app = create_app(settings)

    @app.get("/broken-response")
    async def broken_response():
        return BrokenResponse()

    with caplog.at_level(logging.INFO, logger="llm_decision_api"), TestClient(app) as client:
        with pytest.raises(RuntimeError, match="HTTP response failed after it started") as error:
            client.get("/broken-response")
    rendered_exception = "".join(traceback.format_exception(error.value))
    assert "private-response-exception" not in rendered_exception
    assert "private-response-exception" not in caplog.text
    events = [json.loads(record.getMessage()) for record in caplog.records if record.name == "llm_decision_api"]
    errors = [event for event in events if event["event"] == "internal_error"]
    assert len(errors) == 1 and errors[0]["exception_type"] == "RuntimeError"
