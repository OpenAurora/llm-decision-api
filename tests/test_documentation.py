import ast
import json
import shlex
from html.parser import HTMLParser

import pytest

from llm_decision_api.documentation import render_guide
from llm_decision_api.schemas import DecisionResponse, request_model


class CodeExamples(HTMLParser):
    def __init__(self, html):
        super().__init__(convert_charrefs=True)
        self.examples = {}
        self.active = None
        self.feed(html)

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if tag == "code" and "data-example" in attributes:
            self.active = attributes["id"]
            self.examples[self.active] = ""

    def handle_endtag(self, tag):
        if tag == "code":
            self.active = None

    def handle_data(self, data):
        if self.active:
            self.examples[self.active] += data


@pytest.mark.parametrize("max_options", [2, 20, 62])
def test_reference_examples_follow_configured_limits_without_inference(client_factory, max_options):
    def no_inference(_):
        pytest.fail("Reading documentation must not call the model")

    with client_factory(no_inference, max_options=max_options) as client:
        response = client.get("/docs")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/html")
        assert response.headers["x-request-id"]
        assert 'lang="zh-CN"' in response.text
        assert "API 文档" in response.text
        assert "{{" not in response.text
        assert f"2～{max_options} 项" in response.text
        assert f"2～{max_options} 项，按等级顺序排列" in response.text
        examples = CodeExamples(response.text).examples
        curl = shlex.split(examples["curl-code"].replace("\\\n", ""))
        assert curl[:2] == ["curl", "http://testserver/v1/decide"]
        body = json.loads(curl[curl.index("-d") + 1])
        validated = request_model(max_options).model_validate(body)
        assert "Content-Type: application/json" in curl
        assert "Authorization:" not in " ".join(curl)
        python = examples["python-code"]
        compile(python, "<docs example>", "exec")
        assignment = next(
            node for node in ast.parse(python).body
            if isinstance(node, ast.Assign) and node.targets[0].id == "body"
        )
        assert ast.literal_eval(assignment.value) == body
        result = DecisionResponse.model_validate_json(examples["response-example"])
        assert result.answers.keys() == validated.questions.keys()
        questions = {}
        answers = {}
        for kind in ("choice", "noul", "score"):
            questions[kind] = json.loads(examples[f"{kind}-question"])
            answers[kind] = json.loads(examples[f"{kind}-answer-code"])
            assert questions[kind]["type"] == answers[kind]["type"] == kind
        request_model(max_options).model_validate({"state": body["state"], "questions": questions})
        DecisionResponse.model_validate({
            "model": "example", "answers": answers, "usage": {"input_tokens": 0, "output_tokens": 0},
        })


def test_reference_uses_safe_auth_examples_and_removes_generated_documentation(client_factory):
    with client_factory(
        service_api_key="private-service-key",
        max_body_bytes=2048,
        body_timeout_seconds=3,
        timeout_seconds=7,
    ) as client:
        guide = client.get("/docs")
        assert guide.status_code == 200
        for removed in ("/healthz", "/readyz", "/docs/download", "/api-docs", "/openapi.json", "/redoc"):
            assert client.get(removed).status_code == 404
            assert removed not in guide.text
        assert "OpenAPI" not in guide.text and "接口调试" not in guide.text
        assert 'data-language="request"' not in guide.text
        assert '>JSON</button>' not in guide.text
        assert "需要 Bearer 认证" in guide.text
        assert "2,048 字节" in guide.text
        assert "<td>3 秒</td>" in guide.text
        assert "<td>7 秒</td>" in guide.text
        for secret in ("private-service-key", "private-model-key", "model.invalid"):
            assert secret not in guide.text
        examples = CodeExamples(guide.text).examples
        curl = shlex.split(examples["curl-code"].replace("\\\n", ""))
        assert "Authorization: Bearer YOUR_SERVICE_KEY" in curl
        assert "'Authorization': 'Bearer YOUR_SERVICE_KEY'" in examples["python-code"]
        assert "Authorization: Bearer YOUR_SERVICE_KEY" in examples["endpoint-code"]
        assert '<script src=' not in guide.text
        assert '<link rel="stylesheet"' not in guide.text


def test_reference_escapes_base_url_and_preserves_valid_examples_under_a_prefix(settings):
    guide = render_guide(settings, 'https://example.invalid/gateway/')
    assert 'href="https://example.invalid/gateway/docs"' in guide
    assert 'https://example.invalid/gateway/v1/decide' in guide
    hostile_url = 'https://example.invalid/\"><img src=x onerror=alert(1)>'
    escaped = render_guide(settings, hostile_url)
    assert '<img src=x' not in escaped
    assert '&quot;&gt;&lt;img src=x onerror=alert(1)&gt;' in escaped
    examples = CodeExamples(escaped).examples
    compile(examples["python-code"], "<docs example>", "exec")
    curl = shlex.split(examples["curl-code"].replace("\\\n", ""))
    assert curl[1] == hostile_url + "/v1/decide"


@pytest.mark.parametrize("origin", ["http://localhost:9000", "https://api.example.com"])
def test_reference_uses_request_origin_and_deployment_prefix(client_factory, origin):
    with client_factory() as client:
        client.app.root_path = "/gateway"
        response = client.get(origin + "/gateway/docs")
    assert response.status_code == 200
    examples = CodeExamples(response.text).examples
    curl = shlex.split(examples["curl-code"].replace("\\\n", ""))
    assert curl[:2] == ["curl", origin + "/gateway/v1/decide"]
    assert f'href="{origin}/gateway/docs"' in response.text
