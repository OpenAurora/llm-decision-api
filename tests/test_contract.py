import json

import httpx
import pytest
from conftest import model_response
from pydantic import ValidationError

from llm_decision_api.config import Settings
from llm_decision_api.prompt import CHOICE_SYSTEM, NOUL_SYSTEM, SYSTEM, prepare_question
from llm_decision_api.schemas import ChoiceQuestion, NoulQuestion, request_model


@pytest.mark.parametrize(
    "state",
    [
        "保留这段原文。",
        {"value": [1, True, None, {"amount": 2.5}]},
        ["第一条", {"role": "user", "text": "第二条"}],
        {},
        [],
    ],
)
def test_structured_state_round_trips_into_prompt(body, state):
    body["state"] = state
    parsed = request_model(20).model_validate(body)
    prepared = prepare_question(parsed.state, parsed.questions["department"])
    content = json.loads(prepared.messages[1]["content"])
    assert set(content) == {"state", "instructions", "criteria"}
    assert content["state"] == state
    assert content["criteria"] == [
        {"id": "A", "description": "refund"},
        {"id": "B", "description": {"technical_support": {"responsibility": "处理软件故障"}}},
    ]
    assert prepared.option_names == ("refund", "technical_support")
    assert "department" not in prepared.messages[1]["content"]


@pytest.mark.parametrize("state", [None, 12, False, float("nan"), {"value": float("inf")}])
def test_non_json_and_scalar_state_rejected(body, state):
    body["state"] = state
    with pytest.raises(ValidationError):
        request_model(20).model_validate(body)


@pytest.mark.parametrize(
    "instructions",
    [
        "先判断当前诉求。",
        {"rules": ["先判断当前诉求"], "reference": {"kind": 3}},
        ["先判断当前诉求", {"reference": {"kind": 3}}],
    ],
)
def test_structured_question_and_criteria_preserve_fields(body, instructions):
    body["questions"]["department"]["instructions"] = instructions
    parsed = request_model(20).model_validate(body)
    content = json.loads(
        prepare_question(parsed.state, parsed.questions["department"]).messages[1]["content"]
    )
    assert content["instructions"] == body["questions"]["department"]["instructions"]


@pytest.mark.parametrize(
    "description, expected",
    [
        (None, "refund"),
        ("The customer requests a refund.", "refund: The customer requests a refund."),
        ("", "refund: "),
        (
            {"condition": "duplicate_charge", "limit": 2},
            {"refund": {"condition": "duplicate_charge", "limit": 2}},
        ),
        (["duplicate charge", {"confirmed": True}], {"refund": ["duplicate charge", {"confirmed": True}]}),
        ({}, {"refund": {}}),
        ([], {"refund": []}),
    ],
)
def test_choice_preserves_business_meaning_without_exposing_name_field(description, expected):
    question = ChoiceQuestion(
        type="choice", instructions="Route the request.", criteria={"refund": description, "delivery": None}
    )
    prepared = prepare_question("Please refund my payment.", question)
    content = json.loads(prepared.messages[1]["content"])
    assert content["criteria"] == [
        {"id": "A", "description": expected},
        {"id": "B", "description": "delivery"},
    ]
    assert prepared.option_names == ("refund", "delivery")
    assert prepared.messages[0] == {"role": "system", "content": CHOICE_SYSTEM}
    assert question.criteria["refund"] == description


@pytest.mark.parametrize(
    "criteria, expected",
    [
        (None, ["true: The proposition is true.", "false: The proposition is false."]),
        ({}, ["true: The proposition is true.", "false: The proposition is false."]),
        ({"true": "Refund requested."}, ["true: Refund requested.", "false: The proposition is false."]),
        ({"false": "Delivery only."}, ["true: The proposition is true.", "false: Delivery only."]),
        (
            {"false": "Delivery only.", "true": "Refund requested."},
            ["true: Refund requested.", "false: Delivery only."],
        ),
        (
            {"true": {"condition": "refund_requested"}, "false": ["delivery only", {"refund": False}]},
            [{"true": {"condition": "refund_requested"}}, {"false": ["delivery only", {"refund": False}]}],
        ),
        ({"true": "", "false": []}, ["true: ", {"false": []}]),
    ],
)
def test_noul_user_descriptions_take_precedence_and_keep_true_false_mapping(criteria, expected):
    question = NoulQuestion(type="noul", instructions="Is a refund requested?", criteria=criteria)
    prepared = prepare_question("Please refund my payment.", question)
    content = json.loads(prepared.messages[1]["content"])
    assert content["criteria"] == [
        {"id": "t", "description": expected[0]},
        {"id": "f", "description": expected[1]},
    ]
    assert prepared.option_names == ("true", "false")
    assert prepared.labels == ("t", "f")
    assert prepared.messages[0] == {"role": "system", "content": NOUL_SYSTEM}
    assert question.criteria == criteria


@pytest.mark.parametrize("max_options", [2, 20, 32, 62])
@pytest.mark.parametrize("question_type", ["choice", "score"])
def test_config_controls_request_limits_without_global_schema_state(body, max_options, question_type):
    question = body["questions"]["department"]
    question["type"] = question_type
    question["criteria"] = (
        {f"c{i}": None for i in range(max_options)}
        if question_type == "choice" else [f"等级{i}" for i in range(max_options)]
    )
    request_model(max_options).model_validate(body)
    if question_type == "choice":
        question["criteria"]["overflow"] = None
    else:
        question["criteria"].append("overflow")
    with pytest.raises(ValidationError):
        request_model(max_options).model_validate(body)


@pytest.mark.parametrize("max_options", [1, 63, 255])
def test_configuration_rejects_unsupported_label_capacity(max_options):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, api_key="secret", model="test-model", max_options=max_options)


def test_environment_defaults_and_configurable_candidate_limit(monkeypatch):
    monkeypatch.setenv("LLM_DECISION_API_KEY", "secret")
    monkeypatch.setenv("LLM_DECISION_MODEL", "test-model")
    settings = Settings(_env_file=None)
    assert settings.model == "test-model"
    assert settings.api_key.get_secret_value() == "secret"
    assert settings.max_options == 20
    monkeypatch.setenv("LLM_DECISION_MAX_OPTIONS", "62")
    settings = Settings(_env_file=None)
    assert settings.max_options == 62
    assert "secret" not in repr(settings)


def test_model_name_must_be_configured(monkeypatch):
    monkeypatch.delenv("LLM_DECISION_MODEL", raising=False)
    with pytest.raises(ValidationError) as error:
        Settings(_env_file=None, api_key="synthetic-key")
    assert any(item["loc"] == ("model",) and item["type"] == "missing" for item in error.value.errors())


def test_environment_overrides_dotenv_configuration(tmp_path, monkeypatch):
    for key in ("BASE_URL", "MODEL", "API_KEY", "SERVICE_API_KEY"):
        monkeypatch.delenv(f"LLM_DECISION_{key}", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "LLM_DECISION_BASE_URL=https://model.example.com/v1\n"
        "LLM_DECISION_MODEL=file-model\n"
        "LLM_DECISION_API_KEY=synthetic-upstream-key\n"
        "LLM_DECISION_SERVICE_API_KEY=synthetic-service-key\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("LLM_DECISION_MODEL", "environment-model")
    settings = Settings(_env_file=env_file)
    assert settings.base_url == "https://model.example.com/v1"
    assert settings.model == "environment-model"
    assert settings.api_key.get_secret_value() == "synthetic-upstream-key"
    assert settings.service_api_key.get_secret_value() == "synthetic-service-key"
    assert "synthetic-" not in repr(settings)


def test_mixed_types_are_independent_and_output_semantics_are_defined(client_factory):
    captured = []

    def handler(request):
        payload = json.loads(request.content)
        captured.append(payload)
        content = json.loads(payload["messages"][1]["content"])
        labels = payload["structured_outputs"]["choice"]
        scores = {"选择部门": [0.8, 0.2], "是否退款": [0.7, 0.3], "评估紧急程度": [0.1, 0.3, 0.6]}
        return httpx.Response(200, json=model_response(labels, scores[content["instructions"]]))

    body = {
        "state": ["共享资料"],
        "questions": {
            "private-choice-id": {
                "type": "choice",
                "instructions": "选择部门",
                "criteria": {"refund": None, "technical_support": None},
            },
            "private-noul-id": {"type": "noul", "instructions": "是否退款"},
            "private-score-id": {
                "type": "score",
                "instructions": "评估紧急程度",
                "criteria": ["低", {"level": "中", "detail": [1, True, None]}, ["高", {"limit": 2.5}]],
            },
        },
    }
    with client_factory(handler) as client:
        response = client.post("/v1/decide", json=body)
    assert response.status_code == 200, response.text
    data = response.json()
    assert list(data["answers"]) == list(body["questions"])
    assert set(data) == {"model", "answers", "usage"}
    assert data["answers"]["private-choice-id"] == {
        "type": "choice",
        "choice": "refund",
        "confidence": pytest.approx(0.6),
        "probabilities": {"refund": pytest.approx(0.8), "technical_support": pytest.approx(0.2)},
    }
    assert data["answers"]["private-noul-id"] == {"type": "noul", "noul": pytest.approx(0.7)}
    score = data["answers"]["private-score-id"]
    assert set(score) == {"type", "score", "confidence", "legend", "probabilities"}
    assert score["score"] == pytest.approx(1.5)
    assert score["confidence"] == pytest.approx(0.25)
    assert score["legend"] == {
        "0": "低",
        "1": {"level": "中", "detail": [1, True, None]},
        "2": ["高", {"limit": 2.5}],
    }
    assert score["probabilities"] == {
        "0": pytest.approx(0.1),
        "1": pytest.approx(0.3),
        "2": pytest.approx(0.6),
    }
    assert data["usage"] == {"input_tokens": 120, "output_tokens": 3}
    assert len(captured) == 3
    for payload in captured:
        content = json.loads(payload["messages"][1]["content"])
        assert set(content) == {"state", "instructions", "criteria"}
        assert all(set(item) == {"id", "description"} for item in content["criteria"])
        assert [item["id"] for item in content["criteria"]] == payload["structured_outputs"]["choice"]
        assert content["state"] == body["state"]
        assert "private-" not in json.dumps(payload)
        assert "questions" not in content
        assert payload["max_tokens"] == 1 and payload["top_logprobs"] == 20
        assert payload["logprobs"] is True
        assert payload["chat_template_kwargs"] == {"enable_thinking": False}
        if content["instructions"] == "是否退款":
            assert payload["structured_outputs"]["choice"] == ["t", "f"]
            assert payload["messages"][0]["content"] == NOUL_SYSTEM
        elif content["instructions"] == "选择部门":
            assert payload["messages"][0]["content"] == CHOICE_SYSTEM
        else:
            assert payload["messages"][0]["content"] == SYSTEM
        if content["instructions"] == "评估紧急程度":
            assert content["criteria"] == [
                {"id": "A", "description": "低"},
                {"id": "B", "description": {"level": "中", "detail": [1, True, None]}},
                {"id": "C", "description": ["高", {"limit": 2.5}]},
            ]


def test_62_labels_preserve_case_and_original_business_names(client_factory):
    captured = []

    def handler(request):
        payload = json.loads(request.content)
        captured.append(payload)
        labels = payload["structured_outputs"]["choice"]
        return httpx.Response(200, json=model_response(labels, [0.2 / 61] * 61 + [0.8]))

    body = {
        "state": "x",
        "questions": {
            "q": {
                "type": "choice",
                "instructions": "选最后一个",
                "criteria": {f"item-{i}": None for i in range(62)},
            }
        },
    }
    with client_factory(handler, max_options=62) as client:
        response = client.post("/v1/decide", json=body)
    assert response.status_code == 200, response.text
    assert response.json()["answers"]["q"]["choice"] == "item-61"
    assert len(response.json()["answers"]["q"]["probabilities"]) == 62
    assert response.json()["answers"]["q"]["confidence"] == pytest.approx(0.7967213114754098)
    labels = captured[0]["structured_outputs"]["choice"]
    assert labels[0] == "A" and labels[26] == "a" and labels[-1] == "9"
    criteria = json.loads(captured[0]["messages"][1]["content"])["criteria"]
    assert [item["id"] for item in criteria] == labels
    assert criteria[26] == {"id": "a", "description": "item-26"}
    assert captured[0]["top_logprobs"] == 62


@pytest.mark.parametrize("max_options", [2, 20, 62])
def test_score_at_configured_limit_returns_all_levels_and_rejects_overflow(client_factory, max_options):
    captured = []

    def handler(request):
        payload = json.loads(request.content)
        captured.append(payload)
        labels = payload["structured_outputs"]["choice"]
        return httpx.Response(200, json=model_response(labels, [1 / max_options] * max_options))

    criteria = [{"level": i} for i in range(max_options)]
    body = {
        "state": "资料",
        "questions": {"q": {"type": "score", "instructions": "评分", "criteria": criteria}},
    }
    with client_factory(handler, max_options=max_options) as client:
        response = client.post("/v1/decide", json=body)
        assert response.status_code == 200, response.text
        answer = response.json()["answers"]["q"]
        assert answer["score"] == pytest.approx((max_options - 1) / 2)
        assert answer["confidence"] == 0
        assert answer["legend"] == {str(i): description for i, description in enumerate(criteria)}
        assert answer["probabilities"] == {
            str(i): pytest.approx(1 / max_options) for i in range(max_options)
        }
        assert len(captured[0]["structured_outputs"]["choice"]) == max_options
        assert captured[0]["top_logprobs"] == max_options
        criteria.append({"level": max_options})
        response = client.post("/v1/decide", json=body)
        assert response.status_code == 422
        assert len(captured) == 1


def test_http_numbers_keep_precision_and_single_question_envelope(client_factory):
    probabilities = [0.0123456789, 0.2345678912, 0.7530864299]

    def handler(request):
        labels = json.loads(request.content)["structured_outputs"]["choice"]
        return httpx.Response(200, json=model_response(labels, probabilities))

    question = {"type": "score", "instructions": "评分", "criteria": ["低", "中", "高"]}
    with client_factory(handler) as client:
        response = client.post("/v1/decide", json={"state": "资料", "questions": {"q": question}})
    assert response.status_code == 200
    data = response.json()
    assert set(data) == {"model", "answers", "usage"}
    assert set(data["answers"]) == {"q"}
    answer = data["answers"]["q"]
    assert answer["score"] == pytest.approx(1.740740751, abs=1e-12)
    assert answer["confidence"] == pytest.approx(0.6111111265, abs=1e-12)
    assert list(answer["probabilities"].values()) == pytest.approx(probabilities, abs=1e-12)
    assert data["usage"]["output_tokens"] == 1
