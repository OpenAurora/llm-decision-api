import copy

import pytest
from conftest import model_response

from llm_decision_api.errors import ServiceError
from llm_decision_api.prompt import prepare_question
from llm_decision_api.schemas import ChoiceQuestion, ScoreQuestion
from llm_decision_api.scoring import parse_scores, to_answer


def prepared():
    return prepare_question(
        "资料", ChoiceQuestion(type="choice", instructions="判断", criteria={"first": None, "second": None})
    )


@pytest.mark.parametrize(
    "kind",
    [
        "missing",
        "duplicate",
        "nan",
        "masked",
        "positive",
        "thinking",
        "reasoning",
        "extra_position",
        "two_tokens",
        "no_usage",
        "bad_label",
        "lowercase",
        "quoted_label",
        "newline",
        "explanation",
        "mismatch",
        "finish",
        "bool_usage",
    ],
)
def test_incomplete_or_invalid_model_scores_never_become_answers(kind):
    body = model_response()
    choice = body["choices"][0]
    message = choice["message"]
    positions = choice["logprobs"]["content"]
    top = positions[0]["top_logprobs"]
    if kind == "missing":
        top.pop()
    elif kind == "duplicate":
        top.append(copy.deepcopy(top[0]))
    elif kind == "nan":
        top[1]["logprob"] = float("nan")
    elif kind == "masked":
        top[1]["logprob"] = -9999
    elif kind == "positive":
        top[1]["logprob"] = 1
    elif kind == "thinking":
        message["reasoning_content"] = "analysis"
    elif kind == "reasoning":
        message["provider_specific_fields"] = {"reasoning": "analysis"}
    elif kind == "extra_position":
        positions.append(copy.deepcopy(positions[0]))
    elif kind == "two_tokens":
        body["usage"]["completion_tokens"] = 2
    elif kind == "no_usage":
        body.pop("usage")
    elif kind == "bad_label":
        message["content"] = " A"
    elif kind == "lowercase":
        message["content"] = "a"
    elif kind == "quoted_label":
        message["content"] = '"A"'
    elif kind == "newline":
        message["content"] = "A\n"
    elif kind == "explanation":
        message["content"] = "A because the description matches."
    elif kind == "mismatch":
        top[1]["logprob"] = 0
    elif kind == "finish":
        choice["finish_reason"] = "content_filter"
    elif kind == "bool_usage":
        body["usage"]["completion_tokens"] = True
    with pytest.raises(ServiceError) as error:
        parse_scores(body, prepared())
    assert error.value.status == 502


def test_stable_normalization_and_tie_respect_returned_maximum():
    body = model_response(probabilities=[0.5, 0.5], selected="B")
    scored = parse_scores(body, prepared())
    assert scored.best_index == 1
    assert scored.probabilities == pytest.approx((0.5, 0.5))
    body["choices"][0]["logprobs"]["content"][0]["top_logprobs"] = [
        {"token": "A", "logprob": -9000},
        {"token": "B", "logprob": -8999},
    ]
    scored = parse_scores(body, prepared())
    assert sum(scored.probabilities) == pytest.approx(1)
    assert scored.probabilities[1] > scored.probabilities[0]


@pytest.mark.parametrize(
    "probabilities, expected_confidence",
    [
        ([0.5, 0.5], 0.0),
        ([1.0, 0.0], 1.0),
        ([0.82, 0.18], 0.64),
        ([0.8, 0.15, 0.05], 0.7),
        ([1 / 62] * 62, 0.0),
    ],
)
def test_choice_confidence_scales_peak_above_uniform(probabilities, expected_confidence):
    question = ChoiceQuestion(
        type="choice", instructions="选择", criteria={f"业务选项{i}": None for i in range(len(probabilities))}
    )
    prepared = prepare_question("资料", question)
    scored = parse_scores(model_response(prepared.labels, probabilities), prepared)
    answer = to_answer(prepared, scored)
    assert answer.confidence == pytest.approx(expected_confidence, abs=1e-12)
    assert tuple(answer.probabilities.values()) == pytest.approx(probabilities)


@pytest.mark.parametrize(
    "probabilities, selected, expected_score, expected_confidence",
    [
        ([1.0, 0.0, 0.0], "A", 0.0, 1.0),
        ([0.0, 1.0, 0.0], "B", 1.0, 1.0),
        ([0.0, 0.0, 1.0], "C", 2.0, 1.0),
        ([0.25] * 4, "A", 1.5, 0.0),
        ([0.2] * 5, "A", 2.0, 0.0),
        ([0.0] * 19 + [1.0], "T", 19.0, 1.0),
        ([0.0] * 61 + [1.0], "9", 61.0, 1.0),
        ([0.82, 0.18], "A", 0.18, 0.64),
        ([0.0, 0.76, 0.24], "B", 1.24, 0.64),
        ([0.8, 0.2, 0.0], "A", 0.2, 0.7),
        ([0.8, 0.0, 0.2], "A", 0.4, 0.4),
        ([0.01, 0.02, 0.07, 0.3, 0.6], "E", 3.46, 0.55),
        # Score confidence uses the first mode, even when the model emits another tied label.
        ([0.4, 0.4, 0.2], "B", 0.8, 0.0),
        ([0.2, 0.4, 0.4], "C", 1.2, 0.1),
    ],
)
def test_score_value_and_confidence_follow_separate_metrics(
    probabilities, selected, expected_score, expected_confidence
):
    question = ScoreQuestion(
        type="score", instructions="评分", criteria=[f"等级{i}" for i in range(len(probabilities))]
    )
    prepared = prepare_question("资料", question)
    scored = parse_scores(model_response(prepared.labels, probabilities, selected=selected), prepared)
    answer = to_answer(prepared, scored)
    assert answer.score == pytest.approx(expected_score, abs=1e-12)
    assert answer.confidence == pytest.approx(expected_confidence, abs=1e-12)
    assert 0 <= answer.confidence <= 1


def test_choice_tie_keeps_model_selection_with_zero_confidence():
    question = prepared()
    scored = parse_scores(model_response(probabilities=[0.5, 0.5], selected="B"), question)
    answer = to_answer(question, scored)
    assert answer.choice == "second"
    assert answer.confidence == 0.0
