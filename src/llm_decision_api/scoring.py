"""Validate every candidate before computing a conditional distribution."""

import math
from dataclasses import dataclass

from .errors import ServiceError, invalid_scores
from .prompt import PreparedQuestion
from .schemas import ChoiceAnswer, ChoiceQuestion, NoulAnswer, NoulQuestion, ScoreAnswer


@dataclass(frozen=True)
class ScoredQuestion:
    probabilities: tuple[float, ...]
    best_index: int
    input_tokens: int
    output_tokens: int


def parse_scores(body: dict, prepared: PreparedQuestion) -> ScoredQuestion:
    try:
        choices = body["choices"]
        if not isinstance(choices, list) or len(choices) != 1:
            raise invalid_scores("invalid_choice_count")
        choice = choices[0]
        message = choice["message"]
        if (
            message.get("reasoning")
            or message.get("reasoning_content")
            or (message.get("provider_specific_fields") or {}).get("reasoning")
        ):
            raise invalid_scores("thinking_not_disabled")
        content = message.get("content")
        if content not in prepared.labels:
            raise invalid_scores("invalid_output_label")
        if choice.get("finish_reason") not in ("stop", "length"):
            raise invalid_scores("invalid_finish_reason")
        positions = choice["logprobs"]["content"]
        if len(positions) != 1 or positions[0]["token"] != content:
            raise invalid_scores("not_one_answer_token")
        usage = body["usage"]
        if type(usage.get("completion_tokens")) is not int or usage["completion_tokens"] != 1:
            raise invalid_scores("not_one_completion_token")
        if type(usage.get("prompt_tokens")) is not int or usage["prompt_tokens"] < 0:
            raise invalid_scores("invalid_token_usage")
        found = {}
        for item in positions[0]["top_logprobs"]:
            label = item["token"]
            if label not in prepared.labels:
                continue
            if label in found:
                raise invalid_scores("duplicate_candidate_score")
            value = item["logprob"]
            if type(value) not in (int, float) or not math.isfinite(value) or not -9999 < value <= 1e-5:
                raise invalid_scores("invalid_candidate_score")
            found[label] = value
        if set(found) != set(prepared.labels):
            raise invalid_scores("incomplete_candidate_scores")
        values = [found[label] for label in prepared.labels]
        peak = max(values)
        selected_index = prepared.labels.index(content)
        if values[selected_index] != peak:
            raise invalid_scores("output_score_disagreement")
        weights = [math.exp(value - peak) for value in values]
        total = math.fsum(weights)
        return ScoredQuestion(
            tuple(value / total for value in weights),
            selected_index,
            usage["prompt_tokens"],
            usage["completion_tokens"],
        )
    except ServiceError:
        raise
    except (KeyError, IndexError, TypeError, ValueError, AttributeError):
        raise invalid_scores("malformed_model_response") from None


def _choice_confidence(probabilities: tuple[float, ...]) -> float:
    """Scale the peak above a uniform distribution; candidates were validated upstream."""
    count = len(probabilities)
    peak = max(probabilities) / math.fsum(probabilities)
    return max(0.0, min(1.0, (count * peak - 1) / (count - 1)))


def _score_confidence(probabilities: tuple[float, ...]) -> float:
    """Use the approved TypeSafe adapter metric; see README.md for its source."""
    count = len(probabilities)
    # The adapter uses the first mode on ties, regardless of the emitted label.
    mode = max(range(count), key=probabilities.__getitem__)
    deviation = math.fsum(
        probability * abs(index - mode) for index, probability in enumerate(probabilities)
    ) / math.fsum(probabilities)
    midpoint = (count - 1) / 2
    uniform_deviation = math.fsum(abs(index - midpoint) for index in range(count)) / count
    return max(0.0, min(1.0, 1 - deviation / uniform_deviation))


def to_answer(prepared: PreparedQuestion, scored: ScoredQuestion):
    probabilities = dict(zip(prepared.option_names, scored.probabilities))
    question = prepared.question
    if isinstance(question, ChoiceQuestion):
        return ChoiceAnswer(
            choice=prepared.option_names[scored.best_index],
            confidence=_choice_confidence(scored.probabilities),
            probabilities=probabilities,
        )
    if isinstance(question, NoulQuestion):
        return NoulAnswer(noul=probabilities["true"])
    return ScoreAnswer(
        score=math.fsum(index * probability for index, probability in enumerate(scored.probabilities)),
        confidence=_score_confidence(scored.probabilities),
        legend={str(index): description for index, description in enumerate(question.criteria)},
        probabilities=probabilities,
    )
