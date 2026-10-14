"""Public request and response types; no model-specific transport fields."""

from __future__ import annotations

import json
import math
from typing import Annotated, Literal, Union

from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    JsonValue,
    create_model,
)

from .labels import LABEL_CAPACITY

MAX_QUESTIONS = 32


def unicode_text(value: str):
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        raise ValueError("Strings must contain valid Unicode characters") from None
    return value


def check_json(value):
    if type(value) not in (str, dict, list):
        raise ValueError("Expected a string, JSON object or array")
    pending = [(value, 0)]
    while pending:
        item, depth = pending.pop()
        if depth > 64:
            raise ValueError("JSON nesting must not exceed 64 levels")
        if isinstance(item, dict):
            if any(type(key) is not str for key in item):
                raise ValueError("JSON object keys must be strings")
            for key in item:
                unicode_text(key)
            pending.extend((child, depth + 1) for child in item.values())
        elif isinstance(item, list):
            pending.extend((child, depth + 1) for child in item)
        elif type(item) is str:
            unicode_text(item)
        elif type(item) is float:
            if not math.isfinite(item):
                raise ValueError("JSON numbers must be finite")
        elif item is not None and type(item) not in (str, int, bool):
            raise ValueError("Only JSON values are accepted")
    return value


def nonempty(value):
    if not value or (isinstance(value, str) and not value.strip()):
        raise ValueError("Question instructions must not be empty")
    return value


def identifier(value: str) -> str:
    unicode_text(value)
    if not value.strip():
        raise ValueError("Identifiers and option names must not be blank")
    return value


SemanticValue = Annotated[Union[str, dict[str, JsonValue], list[JsonValue]], BeforeValidator(check_json)]
Instructions = Annotated[SemanticValue, AfterValidator(nonempty)]
Identifier = Annotated[str, AfterValidator(identifier)]
Description = Union[SemanticValue, None]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class ChoiceQuestion(StrictModel):
    type: Literal["choice"]
    instructions: Instructions
    criteria: dict[Identifier, Description] = Field(min_length=2, max_length=LABEL_CAPACITY)


class NoulQuestion(StrictModel):
    type: Literal["noul"]
    instructions: Instructions
    criteria: dict[Literal["true", "false"], SemanticValue] | None = None


class ScoreQuestion(StrictModel):
    type: Literal["score"]
    instructions: Instructions
    criteria: list[SemanticValue] = Field(min_length=2, max_length=LABEL_CAPACITY)


Question = Union[ChoiceQuestion, NoulQuestion, ScoreQuestion]


def request_model(max_options: int) -> type[BaseModel]:
    """Each application gets validation and OpenAPI matching its configuration."""
    choice = create_model(
        "ChoiceQuestion",
        __base__=ChoiceQuestion,
        criteria=(dict[Identifier, Description], Field(min_length=2, max_length=max_options)),
    )
    score = create_model(
        "ScoreQuestion",
        __base__=ScoreQuestion,
        criteria=(list[SemanticValue], Field(min_length=2, max_length=max_options)),
    )
    typed_question = Annotated[Union[choice, NoulQuestion, score], Field(discriminator="type")]
    return create_model(
        "DecisionRequest",
        __base__=StrictModel,
        state=(SemanticValue, Field(description="所有问题共用的资料：字符串、JSON 对象或数组。")),
        questions=(
            dict[Identifier, typed_question],
            Field(min_length=1, max_length=MAX_QUESTIONS, description="按问题 ID 组织，逐题独立判断。"),
        ),
    )


class ChoiceAnswer(StrictModel):
    type: Literal["choice"] = "choice"
    choice: str
    confidence: float = Field(ge=0, le=1, description="按候选数量缩放的最高概率优势，范围 0～1。")
    probabilities: dict[str, float]


class NoulAnswer(StrictModel):
    type: Literal["noul"] = "noul"
    noul: float = Field(ge=0, le=1, description="命题成立的概率，保留数值，不转换为布尔值。")


class ScoreAnswer(StrictModel):
    type: Literal["score"] = "score"
    score: float = Field(description="等级下标的概率加权平均，下标从 0 开始。")
    confidence: float = Field(ge=0, le=1, description="按官方适配器的等级分布集中程度计算，范围 0～1。")
    legend: dict[str, SemanticValue] = Field(
        description="等级下标到原始 criteria 描述的映射，保留 JSON 结构。"
    )
    probabilities: dict[str, float]


Answer = Annotated[Union[ChoiceAnswer, NoulAnswer, ScoreAnswer], Field(discriminator="type")]


class Usage(StrictModel):
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)


class DecisionResponse(StrictModel):
    model: str
    answers: dict[str, Answer]
    usage: Usage


def json_text(value) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
