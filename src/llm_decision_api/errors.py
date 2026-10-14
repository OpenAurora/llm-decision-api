from typing import Any

from pydantic import BaseModel


class ErrorDetail(BaseModel):
    code: str
    message: str
    request_id: str
    question_id: str | None = None
    details: list[dict[str, Any]] | None = None


class ErrorResponse(BaseModel):
    error: ErrorDetail


class ServiceError(Exception):
    """An error whose code and message are safe to return to API clients."""

    def __init__(
        self,
        status: int,
        code: str,
        message: str,
        *,
        question_id: str | None = None,
        details: list[dict] | None = None,
    ):
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.question_id = question_id
        self.details = details


def invalid_scores(code: str) -> ServiceError:
    return ServiceError(502, code, "模型未返回完整、有效的单 token 候选评分。")
