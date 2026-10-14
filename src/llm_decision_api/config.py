from urllib.parse import urlsplit

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .labels import LABEL_CAPACITY


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="LLM_DECISION_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        hide_input_in_errors=True,
        allow_inf_nan=False,
    )

    base_url: str = "http://127.0.0.1:8000/v1"
    model: str = Field(min_length=1)
    api_key: SecretStr = Field(min_length=1, repr=False)
    service_api_key: SecretStr | None = Field(default=None, repr=False)
    max_options: int = Field(default=20, ge=2, le=LABEL_CAPACITY)
    body_timeout_seconds: float = Field(default=10, gt=0)
    timeout_seconds: float = Field(default=10, gt=0)
    max_body_bytes: int = Field(default=1024 * 1024, ge=1024)
    verify_upstream_on_startup: bool = True
    log_level: str = "INFO"

    @field_validator("base_url")
    @classmethod
    def valid_base_url(cls, value: str) -> str:
        parsed = urlsplit(value)
        if (
            parsed.scheme not in ("http", "https")
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("Model base URL must be HTTP(S), without credentials, query or fragment")
        return value.rstrip("/")

    @field_validator("service_api_key", mode="before")
    @classmethod
    def empty_service_key(cls, value):
        return None if value == "" else value

    @field_validator("log_level")
    @classmethod
    def valid_log_level(cls, value: str) -> str:
        value = value.upper()
        if value not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError("Unsupported log level")
        return value
