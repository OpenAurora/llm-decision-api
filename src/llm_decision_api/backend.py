"""HTTP model transport, owned and closed by the application lifespan."""

import httpx

from .config import Settings
from .errors import ServiceError
from .prompt import PreparedQuestion
from .scoring import ScoredQuestion, parse_scores


class ModelBackend:
    def __init__(self, settings: Settings, transport: httpx.AsyncBaseTransport | None = None):
        self.settings = settings
        self.client = httpx.AsyncClient(
            base_url=settings.base_url + "/",
            headers={"Authorization": "Bearer " + settings.api_key.get_secret_value()},
            timeout=settings.timeout_seconds,
            limits=httpx.Limits(
                max_connections=None,
                max_keepalive_connections=None,
            ),
            trust_env=False,
            follow_redirects=False,
            transport=transport,
        )

    async def close(self):
        await self.client.aclose()

    def payload(self, prepared: PreparedQuestion) -> dict:
        if len(prepared.labels) > self.settings.max_options:
            raise ServiceError(422, "too_many_options", "候选数量超过服务配置上限。")
        return {
            "model": self.settings.model,
            "messages": prepared.messages,
            "temperature": 0,
            "max_tokens": 1,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
            "structured_outputs": {"choice": list(prepared.labels)},
            "logprobs": True,
            "top_logprobs": self.settings.max_options,
            "return_tokens_as_token_ids": False,
        }

    async def score(self, prepared: PreparedQuestion) -> ScoredQuestion:
        try:
            response = await self.client.post("chat/completions", json=self.payload(prepared))
        except httpx.TimeoutException:
            raise ServiceError(504, "upstream_timeout", "模型服务请求超时。") from None
        except httpx.RequestError:
            raise ServiceError(502, "upstream_unavailable", "无法连接模型服务。") from None
        if response.status_code != 200:
            if response.status_code == 504:
                raise ServiceError(504, "upstream_timeout", "模型服务请求超时。")
            if response.status_code == 400:
                text = response.text[:16000].lower()
                if "logprob" in text and "max allowed" in text:
                    raise ServiceError(
                        502,
                        "upstream_logprobs_limit",
                        "模型服务取分上限不足，请核对 LLM_DECISION_MAX_OPTIONS 与上游 max_logprobs 设置。",
                    )
            if response.status_code in (429, 503, 529):
                raise ServiceError(503, "upstream_busy", "模型服务暂时繁忙。")
            raise ServiceError(502, "upstream_rejected", "模型服务未能处理请求，请检查服务端配置。")
        try:
            body = response.json()
        except ValueError:
            raise ServiceError(502, "invalid_upstream_json", "模型服务返回了无效 JSON。") from None
        return parse_scores(body, prepared)
