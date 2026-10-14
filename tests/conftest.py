import json
import math

import httpx
import pytest
from fastapi.testclient import TestClient

from llm_decision_api.api import create_app
from llm_decision_api.backend import ModelBackend
from llm_decision_api.config import Settings


def model_response(labels=("A", "B"), probabilities=None, selected=None, input_tokens=40):
    if probabilities is None:
        probabilities = [0.8] + [0.2 / (len(labels) - 1)] * (len(labels) - 1)
    values = [math.log(value) if value else -1000.0 for value in probabilities]
    if selected is None:
        selected = labels[max(range(len(values)), key=values.__getitem__)]
    return {
        "choices": [
            {
                "finish_reason": "length",
                "message": {"role": "assistant", "content": selected},
                "logprobs": {
                    "content": [
                        {
                            "token": selected,
                            "logprob": values[labels.index(selected)],
                            "top_logprobs": [
                                {"token": label, "logprob": value} for label, value in zip(labels, values)
                            ],
                        }
                    ]
                },
            }
        ],
        "usage": {"prompt_tokens": input_tokens, "completion_tokens": 1, "total_tokens": input_tokens + 1},
    }


@pytest.fixture
def settings():
    return Settings(
        _env_file=None,
        api_key="private-model-key",
        model="test-model",
        base_url="http://model.invalid/v1",
        verify_upstream_on_startup=False,
    )


@pytest.fixture
def body():
    return {
        "state": {"customer_message": "请退回重复扣款", "charge_count": 2},
        "questions": {
            "department": {
                "type": "choice",
                "instructions": "分配处理部门。",
                "criteria": {"refund": None, "technical_support": {"responsibility": "处理软件故障"}},
            }
        },
    }


@pytest.fixture
def client_factory(settings):
    def build(handler=None, **overrides):
        cfg = Settings(**(settings.model_dump() | overrides), _env_file=None)
        if handler is None:

            def handler(request):
                labels = json.loads(request.content)["structured_outputs"]["choice"]
                return httpx.Response(200, json=model_response(labels))

        app = create_app(
            cfg, backend_factory=lambda config: ModelBackend(config, transport=httpx.MockTransport(handler))
        )
        return TestClient(app)

    return build
