"""Chinese API reference with examples matching the deployed configuration."""

import json
import re
import shlex
from html import escape
from pathlib import Path

from . import __version__
from .config import Settings
from .schemas import MAX_QUESTIONS

GUIDE_TEMPLATE = Path(__file__).with_name("api_reference.html").read_text(encoding="utf-8")
DOCS_BASE_URL = "http://127.0.0.1:8010"


def render_guide(settings: Settings, base_url: str = DOCS_BASE_URL) -> str:
    state = {"message": "同一笔订单扣了两次款，请今天退回多扣部分。"}
    questions = {
        "department": {
            "type": "choice",
            "instructions": "选择负责处理的部门。",
            "criteria": {"refund": "退款问题", "support": "技术故障"},
        },
        "refund_requested": {"type": "noul", "instructions": "客户是否明确提出退款要求？"},
        "urgency": {
            "type": "score",
            "instructions": "按客户要求的处理时限判断紧急程度。",
            "criteria": ["未指定期限", "尽快处理", "今天处理"],
        },
    }
    answers = {
        "department": {
            "type": "choice",
            "choice": "refund",
            "confidence": 0.8,
            "probabilities": {"refund": 0.9, "support": 0.1},
        },
        "refund_requested": {"type": "noul", "noul": 0.96},
        "urgency": {
            "type": "score",
            "score": 1.5,
            "confidence": 0.25,
            "legend": {"0": "未指定期限", "1": "尽快处理", "2": "今天处理"},
            "probabilities": {"0": 0.1, "1": 0.3, "2": 0.6},
        },
    }
    if settings.max_options == 2:
        questions["urgency"]["criteria"] = ["未指定期限", "今天处理"]
        answers["urgency"] = {
            "type": "score",
            "score": 0.8,
            "confidence": 0.6,
            "legend": {"0": "未指定期限", "1": "今天处理"},
            "probabilities": {"0": 0.2, "1": 0.8},
        }

    def pretty(value):
        return json.dumps(value, ensure_ascii=False, indent=2)

    values = {
        "version": __version__,
        "base_url": base_url.rstrip("/"),
        "max_questions": str(MAX_QUESTIONS),
        "max_options": str(settings.max_options),
        "max_body_bytes": f"{settings.max_body_bytes:,}",
        "body_timeout": f"{settings.body_timeout_seconds:g}",
        "decision_timeout": f"{settings.timeout_seconds:g}",
        "auth_note": (
            "需要 Bearer 认证。示例中的 YOUR_SERVICE_KEY 为服务访问密钥。"
            if settings.service_api_key
            else "当前服务未启用访问密钥认证。"
        ),
    }
    for question_id, question in questions.items():
        kind = question["type"]
        values[f"{kind}_question"] = pretty(question)
        values[f"{kind}_answer"] = pretty(answers[question_id])
    endpoint = base_url.rstrip("/") + "/v1/decide"
    headers = {"Content-Type": "application/json"}
    if settings.service_api_key:
        headers["Authorization"] = "Bearer YOUR_SERVICE_KEY"
    values["endpoint"] = "\n".join([
        "POST " + endpoint,
        *(f"{key}: {value}" for key, value in headers.items()),
    ])
    body = {"state": state, "questions": {"refund_requested": questions["refund_requested"]}}
    request = pretty(body)
    values["response_example"] = pretty({
        "model": "your-model",
        "answers": {"refund_requested": answers["refund_requested"]},
        "usage": {"input_tokens": 300, "output_tokens": 1},
    })
    values["curl_example"] = " \\\n".join([
        "curl " + shlex.quote(endpoint),
        *(f"  -H {shlex.quote(key + ': ' + value)}" for key, value in headers.items()),
        "  -d " + shlex.quote(request),
    ])
    # This example contains only strings and objects, also valid as Python literals.
    values["python_example"] = (
        "import json\n"
        "from urllib.request import Request, urlopen\n\n"
        f"body = {request}\n\n"
        "request = Request(\n"
        f"    {endpoint!r},\n"
        '    data=json.dumps(body).encode("utf-8"),\n'
        f"    headers={headers!r},\n"
        '    method="POST",\n'
        ")\n"
        "with urlopen(request, timeout=30) as response:\n"
        '    print(json.load(response)["answers"])'
    )

    return re.sub(r"\{\{([a-z_]+)\}\}", lambda match: escape(values[match[1]], quote=True), GUIDE_TEMPLATE)
