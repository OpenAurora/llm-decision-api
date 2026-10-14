"""Manually check the HTTP API against the configured live model.

Starts and stops its own local server. Never stores credentials or headers.
This is functional integration verification, not a latency or accuracy benchmark.
"""

import argparse
import json
import os
import secrets
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

from llm_decision_api import __version__
from llm_decision_api.config import Settings


def check_answers(body, result):
    assert set(result) == {"model", "answers", "usage"}
    assert set(result["answers"]) == set(body["questions"])
    for question_id, question in body["questions"].items():
        answer = result["answers"][question_id]
        assert answer["type"] == question["type"]
        if question["type"] == "noul":
            assert set(answer) == {"type", "noul"}
            assert type(answer["noul"]) is float and 0 <= answer["noul"] <= 1
            continue
        assert 0 <= answer["confidence"] <= 1
        assert abs(sum(answer["probabilities"].values()) - 1) < 1e-10
        if question["type"] == "choice":
            assert set(answer) == {"type", "choice", "confidence", "probabilities"}
            assert set(answer["probabilities"]) == set(question["criteria"])
        else:
            assert set(answer) == {"type", "score", "confidence", "legend", "probabilities"}
            assert answer["legend"] == {str(i): value for i, value in enumerate(question["criteria"])}
            assert set(answer["probabilities"]) == set(answer["legend"])
    assert result["usage"]["output_tokens"] == len(body["questions"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    settings = Settings()
    model_key = settings.api_key.get_secret_value()
    args.output.mkdir(parents=True, exist_ok=False)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    service_key = secrets.token_urlsafe(24)
    env = os.environ.copy()
    env.update(
        LLM_DECISION_API_KEY=model_key,
        LLM_DECISION_BASE_URL=settings.base_url,
        LLM_DECISION_MODEL=settings.model,
        LLM_DECISION_SERVICE_API_KEY=service_key,
        LLM_DECISION_MAX_OPTIONS="20",
        LLM_DECISION_VERIFY_UPSTREAM_ON_STARTUP="true",
        PYTHONIOENCODING="utf-8",
    )
    process = subprocess.Popen(
        [sys.executable, "-m", "llm_decision_api", "--host", "127.0.0.1", "--port", str(port)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    report = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "kind": "functional_live_http_check",
        "service_version": __version__,
        "model": settings.model,
        "passed": False,
        "checks": [],
    }
    try:
        with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=20, trust_env=False) as client:
            deadline = time.monotonic() + 35
            while True:
                if process.poll() is not None:
                    raise RuntimeError("Local API stopped during startup; inspect the sanitized server log")
                try:
                    if client.get("/docs").status_code == 200:
                        break
                except httpx.RequestError:
                    pass
                if time.monotonic() > deadline:
                    raise RuntimeError("Local API did not become ready")
                time.sleep(0.15)
            print("startup_check_passed", flush=True)

            def post(name, body, authorized=True, expected_status=200):
                started = time.perf_counter()
                response = client.post(
                    "/v1/decide",
                    json=body,
                    headers={"Authorization": "Bearer " + service_key} if authorized else {},
                )
                row = {
                    "name": name,
                    "http_status": response.status_code,
                    "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
                    "response": response.json(),
                }
                report["checks"].append(row)
                if response.status_code != expected_status:
                    raise AssertionError(f"{name}: unexpected HTTP status {response.status_code}")
                if response.status_code == 200:
                    check_answers(body, row["response"])
                    assert row["response"]["model"] == settings.model
                print(json.dumps({"check": name, "status": response.status_code}), flush=True)
                return row["response"]

            route = {
                "type": "choice",
                "instructions": "根据客户当前诉求选择处理部门。",
                "criteria": {"退款部门": None, "技术部门": None, "销售部门": None},
            }
            mixed = {
                "state": {"客户留言": "订单重复扣款，请今天退回多扣部分。", "扣款次数": 2},
                "questions": {
                    "department": route,
                    "refund_requested": {"type": "noul", "instructions": "客户是否明确要求退款？"},
                    "urgency": {
                        "type": "score",
                        "instructions": "按客户明确提出的时间要求判断紧急程度。",
                        "criteria": ["没有期限", {"时间要求": "希望尽快"}, ["明确要求今天处理"]],
                    },
                },
            }
            post("unauthorized", mixed, authorized=False, expected_status=401)
            result = post("object_state_and_three_types", mixed)
            assert result["answers"]["department"]["choice"] == "退款部门"
            assert result["answers"]["refund_requested"]["noul"] > 0.5
            assert 0 <= result["answers"]["urgency"]["score"] <= 2
            assert result["usage"]["output_tokens"] == 3

            # Regression: an explicit exclusion must survive the Noul label mapping.
            development = {
                "type": "noul",
                "instructions": "根据 state.task.name，判断当前任务是否属于开发阶段具体的开发工作。",
                "criteria": {
                    "true": (
                        "任务属于开发或编码阶段，工作内容为具体的业务功能、接口、扩展、报表、"
                        "数据同步或系统对接等开发实现。任务名称可以只描述业务功能或数据对象，"
                        "不要求写出实现动作。预留工作量不属于开发任务。"
                    ),
                    "false": "任务不属于开发或编码阶段的具体开发工作，或属于预留工作量。",
                },
            }
            for name, task, expected in (
                ("noul_reserved_work", {"id": 141, "name": "开发编码-系统集成-RFID集成-新RFID集成改造预留"}, False),
                ("noul_development_work", {"id": 139, "name": "开发编码-系统集成-机队管理系统集成"}, True),
                ("noul_reserved_field", {"name": "开发编码-系统集成-RFID集成-实现接口预留字段映射"}, True),
            ):
                result = post(name, {"state": {"task": task}, "questions": {"task_is_development_work": development}})
                assert (result["answers"]["task_is_development_work"]["noul"] >= 0.5) == expected

            for name, state in (
                ("string_state", "客户说同一订单重复扣款，请退款。"),
                ("array_state", [{"role": "customer", "text": "订单多扣了钱，请退款。"}]),
            ):
                result = post(name, {"state": state, "questions": {"department": route}})
                assert result["answers"]["department"]["choice"] == "退款部门"

            question20 = {
                "type": "choice",
                "instructions": "选择业务编码与 state.target 完全相同的处理选项。",
                "criteria": {f"handler_{i:02}": {"business_code": f"CODE-{i:02}"} for i in range(20)},
            }
            result = post(
                "twenty_candidates", {"state": {"target": "CODE-19"}, "questions": {"route": question20}}
            )
            answer = result["answers"]["route"]
            assert answer["choice"] == "handler_19"
            assert len(answer["probabilities"]) == 20
            assert abs(sum(answer["probabilities"].values()) - 1) < 1e-10
            overflow = question20 | {"criteria": question20["criteria"] | {"extra": None}}
            post(
                "over_configured_limit", {"state": "x", "questions": {"route": overflow}}, expected_status=422
            )
            guide = client.get("/docs")
            assert guide.status_code == 200 and "API 文档" in guide.text
            assert f"v{__version__}" in guide.text
            for removed_route in ("/healthz", "/readyz", "/docs/download", "/openapi.json", "/api-docs"):
                assert client.get(removed_route).status_code == 404
            report["passed"] = True
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        process.terminate()
        try:
            log, _ = process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            log, _ = process.communicate()
        secrets_to_remove = (model_key, service_key)
        log_text = log.decode("utf-8", errors="replace")
        report["completed_utc"] = datetime.now(timezone.utc).isoformat()
        report_text = json.dumps(report, ensure_ascii=False, indent=2)
        for secret in secrets_to_remove:
            log_text = log_text.replace(secret, "[REDACTED]")
            report_text = report_text.replace(secret, "[REDACTED]")
        (args.output / "server.log").write_text(log_text, encoding="utf-8")
        (args.output / "results.json").write_text(report_text, encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "checks": len(report["checks"])}), flush=True)


if __name__ == "__main__":
    main()
