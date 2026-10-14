import json

import httpx
import pytest
from conftest import model_response


@pytest.mark.parametrize("true_mass,false_mass", [(0.02, 0.78), (0.72, 0.08)])
def test_noul_normalizes_t_probability_with_fixed_semantics(client_factory, true_mass, false_mass):
    def handler(request):
        payload = json.loads(request.content)
        content = json.loads(payload["messages"][1]["content"])
        assert content["criteria"] == [
            {"id": "t", "description": "true: 具体开发，排除预留工作量。"},
            {"id": "f", "description": "false: 非开发或预留工作量。"},
        ]
        assert payload["max_tokens"] == 1
        assert payload["structured_outputs"]["choice"] == ["t", "f"]
        return httpx.Response(200, json=model_response(("t", "f"), [true_mass, false_mass]))

    # Reversed user key order must never reverse the meaning of t/f.
    body = {"state": {"task": "资料"}, "questions": {"development": {
        "type": "noul", "instructions": "判断是否为具体开发。",
        "criteria": {"false": "非开发或预留工作量。", "true": "具体开发，排除预留工作量。"},
    }}}
    with client_factory(handler) as client:
        response = client.post("/v1/decide", json=body)
    assert response.status_code == 200
    result = response.json()
    assert result["answers"]["development"] == {
        "type": "noul", "noul": pytest.approx(true_mass / (true_mass + false_mass)),
    }
    assert result["usage"]["output_tokens"] == 1


def test_noul_never_interprets_uppercase_scores_as_lowercase_tf(client_factory):
    def handler(_):
        return httpx.Response(200, json=model_response(("T", "F")))

    with client_factory(handler) as client:
        response = client.post("/v1/decide", json={
            "state": "资料", "questions": {"q": {"type": "noul", "instructions": "条件是否成立？"}},
        })
    assert response.status_code == 502
    assert response.json()["error"]["question_id"] == "q"
