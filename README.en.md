# llm-decision-api

[简体中文](README.md) | **English**

An HTTP API for semantic decisions, developed and tested with **Qwen3.8-27B**. It uses candidate token probabilities to produce structured decisions. Send a shared context and a set of independent questions to `POST /v1/decide` and receive structured JSON answers.

- **Choice**: select an option and return candidate probabilities and confidence.
- **Noul**: return the relative probability that a proposition holds under the given candidate definitions.
- **Score**: compute a score from the probability distribution over ordered levels.

Features include Chinese API documentation, optional Bearer authentication, request size limits, timeouts, and upstream candidate scoring checks. Neither candidate probabilities nor confidence values are calibrated probabilities of a correct decision.

The tested deployment used the model identifier `qwen3.8-27b`, served through an endpoint supporting vLLM-style extensions. This repository contains the API service, not model weights. Other models can be configured if they meet the protocol requirements, but the benchmark results below apply only to the tested Qwen3.8-27B deployment.

## Benchmarks

These results come from **historical deployment tests on October 8, 2026**. Model benchmarks were not rerun when preparing this public repository. The comparison covers this service (`qwen3.8-27b`), Jev (`jev-1.13.0`), and Laya (response model identifier `laya-rl-agent`). Model identifiers were recorded from request configurations and responses; the actual model weights were not independently verified.

### Choice / Noul: agreement with reference answers

Each service received the same **492 Chinese and English inputs**, representing two language versions of 246 original questions. The table reports the percentage of answers matching the reference labels; higher is better. Noul predictions were classified using a probability threshold of 0.5.

| Question group | llm-decision-api (Qwen3.8-27B) | Jev | Laya |
| --- | ---: | ---: | ---: |
| SemIf · Chinese (144 questions) | 96.53% | 95.83% | 50.69% |
| TypeSafe · Chinese (102 questions) | 90.20% | 89.22% | 33.33% |
| SemIf · English (144 questions) | 97.22% | 97.22% | 59.03% |
| TypeSafe · English (102 questions) | 87.25% | 88.24% | 36.27% |

SemIf reference answers were authored by the project and reviewed with model assistance. TypeSafe labels were derived from public reference model distributions. Agreement with these labels is not equivalent to independently human-validated accuracy on real business tasks. Paired resampling by source scenario or case produced intervals containing zero for the differences between this service and Jev across all four primary group metrics. This sample does not establish a statistically significant overall advantage for either service.

### Choice / Noul: response latency

Requests were sent sequentially, with one question per request, a 10-second timeout, and no automatic retries. Each service received four warm-up requests, excluded from the statistics. Latencies are client-observed values in milliseconds and include network round trips.

| Service | Min | Max | Mean | P95 | Failures / requests |
| --- | ---: | ---: | ---: | ---: | ---: |
| llm-decision-api (Qwen3.8-27B) | 356.67 | 4,204.87 | 750.96 | 1,303.92 | 0 / 492 |
| Jev | 461.29 | 1,532.26 | 532.75 | 707.54 | 0 / 492 |
| Laya | 51.38 | 1,803.91 | 95.53 | 118.25 | 0 / 492 |

This service averaged approximately 751 ms, with 86.38% of requests completing in under one second, while P95 exceeded one second. Jev had lower mean and P95 latency. Laya had lower latency but a strong positive-class bias in Noul predictions in this deployment. Its cause remains unresolved, and these results should not be generalized to all Laya models or deployments.

### Score: scoring error and latency

Score was evaluated in a separate run on the same day, using **54 inputs**: Chinese and English versions of 27 original questions from 15 source cases, each with four or five ordered levels. MAE is the mean absolute difference from the reference score, in score units; lower is better. Reference scores were calculated as weighted means of public reference probability distributions.

| Service | MAE (score units) | Case-balanced normalized MAE | Mean latency (ms) | P95 (ms) | Failures / requests |
| --- | ---: | ---: | ---: | ---: | ---: |
| llm-decision-api (Qwen3.8-27B) | 0.3580 | 11.53% | 743.54 | 1,202.34 | 0 / 54 |
| Jev | 0.3181 | 10.52% | 503.81 | 532.68 | 0 / 54 |
| Laya | 0.9610 | 26.96% | 88.72 | 189.02 | 0 / 54 |

Normalized errors are divided by each question's score range, averaged within each case, and then averaged equally across cases. This metric is not a classification error rate: `100% − error` does not yield accuracy. In this sample, Jev's overall scores were closest to the reference values, followed by this service.

**Scope and limitations:** hardware, networks, caches, and service load were not controlled across deployments. Chinese and English inputs are paired rather than independent samples. The two test runs were analyzed separately; concurrent throughput, long-term stability, and confidence calibration were not evaluated. This repository publishes the methodology and aggregate results, but not the complete frozen datasets or per-request records, so it is insufficient to reproduce the full experiment on its own.

See the [detailed benchmark report (Chinese)](docs/benchmark-2026-10-08.md) for methods, results by question type, uncertainty intervals, data sources, and limitations.

## Requirements

- Python 3.10+ and [uv](https://docs.astral.sh/uv/getting-started/installation/).
- An accessible model inference server that meets the upstream protocol requirements below.
- Linux for deployment with systemd. Windows is also supported for local development.

### Upstream protocol requirements

The backend calls `POST {LLM_DECISION_BASE_URL}/chat/completions` with vLLM-style extension fields. A server that supports only standard Chat Completions may not be compatible.

| Request or response requirement | Details |
| --- | --- |
| `structured_outputs.choice` | Constrain the output to the specified candidate labels |
| `logprobs=true`, `top_logprobs` | Return valid scores for every candidate label; the default configuration requires support for 20 candidates |
| `max_tokens=1` | Each candidate label must be emitted as a single token |
| `chat_template_kwargs.enable_thinking=false` | Disable thinking output; the response must contain only a candidate label |
| `return_tokens_as_token_ids=false` | Return label text rather than token IDs |
| `usage` | Return the input token count and a single output token |

By default, startup makes one upstream model call to verify candidate scoring at the configured capacity. The service will not start if this check fails. Task performance must still be evaluated when switching models. This repository does not include model weights or claim universal accuracy or latency figures.

## Installation and startup

After downloading or cloning this repository, run the following in the project root:

```bash
uv sync --frozen
```

Copy [.env.example](.env.example) to `.env`, preserving any existing configuration:

```bash
# Linux / macOS
cp -n .env.example .env
chmod 600 .env
```

```powershell
# Windows PowerShell
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

Edit `.env` with your model server's configuration. The example URL points to localhost, the example model name is `qwen3.8-27b`, and the API key is a placeholder. If your upstream server uses a different model alias, set `LLM_DECISION_MODEL` to the exact name exposed by that server.

| Setting | Purpose and default |
| --- | --- |
| `LLM_DECISION_BASE_URL` | Upstream model URL; defaults to `http://127.0.0.1:8000/v1` |
| `LLM_DECISION_MODEL` | Required model name exposed by the upstream server; the example uses `qwen3.8-27b` |
| `LLM_DECISION_API_KEY` | Required upstream API key; a nonempty placeholder can be used for a local upstream without authentication |
| `LLM_DECISION_SERVICE_API_KEY` | Optional Bearer key for this API; empty disables authentication |
| `LLM_DECISION_MAX_OPTIONS` | Maximum Choice options or Score levels; default 20, allowed range 2–62 |
| `LLM_DECISION_BODY_TIMEOUT_SECONDS` | Request body reception timeout; default 10 seconds |
| `LLM_DECISION_TIMEOUT_SECONDS` | Decision processing timeout after request validation; default 10 seconds |
| `LLM_DECISION_MAX_BODY_BYTES` | Maximum request body size; default 1048576 bytes |
| `LLM_DECISION_VERIFY_UPSTREAM_ON_STARTUP` | Verify candidate scoring at startup; default `true` |
| `LLM_DECISION_LOG_LEVEL` | Logging level; default `INFO` |

Process environment variables take precedence over `.env`. The upstream API key and the service access key are independent. Restart the service after configuration changes. Git ignores `.env`; commit only example configurations containing placeholders.

Start the service locally:

```bash
uv run --frozen llm-decision-api --host 127.0.0.1 --port 8010
```

You can also run `uv run --frozen python -m llm_decision_api`. The command defaults to listening on `0.0.0.0:8010`. When exposing it to a network, configure the service access key or gateway authentication and provide HTTPS through a reverse proxy.

| Endpoint | Purpose |
| --- | --- |
| `GET http://127.0.0.1:8010/docs` | Chinese API documentation; reading it does not invoke the model |
| `POST http://127.0.0.1:8010/v1/decide` | Semantic decision API |

For a deployment on another host, use that host's address. `/docs` generates examples using the address through which it is accessed and does not require external resources. Documentation remains publicly readable when a service key is configured, but never displays real keys or the upstream URL.

## API usage

`POST /v1/decide` accepts `application/json`, with only `state` and `questions` at the top level. When service authentication is enabled, include `Authorization: Bearer <service-access-key>`.

- `state`: context shared by all questions; a string, JSON object, or array.
- `questions`: an object keyed by question ID, with 1–32 questions per request.
- `instructions`: the task for each question; a nonempty string, JSON object, or array.

### Example request

```json
{
  "state": {"message": "I was charged twice for the same order. Please refund the extra charge today."},
  "questions": {
    "department": {
      "type": "choice",
      "instructions": "Choose the department responsible for handling this request.",
      "criteria": {"refund": "Refund issues", "support": "Technical issues"}
    },
    "refund_requested": {
      "type": "noul",
      "instructions": "Has the customer explicitly requested a refund?"
    },
    "urgency": {
      "type": "score",
      "instructions": "Assess urgency based on the customer's requested deadline.",
      "criteria": ["No deadline specified", "As soon as possible", "Today"]
    }
  }
}
```

| Type | `criteria` rules | Output |
| --- | --- | --- |
| `choice` | Map option names to descriptions; at least 2 options, up to 20 by default | `choice`, `confidence`, and the full `probabilities` distribution |
| `noul` | Optional; may provide descriptions for `true` and `false` | Only `type` and the numeric probability `noul` |
| `score` | Array of descriptions in level order; at least 2 levels, up to 20 by default | `score`, `confidence`, `legend`, and the full `probabilities` distribution |

Choice option counts and Score level counts share the `LLM_DECISION_MAX_OPTIONS` limit. Exceeding the configured limit returns HTTP 422.

Both Choice option names and descriptions contribute to the decision. Descriptions may be strings, objects, arrays, or `null` (which uses the option name). Noul supplies defaults only for missing descriptions. Score uses zero-based level indices to calculate the weighted score; `legend` preserves the original descriptions and JSON structure.

Choice uses a dedicated classification prompt that considers the full meaning of each option, necessary conditions, negations, exclusions, and exceptions. The model returns internal single-token labels, which the service maps back to the original option names; callers do not need to use these labels.

Noul uses a dedicated proposition-evaluation prompt with fixed lowercase output labels: `t` for true and `f` for false. The service normalizes the two candidate scores and returns the probability of `t` as `noul`. Callers use `criteria.true` and `criteria.false` rather than the internal labels. User-supplied descriptions, including negations and exclusions, are passed to the model in full.

### Example response

The numbers below illustrate the response structure. Keys in `answers` match the request's question IDs, and `usage` aggregates model usage across all questions.

```json
{
  "model": "qwen3.8-27b",
  "answers": {
    "department": {
      "type": "choice",
      "choice": "refund",
      "confidence": 0.8,
      "probabilities": {"refund": 0.9, "support": 0.1}
    },
    "refund_requested": {"type": "noul", "noul": 0.96},
    "urgency": {
      "type": "score",
      "score": 1.5,
      "confidence": 0.25,
      "legend": {"0": "No deadline specified", "1": "As soon as possible", "2": "Today"},
      "probabilities": {"0": 0.1, "1": 0.3, "2": 0.6}
    }
  },
  "usage": {"input_tokens": 900, "output_tokens": 3}
}
```

Each question triggers one independent model call, with calls executed concurrently. Inputs retain their original JSON structure. Probabilities describe the relative distribution within the candidate set; neither probabilities nor `confidence` are the actual probability that an answer is correct.

<details>
<summary>Scoring formulas and references</summary>

Let `p_i` be the normalized candidate probability, `N` the number of candidates, and `i` a zero-based index. Let `m` be the first level index with the highest probability.

```text
Choice confidence = (N × max(p_i) - 1) / (N - 1)
Score score = Σ(i × p_i)
Score confidence = 1 - D / U
  D = Σ(p_i × |i-m|)
  U = Σ(|i-(N-1)/2|) / N
```

Both confidence metrics are clamped to 0–1. Choice selects the option with the highest probability, preserving the model's selected option in a tie. Noul does not return a separate confidence value.

The formulas follow the [official TypeSafe adapter](https://github.com/typesafe-ai/system-one-adapter-python/blob/e1d4cc938204b22fc5a3c3aca7044072fe3f712d/src/system_one_adapter/_utils/confidence_metrics.py). Structured `legend` values follow the [official SDK schema](https://github.com/typesafe-ai/typesafe-sdk-python/blob/f078f1e208a0d885154dc758344ae4fce77ac168/src/typesafe_sdk/_schemas/models.py). This service implements these rules independently.

</details>

## Error handling

If any question fails, the entire batch returns an error and cancels unfinished tasks. Partial answers are not returned. Body reception and decision processing have separate time limits; neither is a total deadline for the entire HTTP request.

```json
{"error":{"code":"upstream_timeout","message":"模型服务请求超时。","request_id":"...","question_id":"department"}}
```

Error messages are currently returned in Chinese; this example means that the upstream model request timed out. Each response includes `X-Request-ID`. Errors include `question_id` when a specific question can be identified.

| Status | Meaning |
| --- | --- |
| 400 | Invalid JSON |
| 401 | Invalid service access key |
| 405 | Unsupported HTTP method; see the `Allow` response header |
| 408 | Request body reception timeout: `request_body_timeout` |
| 413 | Request body exceeds the size limit |
| 415 | Unsupported request body media type |
| 422 | Invalid fields, question count, or candidate count |
| 500 | Internal server error |
| 502 | Upstream request failure, invalid response, or incomplete candidate scores |
| 503 | Service temporarily unavailable |
| 504 | Model request timeout or upstream HTTP 504: `upstream_timeout`; decision deadline exceeded: `decision_timeout` |

## Linux deployment with systemd

The [llm-decision-api.service](llm-decision-api.service) example runs under a dedicated account from `/opt/llm-decision-api`. Update the service file if you use a different directory or account.

Place the project in that directory, run `uv sync --frozen --no-dev`, and create and configure `.env`. Then create the service account and grant read access:

```bash
sudo useradd --system --user-group --no-create-home --shell /usr/sbin/nologin llm-decision-api
sudo chgrp -R llm-decision-api /opt/llm-decision-api
sudo chmod -R g+rX /opt/llm-decision-api
sudo chmod 640 /opt/llm-decision-api/.env
```

Skip account creation if the account already exists. The service account must also be able to access the Python interpreter used by the virtual environment. For a uv-managed Python installation, install the interpreter in an accessible directory or create the virtual environment with a system-installed Python.

```bash
cd /opt/llm-decision-api
sudo cp llm-decision-api.service /etc/systemd/system/llm-decision-api.service
sudo systemctl daemon-reload
sudo systemctl enable --now llm-decision-api
sudo systemctl status llm-decision-api --no-pager
curl -fsS http://127.0.0.1:8010/docs -o /dev/null
```

The service reads `.env` from its working directory and invokes the executable in the virtual environment directly; uv does not need to be on systemd's `PATH`. The service account needs read access to the project, not write access.

| Operation | Command |
| --- | --- |
| Recent logs | `sudo journalctl -u llm-decision-api -n 100 --no-pager` |
| Follow logs | `sudo journalctl -u llm-decision-api -f` |
| Restart | `sudo systemctl restart llm-decision-api` |
| Stop | `sudo systemctl stop llm-decision-api` |

Preserve the server's `.env` when updating the project. After dependency changes, run `uv sync --frozen --no-dev` and restart the service. After editing `.service`, copy it again and run `systemctl daemon-reload`. Dependency installation and updates should be performed by a deployment account with write access to the project directory.

## Development and validation

```bash
uv sync --frozen
uv run --frozen pytest -q
uv run --frozen ruff check src tests
uv build
```

Tests use a simulated upstream and do not require a real model or credentials. To check a live model, configure `.env`, then run the manual integration check with a new output directory outside the project:

```bash
uv run --frozen python tests/integration/smoke_live.py --output ../llm-decision-smoke-results
```

The script temporarily starts a local service and calls the configured model. Reports include the model identifier and response data; review these artifacts before sharing them.

[examples/decide.json](examples/decide.json) contains a fictional customer-service scenario in Chinese with all three question types:

```bash
curl http://127.0.0.1:8010/v1/decide \
  -H 'Content-Type: application/json' \
  --data-binary @examples/decide.json
```

When service authentication is enabled, add an `Authorization: Bearer <service-access-key>` header. In PowerShell, use `curl.exe` and put the command on one line.

Project layout:

```text
src/llm_decision_api/       Service code and Chinese API documentation page
tests/                      Automated regression tests
tests/integration/          Manual live-model checks
examples/                   Fictional request examples
docs/                       Benchmark report and methodology
.env.example                Configuration template
llm-decision-api.service     Example systemd unit
```

The distribution, CLI command, and systemd service are named `llm-decision-api`. The Python package is `llm_decision_api`, and configuration variables use the `LLM_DECISION_` prefix.

## License

Released under the [MIT License](LICENSE).
