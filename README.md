# llm-decision-api

**简体中文** | [English](README.en.md)

基于 **Qwen3.8-27B** 开发与测试的 HTTP 语义判断服务，利用模型候选 token 的概率完成结构化决策。通过 `POST /v1/decide` 对同一份上下文中的多个问题独立判断，返回结构化 JSON。

- **Choice**：从候选项中选择答案，返回候选概率与置信度。
- **Noul**：返回命题在当前候选定义下成立的相对概率。
- **Score**：根据有序等级的概率分布计算评分。

提供中文 API 文档、可选 Bearer 认证、请求大小限制、超时控制及上游候选评分检查。概率与置信度均不等同于实际判断正确率。

测试部署使用的模型标识为 `qwen3.8-27b`，通过兼容 vLLM 扩展协议的模型服务调用。本项目提供 API 服务代码，不包含模型权重；也可配置其他满足协议要求的模型，但下列评测结果仅对应本次 Qwen3.8-27B 部署。

## 对比测试

以下结果来自 **2026-10-08 的历史部署实测**，此次开源整理未重新执行模型评测。对比对象为本服务（`qwen3.8-27b`）、Jev（`jev-1.13.0`）和 Laya（返回标识 `laya-rl-agent`）。模型名称来自当时的请求配置与响应，未独立核查实际权重版本。

### Choice / Noul：参考答案一致率

每服务使用相同的 **492 条中英文输入**，对应 246 道原始问题的两种语言版本。表内为命中题数除以题数，越高越好；Noul 以 0.5 为正负类判定阈值。

| 题组 | llm-decision-api（Qwen3.8-27B） | Jev | Laya |
| --- | ---: | ---: | ---: |
| SemIf · 中文（144 题） | 96.53% | 95.83% | 50.69% |
| TypeSafe · 中文（102 题） | 90.20% | 89.22% | 33.33% |
| SemIf · 英文（144 题） | 97.22% | 97.22% | 59.03% |
| TypeSafe · 英文（102 题） | 87.25% | 88.24% | 36.27% |

SemIf 使用项目编写并经模型复核的参考答案，TypeSafe 使用公开参考模型分布生成的标签；这些一致率不等同于独立人工验证的业务正确率。按源场景或案例成组配对重采样后，本服务与 Jev 的四组主指标差值区间均包含零，当前样本不足以认定两者整体上存在显著优胜关系。

### Choice / Noul：响应耗时

每请求一道问题、串行调用、超时 10 秒、无自动重试；每服务预热 4 次，预热不计入统计。耗时为客户端观测值，包含网络往返，单位为毫秒。

| 服务 | 最低 | 最高 | 平均 | P95 | 失败 / 请求数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| llm-decision-api（Qwen3.8-27B） | 356.67 | 4,204.87 | 750.96 | 1,303.92 | 0 / 492 |
| Jev | 461.29 | 1,532.26 | 532.75 | 707.54 | 0 / 492 |
| Laya | 51.38 | 1,803.91 | 95.53 | 118.25 | 0 / 492 |

本服务平均耗时约 751 ms，86.38% 的请求在 1 秒内完成，但 P95 超过 1 秒。Jev 的平均和 P95 耗时更低。Laya 耗时较低，但该部署的 Noul 结果存在明显正类偏向，原因未定位，不能将此结果概括为其所有模型或部署的能力。

### Score：评分误差与耗时

Score 在同日另行补测，包含 **54 条输入**，对应 27 道原题的中英文版本、15 个源案例，使用 4 或 5 个有序等级。MAE 为相对参考分数的平均绝对误差，越低越好；参考分数来自公开参考概率分布的加权均值。

| 服务 | MAE（分） | 案例等权归一化 MAE | 平均耗时（ms） | P95（ms） | 失败 / 请求数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| llm-decision-api（Qwen3.8-27B） | 0.3580 | 11.53% | 743.54 | 1,202.34 | 0 / 54 |
| Jev | 0.3181 | 10.52% | 503.81 | 532.68 | 0 / 54 |
| Laya | 0.9610 | 26.96% | 88.72 | 189.02 | 0 / 54 |

归一化误差先按各题量程归一化，再在案例内及案例间等权平均；它不是错误率，不能用 `100% − 误差` 得到正确率。本批样本中 Jev 的整体评分最接近参考值，本服务次之。

**适用范围：** 三方硬件、网络、缓存和服务负载未统一控制；中英文输入存在配对关系；两批测试分开统计，未验证并发吞吐、长期稳定性或置信度校准。当前仓库公开方法与汇总结果，未附完整冻结题集和逐次调用记录，因此无法仅凭本仓库完整复现实验。

详细方法、分题型结果、误差区间、数据来源及限制见 [对比测试报告](docs/benchmark-2026-10-08.md)。

## 环境要求

- Python 3.10+ 和 [uv](https://docs.astral.sh/uv/getting-started/installation/)。
- 可访问的模型推理服务，满足下文的上游协议要求。
- 使用 systemd 常驻部署时需要 Linux；本地开发也支持 Windows。

### 上游协议要求

当前后端通过 `POST {LLM_DECISION_BASE_URL}/chat/completions` 调用模型，使用 vLLM 风格的扩展字段。仅支持普通 Chat Completions 的服务不一定兼容。

| 请求或响应要求 | 说明 |
| --- | --- |
| `structured_outputs.choice` | 将输出约束在指定候选标签中 |
| `logprobs=true`、`top_logprobs` | 必须返回全部候选标签的有效分数；默认需要支持 20 个候选 |
| `max_tokens=1` | 每个候选标签必须能作为单个 token 输出 |
| `chat_template_kwargs.enable_thinking=false` | 禁用思考输出，响应只能包含候选标签 |
| `return_tokens_as_token_ids=false` | 返回标签文本，不能以 token ID 替代 |
| `usage` | 返回输入 token 数与一个输出 token 的用量 |

默认启动检查会调用一次上游模型，验证当前配置的候选评分能力；失败则停止启动。更换模型后仍需验证实际任务效果。本项目不包含模型权重，也不声明通用准确率或延迟指标。

## 安装与启动

下载或克隆本仓库后，在项目根目录执行：

```bash
uv sync --frozen
```

将 [.env.example](.env.example) 复制为 `.env`，已有文件请保留：

```bash
# Linux / macOS
cp -n .env.example .env
chmod 600 .env
```

```powershell
# Windows PowerShell
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

编辑 `.env`，填写模型服务的实际配置。示例地址指向本机，示例模型名称为 `qwen3.8-27b`，密钥为占位值。如果上游使用其他模型别名，请将 `LLM_DECISION_MODEL` 改为该服务实际暴露的名称。

| 配置项 | 用途与默认值 |
| --- | --- |
| `LLM_DECISION_BASE_URL` | 上游模型地址；默认 `http://127.0.0.1:8000/v1` |
| `LLM_DECISION_MODEL` | 必填，上游暴露的模型名称；配置示例为 `qwen3.8-27b` |
| `LLM_DECISION_API_KEY` | 必填，调用上游使用的密钥；无认证的本地上游可使用非空占位值 |
| `LLM_DECISION_SERVICE_API_KEY` | 可选，调用本服务的 Bearer 密钥；留空不启用认证 |
| `LLM_DECISION_MAX_OPTIONS` | Choice 选项或 Score 等级数量上限；默认 20，范围 2～62 |
| `LLM_DECISION_BODY_TIMEOUT_SECONDS` | 请求体接收时限；默认 10 秒 |
| `LLM_DECISION_TIMEOUT_SECONDS` | 请求验证后决策处理时限；默认 10 秒 |
| `LLM_DECISION_MAX_BODY_BYTES` | 请求体大小上限；默认 1048576 字节 |
| `LLM_DECISION_VERIFY_UPSTREAM_ON_STARTUP` | 启动时验证候选评分；默认 `true` |
| `LLM_DECISION_LOG_LEVEL` | 日志级别；默认 `INFO` |

进程环境变量优先于 `.env`。上游密钥与服务访问密钥相互独立，修改配置后需重启。`.env` 已被 Git 忽略，请仅提交占位配置示例。

启动本地服务：

```bash
uv run --frozen llm-decision-api --host 127.0.0.1 --port 8010
```

也可执行 `uv run --frozen python -m llm_decision_api`。命令默认监听 `0.0.0.0:8010`；对外开放时应配置服务访问密钥或网关认证，并通过反向代理提供 HTTPS。

| 入口 | 用途 |
| --- | --- |
| `GET http://127.0.0.1:8010/docs` | 中文 API 文档，无需模型调用即可读取 |
| `POST http://127.0.0.1:8010/v1/decide` | 语义判断接口 |

部署到其他主机时使用该主机的访问地址。`/docs` 根据实际访问地址生成调用示例，页面无需外部资源；配置服务密钥后，文档仍可公开读取，但不会展示真实密钥或上游地址。

## API 使用

`POST /v1/decide` 接收 `application/json`，请求体只有 `state` 和 `questions`。启用业务接口认证时，请附带 `Authorization: Bearer <服务访问密钥>`。

- `state`：所有问题共用的上下文，支持字符串、JSON 对象或数组。
- `questions`：以问题 ID 为键的对象，每次 1～32 题。
- `instructions`：每题的判断要求，支持非空字符串、JSON 对象或数组。

### 请求示例

```json
{
  "state": {"message": "同一笔订单扣了两次款，请今天退回多扣部分。"},
  "questions": {
    "department": {
      "type": "choice",
      "instructions": "选择负责处理的部门。",
      "criteria": {"refund": "退款问题", "support": "技术故障"}
    },
    "refund_requested": {
      "type": "noul",
      "instructions": "客户是否明确提出退款要求？"
    },
    "urgency": {
      "type": "score",
      "instructions": "按客户要求的处理时限判断紧急程度。",
      "criteria": ["未指定期限", "尽快处理", "今天处理"]
    }
  }
}
```

| 类型 | `criteria` 规则 | 结果 |
| --- | --- | --- |
| `choice` | 选项名称到描述的映射；2 项起，默认最多 20 项 | `choice`、`confidence`、完整 `probabilities` |
| `noul` | 可省略，也可提供 `true`、`false` 的描述 | 仅 `type` 和数值概率 `noul` |
| `score` | 按等级顺序排列的描述数组；2 档起，默认最多 20 档 | `score`、`confidence`、`legend`、完整 `probabilities` |

Choice 的选项数与 Score 的等级数共用 `LLM_DECISION_MAX_OPTIONS` 上限，超过配置上限返回 HTTP 422。

Choice 的名称与描述共同参与判断，描述支持字符串、对象、数组或 `null`（使用名称）。Noul 仅为缺失的描述补默认值。Score 按从 0 开始的等级下标加权评分，`legend` 保留原始描述及 JSON 结构。

Choice 使用独立的候选分类提示词，要求结合完整业务含义、必要条件、否定、排除条件及例外选择候选。模型输出仍使用内部单 token 标签，服务映射回原始选项名称；调用方的入参及返回格式不变。

Noul 使用独立的命题真假判断提示词，模型输出固定为小写 `t`（true）或 `f`（false）。服务将两者的候选评分归一化，返回 `t` 的概率作为 `noul`。调用方仍使用 `criteria.true`、`criteria.false`，无需填写内部标签。用户提供的描述及其中的否定、排除条件会完整传给模型。

### 返回示例

以下数值仅演示结构。`answers` 与请求中的问题 ID 对应，`usage` 汇总所有问题的模型用量。

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
      "legend": {"0": "未指定期限", "1": "尽快处理", "2": "今天处理"},
      "probabilities": {"0": 0.1, "1": 0.3, "2": 0.6}
    }
  },
  "usage": {"input_tokens": 900, "output_tokens": 3}
}
```

各题独立、并发调用一次模型，输入保留原有 JSON 结构。概率是候选集合内的相对分布，概率与 `confidence` 均不等同于实际判断正确率。

<details>
<summary>评分公式与来源</summary>

`p_i` 为归一化候选概率，`N` 为候选数，`i` 从 0 开始；`m` 为概率最高的首个等级下标。

```text
Choice confidence = (N × max(p_i) - 1) / (N - 1)
Score score = Σ(i × p_i)
Score confidence = 1 - D / U
  D = Σ(p_i × |i-m|)
  U = Σ(|i-(N-1)/2|) / N
```

两种置信度均限制在 0～1。Choice 取最高概率选项，并列时保留模型输出的选项。Noul 不额外返回置信度。

公式依据 [TypeSafe 官方适配器](https://github.com/typesafe-ai/system-one-adapter-python/blob/e1d4cc938204b22fc5a3c3aca7044072fe3f712d/src/system_one_adapter/_utils/confidence_metrics.py)，结构化 `legend` 依据 [官方 SDK 定义](https://github.com/typesafe-ai/typesafe-sdk-python/blob/f078f1e208a0d885154dc758344ae4fce77ac168/src/typesafe_sdk/_schemas/models.py)。本服务独立实现这些计算规则。

</details>

## 错误处理

任意一题失败，整批返回错误并取消未完成任务，不返回部分答案。请求体接收和决策处理使用独立时限，不代表整个 HTTP 请求的总时限。

```json
{"error":{"code":"upstream_timeout","message":"模型服务请求超时。","request_id":"...","question_id":"department"}}
```

每个响应包含 `X-Request-ID`；错误中的 `question_id` 在能定位到具体问题时提供。

| 状态码 | 含义 |
| --- | --- |
| 400 | 无效 JSON |
| 401 | 服务访问密钥无效 |
| 405 | 请求方法不支持，允许的方法见 `Allow` 响应头 |
| 408 | 请求体接收超时：`request_body_timeout` |
| 413 | 请求体超过大小限制 |
| 415 | 请求体媒体类型不支持 |
| 422 | 字段、题目数或候选数量不合法 |
| 500 | 服务内部错误 |
| 502 | 上游调用失败、响应无效或候选评分不完整 |
| 503 | 服务暂不可用 |
| 504 | 模型调用超时或上游返回 504：`upstream_timeout`；决策超时：`decision_timeout` |

## Linux 常驻部署

提供 [llm-decision-api.service](llm-decision-api.service) 示例，使用专用账号运行，部署目录为 `/opt/llm-decision-api`。如果选择其他目录或账号，请同步修改服务文件。

将项目放入部署目录，在该目录执行 `uv sync --frozen --no-dev`，创建 `.env` 并填写实际配置，再创建服务账号、授予读取权限：

```bash
sudo useradd --system --user-group --no-create-home --shell /usr/sbin/nologin llm-decision-api
sudo chgrp -R llm-decision-api /opt/llm-decision-api
sudo chmod -R g+rX /opt/llm-decision-api
sudo chmod 640 /opt/llm-decision-api/.env
```

账号已存在时跳过创建命令。虚拟环境使用的 Python 解释器也必须能被该账号访问；使用 uv 管理的 Python 时，请将其安装到服务账号可访问的目录，或使用系统安装的 Python 创建虚拟环境。

```bash
cd /opt/llm-decision-api
sudo cp llm-decision-api.service /etc/systemd/system/llm-decision-api.service
sudo systemctl daemon-reload
sudo systemctl enable --now llm-decision-api
sudo systemctl status llm-decision-api --no-pager
curl -fsS http://127.0.0.1:8010/docs -o /dev/null
```

服务从工作目录读取 `.env`，直接使用虚拟环境中的启动命令，不依赖 systemd 的 `PATH` 中有 uv。服务账号只需读取项目文件，无需写入权限。

| 操作 | 命令 |
| --- | --- |
| 查看日志 | `sudo journalctl -u llm-decision-api -n 100 --no-pager` |
| 持续查看日志 | `sudo journalctl -u llm-decision-api -f` |
| 重启 | `sudo systemctl restart llm-decision-api` |
| 停止 | `sudo systemctl stop llm-decision-api` |

更新项目时保留服务器 `.env`。依赖变化后执行 `uv sync --frozen --no-dev`，再重启服务；修改 `.service` 后需要重新复制文件并执行 `systemctl daemon-reload`。安装和更新依赖由具有项目目录写入权限的部署账号完成。

## 开发与验证

```bash
uv sync --frozen
uv run --frozen pytest -q
uv run --frozen ruff check src tests
uv build
```

pytest 使用模拟上游，不需要真实模型或密钥。验证真实模型时，先配置 `.env`，再运行手动集成检查，将输出写入工程外的新目录：

```bash
uv run --frozen python tests/integration/smoke_live.py --output ../llm-decision-smoke-results
```

脚本会临时启动本地服务并调用已配置的模型，报告包含模型标识和响应结果，请按自己的数据公开范围管理这些产物。

[examples/decide.json](examples/decide.json) 是虚构业务场景，可用于发送包含三种题型的请求：

```bash
curl http://127.0.0.1:8010/v1/decide \
  -H 'Content-Type: application/json' \
  --data-binary @examples/decide.json
```

启用服务认证时，增加 `Authorization: Bearer <服务访问密钥>` 请求头。PowerShell 中可使用 `curl.exe` 并写成单行。

项目结构：

```text
src/llm_decision_api/       服务代码与中文 API 文档页面
tests/                      自动化回归测试
tests/integration/          手动真实模型检查
examples/                   虚构请求示例
docs/                       对比测试报告与方法说明
.env.example                配置模板
llm-decision-api.service     systemd 部署示例
```

安装包、启动命令、systemd 服务统一使用 `llm-decision-api`，Python 导入使用 `llm_decision_api`，配置统一使用 `LLM_DECISION_` 前缀。

## 许可证

采用 [MIT License](LICENSE)。
