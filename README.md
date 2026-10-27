# 个人学习伴侣

基于 Python 和 LangChain 的引导式学习 Agent。当前支持命令行对话、分级学习提示、SQLite 会话恢复、长对话摘要，以及个人知识库导入、检索、更新、删除和来源引用。

## 本地 Web 界面

在配置好下文的 `agent_env` 和 `.env` 后运行：

```powershell
python -m learning_companion.web
```

在浏览器打开 http://127.0.0.1:8765，使用 Ctrl+C 停止服务。Web 版复用现有数据库，支持新建和恢复会话、失败后重试、按本地文件路径导入及管理知识库，并显示对话中的来源。服务仅监听本机；同一会话同时只处理一个 Web 请求，也不要同时在命令行中操作同一会话。

如果 8765 端口访问超时或已被占用，可运行 `python -m learning_companion.web --port 8766`，再打开 http://127.0.0.1:8766。

设置页支持填写模型提供方、模型名称、服务地址和 API Key，保存到本机 `.env` 后对下一次请求生效，无需重启 Web 服务。密钥留空会保留原值，保存后不回显；切换提供方时需同时填写对应密钥。由进程环境变量管理的字段需要在启动环境中修改，不能通过页面覆盖。

沙箱由 Agent 按需调用；整轮完成后，回复上方会显示独立的 Python 沙箱结果区，包含执行代码、真实状态、标准输出、错误输出、退出码和耗时。结果随会话保存。环境不可用时，可点击“重新检查沙箱”，或在设置页点击“检查沙箱环境”；检查就绪不代表历史失败的代码已重新执行，需要在对话中请求再次运行。启动代码沙箱前需启动 Docker Desktop，并准备 Python 镜像。

目前回复在整轮完成后显示，尚不支持流式输出、取消请求或实时工具执行状态；可视化功能尚未实现。Web 导入接受单个文件路径，目录导入仍可通过命令行完成。

## 使用 agent_env

使用现有 Conda 环境，不在 base 环境安装依赖：

```powershell
conda activate agent_env
python -m pip install -r requirements-dev.txt
```

本机环境的 Python 路径为 `C:\Users\zcg\.conda\envs\agent_env\python.exe`。如果当前终端无法使用 `conda activate`，可以直接调用该解释器：

```powershell
& "C:\Users\zcg\.conda\envs\agent_env\python.exe" -m pip install -r requirements-dev.txt
& "C:\Users\zcg\.conda\envs\agent_env\python.exe" main.py --help
```

已在该环境的 Python 3.11 上开发。SQLite 随 Python 提供，无需启动数据库服务；LangGraph 的 SQLite 适配包已列入依赖。

## 配置模型

首次使用时复制配置文件；已有 `.env` 时直接编辑，避免覆盖：

```powershell
Copy-Item .env.example .env
```

填写实际使用的模型和密钥：

```dotenv
LLM_PROVIDER=deepseek
LLM_MODEL=deepseek-flash
LLM_API_KEY=你的密钥
LLM_BASE_URL=https://api.deepseek.com
DEEPSEEK_THINKING=disabled
```

支持 DeepSeek 和 OpenAI 两种模型适配。DeepSeek 使用 `langchain-deepseek`，默认地址为 `https://api.deepseek.com`，使用其文档规定的 `max_tokens` 参数；普通学习对话默认关闭思考模式以减少等待。可通过 `DEEPSEEK_THINKING=enabled` 开启，但需要为推理和正文预留足够输出额度。

`LLM_PROVIDER` 支持 `auto`、`deepseek` 和 `openai`。默认 `auto` 会根据 `deepseek-*` 模型名选择 DeepSeek 和正确服务地址，其他模型使用 OpenAI；明确指定提供方时以指定值为准。DeepSeek 模型若误配 OpenAI 官方地址，会在发送请求前报错。

OpenAI 使用 `langchain-openai` 的 Chat Completions 适配器和对应默认地址。自定义网关可填写 `LLM_BASE_URL`；其他协议可在 `learning_companion/agent.py` 的模型工厂中增加适配。DeepSeek 官方调用方式见[首次调用 API](https://api-docs.deepseek.com/zh-cn/)，具体参数见[Chat Completions API](https://api-docs.deepseek.com/zh-cn/api/create-chat-completion/)。

进程环境变量优先于 `.env`。未设置 `LLM_API_KEY` 时，DeepSeek 可以读取 `DEEPSEEK_API_KEY`，OpenAI 可以读取 `OPENAI_API_KEY`，不会将另一家服务的 Key 自动作为后备。`.env` 已被 Git 忽略，程序不打印密钥或服务错误响应正文。

```powershell
python main.py --check
python main.py
```

`--check` 只检查配置，不访问模型。实际对话将发送当前会话上下文到所配置的模型服务。

若输入问题后长时间等待，可单独测试连接，不创建或修改学习会话：

```powershell
python main.py --check-connection
```

连接诊断使用 10 秒的单次网络超时并关闭自动重试。网络超时并非严格的总耗时上限，多个连接阶段可能使实际耗时更长。普通对话会立即显示等待状态、当前超时与重试配置。

如果连接超时，先用 `--check` 确认提供方和服务地址，再确认终端能访问对应 API。确实需要本机代理时，可将实际 HTTP(S) 代理地址长期保存在 `.env` 的 `LLM_PROXY_URL` 中，例如 `http://127.0.0.1:<实际HTTP代理端口>`；不必修改系统环境变量。代理地址与模型的 `LLM_BASE_URL` 是两项不同配置，能够直连时不需要设置代理。程序不会自动猜测或扫描代理端口。

新配置默认 `LLM_TIMEOUT_SECONDS=30`、`LLM_MAX_RETRIES=0`，便于失败后手动 `/retry`。已有 `.env` 中的值继续生效；例如单次 60 秒加一次重试可能造成较长等待。

## 会话与代码输入

```powershell
python main.py --list
python main.py --session <完整会话ID>
python main.py --history <完整会话ID>
```

`--list` 和 `--history` 无需模型密钥。直接启动会新建会话，已有会话使用 `--session` 恢复。对话内可使用：

| 命令 | 功能 |
| --- | --- |
| `/help` | 查看命令 |
| `/new` | 新建会话 |
| `/sessions` | 列出会话 |
| `/use <完整ID>` | 切换会话 |
| `/history` | 查看完整历史 |
| `/retry` | 继续未完成的请求，不重复追加提问 |
| `/paste` | 输入多行内容，以单独一行 `/send` 提交，`/cancel` 取消 |
| `/kb` | 管理知识库，支持 list、import、search、show、delete |
| `/exit` | 退出并显示恢复命令 |

例如使用 `/paste` 输入带缩进的 Python 代码，模型会分析思路并引导定位问题，需要验证时可调用隔离的 Docker 沙箱运行最小示例。

## Python 代码沙箱

本机已安装 Docker Desktop，并使用代理 `http://127.0.0.1:65533` 完成官方 Python 镜像准备。Docker 镜像下载走代理，学习代码容器默认断网。

```powershell
python main.py --sandbox-check
python main.py --sandbox-run examples/sandbox/loop_sum.py
python main.py --sandbox-self-test
```

每次执行使用非 root、只读根目录和临时工作区，不挂载项目文件或传入密钥。默认限制为 5 秒、256 MiB 内存、1 CPU、64 个进程和 8000 字节合计输出，超时或输出超限时强制清理容器。

聊天中的 `run_python` 工具已接入；Docker 不可用时会明确返回失败，不会在宿主 Python 中降级执行。完整配置、代理说明和能力边界见 [Docker 沙箱指南](docs/DOCKER_SANDBOX.md)。

## 个人知识库

```powershell
python main.py --kb-import examples/knowledge/python_copy.md
python main.py --kb-search "浅拷贝"
python main.py --kb-list
python main.py
```

进入聊天后可问：“结合我的笔记，引导我理解浅拷贝为什么会影响嵌套列表。”Agent 可以调用只读检索工具，引用真实片段，并列出本轮检索资料、文件和行号。

支持 Markdown、纯文本和目录递归导入；同一文件重新导入可更新资料。默认使用本地 SQLite FTS5 全文检索，不需要额外密钥；如需语义向量检索，可独立配置 Embedding 服务并切换到混合模式。默认全文检索不等同于语义搜索。

完整命令、配置和数据说明见 [个人知识库使用指南](docs/KNOWLEDGE_BASE.md)。建议将个人原始资料放在已被 Git 忽略的 `data/knowledge/` 下。

## 上下文与数据

- 数据库默认为项目下的 `data/learning.db`，相对路径始终以项目目录为基准。
- 各会话上下文独立，同一会话可重启恢复；不建立跨对话学习画像。主动导入的知识库资料可供不同会话检索。
- LangGraph 检查点负责恢复 Agent 状态；应用表保存会话元数据和完整用户、助手消息。
- 接近摘要阈值时，使用同一个模型压缩较早历史，保留题目、尝试、提示进度和待解决问题，再继续对话。摘要会产生额外模型调用。
- 摘要只影响 Agent 的活动上下文，`/history` 仍可查看原始对话。
- `CONTEXT_SUMMARY_TOKENS` 使用近似 token 计数，不能代替模型服务的精确上下文限制。需要根据所选模型调整阈值、保留消息数、输入和输出长度。
- SQLite 文件和 WAL 辅助文件均被 Git 忽略。本版本面向本地单用户，同一个会话请在一个终端中操作。

## 验证

```powershell
python -m pytest -q
python -m ruff check .
python -m ruff format --check .
python -m pip check
```

如果受限环境不能写默认临时目录，可为 pytest 设置项目内的新临时目录：

```powershell
$testRun = Join-Path (Get-Location).Path ('.pytest_cache/run-' + [guid]::NewGuid().ToString('N'))
python -m pytest -q --basetemp $testRun
```

自动测试使用离线测试模型和本地 HTTP 模拟服务，不消耗真实模型额度。覆盖会话恢复、隔离、摘要、失败恢复、配置校验和命令行模型适配。

当前验证结果：启用 `RUN_DOCKER_TESTS=1` 后完整回归 97 项测试通过，包含本机 HTTP 模拟接口和真实 Docker 的 Agent 工具链路；另有真实容器自检验证隔离、超时、输出和 cgroup 资源限制。Ruff 检查与格式检查通过，`pip check` 未发现依赖冲突。受限执行环境运行 HTTP/Docker 集成测试时需要对应的本机访问权限。

已使用真实 DeepSeek 服务验证直连、认证及原会话恢复后回复。完整教学质量评估仍需按下方用例进行。

经用户批准，已用自带的 Python 容器复制笔记完成真实知识库对话验证：DeepSeek 实际调用检索工具，并返回引用标记、文件名和正确行号，同时以预测问题继续引导。示例已导入本地知识库；测试对话保存在临时数据库。

真实教学效果需要配置模型后按 `docs/TEACHING_EVALUATION.md` 逐项评估；离线测试通过不代表模型一定遵守教学规则。

## 项目结构

```text
main.py                         命令行入口
learning_companion/
  config.py                     模型、数据库和上下文配置
  agent.py                      Agent、摘要和失败恢复
  storage.py                    SQLite 会话与完整历史
  cli.py                        命令行交互
  knowledge.py                  资料快照、切分、全文与可选向量检索
  knowledge_tools.py            只读检索工具与来源归档
  knowledge_cli.py              知识库管理命令
  skills.py                     可选的外部 Skill 协议与工具适配
  sandbox.py                    本地 Docker 执行、资源限制与清理
  sandbox_tools.py              Python 执行工具
  sandbox_cli.py                沙箱诊断与真实隔离自检
  prompts/system.md             引导式教学系统提示词
  prompts/summary.md            学习进度摘要提示词
tests/                          自动测试
docs/TEACHING_EVALUATION.md      教学行为评估用例
docs/KNOWLEDGE_BASE.md           知识库配置与使用
docs/SKILL_INTERFACE.md          外接 Skill 接口约定
docs/DOCKER_SANDBOX.md           本地沙箱、代理和隔离说明
examples/knowledge/             可导入的示例笔记
examples/sandbox/               沙箱运行示例
PLAN.md                         分阶段实施计划
```

Skill 目前只保留可选接入接口，具体内容后续外接，默认不启用。接入约定见 [Skill 扩展接口](docs/SKILL_INTERFACE.md)。Docker 沙箱已接入，后续按 `PLAN.md` 增加可视化和 Web 界面。
