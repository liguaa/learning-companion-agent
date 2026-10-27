# 本地 Docker Python 沙箱

学习 Agent 运行于 `agent_env`，学习代码运行于本机 Linux Docker 容器。工具名为 `run_python`，只接受 Python 代码和标准输入，镜像、资源和挂载设置由宿主应用控制。

## 本机已完成的准备

- 已通过 `http://127.0.0.1:65533` 下载 Docker 官方安装包并验证 Docker Inc 签名。
- 已按当前用户安装 Docker Desktop，使用本机已有的 WSL 2 后端。
- 已配置 Docker Desktop 的 HTTP/HTTPS 手动代理为上述地址，拉取官方 `python:3.11-slim` 镜像。
- 应用会在 PATH 之外检查 Docker Desktop 的标准安装位置，并为 Docker 子进程补齐凭据辅助程序的 PATH。

Docker Desktop 关闭后，需要重新启动再使用沙箱。新开终端才能获得安装时更新的 PATH；应用自身也支持在旧终端查找 Docker。

## 使用

在 `agent_env` 中运行：

```powershell
python main.py --sandbox-check
python main.py --sandbox-prepare
python main.py --sandbox-run examples/sandbox/loop_sum.py
python main.py --sandbox-self-test
python main.py
```

`--sandbox-prepare` 通过 Docker 引擎下载镜像，仅在首次准备或主动更新镜像时需要。代码执行阶段使用已有镜像，禁止自动拉取。

聊天中可以使用 `/sandbox` 检查状态，然后要求验证一个最小 Python 示例。模型会调用工具获取真实输出，仍需遵循引导式教学策略。

独立沙箱命令不需要模型密钥，也不向聊天模型发送代码。普通聊天调用沙箱后，其输出会作为对话上下文发送到已配置的模型服务。

## 配置

```dotenv
SANDBOX_BACKEND=docker
SANDBOX_IMAGE=python:3.11-slim
SANDBOX_TIMEOUT_SECONDS=5
SANDBOX_MEMORY_MB=256
SANDBOX_OUTPUT_BYTES=8000
```

运行时间限制适用于容器启动与代码执行阶段；引擎检查、创建和清理另有控制超时，因此总调用耗时可能更长。默认使用 1 个 CPU、64 个进程、256 MiB 内存，禁用额外 swap；输出预算由 stdout 与 stderr 共同使用，按字节计。

可配置执行时间为 1～30 秒、内存为 64～1024 MiB、输出为 1024～32768 字节。`SANDBOX_BACKEND=none` 关闭 Agent 的执行工具；显式调用独立沙箱 CLI 命令仍表示用户主动请求执行。

执行文件可通过 `--sandbox-stdin` 提供输入文本，或在聊天工具调用中使用 `stdin`。当前镜像面向 Python 标准库练习，不在执行过程中联网安装依赖；额外库需由宿主操作者准备可信镜像后配置。

## 代理

镜像下载由 Docker 引擎执行。仅在 `.env` 配置 `LLM_PROXY_URL`、或在终端设置 `HTTP_PROXY`，不能代替 Docker Desktop 的代理配置。

若代理端口改变，在 Docker Desktop 的 Settings → Resources → Proxies 中更新手动 HTTP/HTTPS 代理，并确认 Containers proxy 使用同一代理，然后重新拉取镜像。配置方式见 [Docker Desktop 代理文档](https://docs.docker.com/desktop/settings-and-maintenance/settings/#proxies)。

学习代码容器使用 `--network=none`，不传入模型密钥；同时清空 Docker 客户端可能自动注入的代理变量。下载镜像走代理，执行代码保持断网。

## 隔离与返回结果

- 仅连接本机 named pipe 或 Unix socket，拒绝远程 Docker endpoint。
- 每次解析并固定镜像 ID，覆盖镜像入口，拒绝声明持久数据卷的镜像。
- 使用非 root 用户、只读根目录、移除 capabilities、禁止提升权限，保留 Docker 默认 seccomp。
- 不挂载项目、用户目录或 Docker socket；工作目录和临时目录为有容量上限的 tmpfs。
- 容器禁用持久日志，宿主同时有界读取输出。超时、输出超限、异常和中断时均尝试强制移除本次容器。
- 验证容器实际退出状态，启动失败、输出连接异常或清理未确认均不报告成功。

结果包括 `status`、`stdout`、`stderr`、`exit_code`、耗时、截断标记和 `cleanup_confirmed`。常见状态为 `success`、`error`、`timeout`、`output_limit`、`unavailable`、`invalid_input` 和 `internal_error`。`error` 可以包含真实 Python traceback，便于引导调试。

若 Docker 引擎故障导致无法确认清理，结果会返回本次容器名称。检查 Docker 后仅移除该指定容器，不应使用全局 prune 清理其他工作。

这是面向个人本地学习的容器隔离方案，仍依赖 Docker、WSL、镜像和宿主系统的安全边界。Python 可以在容器内启动子进程，但所有进程都受统一的容器资源与权限限制；工具不提供任意宿主 Shell 接口。

## 验证

`--sandbox-self-test` 验证基本运行、标准输入、Python 运行和语法报错、无限循环超时、输出上限、网络阻断、用户与文件系统隔离、新工作区，以及实际 cgroup v2 的内存、swap、CPU、进程限制；每次检查清理状态。

自动回归默认跳过真实 Docker 用例。准备好本地 Docker 后可显式启用：

```powershell
$env:RUN_DOCKER_TESTS = '1'
python -m pytest -q
```

该集成测试使用离线模型驱动真实 Docker 工具，不调用外部大模型。

沙箱还提供 `SandboxProvider.run_python(code, stdin) -> SandboxResult` 接口，可通过 `open_companion(..., sandbox_provider=...)` 注入其他适配器，未来更换后端时无需修改对话循环。当前只实现本地 Docker 后端；暂不导出容器内文件，文件产物支持随可视化阶段增加。
