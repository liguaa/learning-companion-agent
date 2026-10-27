"""Python execution in disposable local Linux Docker containers."""

import json
import os
import re
import shutil
import subprocess
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from .config import Settings

MAX_CODE_CHARS = 16000
MAX_STDIN_CHARS = 8000
CONTAINER_LABEL = "learning-companion.sandbox=true"
RUNNER = """import io,json,pathlib,runpy,sys
payload=json.load(sys.stdin)
sys.stdin=io.StringIO(payload["stdin"])
path=pathlib.Path("/work/main.py")
path.write_text(payload["code"],encoding="utf-8")
runpy.run_path(str(path),run_name="__main__")
"""


class SandboxError(ValueError):
    pass


@dataclass
class SandboxResult:
    status: str
    stdout: str = ""
    stderr: str = ""
    exit_code: int | None = None
    duration_seconds: float = 0
    truncated: bool = False
    cleanup_confirmed: bool = True
    message: str = ""
    container_name: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class SandboxProvider(Protocol):
    def run_python(self, code: str, stdin: str = "") -> SandboxResult: ...


def find_docker() -> str | None:
    found = shutil.which("docker")
    if found:
        return found
    roots = [Path("C:/Program Files/Docker/Docker")]
    if os.environ.get("LOCALAPPDATA"):
        roots.insert(0, Path(os.environ["LOCALAPPDATA"]) / "Programs" / "DockerDesktop")
    return next(
        (
            str(root / "resources/bin/docker.exe")
            for root in roots
            if (root / "resources/bin/docker.exe").is_file()
        ),
        None,
    )


def docker_environment() -> dict[str, str]:
    # Host model keys and the rest of the application environment are unnecessary here.
    allowed = {
        "PATH",
        "SYSTEMROOT",
        "WINDIR",
        "COMSPEC",
        "PATHEXT",
        "TEMP",
        "TMP",
        "USERPROFILE",
        "APPDATA",
        "LOCALAPPDATA",
        "HOME",
        "HOMEDRIVE",
        "HOMEPATH",
        "DOCKER_CONFIG",
        "DOCKER_CERT_PATH",
        "DOCKER_TLS_VERIFY",
    }
    environment = {key: value for key, value in os.environ.items() if key.upper() in allowed}
    executable = find_docker()
    if executable:
        # Freshly installed Desktop's credential helper may not be in this terminal's PATH.
        directory = str(Path(executable).parent)
        environment["PATH"] = directory + os.pathsep + environment.get("PATH", "")
    return environment


def capture_process(args: list[str], payload: bytes, env: dict, timeout: float, limit: int):
    """Drain both pipes with a shared byte budget and terminate the CLI on limits."""
    process = subprocess.Popen(
        args,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        bufsize=0,
        shell=False,
    )
    buffers = [bytearray(), bytearray()]
    lock = threading.Lock()
    overflow = threading.Event()
    total = 0

    def read(pipe, index):
        nonlocal total
        try:
            while chunk := pipe.read(4096):
                with lock:
                    remaining = max(0, limit - total)
                    buffers[index].extend(chunk[:remaining])
                    total += min(len(chunk), remaining)
                    if len(chunk) > remaining:
                        overflow.set()
        except (OSError, ValueError):
            pass
        finally:
            pipe.close()

    def write():
        try:
            remaining = memoryview(payload)
            while remaining:
                written = process.stdin.write(remaining)
                if not written:
                    break
                remaining = remaining[written:]
        except (OSError, ValueError):
            pass
        finally:
            process.stdin.close()

    threads = [
        threading.Thread(target=read, args=(process.stdout, 0), daemon=True),
        threading.Thread(target=read, args=(process.stderr, 1), daemon=True),
        threading.Thread(target=write, daemon=True),
    ]
    deadline = time.monotonic() + timeout
    status = "completed"
    try:
        for thread in threads:
            thread.start()
        while process.poll() is None:
            if overflow.is_set():
                status = "output_limit"
                break
            if time.monotonic() >= deadline:
                status = "timeout"
                break
            time.sleep(0.02)
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=3)
        for thread in threads:
            thread.join(timeout=1)
    if overflow.is_set():
        status = "output_limit"
    return status, bytes(buffers[0]), bytes(buffers[1]), process.returncode


class DockerSandbox:
    def __init__(self, settings: Settings):
        self.settings = settings

    def command(self, args: list[str], *, host: str | None = None, timeout: float = 10):
        executable = find_docker()
        if not executable:
            raise SandboxError("未找到 Docker，请先安装并启动 Docker Desktop。")
        prefix = [executable, "--host", host] if host else [executable]
        try:
            return subprocess.run(
                [*prefix, *args],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                env=docker_environment(),
                shell=False,
            )
        except subprocess.TimeoutExpired:
            raise SandboxError(
                "Docker 控制请求超时，请检查 Docker Desktop 是否正常运行。"
            ) from None
        except OSError:
            raise SandboxError("无法启动 Docker 命令，请检查安装和访问权限。") from None

    def local_engine(self) -> str:
        context = os.environ.get("DOCKER_CONTEXT")
        host = os.environ.get("DOCKER_HOST") if not context else None
        if not host:
            args = ["context", "inspect"]
            if context:
                args.append(context)
            result = self.command([*args, "--format", "{{json .Endpoints.docker.Host}}"])
            if result.returncode:
                raise SandboxError("无法读取 Docker context，请确认 Docker Desktop 已完成启动。")
            try:
                host = json.loads(result.stdout)
            except (ValueError, TypeError):
                raise SandboxError("Docker context 返回了无效地址。") from None
        local_pipe = isinstance(host, str) and re.fullmatch(
            r"npipe:/{2,4}\./pipe/[A-Za-z0-9_.-]+", host
        )
        local_socket = isinstance(host, str) and host.startswith("unix:///")
        if not local_pipe and not local_socket:
            raise SandboxError("沙箱仅允许本机 Docker socket，请切换到本地 Docker context。")
        result = self.command(["info", "--format", "{{json .OSType}}"], host=host)
        if result.returncode:
            raise SandboxError("无法连接本机 Docker 引擎，请启动 Docker Desktop 后重试。")
        if result.stdout.strip() != '"linux"':
            raise SandboxError("请将 Docker Desktop 切换为 Linux containers。")
        return host

    def inspect_image(self, host: str) -> str:
        result = self.command(
            [
                "image",
                "inspect",
                self.settings.sandbox_image,
            ],
            host=host,
        )
        if result.returncode:
            raise SandboxError("沙箱镜像尚未准备，请先运行 python main.py --sandbox-prepare。")
        try:
            image = json.loads(result.stdout)[0]
            if image["Os"] != "linux" or image.get("Config", {}).get("Volumes"):
                raise ValueError("Image must not declare writable volumes")
            if not re.fullmatch(r"sha256:[a-f0-9]{64}", image["Id"]):
                raise ValueError("Invalid image identity")
            return image["Id"]
        except (ValueError, TypeError, KeyError, IndexError, AttributeError):
            raise SandboxError("沙箱镜像必须是无预声明数据卷的可信 Linux Python 镜像。") from None

    def check(self) -> dict:
        try:
            host = self.local_engine()
            self.inspect_image(host)
            return {"ready": True, "message": "本机 Linux Docker 引擎和沙箱镜像已就绪。"}
        except SandboxError as error:
            return {"ready": False, "message": str(error)}

    def prepare(self) -> dict:
        host = self.local_engine()
        result = self.command(
            ["image", "pull", "--quiet", self.settings.sandbox_image], host=host, timeout=600
        )
        if result.returncode:
            raise SandboxError(
                "镜像拉取失败，请检查 Docker Desktop 的镜像下载代理；"
                "只设置终端 HTTP_PROXY 或 LLM_PROXY_URL 不会配置 Docker 引擎代理。"
            )
        self.inspect_image(host)
        return {"ready": True, "message": "沙箱镜像已准备，执行代码时不再自动下载镜像。"}

    def create_args(self, name: str, image_id: str) -> list[str]:
        args = [
            "container",
            "create",
            "--name",
            name,
            "--label",
            CONTAINER_LABEL,
            "--pull=never",
            "--interactive",
            "--init",
            "--network=none",
            "--ipc=private",
            "--cgroupns=private",
            "--read-only",
            "--user=10001:10001",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges",
            "--pids-limit=64",
            "--cpus=1",
            f"--memory={self.settings.sandbox_memory_mb}m",
            f"--memory-swap={self.settings.sandbox_memory_mb}m",
            "--log-driver=none",
            "--ulimit=nofile=64:64",
            "--ulimit=fsize=1048576:1048576",
            "--ulimit=core=0",
            "--tmpfs=/work:rw,noexec,nosuid,size=32m,uid=10001,gid=10001,mode=700",
            "--tmpfs=/tmp:rw,noexec,nosuid,size=16m,mode=1777",
            "--workdir=/work",
            "--hostname=learning-sandbox",
            "--env=HOME=/work",
        ]
        # Docker client proxy defaults must not be inherited by learning code.
        for key in (
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "ALL_PROXY",
            "NO_PROXY",
            "http_proxy",
            "https_proxy",
            "all_proxy",
            "no_proxy",
        ):
            args.append(f"--env={key}=")
        return [*args, "--entrypoint=python", image_id, "-I", "-B", "-u", "-c", RUNNER]

    def cleanup(self, host: str, name: str) -> bool:
        try:
            result = self.command(["container", "rm", "--force", name], host=host, timeout=5)
            if result.returncode == 0:
                return True
            remaining = self.command(
                [
                    "container",
                    "ls",
                    "--all",
                    "--filter",
                    f"name=^/{name}$",
                    "--format",
                    "{{.Names}}",
                ],
                host=host,
                timeout=5,
            )
            return remaining.returncode == 0 and not remaining.stdout.strip()
        except SandboxError:
            return False

    def run_python(self, code: str, stdin: str = "") -> SandboxResult:
        if not code.strip() or len(code) > MAX_CODE_CHARS or len(stdin) > MAX_STDIN_CHARS:
            raise SandboxError("代码需为 1～16000 个字符，标准输入最多 8000 个字符。")
        started = time.monotonic()
        name = "learning-sandbox-" + uuid4().hex
        host = None
        may_exist = False
        result = SandboxResult(status="unavailable")
        try:
            host = self.local_engine()
            image_id = self.inspect_image(host)
            may_exist = True
            created = self.command(self.create_args(name, image_id), host=host)
            if created.returncode:
                raise SandboxError("Docker 容器创建失败；请检查引擎是否支持所需的隔离和资源限制。")
            command = [
                find_docker(),
                "--host",
                host,
                "container",
                "start",
                "--attach",
                "--interactive",
                name,
            ]
            status, stdout, stderr, client_exit_code = capture_process(
                command,
                json.dumps({"code": code, "stdin": stdin}).encode("utf-8"),
                docker_environment(),
                self.settings.sandbox_timeout_seconds,
                self.settings.sandbox_output_bytes,
            )
            result = SandboxResult(
                status=status,
                stdout=stdout.decode("utf-8", errors="replace"),
                stderr=stderr.decode("utf-8", errors="replace"),
                truncated=status == "output_limit",
            )
            if status == "completed":
                state = self.command(
                    ["container", "inspect", name, "--format", "{{json .State}}"], host=host
                )
                if state.returncode:
                    raise SandboxError("无法确认容器执行状态，本轮结果不视为成功。")
                details = json.loads(state.stdout)
                if details["Running"]:
                    raise SandboxError("Docker 输出连接中断，容器仍在运行，本轮将强制清理。")
                if details.get("Status") != "exited" or details.get("Error"):
                    raise SandboxError("容器未正常完成启动或退出，本轮结果不视为成功。")
                if client_exit_code not in {0, details["ExitCode"]}:
                    raise SandboxError("Docker 输出连接异常，本轮结果不视为成功。")
                result.exit_code = details["ExitCode"]
                result.status = "success" if result.exit_code == 0 else "error"
                if details.get("OOMKilled"):
                    result.message = "代码超出容器内存限制，已被终止。"
            elif status == "timeout":
                result.message = "代码运行超时，已请求终止并清理容器。"
            else:
                result.message = "代码输出超过限制，已截断并请求终止容器。"
        except SandboxError as error:
            result.status = "unavailable"
            result.message = str(error)
        except Exception:
            result.status = "internal_error"
            result.message = "沙箱执行异常，本轮结果不视为成功。"
        finally:
            if may_exist:
                result.cleanup_confirmed = self.cleanup(host, name)
                if not result.cleanup_confirmed:
                    result.status = "internal_error"
                    result.container_name = name
                    result.message = "无法确认容器已清理，请检查 Docker 并手动移除该容器。"
            result.duration_seconds = round(time.monotonic() - started, 3)
        return result
