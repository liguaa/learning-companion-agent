import json
import subprocess
import sys
from dataclasses import replace

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from learning_companion import sandbox
from learning_companion.agent import open_companion
from learning_companion.sandbox import (
    DockerSandbox,
    SandboxResult,
    capture_process,
    docker_environment,
)
from tests.conftest import ToolCapableFakeModel


class FakeDocker(DockerSandbox):
    def __init__(self, settings):
        super().__init__(settings)
        self.calls = []
        self.cleanup_fails = False
        self.create_fails = False
        self.volumes = None
        self.state = {"Running": False, "ExitCode": 0, "OOMKilled": False, "Status": "exited"}

    def command(self, args, **kwargs):
        self.calls.append(args)
        code = 0
        if args[:2] == ["context", "inspect"]:
            output = json.dumps("npipe:////./pipe/docker_engine")
        elif args[0] == "info":
            output = '"linux"'
        elif args[:2] == ["image", "inspect"]:
            output = json.dumps(
                [
                    {
                        "Id": "sha256:" + "a" * 64,
                        "Os": "linux",
                        "Config": {"Volumes": self.volumes} if self.volumes else {},
                    }
                ]
            )
        elif args[:2] == ["container", "inspect"]:
            output = json.dumps(self.state)
        elif args[:2] == ["container", "create"]:
            code, output = (1 if self.create_fails else 0), "created"
        elif args[:2] in (["container", "rm"], ["container", "ls"]):
            code, output = (1 if self.cleanup_fails else 0), ""
        else:
            raise AssertionError(args)
        return subprocess.CompletedProcess(args, code, output, "")


@pytest.fixture
def docker(settings, monkeypatch):
    monkeypatch.delenv("DOCKER_CONTEXT", raising=False)
    monkeypatch.delenv("DOCKER_HOST", raising=False)
    monkeypatch.setattr(sandbox, "find_docker", lambda: "docker.exe")
    monkeypatch.setattr(sandbox, "capture_process", lambda *args: ("completed", b"6\n", b"", 0))
    return FakeDocker(settings)


def test_container_has_fixed_isolation_and_no_host_mounts(docker):
    result = docker.run_python("print(6)")
    assert result.status == "success" and result.exit_code == 0 and result.cleanup_confirmed
    created = next(args for args in docker.calls if args[:2] == ["container", "create"])
    for option in (
        "--network=none",
        "--read-only",
        "--user=10001:10001",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--pids-limit=64",
        "--memory=256m",
        "--memory-swap=256m",
        "--cpus=1",
        "--pull=never",
        "--log-driver=none",
    ):
        assert option in created
    assert "--mount" not in created and "--volume" not in created and "-v" not in created
    assert "--env=HTTP_PROXY=" in created and "--entrypoint=python" in created
    assert any(args[:2] == ["container", "rm"] for args in docker.calls)


@pytest.mark.parametrize("status", ["timeout", "output_limit"])
def test_limits_terminate_and_cleanup_container(docker, monkeypatch, status):
    monkeypatch.setattr(sandbox, "capture_process", lambda *args: (status, b"output", b"", -1))
    result = docker.run_python("print(6)")
    assert result.status == status and result.cleanup_confirmed
    assert result.truncated == (status == "output_limit")
    assert any(args[:2] == ["container", "rm"] for args in docker.calls)


def test_oom_is_reported_as_execution_error(docker):
    docker.state.update(ExitCode=137, OOMKilled=True)
    result = docker.run_python("print(6)")
    assert result.status == "error" and result.exit_code == 137
    assert "内存" in result.message


def test_partial_create_failure_still_attempts_cleanup(docker):
    docker.create_fails = True
    result = docker.run_python("print(6)")
    assert result.status == "unavailable" and result.cleanup_confirmed
    assert any(args[:2] == ["container", "rm"] for args in docker.calls)


def test_container_that_never_started_is_not_reported_as_success(docker):
    docker.state["Status"] = "created"
    result = docker.run_python("print(6)")
    assert result.status == "unavailable" and result.cleanup_confirmed


def test_cleanup_failure_is_not_reported_as_success(docker):
    docker.cleanup_fails = True
    result = docker.run_python("print(6)")
    assert result.status == "internal_error" and not result.cleanup_confirmed
    assert result.container_name.startswith("learning-sandbox-")


def test_keyboard_interrupt_still_cleans_container(docker, monkeypatch):
    def interrupt(*args):
        raise KeyboardInterrupt

    monkeypatch.setattr(sandbox, "capture_process", interrupt)
    with pytest.raises(KeyboardInterrupt):
        docker.run_python("print(6)")
    assert any(args[:2] == ["container", "rm"] for args in docker.calls)


def test_remote_daemon_is_rejected(docker, monkeypatch):
    monkeypatch.setenv("DOCKER_HOST", "tcp://remote.example:2375")
    result = docker.run_python("print(6)")
    assert result.status == "unavailable" and "本机" in result.message
    assert docker.calls == []


def test_missing_docker_never_falls_back_to_host_python(settings, monkeypatch):
    monkeypatch.setattr(sandbox, "find_docker", lambda: None)
    result = DockerSandbox(settings).run_python("raise RuntimeError('should not run')")
    assert result.status == "unavailable"
    assert "Docker" in result.message


def test_image_declared_volumes_are_rejected(docker):
    docker.volumes = {"/unsafe-volume": {}}
    result = docker.run_python("print(6)")
    assert result.status == "unavailable"
    assert not any(args[:2] == ["container", "create"] for args in docker.calls)


def test_host_keys_are_not_in_docker_cli_environment(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "host-secret")
    monkeypatch.setenv("OPENAI_API_KEY", "host-secret")
    monkeypatch.setenv("EMBEDDING_API_KEY", "host-secret")
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:65533")
    environment = docker_environment()
    assert all("API_KEY" not in name for name in environment)
    assert "HTTPS_PROXY" not in environment
    assert "host-secret" not in str(environment)


def test_pipe_capture_shares_output_budget_between_streams():
    args = [sys.executable, "-c", "import sys; print('x'*3000); sys.stderr.write('y'*3000)"]
    status, stdout, stderr, _ = capture_process(args, b"", docker_environment(), 3, 1024)
    assert status == "output_limit" and len(stdout) + len(stderr) <= 1024


def test_pipe_capture_has_wall_timeout():
    args = [sys.executable, "-c", "while True: pass"]
    status, _, _, _ = capture_process(args, b"", docker_environment(), 0.2, 1024)
    assert status == "timeout"


def test_pipe_capture_sends_entire_large_input():
    args = [sys.executable, "-c", "import sys; print(len(sys.stdin.buffer.read()))"]
    status, stdout, _, _ = capture_process(args, b"x" * 100000, docker_environment(), 3, 1024)
    assert status == "completed" and stdout.strip() == b"100000"


def test_agent_can_invoke_injected_sandbox(settings):
    class ExampleSandbox:
        def run_python(self, code, stdin=""):
            assert code == "print(6)"
            return SandboxResult(status="success", stdout="6\n", exit_code=0)

    model = ToolCapableFakeModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "run",
                        "name": "run_python",
                        "args": {"code": "print(6)"},
                    }
                ],
            ),
            AIMessage(content="运行输出为 6，你会如何解释？"),
        ]
    )
    with open_companion(settings, model, sandbox_provider=ExampleSandbox()) as companion:
        session = companion.store.create_session()
        assert "输出为 6" in companion.ask(session, "请验证")
        snapshot = companion.graph.get_state({"configurable": {"thread_id": session}})
        result = next(m for m in snapshot.values["messages"] if isinstance(m, ToolMessage))
        assert result.name == "run_python" and json.loads(result.text)["status"] == "success"
        runs = companion.store.sandbox_history(session)
        assert len(runs) == 1 and runs[0]["code"] == "print(6)"
        assert runs[0]["result"]["stdout"] == "6\n"
    from learning_companion.storage import SQLiteStore

    store = SQLiteStore(settings.database_path)
    try:
        assert store.sandbox_history(session) == runs
    finally:
        store.close()


def test_sandbox_disabled_does_not_advertise_execution(settings):
    from tests.conftest import RecordingModel

    model = RecordingModel()
    with open_companion(replace(settings, sandbox_backend="none"), model) as companion:
        session = companion.store.create_session()
        companion.ask(session, "你好")
    assert "应用提供 run_python" not in model.calls[0][0].text


def test_invalid_tool_input_does_not_leave_graph_stuck(settings):
    model = ToolCapableFakeModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "invalid-run",
                        "name": "run_python",
                        "args": {"code": " "},
                    }
                ],
            ),
            AIMessage(content="请先提供需要验证的代码。"),
        ]
    )
    with open_companion(settings, model) as companion:
        session = companion.store.create_session()
        assert "提供" in companion.ask(session, "请验证代码")
        assert companion.store.session(session)["pending_message_id"] is None
