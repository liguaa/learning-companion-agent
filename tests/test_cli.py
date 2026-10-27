import json
import os
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import pytest
from openai import APITimeoutError

from learning_companion import cli
from learning_companion.config import PROJECT_ROOT, Settings
from tests.conftest import RecordingModel


@pytest.fixture
def model_server():
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            assert self.path == "/v1/chat/completions"
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(request)
            body = json.dumps(
                {
                    "id": "test-completion",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "test-model",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "你准备如何验证这个思路？"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
                },
                ensure_ascii=False,
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def run_cli(env, *args, content=""):
    return subprocess.run(
        [sys.executable, "main.py", "--env-file", str(PROJECT_ROOT / "missing-test.env"), *args],
        input=content,
        encoding="utf-8",
        capture_output=True,
        cwd=PROJECT_ROOT,
        env={**env, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"},
        timeout=30,
    )


@pytest.mark.parametrize("provider", ["openai", "deepseek"])
def test_cli_real_adapter_multiline_and_restart(tmp_path, model_server, provider):
    url, requests = model_server
    env = {
        **os.environ,
        "DATABASE_PATH": str(tmp_path / "cli.db"),
        "LLM_API_KEY": "local-test-secret",
        "LLM_MODEL": "test-model",
        "LLM_PROVIDER": provider,
        "DEEPSEEK_THINKING": "disabled",
        "LLM_BASE_URL": url,
        "LLM_PROXY_URL": "",
        "LLM_MAX_RETRIES": "0",
        "LANGSMITH_TRACING": "false",
        "LANGCHAIN_TRACING_V2": "false",
    }
    probe = run_cli(env, "--check-connection")
    assert probe.returncode == 0, probe.stderr
    assert "连接成功" in probe.stdout
    assert not (tmp_path / "cli.db").exists()
    requests.clear()
    first = run_cli(
        env, content="/paste\n我在学循环\nfor i in range(3):\n    print(i)\n/send\n/exit\n"
    )
    assert first.returncode == 0, first.stderr
    assert "你准备如何验证" in first.stdout
    assert first.stdout.index("正在请求模型") < first.stdout.index("你准备如何验证")
    listed = run_cli(env, "--list")
    assert listed.returncode == 0, listed.stderr
    session_id = listed.stdout.split()[0]
    second = run_cli(env, "--session", session_id, content="继续刚才的问题\n/exit\n")
    assert second.returncode == 0, second.stderr
    assert len(requests) == 2
    assert requests[0]["messages"][0]["role"] == "system"
    if provider == "deepseek":
        assert requests[0]["max_tokens"] > 0
        assert "max_completion_tokens" not in requests[0]
        assert requests[0]["thinking"] == {"type": "disabled"}
    assert "    print(i)" in requests[1]["messages"][1]["content"]
    assert len(requests[1]["messages"]) == 4
    assert "local-test-secret" not in first.stdout + first.stderr + second.stderr

    env["LLM_API_KEY"] = ""
    env["OPENAI_API_KEY"] = ""
    history = run_cli(env, "--history", session_id)
    assert history.returncode == 0
    assert "我在学循环" in history.stdout
    assert "继续刚才的问题" in history.stdout


def test_missing_config_is_actionable_and_does_not_create_database(tmp_path):
    database = tmp_path / "not-created.db"
    env = {
        **os.environ,
        "LLM_MODEL": "",
        "LLM_API_KEY": "",
        "OPENAI_API_KEY": "",
        "DATABASE_PATH": str(database),
    }
    result = run_cli(env)
    assert result.returncode == 2
    assert ".env.example" in result.stderr
    assert not database.exists()


def test_waiting_status_is_flushed_before_model_call(settings, capsys):
    def respond():
        output = capsys.readouterr().out
        assert "正在请求模型" in output
        assert "Ctrl+C" in output
        return "回复内容"

    cli.request_reply(settings, respond)
    assert "伴侣：回复内容" in capsys.readouterr().out


def test_connection_check_does_not_create_database(tmp_path, monkeypatch, capsys):
    settings = Settings(
        model="test-model", api_key="never-print-me", database_path=tmp_path / "not-created.db"
    )
    captured = []

    def create_model(probe_settings):
        captured.append(probe_settings)
        return RecordingModel()

    monkeypatch.setattr(cli, "create_model", create_model)
    assert cli.check_connection(settings) == 0
    assert captured[0].timeout_seconds == 10
    assert captured[0].max_retries == 0
    assert not settings.database_path.exists()
    output = capsys.readouterr().out
    assert "连接成功" in output
    assert "never-print-me" not in output


def test_connection_timeout_has_proxy_advice_and_no_session_retry_hint(monkeypatch, capsys):
    def fail(settings):
        raise APITimeoutError(request=None)

    monkeypatch.setattr(cli, "create_model", fail)
    settings = Settings(model="test-model", api_key="never-print-me")
    assert cli.check_connection(settings) == 1
    output = capsys.readouterr().err
    assert "LLM_PROXY_URL" in output
    assert "/retry" not in output
    assert "never-print-me" not in output
