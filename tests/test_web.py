"""Exercise the Web endpoints without a model or network access."""

import io
import json
from dataclasses import replace
from email.message import Message
from types import SimpleNamespace

from learning_companion import web
from learning_companion.config import Settings


def request(method, path, payload=None, *, origin="http://127.0.0.1:8765"):
    handler = object.__new__(web.Handler)
    handler.server = SimpleNamespace(server_port=8765)
    handler.path = path
    handler.headers = Message()
    handler.headers["Host"] = "127.0.0.1:8765"
    handler.headers["Origin"] = origin
    handler.headers["Content-Type"] = "application/json"
    body = json.dumps(payload or {}).encode()
    handler.headers["Content-Length"] = str(len(body))
    handler.rfile = io.BytesIO(body)
    handler.wfile = io.BytesIO()
    handler.send_response = lambda status: setattr(handler, "status", status)
    handler.send_header = lambda *args: None
    handler.end_headers = lambda: None
    getattr(handler, "do_" + method)()
    return handler.status, json.loads(handler.wfile.getvalue())


def test_session_and_knowledge_endpoints(tmp_path, monkeypatch):
    monkeypatch.setattr(web, "settings", replace(Settings(), database_path=tmp_path / "db.sqlite"))
    status, session = request("POST", "/api/sessions")
    assert status == 200
    status, history = request("GET", f"/api/sessions/{session['id']}")
    assert status == 200 and history["messages"] == []
    file = tmp_path / "notes.md"
    file.write_text("# 学习笔记\nPython 列表", encoding="utf-8")
    status, result = request("POST", "/api/documents/import", {"path": str(file)})
    assert status == 200 and result["status"] == "imported"
    _, documents = request("GET", "/api/documents")
    assert documents[0]["filename"] == "notes.md"
    _, document = request("GET", f"/api/documents/{documents[0]['id']}")
    assert "Python 列表" in document["content"]
    status, _ = request("POST", f"/api/documents/{documents[0]['id']}/delete")
    assert status == 200 and file.exists()
    assert request("GET", "/api/documents")[1] == []


def test_rejects_cross_origin_write(tmp_path, monkeypatch):
    monkeypatch.setattr(web, "settings", replace(Settings(), database_path=tmp_path / "db.sqlite"))
    status, _ = request("POST", "/api/sessions", origin="https://example.com")
    assert status == 400
    assert not (tmp_path / "db.sqlite").exists()


def test_model_settings_save_preserve_and_redact(tmp_path, monkeypatch):
    for name in ("LLM_PROVIDER", "LLM_MODEL", "LLM_BASE_URL", "LLM_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    env = tmp_path / ".env"
    env.write_text("# Keep local options\nMAX_INPUT_CHARS=7000\n", encoding="utf-8")
    monkeypatch.setattr(web, "ROOT", tmp_path)
    monkeypatch.setattr(web, "settings", Settings())
    payload = {
        "provider": "deepseek",
        "model": "deepseek-test",
        "base_url": "",
        "api_key": "test-secret",
    }
    status, result = request("POST", "/api/settings", payload)
    assert status == 200 and result["has_api_key"]
    assert "test-secret" not in json.dumps(result)
    assert "api_key" not in result
    assert web.settings.api_key == "test-secret"
    assert web.settings.max_input_chars == 7000
    assert "# Keep local options" in env.read_text(encoding="utf-8")
    payload.update(model="deepseek-new", api_key="")
    assert request("POST", "/api/settings", payload)[0] == 200
    assert web.settings.api_key == "test-secret"
    assert web.settings.model == "deepseek-new"
    assert "test-secret" not in json.dumps(request("GET", "/api/settings")[1])
    saved = env.read_bytes()
    payload["base_url"] = "not-a-url"
    assert request("POST", "/api/settings", payload)[0] == 400
    assert env.read_bytes() == saved
    assert not list(tmp_path.glob(".env-web-*"))


def test_settings_respects_process_environment(tmp_path, monkeypatch):
    monkeypatch.setattr(web, "ROOT", tmp_path)
    monkeypatch.setattr(web, "settings", Settings(model="managed", api_key="existing"))
    monkeypatch.setenv("LLM_MODEL", "managed")
    payload = {"provider": "auto", "model": "changed", "base_url": "", "api_key": ""}
    assert request("POST", "/api/settings", payload)[0] == 400
    assert not (tmp_path / ".env").exists()
