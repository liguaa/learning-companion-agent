"""Local-only HTTP interface for the learning companion."""

import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from langgraph.checkpoint.sqlite import SqliteSaver

from .agent import ModelCallError, open_companion
from .config import ConfigurationError, load_settings
from .knowledge import KnowledgeBase, KnowledgeBaseError
from .markdown import render_markdown
from .sandbox import DockerSandbox, SandboxError
from .sandbox_history import sandbox_runs
from .storage import SQLiteStore
from .web_settings import public_settings, save_model_settings

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"
settings = load_settings()
settings_lock = threading.Lock()
locks: dict[str, threading.Lock] = {}
locks_guard = threading.Lock()


def session_lock(session_id: str) -> threading.Lock:
    with locks_guard:
        return locks.setdefault(session_id, threading.Lock())


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def respond(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def read_json(self):
        if self.headers.get("Origin") != f"http://127.0.0.1:{self.server.server_port}":
            raise ValueError("请求来源无效，请通过本地地址打开页面。")
        if self.headers.get_content_type() != "application/json":
            raise ValueError("请求格式无效。")
        size = int(self.headers.get("Content-Length", "0"))
        if not 0 < size <= 3 * 1024 * 1024:
            raise ValueError("请求大小无效。")
        data = json.loads(self.rfile.read(size))
        if not isinstance(data, dict):
            raise ValueError("请求格式无效。")
        return data

    def do_GET(self):
        if self.headers.get("Host") != f"127.0.0.1:{self.server.server_port}":
            self.respond({"error": "请求地址无效。"}, 403)
            return
        path = urlsplit(self.path).path
        try:
            if path in {"/", "/app.js", "/style.css"}:
                name = "index.html" if path == "/" else path[1:]
                body = (WEB / name).read_bytes()
                self.send_response(200)
                self.send_header(
                    "Content-Type",
                    {
                        "index.html": "text/html",
                        "app.js": "text/javascript",
                        "style.css": "text/css",
                    }[name]
                    + "; charset=utf-8",
                )
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.end_headers()
                self.wfile.write(body)
            elif path == "/api/sessions":
                store = SQLiteStore(settings.database_path)
                try:
                    self.respond([dict(row) for row in store.sessions()])
                finally:
                    store.close()
            elif path.startswith("/api/sessions/"):
                sid = path.removeprefix("/api/sessions/")
                store = SQLiteStore(settings.database_path)
                try:
                    history = [dict(row) for row in store.history(sid)]
                    for message in history:
                        if message["role"] == "assistant":
                            message["html"] = render_markdown(message["content"])
                    runs = store.sandbox_history(sid)
                    # Older sessions predate transcript archival of tool results.
                    saver = SqliteSaver(store.conn)
                    saver.setup()
                    checkpoint = saver.get_tuple({"configurable": {"thread_id": sid}})
                    if checkpoint:
                        archived = {(run["reply_id"], run["call_id"]) for run in runs}
                        runs.extend(
                            run
                            for run in sandbox_runs(
                                checkpoint.checkpoint.get("channel_values", {}).get("messages", [])
                            )
                            if (run["reply_id"], run["call_id"]) not in archived
                        )
                    self.respond(
                        {
                            "session": dict(store.session(sid)),
                            "messages": history,
                            "sandbox_runs": runs,
                        }
                    )
                finally:
                    store.close()
            elif path == "/api/documents":
                self.respond(KnowledgeBase(settings).documents())
            elif path.startswith("/api/documents/"):
                self.respond(KnowledgeBase(settings).document(path.removeprefix("/api/documents/")))
            elif path == "/api/settings":
                self.respond(public_settings(settings))
            else:
                self.respond({"error": "页面不存在。"}, 404)
        except (ValueError, KnowledgeBaseError) as error:
            self.respond({"error": str(error)}, 400)
        except Exception:
            self.respond({"error": "读取失败，请检查本地服务。"}, 500)

    def do_POST(self):
        global settings
        if self.headers.get("Host") != f"127.0.0.1:{self.server.server_port}":
            self.respond({"error": "请求地址无效。"}, 403)
            return
        path = urlsplit(self.path).path
        try:
            data = self.read_json()
            if path == "/api/settings":
                with settings_lock:
                    settings = save_model_settings(data, settings, ROOT / ".env")
                self.respond(public_settings(settings))
            elif path == "/api/sandbox/check":
                if settings.sandbox_backend != "docker":
                    self.respond({"status": "unavailable", "message": "当前未启用 Docker 沙箱。"})
                else:
                    try:
                        DockerSandbox(settings).check()
                        self.respond(
                            {"status": "ready", "message": "Docker 引擎和 Python 镜像已就绪。"}
                        )
                    except SandboxError as error:
                        self.respond({"status": "unavailable", "message": str(error)})
            elif path == "/api/sessions":
                store = SQLiteStore(settings.database_path)
                try:
                    self.respond({"id": store.create_session()})
                finally:
                    store.close()
            elif path.startswith("/api/sessions/") and path.endswith(("/ask", "/retry")):
                parts = path.split("/")
                sid, action = parts[3], parts[4]
                lock = session_lock(sid)
                if not lock.acquire(blocking=False):
                    raise ValueError("此会话正在处理请求，请稍候。")
                try:
                    with open_companion(settings) as companion:
                        result = (
                            companion.ask(sid, data.get("content", ""))
                            if action == "ask"
                            else companion.retry(sid)
                        )
                    self.respond({"content": result})
                finally:
                    lock.release()
            elif path == "/api/documents/import":
                # A local path preserves the knowledge base's existing update semantics.
                raw = data.get("path", "")
                if not isinstance(raw, str) or not raw.strip():
                    raise ValueError("请输入资料文件的本地路径。")
                result = KnowledgeBase(settings).import_file(Path(raw.strip()))
                self.respond({"status": result.status, "filename": result.filename})
            elif path.startswith("/api/documents/") and path.endswith("/delete"):
                KnowledgeBase(settings).delete(path.split("/")[3])
                self.respond({"ok": True})
            else:
                self.respond({"error": "接口不存在。"}, 404)
        except (
            ValueError,
            KnowledgeBaseError,
            ModelCallError,
            ConfigurationError,
            json.JSONDecodeError,
        ) as error:
            self.respond(
                {
                    "error": str(error).replace(
                        "输入 /retry 可继续本轮，或 /new 新建会话。", "可在页面中重试，或新建会话。"
                    )
                },
                400,
            )
        except Exception:
            self.respond({"error": "操作失败，请检查本地服务和配置。"}, 500)


def main():
    parser = argparse.ArgumentParser(description="启动本地学习伴侣 Web 界面")
    parser.add_argument("--port", type=int, default=8765, help="本地监听端口（默认 8765）")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("端口必须在 1 到 65535 之间。")
    try:
        server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    except OSError:
        parser.exit(1, f"无法监听端口 {args.port}，请尝试 --port {args.port + 1}。\n")
    print(f"学习伴侣已启动：http://127.0.0.1:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
