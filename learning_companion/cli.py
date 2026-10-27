"""Command line interface. Read-only session commands work without model credentials."""

import argparse
import logging
import sqlite3
import sys
from collections.abc import Callable
from dataclasses import replace
from functools import partial
from pathlib import Path
from time import monotonic
from urllib.parse import urlsplit

from .agent import ModelCallError, create_model, model_error_message, open_companion
from .config import ConfigurationError, Settings, load_settings
from .knowledge import KnowledgeBase
from .knowledge_cli import (
    delete_document,
    handle_knowledge_command,
    import_path,
    list_documents,
    search_documents,
    show_document,
)
from .sandbox import DockerSandbox
from .sandbox_cli import run_file, self_test
from .storage import SQLiteStore

HELP = """命令：
  /help          查看帮助
  /new           新建学习会话
  /sessions      列出会话及完整 ID
  /use <ID>      切换到已有会话
  /history       查看当前会话完整历史
  /retry         重试本轮未完成的请求
  /paste         输入多行代码或问题，以单独一行 /send 结束，/cancel 取消
  /kb            查看知识库（/kb import、search、show、delete 管理资料）
  /sandbox       检查 Docker 沙箱是否就绪
  /exit          退出（会话自动保存）"""


def show_sessions(store: SQLiteStore) -> None:
    sessions = store.sessions()
    if not sessions:
        print("暂无学习会话。")
    for row in sessions:
        status = " [待完成]" if row["pending_message_id"] else ""
        print(f"{row['id']}  {row['title']}{status}")


def show_history(store: SQLiteStore, session_id: str) -> None:
    history = store.history(session_id)
    if not history:
        print("当前会话暂无消息。")
    for row in history:
        role = "你" if row["role"] == "user" else "伴侣"
        print(f"\n{role}：{row['content']}")


def read_multiline(max_chars: int) -> str:
    print("请输入多行内容，单独一行 /send 提交，/cancel 取消。")
    lines = []
    size = 0
    while True:
        line = input("... ")
        if line == "/cancel":
            return ""
        if line == "/send":
            return "\n".join(lines)
        size += len(line) + 1
        if size > max_chars + 1:
            raise ValueError(f"输入超过 {max_chars} 个字符，本次多行输入已取消。")
        lines.append(line)


def print_waiting(settings: Settings, *, save_turn: bool = True) -> None:
    cancel_hint = "Ctrl+C 可退出并保留本轮" if save_turn else "Ctrl+C 可取消测试"
    print(
        f"正在请求模型（单次网络超时 {settings.timeout_seconds:g} 秒，"
        f"自动重试 {settings.max_retries} 次；{cancel_hint}）…",
        flush=True,
    )


def request_reply(settings: Settings, operation: Callable[[], str]) -> None:
    print_waiting(settings)
    print("伴侣：" + operation(), flush=True)


def check_connection(settings: Settings) -> int:
    settings.validate_model()
    # Explicit diagnostic: no database writes or automatic retries.
    probe_settings = replace(settings, timeout_seconds=10, max_retries=0)
    print_waiting(probe_settings, save_turn=False)
    started = monotonic()
    try:
        model = create_model(probe_settings)
        response = model.invoke("请只回复：连接正常。")
    except KeyboardInterrupt:
        print("\n连接测试已取消。")
        return 130
    except Exception as error:
        print(
            f"连接测试失败（耗时 {monotonic() - started:.1f} 秒）："
            + model_error_message(error, retry_hint=False),
            file=sys.stderr,
        )
        return 1
    if not response.text.strip():
        print("服务已响应，但返回正文为空，请检查模型类型与输出长度设置。", file=sys.stderr)
        return 1
    print(f"连接成功（耗时 {monotonic() - started:.1f} 秒），未写入学习会话。")
    return 0


def chat(settings: Settings, session_id: str | None) -> int:
    with open_companion(settings) as companion:
        store = companion.store
        if session_id:
            store.session(session_id)
        else:
            session_id = store.create_session()
        print("个人学习伴侣 · 引导式学习")
        print(f"当前会话：{session_id}")
        print("输入 /help 查看命令；代码可通过 /paste 多行输入。")
        if store.session(session_id)["pending_message_id"]:
            print("该会话有未完成的请求，输入 /retry 继续。")
        while True:
            try:
                content = input("\n你> ").strip()
                if not content:
                    continue
                if content == "/exit":
                    break
                if content == "/help":
                    print(HELP)
                elif content == "/sessions":
                    show_sessions(store)
                elif content == "/history":
                    show_history(store, session_id)
                elif content == "/sandbox":
                    print(DockerSandbox(settings).check()["message"])
                elif content == "/kb" or content.startswith("/kb "):
                    handle_knowledge_command(companion.knowledge, content[3:])
                elif content == "/new":
                    session_id = store.create_session()
                    print(f"当前会话：{session_id}")
                elif content.startswith("/use "):
                    target = content.removeprefix("/use ").strip()
                    store.session(target)
                    session_id = target
                    print(f"当前会话：{session_id}")
                    if store.session(session_id)["pending_message_id"]:
                        print("该会话有未完成的请求，输入 /retry 继续。")
                elif content == "/retry":
                    request_reply(settings, partial(companion.retry, session_id))
                elif content == "/paste":
                    content = read_multiline(settings.max_input_chars)
                    if content.strip():
                        request_reply(settings, partial(companion.ask, session_id, content))
                elif content.startswith("/"):
                    print("未知命令，输入 /help 查看帮助。")
                else:
                    request_reply(settings, partial(companion.ask, session_id, content))
            except (EOFError, KeyboardInterrupt):
                break
            except (ValueError, ModelCallError) as error:
                print(f"提示：{error}")
        print(f"\n会话已保存。恢复命令：python main.py --session {session_id}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="个人学习伴侣：引导式学习与 SQLite 会话恢复")
    parser.add_argument("--env-file", type=Path, help="指定 .env 文件路径")
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--session", help="继续指定会话（使用完整 ID）")
    actions.add_argument("--list", action="store_true", help="列出本地会话，无需模型密钥")
    actions.add_argument("--history", metavar="ID", help="查看指定会话历史，无需模型密钥")
    actions.add_argument("--check", action="store_true", help="检查配置，不发起模型请求")
    actions.add_argument(
        "--check-connection", action="store_true", help="发送最小请求检查连接，不写入会话"
    )
    actions.add_argument("--kb-import", type=Path, metavar="PATH", help="导入资料文件或目录")
    actions.add_argument("--kb-list", action="store_true", help="列出资料，无需聊天模型密钥")
    actions.add_argument("--kb-search", metavar="QUERY", help="检索已导入的资料")
    actions.add_argument("--kb-show", metavar="ID", help="查看已导入的资料和行号")
    actions.add_argument("--kb-delete", metavar="ID", help="删除资料和索引，保留原文件")
    actions.add_argument("--sandbox-check", action="store_true", help="检查本机 Docker 与镜像")
    actions.add_argument(
        "--sandbox-prepare", action="store_true", help="通过 Docker 引擎下载沙箱镜像"
    )
    actions.add_argument(
        "--sandbox-run", type=Path, metavar="PYTHON_FILE", help="在容器中运行 Python 文件"
    )
    actions.add_argument(
        "--sandbox-self-test", action="store_true", help="测试真实 Docker 隔离和资源限制"
    )
    parser.add_argument("--sandbox-stdin", default="", help="--sandbox-run 的标准输入文本")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    # Provider HTTP logs may contain endpoint details; keep the default CLI output minimal.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    try:
        settings = load_settings(args.env_file)
        if args.sandbox_check:
            result = DockerSandbox(settings).check()
            print(result["message"])
            return 0 if result["ready"] else 1
        if args.sandbox_prepare:
            print(
                "正在通过 Docker 引擎准备镜像；镜像下载使用 Docker Desktop 的代理设置。", flush=True
            )
            print(DockerSandbox(settings).prepare()["message"])
            return 0
        if args.sandbox_run is not None:
            return run_file(settings, args.sandbox_run, args.sandbox_stdin)
        if args.sandbox_self_test:
            return self_test(settings)
        if args.kb_list or any(
            value is not None
            for value in (args.kb_import, args.kb_search, args.kb_show, args.kb_delete)
        ):
            knowledge = KnowledgeBase(settings)
            if args.kb_import:
                import_path(knowledge, args.kb_import)
            elif args.kb_list:
                list_documents(knowledge)
            elif args.kb_search is not None:
                search_documents(knowledge, args.kb_search)
            elif args.kb_show is not None:
                show_document(knowledge, args.kb_show)
            else:
                delete_document(knowledge, args.kb_delete)
            return 0
        if args.check_connection:
            return check_connection(settings)
        if args.check:
            settings.validate_model()
            print(f"配置有效。模型：{settings.model}")
            host = urlsplit(settings.resolved_base_url or "https://api.openai.com").hostname
            print(f"服务：{settings.resolved_provider}（{host}）")
            print(f"SQLite：{settings.database_path}")
            print("未发起模型请求；连接情况需要在实际对话时验证。")
            return 0
        if args.list or args.history:
            store = SQLiteStore(settings.database_path)
            try:
                if args.list:
                    show_sessions(store)
                else:
                    show_history(store, args.history)
            finally:
                store.close()
            return 0
        return chat(settings, args.session)
    except (ConfigurationError, ValueError) as error:
        print(f"配置或参数错误：{error}", file=sys.stderr)
        return 2
    except (OSError, sqlite3.Error):
        print("无法访问本地数据，请检查 DATABASE_PATH、目录权限或数据库占用情况。", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
