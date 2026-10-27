"""Sandbox diagnostics and local smoke tests, independent of model credentials."""

import json
from dataclasses import replace
from pathlib import Path

from .config import Settings
from .sandbox import MAX_CODE_CHARS, DockerSandbox, SandboxError


def run_file(settings: Settings, path: Path, stdin: str) -> int:
    try:
        with path.open(encoding="utf-8-sig") as handle:
            code = handle.read(MAX_CODE_CHARS + 1)
    except (OSError, UnicodeDecodeError):
        raise SandboxError("无法读取 Python 文件，请检查路径、权限和 UTF-8 编码。") from None
    result = DockerSandbox(settings).run_python(code, stdin)
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    return 0 if result.status == "success" else 1


def self_test(settings: Settings) -> int:
    provider = DockerSandbox(settings)
    cases = [
        ("basic", "print(sum([1, 2, 3]))", "", "success", "6"),
        ("stdin", "print(input().upper())", "hello\n", "success", "HELLO"),
        ("python_error", "1 / 0", "", "error", "ZeroDivisionError"),
        ("syntax_error", "def broken(:\n pass", "", "error", "SyntaxError"),
        ("timeout", "while True: pass", "", "timeout", ""),
        (
            "output_limit",
            f"print('x' * {settings.sandbox_output_bytes + 1000})",
            "",
            "output_limit",
            "",
        ),
        (
            "network",
            "import socket\ntry:\n"
            " socket.create_connection(('1.1.1.1',80),timeout=0.5)\n"
            "except OSError:\n print('network_blocked')\n"
            "else:\n raise RuntimeError('network was accessible')",
            "",
            "success",
            "network_blocked",
        ),
        (
            "isolation",
            "import os,pathlib\nassert os.getuid() == 10001\n"
            "assert 'LLM_API_KEY' not in os.environ\n"
            "assert 'OPENAI_API_KEY' not in os.environ\n"
            "assert not os.environ.get('HTTP_PROXY')\n"
            "pathlib.Path('/work/probe.txt').write_text('ephemeral')\n"
            "try:\n pathlib.Path('/host-write-probe').write_text('forbidden')\n"
            "except OSError:\n print('isolation_ok')\n"
            "else:\n raise RuntimeError('root filesystem was writable')",
            "",
            "success",
            "isolation_ok",
        ),
        (
            "fresh_workspace",
            "from pathlib import Path\nassert not Path('/work/probe.txt').exists()\nprint('fresh')",
            "",
            "success",
            "fresh",
        ),
    ]
    passed = True
    for name, code, stdin, expected_status, text in cases:
        result = provider.run_python(code, stdin)
        success = (
            result.status == expected_status
            and result.cleanup_confirmed
            and text in result.stdout + result.stderr
        )
        passed = passed and success
        print(
            f"{'PASS' if success else 'FAIL'} {name}: {result.status}, "
            f"cleanup={result.cleanup_confirmed}",
            flush=True,
        )
        if not success and result.message:
            print(result.message)
        if result.status == "unavailable":
            break
    # Check the actual cgroup limits inside a disposable container as well.
    if passed:
        code = (
            "from pathlib import Path\n"
            "limit=int(Path('/sys/fs/cgroup/memory.max').read_text())\n"
            f"assert limit == {settings.sandbox_memory_mb * 1024 * 1024}\n"
            "assert int(Path('/sys/fs/cgroup/memory.swap.max').read_text()) == 0\n"
            "assert int(Path('/sys/fs/cgroup/pids.max').read_text()) == 64\n"
            "quota,period=map(int,Path('/sys/fs/cgroup/cpu.max').read_text().split())\n"
            "assert quota == period\nprint('limits_ok')"
        )
        result = DockerSandbox(replace(settings, sandbox_timeout_seconds=10)).run_python(code)
        success = (
            result.status == "success" and result.cleanup_confirmed and "limits_ok" in result.stdout
        )
        passed = passed and success
        print(f"{'PASS' if success else 'FAIL'} cgroup_limits: {result.status}", flush=True)
    return 0 if passed else 1
