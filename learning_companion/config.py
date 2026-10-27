"""Load configuration without exposing credentials or changing process environment."""

import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

from dotenv import dotenv_values

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class ConfigurationError(ValueError):
    pass


@dataclass(frozen=True)
class Settings:
    provider: str = "auto"
    model: str = ""
    api_key: str = field(default="", repr=False)
    base_url: str | None = None
    proxy_url: str | None = None
    deepseek_thinking: str = "disabled"
    database_path: Path = PROJECT_ROOT / "data" / "learning.db"
    timeout_seconds: float = 30
    max_retries: int = 0
    max_output_tokens: int = 1500
    summary_tokens: int = 6000
    keep_messages: int = 6
    max_input_chars: int = 6000
    knowledge_search_mode: str = "keyword"
    knowledge_chunk_size: int = 900
    knowledge_chunk_overlap: int = 120
    embedding_model: str = ""
    embedding_api_key: str = field(default="", repr=False)
    embedding_base_url: str | None = None
    sandbox_backend: str = "docker"
    sandbox_image: str = "python:3.11-slim"
    sandbox_timeout_seconds: float = 5
    sandbox_memory_mb: int = 256
    sandbox_output_bytes: int = 8000

    @property
    def resolved_provider(self) -> str:
        if self.provider != "auto":
            return self.provider
        return "deepseek" if self.model.lower().startswith("deepseek-") else "openai"

    @property
    def resolved_base_url(self) -> str | None:
        if self.base_url:
            return self.base_url
        return "https://api.deepseek.com" if self.resolved_provider == "deepseek" else None

    def validate_provider(self) -> None:
        if self.provider not in {"auto", "openai", "deepseek"}:
            raise ConfigurationError("LLM_PROVIDER 只能是 auto、openai 或 deepseek。")
        if self.deepseek_thinking not in {"enabled", "disabled"}:
            raise ConfigurationError("DEEPSEEK_THINKING 只能是 enabled 或 disabled。")
        host = urlsplit(self.base_url or "").hostname
        if self.resolved_provider == "deepseek" and host == "api.openai.com":
            raise ConfigurationError("DeepSeek 模型不能使用 OpenAI 官方地址，请检查 LLM_BASE_URL。")
        if (
            self.model.lower().startswith("deepseek-")
            and self.resolved_provider == "openai"
            and (not self.base_url or host == "api.openai.com")
        ):
            raise ConfigurationError(
                "DeepSeek 模型请使用 LLM_PROVIDER=deepseek 或正确的兼容服务地址。"
            )

    def validate_model(self) -> None:
        self.validate_provider()
        missing = []
        if not self.model:
            missing.append("LLM_MODEL")
        if not self.api_key:
            alias = "DEEPSEEK_API_KEY" if self.resolved_provider == "deepseek" else "OPENAI_API_KEY"
            missing.append(f"LLM_API_KEY（也可使用 {alias}）")
        if missing:
            raise ConfigurationError("请先复制 .env.example 为 .env 并配置：" + "、".join(missing))


def load_settings(env_file: Path | None = None) -> Settings:
    values = {**dotenv_values(env_file or PROJECT_ROOT / ".env"), **os.environ}

    def text(name: str, default: str = "") -> str:
        return str(values.get(name) or default).strip()

    def number(name: str, default: int | float, minimum: int, *, integer: bool = True):
        try:
            raw = text(name, str(default))
            value = int(raw) if integer else float(raw)
        except ValueError:
            raise ConfigurationError(f"{name} 必须为有效数字。") from None
        if not math.isfinite(value) or value < minimum:
            raise ConfigurationError(f"{name} 必须大于或等于 {minimum}。")
        return value

    def url(name: str) -> str | None:
        value = text(name) or None
        if value is None:
            return None
        try:
            parsed = urlsplit(value)
            valid = (
                parsed.scheme in {"http", "https"}
                and parsed.hostname
                and not parsed.username
                and not parsed.password
                and not parsed.query
                and not parsed.fragment
            )
            _ = parsed.port
        except ValueError:
            valid = False
        if not valid:
            raise ConfigurationError(f"{name} 必须是有效的 HTTP(S) 地址，且不含凭据或查询参数。")
        if name == "LLM_PROXY_URL" and parsed.path not in {"", "/"}:
            raise ConfigurationError("LLM_PROXY_URL 应为代理地址和端口，不能带 API 路径。")
        return value

    database_path = Path(text("DATABASE_PATH", "data/learning.db")).expanduser()
    if not database_path.is_absolute():
        database_path = PROJECT_ROOT / database_path

    provider = text("LLM_PROVIDER", "auto").lower()
    model = text("LLM_MODEL")
    is_deepseek = provider == "deepseek" or (
        provider == "auto" and model.lower().startswith("deepseek-")
    )
    settings = Settings(
        provider=provider,
        model=model,
        api_key=text("LLM_API_KEY")
        or text("DEEPSEEK_API_KEY" if is_deepseek else "OPENAI_API_KEY"),
        base_url=url("LLM_BASE_URL"),
        proxy_url=url("LLM_PROXY_URL"),
        deepseek_thinking=text("DEEPSEEK_THINKING", "disabled").lower(),
        database_path=database_path.resolve(),
        timeout_seconds=number("LLM_TIMEOUT_SECONDS", 30, 1, integer=False),
        max_retries=number("LLM_MAX_RETRIES", 0, 0),
        max_output_tokens=number("LLM_MAX_OUTPUT_TOKENS", 1500, 1),
        summary_tokens=number("CONTEXT_SUMMARY_TOKENS", 6000, 256),
        keep_messages=number("CONTEXT_KEEP_MESSAGES", 6, 1),
        max_input_chars=number("MAX_INPUT_CHARS", 6000, 1),
        knowledge_search_mode=text("KB_SEARCH_MODE", "keyword").lower(),
        knowledge_chunk_size=number("KB_CHUNK_SIZE", 900, 128),
        knowledge_chunk_overlap=number("KB_CHUNK_OVERLAP", 120, 0),
        embedding_model=text("EMBEDDING_MODEL"),
        embedding_api_key=text("EMBEDDING_API_KEY"),
        embedding_base_url=url("EMBEDDING_BASE_URL"),
        sandbox_backend=text("SANDBOX_BACKEND", "docker").lower(),
        sandbox_image=text("SANDBOX_IMAGE", "python:3.11-slim"),
        sandbox_timeout_seconds=number("SANDBOX_TIMEOUT_SECONDS", 5, 1, integer=False),
        sandbox_memory_mb=number("SANDBOX_MEMORY_MB", 256, 64),
        sandbox_output_bytes=number("SANDBOX_OUTPUT_BYTES", 8000, 1024),
    )
    settings.validate_provider()
    if settings.knowledge_search_mode not in {"keyword", "hybrid"}:
        raise ConfigurationError("KB_SEARCH_MODE 只能是 keyword 或 hybrid。")
    if not 0 <= settings.knowledge_chunk_overlap < settings.knowledge_chunk_size <= 2000:
        raise ConfigurationError("知识库切分需满足 0 <= KB_CHUNK_OVERLAP < KB_CHUNK_SIZE <= 2000。")
    if settings.sandbox_backend not in {"none", "docker"}:
        raise ConfigurationError("SANDBOX_BACKEND 只能是 none 或 docker。")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:/@-]{0,199}", settings.sandbox_image):
        raise ConfigurationError("SANDBOX_IMAGE 必须是有效的镜像引用，不能包含选项或空白。")
    if settings.sandbox_timeout_seconds > 30 or settings.sandbox_memory_mb > 1024:
        raise ConfigurationError("沙箱运行时间不能超过 30 秒，内存不能超过 1024 MiB。")
    if settings.sandbox_output_bytes > 32768:
        raise ConfigurationError("SANDBOX_OUTPUT_BYTES 不能超过 32768。")
    return settings
