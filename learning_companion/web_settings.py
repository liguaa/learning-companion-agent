"""Validate and atomically persist model settings without exposing secrets."""

import os
import tempfile
from pathlib import Path

from dotenv import set_key

from .config import ConfigurationError, Settings, load_settings

FIELDS = {
    "provider": "LLM_PROVIDER",
    "model": "LLM_MODEL",
    "base_url": "LLM_BASE_URL",
    "api_key": "LLM_API_KEY",
}


def public_settings(settings: Settings) -> dict:
    return {
        "provider": settings.provider,
        "model": settings.model,
        "base_url": settings.base_url or "",
        "has_api_key": bool(settings.api_key),
        "database": str(settings.database_path),
        "sandbox": settings.sandbox_backend,
    }


def save_model_settings(data: dict, current: Settings, env_file: Path) -> Settings:
    changes = {}
    for field, name in FIELDS.items():
        value = data.get(field, "")
        if not isinstance(value, str) or len(value) > 4096:
            raise ConfigurationError("模型设置必须是有效文本。")
        value = value.strip()
        if any(char in value for char in ("\n", "\r", "\x00", "${")):
            raise ConfigurationError("模型设置不能包含换行、空字符或环境变量表达式。")
        if field == "api_key" and not value:
            continue  # Blank means preserve the existing key.
        existing = getattr(current, field) or ""
        if value != existing:
            if name in os.environ:
                raise ConfigurationError(f"{name} 由进程环境变量管理，请先在启动环境中修改。")
            changes[name] = value
    if not changes:
        current.validate_model()
        return current

    env_file.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".env-web-", dir=env_file.parent)
    candidate = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            if env_file.exists():
                handle.write(env_file.read_text(encoding="utf-8"))
        for name, value in changes.items():
            set_key(str(candidate), name, value, quote_mode="always")
        updated = load_settings(candidate)
        updated.validate_model()
        if (
            updated.resolved_provider != current.resolved_provider
            and not data.get("api_key", "").strip()
        ):
            raise ConfigurationError("切换模型提供方时，请同时填写对应的 API Key。")
        os.replace(candidate, env_file)
        return updated
    finally:
        candidate.unlink(missing_ok=True)
