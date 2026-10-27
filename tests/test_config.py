import pytest

from learning_companion.config import PROJECT_ROOT, ConfigurationError, load_settings


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    for name in (
        "LLM_MODEL",
        "LLM_PROVIDER",
        "LLM_API_KEY",
        "DEEPSEEK_API_KEY",
        "DEEPSEEK_THINKING",
        "OPENAI_API_KEY",
        "LLM_BASE_URL",
        "LLM_PROXY_URL",
        "DATABASE_PATH",
        "LLM_TIMEOUT_SECONDS",
        "LLM_MAX_RETRIES",
        "LLM_MAX_OUTPUT_TOKENS",
        "CONTEXT_SUMMARY_TOKENS",
        "CONTEXT_KEEP_MESSAGES",
        "MAX_INPUT_CHARS",
        "KB_SEARCH_MODE",
        "KB_CHUNK_SIZE",
        "KB_CHUNK_OVERLAP",
        "EMBEDDING_MODEL",
        "EMBEDDING_API_KEY",
        "EMBEDDING_BASE_URL",
        "SANDBOX_BACKEND",
        "SANDBOX_IMAGE",
        "SANDBOX_TIMEOUT_SECONDS",
        "SANDBOX_MEMORY_MB",
        "SANDBOX_OUTPUT_BYTES",
    ):
        monkeypatch.delenv(name, raising=False)


def test_dotenv_environment_priority_and_secret_redaction(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "LLM_MODEL=file-model\nLLM_API_KEY=do-not-print-me\nDATABASE_PATH=data/test.db\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("LLM_MODEL", "process-model")
    settings = load_settings(env_file)
    settings.validate_model()
    assert settings.model == "process-model"
    assert settings.api_key == "do-not-print-me"
    assert "do-not-print-me" not in repr(settings)
    assert settings.database_path == PROJECT_ROOT / "data" / "test.db"


@pytest.mark.parametrize("value", ["oops", "-1", "nan", "inf", "0"])
def test_reject_invalid_timeout(tmp_path, monkeypatch, value):
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", value)
    with pytest.raises(ConfigurationError, match="LLM_TIMEOUT_SECONDS"):
        load_settings(tmp_path / "missing.env")


@pytest.mark.parametrize("value", ["not-a-url", "https://secret@host/v1", "https://h/v1?key=x"])
def test_reject_unsafe_or_invalid_url(tmp_path, monkeypatch, value):
    monkeypatch.setenv("LLM_BASE_URL", value)
    with pytest.raises(ConfigurationError, match="LLM_BASE_URL"):
        load_settings(tmp_path / "missing.env")


def test_missing_model_configuration_has_actionable_message(tmp_path):
    settings = load_settings(tmp_path / "missing.env")
    with pytest.raises(ConfigurationError, match=".env.example"):
        settings.validate_model()


def test_proxy_can_be_persisted_in_dotenv(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("LLM_PROXY_URL=http://127.0.0.1:12345\n", encoding="utf-8")
    settings = load_settings(env_file)
    assert settings.proxy_url == "http://127.0.0.1:12345"


@pytest.mark.parametrize(
    "value", ["not-a-proxy", "http://127.0.0.1:12345/v1", "http://user:secret@host:12345"]
)
def test_proxy_rejects_invalid_address_without_exposing_contents(tmp_path, monkeypatch, value):
    monkeypatch.setenv("LLM_PROXY_URL", value)
    with pytest.raises(ConfigurationError, match="LLM_PROXY_URL") as error:
        load_settings(tmp_path / "missing.env")
    assert "secret" not in str(error.value)


def test_deepseek_model_selects_correct_provider_and_endpoint(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("LLM_MODEL=deepseek-flash\nLLM_API_KEY=local-test\n", encoding="utf-8")
    settings = load_settings(env_file)
    settings.validate_model()
    assert settings.resolved_provider == "deepseek"
    assert settings.resolved_base_url == "https://api.deepseek.com"
    assert settings.deepseek_thinking == "disabled"


def test_deepseek_uses_provider_specific_key_alias(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "deepseek-flash")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-local-test")
    monkeypatch.setenv("OPENAI_API_KEY", "unrelated-openai-key")
    settings = load_settings(tmp_path / "missing.env")
    assert settings.api_key == "deepseek-local-test"
    monkeypatch.delenv("DEEPSEEK_API_KEY")
    settings = load_settings(tmp_path / "missing.env")
    with pytest.raises(ConfigurationError, match="DEEPSEEK_API_KEY"):
        settings.validate_model()


@pytest.mark.parametrize(
    ("provider", "base_url"),
    [
        ("deepseek", "https://api.openai.com/v1"),
        ("openai", ""),
        ("openai", "https://api.openai.com/v1"),
    ],
)
def test_deepseek_cannot_accidentally_use_openai_endpoint(
    tmp_path, monkeypatch, provider, base_url
):
    monkeypatch.setenv("LLM_PROVIDER", provider)
    monkeypatch.setenv("LLM_MODEL", "deepseek-flash")
    monkeypatch.setenv("LLM_BASE_URL", base_url)
    with pytest.raises(ConfigurationError, match="DeepSeek"):
        load_settings(tmp_path / "missing.env")


@pytest.mark.parametrize(
    ("name", "value"), [("LLM_PROVIDER", "unknown"), ("DEEPSEEK_THINKING", "maybe")]
)
def test_provider_options_are_validated(tmp_path, monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ConfigurationError, match=name):
        load_settings(tmp_path / "missing.env")


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("KB_SEARCH_MODE", "unknown"),
        ("KB_CHUNK_SIZE", "100"),
        ("KB_CHUNK_SIZE", "2001"),
        ("KB_CHUNK_OVERLAP", "900"),
    ],
)
def test_knowledge_configuration_is_validated(tmp_path, monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ConfigurationError):
        load_settings(tmp_path / "missing.env")


def test_embedding_configuration_uses_independent_secret(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "chat-only-secret")
    monkeypatch.setenv("EMBEDDING_API_KEY", "embedding-only-secret")
    settings = load_settings(tmp_path / "missing.env")
    assert settings.embedding_api_key == "embedding-only-secret"
    assert "embedding-only-secret" not in repr(settings)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("SANDBOX_BACKEND", "shell"),
        ("SANDBOX_IMAGE", "--privileged"),
        ("SANDBOX_TIMEOUT_SECONDS", "31"),
        ("SANDBOX_MEMORY_MB", "2048"),
        ("SANDBOX_OUTPUT_BYTES", "1000000"),
    ],
)
def test_sandbox_configuration_is_bounded(tmp_path, monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ConfigurationError):
        load_settings(tmp_path / "missing.env")
