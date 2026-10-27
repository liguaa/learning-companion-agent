from dataclasses import replace

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_deepseek import ChatDeepSeek

from learning_companion import agent
from learning_companion.agent import ModelCallError, open_companion
from learning_companion.storage import SessionNotFound
from tests.conftest import RecordingModel, ToolCapableFakeModel


def test_context_is_restored_from_sqlite_after_reopening(settings):
    first_model = RecordingModel()
    with open_companion(settings, first_model) as companion:
        session = companion.store.create_session()
        companion.ask(session, "我正在学习循环，题目是计算列表总和。")

    second_model = RecordingModel()
    with open_companion(settings, second_model) as companion:
        companion.ask(session, "继续刚才的题目，我想先定义一个累加变量。")
        sent = second_model.calls[-1]
        assert isinstance(sent[0], SystemMessage)
        assert "计算列表总和" in sent[1].content
        assert [m.type for m in sent] == ["system", "human", "ai", "human"]
        assert len(companion.store.history(session)) == 4
        assert companion.store.session(session)["pending_message_id"] is None


def test_sessions_do_not_share_messages(settings):
    model = RecordingModel()
    with open_companion(settings, model) as companion:
        a = companion.store.create_session()
        b = companion.store.create_session()
        companion.ask(a, "A 会话的私有学习主题")
        companion.ask(b, "B 会话的问题")
        assert all("A 会话" not in str(m.content) for m in model.calls[-1])
        assert len(companion.store.history(a)) == len(companion.store.history(b)) == 2


def test_failed_turn_resumes_after_restart_without_duplicate_input(settings, caplog):
    with open_companion(settings, RecordingModel(failures_remaining=1)) as companion:
        session = companion.store.create_session()
        with pytest.raises(ModelCallError, match="/retry") as failure:
            companion.ask(session, "需要重试的问题")
        assert "test-provider-payload-with-secret" not in str(failure.value)
        assert "test-provider-payload-with-secret" not in caplog.text
        assert len(companion.store.history(session)) == 1
        with pytest.raises(ValueError, match="/retry"):
            companion.ask(session, "不应追加的新问题")

    model = RecordingModel()
    with open_companion(settings, model) as companion:
        companion.retry(session)
        assert len(companion.store.history(session)) == 2
        assert len([m for m in model.calls[-1] if isinstance(m, HumanMessage)]) == 1
        with pytest.raises(ValueError, match="没有待重试"):
            companion.retry(session)


def test_completed_checkpoint_can_finish_archival_after_restart(settings, monkeypatch):
    with open_companion(settings, RecordingModel()) as companion:
        session = companion.store.create_session()

        def fail_archival(*args):
            raise OSError("simulated disk failure")

        monkeypatch.setattr(companion.store, "finish_turn", fail_archival)
        with pytest.raises(ModelCallError):
            companion.ask(session, "已经得到模型回复，但未归档")

    model = RecordingModel()
    with open_companion(settings, model) as companion:
        companion.retry(session)
        assert model.calls == []
        assert len(companion.store.history(session)) == 2


def test_summarization_keeps_full_transcript_and_restores_summary(settings):
    settings = replace(settings, summary_tokens=256, keep_messages=2)
    model = RecordingModel()
    with open_companion(settings, model) as companion:
        session = companion.store.create_session()
        for turn in range(5):
            companion.ask(session, f"第 {turn} 轮，学习循环和列表求和。" + "练习描述。" * 300)
        history = companion.store.history(session)
        checkpoint = companion.graph.get_state({"configurable": {"thread_id": session}})
        assert len(history) == 10
        assert len(checkpoint.values["messages"]) < len(history)
        assert any("当前学习循环" in m.text for m in checkpoint.values["messages"])
        assert "第 0 轮" in history[0]["content"]

    resumed_model = RecordingModel()
    with open_companion(settings, resumed_model) as companion:
        companion.ask(session, "继续")
        assert any("当前学习循环" in m.text for m in resumed_model.calls[-1])


def test_invalid_input_does_not_write_messages(settings):
    settings = replace(settings, max_input_chars=10)
    with open_companion(settings, RecordingModel()) as companion:
        session = companion.store.create_session()
        for content in ("   ", "x" * 11):
            with pytest.raises(ValueError):
                companion.ask(session, content)
        assert companion.store.history(session) == []
        with pytest.raises(SessionNotFound):
            companion.ask("unknown", "问题")


def test_empty_response_can_be_retried(settings):
    model = ToolCapableFakeModel(
        responses=[AIMessage(content=""), AIMessage(content="先从一个小例子开始。")]
    )
    with open_companion(settings, model) as companion:
        session = companion.store.create_session()
        with pytest.raises(ModelCallError):
            companion.ask(session, "如何理解循环？")
        assert companion.retry(session) == "先从一个小例子开始。"
        assert len(companion.store.history(session)) == 2


def test_model_factory_passes_proxy_without_changing_environment(settings, monkeypatch):
    import os

    captured = {}

    def model(**kwargs):
        captured.update(kwargs)
        return RecordingModel()

    before = dict(os.environ)
    monkeypatch.setattr(agent, "ChatOpenAI", model)
    agent.create_model(
        replace(
            settings, model="test-model", api_key="local-test", proxy_url="http://127.0.0.1:12345"
        )
    )
    assert captured["openai_proxy"] == "http://127.0.0.1:12345"
    assert dict(os.environ) == before


def test_deepseek_factory_uses_documented_parameters_and_defaults(settings):
    model = agent.create_model(replace(settings, model="deepseek-flash", api_key="local-test"))
    assert isinstance(model, ChatDeepSeek)
    assert str(model.api_base) == "https://api.deepseek.com"
    payload = model._get_request_payload("hello")
    assert payload["max_tokens"] == settings.max_output_tokens
    assert "max_completion_tokens" not in payload
    assert payload["extra_body"] == {"thinking": {"type": "disabled"}}
