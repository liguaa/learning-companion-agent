from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from learning_companion.config import Settings


class RecordingModel(BaseChatModel):
    calls: list[list[BaseMessage]] = Field(default_factory=list)
    failures_remaining: int = 0

    def bind_tools(self, tools, **kwargs):
        return self

    @property
    def _llm_type(self) -> str:
        return "offline-test-model"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs: Any):
        self.calls.append(list(messages))
        if self.failures_remaining:
            self.failures_remaining -= 1
            raise RuntimeError("test-provider-payload-with-secret")
        if any(isinstance(m, SystemMessage) for m in messages):
            content = "先观察循环中累加值如何变化，你准备如何验证？"
        else:
            content = "当前学习循环和列表求和；已讨论累加变量，下一步验证循环中的变化。"
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])


class ToolCapableFakeModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


@pytest.fixture
def settings(tmp_path):
    return Settings(database_path=tmp_path / "learning.db")
