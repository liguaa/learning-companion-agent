import json

import pytest
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from pydantic import ValidationError

from learning_companion.agent import open_companion
from learning_companion.skills import SkillInfo, create_skill_tools
from tests.conftest import RecordingModel, ToolCapableFakeModel


class ExampleProvider:
    def __init__(self):
        self.loaded = []

    def list_skills(self):
        return [SkillInfo(name="external_example", description="外部提供方测试用指导")]

    def load_skill(self, name):
        self.loaded.append(name)
        return "先确认已有尝试，再给一个下一步提示。"


def test_unknown_skill_does_not_reach_provider_loader():
    provider = ExampleProvider()
    tools = {item.name: item for item in create_skill_tools(provider)}
    result = json.loads(tools["load_skill"].invoke({"name": "not_registered"}))
    assert "error" in result and provider.loaded == []
    with pytest.raises(ValidationError):
        tools["load_skill"].invoke({"name": "../private"})


def test_provider_errors_do_not_expose_details():
    class FailingProvider(ExampleProvider):
        def load_skill(self, name):
            raise RuntimeError("private-provider-key")

    result = create_skill_tools(FailingProvider())[1].invoke({"name": "external_example"})
    assert "private-provider-key" not in result
    assert "error" in json.loads(result)


def test_agent_can_load_injected_provider_without_changing_dialog_loop(settings):
    provider = ExampleProvider()
    model = ToolCapableFakeModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "list",
                        "name": "list_skills",
                        "args": {},
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "load",
                        "name": "load_skill",
                        "args": {"name": "external_example"},
                    }
                ],
            ),
            AIMessage(content="你目前尝试了哪一步？"),
        ]
    )
    with open_companion(settings, model, skill_provider=provider) as companion:
        session = companion.store.create_session()
        assert companion.ask(session, "请使用外部指导帮助我") == "你目前尝试了哪一步？"
        snapshot = companion.graph.get_state({"configurable": {"thread_id": session}})
        results = [m for m in snapshot.values["messages"] if isinstance(m, ToolMessage)]
        assert [m.name for m in results] == ["list_skills", "load_skill"]
        assert provider.loaded == ["external_example"]


def test_default_agent_does_not_advertise_skill_tools(settings):
    model = RecordingModel()
    with open_companion(settings, model) as companion:
        session = companion.store.create_session()
        companion.ask(session, "你好")
    prompt = next(m.text for m in model.calls[0] if isinstance(m, SystemMessage))
    assert "list_skills" not in prompt and "load_skill" not in prompt
