import json

from langchain_core.messages import AIMessage, ToolMessage

from learning_companion.agent import open_companion
from learning_companion.knowledge import KnowledgeBase
from learning_companion.knowledge_tools import retrieved_sources
from tests.conftest import ToolCapableFakeModel


def test_agent_calls_real_search_tool_and_archives_sources(settings, tmp_path):
    path = tmp_path / "notes.md"
    path.write_text("# 浅拷贝\n内部对象被共享。", encoding="utf-8")
    imported = KnowledgeBase(settings).import_file(path)
    citation = f"[KB:{imported.document_id}:r1:c1]"
    model = ToolCapableFakeModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "search-call",
                        "name": "search_knowledge",
                        "args": {"query": "浅拷贝"},
                    }
                ],
            ),
            AIMessage(content=f"笔记指出内部对象共享 {citation}。你会如何验证？"),
            AIMessage(content="继续观察外层容器与内部对象的区别。"),
        ]
    )
    with open_companion(settings, model) as companion:
        session = companion.store.create_session()
        answer = companion.ask(session, "根据我的笔记解释浅拷贝。")
        assert citation in answer and "本轮检索资料" in answer
        assert "notes.md" in answer and "第 1～2 行" in answer
        assert str(tmp_path) not in answer
        assert companion.store.history(session)[-1]["content"] == answer
        snapshot = companion.graph.get_state({"configurable": {"thread_id": session}})
        messages = snapshot.values["messages"]
        tool_result = next(m for m in messages if isinstance(m, ToolMessage))
        assert json.loads(tool_result.text)["results"][0]["document_id"] == imported.document_id
        next_answer = companion.ask(session, "继续")
        assert "本轮检索资料" not in next_answer
        companion.knowledge.delete(imported.document_id)
        original_user_id = next(m.id for m in messages if m.type == "human")
        assert retrieved_sources(messages, original_user_id, companion.knowledge) == ""


def test_empty_knowledge_tool_results_do_not_invent_sources(settings):
    model = ToolCapableFakeModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "empty-search",
                        "name": "search_knowledge",
                        "args": {"query": "不存在的主题"},
                    }
                ],
            ),
            AIMessage(content="还没有找到相关资料，可以先导入笔记。"),
        ]
    )
    with open_companion(settings, model) as companion:
        session = companion.store.create_session()
        answer = companion.ask(session, "查一下我的资料")
        assert "[KB:" not in answer
        assert "本轮检索资料" not in answer
