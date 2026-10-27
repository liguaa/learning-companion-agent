import json

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from learning_companion.sandbox_history import sandbox_runs


def test_results_stay_with_their_turn_and_preserve_failure():
    messages = [
        HumanMessage(content="第一题", id="first"),
        AIMessage(
            content="",
            tool_calls=[{"name": "run_python", "id": "one", "args": {"code": "print(1)"}}],
        ),
        ToolMessage(
            content=json.dumps({"status": "unavailable", "message": "Docker 未启动"}),
            tool_call_id="one",
            name="run_python",
        ),
        HumanMessage(content="第二题", id="second"),
        AIMessage(
            content="",
            tool_calls=[{"name": "run_python", "id": "two", "args": {"code": "print(2)"}}],
        ),
        ToolMessage(
            content=json.dumps({"status": "success", "stdout": "2\n"}),
            tool_call_id="two",
            name="run_python",
        ),
    ]
    first = sandbox_runs(messages, "first")
    assert len(first) == 1 and first[0]["result"]["status"] == "unavailable"
    assert first[0]["reply_id"] == "reply-first" and first[0]["code"] == "print(1)"
    assert len(sandbox_runs(messages)) == 2
