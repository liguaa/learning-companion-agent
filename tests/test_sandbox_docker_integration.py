import json
import os

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from learning_companion.agent import open_companion
from tests.conftest import ToolCapableFakeModel


@pytest.mark.skipif(os.environ.get("RUN_DOCKER_TESTS") != "1", reason="Set RUN_DOCKER_TESTS=1")
def test_agent_tool_executes_in_real_local_docker(settings):
    model = ToolCapableFakeModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "real-docker-run",
                        "name": "run_python",
                        "args": {"code": "print('学习', sum([1,2,3]))"},
                    }
                ],
            ),
            AIMessage(content="运行输出为学习 6。"),
        ]
    )
    with open_companion(settings, model) as companion:
        session = companion.store.create_session()
        companion.ask(session, "请验证这段示例代码")
        state = companion.graph.get_state({"configurable": {"thread_id": session}})
        result = next(m for m in state.values["messages"] if isinstance(m, ToolMessage))
        payload = json.loads(result.text)
        assert payload["status"] == "success"
        assert payload["stdout"].strip() == "学习 6"
        assert payload["cleanup_confirmed"] and payload["exit_code"] == 0
