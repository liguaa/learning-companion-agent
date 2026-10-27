"""Extract actual sandbox calls from graph messages for transcript display."""

import json

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage


def sandbox_runs(messages, pending_id: str | None = None) -> list[dict]:
    runs = []
    human_id = None
    calls = {}
    for message in messages:
        if isinstance(message, HumanMessage):
            human_id = message.id
            calls = {}
        elif isinstance(message, AIMessage):
            for call in message.tool_calls:
                if call.get("name") == "run_python":
                    calls[call["id"]] = call.get("args", {})
        elif (
            isinstance(message, ToolMessage)
            and message.name == "run_python"
            and human_id
            and (pending_id is None or human_id == pending_id)
        ):
            try:
                result = json.loads(message.text)
                if not isinstance(result, dict) or "status" not in result:
                    continue
            except (ValueError, TypeError):
                continue
            args = calls.get(message.tool_call_id, {})
            runs.append(
                {
                    "reply_id": f"reply-{human_id}",
                    "call_id": message.tool_call_id,
                    "code": args.get("code", ""),
                    "stdin": args.get("stdin", ""),
                    "result": result,
                }
            )
    return runs
