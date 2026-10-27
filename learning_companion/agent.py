"""LangChain agent with SQLite checkpoints and recoverable turns."""

import logging
from contextlib import contextmanager
from importlib.resources import files

from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage
from langchain_deepseek import ChatDeepSeek
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
)

from .config import Settings
from .knowledge import KnowledgeBase
from .knowledge_tools import create_knowledge_tool, retrieved_sources
from .sandbox import DockerSandbox, SandboxProvider
from .sandbox_history import sandbox_runs
from .sandbox_tools import SANDBOX_GUIDANCE, create_sandbox_tool
from .skills import SKILL_GUIDANCE, SkillProvider, create_skill_tools
from .storage import SQLiteStore

logger = logging.getLogger(__name__)


class ModelCallError(RuntimeError):
    pass


def read_prompt(name: str) -> str:
    return files("learning_companion").joinpath("prompts", name).read_text(encoding="utf-8")


def create_model(settings: Settings) -> BaseChatModel:
    settings.validate_model()
    common = {
        "model": settings.model,
        "api_key": settings.api_key,
        "base_url": settings.resolved_base_url,
        "openai_proxy": settings.proxy_url,
        "timeout": settings.timeout_seconds,
        "max_retries": settings.max_retries,
        "use_responses_api": False,
    }
    if settings.resolved_provider == "deepseek":
        return ChatDeepSeek(
            **common,
            max_tokens=settings.max_output_tokens,
            extra_body={"thinking": {"type": settings.deepseek_thinking}},
        )
    return ChatOpenAI(
        **common,
        max_completion_tokens=settings.max_output_tokens,
    )


def model_error_message(error: Exception, *, retry_hint: bool = True) -> str:
    if isinstance(error, AuthenticationError):
        reason = "模型认证失败，请检查密钥和服务地址。"
    elif isinstance(error, RateLimitError):
        reason = "模型服务限流或额度不足，请稍后重试并检查账户。"
    elif isinstance(error, APITimeoutError):
        reason = (
            "模型请求超时。请确认终端能访问模型服务；需要代理时，"
            "可在 .env 配置 LLM_PROXY_URL。LLM_TIMEOUT_SECONDS 控制单次网络超时。"
        )
    elif isinstance(error, APIConnectionError):
        reason = "无法连接模型服务，请检查网络、LLM_BASE_URL 和需要时的 LLM_PROXY_URL。"
    elif isinstance(error, PermissionDeniedError):
        reason = "模型服务拒绝访问，请检查账户或模型访问权限。"
    elif isinstance(error, NotFoundError):
        reason = "找不到模型或接口，请检查 LLM_MODEL 和 LLM_BASE_URL。"
    elif isinstance(error, BadRequestError):
        reason = "模型服务不接受当前请求参数，请检查模型名称、接口协议和输出长度设置。"
    elif isinstance(error, APIStatusError):
        reason = f"模型服务返回 HTTP {error.status_code}，请检查服务状态。"
    else:
        reason = "本轮请求未完成，请检查模型配置或服务状态。"
    if retry_hint:
        return reason + " 输入 /retry 可继续本轮，或 /new 新建会话。"
    return reason


class LearningCompanion:
    def __init__(
        self,
        settings: Settings,
        store: SQLiteStore,
        model: BaseChatModel,
        *,
        skill_provider: SkillProvider | None = None,
        sandbox_provider: SandboxProvider | None = None,
    ):
        self.settings = settings
        self.store = store
        self.knowledge = KnowledgeBase(settings)
        checkpointer = SqliteSaver(store.conn)
        checkpointer.setup()
        tools = [create_knowledge_tool(self.knowledge)]
        system_prompt = read_prompt("system.md")
        active_sandbox = sandbox_provider
        if active_sandbox is None and settings.sandbox_backend == "docker":
            active_sandbox = DockerSandbox(settings)
        if active_sandbox is not None:
            tools.append(create_sandbox_tool(active_sandbox))
            system_prompt += "\n" + SANDBOX_GUIDANCE
        if skill_provider is not None:
            tools.extend(create_skill_tools(skill_provider))
            system_prompt += "\n" + SKILL_GUIDANCE
        self.graph = create_agent(
            model=model,
            tools=tools,
            system_prompt=system_prompt,
            checkpointer=checkpointer,
            middleware=[
                SummarizationMiddleware(
                    model=model,
                    trigger=("tokens", settings.summary_tokens),
                    keep=("messages", settings.keep_messages),
                    summary_prompt=read_prompt("summary.md"),
                    trim_tokens_to_summarize=None,
                )
            ],
        )

    def ask(self, session_id: str, content: str) -> str:
        content = content.strip()
        if not content:
            raise ValueError("请输入学习问题。")
        if len(content) > self.settings.max_input_chars:
            raise ValueError(
                f"单次输入不能超过 {self.settings.max_input_chars} 个字符，请分段发送。"
            )
        self.store.session(session_id)
        self.store.start_turn(session_id, content)
        return self.retry(session_id)

    def retry(self, session_id: str) -> str:
        session = self.store.session(session_id)
        pending_id = session["pending_message_id"]
        if pending_id is None:
            raise ValueError("当前没有待重试的请求。")
        pending = next(m for m in self.store.history(session_id) if m["id"] == pending_id)
        config = {"configurable": {"thread_id": session_id}}
        try:
            state = self.graph.get_state(config)
            messages = state.values.get("messages", [])
            if state.next:
                # Resume a failed graph node without adding the learner's message twice.
                result = self.graph.invoke(None, config)
            elif (
                any(m.id == pending_id for m in messages)
                and messages
                and isinstance(messages[-1], AIMessage)
                and messages[-1].text.strip()
            ):
                # The model finished, but the process stopped before transcript archival.
                result = state.values
            else:
                inputs = [HumanMessage(content=pending["content"], id=pending_id)]
                if (
                    messages
                    and isinstance(messages[-1], AIMessage)
                    and not messages[-1].text.strip()
                ):
                    inputs.append(RemoveMessage(id=messages[-1].id))
                result = self.graph.invoke(
                    {"messages": inputs},
                    config,
                )
            response = result["messages"][-1]
            if not isinstance(response, AIMessage) or not response.text.strip():
                raise RuntimeError("Empty model response")
            text = response.text + retrieved_sources(result["messages"], pending_id, self.knowledge)
            self.store.finish_turn(
                session_id,
                f"reply-{pending_id}",
                text,
                sandbox_runs(result["messages"], pending_id),
            )
            return text
        except Exception as error:
            # Do not log provider payloads, prompts, URLs or credentials.
            logger.warning("Learning turn failed (%s)", type(error).__name__)
            raise ModelCallError(model_error_message(error)) from None


@contextmanager
def open_companion(
    settings: Settings,
    model: BaseChatModel | None = None,
    *,
    skill_provider: SkillProvider | None = None,
    sandbox_provider: SandboxProvider | None = None,
):
    active_model = model if model is not None else create_model(settings)
    store = SQLiteStore(settings.database_path)
    try:
        yield LearningCompanion(
            settings,
            store,
            active_model,
            skill_provider=skill_provider,
            sandbox_provider=sandbox_provider,
        )
    finally:
        store.close()
