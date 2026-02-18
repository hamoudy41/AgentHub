"""ReAct agent with LangGraph - tools: calculator, search, document lookup."""

from __future__ import annotations

import re
from typing import Annotated, Any, AsyncIterator, Sequence, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from app.core.logging import get_logger
from app.security import sanitize_user_input

from .tools import BASE_TOOLS, calculator_tool, create_document_lookup_tool, search_tool

logger = get_logger(__name__)


def _translate_math_intent(message: str) -> tuple[str, str] | None:
    """Parse natural-language math (average, mean, sum, product) into (expression, intent)."""
    msg_lower = message.lower().strip()
    numbers = re.findall(r"-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?", message)
    nums = [float(n.replace(",", "")) for n in numbers]

    if len(nums) < 1:
        return None

    if "average" in msg_lower or "mean" in msg_lower:
        expr = (
            "(" + "+".join(str(int(n) if n == int(n) else n) for n in nums) + ")/" + str(len(nums))
        )
        return (expr, "average")
    if "sum of" in msg_lower or "add up" in msg_lower:
        expr = "+".join(str(int(n) if n == int(n) else n) for n in nums)
        return (expr, "sum")
    if "product" in msg_lower or "multiply" in msg_lower:
        expr = "*".join(str(int(n) if n == int(n) else n) for n in nums)
        return (expr, "product")

    return None


def _is_malformed(text: str) -> bool:
    """True if text looks like malformed tool-call JSON."""
    t = text.strip()
    return t.startswith("{") and '"parameters"' in t and '"name"' in t


_STOP_WORDS = frozenset(
    "the a an is are was were be been being of in to for on with at by from as "
    "what who how when where which why best good great top experience things thing "
    "i me my you your we our it its".split()
)


def _search_query_from_message(message: str) -> str:
    """Derive a focused search query from the user message."""
    q = message.strip().rstrip("?.").strip()
    if not q:
        return message
    lower = q.lower()
    for prefix in (
        "what is the ",
        "what is ",
        "what are the ",
        "what are ",
        "who is the ",
        "who are the ",
        "who is ",
        "who are ",
        "how is the ",
        "how are the ",
        "how does ",
        "how do ",
        "tell me about ",
        "explain ",
        "describe ",
    ):
        if lower.startswith(prefix):
            q = q[len(prefix) :].strip()
            lower = q.lower()
            break
    words = q.split()
    kept = [w for w in words if w.lower() not in _STOP_WORDS]
    q = " ".join(kept) if kept else q
    return q.strip() or message


class AgentState(TypedDict):
    """State for the ReAct agent."""

    messages: Annotated[Sequence[BaseMessage], add_messages]


def _get_chat_model(provider: str, base_url: str, model: str, api_key: str | None = None):
    """Create LangChain ChatModel from our config."""
    base = str(base_url).rstrip("/")
    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            base_url=base,
            model=model,
            temperature=0,
            num_ctx=2048,
            num_predict=512,
        )
    # openai_compatible
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        base_url=f"{base}/v1",
        api_key=api_key or "not-needed",
        model=model,
        temperature=0,
    )


def _create_agent_graph(tools: list) -> Any:
    """Build the ReAct graph with given tools."""
    from app.core.config import get_settings

    settings = get_settings()
    if not settings.llm_base_url or not settings.llm_provider:
        return None

    model = _get_chat_model(
        provider=settings.llm_provider,
        base_url=str(settings.llm_base_url),
        model=settings.llm_model,
        api_key=settings.llm_api_key,
    )
    model_with_tools = model.bind_tools(tools)

    def call_model(state: AgentState, config: RunnableConfig) -> dict:
        system = SystemMessage(
            content="You are a helpful, friendly AI assistant with access to tools. "
            "Always respond in English. "
            "Use the calculator for math, search for web info, document_lookup for document content. "
            "When using search_tool: use the most specific, distinctive terms from the question. "
            "Avoid single generic words that match unrelated topics. "
            "When no tool is needed, answer directly in plain text—never output JSON or tool schemas. "
            "Be conversational and helpful. Answer any question the user asks—do not refuse or deflect. "
            "Use your knowledge freely. If search returns no results, answer from what you know. "
            "Respond concisely but naturally. If you truly cannot help, say so."
        )
        response = model_with_tools.invoke(
            [system] + list(state["messages"]),
            config,
        )
        return {"messages": [response]}

    tool_node = ToolNode(tools)

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node)
    workflow.set_entry_point("agent")

    def should_continue(state: AgentState) -> str:
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tools"
        return "end"

    workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
    workflow.add_edge("tools", "agent")
    return workflow.compile()


def agent_graph(
    tenant_id: str,
    get_document_fn: Any,
) -> Any:
    """Create the agent graph with document lookup bound to tenant."""
    doc_tool = create_document_lookup_tool(tenant_id, get_document_fn)
    tools = BASE_TOOLS + [doc_tool]
    return _create_agent_graph(tools)


def _get_model_without_tools() -> Any:
    """Get chat model without tool binding, for summarization."""
    from app.core.config import get_settings

    settings = get_settings()
    if not settings.llm_base_url or not settings.llm_provider:
        return None
    return _get_chat_model(
        provider=settings.llm_provider,
        base_url=str(settings.llm_base_url),
        model=settings.llm_model,
        api_key=settings.llm_api_key,
    )


async def _search_and_summarize(message: str) -> str | None:
    """Search the web and use LLM to summarize. Returns None if search or summarization fails."""
    query = _search_query_from_message(message)
    search_content = search_tool.invoke({"query": query})
    if (
        not search_content
        or "No web results" in search_content
        or "Search failed" in search_content
    ):
        return None

    model = _get_model_without_tools()
    if not model:
        return None

    prompt = (
        f"Summarize the following web search results in 2-4 concise sentences for the user. "
        f"Answer the question directly. Always respond in English. Use plain text only.\n\n"
        f"Question: {message}\n\n"
        f"Search results:\n{search_content[:2000]}"
    )
    try:
        response = await model.ainvoke([HumanMessage(content=prompt)])
        if hasattr(response, "content") and response.content:
            text = str(response.content).strip()
            if text and not _is_malformed(text):
                return text
    except Exception:
        pass
    return None


async def run_agent(
    tenant_id: str,
    message: str,
    get_document_fn: Any,
) -> dict[str, Any]:
    """Run the agent and return the final response."""
    # Sanitize user input for security
    try:
        message = sanitize_user_input(
            message,
            max_length=4000,
            check_injection=True,
            tenant_id=tenant_id,
        )
    except ValueError as e:
        logger.warning(
            "agent.input_validation_failed",
            tenant_id=tenant_id,
            error=str(e),
        )
        return {
            "answer": "Input validation failed. Please check your input and try again.",
            "tools_used": [],
            "error": "input_validation_failed",
        }

    translated = _translate_math_intent(message)
    if translated:
        expr, intent = translated
        try:
            result = calculator_tool.invoke({"expression": expr})
            if result.startswith("Error:"):
                answer = result
            else:
                label = (
                    "average" if intent == "average" else "sum" if intent == "sum" else "product"
                )
                answer = f"The {label} is {result}."
            return {"answer": answer, "tools_used": ["calculator_tool"]}
        except Exception as e:
            logger.warning(
                "react_agent.math_intent_failed", error=str(e), expression=expr, intent=intent
            )

    graph = agent_graph(tenant_id, get_document_fn)
    if not graph:
        return {"answer": "LLM not configured.", "tools_used": [], "error": "llm_not_configured"}

    def _extract_result(res: dict) -> tuple[str, list[str]]:
        final = None
        used: list[str] = []
        for m in reversed(res["messages"]):
            if isinstance(m, AIMessage):
                final = m.content or "No response."
                break
        for m in res["messages"]:
            if isinstance(m, ToolMessage):
                used.append(m.name)
        return final or "No response.", list(dict.fromkeys(used))

    inputs = {"messages": [HumanMessage(content=message)]}
    result = await graph.ainvoke(inputs)
    answer, tools_used = _extract_result(result)

    if _is_malformed(answer) or not answer or answer == "No response.":
        summary = await _search_and_summarize(message)
        if summary:
            answer = summary
            tools_used = list(dict.fromkeys(tools_used + ["search_tool"]))
        else:
            query = _search_query_from_message(message)
            search_content = search_tool.invoke({"query": query})
            if (
                search_content
                and "No web results" not in search_content
                and "Search failed" not in search_content
            ):
                first_block = search_content.split("\n\n")[0] if search_content else ""
                if len(first_block) > 800:
                    first_block = first_block[:797] + "..."
                answer = f"Based on web search:\n\n{first_block}"
                tools_used = list(dict.fromkeys(tools_used + ["search_tool"]))
            else:
                answer = (
                    "I couldn't find an answer. Try asking again or rephrasing. "
                    "You can also try math (e.g. average of 1,2,5,6) or document lookup."
                )

    return {"answer": answer, "tools_used": tools_used}


async def run_agent_stream(
    tenant_id: str,
    message: str,
    get_document_fn: Any,
) -> AsyncIterator[str]:
    """Stream agent response tokens."""
    translated = _translate_math_intent(message)
    if translated:
        expr, intent = translated
        try:
            result = calculator_tool.invoke({"expression": expr})
            if result.startswith("Error:"):
                answer = result
            else:
                label = (
                    "average" if intent == "average" else "sum" if intent == "sum" else "product"
                )
                answer = f"The {label} is {result}."
            yield answer
            return
        except Exception as e:
            logger.warning(
                "react_agent.math_intent_failed", error=str(e), expression=expr, intent=intent
            )

    graph = agent_graph(tenant_id, get_document_fn)
    if not graph:
        yield "LLM not configured. Set LLM_PROVIDER and LLM_BASE_URL."
        return

    inputs = {"messages": [HumanMessage(content=message)]}
    async for msg, metadata in graph.astream(inputs, stream_mode="messages"):
        if hasattr(msg, "content") and msg.content and isinstance(msg.content, str):
            yield msg.content
        elif isinstance(msg, dict) and msg.get("content"):
            yield msg["content"]
