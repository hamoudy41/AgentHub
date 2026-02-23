"""ReAct agent with LangGraph - tools: calculator, search, document lookup."""

from __future__ import annotations

import re
from typing import Annotated, Any, AsyncIterator, Awaitable, Callable, Literal, Sequence, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from app.core.logging import get_logger
from app.security import sanitize_user_input

from .chat_models import create_chat_model
from .prompts import REACT_SYSTEM_PROMPT
from .tools import BASE_TOOLS, calculator_tool, create_document_lookup_tool
from .tools.search import SearchToolError, SearchToolNoResults, search_web

logger = get_logger(__name__)

GetDocumentFn = Callable[[str, str], Awaitable[dict[str, Any] | None]]


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
    if not isinstance(text, str):
        return False
    t = text.strip()
    if not t:
        return False
    if t.startswith("{") and '"parameters"' in t and '"name"' in t:
        return True
    # Common tool-call-ish payloads from various providers
    return t.startswith("{") and ('"tool_calls"' in t or '"function"' in t)


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


def _create_agent_graph(tools: list) -> Any:
    """Build the ReAct graph with given tools."""
    from app.core.config import get_settings

    settings = get_settings()
    model = create_chat_model(settings)
    if not model:
        return None
    model_with_tools = model.bind_tools(tools)

    def call_model(state: AgentState, config: RunnableConfig) -> dict:
        system = SystemMessage(content=REACT_SYSTEM_PROMPT)
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
    get_document_fn: GetDocumentFn,
) -> Any:
    """Create the agent graph with document lookup bound to tenant."""
    doc_tool = create_document_lookup_tool(tenant_id, get_document_fn)
    tools = BASE_TOOLS + [doc_tool]
    return _create_agent_graph(tools)


def _get_model_without_tools() -> Any:
    """Get chat model without tool binding, for summarization."""
    from app.core.config import get_settings

    settings = get_settings()
    return create_chat_model(settings)


async def _summarize_search_results(
    message: str,
    *,
    search_content: str,
    strict_english: bool,
) -> str | None:
    """Summarize search results for a user question."""
    model = _get_model_without_tools()
    if not model:
        return None

    if strict_english:
        system = SystemMessage(
            content=(
                "You are a concise summarizer. Respond ONLY in English. "
                "If the source text is not English, translate and summarize it in English. "
                "Never output tool-call JSON or schemas."
            )
        )
    else:
        system = SystemMessage(
            content=(
                "You are a concise summarizer. Respond in English using plain text only. "
                "Never output tool-call JSON or schemas."
            )
        )

    prompt = (
        "Summarize the following web search results in 2–4 concise sentences. "
        "Answer the user's question directly.\n\n"
        f"Question: {message}\n\n"
        f"Search results:\n{search_content[:2000]}"
    )

    try:
        response = await model.ainvoke([system, HumanMessage(content=prompt)])
    except Exception:
        return None

    content = getattr(response, "content", None)
    if not content:
        return None

    text = str(content).strip()
    if not text or _is_malformed(text):
        return None
    return text


WebFallbackStatus = Literal["ok", "no_results", "search_failed", "summarize_failed"]


async def _web_fallback_answer(message: str) -> tuple[str | None, WebFallbackStatus]:
    """Try answering by searching the web and summarizing results."""
    query = _search_query_from_message(message)
    try:
        search_content = search_web(query)
    except SearchToolNoResults:
        return None, "no_results"
    except SearchToolError as e:
        logger.warning("react_agent.search_failed", error=str(e), query=query)
        return None, "search_failed"

    summary = await _summarize_search_results(
        message,
        search_content=search_content,
        strict_english=False,
    )
    if summary:
        return summary, "ok"

    summary = await _summarize_search_results(
        message,
        search_content=search_content,
        strict_english=True,
    )
    if summary:
        return summary, "ok"

    return None, "summarize_failed"


async def run_agent(
    tenant_id: str,
    message: str,
    get_document_fn: GetDocumentFn,
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

    def _extract_result(res: dict[str, Any]) -> tuple[str, list[str]]:
        messages = res.get("messages") or []
        final: str | None = None
        used: list[str] = []

        for m in reversed(messages):
            if isinstance(m, AIMessage):
                content = m.content
                final = content if isinstance(content, str) else str(content)
                final = final.strip() or "No response."
                break

        for m in messages:
            if isinstance(m, ToolMessage) and getattr(m, "name", None):
                used.append(m.name)

        return final or "No response.", list(dict.fromkeys(used))

    inputs = {"messages": [HumanMessage(content=message)]}
    try:
        result = await graph.ainvoke(inputs)
    except Exception as e:
        logger.exception("react_agent.graph_invoke_failed", tenant_id=tenant_id, error=str(e))
        return {
            "answer": "Something went wrong while generating a response. Please try again.",
            "tools_used": [],
            "error": "agent_failed",
        }
    answer, tools_used = _extract_result(result)

    if _is_malformed(answer) or not answer or answer == "No response.":
        summary, status = await _web_fallback_answer(message)
        if summary:
            answer = summary
            tools_used = list(dict.fromkeys(tools_used + ["search_tool"]))
        elif status == "summarize_failed":
            answer = (
                "I found some information but couldn't format it properly. "
                "Try rephrasing your question, or use math (e.g. average of 1,2,5,6) "
                "or document lookup."
            )
            tools_used = list(dict.fromkeys(tools_used + ["search_tool"]))
        elif status == "search_failed":
            answer = (
                "I couldn't perform a web search right now. Try rephrasing your question "
                "or try again later."
            )
        else:
            answer = (
                "I couldn't find an answer. Try asking again or rephrasing. "
                "You can also try math (e.g. average of 1,2,5,6) or document lookup."
            )

    return {"answer": answer, "tools_used": tools_used}


async def run_agent_stream(
    tenant_id: str,
    message: str,
    get_document_fn: GetDocumentFn,
) -> AsyncIterator[str]:
    """Stream agent response tokens."""
    # Sanitize user input for security (match run_agent behavior)
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
        yield "Input validation failed. Please check your input and try again."
        return

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
    try:
        async for msg, metadata in graph.astream(inputs, stream_mode="messages"):
            if isinstance(msg, AIMessage) and msg.content:
                yield msg.content if isinstance(msg.content, str) else str(msg.content)
            elif isinstance(msg, dict):
                content = msg.get("content")
                if content:
                    yield content if isinstance(content, str) else str(content)
    except Exception as e:
        logger.exception("react_agent.graph_stream_failed", tenant_id=tenant_id, error=str(e))
        yield "Something went wrong while generating a response. Please try again."
