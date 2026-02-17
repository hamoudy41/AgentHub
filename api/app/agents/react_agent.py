"""ReAct agent with LangGraph - tools: calculator, search, document lookup."""

from __future__ import annotations

from typing import Annotated, Any, AsyncIterator, Sequence, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from .tools import BASE_TOOLS, create_document_lookup_tool


class AgentState(TypedDict):
    """State for the ReAct agent."""

    messages: Annotated[Sequence[BaseMessage], add_messages]


def _get_chat_model(provider: str, base_url: str, model: str, api_key: str | None = None):
    """Create LangChain ChatModel from our config."""
    base = str(base_url).rstrip("/")
    if provider == "ollama":
        from langchain_community.chat_models import ChatOllama

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
            content="You are a helpful AI assistant with access to tools. "
            "Use the calculator for math, search for web info, and document_lookup for document content. "
            "Respond concisely. If you cannot help, say so."
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


async def run_agent(
    tenant_id: str,
    message: str,
    get_document_fn: Any,
) -> dict[str, Any]:
    """Run the agent and return the final response."""
    graph = agent_graph(tenant_id, get_document_fn)
    if not graph:
        return {"answer": "LLM not configured.", "tools_used": [], "error": "llm_not_configured"}

    inputs = {"messages": [HumanMessage(content=message)]}
    result = await graph.ainvoke(inputs)
    final = None
    tools_used: list[str] = []
    for m in reversed(result["messages"]):
        if isinstance(m, AIMessage):
            final = m.content or "No response."
            break
    for m in result["messages"]:
        if isinstance(m, ToolMessage):
            tools_used.append(m.name)
    return {"answer": final or "No response.", "tools_used": list(dict.fromkeys(tools_used))}


async def run_agent_stream(
    tenant_id: str,
    message: str,
    get_document_fn: Any,
) -> AsyncIterator[str]:
    """Stream agent response tokens."""
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
