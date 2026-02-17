"""Tests for ReAct agent (run_agent, run_agent_stream) - TDD."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from langchain_core.messages import AIMessage, ToolMessage

from app.agents import run_agent, run_agent_stream
from app.agents.react_agent import agent_graph


@pytest.mark.asyncio
async def test_run_agent_returns_llm_not_configured_when_no_llm():
    """run_agent returns fallback when LLM not configured."""
    with patch("app.core.config.get_settings") as mock_settings:
        mock_settings.return_value.llm_base_url = None
        mock_settings.return_value.llm_provider = ""

        result = await run_agent(
            tenant_id="t1",
            message="What is 2+2?",
            get_document_fn=AsyncMock(return_value=None),
        )
        assert "LLM not configured" in result["answer"]
        assert result.get("error") == "llm_not_configured"


@pytest.mark.asyncio
async def test_run_agent_invokes_graph_when_configured():
    """run_agent returns answer when LLM configured and graph runs."""
    mock_graph = AsyncMock()
    mock_graph.ainvoke = AsyncMock(
        return_value={
            "messages": [
                ToolMessage(content="4", tool_call_id="x", name="calculator_tool"),
                AIMessage(content="4", tool_calls=[]),
            ]
        }
    )

    with patch("app.agents.react_agent.agent_graph") as mock_agent_graph:
        mock_agent_graph.return_value = mock_graph

        result = await run_agent(
            tenant_id="t1",
            message="What is 2+2?",
            get_document_fn=AsyncMock(return_value=None),
        )
        assert result["answer"] == "4"
        assert "calculator_tool" in result.get("tools_used", [])


@pytest.mark.asyncio
async def test_run_agent_stream_returns_fallback_when_not_configured():
    """run_agent_stream yields fallback when LLM not configured."""
    with patch("app.core.config.get_settings") as mock_settings:
        mock_settings.return_value.llm_base_url = None
        mock_settings.return_value.llm_provider = ""

        tokens = []
        async for t in run_agent_stream(
            tenant_id="t1",
            message="Hi",
            get_document_fn=AsyncMock(return_value=None),
        ):
            tokens.append(t)
        assert "".join(tokens) == "LLM not configured. Set LLM_PROVIDER and LLM_BASE_URL."


@pytest.mark.asyncio
async def test_run_agent_stream_yields_tokens_when_configured():
    """run_agent_stream yields tokens when LLM configured."""

    async def mock_astream(*args, stream_mode=None, **kwargs):
        yield (type("Msg", (), {"content": "The answer is "})(), {})
        yield (type("Msg", (), {"content": "4."})(), {})

    mock_graph = AsyncMock()
    mock_graph.astream = mock_astream

    with patch("app.agents.react_agent.agent_graph") as mock_agent_graph:
        mock_agent_graph.return_value = mock_graph

        tokens = []
        async for t in run_agent_stream(
            tenant_id="t1",
            message="What is 2+2?",
            get_document_fn=AsyncMock(return_value=None),
        ):
            tokens.append(t)
        assert "".join(tokens) == "The answer is 4."


@pytest.mark.asyncio
async def test_run_agent_returns_no_response_when_empty_content():
    """run_agent returns 'No response.' when ainvoke has AIMessage with empty content."""
    mock_graph = AsyncMock()
    mock_graph.ainvoke = AsyncMock(
        return_value={
            "messages": [
                AIMessage(content="", tool_calls=[]),
            ]
        }
    )

    with patch("app.agents.react_agent.agent_graph") as mock_agent_graph:
        mock_agent_graph.return_value = mock_graph

        result = await run_agent(
            tenant_id="t1",
            message="Hi",
            get_document_fn=AsyncMock(return_value=None),
        )
        assert result["answer"] == "No response."


@pytest.mark.asyncio
async def test_run_agent_returns_final_ai_message():
    """run_agent returns final AIMessage content from ainvoke result."""
    mock_graph = AsyncMock()
    mock_graph.ainvoke = AsyncMock(
        return_value={
            "messages": [
                ToolMessage(content="42", tool_call_id="x", name="calculator_tool"),
                AIMessage(content="The answer is 42.", tool_calls=[]),
            ]
        }
    )

    with patch("app.agents.react_agent.agent_graph") as mock_agent_graph:
        mock_agent_graph.return_value = mock_graph

        result = await run_agent(
            tenant_id="t1",
            message="What is 6*7?",
            get_document_fn=AsyncMock(return_value=None),
        )
        assert result["answer"] == "The answer is 42."


def test_agent_graph_returns_compiled_graph_when_configured():
    """agent_graph returns compiled LangGraph when LLM is configured."""
    mock_settings = MagicMock()
    mock_settings.llm_base_url = "http://localhost:11434"
    mock_settings.llm_provider = "ollama"
    mock_settings.llm_model = "llama3"
    mock_settings.llm_api_key = None

    mock_model = MagicMock()
    mock_model.bind_tools.return_value = mock_model

    with (
        patch("app.core.config.get_settings", return_value=mock_settings),
        patch(
            "app.agents.react_agent._get_chat_model",
            return_value=mock_model,
        ),
    ):
        graph = agent_graph("t1", AsyncMock(return_value=None))
    assert graph is not None
    assert hasattr(graph, "ainvoke")


@pytest.mark.asyncio
async def test_run_agent_stream_yields_dict_content():
    """run_agent_stream yields content when message is dict with content key."""

    async def mock_astream(*args, stream_mode=None, **kwargs):
        yield ({"content": "Hello "}, {})
        yield ({"content": "world."}, {})

    mock_graph = AsyncMock()
    mock_graph.astream = mock_astream

    with patch("app.agents.react_agent.agent_graph") as mock_agent_graph:
        mock_agent_graph.return_value = mock_graph

        tokens = []
        async for t in run_agent_stream(
            tenant_id="t1",
            message="Hi",
            get_document_fn=AsyncMock(return_value=None),
        ):
            tokens.append(t)
        assert "".join(tokens) == "Hello world."
