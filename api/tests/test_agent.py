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
async def test_run_agent_translates_average_to_calculator():
    """run_agent computes average directly via calculator—bypasses LLM."""
    with patch("app.agents.react_agent.calculator_tool") as mock_calc:
        mock_calc.invoke.return_value = "3.5"

        result = await run_agent(
            tenant_id="t1",
            message="find the average of 1, 2, 5, 6",
            get_document_fn=AsyncMock(return_value=None),
        )
        assert "3.5" in result["answer"]
        assert "average" in result["answer"].lower()
        assert "calculator_tool" in result.get("tools_used", [])
        mock_calc.invoke.assert_called_once_with({"expression": "(1+2+5+6)/4"})


@pytest.mark.asyncio
async def test_run_agent_average_with_spaces_and_and():
    """run_agent handles '1 2 5 and 6' format for average."""
    with patch("app.agents.react_agent.calculator_tool") as mock_calc:
        mock_calc.invoke.return_value = "3.5"

        result = await run_agent(
            tenant_id="t1",
            message="find the average of 1 2 5 and 6",
            get_document_fn=AsyncMock(return_value=None),
        )
        assert "3.5" in result["answer"]
        mock_calc.invoke.assert_called_once_with({"expression": "(1+2+5+6)/4"})


@pytest.mark.asyncio
async def test_run_agent_average_with_thousands_separators():
    """run_agent treats 1,000 and 2,000 as numbers with thousands separators."""
    with patch("app.agents.react_agent.calculator_tool") as mock_calc:
        mock_calc.invoke.return_value = "1500"

        result = await run_agent(
            tenant_id="t1",
            message="average of 1,000 and 2,000",
            get_document_fn=AsyncMock(return_value=None),
        )
        assert "1500" in result["answer"]
        mock_calc.invoke.assert_called_once_with({"expression": "(1000+2000)/2"})


@pytest.mark.asyncio
async def test_run_agent_average_plain_multi_digit():
    """run_agent treats 1000 and 2000 (no commas) as single numbers."""
    with patch("app.agents.react_agent.calculator_tool") as mock_calc:
        mock_calc.invoke.return_value = "1500"

        result = await run_agent(
            tenant_id="t1",
            message="average of 1000 and 2000",
            get_document_fn=AsyncMock(return_value=None),
        )
        assert "1500" in result["answer"]
        mock_calc.invoke.assert_called_once_with({"expression": "(1000+2000)/2"})


@pytest.mark.asyncio
async def test_translate_math_intent_average():
    """_translate_math_intent returns (expression, intent) for average/mean."""
    from app.agents.react_agent import _translate_math_intent

    result = _translate_math_intent("find the average of 1, 2, 5, 6")
    assert result is not None
    expr, intent = result
    assert expr == "(1+2+5+6)/4"
    assert intent == "average"

    result = _translate_math_intent("mean of 10, 20, 30")
    assert result is not None
    expr, intent = result
    assert expr == "(10+20+30)/3"
    assert intent == "average"


@pytest.mark.asyncio
async def test_translate_math_intent_thousands_separators():
    """_translate_math_intent treats 1,000 and 2,000 as single numbers, not 1, 0, 2, 0."""
    from app.agents.react_agent import _translate_math_intent

    result = _translate_math_intent("average of 1,000 and 2,000")
    assert result is not None
    expr, intent = result
    assert expr == "(1000+2000)/2"
    assert intent == "average"


@pytest.mark.asyncio
async def test_translate_math_intent_plain_multi_digit():
    """_translate_math_intent treats 1000 and 2000 (no commas) as single numbers, not 100, 0, 200, 0."""
    from app.agents.react_agent import _translate_math_intent

    result = _translate_math_intent("average of 1000 and 2000")
    assert result is not None
    expr, intent = result
    assert expr == "(1000+2000)/2"
    assert intent == "average"


@pytest.mark.asyncio
async def test_translate_math_intent_no_match():
    """_translate_math_intent returns None when no math intent."""
    from app.agents.react_agent import _translate_math_intent

    assert _translate_math_intent("how to learn python") is None
    assert _translate_math_intent("average of 5") is None  # single number


@pytest.mark.asyncio
async def test_search_query_strips_question_prefix():
    """_search_query_from_message strips common question prefixes."""
    from app.agents.react_agent import _search_query_from_message

    q = _search_query_from_message("what is the capital of France?")
    assert not q.lower().startswith("what is")
    assert len(q) > 0


@pytest.mark.asyncio
async def test_search_query_removes_stop_words():
    """_search_query_from_message removes generic stop words so substantive terms lead."""
    from app.agents.react_agent import _search_query_from_message

    q = _search_query_from_message("who is the best player in the world?")
    assert "best" not in q.lower()
    assert len(q) > 0


@pytest.mark.asyncio
async def test_run_agent_search_and_summarize_when_malformed_json():
    """run_agent searches and summarizes when model returns malformed JSON."""
    mock_graph = AsyncMock()
    mock_graph.ainvoke = AsyncMock(
        return_value={
            "messages": [
                AIMessage(
                    content='{"name": "software engineering", "parameters": {"type": "learning", "description": "methods"}}',
                    tool_calls=[],
                ),
            ]
        }
    )

    async def mock_summarize(msg: str) -> str:
        return "To learn software engineering: practice with projects, read documentation, and build things."

    with (
        patch("app.agents.react_agent.agent_graph") as mock_agent_graph,
        patch(
            "app.agents.react_agent._search_and_summarize",
            new_callable=AsyncMock,
            side_effect=mock_summarize,
        ),
    ):
        mock_agent_graph.return_value = mock_graph

        result = await run_agent(
            tenant_id="t1",
            message="how to learn software engineering?",
            get_document_fn=AsyncMock(return_value=None),
        )
        assert (
            "learn software engineering" in result["answer"].lower()
            or "projects" in result["answer"].lower()
        )
        assert mock_graph.ainvoke.call_count == 1


@pytest.mark.asyncio
async def test_run_agent_fallback_when_search_and_summarize_fails():
    """run_agent returns fallback when model fails and search/summarize yields nothing useful."""
    mock_graph = AsyncMock()
    mock_graph.ainvoke = AsyncMock(
        return_value={
            "messages": [
                AIMessage(
                    content='{"name": "x", "parameters": {"y": "z"}}',
                    tool_calls=[],
                ),
            ]
        }
    )

    with (
        patch("app.agents.react_agent.agent_graph") as mock_agent_graph,
        patch(
            "app.agents.react_agent._search_and_summarize",
            new_callable=AsyncMock,
            return_value=None,
        ),
        patch("app.agents.react_agent.search_tool") as mock_search,
    ):
        mock_agent_graph.return_value = mock_graph
        mock_search.invoke.return_value = "Search failed: error. Try rephrasing."

        result = await run_agent(
            tenant_id="t1",
            message="how to learn software engineering?",
            get_document_fn=AsyncMock(return_value=None),
        )
        assert "couldn't find" in result["answer"].lower() or "rephrase" in result["answer"].lower()
        assert mock_graph.ainvoke.call_count == 1


@pytest.mark.asyncio
async def test_run_agent_search_and_summarize_when_model_fails():
    """run_agent searches and summarizes when model returns malformed."""
    mock_graph = AsyncMock()
    mock_graph.ainvoke = AsyncMock(
        return_value={
            "messages": [
                AIMessage(content='{"name": "x", "parameters": {}}', tool_calls=[]),
            ]
        }
    )

    async def mock_summarize(msg: str) -> str:
        return "A regular expression (regex) is a sequence of characters that defines a search pattern, used for text matching and validation."

    with (
        patch("app.agents.react_agent.agent_graph") as mock_agent_graph,
        patch(
            "app.agents.react_agent._search_and_summarize",
            new_callable=AsyncMock,
            side_effect=mock_summarize,
        ),
    ):
        mock_agent_graph.return_value = mock_graph

        result = await run_agent(
            tenant_id="t1",
            message="what is regular expression",
            get_document_fn=AsyncMock(return_value=None),
        )
        assert "regular expression" in result["answer"].lower()
        assert "search pattern" in result["answer"].lower()
        assert "search_tool" in result.get("tools_used", [])


@pytest.mark.asyncio
async def test_run_agent_uses_raw_search_when_summarize_fails():
    """run_agent returns raw search results when summarization fails."""
    mock_graph = AsyncMock()
    mock_graph.ainvoke = AsyncMock(
        return_value={
            "messages": [
                AIMessage(content='{"name": "x", "parameters": {}}', tool_calls=[]),
            ]
        }
    )

    with (
        patch("app.agents.react_agent.agent_graph") as mock_agent_graph,
        patch(
            "app.agents.react_agent._search_and_summarize",
            new_callable=AsyncMock,
            return_value=None,
        ),
        patch("app.agents.react_agent.search_tool") as mock_search,
    ):
        mock_agent_graph.return_value = mock_graph
        mock_search.invoke.return_value = (
            "[1] Regular expression - Wikipedia\n"
            "A regular expression is a sequence of characters that defines a search pattern.\n"
            "Source: https://example.com/regex"
        )

        result = await run_agent(
            tenant_id="t1",
            message="what is regular expression",
            get_document_fn=AsyncMock(return_value=None),
        )
        assert "Based on web search" in result["answer"]
        assert "regular expression" in result["answer"].lower()
        assert "search pattern" in result["answer"].lower()


@pytest.mark.asyncio
async def test_run_agent_stream_math_shortcut_without_llm():
    """run_agent_stream computes average via calculator—works even without LLM (matches run_agent)."""
    with patch("app.agents.react_agent.calculator_tool") as mock_calc:
        mock_calc.invoke.return_value = "3.5"

        tokens = []
        async for t in run_agent_stream(
            tenant_id="t1",
            message="find the average of 1, 2, 5, 6",
            get_document_fn=AsyncMock(return_value=None),
        ):
            tokens.append(t)
        assert "".join(tokens) == "The average is 3.5."
        mock_calc.invoke.assert_called_once_with({"expression": "(1+2+5+6)/4"})


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
async def test_run_agent_returns_fallback_when_empty_content_and_search_fails():
    """run_agent returns fallback when model has empty content and search yields nothing."""
    mock_graph = AsyncMock()
    mock_graph.ainvoke = AsyncMock(
        return_value={
            "messages": [
                AIMessage(content="", tool_calls=[]),
            ]
        }
    )

    with (
        patch("app.agents.react_agent.agent_graph") as mock_agent_graph,
        patch(
            "app.agents.react_agent._search_and_summarize",
            new_callable=AsyncMock,
            return_value=None,
        ),
        patch("app.agents.react_agent.search_tool") as mock_search,
    ):
        mock_agent_graph.return_value = mock_graph
        mock_search.invoke.return_value = "No web results found for: Hi"

        result = await run_agent(
            tenant_id="t1",
            message="Hi",
            get_document_fn=AsyncMock(return_value=None),
        )
        assert "couldn't find" in result["answer"].lower() or "rephrase" in result["answer"].lower()


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
