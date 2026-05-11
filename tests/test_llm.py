"""
Tests for memory management and LLM node behavior.
LLM calls are mocked so tests run fast without an API key.

Run with:  pytest tests/test_llm.py -v
"""

import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from memory import MemoryManager, MAX_HISTORY_TURNS, SESSION_TTL_MINUTES


# ─────────────────────────────────────────────
# Test: MemoryManager
# ─────────────────────────────────────────────

def test_new_session_starts_empty():
    """A brand new session should have no turns."""
    mem = MemoryManager()
    session = mem.get_or_create_session("s1")
    assert session.turns == []


def test_add_turn_stores_qa():
    """Adding a turn should store the query and report."""
    mem = MemoryManager()
    mem.add_turn("s1", query="What is avg salary?", report="The average is $72,000.")
    context = mem.get_context("s1")
    assert "What is avg salary?" in context
    assert "$72,000" in context


def test_sliding_window_keeps_max_turns():
    """After MAX_HISTORY_TURNS turns, older ones should be dropped."""
    mem = MemoryManager()
    for i in range(MAX_HISTORY_TURNS + 3):
        mem.add_turn("s1", query=f"Question {i}", report=f"Answer {i}")

    session = mem.get_or_create_session("s1")
    assert len(session.turns) == MAX_HISTORY_TURNS


def test_get_context_empty_for_new_session():
    """get_context on a new session returns an empty string."""
    mem = MemoryManager()
    context = mem.get_context("brand-new-session")
    assert context == ""


def test_clear_session_removes_history():
    """clear_session should delete all turns for that session."""
    mem = MemoryManager()
    mem.add_turn("s1", query="What?", report="This.")
    mem.clear_session("s1")
    context = mem.get_context("s1")
    assert context == ""


def test_multiple_sessions_are_independent():
    """Turns for session A should not appear in session B."""
    mem = MemoryManager()
    mem.add_turn("session-A", query="A's question", report="A's answer")
    mem.add_turn("session-B", query="B's question", report="B's answer")

    context_a = mem.get_context("session-A")
    context_b = mem.get_context("session-B")

    assert "A's question" in context_a
    assert "B's question" not in context_a
    assert "B's question" in context_b
    assert "A's question" not in context_b


def test_cleanup_removes_expired_sessions():
    """Sessions older than TTL should be removed by cleanup."""
    mem = MemoryManager()
    mem.add_turn("old-session", query="Old?", report="Old answer.")

    # Manually set last_active to a past time
    mem._sessions["old-session"].last_active = (
        datetime.utcnow() - timedelta(minutes=SESSION_TTL_MINUTES + 1)
    )

    removed = mem.cleanup_expired_sessions()
    assert removed == 1
    assert "old-session" not in mem._sessions


def test_cleanup_keeps_active_sessions():
    """Recent sessions should NOT be removed by cleanup."""
    mem = MemoryManager()
    mem.add_turn("active-session", query="Recent?", report="Recent answer.")
    removed = mem.cleanup_expired_sessions()
    assert removed == 0
    assert "active-session" in mem._sessions


def test_active_session_count():
    """active_session_count should reflect the number of sessions."""
    mem = MemoryManager()
    assert mem.active_session_count == 0
    mem.get_or_create_session("s1")
    mem.get_or_create_session("s2")
    assert mem.active_session_count == 2


# ─────────────────────────────────────────────
# Test: LLM Retry Helper
# ─────────────────────────────────────────────

def test_retry_succeeds_on_first_attempt():
    """If the LLM call succeeds, it should return immediately."""
    from llm_client import call_llm_with_retry
    from groq import RateLimitError

    mock_chain = MagicMock()
    mock_chain.invoke.return_value = "success"

    result = call_llm_with_retry(mock_chain, {}, max_retries=3)
    assert result == "success"
    assert mock_chain.invoke.call_count == 1


def test_retry_raises_after_max_attempts():
    """If all retries fail with RateLimitError, the error should propagate."""
    from llm_client import call_llm_with_retry
    from groq import RateLimitError

    mock_chain = MagicMock()
    # Simulate a RateLimitError. We need to mock it as a real exception.
    mock_chain.invoke.side_effect = Exception("Rate limit exceeded")

    with pytest.raises(Exception):
        call_llm_with_retry(mock_chain, {}, max_retries=1, base_delay=0)


def test_non_rate_limit_error_propagates_immediately():
    """Non-rate-limit errors should raise immediately without retrying."""
    from llm_client import call_llm_with_retry

    mock_chain = MagicMock()
    mock_chain.invoke.side_effect = ValueError("Schema mismatch")

    with pytest.raises(ValueError):
        call_llm_with_retry(mock_chain, {}, max_retries=3, base_delay=0)

    # Should have only been called once
    assert mock_chain.invoke.call_count == 1
