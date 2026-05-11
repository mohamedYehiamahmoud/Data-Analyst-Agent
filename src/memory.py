"""
memory.py — Conversation Memory Management
==========================================
LLMs are stateless — they remember nothing between calls.
This module implements a simple in-memory session store so
that users can ask follow-up questions in a conversation.

How it works:
  - Each "session" gets a unique ID (like a conversation ID)
  - We store past Q&A pairs for that session
  - When the user asks a new question, we inject the recent
    history into the LLM prompt so it has context
  - We limit context length to avoid exceeding the LLM's
    context window (which would cause errors or extra cost)

Production note:
  For a real deployment, replace the in-memory dict with
  Redis or a database so sessions survive server restarts.
"""

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

MAX_HISTORY_TURNS = 5          # How many Q&A pairs to remember per session
SESSION_TTL_MINUTES = 60       # Sessions expire after this many minutes of inactivity
MAX_CONTEXT_CHARS = 4000       # Max characters of history to inject into prompts


# ─────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────

@dataclass
class Turn:
    """A single exchange: one user question + one assistant answer."""
    query: str
    report: str
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class Session:
    """All history for one conversation session."""
    session_id: str
    turns: list[Turn] = field(default_factory=list)
    last_active: datetime = field(default_factory=datetime.utcnow)
    csv_filename: Optional[str] = None   # Track which file this session is about


# ─────────────────────────────────────────────
# Memory Manager
# ─────────────────────────────────────────────

class MemoryManager:
    """
    Manages conversation history across multiple sessions.

    Usage:
        memory = MemoryManager()

        # Save a completed Q&A turn
        memory.add_turn("session-123", query="What is avg salary?", report="The average is...")

        # Get formatted history to inject into the next prompt
        context = memory.get_context("session-123")
    """

    def __init__(self):
        # dict[session_id -> Session]
        # defaultdict automatically creates a new Session if the key doesn't exist
        self._sessions: dict[str, Session] = {}

    def get_or_create_session(self, session_id: str, csv_filename: Optional[str] = None) -> Session:
        """Get an existing session or create a new one."""
        if session_id not in self._sessions:
            self._sessions[session_id] = Session(
                session_id=session_id,
                csv_filename=csv_filename,
            )
            logger.info(f"New session created: {session_id}")
        session = self._sessions[session_id]
        session.last_active = datetime.utcnow()
        return session

    def add_turn(self, session_id: str, query: str, report: str) -> None:
        """
        Record a completed Q&A exchange in the session's history.

        If we've hit MAX_HISTORY_TURNS, the oldest turn is dropped
        (sliding window) to keep the context window from growing forever.
        """
        session = self.get_or_create_session(session_id)
        session.turns.append(Turn(query=query, report=report))

        # Keep only the most recent N turns (sliding window)
        if len(session.turns) > MAX_HISTORY_TURNS:
            session.turns = session.turns[-MAX_HISTORY_TURNS:]

    def get_context(self, session_id: str) -> str:
        """
        Return a formatted string of recent conversation history,
        ready to be injected into the next LLM prompt.

        The output looks like:
            [Previous conversation]
            Q: What is the average age?
            A: The average age is 35.2 years...

            Q: How does that compare by gender?
            A: Female average: 34.1, Male average: 36.3...

        If there's no history (first question), returns an empty string.
        """
        if session_id not in self._sessions:
            return ""

        session = self._sessions[session_id]
        if not session.turns:
            return ""

        parts = ["[Previous conversation context]"]
        total_chars = 0

        # Walk turns in reverse so we include the most recent ones
        # if we run out of character budget
        for turn in reversed(session.turns):
            # Truncate long reports so one giant report doesn't crowd out others
            report_snippet = turn.report[:500] + "..." if len(turn.report) > 500 else turn.report
            entry = f"Q: {turn.query}\nA: {report_snippet}\n"

            if total_chars + len(entry) > MAX_CONTEXT_CHARS:
                break   # Stop adding history once we hit the budget

            parts.insert(1, entry)   # Insert after the header, in chronological order
            total_chars += len(entry)

        return "\n".join(parts)

    def clear_session(self, session_id: str) -> None:
        """Delete a session (e.g. when user uploads a new CSV)."""
        if session_id in self._sessions:
            del self._sessions[session_id]
            logger.info(f"Session cleared: {session_id}")

    def cleanup_expired_sessions(self) -> int:
        """
        Remove sessions that haven't been used recently.

        Call this periodically (e.g. via a background task) to prevent
        memory from growing unboundedly in a long-running server.

        Returns the number of sessions removed.
        """
        cutoff = datetime.utcnow() - timedelta(minutes=SESSION_TTL_MINUTES)
        expired = [
            sid for sid, session in self._sessions.items()
            if session.last_active < cutoff
        ]
        for sid in expired:
            del self._sessions[sid]

        if expired:
            logger.info(f"Cleaned up {len(expired)} expired sessions.")

        return len(expired)

    @property
    def active_session_count(self) -> int:
        return len(self._sessions)


# ─────────────────────────────────────────────
# Singleton instance
# ─────────────────────────────────────────────
# We create ONE MemoryManager for the whole app.
# FastAPI's dependency injection system will use this.

memory_manager = MemoryManager()
