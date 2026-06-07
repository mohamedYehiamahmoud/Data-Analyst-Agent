"""#185
memory.py — Conversation Memory Management

KEY IMPROVEMENTS over original:
  1. FIX: get_context() used `parts.insert(1, entry)` to build the context
     string, but then called "\n".join(parts). Because entries were inserted
     at index 1, the final join added newlines BETWEEN the header and each
     entry, producing messy output. Fixed to build a list in order and join
     at the end.
  2. FIX: The method iterated in reverse to stay within the char budget, but
     inserted each entry at index 1 — so the final order was REVERSED
     (most recent turn appeared first). Fixed to collect in reverse then
     reverse again before joining, preserving chronological order in output.
  3. ADDED: session_count property renamed to active_session_count for
     clarity, and a get_session() method added for inspection/testing.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

MAX_HISTORY_TURNS = 5       # How many Q&A pairs to keep per session
SESSION_TTL_MINUTES = 60    # Sessions expire after this many idle minutes
MAX_CONTEXT_CHARS = 4000    # Max characters of history injected into prompts


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
    csv_filename: Optional[str] = None
    last_image_dir: Optional[str] = None   # session subfolder of images/ for the most recent query


# ─────────────────────────────────────────────
# Memory Manager
# ─────────────────────────────────────────────

class MemoryManager:
    """
    Manages conversation history across multiple sessions.

    Usage:
        memory = MemoryManager()
        memory.add_turn("session-123", query="What is avg salary?", report="...")
        context = memory.get_context("session-123")
    """

    def __init__(self):
        self._sessions: dict[str, Session] = {}

    def get_or_create_session(
        self,
        session_id: str,
        csv_filename: Optional[str] = None,
    ) -> Session:
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

    def get_session(self, session_id: str) -> Optional[Session]:
        """Return an existing session, or None if it doesn't exist."""
        return self._sessions.get(session_id)

    def add_turn(self, session_id: str, query: str, report: str, image_dir: Optional[str] = None) -> None:
        """
        Record a completed Q&A exchange.
        Keeps only the most recent MAX_HISTORY_TURNS turns (sliding window).
        Also stores the image directory for the latest query so the email
        endpoint can embed charts without needing extra state.
        """
        session = self.get_or_create_session(session_id)
        session.turns.append(Turn(query=query, report=report))
        if image_dir:
            session.last_image_dir = image_dir

        if len(session.turns) > MAX_HISTORY_TURNS:
            session.turns = session.turns[-MAX_HISTORY_TURNS:]

    def get_context(self, session_id: str) -> str:
        """
        Return a formatted string of recent conversation history for
        injection into the next LLM prompt.

        FIX: Original code had a reversed-order bug where the most recent
        turn appeared first in the output. This version builds the list
        correctly in chronological order within the character budget.

        Returns an empty string if there's no history.
        """
        if session_id not in self._sessions:
            return ""

        session = self._sessions[session_id]
        if not session.turns:
            return ""

        collected_entries = []
        total_chars = 0

        # Walk in reverse to prioritise recent turns within the char budget
        for turn in reversed(session.turns):
            report_snippet = (
                turn.report[:1500] + "..."
                if len(turn.report) > 1500
                else turn.report
            )
            entry = f"Q: {turn.query}\nA: {report_snippet}\n"

            if total_chars + len(entry) > MAX_CONTEXT_CHARS:
                break

            collected_entries.append(entry)
            total_chars += len(entry)

        # FIX: Reverse so the oldest collected turn comes first (chronological)
        collected_entries.reverse()

        header = "[Previous conversation context]"
        return header + "\n" + "\n".join(collected_entries)

    def clear_session(self, session_id: str) -> None:
        """Delete a session (e.g. when user uploads a new CSV)."""
        if session_id in self._sessions:
            del self._sessions[session_id]
            logger.info(f"Session cleared: {session_id}")

    def cleanup_expired_sessions(self) -> int:
        """
        Remove sessions that haven't been used recently.
        Returns the number of sessions removed.
        """
        cutoff = datetime.utcnow() - timedelta(minutes=SESSION_TTL_MINUTES)
        expired = [
            sid
            for sid, session in self._sessions.items()
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

memory_manager = MemoryManager()
