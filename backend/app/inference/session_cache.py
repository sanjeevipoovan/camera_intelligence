"""
Session and KV Cache Alignment Manager.

Ensures byte-level deterministic prefix formatting so that vLLM and Ollama
hit the GPU Key-Value (KV) cache on follow-up questions.
"""

import hashlib
import time
from dataclasses import dataclass, field
from pydantic import BaseModel


@dataclass
class ConversationTurn:
    role: str
    content: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class VisualSession:
    session_id: str
    image_hash: str
    image_bytes: bytes
    history: list[ConversationTurn] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)


class SessionCacheManager:
    """In-memory session registry enabling KV cache reuse across follow-up queries."""

    def __init__(self, ttl_seconds: float = 600.0) -> None:
        self._sessions: dict[str, VisualSession] = {}
        self.ttl = ttl_seconds

    def compute_image_hash(self, image_bytes: bytes) -> str:
        """Create unique SHA256 digest of image to verify prefix match."""
        return hashlib.sha256(image_bytes).hexdigest()

    def get_or_create_session(
        self, session_id: str | None, image_bytes: bytes
    ) -> VisualSession:
        """Retrieve existing visual session or register a new one."""
        self._cleanup_expired()
        img_hash = self.compute_image_hash(image_bytes)

        if session_id and session_id in self._sessions:
            session = self._sessions[session_id]
            # Verify if the same image is being referred to
            if session.image_hash == img_hash:
                session.last_accessed = time.time()
                return session

        # Generate new deterministic session ID
        new_id = session_id or f"sess_{img_hash[:16]}"
        session = VisualSession(
            session_id=new_id,
            image_hash=img_hash,
            image_bytes=image_bytes,
            history=[],
        )
        self._sessions[new_id] = session
        return session

    def build_deterministic_messages(
        self, session: VisualSession, current_prompt: str, image_b64: str
    ) -> list[dict]:
        """
        Builds the OpenAI/vLLM message payload.
        
        CRITICAL FOR KV CACHING:
        The image MUST always be in the first turn with static system instructions.
        Follow-up questions are appended to the tail so the engine reuses the
        entire prefix KV cache of previous tokens.
        """
        messages = []

        # Turn 1: System prompt + Image + Initial Question
        first_content = [
            {"type": "text", "text": "You are a visual assistant analyzing this mobile image."},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
        ]

        if not session.history:
            first_content.append({"type": "text", "text": current_prompt})
            messages.append({"role": "user", "content": first_content})
        else:
            # Reconstruct history deterministically
            first_turn = session.history[0]
            first_content.append({"type": "text", "text": first_turn.content})
            messages.append({"role": "user", "content": first_content})

            # Append intermediate turns
            for turn in session.history[1:]:
                messages.append({"role": turn.role, "content": turn.content})

            # Append current question
            messages.append({"role": "user", "content": current_prompt})

        return messages

    def record_turn(self, session: VisualSession, user_prompt: str, assistant_response: str) -> None:
        """Record dialog turns to keep conversation context updated."""
        session.history.append(ConversationTurn(role="user", content=user_prompt))
        session.history.append(ConversationTurn(role="assistant", content=assistant_response))
        session.last_accessed = time.time()

    def _cleanup_expired(self) -> None:
        """Evict expired sessions to free memory."""
        now = time.time()
        expired = [sid for sid, s in self._sessions.items() if now - s.last_accessed > self.ttl]
        for sid in expired:
            del self._sessions[sid]


session_cache = SessionCacheManager()