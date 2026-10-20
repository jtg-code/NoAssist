"""Provider-neutrale Typen für Chats: Nachrichten, Events, Fehler und Turn-Anfragen.

Dieses Modul hat bewusst keine schweren Abhängigkeiten (nur stdlib), damit es
von allen Schichten (Provider, Session, Speicher, Server) importiert werden kann.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Callable, Literal

if TYPE_CHECKING:
    from toolset import ToolSet


# --- Fehlercodes -------------------------------------------------------------

AUTH_REQUIRED = "auth_required"
CLI_MISSING = "cli_missing"
PROVIDER_UNREACHABLE = "provider_unreachable"
MODEL_NOT_FOUND = "model_not_found"
RATE_LIMIT = "rate_limit"
BAD_REQUEST = "bad_request"
TOOL_UNSUPPORTED = "tool_unsupported"
INTERNAL = "internal"
PROVIDER_ERROR = "provider_error"  # Standardcode von ProviderError

ERROR_CODES = (
    AUTH_REQUIRED,
    CLI_MISSING,
    PROVIDER_UNREACHABLE,
    MODEL_NOT_FOUND,
    RATE_LIMIT,
    BAD_REQUEST,
    TOOL_UNSUPPORTED,
    INTERNAL,
)

Role = Literal["user", "assistant", "tool"]
DoneStatus = Literal["completed", "cancelled", "error"]


def iso_now() -> str:
    """Aktuelle Zeit als ISO-8601-String in UTC (Sekundengenauigkeit)."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_message_id() -> str:
    """Erzeugt eine neue Nachrichten-ID der Form ``m_<32 hex>``."""
    return "m_" + uuid.uuid4().hex


# --- Nachrichten -------------------------------------------------------------

@dataclass
class ToolCall:
    """Ein Tool-Aufruf, den das Modell in einer Assistenten-Nachricht anfordert."""

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name, "arguments": dict(self.arguments)}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ToolCall:
        return cls(id=d["id"], name=d["name"], arguments=dict(d.get("arguments") or {}))


@dataclass
class Message:
    """Eine provider-neutrale Chat-Nachricht.

    - ``tool_calls`` nur bei ``role == "assistant"``.
    - ``tool_call_id``, ``name`` und ``is_error`` nur bei ``role == "tool"``.
    - ``provider_id`` und ``model`` halten fest, wer die Nachricht erzeugt hat.
    """

    role: Role
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None
    is_error: bool = False
    provider_id: str | None = None
    model: str | None = None
    id: str = field(default_factory=new_message_id)
    created_at: str = field(default_factory=iso_now)

    def to_dict(self) -> dict[str, Any]:
        """Wandelt die Nachricht in ein JSON-taugliches Dict um."""
        return {
            "id": self.id,
            "role": self.role,
            "content": self.content,
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
            "tool_call_id": self.tool_call_id,
            "name": self.name,
            "is_error": self.is_error,
            "provider_id": self.provider_id,
            "model": self.model,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Message:
        """Baut eine Nachricht aus einem Dict (Gegenstück zu ``to_dict``).

        Fehlende Felder bekommen ihre Standardwerte; ``id`` und ``created_at``
        werden nur neu erzeugt, wenn sie fehlen.
        """
        role = d.get("role")
        if role not in ("user", "assistant", "tool"):
            raise ValueError(f"Ungültige Rolle: {role!r}")
        kwargs: dict[str, Any] = {
            "role": role,
            "content": d.get("content") or "",
            "tool_calls": [ToolCall.from_dict(tc) for tc in d.get("tool_calls") or []],
            "tool_call_id": d.get("tool_call_id"),
            "name": d.get("name"),
            "is_error": bool(d.get("is_error", False)),
            "provider_id": d.get("provider_id"),
            "model": d.get("model"),
        }
        if d.get("id"):
            kwargs["id"] = d["id"]
        if d.get("created_at"):
            kwargs["created_at"] = d["created_at"]
        return cls(**kwargs)


# --- Events ------------------------------------------------------------------

Event = dict[str, Any]
"""Ein Event ist ein JSON-taugliches Dict mit dem Schlüssel ``type``."""

Emit = Callable[[Event], None]
"""Callback zum Senden von Events. Muss threadsicher sein."""


def ev_turn_start(
    turn_id: str | None = None,
    chat_id: str | None = None,
    provider_id: str | None = None,
    model: str | None = None,
) -> Event:
    return {
        "type": "turn_start",
        "turn_id": turn_id,
        "chat_id": chat_id,
        "provider_id": provider_id,
        "model": model,
    }


def ev_delta(text: str) -> Event:
    """Ein Stück gestreamter Antworttext."""
    return {"type": "delta", "text": text}


def ev_reasoning(text: str) -> Event:
    """Ein Stück gestreamtes Denken (Reasoning/Thinking)."""
    return {"type": "reasoning", "text": text}


def ev_status(text: str) -> Event:
    """Statusmeldung für die Anzeige, z. B. "Modell wird geladen…"."""
    return {"type": "status", "text": text}


def ev_tool_call(call_id: str, name: str, arguments: dict[str, Any], source: str = "local") -> Event:
    return {
        "type": "tool_call",
        "call_id": call_id,
        "name": name,
        "arguments": arguments,
        "source": source,
    }


def ev_tool_result(
    call_id: str,
    name: str,
    result: str,
    is_error: bool = False,
    duration_ms: int = 0,
    truncated: bool = False,
) -> Event:
    return {
        "type": "tool_result",
        "call_id": call_id,
        "name": name,
        "result": result,
        "is_error": is_error,
        "truncated": truncated,
        "duration_ms": duration_ms,
    }


def ev_message(msg: Message) -> Event:
    """Eine fertige Nachricht; ``message`` ist ``msg.to_dict()`` (JSON-tauglich).

    Empfänger stellen die Nachricht mit ``Message.from_dict(ev["message"])`` wieder her.
    """
    return {"type": "message", "message": msg.to_dict()}


def ev_error(code: str, message: str, hint: str | None = None) -> Event:
    ev: Event = {"type": "error", "code": code, "message": message}
    if hint is not None:
        ev["hint"] = hint
    return ev


def ev_done(status: DoneStatus = "completed") -> Event:
    """Letztes Event eines Turns: ``completed``, ``cancelled`` oder ``error``."""
    if status not in ("completed", "cancelled", "error"):
        raise ValueError(f"Ungültiger Status: {status!r}")
    return {"type": "done", "status": status}


# --- Fehler und Provider-Infos -----------------------------------------------

class ProviderError(Exception):
    """Erwartbarer Fehler eines Providers. Der Text darf keine Geheimnisse enthalten."""

    def __init__(self, message: str, code: str = PROVIDER_ERROR, hint: str | None = None):
        super().__init__(message)
        self.message = message
        self.code = code
        self.hint = hint

    def to_event(self) -> Event:
        return ev_error(self.code, self.message, self.hint)


@dataclass
class ModelInfo:
    id: str
    label: str
    loaded: bool | None = None
    description: str | None = None


@dataclass
class ProviderStatus:
    ok: bool
    message: str
    hint: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class TurnRequest:
    """Alles, was ein Provider für einen Turn braucht."""

    history: list[Message]  # inkl. der neuen User-Nachricht am Ende
    model: str | None
    system_prompt: str
    tools: ToolSet
    cancel: threading.Event = field(default_factory=threading.Event)
    state: dict[str, Any] = field(default_factory=dict)  # provider-spezifischer, persistierter Chat-Zustand (veränderbar)
    reasoning: str | None = None  # Reasoning-Stufe (z. B. "low", "high"); None = Standard des Modells


if __name__ == "__main__":
    import json

    call = ToolCall(id="call_1", name="get_datetime", arguments={"tz": "Europe/Berlin"})
    msgs = [
        Message(role="user", content="Wie spät ist es? äöü"),
        Message(role="assistant", tool_calls=[call], provider_id="lmstudio", model="qwen"),
        Message(role="tool", content="12:00", tool_call_id="call_1", name="get_datetime"),
        Message(role="assistant", content="Es ist 12:00 Uhr."),
    ]
    for m in msgs:
        d = m.to_dict()
        roundtrip = Message.from_dict(json.loads(json.dumps(d, ensure_ascii=False)))
        assert roundtrip == m, (roundtrip, m)
        print(f"OK  {m.role:9} {m.id}  {m.content or m.tool_calls}")

    events = [
        ev_turn_start("t_1", "c_1", "lmstudio", "qwen"),
        ev_delta("Hallo"),
        ev_tool_call("call_1", "get_datetime", {}, "local"),
        ev_tool_result("call_1", "get_datetime", "12:00", duration_ms=3),
        ev_message(msgs[-1]),
        ProviderError("Nicht angemeldet", AUTH_REQUIRED, "claude auth login").to_event(),
        ev_done("completed"),
    ]
    for e in events:
        print(json.dumps(e, ensure_ascii=False))
