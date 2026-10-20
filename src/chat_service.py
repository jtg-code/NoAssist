"""ChatService: Chats, Turns, Provider, Tools und MCP-Server für den Server (ohne Flask).

Der Service verbindet ``ChatStore``/``ConfigStore`` (storage.py), die Provider
(providers/registry.py) und die provider-unabhängige ``Session`` (session.py):

- **Turns** laufen in eigenen Worker-Threads. ``start_turn`` speichert die
  User-Nachricht sofort; der Worker baut pro Turn eine ``Session`` aus dem
  gespeicherten Verlauf, dem Provider des Chats, dem aktuellen gemeinsamen
  ``ToolSet`` und dem gespeicherten Provider-Zustand. Jede ``message`` wird
  sofort gespeichert, der Provider-Zustand am Ende (auch nach Abbruch/Fehler).
  Alle Events landen in einem Puffer (``Turn``), aus dem beliebig viele
  SSE-Clients ab einer ``seq`` lesen. Pro Chat läuft höchstens ein Turn
  (sonst ``Busy``); beendete Turns bleiben 5 Minuten abrufbar.
- **Provider** werden aus der Konfiguration gebaut und gecacht. Nach einer
  Änderung (``update_provider``/``delete_provider`` oder Bearbeiten der
  ``config.json``) wird die alte Instanz geschlossen, sobald kein Turn sie
  mehr nutzt.
- **Tools**: ``tools.get_tools()`` abzüglich ``config.tools.disabled`` plus die
  Tools der in ``config.mcp_servers`` eingetragenen MCP-Server. Die MCP-Server
  starten bei ``start()`` parallel im Hintergrund (je mit Zeitlimit
  ``startup_timeout``, Standard 30 s); ein fehlerhafter Server blockiert nichts
  und erscheint in ``list_tools`` mit ``status: "error"``. Jeder Tool-Aufruf
  hat ein Zeitlimit (``call_timeout``, Standard 120 s, siehe ``mcp_tools``).
- **Geheimnisse**: API-Keys, Werte aus ``api_key_env`` und ``env``-Werte der
  MCP-Server werden in Fehler-, Status- und Hinweistexten durch ``***``
  ersetzt (``redact``), aber nur ab ``MIN_SECRET_LEN`` (8) Zeichen: kurze
  Platzhalter-Keys wie ``"ollama"`` sind erlaubt, dienen aber nie als
  Schwärz-Muster. Chat-Inhalte (Titel, Nachrichten) werden nicht verändert.
  ``env`` der MCP-Server wird nie ausgegeben.

Wiederholen (``start_turn(..., retry=True, user_message_id=...)``): Die
User-Nachricht ``user_message_id`` wird erneut ausgeführt, ohne eine neue
anzulegen; alles, was hinter ihr gespeichert ist (Teilantworten und
Tool-Nachrichten des gescheiterten oder abgebrochenen Versuchs), wird gelöscht.
Damit ein veralteter "Erneut versuchen"-Knopf (z. B. in einem zweiten Fenster)
keine inzwischen fertige Antwort löscht, gilt (sonst ``RetryStale``, HTTP 409):

- ``user_message_id`` ist die letzte User-Nachricht des Chats, und
- der letzte Turn zu dieser Nachricht ist nicht erfolgreich abgeschlossen.
  Dazu merkt sich ``ChatStore`` pro Chat den zuletzt gestarteten Turn, seine
  User-Nachricht und seinen Status (``running``/``completed``/``cancelled``/
  ``error``; Spalten ``last_turn_*``, Schema-Version 2). ``running`` ohne
  laufenden Turn bedeutet: der Server wurde mitten im Turn beendet; das darf
  wiederholt werden. Ist kein Status vermerkt (Chats von vor Version 2), ist
  die Wiederholung nur erlaubt, wenn hinter der Nachricht nichts steht.

Läuft im Chat gerade ein Turn, gibt es wie beim Senden ``Busy``.

Demo (aus ``src``): ``python -m chat_service`` (Echo-Provider, offline) oder
``python -m chat_service --provider lmstudio "Wie spät ist es?"``.
"""

# Validierungsfehler sind bewusst ValueError (auch bei falschen Typen): der Server macht daraus HTTP 400.
# ruff: noqa: TRY004

from __future__ import annotations

import copy
import logging
import os
import re
import secrets
import threading
import time
import uuid
from collections.abc import Callable, Generator, Iterable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import urlsplit

from agents import MAIN_EXCLUDED_TOOLS, AgentStore
from chat_types import (
    INTERNAL,
    Event,
    Message,
    ProviderError,
    ev_done,
    ev_error,
    ev_message,
    ev_turn_start,
)
from storage import MIN_SECRET_LEN, ChatStore, ConfigStore

if TYPE_CHECKING:
    from chat_types import DoneStatus
    from mcp_tools import McpToolSource
    from providers.base import Provider
    from toolset import ToolSet, ToolSpec

log = logging.getLogger(__name__)

__all__ = ["Busy", "ChatService", "NotFound", "RetryStale", "ServiceClosed", "Turn"]

# --- Konstanten ------------------------------------------------------------------

BUILTIN_PROVIDER_IDS = ("lmstudio", "claude")
"""Eingebaute Provider: können deaktiviert, aber nicht gelöscht werden."""
ADDABLE_PROVIDER_TYPES = ("openai",)

TURN_TTL_S = 300.0
"""So lange bleiben beendete Turns (samt Event-Puffer) abrufbar."""
FORCE_FINISH_S = 10.0
"""Reagiert ein Provider so lange nach dem Abbruch nicht, wird der Turn ohne ihn beendet."""
MCP_START_TIMEOUT_S = 30.0
"""Standard-Zeitlimit für den Start eines MCP-Servers (änderbar per ``startup_timeout``)."""
MCP_MAX_CALL_TIMEOUT_S = 3600.0
"""Obergrenze für ``call_timeout`` eines MCP-Servers (Standard siehe ``mcp_tools.DEFAULT_CALL_TIMEOUT_S``)."""

MAX_CONTENT_CHARS = 1_000_000
MAX_TITLE_CHARS = 500
MAX_SYSTEM_PROMPT_CHARS = 200_000
MAX_NAME_CHARS = 100
MAX_ID_CHARS = 200
# MIN_SECRET_LEN (aus storage): kürzere Keys/env-Werte werden nie geschwärzt (z. B. "ollama", "1").

THEMES = ("system", "light", "dark")
REASONING_LEVELS = ("off", "minimal", "low", "medium", "high", "xhigh", "max")
AGENT_ROLES = ("agent", "subagent", "assistant")
"""Rollen mit eigenem Standard-Provider/-Modell (Einstellungen ``<rolle>_provider`` und ``<rolle>_model``)."""
AUTH_MODES = ("login", "api_key")

_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")

_PROVIDER_FIELDS: dict[str, frozenset[str]] = {
    "openai": frozenset({
        "name", "enabled", "base_url", "api_key", "api_key_env", "default_model", "models", "supports_tools", "timeout",
    }),
    "claude_agent": frozenset({"name", "enabled", "auth_mode", "api_key", "cli_path", "default_model"}),
    "lmstudio": frozenset({"name", "enabled", "default_model", "api_host"}),
}
_READONLY_PROVIDER_FIELDS = frozenset({"id", "type", "builtin", "has_api_key"})
_CACHE_IGNORED_FIELDS = frozenset({"enabled"})
"""Änderungen an diesen Feldern erfordern keinen neuen Provider."""


# --- Fehler ----------------------------------------------------------------------

class NotFound(LookupError):
    """Chat, Provider oder Turn existiert nicht (Server: HTTP 404)."""


class Busy(RuntimeError):
    """In diesem Chat läuft bereits ein Turn (Server: HTTP 409)."""


class RetryStale(RuntimeError):
    """Die Wiederholung passt nicht (mehr) zum Chat (Server: HTTP 409, Code ``retry_stale``).

    Die User-Nachricht ist nicht mehr die letzte, oder zu ihr gibt es schon
    eine fertige Antwort. Der Client sollte den Chat neu laden.
    """


class ServiceClosed(RuntimeError):
    """Der Service wurde bereits beendet."""


# --- Turn ------------------------------------------------------------------------

class Turn:
    """Ein laufender oder beendeter Turn mit Event-Puffer.

    Events bekommen eine laufende Nummer (``seq``, ab 1). Der Puffer hält alle
    Events des Turns; ``events_after(seq, timeout)`` liefert die neueren (und
    wartet bei Bedarf). Nach ``done`` nimmt ``publish`` nichts mehr an.
    """

    def __init__(self, chat_id: str, provider_id: str | None = None, model: str | None = None):
        self.id = "t_" + uuid.uuid4().hex
        self.chat_id = chat_id
        self.provider_id = provider_id
        self.model = model
        self.cancel = threading.Event()
        self.status: str | None = None
        """Endstatus (``completed``, ``cancelled`` oder ``error``), sobald ``done`` veröffentlicht ist."""
        self.created_at = time.monotonic()
        self.finished_at: float | None = None
        self._events: list[Event] = []
        self._cond = threading.Condition()
        self._finished = False
        # Zustand für den Service (Speichern, Abbruch, Wächter)
        self._store_lock = threading.Lock()
        self._closing = False  # Worker schließt ab: keine Nachrichten mehr speichern
        self._abandoned = False  # ohne Worker beendet (Provider reagierte nicht auf den Abbruch)
        self._chat_gone = False
        self._store_failed = False
        self._watchdog: threading.Timer | None = None
        self._thread: threading.Thread | None = None

    @property
    def finished(self) -> bool:
        """``True``, sobald ``done`` veröffentlicht wurde."""
        with self._cond:
            return self._finished

    def publish(self, event: Event) -> int:
        """Hängt ein Event an und gibt seine ``seq`` zurück (ab 1).

        Nach ``done`` werden keine Events mehr angenommen (Rückgabe ``0``),
        z. B. verspätete Tool-Ergebnisse eines abgebrochenen Turns.
        """
        with self._cond:
            if self._finished:
                log.debug("Event %r nach Turn-Ende verworfen (%s)", event.get("type"), self.id)
                return 0
            self._events.append(dict(event))
            seq = len(self._events)
            if event.get("type") == "done":
                self._finished = True
                self.status = event.get("status")
                self.finished_at = time.monotonic()
            self._cond.notify_all()
            return seq

    def events_after(self, seq: int, timeout: float | None) -> tuple[list[tuple[int, Event]], bool]:
        """Events mit einer ``seq`` größer als ``seq`` und ob der Turn beendet ist.

        Gibt es noch keine neueren Events, wird bis zu ``timeout`` Sekunden
        gewartet (``<= 0`` oder ``None``: nicht warten).
        """
        start = max(0, int(seq or 0))
        wait = float(timeout or 0)
        deadline = time.monotonic() + wait
        with self._cond:
            while len(self._events) <= start and not self._finished:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._cond.wait(remaining)
            items = [(i + 1, ev) for i, ev in enumerate(self._events[start:], start=start)]
            return items, self._finished

    def __repr__(self) -> str:
        state = f"finished={self.status}" if self.finished else "running"
        return f"<Turn {self.id} chat={self.chat_id} {state}>"


# --- interne Datentypen ------------------------------------------------------------

@dataclass(eq=False)
class _ProviderSlot:
    """Gecachte Provider-Instanz samt Konfigurations-Schnappschuss und Nutzungszähler."""

    provider: Provider
    cfg: dict[str, Any]
    users: int = 0
    retired: bool = False
    closed: bool = False


@dataclass(eq=False)
class _TurnContext:
    slot: _ProviderSlot
    provider_id: str
    model: str | None
    system_prompt: str
    state: dict[str, Any]
    user: Message
    exclude_tools: frozenset[str] = frozenset()
    reasoning: str | None = None


@dataclass(eq=False)
class _McpEntry:
    """Ein konfigurierter MCP-Server und sein Zustand."""

    name: str
    transport: str  # "stdio" | "http"
    params: dict[str, Any]  # Argumente für McpToolSource (ohne name); kann env und call_timeout enthalten
    timeout: float  # Zeitlimit für den Start
    status: str = "starting"  # starting | connected | error | disabled | stopped
    error: str | None = None
    source: McpToolSource | None = None
    tool_names: list[str] = field(default_factory=list)
    thread: threading.Thread | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)
    start_done: bool = False
    abort_reason: str | None = None  # "closing" (Start beim Beenden abgebrochen)

    def public(self) -> dict[str, Any]:
        """Anzeige für ``list_tools`` (ohne ``env``, ``args`` und URL)."""
        out: dict[str, Any] = {
            "name": self.name,
            "transport": self.transport,
            "status": self.status,
            "tools": list(self.tool_names),
        }
        if self.error:
            out["error"] = self.error
        return out


# --- Hilfsfunktionen -----------------------------------------------------------------

def _close_quietly(obj: Any, what: str) -> None:
    try:
        obj.close()
    except Exception:
        log.warning("%s %r ließ sich nicht sauber schließen", what, obj, exc_info=True)


def _leaf_exception(exc: BaseException) -> BaseException:
    """Erste "echte" Ausnahme aus (verschachtelten) ExceptionGroups."""
    while isinstance(exc, BaseExceptionGroup) and exc.exceptions:
        exc = exc.exceptions[0]
    return exc


def _exc_text(exc: BaseException) -> str:
    exc = _leaf_exception(exc)
    text = str(exc).strip()
    return f"{type(exc).__name__}: {text}" if text else type(exc).__name__


def _model_dict(m: Any) -> dict[str, Any]:
    out: dict[str, Any] = {"id": str(m.id), "label": str(getattr(m, "label", None) or m.id)}
    if getattr(m, "loaded", None) is not None:
        out["loaded"] = bool(m.loaded)
    if getattr(m, "description", None):
        out["description"] = str(m.description)
    return out


def _cache_key(cfg: Mapping[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in cfg.items() if k not in _CACHE_IGNORED_FIELDS}


def _short_secret(cfg: Mapping[str, Any]) -> bool:
    """``True``, wenn der Provider einen Key unter ``MIN_SECRET_LEN`` Zeichen nutzt (Feld oder ``api_key_env``)."""
    key = cfg.get("api_key")
    if isinstance(key, str) and 0 < len(key) < MIN_SECRET_LEN:
        return True
    env_name = cfg.get("api_key_env")
    if isinstance(env_name, str) and env_name:
        value = os.environ.get(env_name)
        if value and len(value) < MIN_SECRET_LEN:
            return True
    return False


# --- Validierung -------------------------------------------------------------------

def _v_str(key: str, value: Any, *, max_len: int = 500, allow_empty: bool = True) -> str | None:
    """String (getrimmt) oder ``None``; leerer String wird zu ``None``."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"Feld '{key}' muss ein String oder null sein.")
    value = value.strip()
    if len(value) > max_len:
        raise ValueError(f"Feld '{key}' ist zu lang (höchstens {max_len} Zeichen).")
    if not value:
        if not allow_empty:
            raise ValueError(f"Feld '{key}' darf nicht leer sein.")
        return None
    return value


def _v_bool(key: str, value: Any) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"Feld '{key}' muss true oder false sein.")
    return value


def _v_name(value: Any) -> str:
    name = _v_str("name", value, max_len=MAX_NAME_CHARS, allow_empty=False)
    assert name is not None
    return name


def _v_base_url(value: Any) -> str:
    url = _v_str("base_url", value, max_len=2000, allow_empty=False)
    assert url is not None
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.netloc or any(c.isspace() for c in url):
        raise ValueError("Feld 'base_url' muss eine http(s)-URL sein, z. B. http://localhost:11434/v1.")
    return url.rstrip("/")


def _v_api_key(value: Any) -> str | None:
    """``None``/``""`` = löschen, sonst der Key (Wert wird nie in Meldungen übernommen).

    Kurze Keys sind erlaubt (Ollama braucht einen Platzhalter wie ``"ollama"``);
    unter ``MIN_SECRET_LEN`` Zeichen werden sie aber nirgends geschwärzt.
    """
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("Feld 'api_key' muss ein String oder null sein.")
    value = value.strip()
    if len(value) > 4000 or any(c.isspace() or ord(c) < 32 for c in value):
        raise ValueError("Feld 'api_key' enthält ungültige Zeichen oder ist zu lang.")
    return value or None


def _v_env_name(value: Any) -> str | None:
    name = _v_str("api_key_env", value, max_len=128)
    if name is not None and not _ENV_NAME_RE.fullmatch(name):
        raise ValueError("Feld 'api_key_env' muss der Name einer Umgebungsvariablen sein (z. B. OPENAI_API_KEY).")
    return name


def _v_models(value: Any) -> list[Any]:
    if value is None:
        return []
    if not isinstance(value, list) or len(value) > 1000:
        raise ValueError("Feld 'models' muss eine Liste (höchstens 1000 Einträge) sein.")
    result: list[Any] = []
    for item in value:
        if isinstance(item, str) and item.strip():
            result.append(item.strip())
        elif isinstance(item, Mapping) and isinstance(item.get("id"), str) and item["id"].strip():
            entry: dict[str, Any] = {"id": item["id"].strip()}
            for key in ("label", "description"):
                if isinstance(item.get(key), str) and item[key].strip():
                    entry[key] = item[key].strip()
            result.append(entry)
        else:
            raise ValueError("Einträge in 'models' müssen Modellnamen oder Objekte mit 'id' sein.")
    return result


def _v_timeout(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 1 <= value <= 3600:
        raise ValueError("Feld 'timeout' muss eine Zahl zwischen 1 und 3600 (Sekunden) sein.")
    return value


def _v_auth_mode(value: Any) -> str:
    if value not in AUTH_MODES:
        raise ValueError(f"Feld 'auth_mode' muss {' oder '.join(repr(m) for m in AUTH_MODES)} sein.")
    return value


def _v_api_host(value: Any) -> str | None:
    host = _v_str("api_host", value, max_len=200)
    if host is not None and any(c.isspace() for c in host):
        raise ValueError("Feld 'api_host' darf keine Leerzeichen enthalten (z. B. 127.0.0.1:1234).")
    return host


_FIELD_VALIDATORS: dict[str, Callable[[Any], Any]] = {
    "name": _v_name,
    "enabled": lambda v: _v_bool("enabled", v),
    "base_url": _v_base_url,
    "api_key": _v_api_key,
    "api_key_env": _v_env_name,
    "default_model": lambda v: _v_str("default_model", v),
    "models": _v_models,
    "supports_tools": lambda v: _v_bool("supports_tools", v),
    "timeout": _v_timeout,
    "auth_mode": _v_auth_mode,
    "cli_path": lambda v: _v_str("cli_path", v, max_len=2000),
    "api_host": _v_api_host,
}


def _validate_provider_fields(ptype: str, data: Mapping[str, Any]) -> dict[str, Any]:
    """Prüft die änderbaren Felder eines Providers und gibt sie normalisiert zurück."""
    allowed = _PROVIDER_FIELDS.get(ptype)
    if allowed is None:
        raise ValueError(f"Provider-Typ {ptype!r} ist unbekannt.")
    unknown = sorted(k for k in data if k not in allowed and k not in _READONLY_PROVIDER_FIELDS)
    if unknown:
        raise ValueError(f"Unbekannte Felder für diesen Provider: {', '.join(unknown)}.")
    return {key: _FIELD_VALIDATORS[key](value) for key, value in data.items() if key in allowed}


def _parse_mcp_entry(index: int, raw: Any) -> _McpEntry:
    """Liest einen Eintrag aus ``config["mcp_servers"]``; Fehler ergeben ``status: "error"``."""
    fallback = f"mcp{index + 1}"
    if not isinstance(raw, Mapping):
        return _McpEntry(fallback, "?", {}, 0.0, status="error", error="Ungültiger Eintrag (kein Objekt).")
    command, url, name = raw.get("command"), raw.get("url"), raw.get("name")
    errors: list[str] = []
    if isinstance(name, str) and name.strip():
        display = name.strip()[:MAX_NAME_CHARS]
    else:
        if name is not None:
            errors.append("'name' muss ein nicht leerer String sein")
        if isinstance(command, str) and command.strip():
            display = Path(command.strip()).stem or fallback
        elif isinstance(url, str) and url.strip():
            display = urlsplit(url.strip()).hostname or fallback
        else:
            display = fallback

    has_cmd = isinstance(command, str) and bool(command.strip())
    has_url = isinstance(url, str) and bool(url.strip())
    params: dict[str, Any] = {}
    transport = "stdio" if has_cmd else "http" if has_url else "?"
    if has_cmd == has_url:
        errors.append("entweder 'command' (stdio) oder 'url' (HTTP) angeben")
    elif has_cmd:
        assert isinstance(command, str)  # folgt aus has_cmd
        params["command"] = command.strip()
        args = raw.get("args")
        if args is None:
            params["args"] = []
        elif isinstance(args, list) and all(isinstance(a, (str, int, float)) and not isinstance(a, bool) for a in args):
            params["args"] = [str(a) for a in args]
        else:
            errors.append("'args' muss eine Liste von Strings sein")
        env = raw.get("env")
        if env is not None:
            if isinstance(env, Mapping) and all(isinstance(k, str) for k in env):
                params["env"] = {k: "" if v is None else str(v) for k, v in env.items()}
            else:
                errors.append("'env' muss ein Objekt mit String-Werten sein")
    else:
        assert isinstance(url, str)  # folgt aus has_url (has_cmd != has_url)
        u = url.strip()
        parts = urlsplit(u)
        if parts.scheme not in ("http", "https") or not parts.netloc:
            errors.append("'url' muss eine http(s)-URL sein")
        params["url"] = u

    timeout_raw = raw.get("startup_timeout", raw.get("timeout"))
    timeout = MCP_START_TIMEOUT_S
    if timeout_raw is not None:
        if isinstance(timeout_raw, bool) or not isinstance(timeout_raw, (int, float)) or not 0 < timeout_raw <= 600:
            errors.append("'startup_timeout' muss eine Zahl zwischen 0 und 600 (Sekunden) sein")
        else:
            timeout = float(timeout_raw)

    call_raw = raw.get("call_timeout")
    if call_raw is not None:  # fehlt/null: Standard von McpToolSource (120 s)
        if (isinstance(call_raw, bool) or not isinstance(call_raw, (int, float))
                or not 0 < call_raw <= MCP_MAX_CALL_TIMEOUT_S):
            errors.append(f"'call_timeout' muss eine Zahl zwischen 0 und {MCP_MAX_CALL_TIMEOUT_S:g} (Sekunden) sein")
        else:
            params["call_timeout"] = float(call_raw)

    entry = _McpEntry(display, transport, params, timeout)
    if errors:
        entry.status = "error"
        entry.error = "Ungültige Konfiguration: " + "; ".join(errors) + "."
    elif raw.get("enabled") is False:
        entry.status = "disabled"
    return entry


def _default_factory(cfg: dict[str, Any], data_dir: Path) -> Provider:
    from providers.registry import build_provider

    return build_provider(cfg, data_dir)


# --- ChatService -------------------------------------------------------------------

class ChatService:
    """Fachlogik hinter der REST-API (siehe Plan, Abschnitt "API-Schema").

    ``provider_factory(cfg, data_dir) -> Provider`` ersetzt bei Bedarf
    ``providers.registry.build_provider`` (z. B. für Tests mit Fake-Providern).
    """

    def __init__(
        self,
        store: ChatStore,
        config: ConfigStore,
        data_dir: str | os.PathLike[str],
        *,
        provider_factory: Callable[[dict[str, Any], Path], Provider] | None = None,
    ):
        self.store = store
        self.config = config
        self.data_dir = Path(data_dir)
        self.agents = AgentStore(self.data_dir / "agents")
        self._provider_factory = provider_factory or _default_factory

        self._lock = threading.RLock()  # Turns, aktive Turns, Zustand des Service
        self._cfg_lock = threading.RLock()  # Konfiguration lesen/ändern/speichern
        self._prov_lock = threading.RLock()  # Provider-Cache
        self._mcp_lock = threading.RLock()  # MCP-Server und gemeinsames ToolSet
        self._bg_lock = threading.Lock()

        self._turns: dict[str, Turn] = {}
        self._active: dict[str, Turn] = {}  # chat_id -> laufender Turn
        self._providers: dict[str, _ProviderSlot] = {}
        self._retired: list[_ProviderSlot] = []
        self._bg: list[threading.Thread] = []
        self._mcp: list[_McpEntry] = []
        self._local_specs: list[ToolSpec] | None = None
        self._shared_tools: ToolSet | None = None
        self._startup_thread: threading.Thread | None = None
        self._started = False
        self._closed = threading.Event()
        self._config_stamp: tuple[int, int] | None = None
        self._config_bad_stamp: tuple[int, int] | None = None

    # ------------------------------------------------------------------ Lebenszyklus

    def start(self) -> None:
        """Startet die MCP-Server aus der Konfiguration (parallel im Hintergrund) und kehrt sofort zurück.

        Bis ein MCP-Server verbunden ist, steht er in ``list_tools`` mit
        ``status: "starting"``; Turns nutzen jeweils die gerade verfügbaren Tools.
        """
        with self._lock:
            if self._closed.is_set():
                raise ServiceClosed("Der ChatService wurde bereits beendet.")
            if self._started:
                return
            self._started = True
        self._sync_config()
        with self._cfg_lock:
            raw = self.config.data.get("mcp_servers")
        raw_list = raw if isinstance(raw, list) else []
        if raw is not None and not isinstance(raw, list):
            log.warning("config.json: 'mcp_servers' ist keine Liste und wird ignoriert")
        entries = [_parse_mcp_entry(i, item) for i, item in enumerate(raw_list)]
        with self._mcp_lock:
            self._mcp = entries
        for entry in entries:
            if entry.status == "error":
                log.warning("MCP-Server %s: %s", entry.name, entry.error)
        self._startup_thread = threading.Thread(target=self._startup, name="chat-service-start", daemon=True)
        self._startup_thread.start()

    def wait_mcp_ready(self, timeout: float | None = None) -> bool:
        """Wartet, bis kein MCP-Server mehr startet (für Tests und Demos). ``True`` = alle fertig."""
        deadline = None if timeout is None else time.monotonic() + timeout
        threads = [self._startup_thread] if self._startup_thread else []
        with self._mcp_lock:
            threads += [e.thread for e in self._mcp if e.thread is not None]
        for th in threads:
            if th is None:
                continue
            th.join(None if deadline is None else max(0.0, deadline - time.monotonic()))
        with self._mcp_lock:
            return not any(e.status == "starting" for e in self._mcp)

    def close(self, timeout: float = 5) -> None:
        """Bricht laufende Turns ab und schließt Provider und MCP-Server (höchstens ``timeout`` Sekunden).

        Alles wird parallel geschlossen; was nach Ablauf der Zeit noch läuft,
        wird im Log gemeldet und läuft als Daemon-Thread weiter. Idempotent.
        """
        deadline = time.monotonic() + max(0.5, float(timeout))
        with self._lock:
            if self._closed.is_set():
                return
            self._closed.set()
            turns = list(self._active.values())
            workers = [t._thread for t in self._turns.values() if t._thread is not None]
        for turn in turns:
            turn.cancel.set()
            if turn._watchdog is not None:
                turn._watchdog.cancel()

        closers: list[threading.Thread] = []

        def spawn(target: Callable[..., Any], *args: Any, name: str) -> None:
            th = threading.Thread(target=target, args=args, name=name, daemon=True)
            th.start()
            closers.append(th)

        # MCP-Server: startende abbrechen, verbundene schließen
        with self._mcp_lock:
            entries = list(self._mcp)
        mcp_threads: list[threading.Thread] = []
        for entry in entries:
            if entry.thread is not None and entry.thread.is_alive():
                mcp_threads.append(entry.thread)
                spawn(self._mcp_abort, entry, "closing", name=f"mcp-abort-{entry.name}")
            elif entry.status == "connected" and entry.source is not None:
                source = entry.source
                spawn(self._close_mcp_source, entry, source, name=f"mcp-close-{entry.name}")

        # Provider (auch von laufenden Turns: Claude bricht dabei selbst ab)
        with self._prov_lock:
            slots = list(self._providers.values()) + list(self._retired)
            self._providers.clear()
            self._retired.clear()
            for slot in slots:
                slot.retired = True
            to_close = [slot for slot in slots if not slot.closed]
            for slot in to_close:
                slot.closed = True
        for slot in to_close:
            spawn(_close_quietly, slot.provider, "Provider", name=f"provider-close-{slot.provider.id}")

        with self._bg_lock:
            background = list(self._bg)
        waits = [th for th in workers + mcp_threads + closers + background if th is not None]
        if self._startup_thread is not None:
            waits.append(self._startup_thread)
        for th in waits:
            th.join(max(0.0, deadline - time.monotonic()))

        # Turns, deren Worker nicht rechtzeitig fertig wurde, trotzdem abschließen (SSE-Streams enden).
        for turn in turns:
            if not turn.finished:
                self._force_finish(turn)
        still = [th.name for th in waits if th.is_alive()]
        if still:
            log.warning("ChatService.close: nach %.1f s laufen noch: %s", timeout, ", ".join(still))

    # ------------------------------------------------------------------ Konfiguration

    def _stat_config(self) -> tuple[int, int] | None:
        try:
            st = os.stat(self.config.path)
        except OSError:
            return None
        return st.st_mtime_ns, st.st_size

    def _sync_config(self, strict: bool = False) -> None:
        """Lädt ``config.json`` neu, wenn sie außerhalb des Service geändert wurde.

        So überschreibt ein späteres Speichern keine Änderungen, die von Hand
        gemacht wurden (z. B. an ``mcp_servers``). Ist die Datei ungültig, wird
        sie nie überschrieben: ``strict=True`` (vor Änderungen) wirft dann
        ``ValueError``, sonst gilt weiter der zuletzt geladene Stand.
        """
        with self._cfg_lock:
            stamp = self._stat_config()
            if stamp is None or stamp == self._config_stamp:
                return
            try:
                self.config.load()
            except (ValueError, OSError) as e:
                if strict:
                    raise ValueError(
                        f"Die Konfigurationsdatei ist ungültig und wird nicht überschrieben: {e}"
                    ) from None
                if self._config_bad_stamp != stamp:
                    self._config_bad_stamp = stamp
                    log.warning("Konfiguration nicht neu geladen: %s", e)
                return
            self._config_stamp = stamp

    def _save_config(self) -> None:
        with self._cfg_lock:
            self.config.save()
            self._config_stamp = self._stat_config()

    def _providers_cfg(self) -> list[dict[str, Any]]:
        data = self.config.data
        providers = data.get("providers")
        if not isinstance(providers, list):
            providers = []
            data["providers"] = providers
        return [p for p in providers if isinstance(p, dict) and isinstance(p.get("id"), str)]

    def _provider_cfg(self, pid: str) -> dict[str, Any] | None:
        for p in self._providers_cfg():
            if p["id"] == pid:
                return p
        return None

    def _settings(self) -> dict[str, Any]:
        settings = self.config.data.get("settings")
        if not isinstance(settings, dict):
            settings = {}
            self.config.data["settings"] = settings
        return settings

    def _default_provider_id(self) -> str | None:
        """Standard-Provider aus den Einstellungen, sonst der erste aktivierte."""
        providers = self._providers_cfg()
        wanted = self._settings().get("default_provider")
        for p in providers:
            if p["id"] == wanted and p.get("enabled", True) is not False:
                return p["id"]
        for p in providers:
            if p.get("enabled", True) is not False:
                return p["id"]
        return providers[0]["id"] if providers else None

    def role_defaults(self, role: str, with_agent: bool = False) -> tuple[str | None, str | None]:
        """Provider und Modell für ``role`` (``subagent`` oder ``assistant``) laut Einstellungen.

        Mit ``with_agent`` (ein vorgefertigter Agent wird verwendet) gilt zuerst die Rolle ``agent``.
        Ist keine Rolle (mit aktiviertem Provider) eingestellt, kommt der Standard-Provider ohne Modell zurück.
        """
        if role not in AGENT_ROLES:
            raise ValueError(f"Unbekannte Rolle: {role!r}")
        self._sync_config()
        with self._cfg_lock:
            settings = self._settings()
            for name in (("agent", role) if with_agent else (role,)):
                cfg = self._provider_cfg(settings.get(f"{name}_provider") or "")
                if cfg is not None and cfg.get("enabled", True) is not False:
                    return cfg["id"], settings.get(f"{name}_model") or None
            return self._default_provider_id(), None

    def _disabled_tools(self) -> set[str]:
        with self._cfg_lock:
            tools_cfg = self.config.data.get("tools")
            disabled = tools_cfg.get("disabled") if isinstance(tools_cfg, Mapping) else None
            return {n for n in disabled if isinstance(n, str)} if isinstance(disabled, list) else set()

    @staticmethod
    def _public_provider(cfg: Mapping[str, Any]) -> dict[str, Any]:
        out = {k: copy.deepcopy(v) for k, v in cfg.items() if k != "api_key"}
        out["has_api_key"] = bool(cfg.get("api_key"))
        out["builtin"] = cfg.get("id") in BUILTIN_PROVIDER_IDS
        out.setdefault("name", cfg.get("id"))
        out.setdefault("enabled", True)
        out.setdefault("default_model", None)
        return out

    # ------------------------------------------------------------------ Geheimnisse

    def _secrets(self) -> list[str]:
        """Alle bekannten Geheimnisse ab ``MIN_SECRET_LEN`` Zeichen, längste zuerst.

        API-Keys und Werte aus ``api_key_env`` der aktuellen Konfiguration und
        der gecachten (auch ausgemusterten) Provider-Instanzen sowie die
        ``env``-Werte der MCP-Server. Kürzere Werte werden nie geschwärzt.
        """
        found: set[str] = set()

        def add(value: Any) -> None:
            if isinstance(value, str) and len(value) >= MIN_SECRET_LEN:
                found.add(value)

        def add_provider(cfg: Mapping[str, Any]) -> None:
            add(cfg.get("api_key"))
            env_name = cfg.get("api_key_env")
            if isinstance(env_name, str) and env_name:
                add(os.environ.get(env_name))

        with self._cfg_lock:
            for p in self._providers_cfg():
                add_provider(p)
            servers = self.config.data.get("mcp_servers")
            for server in servers if isinstance(servers, list) else []:
                env = server.get("env") if isinstance(server, Mapping) else None
                if isinstance(env, Mapping):
                    for value in env.values():
                        add(value)
        with self._prov_lock:
            cfgs = [s.cfg for s in self._providers.values()] + [s.cfg for s in self._retired]
        for cfg in cfgs:
            add_provider(cfg)
        with self._mcp_lock:
            for entry in self._mcp:
                for value in (entry.params.get("env") or {}).values():
                    add(value)
        return sorted(found, key=len, reverse=True)

    def redact(self, text: str) -> str:
        """Ersetzt alle bekannten Geheimnisse (ab ``MIN_SECRET_LEN`` Zeichen) in ``text`` durch ``***``.

        Für Fehler-, Status- und Hinweistexte, nicht für Chat-Inhalte. Neben
        ``_secrets()`` schwärzen auch die Provider selbst (``provider.redact``/
        ``provider.scrub``, z. B. Muster für Anthropic-Keys) - aber nur Provider,
        deren Keys nicht zu kurz sind: deren Funktionen kennen keine
        Mindestlänge und würden sonst ``"ollama"`` & Co. überall ersetzen.
        """
        if not text or not isinstance(text, str):
            return text
        for secret in self._secrets():
            text = text.replace(secret, "***")
        with self._prov_lock:
            slots = list(self._providers.values()) + list(self._retired)
        for slot in slots:
            if _short_secret(slot.cfg):
                continue
            fn = getattr(slot.provider, "redact", None) or getattr(slot.provider, "scrub", None)
            if callable(fn):
                try:
                    text = str(fn(text))
                except Exception:
                    log.debug("Schwärzen durch %r fehlgeschlagen", slot.provider, exc_info=True)
        return text

    def _redact_event(self, ev: Event) -> Event:
        out = dict(ev)
        for key in ("message", "hint"):
            if isinstance(out.get(key), str):
                out[key] = self.redact(out[key])
        return out

    # ------------------------------------------------------------------ Provider-Cache

    def _acquire_provider(self, pid: str, cfg: Mapping[str, Any]) -> _ProviderSlot:
        """Provider-Instanz für die aktuelle Konfiguration (gecacht; bei Änderung neu gebaut)."""
        key = _cache_key(cfg)
        failure: Exception | None = None
        with self._prov_lock:
            if self._closed.is_set():
                raise ServiceClosed("Der ChatService wurde bereits beendet.")
            slot = self._providers.get(pid)
            if slot is not None and _cache_key(slot.cfg) != key:
                self._retire_slot(pid)
                slot = None
            if slot is None:
                try:
                    provider = self._provider_factory(copy.deepcopy(dict(cfg)), self.data_dir)
                except (ValueError, ProviderError):
                    raise
                except Exception as e:
                    log.exception("Provider %s konnte nicht erzeugt werden", pid)
                    failure = e
                else:
                    slot = _ProviderSlot(provider, copy.deepcopy(dict(cfg)))
                    self._providers[pid] = slot
            if slot is not None:
                slot.users += 1
                return slot
        # außerhalb des Locks schwärzen (redact nimmt selbst Locks)
        assert failure is not None
        raise ValueError(self.redact(f"Provider {pid} kann nicht erzeugt werden: {_exc_text(failure)}"))

    def _release_provider(self, slot: _ProviderSlot) -> None:
        with self._prov_lock:
            slot.users -= 1
            close_now = slot.retired and slot.users <= 0 and not slot.closed
            if close_now:
                slot.closed = True
                if slot in self._retired:
                    self._retired.remove(slot)
        if close_now:
            self._close_in_background(slot.provider)

    def _retire_slot(self, pid: str) -> None:
        """Nimmt die gecachte Instanz aus dem Cache; geschlossen wird sie, sobald sie niemand mehr nutzt."""
        with self._prov_lock:
            slot = self._providers.pop(pid, None)
            if slot is None:
                return
            slot.retired = True
            if slot.users > 0:
                self._retired.append(slot)
                return
            slot.closed = True
        self._close_in_background(slot.provider)

    def _close_in_background(self, provider: Provider) -> None:
        th = threading.Thread(
            target=_close_quietly, args=(provider, "Provider"), name=f"provider-close-{provider.id}", daemon=True
        )
        with self._bg_lock:
            self._bg = [t for t in self._bg if t.is_alive()]
            self._bg.append(th)
        th.start()

    @contextmanager
    def _lease(self, pid: str) -> Generator[Provider, None, None]:
        """Provider für eine kurze Abfrage (Modelle, Status) leihen."""
        self._sync_config()
        with self._cfg_lock:
            cfg = self._provider_cfg(pid)
            if cfg is None:
                raise NotFound(f"Provider {pid!r} nicht gefunden.")
            cfg = copy.deepcopy(cfg)
        slot = self._acquire_provider(pid, cfg)
        try:
            yield slot.provider
        finally:
            self._release_provider(slot)

    # ------------------------------------------------------------------ Provider-API

    def list_providers(self) -> list[dict[str, Any]]:
        """Alle Provider (maskiert: ``has_api_key`` statt ``api_key``, plus ``builtin``)."""
        self._sync_config()
        with self._cfg_lock:
            return [self._public_provider(p) for p in self._providers_cfg()]

    def add_provider(self, data: Mapping[str, Any]) -> dict[str, Any]:
        """Legt einen OpenAI-kompatiblen Provider an (ID ``oai-<6 hex>``) und gibt ihn maskiert zurück."""
        if not isinstance(data, Mapping):
            raise ValueError("Provider-Daten müssen ein Objekt sein.")
        ptype = data.get("type", "openai")
        if ptype not in ADDABLE_PROVIDER_TYPES:
            raise ValueError("Es können nur OpenAI-kompatible Provider (type 'openai') hinzugefügt werden.")
        fields = _validate_provider_fields("openai", {k: v for k, v in data.items() if k not in ("type", "id")})
        if not fields.get("base_url"):
            raise ValueError("Feld 'base_url' fehlt (z. B. http://localhost:11434/v1).")
        with self._cfg_lock:
            self._sync_config(strict=True)
            if self._closed.is_set():
                raise ServiceClosed("Der ChatService wurde bereits beendet.")
            existing = {p["id"] for p in self._providers_cfg()}
            pid = "oai-" + secrets.token_hex(3)
            while pid in existing:
                pid = "oai-" + secrets.token_hex(3)
            cfg: dict[str, Any] = {
                "id": pid,
                "type": "openai",
                "name": fields.get("name") or urlsplit(fields["base_url"]).hostname or "OpenAI-kompatibel",
                "enabled": True,
                "base_url": fields["base_url"],
                "api_key": None,
                "api_key_env": None,
                "default_model": None,
                "models": [],
                "supports_tools": True,
            }
            cfg.update({k: v for k, v in fields.items() if k != "name"})
            providers = self.config.data.get("providers")
            if not isinstance(providers, list):
                providers = []
                self.config.data["providers"] = providers
            providers.append(cfg)
            try:
                self._save_config()
            except BaseException:
                providers.remove(cfg)
                raise
            return self._public_provider(cfg)

    def update_provider(self, pid: str, data: Mapping[str, Any]) -> dict[str, Any]:
        """Teilupdate eines Providers. ``api_key``: fehlt = behalten, ``""``/``null`` = löschen, Wert = setzen."""
        if not isinstance(data, Mapping):
            raise ValueError("Provider-Daten müssen ein Objekt sein.")
        with self._cfg_lock:
            self._sync_config(strict=True)
            cfg = self._provider_cfg(pid)
            if cfg is None:
                raise NotFound(f"Provider {pid!r} nicht gefunden.")
            if "id" in data and data["id"] != pid:
                raise ValueError("Die ID eines Providers kann nicht geändert werden.")
            if "type" in data and data["type"] != cfg.get("type"):
                raise ValueError("Der Typ eines Providers kann nicht geändert werden.")
            fields = _validate_provider_fields(str(cfg.get("type")), data)
            if not fields:
                return self._public_provider(cfg)
            old = copy.deepcopy(cfg)
            cfg.update(fields)
            try:
                self._save_config()
            except BaseException:
                cfg.clear()
                cfg.update(old)
                raise
            result = self._public_provider(cfg)
        self._retire_slot(pid)
        return result

    def delete_provider(self, pid: str) -> None:
        """Löscht einen hinzugefügten Provider. Eingebaute (LM Studio, Claude) ergeben ``ValueError``."""
        with self._cfg_lock:
            self._sync_config(strict=True)
            cfg = self._provider_cfg(pid)
            if cfg is None:
                raise NotFound(f"Provider {pid!r} nicht gefunden.")
            if pid in BUILTIN_PROVIDER_IDS:
                raise ValueError("Eingebaute Provider können nicht gelöscht werden (aber deaktiviert).")
            providers = self.config.data["providers"]
            index = providers.index(cfg)
            providers.pop(index)
            settings = self._settings()
            old_settings = dict(settings)
            if settings.get("default_provider") == pid:
                settings["default_provider"] = self._default_provider_id()
                settings["default_model"] = None
            for role in AGENT_ROLES:
                if settings.get(f"{role}_provider") == pid:
                    settings[f"{role}_provider"] = None
                    settings[f"{role}_model"] = None
            try:
                self._save_config()
            except BaseException:
                providers.insert(index, cfg)
                settings.clear()
                settings.update(old_settings)
                raise
        self._retire_slot(pid)

    def provider_models(self, pid: str) -> dict[str, Any]:
        """``{"models": [{id, label, loaded?, description?}], "error"?}``; Abruffehler stehen in ``error``."""
        try:
            with self._lease(pid) as provider:
                models = provider.list_models()
                error = getattr(provider, "last_models_error", None)
        except (NotFound, ServiceClosed):
            raise
        except ProviderError as e:
            return {"models": [], "error": self.redact(e.message)}
        except ValueError as e:
            return {"models": [], "error": self.redact(str(e))}
        except Exception as e:
            log.exception("Modellliste von %s fehlgeschlagen", pid)
            return {"models": [], "error": self.redact(f"Modellliste nicht abrufbar: {_exc_text(e)}")}
        out: dict[str, Any] = {"models": [_model_dict(m) for m in models or []]}
        if error:
            out["error"] = self.redact(str(error))
        return out

    def provider_status(self, pid: str) -> dict[str, Any]:
        """``{"ok", "message", "hint"?, "details"}`` des Providers."""
        try:
            with self._lease(pid) as provider:
                st = provider.status()
        except (NotFound, ServiceClosed):
            raise
        except ProviderError as e:
            out: dict[str, Any] = {"ok": False, "message": self.redact(e.message), "details": {"code": e.code}}
            if e.hint:
                out["hint"] = self.redact(e.hint)
            return out
        except ValueError as e:
            return {"ok": False, "message": self.redact(str(e)), "details": {}}
        except Exception as e:
            log.exception("Status von %s fehlgeschlagen", pid)
            return {"ok": False, "message": self.redact(f"Status nicht abrufbar: {_exc_text(e)}"), "details": {}}
        out = {
            "ok": bool(st.ok),
            "message": self.redact(st.message or ""),
            "details": copy.deepcopy(dict(st.details or {})),
        }
        if st.hint:
            out["hint"] = self.redact(st.hint)
        return out

    # ------------------------------------------------------------------ Einstellungen

    def get_settings(self) -> dict[str, Any]:
        """Alle Einstellungen (``theme``, ``system_prompt``, ``default_provider``, ``default_model``, …)."""
        self._sync_config()
        with self._cfg_lock:
            return copy.deepcopy(self._settings())

    def update_settings(self, data: Mapping[str, Any]) -> dict[str, Any]:
        """Teilupdate der Einstellungen; unbekannte Felder werden ignoriert. Gibt alle Einstellungen zurück."""
        if not isinstance(data, Mapping):
            raise ValueError("Einstellungen müssen ein Objekt sein.")
        changes: dict[str, Any] = {}
        if "theme" in data:
            if data["theme"] not in THEMES:
                raise ValueError(f"Feld 'theme' muss einer der Werte {', '.join(THEMES)} sein.")
            changes["theme"] = data["theme"]
        if "system_prompt" in data:
            prompt = data["system_prompt"]
            if prompt is None:
                prompt = ""
            if not isinstance(prompt, str):
                raise ValueError("Feld 'system_prompt' muss ein String sein.")
            if len(prompt) > MAX_SYSTEM_PROMPT_CHARS:
                raise ValueError(f"Der System-Prompt ist zu lang (höchstens {MAX_SYSTEM_PROMPT_CHARS} Zeichen).")
            changes["system_prompt"] = prompt
        if "default_model" in data:
            changes["default_model"] = _v_str("default_model", data["default_model"])
        for role in AGENT_ROLES:
            if f"{role}_model" in data:
                changes[f"{role}_model"] = _v_str(f"{role}_model", data[f"{role}_model"])
        with self._cfg_lock:
            self._sync_config(strict=True)
            if "default_provider" in data:
                pid = _v_str("default_provider", data["default_provider"], max_len=200)
                if pid is not None and self._provider_cfg(pid) is None:
                    raise ValueError(f"Provider {pid!r} existiert nicht.")
                changes["default_provider"] = pid
            settings = self._settings()
            for role in AGENT_ROLES:
                key = f"{role}_provider"
                if key in data:
                    pid = _v_str(key, data[key], max_len=200)
                    if pid is not None and self._provider_cfg(pid) is None:
                        raise ValueError(f"Provider {pid!r} existiert nicht.")
                    changes[key] = pid
                if changes.get(key, settings.get(key)) is None:
                    changes[f"{role}_model"] = None  # ein Modell gehört immer zu einem Provider
            if changes:
                old = dict(settings)
                settings.update(changes)
                try:
                    self._save_config()
                except BaseException:
                    settings.clear()
                    settings.update(old)
                    raise
            return copy.deepcopy(settings)

    # ------------------------------------------------------------------ Tools und MCP

    def _get_local_specs(self) -> list[ToolSpec]:
        with self._mcp_lock:
            if self._local_specs is None:
                try:
                    from tools import bind_service, get_tools
                    from toolset import ToolSet

                    bind_service(self)
                    self._local_specs = list(ToolSet.from_any(get_tools()))
                except Exception:
                    log.exception("Lokale Tools konnten nicht geladen werden")
                    self._local_specs = []
            return self._local_specs

    def _rebuild_tools(self) -> ToolSet:
        """Gemeinsames ToolSet neu aufbauen: lokale Tools, dann MCP-Tools in Konfigurationsreihenfolge."""
        from toolset import ToolSet

        local = self._get_local_specs()
        failed: list[tuple[_McpEntry, str]] = []
        with self._mcp_lock:
            tools = ToolSet(local)
            for entry in self._mcp:
                if entry.status != "connected" or entry.source is None:
                    entry.tool_names = []
                    continue
                try:
                    entry.tool_names = [tools.add(spec).name for spec in entry.source.specs(reserved=tools.names())]
                except Exception as e:  # noqa: BLE001 - ein Server darf die übrigen Tools nicht verhindern
                    log.warning("Tools von MCP-Server %s nicht übernehmbar: %s", entry.name, _exc_text(e))
                    entry.status = "error"
                    entry.error = "Tools nicht übernehmbar."
                    entry.tool_names = []
                    failed.append((entry, f"Tools nicht übernehmbar: {_exc_text(e)}"))
            self._shared_tools = tools
        for entry, text in failed:  # außerhalb des Locks schwärzen (redact nimmt selbst Locks)
            entry.error = self.redact(text)
        return tools

    def _turn_tools(self) -> ToolSet:
        """Tools für einen Turn: aktuelles gemeinsames ToolSet ohne ``tools.disabled`` (neues Objekt)."""
        from toolset import ToolSet

        with self._mcp_lock:
            shared = self._shared_tools
        if shared is None:
            shared = self._rebuild_tools()
        disabled = self._disabled_tools()
        result = ToolSet(spec for spec in shared if spec.name not in disabled)
        result.approval_hook = shared.approval_hook
        return result

    def list_tools(self) -> dict[str, Any]:
        """``{"tools": [{name, description, source, parameters}], "mcp_servers": [{name, status, error?, tools}]}``."""
        self._sync_config()
        tools = self._turn_tools()
        with self._mcp_lock:
            servers = [entry.public() for entry in self._mcp]
        return {"tools": tools.describe(), "mcp_servers": servers}

    def _startup(self) -> None:
        self._get_local_specs()
        with self._mcp_lock:
            entries = [e for e in self._mcp if e.status == "starting"]
            for entry in entries:
                entry.thread = threading.Thread(
                    target=self._mcp_start_one, args=(entry,), name=f"mcp-start-{entry.name}", daemon=True
                )
        for entry in entries:
            assert entry.thread is not None
            entry.thread.start()
        self._rebuild_tools()

    def _mcp_start_one(self, entry: _McpEntry) -> None:
        """Startet einen MCP-Server (läuft in eigenem Thread, mit Zeitlimit ``entry.timeout``)."""
        try:
            from mcp_tools import McpStartAborted, McpStartTimeout, McpToolSource

            source = McpToolSource(name=entry.name, **entry.params)
        except Exception as e:  # noqa: BLE001 - Fehler wird als Status angezeigt
            self._mcp_failed(entry, self._mcp_error_text(entry, e))
            return
        with entry.lock:
            entry.source = source  # ab hier bricht _mcp_abort über source.abort() ab
            aborted = entry.abort_reason is not None
        if aborted or self._closed.is_set():
            source.dispose()
            self._mcp_failed(entry, "Beim Beenden abgebrochen.", status="stopped")
            return

        error: BaseException | None = None
        try:
            source.start(timeout=entry.timeout)  # räumt bei Fehler/Zeitüberschreitung/Abbruch selbst auf
        except BaseException as e:  # noqa: BLE001 - jeder Fehler wird als Status angezeigt
            error = e
        with entry.lock:
            entry.start_done = True
            reason = entry.abort_reason
        if error is not None or reason is not None:
            source.dispose()  # idempotent
            if reason == "closing" or isinstance(error, McpStartAborted):
                self._mcp_failed(entry, "Beim Beenden abgebrochen.", status="stopped")
            elif isinstance(error, McpStartTimeout):
                self._mcp_failed(
                    entry, f"Zeitüberschreitung: Der Server hat sich nicht innerhalb von {entry.timeout:g} s gemeldet."
                )
            else:
                assert error is not None
                self._mcp_failed(entry, self._mcp_error_text(entry, error))
            return
        if self._closed.is_set():
            self._close_mcp_source(entry, source)
            return
        with self._mcp_lock:
            entry.status = "connected"
            entry.error = None
        log.info("MCP-Server %s verbunden (%d Tools)", entry.name, len(source.tools))
        self._rebuild_tools()

    def _mcp_error_text(self, entry: _McpEntry, exc: BaseException) -> str:
        leaf = _leaf_exception(exc)
        if isinstance(leaf, FileNotFoundError) and entry.transport == "stdio":
            text = f"Befehl {entry.params.get('command')!r} nicht gefunden ({_exc_text(leaf)})."
        else:
            text = f"Start fehlgeschlagen: {_exc_text(leaf)}"
        return self.redact(text)

    def _mcp_failed(self, entry: _McpEntry, message: str, status: str = "error") -> None:
        with self._mcp_lock:
            entry.status = status
            entry.error = message
            entry.source = None
            entry.tool_names = []
        if status == "error":
            log.warning("MCP-Server %s: %s", entry.name, message)

    def _mcp_abort(self, entry: _McpEntry, reason: str) -> None:
        """Bricht den Start eines MCP-Servers ab (``reason``: ``"closing"``).

        Gibt es die ``McpToolSource`` noch nicht, sieht ``_mcp_start_one`` den
        Grund vor dem Start; sonst wirft ihr ``start()`` ``McpStartAborted``.
        """
        with entry.lock:
            if entry.start_done or entry.abort_reason is not None:
                return
            entry.abort_reason = reason
            source = entry.source
        if source is not None:
            source.abort()

    def _close_mcp_source(self, entry: _McpEntry, source: McpToolSource) -> None:
        with self._mcp_lock:
            entry.status = "stopped"
            entry.tool_names = []
        try:
            source.close()  # räumt bei Problemen selbst hart ab (dispose)
        except Exception:
            log.warning("MCP-Server %s ließ sich nicht sauber schließen", entry.name, exc_info=True)
            source.dispose()

    # ------------------------------------------------------------------ Chats

    @staticmethod
    def _chat_out(chat: Mapping[str, Any]) -> dict[str, Any]:
        return {k: copy.deepcopy(v) for k, v in chat.items() if k != "provider_state"}

    def _check_provider_ref(self, provider_id: str | None) -> None:
        if provider_id is None:
            return
        self._sync_config()
        with self._cfg_lock:
            if self._provider_cfg(provider_id) is None:
                raise ValueError(f"Provider {provider_id!r} existiert nicht.")

    def list_chats(self) -> list[dict[str, Any]]:
        """Alle Chats (zuletzt geänderte zuerst), ohne Nachrichten und ohne Provider-Zustand."""
        return [self._chat_out(c) for c in self.store.list_chats()]

    def create_chat(
        self, provider_id: str | None = None, model: str | None = None, title: str | None = None
    ) -> dict[str, Any]:
        """Legt einen leeren Chat an (ohne Provider: Standard-Provider der Einstellungen)."""
        provider_id = _v_str("provider_id", provider_id, max_len=200)
        model = _v_str("model", model)
        title = _v_str("title", title, max_len=MAX_TITLE_CHARS)
        if provider_id is None:
            self._sync_config()
            with self._cfg_lock:
                provider_id = self._default_provider_id()
        else:
            self._check_provider_ref(provider_id)
        return self._chat_out(self.store.create_chat(title=title, provider_id=provider_id, model=model))

    def get_chat(self, chat_id: str) -> dict[str, Any] | None:
        """Chat mit ``messages`` (Liste von Message-Dicts) und ``active_turn_id`` (``None`` ohne laufenden Turn)."""
        chat = self.store.get_chat(chat_id)
        if chat is None:
            return None
        out = self._chat_out(chat)
        out["messages"] = [m.to_dict() for m in self.store.get_messages(chat_id)]
        with self._lock:
            turn = self._active.get(chat_id)
            out["active_turn_id"] = turn.id if turn is not None and not turn.finished else None
        return out

    def update_chat(self, chat_id: str, data: Mapping[str, Any]) -> dict[str, Any] | None:
        """Ändert ``title``, ``provider_id`` und/oder ``model``. ``None``, wenn es den Chat nicht gibt."""
        if not isinstance(data, Mapping):
            raise ValueError("Chat-Daten müssen ein Objekt sein.")
        unknown = sorted(k for k in data if k not in ("title", "provider_id", "model"))
        if unknown:
            raise ValueError(f"Unbekannte Felder: {', '.join(unknown)}.")
        fields: dict[str, Any] = {}
        if "title" in data:
            fields["title"] = _v_str("title", data["title"], max_len=MAX_TITLE_CHARS)
        if "provider_id" in data:
            fields["provider_id"] = _v_str("provider_id", data["provider_id"], max_len=200)
            self._check_provider_ref(fields["provider_id"])
        if "model" in data:
            fields["model"] = _v_str("model", data["model"])
        if not fields:
            chat = self.store.get_chat(chat_id)
        else:
            chat = self.store.update_chat(chat_id, **fields)
        return self._chat_out(chat) if chat is not None else None

    def delete_chat(self, chat_id: str) -> bool:
        """Löscht den Chat samt Nachrichten (ein laufender Turn wird abgebrochen). ``False``, wenn es ihn nicht gab."""
        with self._lock:
            turn = self._active.get(chat_id)
        if turn is not None and not turn.finished:
            self._request_cancel(turn)
        return self.store.delete_chat(chat_id)

    # ------------------------------------------------------------------ Turns

    def start_turn(
        self,
        chat_id: str,
        content: str,
        provider_id: str | None = None,
        model: str | None = None,
        retry: bool = False,
        user_message_id: str | None = None,
        system_prompt: str | None = None,
        exclude_tools: Iterable[str] | None = None,
        reasoning: str | None = None,
    ) -> tuple[Turn, Message]:
        """Speichert die User-Nachricht und startet den Turn im Hintergrund.

        - Provider: ``provider_id``, sonst der des Chats, sonst der Standard-Provider.
        - Modell: ``model``, sonst das des Chats (bei gleichem Provider), sonst
          der Standard (Einstellungen bzw. ``default_model`` des Providers).
        - Gewählter Provider und Modell werden am Chat gespeichert.
        - ``retry=True``: die User-Nachricht ``user_message_id`` (Pflicht)
          erneut ausführen (``content`` wird ignoriert); alles dahinter wird
          gelöscht. Nur erlaubt, wenn sie die letzte User-Nachricht ist und zu
          ihr keine fertige Antwort vorliegt (siehe Modul-Doku).

        ``system_prompt`` ersetzt für diesen Turn den System-Prompt der Einstellungen;
        ``exclude_tools`` nimmt Tools (nach Name) aus diesem Turn heraus; ``None`` = Standard
        des Haupt-Chats (``agents.MAIN_EXCLUDED_TOOLS``).
        ``reasoning`` ist die Reasoning-Stufe (``REASONING_LEVELS``); ``None`` = Standard des Modells.
        Provider ohne Unterstützung ignorieren sie.

        Gibt ``(Turn, User-Nachricht)`` zurück. Wirft ``NotFound`` (Chat fehlt),
        ``Busy`` (Turn läuft schon), ``RetryStale`` (Wiederholung passt nicht
        mehr) und ``ValueError`` (leere Nachricht, fehlende ``user_message_id``,
        unbekannter oder deaktivierter Provider).
        """
        if self._closed.is_set():
            raise ServiceClosed("Der ChatService wurde bereits beendet.")
        provider_id = _v_str("provider_id", provider_id, max_len=MAX_ID_CHARS)
        model = _v_str("model", model)
        if system_prompt is not None:
            if not isinstance(system_prompt, str):
                raise ValueError("Feld 'system_prompt' muss ein String sein.")
            if len(system_prompt) > MAX_SYSTEM_PROMPT_CHARS:
                raise ValueError(f"Der System-Prompt ist zu lang (höchstens {MAX_SYSTEM_PROMPT_CHARS} Zeichen).")
        excluded = frozenset(MAIN_EXCLUDED_TOOLS if exclude_tools is None else exclude_tools)
        reasoning = _v_str("reasoning", reasoning, max_len=20)
        if reasoning is not None and reasoning not in REASONING_LEVELS:
            raise ValueError(f"Feld 'reasoning' muss einer der Werte {', '.join(REASONING_LEVELS)} sein.")
        if retry:
            user_message_id = _v_str("user_message_id", user_message_id, max_len=MAX_ID_CHARS)
            if user_message_id is None:
                raise ValueError(
                    "Feld 'user_message_id' fehlt: Zum Wiederholen die ID der User-Nachricht angeben, "
                    "deren Antwort neu erzeugt werden soll."
                )
        else:
            if not isinstance(content, str):
                raise ValueError("Feld 'content' muss ein String sein.")
            if not content.strip():
                raise ValueError("Die Nachricht ist leer.")
            if len(content) > MAX_CONTENT_CHARS:
                raise ValueError(f"Die Nachricht ist zu lang (höchstens {MAX_CONTENT_CHARS} Zeichen).")
        if self.store.get_chat(chat_id) is None:
            raise NotFound("Chat nicht gefunden.")

        # Chat reservieren: ab hier kann kein zweiter Turn für diesen Chat starten.
        with self._lock:
            if self._closed.is_set():
                raise ServiceClosed("Der ChatService wurde bereits beendet.")
            self._prune_turns()
            running = self._active.get(chat_id)
            if running is not None and not running.finished:
                raise Busy("In diesem Chat läuft bereits eine Antwort.")
            turn = Turn(chat_id)
            self._active[chat_id] = turn
            self._turns[turn.id] = turn

        slot: _ProviderSlot | None = None
        try:
            chat = self.store.get_chat(chat_id)  # nach der Reservierung: aktueller Provider-Zustand
            if chat is None:
                raise NotFound("Chat nicht gefunden.")
            self._sync_config()
            with self._cfg_lock:
                pid = provider_id or chat.get("provider_id") or self._default_provider_id()
                if not pid:
                    raise ValueError("Es ist kein Provider konfiguriert.")
                cfg = self._provider_cfg(pid)
                if cfg is None:
                    raise ValueError(f"Provider {pid!r} existiert nicht (mehr). Bitte einen anderen Provider wählen.")
                if cfg.get("enabled", True) is False:
                    raise ValueError(f"Provider {cfg.get('name') or pid} ist deaktiviert.")
                cfg = copy.deepcopy(cfg)
                settings = copy.deepcopy(self._settings())
            if model:
                chosen = model
            elif pid == chat.get("provider_id"):
                chosen = chat.get("model") or None
            else:
                chosen = None  # Providerwechsel ohne Modellangabe: Standard des neuen Providers
            default = settings.get("default_model") if pid == settings.get("default_provider") else None
            effective = chosen or default or cfg.get("default_model") or None
            if system_prompt is None:
                system_prompt = settings.get("system_prompt") or ""
                if not isinstance(system_prompt, str):
                    system_prompt = ""

            slot = self._acquire_provider(pid, cfg)
            if retry:
                assert user_message_id is not None
                user = self._prepare_retry(chat_id, user_message_id)
            else:
                user = Message(role="user", content=content)
                self.store.append_messages(chat_id, user)
            self.store.mark_turn_started(chat_id, turn.id, user.id)
            updates: dict[str, Any] = {}
            if chat.get("provider_id") != pid:
                updates["provider_id"] = pid
            if (chat.get("model") or None) != chosen:
                updates["model"] = chosen
            if updates:
                self.store.update_chat(chat_id, **updates)
            state = chat.get("provider_state")
            state = state if isinstance(state, dict) else {}
        except BaseException as e:
            if slot is not None:
                self._release_provider(slot)
            with self._lock:
                if self._active.get(chat_id) is turn:
                    del self._active[chat_id]
                self._turns.pop(turn.id, None)
            if isinstance(e, KeyError) and not isinstance(e, NotFound):
                raise NotFound("Chat nicht gefunden.") from None
            raise

        turn.provider_id, turn.model = pid, effective
        ctx = _TurnContext(slot, pid, effective, system_prompt, state, user, excluded, reasoning)
        thread = threading.Thread(
            target=self._run_turn, args=(turn, ctx), name=f"chat-turn-{turn.id[-8:]}", daemon=True
        )
        turn._thread = thread
        thread.start()
        return turn, user

    def active_turn(self, chat_id: str) -> Turn | None:
        """Laufender Turn des Chats oder ``None``."""
        with self._lock:
            turn = self._active.get(chat_id)
            return turn if turn is not None and not turn.finished else None

    def get_turn(self, turn_id: str) -> Turn | None:
        """Laufender oder vor höchstens 5 Minuten beendeter Turn, sonst ``None``."""
        with self._lock:
            self._prune_turns()
            return self._turns.get(turn_id)

    def cancel_turn(self, turn_id: str) -> bool:
        """Bricht einen laufenden Turn ab. ``False``, wenn er unbekannt oder schon beendet ist."""
        turn = self.get_turn(turn_id)
        if turn is None or turn.finished:
            return False
        self._request_cancel(turn)
        return True

    def _request_cancel(self, turn: Turn) -> None:
        turn.cancel.set()
        with turn._store_lock:
            if turn._watchdog is not None or turn._closing or self._closed.is_set():
                return
            timer = threading.Timer(FORCE_FINISH_S, self._force_finish, args=(turn,))
            timer.daemon = True
            turn._watchdog = timer
        timer.start()

    def _force_finish(self, turn: Turn) -> None:
        """Beendet einen Turn, dessen Provider nicht auf den Abbruch reagiert (Worker läuft weiter, wird ignoriert)."""
        with turn._store_lock:
            if turn._closing or turn._abandoned or turn.finished:
                return
            turn._abandoned = True
        log.warning("Turn %s reagiert nicht auf den Abbruch und wird ohne den Provider beendet", turn.id)
        try:  # vor der Freigabe des Chats, damit eine Wiederholung den Endstatus sieht
            self.store.mark_turn_finished(turn.chat_id, turn.id, "cancelled")
        except Exception:
            log.warning("Status von Turn %s nicht gespeichert", turn.id, exc_info=True)
        with self._lock:
            if self._active.get(turn.chat_id) is turn:
                del self._active[turn.chat_id]
        turn.publish(ev_done("cancelled"))

    def _prune_turns(self) -> None:
        """Entfernt beendete Turns, die älter als ``TURN_TTL_S`` sind (unter ``self._lock``)."""
        now = time.monotonic()
        expired = [
            tid for tid, t in self._turns.items()
            if t.finished_at is not None and now - t.finished_at > TURN_TTL_S
        ]
        for tid in expired:
            del self._turns[tid]

    def _prepare_retry(self, chat_id: str, user_message_id: str) -> Message:
        """Prüft, ob die Wiederholung noch passt, und löscht alles hinter der User-Nachricht.

        Läuft, während der Chat für den neuen Turn reserviert ist: Kein anderer
        Turn kann dazwischen Nachrichten anhängen oder den Status ändern.
        Wirft ``RetryStale``, wenn ``user_message_id`` nicht die letzte
        User-Nachricht ist oder zu ihr schon eine fertige Antwort vorliegt.
        """
        messages = self.store.get_messages(chat_id)
        index = next((i for i in range(len(messages) - 1, -1, -1) if messages[i].role == "user"), None)
        if index is None or messages[index].id != user_message_id:
            raise RetryStale(
                "Diese Nachricht kann nicht mehr wiederholt werden: Im Chat wurde inzwischen weitergeschrieben. "
                "Bitte den Chat neu laden."
            )
        user = messages[index]
        after = len(messages) - index - 1
        last = self.store.get_last_turn(chat_id)
        if last is not None and last["user_message_id"] == user.id:
            # running ohne laufenden Turn = Server wurde mitten im Turn beendet: wiederholbar
            if last["status"] == "completed":
                raise RetryStale(
                    "Zu dieser Nachricht gibt es inzwischen eine vollständige Antwort (z. B. aus einem anderen "
                    "Fenster). Bitte den Chat neu laden."
                )
        elif after:
            raise RetryStale(
                "Es ist nicht bekannt, ob die Antwort auf diese Nachricht vollständig ist; sie wird daher nicht "
                "ersetzt. Bitte den Chat neu laden oder die Nachricht erneut senden."
            )
        if after:
            removed = self.store.delete_messages_after(chat_id, user.id)
            log.info("Wiederholung in Chat %s: %d Nachricht(en) hinter der User-Nachricht entfernt",
                     chat_id, removed)
        return user

    def _history_before(self, chat_id: str, message_id: str) -> list[Message]:
        messages = self.store.get_messages(chat_id)
        for index in range(len(messages) - 1, -1, -1):
            if messages[index].id == message_id:
                return messages[:index]
        raise RuntimeError("Die Nachricht wurde im Chat nicht gefunden (Chat gelöscht?).")

    def _run_turn(self, turn: Turn, ctx: _TurnContext) -> None:
        """Worker: Session bauen, ``stream()`` durchlaufen, Nachrichten speichern, Events veröffentlichen."""
        from session import Session

        status: str | None = None
        session: Session | None = None
        started = False

        def publish_start() -> None:
            nonlocal started
            if not started:
                started = True
                turn.publish(ev_turn_start(turn.id, turn.chat_id, ctx.provider_id, ctx.model))

        try:
            history = self._history_before(turn.chat_id, ctx.user.id)
            tools = self._turn_tools()
            for name in ctx.exclude_tools:
                tools.remove(name)
            session = Session(
                system_prompt=ctx.system_prompt,
                model=ctx.model,
                tools=tools,
                provider=ctx.slot.provider,
                messages=history,
                state=ctx.state,
                reasoning=ctx.reasoning,
            )
            for ev in session.stream(ctx.user, cancel=turn.cancel):
                kind = ev.get("type")
                if kind == "done":
                    status = ev.get("status") or "completed"
                    continue  # erst nach dem Speichern des Zustands veröffentlichen
                if kind == "turn_start":
                    if not started:
                        started = True
                        turn.publish({**ev, "turn_id": turn.id, "chat_id": turn.chat_id,
                                      "provider_id": ctx.provider_id})
                    continue
                publish_start()
                if kind == "message":
                    stored = self._store_message(turn, ctx, ev)
                    if stored is None:
                        continue
                    ev = stored
                elif kind == "error":
                    ev = self._redact_event(ev)
                turn.publish(ev)
        except Exception as e:
            log.exception("Turn %s fehlgeschlagen", turn.id)
            publish_start()
            turn.publish(ev_error(INTERNAL, self.redact(f"Interner Fehler: {_exc_text(e)}")))
            status = "error"
        finally:
            if session is not None:
                _close_quietly(session, "Session")
            self._finish_turn(turn, ctx, status)

    def _store_message(self, turn: Turn, ctx: _TurnContext, ev: Event) -> Event | None:
        """Speichert die Nachricht eines ``message``-Events; gibt das zu veröffentlichende Event zurück."""
        try:
            msg = Message.from_dict(ev["message"])
        except Exception:
            log.warning("Ungültige Nachricht im Turn %s verworfen", turn.id, exc_info=True)
            return None
        if msg.role == "user":
            log.warning("Provider %s hat eine User-Nachricht gesendet; verworfen", ctx.provider_id)
            return None
        if msg.provider_id is None:
            msg.provider_id = ctx.provider_id
        if msg.model is None and ctx.model:
            msg.model = ctx.model
        with turn._store_lock:
            if turn._abandoned:
                return None
            if not turn._chat_gone and not turn._store_failed:
                try:
                    self.store.append_messages(turn.chat_id, msg)
                except KeyError:
                    log.info("Chat %s wurde während des Turns gelöscht; Turn wird abgebrochen", turn.chat_id)
                    turn._chat_gone = True
                    turn.cancel.set()
                except Exception as e:
                    log.exception("Nachricht im Turn %s konnte nicht gespeichert werden", turn.id)
                    turn._store_failed = True
                    turn.cancel.set()
                    turn.publish(ev_error(INTERNAL, self.redact(
                        f"Die Antwort konnte nicht gespeichert werden: {_exc_text(e)}")))
        return ev_message(msg)

    def _finish_turn(self, turn: Turn, ctx: _TurnContext, status: str | None) -> None:
        """Zustand und Endstatus speichern, Provider freigeben, Chat freigeben, ``done`` veröffentlichen."""
        with turn._store_lock:
            abandoned = turn._abandoned
            turn._closing = True
            watchdog = turn._watchdog
        if watchdog is not None:
            watchdog.cancel()
        if status is None:
            status = "cancelled" if turn.cancel.is_set() else "error"
        if turn._store_failed or status not in ("completed", "cancelled", "error"):
            status = "error"
        try:
            if not abandoned and not turn._chat_gone:
                try:
                    self.store.update_chat(turn.chat_id, provider_state=ctx.state)
                except Exception:
                    log.warning("Provider-Zustand von Chat %s nicht gespeichert", turn.chat_id, exc_info=True)
                try:  # vor der Freigabe des Chats, damit eine Wiederholung den Endstatus sieht
                    self.store.mark_turn_finished(turn.chat_id, turn.id, status)
                except Exception:
                    log.warning("Status von Turn %s nicht gespeichert", turn.id, exc_info=True)
        finally:
            self._release_provider(ctx.slot)
            if not abandoned:
                with self._lock:
                    if self._active.get(turn.chat_id) is turn:
                        del self._active[turn.chat_id]
                turn.publish(ev_done(cast("DoneStatus", status)))


# --- Demo ------------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import json
    import sys
    import tempfile

    from chat_types import ModelInfo, ProviderStatus, ToolCall, TurnRequest, ev_delta
    from providers.base import Provider as _Provider

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(prog="python -m chat_service",
                                     description="Demo des ChatService (temporäre Daten).")
    parser.add_argument("prompt", nargs="?", default="Hallo! Wie spät ist es?")
    parser.add_argument("--provider", default="demo", help="demo (Echo, offline), lmstudio, claude, …")
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    class DemoProvider(_Provider):
        """Echo-Provider: ruft get_datetime auf und wiederholt die Nachricht."""

        type = "demo"

        def list_models(self) -> list[ModelInfo]:
            return [ModelInfo(id="echo", label="Echo")]

        def status(self) -> ProviderStatus:
            return ProviderStatus(ok=True, message="bereit")

        def run_turn(self, req: TurnRequest, emit: Any) -> None:
            call = ToolCall(id="call_demo", name="get_datetime", arguments={"format": "time"})
            emit(ev_message(Message(role="assistant", tool_calls=[call])))
            out = req.tools.execute(call.name, call.arguments, emit, call_id=call.id)
            emit(ev_message(Message(role="tool", content=out.text, tool_call_id=call.id, name=call.name,
                                    is_error=out.is_error)))
            answer = f"Echo: {req.history[-1].content} (Uhrzeit laut Tool: {out.text})"
            for word in answer.split(" "):
                emit(ev_delta(word + " "))
            emit(ev_message(Message(role="assistant", content=answer)))
            req.state["demo_turns"] = int(req.state.get("demo_turns", 0)) + 1

    def factory(cfg: dict[str, Any], data_dir: Path) -> Provider:
        if cfg.get("type") == "demo":
            return DemoProvider(cfg["id"], cfg.get("name") or cfg["id"])
        return _default_factory(cfg, data_dir)

    with tempfile.TemporaryDirectory() as tmp:
        data_dir = Path(tmp)
        store = ChatStore(data_dir / "noassist.db")
        config = ConfigStore(data_dir / "config.json")
        config.load()
        config.data["providers"].append({"id": "demo", "type": "demo", "name": "Demo (Echo)", "enabled": True})
        config.save()
        service = ChatService(store, config, data_dir, provider_factory=factory)
        service.start()
        try:
            chat = service.create_chat(provider_id=args.provider, model=args.model, title="Demo")
            turn, user = service.start_turn(chat["id"], args.prompt)
            print(f"Turn {turn.id} gestartet, User-Nachricht {user.id}")
            try:
                service.start_turn(chat["id"], "Noch eine Frage")
            except Busy as e:
                print("Zweiter Turn:", type(e).__name__, "-", e)
            seq, done = 0, False
            while not done:
                events, done = turn.events_after(seq, 1.0)
                for seq, ev in events:
                    if ev["type"] in ("delta", "reasoning"):
                        print(ev["text"], end="", flush=True)
                    else:
                        print(f"\n  [{seq}] {json.dumps(ev, ensure_ascii=False)[:160]}")
            stored = service.get_chat(chat["id"]) or {}
            print("\nGespeichert:")
            for m in stored.get("messages", []):
                print(f"  {m['role']:9} {(m['content'] or json.dumps(m['tool_calls'], ensure_ascii=False))[:100]}")
            print("Provider-Zustand:", (store.get_chat(chat["id"]) or {}).get("provider_state"))
            service.wait_mcp_ready(5)
            tools = service.list_tools()
            print("Tools:", [t["name"] for t in tools["tools"]], "| MCP:", tools["mcp_servers"])
        finally:
            service.close()
        sys.exit(0 if turn.status == "completed" else 1)
