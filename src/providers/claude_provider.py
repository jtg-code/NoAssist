"""Claude-Provider über die Claude Agent SDK (Claude-Code-CLI als Unterprozess).

Anmeldung (``auth_mode``):

- ``"login"`` (Standard): der lokale Claude-Code-Login dieses PCs
  (``claude auth login``). Ein global gesetzter ``ANTHROPIC_API_KEY`` wird
  dafür im Unterprozess neutralisiert.
- ``"api_key"``: ein eigener Anthropic-API-Key aus der Konfiguration.

Anmeldedaten verwaltet ausschließlich die Claude-Code-CLI. Dieses Modul liest
keine Anmeldedateien; der Status kommt aus ``claude auth status --json``.

Ablauf eines Turns:

- Pro Turn wird ein eigener ``ClaudeSDKClient`` (also ein CLI-Prozess)
  gestartet. Alle Coroutinen laufen in einer gemeinsamen Event-Loop in einem
  Hintergrund-Thread (``AsyncRunner``); ``run_turn`` blockiert den aufrufenden
  Thread.
- Eingebaute Claude-Code-Tools sind aus (``tools=[]``); verfügbar sind nur die
  Tools aus ``req.tools`` als In-Process-MCP-Server ``noassist``.
- Gesprächskontext: Passt ``state["claude"]["synced_message_id"]`` zur
  vorletzten Nachricht des Verlaufs, wird die Claude-Session per ``resume``
  fortgesetzt. Sonst (neuer Chat, Providerwechsel, nach Abbruch oder Fehler)
  startet eine neue Session, und der bisherige Verlauf geht als klar
  markiertes Transkript vor der neuen Nachricht mit.

``claude_agent_sdk`` wird nur in diesem Modul importiert, und zwar erst bei Bedarf.
"""

from __future__ import annotations

import asyncio
import collections
import concurrent.futures
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Coroutine, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, TypeVar

from chat_types import (
    AUTH_REQUIRED,
    BAD_REQUEST,
    CLI_MISSING,
    INTERNAL,
    MODEL_NOT_FOUND,
    PROVIDER_ERROR,
    PROVIDER_UNREACHABLE,
    RATE_LIMIT,
    Emit,
    Event,
    Message,
    ModelInfo,
    ProviderError,
    ProviderStatus,
    ToolCall,
    TurnRequest,
    ev_delta,
    ev_message,
    ev_reasoning,
    ev_status,
)
from providers.base import Provider

log = logging.getLogger(__name__)

T = TypeVar("T")

# --- Konstanten ----------------------------------------------------------------

AUTH_MODES = ("login", "api_key")
SERVER_NAME = "noassist"
"""Name des In-Process-MCP-Servers mit den NoAssist-Tools."""
TOOL_PREFIX = f"mcp__{SERVER_NAME}__"
"""Präfix, unter dem Claude die NoAssist-Tools sieht (wird in Nachrichten entfernt)."""
MAX_TURNS = 20
"""Höchstzahl der Agenten-Schritte (Modellaufrufe) pro Turn."""

STATUS_TIMEOUT_S = 20.0
MODELS_TTL_S = 600.0
MODELS_RETRY_TTL_S = 60.0
"""Nach einem Fehler beim Abrufen der Modellliste gilt der Fallback nur so lange."""
MODELS_TIMEOUT_S = 60.0
POLL_S = 0.05
INTERRUPT_TIMEOUT_S = 1.0
CANCEL_GRACE_S = 1.8
"""So lange wartet die Event-Loop nach einem Abbruch auf das Ende des Turns."""
CANCEL_HARD_LIMIT_S = 2.8
"""Spätestens so lange nach einem Abbruch kehrt ``run_turn`` zurück."""
CLOSE_TIMEOUT_S = 3.0
RESUME_WAIT_S = 10.0
"""So lange wird vor einem ``resume`` auf das Beenden des vorigen CLI-Prozesses gewartet."""

TRANSCRIPT_MAX_CHARS = 150_000
TRANSCRIPT_MSG_MAX_CHARS = 20_000
TRANSCRIPT_TOOL_MAX_CHARS = 4_000

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data"

DEFAULT_SYSTEM_PROMPT = (
    "Du bist ein hilfreicher Assistent in der Chat-Anwendung NoAssist. "
    "Antworte in der Sprache des Nutzers. Nutze die bereitgestellten Tools, "
    "wenn sie bei der Antwort helfen."
)

LOGIN_HINT = (
    "Claude Code ist auf diesem PC nicht angemeldet: Terminal öffnen und "
    "`claude auth login` ausführen (oder `claude` -> /login), dann in NoAssist "
    "'Erneut prüfen'."
)
API_KEY_HINT = "Einen gültigen Anthropic-API-Key in den Provider-Einstellungen eintragen."
CLI_HINT = (
    "Claude Code (claude.exe) wurde nicht gefunden. Normalerweise bringt das Paket "
    "claude-agent-sdk die CLI mit (`.venv\\Scripts\\python.exe -m pip install --force-reinstall "
    "claude-agent-sdk`). Alternativ Claude Code installieren oder in data/config.json "
    "beim Claude-Provider 'cli_path' auf claude.exe setzen."
)

FALLBACK_MODELS: tuple[tuple[str, str], ...] = (
    ("default", "Standard"),
    ("sonnet", "Sonnet"),
    ("opus", "Opus"),
    ("haiku", "Haiku"),
)

_RATE_LIMIT_WINDOWS = {
    "five_hour": "5-Stunden-Fenster",
    "seven_day": "7 Tage",
    "seven_day_opus": "7 Tage, Opus",
    "seven_day_sonnet": "7 Tage, Sonnet",
    "overage": "Zusatzkontingent",
}

_KEY_PATTERN = re.compile(r"sk-ant-[A-Za-z0-9_\-]{6,}")


# --- Event-Loop im Hintergrund ------------------------------------------------------

class AsyncRunner:
    """Eine asyncio-Event-Loop in einem eigenen Hintergrund-Thread.

    Der Thread startet beim ersten ``submit``. Nach ``close`` startet ein
    weiteres ``submit`` eine neue Loop. Threadsicher.
    """

    def __init__(self, name: str = "claude-agent-loop") -> None:
        self._name = name
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        with self._lock:
            if self._loop is not None:
                return self._loop
            loop = asyncio.new_event_loop()  # unter Windows: ProactorEventLoop (kann Unterprozesse)
            ready = threading.Event()

            def run() -> None:
                asyncio.set_event_loop(loop)
                ready.set()
                loop.run_forever()

            thread = threading.Thread(target=run, name=self._name, daemon=True)
            thread.start()
            ready.wait()
            self._loop, self._thread = loop, thread
            return loop

    def submit(self, coro: Coroutine[Any, Any, T]) -> concurrent.futures.Future[T]:
        """Plant ``coro`` in der Loop ein und liefert ein ``concurrent.futures.Future``."""
        try:
            loop = self._ensure_loop()
            return asyncio.run_coroutine_threadsafe(coro, loop)
        except BaseException:
            coro.close()
            raise

    def run(self, coro: Coroutine[Any, Any, T], timeout: float | None = None) -> T:
        """Führt ``coro`` aus und blockiert bis zum Ergebnis (bei Zeitüberschreitung wird abgebrochen)."""
        fut = self.submit(coro)
        try:
            return fut.result(timeout)
        except concurrent.futures.TimeoutError:
            fut.cancel()
            raise TimeoutError(f"Zeitüberschreitung nach {timeout} s") from None

    def close(self, timeout: float = 3.0) -> None:
        """Wartet bis zu ``timeout`` Sekunden auf laufende Coroutinen, bricht den Rest ab und beendet die Loop."""
        with self._lock:
            loop, thread = self._loop, self._thread
            self._loop = self._thread = None
        if loop is None or thread is None:
            return

        async def drain() -> None:
            me = asyncio.current_task()
            tasks = {t for t in asyncio.all_tasks() if t is not me}
            if tasks:
                _, pending = await asyncio.wait(tasks, timeout=timeout)
                for task in pending:
                    task.cancel()
                if pending:
                    await asyncio.wait(pending, timeout=1.0)
            await loop.shutdown_asyncgens()
            # Ausstehende Rückrufe (Prozessende, Schließen der Pipes) noch abarbeiten,
            # sonst bleiben Transporte offen und warnen beim Aufräumen.
            await asyncio.sleep(0.2)

        try:
            asyncio.run_coroutine_threadsafe(drain(), loop).result(timeout + 2.0)
        except Exception:
            log.warning("Event-Loop des Claude-Providers ließ sich nicht sauber leeren", exc_info=True)
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=2.0)
        if not thread.is_alive():
            loop.close()


# --- CLI-Suche und Anmeldung ---------------------------------------------------

def _require_sdk() -> None:
    """Stellt sicher, dass ``claude_agent_sdk`` importierbar ist (sonst ``ProviderError``)."""
    try:
        import claude_agent_sdk  # noqa: F401
    except ImportError as e:
        raise ProviderError("Das Paket claude-agent-sdk ist nicht installiert.", CLI_MISSING, CLI_HINT) from e


def _is_native_exe(path: str) -> bool:
    return sys.platform != "win32" or Path(path).suffix.lower() in (".exe", ".com")


def find_claude_cli() -> str:
    """Sucht die Claude-Code-CLI in derselben Reihenfolge wie die Claude Agent SDK.

    1. die mit ``claude-agent-sdk`` mitgelieferte CLI (``_bundled``),
    2. ``claude`` bzw. ``claude.exe`` im ``PATH`` (unter Windows nur echte ``.exe``),
    3. unter Windows ``~/.local/bin/claude.exe``.

    Wirft ``ProviderError(code="cli_missing")``, wenn nichts gefunden wird.
    """
    exe_name = "claude.exe" if sys.platform == "win32" else "claude"
    _require_sdk()
    import claude_agent_sdk

    bundled = Path(claude_agent_sdk.__file__).parent / "_bundled" / exe_name
    if bundled.is_file():
        return str(bundled)
    for candidate in ("claude", "claude.exe"):
        found = shutil.which(candidate)
        if found and _is_native_exe(found):
            return found
    if sys.platform == "win32":
        local = Path.home() / ".local" / "bin" / "claude.exe"
        if local.is_file():
            return str(local)
    raise ProviderError("Claude Code (CLI) wurde nicht gefunden.", CLI_MISSING, CLI_HINT)


def _resolve_cli(cli_path: str | None) -> str:
    """Konfigurierter Pfad (falls gesetzt) oder ``find_claude_cli()``."""
    if not cli_path:
        return find_claude_cli()
    path = Path(os.path.expandvars(os.path.expanduser(cli_path)))
    resolved = str(path) if path.is_file() else shutil.which(cli_path)
    if not resolved:
        raise ProviderError(
            f"Claude Code wurde unter dem konfigurierten Pfad nicht gefunden: {cli_path}",
            CLI_MISSING,
            CLI_HINT,
        )
    if not _is_native_exe(resolved):
        raise ProviderError(
            f"Der konfigurierte Pfad ist kein ausführbares claude.exe: {cli_path}",
            CLI_MISSING,
            "Unter Windows wird nur die native claude.exe unterstützt (kein npm-Skript wie claude.cmd).",
        )
    return resolved


def _base_env() -> dict[str, str]:
    """Umgebung für CLI-Unterprozesse (ohne ``CLAUDECODE``, wie in der SDK)."""
    return {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}


def _login_env() -> dict[str, str]:
    """Umgebungs-Überschreibungen für den Login-Modus (globalen API-Key neutralisieren)."""
    return {"ANTHROPIC_API_KEY": ""}


def start_login(cli_path: str | None = None) -> None:
    """Öffnet ``claude auth login`` in einer neuen Konsole und kehrt sofort zurück.

    Unter Windows läuft die Anmeldung in einem eigenen Konsolenfenster
    (``CREATE_NEW_CONSOLE``); das Fenster bleibt nach dem Ende offen, bis eine
    Taste gedrückt wird, damit Erfolg oder Fehler lesbar sind. Es wird keine
    Shell benutzt. Wirft ``ProviderError(code="cli_missing")``, wenn die CLI
    fehlt.
    """
    cli = _resolve_cli(cli_path)
    env = _base_env()
    env.update(_login_env())
    if sys.platform == "win32":
        python = Path(sys.executable)
        if python.name.lower() == "pythonw.exe" and (python.parent / "python.exe").is_file():
            python = python.parent / "python.exe"  # pythonw hat keine Konsole
        wrapper = (
            "import subprocess, sys\n"
            "print('NoAssist: Anmeldung bei Claude Code ...', flush=True)\n"
            "code = subprocess.call(sys.argv[1:])\n"
            "print()\n"
            "print('Anmeldung erfolgreich.' if code == 0 else f'Anmeldung beendet (Exit-Code {code}).')\n"
            "print('In NoAssist auf \"Erneut prüfen\" klicken.')\n"
            "input('Enter drücken, um dieses Fenster zu schließen ...')\n"
        )
        subprocess.Popen(
            [str(python), "-c", wrapper, cli, "auth", "login"],
            env=env,
            creationflags=subprocess.CREATE_NEW_CONSOLE,
            close_fds=True,
        )
    else:
        subprocess.Popen([cli, "auth", "login"], env=env, start_new_session=True)


# --- Transkript für die Übergabe -------------------------------------------------

def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n[... gekürzt, {len(text) - limit} weitere Zeichen]"


def _transcript_blocks(messages: Iterable[Message]) -> list[str]:
    """Rendert den Verlauf als Textblöcke; Tool-Aufrufe ohne Ergebnis bekommen "Fehler: abgebrochen"."""
    msgs = list(messages)
    answered = {m.tool_call_id for m in msgs if m.role == "tool" and m.tool_call_id}
    blocks: list[str] = []
    missing: list[ToolCall] = []  # Aufrufe ohne Ergebnis; Platzhalter nach den echten Ergebnissen

    def flush_missing() -> None:
        for tc in missing:
            blocks.append(f"[Tool-Ergebnis {tc.name} (ID {tc.id}), Fehler]\nFehler: abgebrochen")
        missing.clear()

    for m in msgs:
        if m.role != "tool":
            flush_missing()
        if m.role == "user":
            if m.content.strip():
                blocks.append("[Benutzer]\n" + _clip(m.content, TRANSCRIPT_MSG_MAX_CHARS))
        elif m.role == "assistant":
            lines = ["[Assistent]"]
            if m.content.strip():
                lines.append(_clip(m.content, TRANSCRIPT_MSG_MAX_CHARS))
            for tc in m.tool_calls:
                args = json.dumps(tc.arguments, ensure_ascii=False, default=str)
                lines.append(f"[Tool-Aufruf {tc.name} (ID {tc.id})] {_clip(args, TRANSCRIPT_TOOL_MAX_CHARS)}")
            if len(lines) > 1:
                blocks.append("\n".join(lines))
            missing.extend(tc for tc in m.tool_calls if tc.id not in answered)
        elif m.role == "tool":
            flag = ", Fehler" if m.is_error else ""
            blocks.append(
                f"[Tool-Ergebnis {m.name or 'unbekannt'} (ID {m.tool_call_id or '?'}){flag}]\n"
                + _clip(m.content, TRANSCRIPT_TOOL_MAX_CHARS)
            )
    flush_missing()
    return blocks


def build_handover_prompt(previous: Iterable[Message], new_text: str) -> str:
    """Prompt für eine neue Claude-Session: bisheriger Verlauf als Transkript plus neue Nachricht.

    Ohne bisherigen Verlauf ist das Ergebnis einfach ``new_text``. Lange
    Verläufe werden von vorne gekürzt (höchstens ``TRANSCRIPT_MAX_CHARS``
    Zeichen; die neuesten Einträge bleiben erhalten).
    """
    blocks = _transcript_blocks(previous)
    if not blocks:
        return new_text
    kept: list[str] = []
    total = 0
    for block in reversed(blocks):
        if kept and total + len(block) > TRANSCRIPT_MAX_CHARS:
            break
        kept.append(block)
        total += len(block) + 2
    kept.reverse()
    body = "\n\n".join(kept)
    omitted = len(blocks) - len(kept)
    if omitted:
        body = f"[... {omitted} ältere Einträge ausgelassen ...]\n\n" + body
    return (
        "<bisheriger_verlauf>\n"
        "Hinweis: Dies ist der bisherige Verlauf dieses Chats als Transkript (geführt in NoAssist, "
        "eventuell mit einem anderen Modell). Er dient nur als Kontext: Die Tool-Aufrufe darin sind "
        "bereits erledigt und werden nicht wiederholt. Setze das Gespräch nahtlos fort und antworte "
        "auf die neue Nachricht unten.\n\n"
        + body
        + "\n</bisheriger_verlauf>\n\n<neue_nachricht>\n"
        + new_text
        + "\n</neue_nachricht>"
    )


# --- Hilfen für einen Turn ----------------------------------------------------------

class _EmitGate:
    """Leitet Events an ``emit`` weiter, bis ``close`` aufgerufen wurde (threadsicher).

    Verhindert Events nach dem Ende von ``run_turn`` (z. B. das verspätete
    ``tool_result`` eines Tools, das beim Abbruch noch lief).
    """

    def __init__(self, emit: Emit) -> None:
        self._emit = emit
        self._lock = threading.RLock()
        self._open = True

    def __call__(self, event: Event) -> None:
        with self._lock:
            if not self._open:
                log.debug("Event %r nach Turn-Ende verworfen", event.get("type"))
                return
            try:
                self._emit(event)
            except Exception:
                log.exception("Event %r konnte nicht gesendet werden", event.get("type"))

    def close(self) -> None:
        with self._lock:
            self._open = False


@dataclass
class _PendingAssistant:
    """Eine Assistenten-Nachricht im Aufbau (Blöcke mit derselben ``message_id``)."""

    key: str
    model: str | None
    texts: list[str] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)


def _strip_prefix(name: str) -> str:
    return name[len(TOOL_PREFIX):] if name.startswith(TOOL_PREFIX) else name


def _tool_result_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for item in content if isinstance(content, list) else [content]:
        if isinstance(item, Mapping):
            if item.get("type") == "text":
                parts.append(str(item.get("text") or ""))
            elif item.get("type") == "image":
                parts.append("[Bild]")
            else:
                parts.append(json.dumps(item, ensure_ascii=False, default=str)[:500])
        else:
            parts.append(str(item))
    return "\n".join(parts)


def _format_reset(resets_at: Any) -> str | None:
    if not isinstance(resets_at, (int, float)) or resets_at <= 0:
        return None
    try:
        return datetime.fromtimestamp(resets_at).strftime("%d.%m. %H:%M")
    except (OverflowError, OSError, ValueError):
        return None


def _rate_limit_text(info: Any, warning: bool) -> str:
    window = _RATE_LIMIT_WINDOWS.get(getattr(info, "rate_limit_type", None) or "", "")
    usage = getattr(info, "utilization", None)
    reset = _format_reset(getattr(info, "resets_at", None))
    if warning:
        text = "Hinweis: Das Claude-Nutzungslimit"
        text += f" ({window})" if window else ""
        text += f" ist zu {usage * 100:.0f} % ausgeschöpft" if isinstance(usage, (int, float)) else " ist fast erreicht"
    else:
        text = "Das Claude-Nutzungslimit"
        text += f" ({window})" if window else ""
        text += " ist erreicht"
    text += f"; zurückgesetzt am {reset}." if reset else "."
    return text


def _retry_text(data: Mapping[str, Any]) -> str:
    """Statustext zu einer ``api_retry``-Meldung der CLI."""
    kind, status = data.get("error"), data.get("error_status")
    if kind == "rate_limit" or status == 429:
        reason = "Nutzungslimit erreicht"
    elif isinstance(status, int):
        reason = f"Fehler {status}"
    else:
        reason = "Verbindungsproblem"
    text = f"Claude-API: {reason}, neuer Versuch"
    attempt, max_retries = data.get("attempt"), data.get("max_retries")
    if isinstance(attempt, int):
        text += f" {attempt}" + (f"/{max_retries}" if isinstance(max_retries, int) else "")
    delay = data.get("retry_delay_ms")
    if isinstance(delay, (int, float)) and delay > 0:
        text += f" in {delay / 1000:.0f} s"
    return text + " ..."


class _ClaudeTurn:
    """Zustand eines Turns. ``handle`` und ``finalize`` laufen in der Event-Loop."""

    def __init__(
        self, provider: ClaudeAgentProvider, req: TurnRequest, gate: _EmitGate, prompt: str, resume: str | None
    ) -> None:
        self.provider = provider
        self.req = req
        self.gate = gate
        self.prompt = prompt
        self.resume = resume
        self.options: Any = None
        self.finished = threading.Event()
        self.force_cancel = False
        self.cancelled = False
        self.completed = False
        self.got_result = False
        self.resume_missing = False
        self.error: ProviderError | None = None
        self.session_id: str | None = resume
        self.last_message_id: str | None = None
        self.emitted_messages = 0
        self.stderr_tail: collections.deque[str] = collections.deque(maxlen=40)
        self.api_error: tuple[str, str] | None = None  # (Art, Text) aus AssistantMessage.error
        self.rate_limit: Any = None  # RateLimitInfo mit status "rejected"
        self._warned_rate_limit = False
        self._pending: _PendingAssistant | None = None
        self._tool_names: dict[str, str] = {}
        self._open_calls: dict[str, str] = {}  # ausgegebene Tool-Aufrufe ohne Ergebnis
        self._stream_msg_id: str | None = None
        self._streamed_text: set[str | None] = set()
        self._streamed_thinking: set[str | None] = set()
        self._partial: list[str] = []  # gestreamter Text ohne abschließenden TextBlock
        self._model: str | None = None

    @property
    def cancel_requested(self) -> bool:
        return self.force_cancel or self.req.cancel.is_set()

    async def wait(self, task: asyncio.Task[Any]) -> bool:
        """Wartet auf ``task``; ``False``, sobald ein Abbruch angefordert wurde."""
        while True:
            done, _ = await asyncio.wait({task}, timeout=POLL_S)
            if done:
                return True
            if self.cancel_requested:
                return False

    def on_stderr(self, line: str) -> None:
        line = self.provider.scrub(line)
        self.stderr_tail.append(line)
        log.debug("claude stderr: %s", line[:500])

    # -- Nachrichten der SDK --

    def handle(self, msg: Any) -> None:
        from claude_agent_sdk import (
            AssistantMessage,
            RateLimitEvent,
            ResultMessage,
            StreamEvent,
            SystemMessage,
            UserMessage,
        )

        if isinstance(msg, StreamEvent):
            self._on_stream(msg)
        elif isinstance(msg, AssistantMessage):
            self._on_assistant(msg)
        elif isinstance(msg, UserMessage):
            self._on_user(msg)
        elif isinstance(msg, ResultMessage):
            self._on_result(msg)
        elif isinstance(msg, RateLimitEvent):
            self._on_rate_limit(msg)
        elif isinstance(msg, SystemMessage):
            self._on_system(msg)

    def _on_stream(self, msg: Any) -> None:
        if msg.parent_tool_use_id:
            return
        event = msg.event if isinstance(msg.event, Mapping) else {}
        typ = event.get("type")
        if typ == "message_start":
            self._stream_msg_id = (event.get("message") or {}).get("id")
        elif typ == "content_block_delta":
            delta = event.get("delta") or {}
            if delta.get("type") == "text_delta" and delta.get("text"):
                self._streamed_text.add(self._stream_msg_id)
                self._partial.append(delta["text"])
                self.gate(ev_delta(delta["text"]))
            elif delta.get("type") == "thinking_delta" and delta.get("thinking"):
                self._streamed_thinking.add(self._stream_msg_id)
                self.gate(ev_reasoning(delta["thinking"]))
        elif (
            typ in ("message_delta", "message_stop")
            and self._pending is not None
            and self._pending.key == self._stream_msg_id
        ):
            # Alle Blöcke der API-Nachricht sind da: Nachricht abschließen.
            self._flush()

    def _on_assistant(self, msg: Any) -> None:
        from claude_agent_sdk import TextBlock, ThinkingBlock, ToolUseBlock

        if msg.parent_tool_use_id:
            return
        if msg.error:
            texts = [b.text for b in msg.content if isinstance(b, TextBlock)]
            self.api_error = (str(msg.error), " ".join(texts))
            return
        key = msg.message_id or msg.uuid or f"anon-{id(msg)}"
        if self._pending is not None and self._pending.key != key:
            self._flush()
        if self._pending is None:
            self._pending = _PendingAssistant(key=key, model=msg.model or self._model)
        self._model = msg.model or self._model
        for block in msg.content:
            if isinstance(block, TextBlock):
                if block.text:
                    self._pending.texts.append(block.text)
                    if key not in self._streamed_text:
                        self.gate(ev_delta(block.text))  # ohne Streaming: Text am Stück
                self._partial.clear()
            elif isinstance(block, ThinkingBlock):
                if key not in self._streamed_thinking and block.thinking:
                    self.gate(ev_reasoning(block.thinking))
            elif isinstance(block, ToolUseBlock):
                name = _strip_prefix(block.name)
                self._tool_names[block.id] = name
                args = dict(block.input) if isinstance(block.input, Mapping) else {"input": block.input}
                self._pending.tool_calls.append(ToolCall(id=block.id, name=name, arguments=args))

    def _on_user(self, msg: Any) -> None:
        from claude_agent_sdk import ToolResultBlock

        if msg.parent_tool_use_id or isinstance(msg.content, str):
            return
        results = [b for b in msg.content if isinstance(b, ToolResultBlock)]
        if not results:
            return
        self._flush()
        for block in results:
            self._open_calls.pop(block.tool_use_id, None)
            # Nach einem Abbruch liefert die CLI selbst eine (englische, ans Modell
            # gerichtete) Ablehnung; im Verlauf einheitlich "Fehler: abgebrochen".
            aborted = bool(block.is_error) and self.cancel_requested
            self._emit_message(
                Message(
                    role="tool",
                    content="Fehler: abgebrochen" if aborted else _tool_result_text(block.content),
                    tool_call_id=block.tool_use_id,
                    name=self._tool_names.get(block.tool_use_id),
                    is_error=bool(block.is_error),
                    provider_id=self.provider.id,
                    model=self._model,
                )
            )

    def _on_result(self, msg: Any) -> None:
        self.got_result = True
        self._flush()
        if msg.session_id:
            self.session_id = msg.session_id
        if not msg.is_error and msg.subtype == "success":
            self.completed = True
            return
        if self.cancel_requested or (msg.terminal_reason or "").startswith("aborted"):
            self.cancelled = True
            return
        if self.error is None:
            self.error = self.provider.result_error(self, msg)

    def _on_rate_limit(self, msg: Any) -> None:
        info = msg.rate_limit_info
        if info.status == "rejected":
            self.rate_limit = info
        elif info.status == "allowed_warning" and not self._warned_rate_limit:
            self._warned_rate_limit = True
            self.gate(ev_status(_rate_limit_text(info, warning=True)))

    def _on_system(self, msg: Any) -> None:
        data = msg.data if isinstance(msg.data, Mapping) else {}
        if msg.subtype == "api_retry":
            self.gate(ev_status(_retry_text(data)))
            return
        if msg.subtype == "status" and data.get("status") == "compacting":
            self.gate(ev_status("Claude fasst den bisherigen Verlauf zusammen ..."))
            return
        if msg.subtype != "init":
            return
        if data.get("session_id"):
            self.session_id = data["session_id"]
        tools = [str(t) for t in data.get("tools") or []]
        self.provider.last_init = {
            "session_id": data.get("session_id"),
            "model": data.get("model"),
            "tools": tools,
            "mcp_servers": [
                {"name": s.get("name"), "status": s.get("status")}
                for s in data.get("mcp_servers") or []
                if isinstance(s, Mapping)
            ],
            "api_key_source": data.get("apiKeySource"),
            "permission_mode": data.get("permissionMode"),
            "cwd": data.get("cwd"),
            "claude_code_version": data.get("claude_code_version"),
        }
        foreign = [t for t in tools if not t.startswith(TOOL_PREFIX)]
        if foreign:
            log.warning("Claude meldet unerwartete Tools: %s", foreign)

    # -- Ausgabe --

    def _emit_message(self, message: Message) -> None:
        self.gate(ev_message(message))
        self.last_message_id = message.id
        self.emitted_messages += 1

    def _flush(self) -> None:
        pending, self._pending = self._pending, None
        if pending is None:
            return
        content = "\n\n".join(t for t in pending.texts if t)
        if not content and not pending.tool_calls:
            return
        self._emit_message(
            Message(
                role="assistant",
                content=content,
                tool_calls=pending.tool_calls,
                provider_id=self.provider.id,
                model=pending.model,
            )
        )
        for tc in pending.tool_calls:
            self._open_calls[tc.id] = tc.name

    def finalize(self) -> None:
        """Schließt offene Nachrichten ab, schließt das Gate und meldet das Ende (idempotent).

        Bei Abbruch oder Fehler bekommen Tool-Aufrufe ohne Ergebnis ein
        synthetisches Ergebnis "Fehler: abgebrochen", und gestreamter, aber
        unvollständiger Text wird als Assistenten-Nachricht ausgegeben.
        """
        if self.finished.is_set():
            return
        try:
            partial = "".join(self._partial).strip() if not self.completed else ""
            self._partial.clear()
            if partial and self._pending is not None and self._pending.key == self._stream_msg_id:
                self._pending.texts.append(partial)  # gehört zur Nachricht im Aufbau
                partial = ""
            self._flush()
            if not self.completed:
                for call_id, name in list(self._open_calls.items()):
                    self._emit_message(
                        Message(
                            role="tool",
                            content="Fehler: abgebrochen",
                            tool_call_id=call_id,
                            name=name,
                            is_error=True,
                            provider_id=self.provider.id,
                            model=self._model,
                        )
                    )
                self._open_calls.clear()
            if partial:
                self._emit_message(
                    Message(role="assistant", content=partial, provider_id=self.provider.id, model=self._model)
                )
        except Exception:
            log.exception("Claude-Turn konnte nicht sauber abgeschlossen werden")
        finally:
            self.gate.close()
            self.finished.set()


# --- Provider -------------------------------------------------------------------------

class ClaudeAgentProvider(Provider):
    """Claude über die Claude Agent SDK (lokaler Claude-Code-Login oder API-Key).

    Konfiguration (Eintrag in ``data/config.json``): ``id``, ``name``,
    ``auth_mode`` (``"login"`` oder ``"api_key"``), ``api_key``, ``cli_path``
    (optional, Standard: mitgelieferte CLI) und ``default_model``
    (``"default"`` = Standardmodell von Claude Code).
    """

    type = "claude_agent"

    def __init__(
        self,
        id: str = "claude",
        name: str = "Claude (Agent SDK)",
        *,
        auth_mode: str = "login",
        api_key: str | None = None,
        cli_path: str | None = None,
        default_model: str | None = "default",
        data_dir: str | Path | None = None,
    ) -> None:
        super().__init__(id, name)
        mode = (auth_mode or "login").strip().lower()
        if mode not in AUTH_MODES:
            raise ValueError(f"Unbekannter auth_mode {auth_mode!r} (erlaubt: login, api_key)")
        self.auth_mode = mode
        self.api_key = (api_key or "").strip() or None
        self.cli_path = (cli_path or "").strip() or None
        self.default_model = (default_model or "").strip() or "default"
        self.data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
        self.cwd = self.data_dir / "claude-cwd"
        self.last_init: dict[str, Any] | None = None
        """Zusammenfassung der letzten Init-Meldung der CLI (Modell, Tools, MCP-Status; ohne Geheimnisse)."""
        self._runner = AsyncRunner(f"claude-agent-{id}")
        self._lock = threading.Lock()
        self._turns: set[_ClaudeTurn] = set()
        self._closing: dict[str, asyncio.Task[Any]] = {}  # nur in der Event-Loop benutzt
        self._models_lock = threading.Lock()
        self._models: list[ModelInfo] | None = None
        self._models_until = 0.0

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any], data_dir: str | Path | None = None) -> ClaudeAgentProvider:
        """Baut den Provider aus einem Konfigurationseintrag (``type == "claude_agent"``)."""
        return cls(
            id=str(cfg.get("id") or "claude"),
            name=str(cfg.get("name") or "Claude (Agent SDK)"),
            auth_mode=str(cfg.get("auth_mode") or "login"),
            api_key=cfg.get("api_key"),
            cli_path=cfg.get("cli_path"),
            default_model=cfg.get("default_model") or "default",
            data_dir=data_dir,
        )

    # -- Hilfen --

    def scrub(self, text: str) -> str:
        """Entfernt API-Keys aus Texten (für Fehlermeldungen und Logs)."""
        text = str(text)
        if self.api_key:
            text = text.replace(self.api_key, "***")
        return _KEY_PATTERN.sub("***", text)

    def _env(self) -> dict[str, str]:
        """Umgebungs-Überschreibungen für die CLI je nach ``auth_mode``."""
        if self.auth_mode == "api_key":
            return {"ANTHROPIC_API_KEY": self.api_key or ""}
        return _login_env()

    def _model_option(self, model: str | None) -> str | None:
        name = (model or self.default_model or "").strip()
        return None if not name or name.lower() == "default" else name

    def _ensure_cwd(self) -> Path:
        try:
            self.cwd.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise ProviderError(f"Arbeitsordner für Claude nicht anlegbar: {self.cwd} ({e})", INTERNAL) from e
        return self.cwd

    def _auth_error(self) -> ProviderError:
        if self.auth_mode == "api_key":
            return ProviderError("Der Anthropic-API-Key fehlt oder wurde abgelehnt.", AUTH_REQUIRED, API_KEY_HINT)
        return ProviderError("Claude Code ist nicht angemeldet.", AUTH_REQUIRED, LOGIN_HINT)

    def _detail(self, text: str) -> str:
        text = " ".join(self.scrub(text or "").split())
        return text[:300] + (" ..." if len(text) > 300 else "")

    def api_error(self, kind: str, text: str, status: int | None, turn: _ClaudeTurn | None = None) -> ProviderError:
        """Ordnet einen API-Fehler (``AssistantMessage.error``, Text, HTTP-Status) einem ``ProviderError`` zu."""
        low = (text or "").lower()
        detail = self._detail(text)
        if (
            kind == "authentication_failed"
            or status == 401
            or "not logged in" in low
            or "please run /login" in low
            or "invalid api key" in low
            or "invalid x-api-key" in low
            or "oauth token has expired" in low
        ):
            return self._auth_error()
        if kind == "rate_limit" or status == 429 or (turn is not None and turn.rate_limit is not None):
            info = turn.rate_limit if turn is not None else None
            message = "Das Claude-Nutzungslimit ist erreicht."
            if info is not None:
                message = _rate_limit_text(info, warning=False)
            return ProviderError(message + " Bitte später erneut versuchen.", RATE_LIMIT)
        if kind == "billing_error" or status == 402 or "credit balance" in low:
            return ProviderError(f"Claude meldet ein Abrechnungsproblem: {detail}", PROVIDER_ERROR)
        if status == 404 or ("model" in low and ("not found" in low or "not_found" in low or "invalid model" in low)):
            return ProviderError(f"Das Modell ist nicht verfügbar: {detail}", MODEL_NOT_FOUND)
        if kind == "invalid_request" or status == 400:
            return ProviderError(f"Claude hat die Anfrage abgelehnt: {detail}", BAD_REQUEST)
        if kind == "server_error" or (isinstance(status, int) and status >= 500):
            return ProviderError(
                f"Die Claude-API meldet einen Serverfehler (eventuell überlastet): {detail}", PROVIDER_UNREACHABLE
            )
        return ProviderError(f"Claude meldet einen Fehler: {detail or kind}", PROVIDER_ERROR)

    def result_error(self, turn: _ClaudeTurn, msg: Any) -> ProviderError:
        """Fehler aus einer ``ResultMessage`` mit ``is_error``."""
        if turn.api_error is not None:
            return self.api_error(*turn.api_error, msg.api_error_status, turn)
        if msg.subtype == "error_max_turns":
            return ProviderError(
                f"Claude hat die Höchstzahl von {MAX_TURNS} Schritten erreicht, ohne fertig zu werden.", PROVIDER_ERROR
            )
        text = "; ".join(msg.errors or []) or (msg.result or "") or msg.subtype
        return self.api_error("unknown", text, msg.api_error_status, turn)

    def map_exception(self, exc: BaseException, turn: _ClaudeTurn) -> ProviderError:
        """Ordnet Ausnahmen aus SDK/CLI einem ``ProviderError`` zu (ohne Geheimnisse)."""
        from claude_agent_sdk import ClaudeSDKError, CLIConnectionError, CLINotFoundError, ProcessError, ResultError

        if isinstance(exc, ProviderError):
            return exc
        text = str(exc)
        if turn.resume and "no conversation found" in text.lower():
            turn.resume_missing = True
            return ProviderError("Die Claude-Session zu diesem Chat wurde nicht gefunden.", PROVIDER_ERROR)
        if turn.api_error is not None:
            return self.api_error(*turn.api_error, getattr(exc, "api_error_status", None), turn)
        if isinstance(exc, CLINotFoundError):
            return ProviderError("Claude Code (CLI) wurde nicht gefunden.", CLI_MISSING, CLI_HINT)
        if isinstance(exc, ResultError):
            if exc.subtype == "error_max_turns":
                return ProviderError(
                    f"Claude hat die Höchstzahl von {MAX_TURNS} Schritten erreicht, ohne fertig zu werden.",
                    PROVIDER_ERROR,
                )
            detail = "; ".join(exc.errors) or exc.result or exc.subtype or ""
            return self.api_error("unknown", detail, exc.api_error_status, turn)
        stderr = "\n".join(turn.stderr_tail).lower()
        if "not logged in" in stderr or "please run /login" in stderr:
            return self._auth_error()
        if isinstance(exc, ProcessError):
            return ProviderError(f"Claude Code wurde unerwartet beendet (Exit-Code {exc.exit_code}).", PROVIDER_ERROR)
        if isinstance(exc, CLIConnectionError):
            return ProviderError(
                f"Verbindung zu Claude Code fehlgeschlagen: {self._detail(text)}", PROVIDER_UNREACHABLE
            )
        if text.startswith("Control request timeout"):
            return ProviderError("Claude Code antwortet nicht (Zeitüberschreitung).", PROVIDER_UNREACHABLE)
        if isinstance(exc, ClaudeSDKError):
            return ProviderError(f"Fehler in der Claude Agent SDK ({type(exc).__name__}).", INTERNAL)
        log.error("Unerwarteter Fehler im Claude-Turn: %s: %s", type(exc).__name__, self._detail(text))
        return ProviderError(f"Interner Fehler im Claude-Provider ({type(exc).__name__}).", INTERNAL)

    # -- Provider-Schnittstelle --

    def status(self) -> ProviderStatus:
        """Prüft CLI und Anmeldung (Login-Modus: ``claude auth status --json``, Timeout 20 s)."""
        details: dict[str, Any] = {"logged_in": None, "auth_method": None, "subscription_type": None}
        try:
            cli = _resolve_cli(self.cli_path)
        except ProviderError as e:
            return ProviderStatus(False, e.message, e.hint, {**details, "code": e.code})
        if self.auth_mode == "api_key":
            details["auth_method"] = "api_key"
            if self.api_key:
                return ProviderStatus(True, "API-Key hinterlegt.", None, details)
            return ProviderStatus(
                False, "Kein Anthropic-API-Key hinterlegt.", API_KEY_HINT, {**details, "code": AUTH_REQUIRED}
            )

        env = _base_env()
        env.update(self._env())
        try:
            proc = subprocess.run(
                [cli, "auth", "status", "--json"],
                stdin=subprocess.DEVNULL,
                capture_output=True,
                timeout=STATUS_TIMEOUT_S,
                env=env,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except subprocess.TimeoutExpired:
            return ProviderStatus(
                False, "Claude Code antwortet nicht (Zeitüberschreitung bei der Statusabfrage).", None,
                {**details, "code": PROVIDER_UNREACHABLE},
            )
        except OSError as e:
            return ProviderStatus(
                False,
                f"Claude Code ließ sich nicht starten ({type(e).__name__}).",
                CLI_HINT,
                {**details, "code": CLI_MISSING},
            )
        try:
            data = json.loads(proc.stdout.decode("utf-8", "replace") or "null")
        except ValueError:
            data = None
        if not isinstance(data, dict):
            log.debug("claude auth status: Exit-Code %s, keine JSON-Antwort", proc.returncode)
            return ProviderStatus(
                False, "Der Anmeldestatus von Claude Code ist nicht lesbar.", LOGIN_HINT, {**details, "code": INTERNAL}
            )
        logged_in = data.get("loggedIn") is True
        details["logged_in"] = logged_in
        if isinstance(data.get("authMethod"), str):
            details["auth_method"] = data["authMethod"]
        if isinstance(data.get("subscriptionType"), str):
            details["subscription_type"] = data["subscriptionType"]
        if isinstance(data.get("email"), str) and data["email"]:
            details["email"] = data["email"]
        if not logged_in:
            return ProviderStatus(
                False, "Claude Code ist nicht angemeldet.", LOGIN_HINT, {**details, "code": AUTH_REQUIRED}
            )
        extra = ", ".join(str(v) for v in (details["auth_method"], details["subscription_type"]) if v)
        return ProviderStatus(True, f"Angemeldet ({extra})." if extra else "Angemeldet.", None, details)

    def list_models(self) -> list[ModelInfo]:
        """Modelle aus ``get_server_info()["models"]`` (10 Minuten gecacht; Fallback: default/sonnet/opus/haiku)."""
        with self._models_lock:
            now = time.monotonic()
            if self._models is not None and now < self._models_until:
                return list(self._models)
            try:
                models = self._runner.run(self._fetch_models(), timeout=MODELS_TIMEOUT_S)
            except Exception as e:
                detail = e.message if isinstance(e, ProviderError) else type(e).__name__
                log.warning("Claude-Modellliste nicht abrufbar (%s); nutze Standardliste", detail)
                models = []
            ttl = MODELS_TTL_S
            if not models:
                models = [ModelInfo(id=mid, label=label) for mid, label in FALLBACK_MODELS]
                ttl = MODELS_RETRY_TTL_S
            self._models, self._models_until = models, time.monotonic() + ttl
            return list(models)

    async def _fetch_models(self) -> list[ModelInfo]:
        from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient

        options = ClaudeAgentOptions(
            tools=[],
            setting_sources=[],
            strict_mcp_config=True,
            permission_mode="dontAsk",
            cwd=str(self._ensure_cwd()),
            cli_path=_resolve_cli(self.cli_path),
            env=self._env(),
            stderr=lambda line: log.debug("claude stderr: %s", self.scrub(line)[:500]),
        )
        client = ClaudeSDKClient(options=options)
        try:
            await client.connect()
            info = await client.get_server_info() or {}
        finally:
            try:
                await client.disconnect()
            except Exception:
                log.debug("Trennen nach Modellabfrage fehlgeschlagen", exc_info=True)
        models: list[ModelInfo] = []
        for entry in info.get("models") or []:
            if not isinstance(entry, Mapping) or not entry.get("value"):
                continue
            value = str(entry["value"])
            models.append(
                ModelInfo(
                    id=value,
                    label=str(entry.get("displayName") or value),
                    description=str(entry["description"]) if entry.get("description") else None,
                )
            )
        return models

    def start_login(self) -> None:
        """Öffnet ``claude auth login`` in einer neuen Konsole (für ``POST /api/providers/claude/login``)."""
        start_login(self.cli_path)

    def run_turn(self, req: TurnRequest, emit: Emit) -> None:
        """Führt einen Turn aus (blockiert; Vertrag siehe ``Provider``)."""
        if req.cancel.is_set():
            return
        turn = self._prepare_turn(req, _EmitGate(emit), allow_resume=True)
        turn.gate(ev_status("Claude Code wird gestartet ..."))
        self._execute(turn)
        if turn.resume_missing and turn.emitted_messages == 0 and not req.cancel.is_set():
            # Session-Datei fehlt (z. B. gelöscht): neue Session mit Transkript.
            log.info("Claude-Session %s nicht gefunden; neue Session mit Transkript", turn.resume)
            turn = self._prepare_turn(req, _EmitGate(emit), allow_resume=False)
            self._execute(turn)
        self._finish(req, turn)

    def _prepare_turn(self, req: TurnRequest, gate: _EmitGate, allow_resume: bool) -> _ClaudeTurn:
        _require_sdk()
        from claude_agent_sdk import ClaudeAgentOptions

        history = list(req.history or [])
        if not history or history[-1].role != "user":
            raise ProviderError("Der Verlauf muss mit einer Benutzernachricht enden.", BAD_REQUEST)
        text = history[-1].content or ""
        if not text.strip():
            raise ProviderError("Die Nachricht ist leer.", BAD_REQUEST)
        if self.auth_mode == "api_key" and not self.api_key:
            raise ProviderError("Kein Anthropic-API-Key hinterlegt.", AUTH_REQUIRED, API_KEY_HINT)
        cli = _resolve_cli(self.cli_path)
        cwd = self._ensure_cwd()

        prev = req.state.get("claude") if isinstance(req.state, dict) else None
        prev = prev if isinstance(prev, Mapping) else {}
        session_id = prev.get("session_id") if isinstance(prev.get("session_id"), str) else None
        synced = prev.get("synced_message_id")
        resume = (
            session_id
            if allow_resume and session_id and synced and len(history) >= 2 and history[-2].id == synced
            else None
        )
        prompt = text if resume else build_handover_prompt(history[:-1], text)

        turn = _ClaudeTurn(self, req, gate, prompt, resume)
        mcp_servers: dict[str, Any] = {}
        allowed: list[str] = []
        if req.tools is not None and len(req.tools) > 0:
            server, allowed = req.tools.for_claude(gate, SERVER_NAME)
            mcp_servers = {SERVER_NAME: server}
        turn.options = ClaudeAgentOptions(
            tools=[],
            mcp_servers=mcp_servers,
            allowed_tools=allowed,
            permission_mode="dontAsk",
            setting_sources=[],
            strict_mcp_config=True,
            include_partial_messages=True,
            verbatim_prompts=True,
            cwd=str(cwd),
            max_turns=MAX_TURNS,
            model=self._model_option(req.model),
            system_prompt=(req.system_prompt or "").strip() or DEFAULT_SYSTEM_PROMPT,
            cli_path=cli,
            env=self._env(),
            resume=resume,
            stderr=turn.on_stderr,
        )
        return turn

    def _execute(self, turn: _ClaudeTurn) -> None:
        """Startet den Turn in der Event-Loop und wartet auf sein Ende (bei Abbruch höchstens ~3 s)."""
        req = turn.req
        with self._lock:
            self._turns.add(turn)
        try:
            fut = self._runner.submit(self._run(turn))
            cancel_seen: float | None = None
            while not turn.finished.wait(POLL_S):
                if fut.done():
                    break  # Coroutine endete ohne finalize (sollte nicht vorkommen)
                if req.cancel.is_set() or turn.force_cancel:
                    cancel_seen = cancel_seen or time.monotonic()
                    if time.monotonic() - cancel_seen >= CANCEL_HARD_LIMIT_S:
                        log.warning("Claude-Turn reagiert nicht rechtzeitig auf den Abbruch; Rest läuft im Hintergrund")
                        turn.cancelled = True
                        break
            if fut.done() and not turn.finished.is_set() and turn.error is None:
                exc = fut.exception() if not fut.cancelled() else None
                turn.error = self.map_exception(exc, turn) if exc else ProviderError(
                    "Claude-Turn wurde unerwartet beendet.", INTERNAL
                )
        finally:
            turn.gate.close()
            with self._lock:
                self._turns.discard(turn)

    def _finish(self, req: TurnRequest, turn: _ClaudeTurn) -> None:
        """Aktualisiert ``req.state["claude"]`` und wirft ggf. den Fehler des Turns."""
        cancelled = not turn.completed and (turn.cancelled or req.cancel.is_set() or turn.force_cancel)
        ok = turn.completed and turn.error is None
        synced = (turn.last_message_id or req.history[-1].id) if ok else None
        prev_raw = req.state.get("claude")
        prev: Mapping[str, Any] = prev_raw if isinstance(prev_raw, Mapping) else {}
        req.state["claude"] = {
            "session_id": turn.session_id or prev.get("session_id"),
            "synced_message_id": synced,
        }
        if cancelled:
            if turn.force_cancel and not req.cancel.is_set():
                raise ProviderError("Der Claude-Provider wurde beendet.", PROVIDER_ERROR)
            return
        if turn.error is not None:
            raise turn.error

    async def _run(self, turn: _ClaudeTurn) -> None:
        """Coroutine eines Turns: verbinden, Prompt senden, Nachrichten verarbeiten, trennen."""
        from claude_agent_sdk import ClaudeSDKClient

        client: Any = None
        connect_task: asyncio.Task[Any] | None = None
        recv_task: asyncio.Task[Any] | None = None
        interrupt_task: asyncio.Task[Any] | None = None
        try:
            if turn.resume:
                previous = self._closing.get(turn.resume)
                if previous is not None and not previous.done():
                    await asyncio.wait({previous}, timeout=RESUME_WAIT_S)
            if turn.cancel_requested:
                turn.cancelled = True
                return
            client = ClaudeSDKClient(options=turn.options)
            connect_task = asyncio.create_task(client.connect())
            if not await turn.wait(connect_task):
                turn.cancelled = True
                return
            connect_task.result()
            if turn.cancel_requested:
                turn.cancelled = True
                return
            await client.query(turn.prompt)
            recv_task = asyncio.create_task(self._receive(client, turn))
            if await turn.wait(recv_task):
                recv_task.result()
                if not turn.got_result and not turn.cancelled and turn.error is None:
                    raise ProviderError("Claude Code hat die Verbindung unerwartet beendet.", PROVIDER_ERROR)
            else:
                turn.cancelled = True
                interrupt_task = asyncio.create_task(self._interrupt(client))
                await asyncio.wait({recv_task}, timeout=CANCEL_GRACE_S)
        except asyncio.CancelledError:
            turn.cancelled = True
        except Exception as e:
            if not turn.cancelled and turn.error is None:
                turn.error = self.map_exception(e, turn)
        finally:
            turn.finalize()
            me = asyncio.current_task()
            if turn.session_id and me is not None:
                self._closing[turn.session_id] = me
            try:
                await self._shutdown(client, connect_task, recv_task, interrupt_task)
            finally:
                if turn.session_id and self._closing.get(turn.session_id) is me:
                    del self._closing[turn.session_id]

    @staticmethod
    async def _receive(client: Any, turn: _ClaudeTurn) -> None:
        async for msg in client.receive_response():
            turn.handle(msg)

    @staticmethod
    async def _interrupt(client: Any) -> None:
        try:
            await asyncio.wait_for(client.interrupt(), INTERRUPT_TIMEOUT_S)
        except Exception as e:
            log.debug("Interrupt an Claude Code fehlgeschlagen: %s", type(e).__name__)

    @staticmethod
    async def _shutdown(
        client: Any,
        connect_task: asyncio.Task[Any] | None,
        recv_task: asyncio.Task[Any] | None,
        interrupt_task: asyncio.Task[Any] | None,
    ) -> None:
        """Beendet Hilfs-Tasks und trennt den Client (läuft nach ``finalize``, also ohne Events)."""
        if connect_task is not None and not connect_task.done():
            await asyncio.wait({connect_task})  # ein halber Verbindungsaufbau lässt sich nicht sauber abbrechen
        for task in (interrupt_task, recv_task):
            if task is not None and not task.done():
                task.cancel()
                await asyncio.wait({task}, timeout=1.0)
        for task in (connect_task, recv_task, interrupt_task):
            if task is not None and task.done() and not task.cancelled():
                task.exception()  # als abgerufen markieren (keine "never retrieved"-Warnung)
        if client is not None:
            try:
                await client.disconnect()
            except Exception:
                log.debug("Trennen von Claude Code fehlgeschlagen", exc_info=True)

    def close(self) -> None:
        """Bricht laufende Turns ab, wartet kurz auf das Beenden der CLI-Prozesse und stoppt die Event-Loop."""
        with self._lock:
            turns = list(self._turns)
        for turn in turns:
            turn.force_cancel = True
        deadline = time.monotonic() + CLOSE_TIMEOUT_S
        for turn in turns:
            turn.finished.wait(max(0.0, deadline - time.monotonic()))
        self._runner.close(timeout=max(0.5, deadline - time.monotonic()))


if __name__ == "__main__":
    import argparse

    from tools import get_tools
    from toolset import ToolSet

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Demo des Claude-Agent-Providers")
    parser.add_argument("prompt", nargs="?", default="Was ist 17+25? Nutze das Tool add.")
    parser.add_argument("--model", default="haiku", help="Modell (Standard: haiku, am günstigsten)")
    parser.add_argument("--status-only", action="store_true", help="nur Status und Modelle anzeigen")
    args = parser.parse_args()

    provider = ClaudeAgentProvider()
    try:
        st = provider.status()
        print("Status:", st.ok, st.message, st.details, st.hint or "")
        print("Modelle:", [m.id for m in provider.list_models()])
        if st.ok and not args.status_only:

            def show(ev: Event) -> None:
                if ev["type"] in ("delta", "reasoning"):
                    print(ev["text"], end="", flush=True)
                elif ev["type"] == "message":
                    m = ev["message"]
                    print(f"\n[message] {m['role']}: {m['content'][:80]!r} tool_calls={m['tool_calls']}")
                else:
                    print(f"\n[{ev['type']}] {json.dumps(ev, ensure_ascii=False)[:200]}")

            req = TurnRequest(
                history=[Message(role="user", content=args.prompt)],
                model=args.model,
                system_prompt="",
                tools=ToolSet.from_any(get_tools("add", "get_datetime")),
            )
            provider.run_turn(req, show)
            print("\nZustand:", req.state)
            print("Init-Tools:", (provider.last_init or {}).get("tools"))
    finally:
        provider.close()
