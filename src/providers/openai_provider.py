"""Provider für OpenAI-kompatible Endpunkte (OpenAI, Ollama, OpenRouter, LM Studio /v1 ...).

Nutzt ``chat.completions.create(stream=True)`` der ``openai``-Bibliothek. Tool-Aufrufe
werden aus den gestreamten Deltas (nach ``index``) zusammengesetzt, über
``req.tools`` ausgeführt und das Ergebnis zurückgegeben, höchstens
``MAX_ROUNDS`` Runden lang.

``openai`` wird nur in diesem Modul importiert. API-Keys tauchen nie in
Fehlermeldungen, Events oder Logs auf; sie werden überall durch ``***`` ersetzt.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import uuid
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any, ClassVar

from chat_types import (
    AUTH_REQUIRED,
    BAD_REQUEST,
    MODEL_NOT_FOUND,
    PROVIDER_ERROR,
    PROVIDER_UNREACHABLE,
    RATE_LIMIT,
    TOOL_UNSUPPORTED,
    Emit,
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

if TYPE_CHECKING:
    from openai import OpenAI

log = logging.getLogger(__name__)

MAX_ROUNDS = 10
"""Höchstzahl der Modellaufrufe pro Turn (jede Tool-Runde zählt als ein Aufruf)."""
DEFAULT_TIMEOUT = 120.0
"""Standard-Timeout (Sekunden) für Lesen; lokale Modelle brauchen beim Laden lange."""
CONNECT_TIMEOUT = 10.0
STATUS_TIMEOUT = 8.0
NO_KEY = "not-needed"
"""Platzhalter, wenn kein Key konfiguriert ist (lokale Server prüfen ihn nicht)."""
CANCELLED_RESULT = "Fehler: abgebrochen"
REASONING_FIELDS = ("reasoning_content", "reasoning")
"""Delta-Felder, in denen Server Reasoning streamen (DeepSeek/LM Studio bzw. Ollama/OpenRouter)."""

HINT_AUTH = "API-Key in den Provider-Einstellungen prüfen (bzw. die Umgebungsvariable aus 'api_key_env')."
HINT_UNREACHABLE = "Läuft der Server? Basis-URL prüfen (meist endet sie auf /v1)."
HINT_MODEL = "Modellnamen bzw. Basis-URL prüfen; die Modellliste des Providers neu laden."
HINT_RATE = "Etwas warten und erneut versuchen."
HINT_TOOLS = (
    "Das Modell bzw. der Server unterstützt anscheinend keine Tools: "
    "'supports_tools' in den Provider-Einstellungen ausschalten oder ein anderes Modell wählen."
)


def _new_call_id() -> str:
    return "call_" + uuid.uuid4().hex[:24]


def _parse_args(raw: str) -> dict[str, Any]:
    """Argument-String als Dict für die gespeicherte Nachricht (bei ungültigem JSON leer)."""
    if not raw or not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except ValueError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def to_openai_messages(history: Iterable[Message], system_prompt: str = "") -> list[dict[str, Any]]:
    """Wandelt den neutralen Verlauf in OpenAI-Chat-Nachrichten um und bereinigt ihn.

    - Ein nicht leerer ``system_prompt`` kommt als erste ``system``-Nachricht.
    - Tool-Aufrufe ohne Ergebnis bekommen ein synthetisches Ergebnis
      ``"Fehler: abgebrochen"``.
    - Tool-Nachrichten ohne passenden vorherigen Aufruf (verwaist oder doppelt)
      werden verworfen.
    - Leere Assistenten-Nachrichten ohne Tool-Aufrufe werden ausgelassen.
    """
    out: list[dict[str, Any]] = []
    if system_prompt and system_prompt.strip():
        out.append({"role": "system", "content": system_prompt})

    pending: dict[str, str] = {}  # offene Aufrufe: id -> name (in Aufrufreihenfolge)

    def close_pending() -> None:
        for call_id in list(pending):
            out.append({"role": "tool", "tool_call_id": call_id, "content": CANCELLED_RESULT})
        pending.clear()

    for msg in history:
        if msg.role == "tool":
            if msg.tool_call_id and msg.tool_call_id in pending:
                pending.pop(msg.tool_call_id)
                out.append({"role": "tool", "tool_call_id": msg.tool_call_id, "content": msg.content or ""})
            else:
                log.debug("Verwaiste Tool-Nachricht %s verworfen", msg.id)
            continue

        close_pending()
        if msg.role == "user":
            out.append({"role": "user", "content": msg.content or ""})
        elif msg.role == "assistant":
            calls = []
            for tc in msg.tool_calls:
                call_id = tc.id or _new_call_id()
                if call_id in pending:  # doppelte ID im fremden Verlauf
                    call_id = _new_call_id()
                pending[call_id] = tc.name
                calls.append({
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments or {}, ensure_ascii=False, default=str),
                    },
                })
            if calls:
                out.append({"role": "assistant", "content": msg.content or None, "tool_calls": calls})
            elif msg.content:
                out.append({"role": "assistant", "content": msg.content})
    close_pending()
    return out


class _CallBuffer:
    """Sammelt die Deltas eines gestreamten Tool-Aufrufs."""

    def __init__(self) -> None:
        self.id = ""
        self.name = ""
        self.arguments = ""

    def feed(self, tc: Any) -> None:
        if tc.id and not self.id:
            self.id = tc.id
        fn = getattr(tc, "function", None)
        if fn is None:
            return
        name = getattr(fn, "name", None)
        if name and name != self.name:
            # Manche Server wiederholen den vollen Namen in jedem Chunk, andere teilen ihn auf.
            self.name = name if not self.name else self.name + name
        args = getattr(fn, "arguments", None)
        if args:
            self.arguments += args


class OpenAIProvider(Provider):
    """Provider für einen OpenAI-kompatiblen Chat-Completions-Endpunkt.

    Konfiguration (wie in ``data/config.json``)::

        {"id": "oai-<6hex>", "type": "openai", "name": ..., "enabled": true,
         "base_url": "http://localhost:11434/v1", "api_key": null, "api_key_env": null,
         "default_model": null, "models": [], "supports_tools": true, "timeout": 120}

    - ``api_key`` hat Vorrang vor ``api_key_env`` (Name einer Umgebungsvariablen).
      Ohne beides wird ``"not-needed"`` gesendet.
    - ``models``: feste Modellliste (Strings oder ``{"id", "label"?}``); leer = ``GET /models``.
    - ``supports_tools``: ``False`` = es werden keine Tools gesendet.
    - Nach ``list_models()`` steht ein eventueller Fehler in ``last_models_error``.
    """

    type: ClassVar[str] = "openai"

    def __init__(
        self,
        id: str,
        name: str,
        base_url: str,
        api_key: str | None = None,
        api_key_env: str | None = None,
        default_model: str | None = None,
        models: Iterable[str | dict[str, Any]] | None = None,
        supports_tools: bool = True,
        timeout: float | None = None,
    ):
        super().__init__(id, name)
        if not base_url or not str(base_url).strip():
            raise ValueError("base_url fehlt")
        self.base_url = str(base_url).strip().rstrip("/")
        self.api_key = api_key or None
        self.api_key_env = (api_key_env or "").strip() or None
        self.default_model = default_model or None
        self.models = list(models or [])
        self.supports_tools = bool(supports_tools)
        self.timeout = float(timeout) if timeout else DEFAULT_TIMEOUT
        self.last_models_error: str | None = None
        self._client: OpenAI | None = None
        self._client_key: str | None = None
        self._lock = threading.Lock()

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> OpenAIProvider:
        """Erzeugt den Provider aus einem Provider-Dict der ``config.json`` (unbekannte Felder werden ignoriert)."""
        return cls(
            id=cfg["id"],
            name=cfg.get("name") or cfg["id"],
            base_url=cfg.get("base_url") or "",
            api_key=cfg.get("api_key"),
            api_key_env=cfg.get("api_key_env"),
            default_model=cfg.get("default_model"),
            models=cfg.get("models") or [],
            supports_tools=cfg.get("supports_tools", True),
            timeout=cfg.get("timeout"),
        )

    def __repr__(self) -> str:
        return f"<OpenAIProvider id={self.id!r} name={self.name!r} base_url={self.base_url!r}>"

    # -- Key und Client --

    def _resolve_key(self) -> str | None:
        if self.api_key:
            return self.api_key
        if self.api_key_env:
            return os.environ.get(self.api_key_env) or None
        return None

    def _secrets(self) -> list[str]:
        keys = {k for k in (self.api_key, os.environ.get(self.api_key_env) if self.api_key_env else None) if k}
        keys.discard(NO_KEY)
        return sorted(keys, key=len, reverse=True)

    def redact(self, text: Any) -> str:
        """Ersetzt alle Keys dieses Providers in ``text`` durch ``***``."""
        text = str(text)
        for key in self._secrets():
            text = text.replace(key, "***")
        return text

    def _get_client(self) -> OpenAI:
        from openai import OpenAI, Timeout

        key = self._resolve_key() or NO_KEY
        with self._lock:
            if self._client is None or self._client_key != key:
                if self._client is not None:
                    self._close_client(self._client)
                self._client = OpenAI(
                    base_url=self.base_url,
                    api_key=key,
                    timeout=Timeout(self.timeout, connect=CONNECT_TIMEOUT),
                    max_retries=1,
                )
                self._client_key = key
            return self._client

    @staticmethod
    def _close_client(client: Any) -> None:
        try:
            client.close()
        except Exception:
            pass

    def close(self) -> None:
        with self._lock:
            if self._client is not None:
                self._close_client(self._client)
            self._client = None
            self._client_key = None

    # -- Fehler --

    def _map_error(self, exc: BaseException, *, tools_sent: bool = False, model: str | None = None) -> ProviderError:
        """Ordnet Exceptions der openai-Bibliothek einem ``ProviderError`` zu (Key geschwärzt)."""
        import openai

        detail = self.redact(getattr(exc, "message", None) or exc).strip()
        if len(detail) > 500:
            detail = detail[:500] + " …"

        if isinstance(exc, openai.APITimeoutError):
            return ProviderError(f"Zeitüberschreitung bei {self.name} ({self.base_url}).", PROVIDER_UNREACHABLE, HINT_UNREACHABLE)
        if isinstance(exc, openai.APIConnectionError):
            return ProviderError(f"{self.name} ist nicht erreichbar ({self.base_url}).", PROVIDER_UNREACHABLE, HINT_UNREACHABLE)
        if isinstance(exc, (openai.AuthenticationError, openai.PermissionDeniedError)):
            return ProviderError(f"Anmeldung bei {self.name} fehlgeschlagen: {detail}", AUTH_REQUIRED, HINT_AUTH)
        if isinstance(exc, openai.NotFoundError):
            what = f"Modell {model!r}" if model else "Endpunkt"
            return ProviderError(f"{what} bei {self.name} nicht gefunden: {detail}", MODEL_NOT_FOUND, HINT_MODEL)
        if isinstance(exc, openai.RateLimitError):
            return ProviderError(f"Rate-Limit bei {self.name} erreicht: {detail}", RATE_LIMIT, HINT_RATE)
        if isinstance(exc, (openai.BadRequestError, openai.UnprocessableEntityError)):
            low = detail.lower()
            if tools_sent and ("tool" in low or "function" in low):
                return ProviderError(f"{self.name} lehnt die Tools ab: {detail}", TOOL_UNSUPPORTED, HINT_TOOLS)
            hint = HINT_TOOLS if tools_sent else None
            return ProviderError(f"Ungültige Anfrage an {self.name}: {detail}", BAD_REQUEST, hint)
        if isinstance(exc, openai.APIStatusError):
            return ProviderError(f"Fehler von {self.name} (HTTP {exc.status_code}): {detail}", PROVIDER_ERROR)
        if isinstance(exc, openai.APIError):
            return ProviderError(f"Fehler von {self.name}: {detail}", PROVIDER_ERROR)
        return ProviderError(f"Fehler bei {self.name}: {type(exc).__name__}: {detail}", PROVIDER_ERROR)

    # -- Modelle und Status --

    def _configured_models(self) -> list[ModelInfo]:
        result: list[ModelInfo] = []
        for m in self.models:
            if isinstance(m, str) and m.strip():
                result.append(ModelInfo(id=m.strip(), label=m.strip()))
            elif isinstance(m, dict) and m.get("id"):
                result.append(ModelInfo(id=str(m["id"]), label=str(m.get("label") or m["id"]), description=m.get("description")))
        return result

    def _fetch_models(self, timeout: float) -> list[ModelInfo]:
        client = self._get_client().with_options(timeout=timeout, max_retries=0)
        ids = sorted({m.id for m in client.models.list() if getattr(m, "id", None)})
        return [ModelInfo(id=i, label=i) for i in ids]

    def list_models(self) -> list[ModelInfo]:
        """Konfigurierte ``models``, sonst ``GET /models``. Fehler: leere Liste, Text in ``last_models_error``."""
        self.last_models_error = None
        configured = self._configured_models()
        if configured:
            return configured
        try:
            return self._fetch_models(STATUS_TIMEOUT)
        except Exception as e:
            err = self._map_error(e)
            self.last_models_error = err.message
            log.warning("Modellliste von %s nicht abrufbar: %s", self.id, err.message)
            return []

    def status(self) -> ProviderStatus:
        """Kurze Prüfung von Erreichbarkeit und Anmeldung über ``GET /models``."""
        details: dict[str, Any] = {"base_url": self.base_url, "has_api_key": bool(self._resolve_key())}
        if self.api_key_env and not self.api_key and not os.environ.get(self.api_key_env):
            details["api_key_env_missing"] = True
        try:
            models = self._fetch_models(STATUS_TIMEOUT)
        except Exception as e:
            err = self._map_error(e)
            if err.code == MODEL_NOT_FOUND and self._configured_models():
                # Server erreichbar, bietet aber kein /models an: mit fester Liste trotzdem nutzbar.
                return ProviderStatus(ok=True, message=f"{self.name} erreichbar (ohne /models).", details=details)
            hint = err.hint
            if err.code == MODEL_NOT_FOUND:
                hint = HINT_UNREACHABLE
            if details.get("api_key_env_missing") and err.code == AUTH_REQUIRED:
                hint = f"Umgebungsvariable {self.api_key_env} ist nicht gesetzt."
            details["code"] = err.code
            return ProviderStatus(ok=False, message=err.message, hint=hint, details=details)
        details["model_count"] = len(models)
        return ProviderStatus(ok=True, message=f"{self.name} erreichbar ({len(models)} Modelle).", details=details)

    # -- Turn --

    def _pick_model(self, requested: str | None) -> str:
        if requested:
            return requested
        if self.default_model:
            return self.default_model
        models = self.list_models()
        if models:
            return models[0].id
        raise ProviderError(
            f"Für {self.name} ist kein Modell ausgewählt.",
            BAD_REQUEST,
            self.last_models_error or "Modell auswählen oder 'default_model' in den Provider-Einstellungen setzen.",
        )

    def run_turn(self, req: TurnRequest, emit: Emit) -> None:
        """Führt einen Turn aus (Vertrag siehe ``Provider``)."""
        model = self._pick_model(req.model)
        messages = to_openai_messages(req.history, req.system_prompt)
        tools = req.tools.for_openai() if (self.supports_tools and req.tools is not None and len(req.tools)) else []

        for _round in range(MAX_ROUNDS):
            if req.cancel.is_set():
                return
            text, calls = self._stream_round(model, messages, tools, req.cancel, emit)
            if req.cancel.is_set():
                # Teilantwort behalten, unvollständige Tool-Aufrufe verwerfen.
                if text:
                    emit(ev_message(Message(role="assistant", content=text, provider_id=self.id, model=model)))
                return

            tool_calls = [
                ToolCall(id=c.id or _new_call_id(), name=c.name, arguments=_parse_args(c.arguments)) for c in calls
            ]
            emit(ev_message(Message(role="assistant", content=text, tool_calls=tool_calls, provider_id=self.id, model=model)))
            if not calls:
                return

            messages.append({
                "role": "assistant",
                "content": text or None,
                "tool_calls": [
                    {"id": tc.id, "type": "function", "function": {"name": c.name, "arguments": c.arguments or "{}"}}
                    for tc, c in zip(tool_calls, calls)
                ],
            })
            for tc, c in zip(tool_calls, calls):
                if req.cancel.is_set():
                    outcome_text, is_error = CANCELLED_RESULT, True
                else:
                    outcome = req.tools.execute(c.name, c.arguments, emit, call_id=tc.id)
                    outcome_text, is_error = outcome.text, outcome.is_error
                emit(ev_message(Message(
                    role="tool", content=outcome_text, tool_call_id=tc.id, name=c.name,
                    is_error=is_error, provider_id=self.id, model=model,
                )))
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": outcome_text})

        emit(ev_status(f"Maximale Anzahl von {MAX_ROUNDS} Modellaufrufen erreicht; Turn beendet."))

    def _stream_round(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        cancel: threading.Event,
        emit: Emit,
    ) -> tuple[str, list[_CallBuffer]]:
        """Ein Modellaufruf mit Streaming. Liefert (Text, Tool-Aufrufe)."""
        kwargs: dict[str, Any] = {"model": model, "messages": messages, "stream": True}
        if tools:
            kwargs["tools"] = tools
        try:
            stream = self._get_client().chat.completions.create(**kwargs)
        except Exception as e:
            if cancel.is_set():
                return "", []
            raise self._map_error(e, tools_sent=bool(tools), model=model) from None

        text_parts: list[str] = []
        buffers: dict[Any, _CallBuffer] = {}
        done = threading.Event()
        # Wächter: schließt den Stream bei Abbruch, auch wenn gerade kein Chunk kommt.
        watcher = threading.Thread(target=self._watch_cancel, args=(cancel, done, stream), daemon=True)
        watcher.start()
        try:
            for chunk in stream:
                if cancel.is_set():
                    break
                for choice in chunk.choices or []:
                    self._handle_delta(choice.delta, text_parts, buffers, emit)
        except Exception as e:
            if not cancel.is_set():
                raise self._map_error(e, tools_sent=bool(tools), model=model) from None
        finally:
            done.set()
            self._close_stream(stream)

        calls = [b for b in buffers.values() if b.name]
        return "".join(text_parts), calls

    @staticmethod
    def _watch_cancel(cancel: threading.Event, done: threading.Event, stream: Any) -> None:
        while not done.is_set():
            if cancel.wait(0.2):
                if not done.is_set():
                    OpenAIProvider._close_stream(stream)
                return

    @staticmethod
    def _close_stream(stream: Any) -> None:
        try:
            stream.close()
        except Exception:
            pass

    @staticmethod
    def _handle_delta(delta: Any, text_parts: list[str], buffers: dict[Any, _CallBuffer], emit: Emit) -> None:
        if delta is None:
            return
        extra = getattr(delta, "model_extra", None) or {}
        for field_name in REASONING_FIELDS:
            value = getattr(delta, field_name, None) or extra.get(field_name)
            if isinstance(value, str) and value:
                emit(ev_reasoning(value))
                break
        content = getattr(delta, "content", None)
        if content:
            text_parts.append(content)
            emit(ev_delta(content))
        for tc in getattr(delta, "tool_calls", None) or []:
            key: Any = tc.index
            if key is None:  # Server ohne index: nach ID, sonst neuer Eintrag
                key = ("id", tc.id) if tc.id else ("pos", len(buffers))
            buffers.setdefault(key, _CallBuffer()).feed(tc)


def _demo_events(prefix: str = "") -> Callable[[dict[str, Any]], None]:
    def show(ev: dict[str, Any]) -> None:
        if ev["type"] in ("delta", "reasoning"):
            print(ev["text"], end="", flush=True)
        else:
            print(f"\n{prefix}[{ev['type']}] {json.dumps(ev, ensure_ascii=False)[:200]}")

    return show


if __name__ == "__main__":
    import sys

    from tools import get_tools
    from toolset import ToolSet

    # Aufruf: cd src && python -m providers.openai_provider [base_url] [model]
    base = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:1234/v1"
    provider = OpenAIProvider("demo", "Demo", base_url=base, default_model=sys.argv[2] if len(sys.argv) > 2 else None)
    print(provider, provider.status())
    models = provider.list_models()
    print("Modelle:", [m.id for m in models][:10], provider.last_models_error or "")
    if models:
        req = TurnRequest(
            history=[Message(role="user", content="Wie spät ist es? Nutze das Tool get_datetime.")],
            model=None,
            system_prompt="Antworte kurz auf Deutsch.",
            tools=ToolSet.from_any(get_tools()),
        )
        try:
            provider.run_turn(req, _demo_events())
        except ProviderError as e:
            print("\nProviderError:", e.code, e.message, e.hint)
    provider.close()
