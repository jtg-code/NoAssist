"""Provider für LM Studio über das native lmstudio-SDK (synchroner Client).

- Der neutrale Verlauf (``Message``) wird über die ``lms.Chat.add_*``-Methoden
  nachgebaut (``Chat.from_history`` lehnt das Dict-Format für Tool-Aufrufe ab).
- Mit Tools läuft ``LLM.act()`` (Tool-Schleife im SDK, höchstens 10 Runden),
  ohne Tools ``LLM.respond()``.
- Gestreamte Fragmente werden zu ``delta``- bzw. ``reasoning``-Events, fertige
  Nachrichten aus ``on_message`` zu neutralen Nachrichten (``ev_message``).
- Der Sync-Client ist nicht threadsicher: Vorhersagen laufen unter einem Lock,
  Modellliste und Status nutzen einen eigenen Client mit eigenem Lock.
- Abbruch: ``req.cancel`` wird von einem Wächter-Thread beobachtet, der die
  laufende Vorhersage beim Server abbricht; zusätzlich beenden die Callbacks
  die Tool-Schleife zwischen den Runden.

``lmstudio`` wird erst in diesem Modul (bei Bedarf) importiert.
"""

from __future__ import annotations

import json
import logging
import threading
import uuid
from collections.abc import Callable, Generator
from contextlib import AbstractContextManager, contextmanager
from typing import TYPE_CHECKING, Any

from chat_types import (
    BAD_REQUEST,
    INTERNAL,
    MODEL_NOT_FOUND,
    PROVIDER_ERROR,
    PROVIDER_UNREACHABLE,
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
    import lmstudio as lms
    from lmstudio._sdk_models import ToolCallRequestDataDict

log = logging.getLogger(__name__)

MAX_PREDICTION_ROUNDS = 10
"""Höchstzahl der Vorhersage-Runden in ``act()`` (inklusive Tool-Runden)."""
ERROR_PREFIX = "Fehler:"
"""Tool-Ergebnisse mit diesem Präfix gelten als Fehler (wie in ``toolset``)."""
ABORTED_RESULT = "Fehler: abgebrochen"
"""Synthetisches Ergebnis für Tool-Aufrufe ohne Ergebnis im Verlauf."""
CANCEL_POLL_S = 0.2
"""So oft (Sekunden) prüfen Wächter und Lock-Wartezeit ``req.cancel``."""

UNREACHABLE_HINT = (
    "LM Studio starten und den lokalen Server aktivieren "
    "(Developer → Start Server oder `lms server start`)."
)
NO_MODEL_HINT = "In LM Studio ein Modell laden oder in NoAssist ein Modell auswählen."


class _Cancelled(BaseException):
    """Bricht ``act()``/``respond()`` aus einem Callback heraus ab.

    Absichtlich ``BaseException``: das SDK fängt in Callbacks nur ``Exception``
    ab (``sdk_callback_invocation``) und lässt alles andere durch.
    """


# --- Verlauf: neutral -> lmstudio ----------------------------------------------


def _tool_content(text: str) -> str:
    """Tool-Ergebnis so kodieren, wie lmstudio es selbst tut (``json.dumps``)."""
    return json.dumps(text, ensure_ascii=False)


def _normalize_history(history: list[Message]) -> list[dict[str, Any]]:
    """Bereinigt einen (evtl. fremden) Verlauf für ``lms.Chat``.

    Ergebnis: Liste aus ``{"role": "user", "text"}``,
    ``{"role": "assistant", "text", "calls": [ToolCall]}`` und
    ``{"role": "tool", "id", "content"}``.

    - Leere Nutzer- und Assistentennachrichten entfallen.
    - Aufeinanderfolgende Assistentennachrichten werden zusammengeführt
      (lmstudio erlaubt keine direkt aufeinanderfolgenden Antworten).
    - Tool-Nachrichten ohne passenden offenen Aufruf werden verworfen.
    - Tool-Aufrufe ohne Ergebnis bekommen ``"Fehler: abgebrochen"``.
    """
    items: list[dict[str, Any]] = []
    pending: dict[str, str] = {}  # offene Aufrufe: id -> Name

    def flush_pending() -> None:
        for call_id in list(pending):
            items.append({"role": "tool", "id": call_id, "content": ABORTED_RESULT})
        pending.clear()

    for msg in history:
        if msg.role == "tool":
            if msg.tool_call_id and msg.tool_call_id in pending:
                pending.pop(msg.tool_call_id)
                items.append({"role": "tool", "id": msg.tool_call_id, "content": msg.content or ""})
            else:
                log.debug("Tool-Nachricht ohne passenden Aufruf verworfen: %s", msg.tool_call_id)
            continue

        flush_pending()
        if msg.role == "user":
            if msg.content and msg.content.strip():
                items.append({"role": "user", "text": msg.content})
            continue

        # Assistent
        calls: list[ToolCall] = []
        seen: set[str] = set()
        for tc in msg.tool_calls:
            call_id = tc.id or "call_" + uuid.uuid4().hex[:24]
            if call_id in seen:
                continue
            seen.add(call_id)
            args = tc.arguments if isinstance(tc.arguments, dict) else {}
            calls.append(ToolCall(id=call_id, name=tc.name, arguments=dict(args)))
        text = msg.content or ""
        if not text.strip() and not calls:
            continue
        last = items[-1] if items else None
        if last is not None and last["role"] == "assistant":
            # Nur möglich, wenn die vorige Antwort keine Tool-Aufrufe hatte.
            if text.strip():
                last["text"] = f"{last['text']}\n\n{text}" if last["text"].strip() else text
            last["calls"].extend(calls)
        else:
            items.append({"role": "assistant", "text": text, "calls": calls})
        for tc in calls:
            pending[tc.id] = tc.name
    flush_pending()
    return items


def build_chat(system_prompt: str, history: list[Message]) -> lms.Chat:
    """Baut ein ``lms.Chat`` aus System-Prompt und neutralem Verlauf."""
    import lmstudio as lms

    chat = lms.Chat(system_prompt) if system_prompt and system_prompt.strip() else lms.Chat()
    for item in _normalize_history(history):
        role = item["role"]
        if role == "user":
            chat.add_user_message(item["text"])
        elif role == "assistant":
            requests: list[ToolCallRequestDataDict] = [
                {
                    "type": "toolCallRequest",
                    "toolCallRequest": {
                        "type": "function",
                        "id": tc.id,
                        "name": tc.name,
                        "arguments": tc.arguments,
                    },
                }
                for tc in item["calls"]
            ]
            chat.add_assistant_response(item["text"], tool_call_requests=requests)
        else:
            chat.add_tool_result(
                {"type": "toolCallResult", "toolCallId": item["id"], "content": _tool_content(item["content"])}
            )
    return chat


# --- Fehler --------------------------------------------------------------------


def _first_line(exc: BaseException) -> str:
    text = str(exc).strip()
    return text.splitlines()[0].strip() if text else type(exc).__name__


def _is_connection_error(exc: BaseException) -> bool:
    import lmstudio as lms

    if isinstance(exc, (ConnectionError, TimeoutError, OSError)):
        return True
    if isinstance(exc, (lms.LMStudioWebsocketError, lms.LMStudioChannelClosedError)):
        return True
    if isinstance(exc, lms.LMStudioClientError) and "not reachable" in str(exc):
        return True
    try:
        import httpx
    except ImportError:  # pragma: no cover - httpx ist Abhängigkeit von lmstudio
        return False
    return isinstance(exc, httpx.TransportError)


def _map_error(exc: BaseException) -> ProviderError:
    """Wandelt SDK-Fehler in ``ProviderError`` (Texte ohne Geheimnisse)."""
    import lmstudio as lms

    if isinstance(exc, ProviderError):
        return exc
    if _is_connection_error(exc):
        return ProviderError("LM Studio ist nicht erreichbar.", PROVIDER_UNREACHABLE, UNREACHABLE_HINT)
    if isinstance(exc, lms.LMStudioModelNotFoundError):
        return ProviderError("Modell nicht gefunden.", MODEL_NOT_FOUND, NO_MODEL_HINT)
    if isinstance(exc, lms.LMStudioPredictionError):
        return ProviderError(f"LM Studio meldet einen Fehler bei der Antwort: {_first_line(exc)}", BAD_REQUEST)
    if isinstance(exc, lms.LMStudioServerError):
        return ProviderError(f"LM Studio meldet einen Fehler: {_first_line(exc)}", PROVIDER_ERROR)
    if isinstance(exc, lms.LMStudioError):
        return ProviderError(f"Fehler im LM-Studio-SDK: {_first_line(exc)}", INTERNAL)
    return ProviderError(f"Unerwarteter Fehler: {type(exc).__name__}: {_first_line(exc)}", INTERNAL)


def _probe_host(host: str) -> bool:
    import lmstudio as lms

    try:
        return lms.Client.is_valid_api_host(host)
    except Exception:
        return False


# --- Abbruch -------------------------------------------------------------------


class _PredictionCanceller:
    """Bricht laufende Vorhersagen beim Server ab, sobald ``cancel`` gesetzt wird.

    Merkt sich dazu Endpoint und Kanal jeder Vorhersage, indem es
    ``_create_channel`` der LLM-Session für die Dauer des Turns umhüllt (die
    Session gehört exklusiv diesem Turn, siehe Prediction-Lock). Beim Abbruch
    passiert dasselbe wie in ``PredictionStream.cancel()``: Endpoint als
    abgebrochen markieren, Abbruch-Nachricht an den Server senden. Fehlen die
    internen Attribute (andere SDK-Version), bleibt nur der Abbruch über die
    Callbacks.
    """

    def __init__(self, session: Any, cancel: threading.Event):
        self._session = session
        self._cancel = cancel
        self._lock = threading.Lock()
        self._active: list[tuple[Any, Any]] = []
        self._stop = threading.Event()
        self._patched = False
        self._thread: threading.Thread | None = None

    def __enter__(self) -> _PredictionCanceller:
        original: Callable[[Any], AbstractContextManager[Any]] | None = getattr(
            self._session, "_create_channel", None
        )
        if callable(original):
            tracker = self

            @contextmanager
            def tracked(endpoint: Any) -> Generator[Any, None, None]:
                with original(endpoint) as channel:
                    entry = (endpoint, channel)
                    with tracker._lock:
                        tracker._active.append(entry)
                    if tracker._cancel.is_set():
                        tracker._cancel_entry(entry)
                    try:
                        yield channel
                    finally:
                        with tracker._lock:
                            if entry in tracker._active:
                                tracker._active.remove(entry)

            try:
                self._session._create_channel = tracked
                self._patched = True
            except Exception:
                log.debug("Abbruch über den Kanal nicht möglich", exc_info=True)
        self._thread = threading.Thread(target=self._watch, name="lmstudio-cancel", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._stop.set()
        if self._patched:
            try:
                del self._session._create_channel
            except AttributeError:
                pass
        if self._thread is not None:
            self._thread.join(timeout=2)

    def _watch(self) -> None:
        while not self._stop.is_set():
            if self._cancel.wait(CANCEL_POLL_S):
                with self._lock:
                    entries = list(self._active)
                for entry in entries:
                    self._cancel_entry(entry)
                return

    @staticmethod
    def _cancel_entry(entry: tuple[Any, Any]) -> None:
        endpoint, channel = entry
        try:
            mark = getattr(endpoint, "mark_cancelled", None)
            if callable(mark):
                mark()
            channel.cancel()
        except Exception:
            log.debug("Vorhersage konnte nicht abgebrochen werden", exc_info=True)


# --- Provider ------------------------------------------------------------------


class LMStudioProvider(Provider):
    """LM Studio über das native lmstudio-SDK.

    ``api_host``: z. B. ``"127.0.0.1:41343"``; ``None`` = Standard-Ports des
    SDK durchsuchen.
    """

    type = "lmstudio"

    def __init__(self, id: str = "lmstudio", name: str = "LM Studio", api_host: str | None = None):
        super().__init__(id, name)
        self.api_host = api_host
        self._predict_lock = threading.Lock()  # Vorhersagen (Client für Turns)
        self._meta_lock = threading.Lock()  # Modellliste und Status (eigener Client)
        self._clients_lock = threading.Lock()
        self._chat_client: lms.Client | None = None
        self._meta_client: lms.Client | None = None

    # -- Clients --

    def _find_host(self) -> str | None:
        """Erreichbaren API-Host ermitteln (nur HTTP-Probe, ohne Websocket)."""
        import lmstudio as lms

        try:
            if self.api_host:
                return self.api_host if lms.Client.is_valid_api_host(self.api_host) else None
            hosts = list(getattr(lms.Client, "_iter_default_api_hosts", lambda: [])())
            if not hosts:
                return lms.Client.find_default_local_api_host()
            # Parallel proben: nacheinander dauert es ohne LM Studio unter Windows ~7 s.
            from concurrent.futures import ThreadPoolExecutor

            pool = ThreadPoolExecutor(len(hosts), thread_name_prefix="lmstudio-probe")
            try:
                futures = [pool.submit(_probe_host, h) for h in hosts]
                # Reihenfolge der Ports beibehalten, aber nicht auf spätere Proben warten.
                for host, fut in zip(hosts, futures):
                    if fut.result():
                        return host
                return None
            finally:
                pool.shutdown(wait=False, cancel_futures=True)
        except Exception:
            log.debug("Probe von LM Studio fehlgeschlagen", exc_info=True)
            return None

    def _client(self, kind: str) -> lms.Client:
        """Client ``"chat"`` oder ``"meta"`` holen bzw. neu verbinden."""
        import lmstudio as lms

        attr = "_chat_client" if kind == "chat" else "_meta_client"
        with self._clients_lock:
            client = getattr(self, attr)
            if client is not None:
                return client
        host = self._find_host()
        if host is None:
            raise ProviderError("LM Studio ist nicht erreichbar.", PROVIDER_UNREACHABLE, UNREACHABLE_HINT)
        client = lms.Client(host)
        with self._clients_lock:
            setattr(self, attr, client)
        return client

    def _drop_client(self, kind: str) -> None:
        """Client verwerfen (z. B. nach Verbindungsabbruch); nächster Aufruf verbindet neu."""
        attr = "_chat_client" if kind == "chat" else "_meta_client"
        with self._clients_lock:
            client = getattr(self, attr)
            setattr(self, attr, None)
        if client is not None:
            try:
                client.close()
            except Exception:
                log.debug("Client konnte nicht geschlossen werden", exc_info=True)

    def close(self) -> None:
        self._drop_client("chat")
        self._drop_client("meta")

    # -- Modelle und Status --

    def list_models(self) -> list[ModelInfo]:
        """Geladene LLMs zuerst (``loaded=True``), danach heruntergeladene (``loaded=False``)."""
        with self._meta_lock:
            try:
                client = self._client("meta")
                loaded = list(client.llm.list_loaded())
                downloaded = list(client.llm.list_downloaded())
            except Exception as e:
                err = _map_error(e)
                if err.code == PROVIDER_UNREACHABLE:
                    self._drop_client("meta")
                raise err from e

        labels: dict[str, str] = {}
        for d in downloaded:
            labels[d.model_key] = getattr(d.info, "display_name", None) or d.model_key
        counts: dict[str, int] = {}
        for label in labels.values():
            counts[label] = counts.get(label, 0) + 1
        for key, label in labels.items():
            if counts[label] > 1:  # gleiche Anzeigenamen (z. B. Quantisierungen) unterscheidbar machen
                labels[key] = f"{label} ({key})"

        result: list[ModelInfo] = []
        loaded_keys: set[str] = set()
        for handle in loaded:
            ident = handle.identifier
            loaded_keys.add(ident)
            label = labels.get(ident, ident)
            result.append(ModelInfo(id=ident, label=label, loaded=True))
        for d in sorted(downloaded, key=lambda m: labels[m.model_key].lower()):
            if d.model_key in loaded_keys:
                continue
            result.append(ModelInfo(id=d.model_key, label=labels[d.model_key], loaded=False))
        return result

    def status(self) -> ProviderStatus:
        host = self._find_host()
        if host is None:
            return ProviderStatus(ok=False, message="LM Studio ist nicht erreichbar.", hint=UNREACHABLE_HINT)
        with self._meta_lock:
            try:
                loaded = [h.identifier for h in self._client("meta").llm.list_loaded()]
            except Exception as e:
                self._drop_client("meta")
                err = _map_error(e)
                return ProviderStatus(ok=False, message=err.message, hint=err.hint, details={"api_host": host})
        if loaded:
            message = f"Verbunden, {len(loaded)} Modell(e) geladen."
            hint = None
        else:
            message = "Verbunden, kein Modell geladen."
            hint = "Ein ausgewähltes Modell wird beim ersten Senden geladen."
        return ProviderStatus(ok=True, message=message, hint=hint, details={"api_host": host, "loaded_models": loaded})

    # -- Turn --

    def run_turn(self, req: TurnRequest, emit: Emit) -> None:
        if req.cancel.is_set():
            return
        try:
            chat = build_chat(req.system_prompt, req.history)
        except Exception as e:
            raise ProviderError(
                f"Verlauf kann nicht an LM Studio übergeben werden: {_first_line(e)}", BAD_REQUEST
            ) from e

        if not self._acquire(req, emit):
            return
        try:
            self._run_locked(req, chat, emit)
        finally:
            self._predict_lock.release()

    def _acquire(self, req: TurnRequest, emit: Emit) -> bool:
        """Prediction-Lock holen; ``False`` bei Abbruch während des Wartens."""
        if self._predict_lock.acquire(blocking=False):
            return True
        emit(ev_status("Wartet auf laufende Anfrage…"))
        while not req.cancel.is_set():
            if self._predict_lock.acquire(timeout=CANCEL_POLL_S):
                return True
        return False

    def _resolve_model(self, client: lms.Client, model: str | None, emit: Emit) -> lms.LLM:
        loaded = [h.identifier for h in client.llm.list_loaded()]
        if not model:
            if not loaded:
                raise ProviderError("In LM Studio ist kein Modell geladen.", MODEL_NOT_FOUND, NO_MODEL_HINT)
            return client.llm.model(loaded[0])
        if model not in loaded:
            emit(ev_status("Modell wird geladen…"))
        return client.llm.model(model)

    def _run_locked(self, req: TurnRequest, chat: lms.Chat, emit: Emit) -> None:
        turn = _TurnState(self, req, emit)
        try:
            client = self._client("chat")
            llm = self._resolve_model(client, req.model, emit)
            turn.model_id = getattr(llm, "identifier", None) or req.model
            if req.cancel.is_set():
                return
            tools = req.tools.for_lmstudio(emit) if req.tools else []
            with _PredictionCanceller(client.llm, req.cancel):
                if tools:
                    llm.act(
                        chat,
                        tools,
                        max_prediction_rounds=MAX_PREDICTION_ROUNDS,
                        on_message=turn.on_message,
                        on_prediction_fragment=turn.on_fragment,
                        on_prompt_processing_progress=turn.on_progress,
                        on_round_start=turn.on_round_boundary,
                        on_round_end=turn.on_round_boundary,
                        handle_invalid_tool_request=turn.on_invalid_tool_request,
                    )
                else:
                    result = llm.respond(
                        chat,
                        on_prediction_fragment=turn.on_fragment,
                        on_prompt_processing_progress=turn.on_progress,
                    )
                    turn.finish_text(result.content)
            turn.flush_partial()
        except _Cancelled:
            turn.flush_partial()
        except ProviderError as e:
            if e.code == PROVIDER_UNREACHABLE:
                self._drop_client("chat")
            raise
        except Exception as e:
            if req.cancel.is_set():
                log.debug("Fehler nach Abbruch ignoriert", exc_info=True)
                turn.flush_partial()
                return
            err = _map_error(e)
            if err.code == PROVIDER_UNREACHABLE:
                self._drop_client("chat")
            raise err from e


class _TurnState:
    """Callbacks und Puffer eines einzelnen Turns."""

    def __init__(self, provider: LMStudioProvider, req: TurnRequest, emit: Emit):
        self.provider = provider
        self.req = req
        self.emit = emit
        self.model_id: str | None = req.model
        self._text: list[str] = []  # sichtbarer Text der laufenden Runde
        self._saw_fragment = False
        self._calls: list[ToolCall] = []  # Aufrufe der letzten Assistentennachricht (lmstudio-IDs)

    # -- Hilfen --

    def _check(self) -> None:
        if self.req.cancel.is_set():
            raise _Cancelled()

    def _message(self, **kwargs: Any) -> Message:
        return Message(provider_id=self.provider.id, model=self.model_id, **kwargs)

    def _take_text(self, fallback: str) -> str:
        text = "".join(self._text) if self._saw_fragment else fallback
        self._text.clear()
        self._saw_fragment = False
        return text

    # -- SDK-Callbacks --

    def on_fragment(self, fragment: Any, *_round: Any) -> None:
        self._check()
        content = getattr(fragment, "content", "") or ""
        kind = getattr(fragment, "reasoning_type", "none")
        # Ab dem ersten Fragment zählt nur der gepufferte sichtbare Text, nicht der
        # Text der SDK-Nachricht (der enthält auch das Reasoning).
        self._saw_fragment = True
        if kind == "reasoning":
            if content:
                self.emit(ev_reasoning(content))
            return
        if kind in ("reasoningStartTag", "reasoningEndTag"):
            return
        if content:
            self._text.append(content)
            self.emit(ev_delta(content))

    def on_progress(self, *_args: Any) -> None:
        self._check()

    def on_round_boundary(self, _round_index: int) -> None:
        self._check()

    def on_invalid_tool_request(self, exc: Exception, _request: Any) -> str | None:
        """Fehlertexte des SDK (z. B. unbekanntes Tool) mit ``Fehler:`` versehen."""
        return f"{ERROR_PREFIX} {_first_line(exc)}"

    def on_message(self, msg: Any) -> None:
        role = getattr(msg, "role", None)
        if role == "assistant":
            self._on_assistant(msg)
        elif role == "tool":
            self._on_tool_results(msg)

    def _on_assistant(self, msg: Any) -> None:
        texts: list[str] = []
        calls: list[ToolCall] = []
        for part in getattr(msg, "content", None) or []:
            request = getattr(part, "tool_call_request", None)
            if request is not None:
                call_id = getattr(request, "id", None) or "call_" + uuid.uuid4().hex[:24]
                args = getattr(request, "arguments", None) or {}
                calls.append(ToolCall(id=call_id, name=request.name, arguments=_plain(args)))
            else:
                text = getattr(part, "text", None)
                if text:
                    texts.append(text)
        content = self._take_text("".join(texts))
        self._calls = calls
        self.emit(ev_message(self._message(role="assistant", content=content, tool_calls=calls)))

    def _on_tool_results(self, msg: Any) -> None:
        calls = self._calls
        by_id = {c.id: c for c in calls}
        unmatched = list(calls)
        for part in getattr(msg, "content", None) or []:
            call_id = getattr(part, "tool_call_id", None)
            call = by_id.get(call_id) if call_id else None
            if call is None:
                call = unmatched[0] if unmatched else None
            if call is not None and call in unmatched:
                unmatched.remove(call)
            text = _decode_result(getattr(part, "content", ""))
            self.emit(
                ev_message(
                    self._message(
                        role="tool",
                        content=text,
                        tool_call_id=call.id if call else (call_id or ""),
                        name=call.name if call else None,
                        is_error=text.startswith(ERROR_PREFIX),
                    )
                )
            )
        self._calls = []

    # -- Abschluss --

    def finish_text(self, fallback: str) -> None:
        """Antwort aus ``respond()`` als Nachricht senden."""
        content = self._take_text(fallback or "")
        if content.strip():
            self.emit(ev_message(self._message(role="assistant", content=content)))

    def flush_partial(self) -> None:
        """Bei Abbruch: bereits gestreamten, noch nicht gesendeten Text als Nachricht sichern."""
        if self._saw_fragment and "".join(self._text).strip():
            self.emit(ev_message(self._message(role="assistant", content=self._take_text(""))))
        else:
            self._text.clear()
            self._saw_fragment = False


def _plain(value: Any) -> Any:
    """Mappings/Sequenzen aus dem SDK in einfache JSON-Typen wandeln."""
    if isinstance(value, dict) or hasattr(value, "items"):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def _decode_result(content: Any) -> str:
    """lmstudio kodiert Tool-Ergebnisse per ``json.dumps``; hier zurück in Text."""
    if not isinstance(content, str):
        return json.dumps(content, ensure_ascii=False, default=str)
    try:
        decoded = json.loads(content)
    except ValueError:
        return content  # z. B. Fehlertexte des SDK, die nicht kodiert sind
    if isinstance(decoded, str):
        return decoded
    return content


if __name__ == "__main__":
    import sys
    import time

    from chat_types import Event
    from toolset import ToolSet
    from tools import get_tools

    def show(ev: Event) -> None:
        if ev["type"] in ("delta", "reasoning"):
            print(ev["text"], end="", flush=True)
        else:
            print("\n  [event]", json.dumps(ev, ensure_ascii=False)[:300])

    provider = LMStudioProvider()
    st = provider.status()
    print("Status:", st)
    if not st.ok:
        t0 = time.perf_counter()
        try:
            provider.run_turn(
                TurnRequest(history=[Message(role="user", content="Hallo")], model=None, system_prompt="", tools=ToolSet()),
                show,
            )
        except ProviderError as e:
            print(f"ProviderError {e.code}: {e.message} ({time.perf_counter() - t0:.1f} s)")
        sys.exit(0)

    models = provider.list_models()
    for m in models:
        print(f"  {'*' if m.loaded else ' '} {m.id}  ({m.label})")
    model = sys.argv[1] if len(sys.argv) > 1 else next((m.id for m in models if m.loaded), None)
    print("Modell:", model)

    tools = ToolSet.from_any(get_tools())
    history = [
        Message(role="user", content="Was ist 2 + 3?"),
        Message(role="assistant", tool_calls=[ToolCall(id="alt_1", name="add", arguments={"a": 2, "b": 3})]),
        Message(role="tool", content="5", tool_call_id="alt_1", name="add"),
        Message(role="tool", content="verwaist", tool_call_id="gibt_es_nicht", name="add"),
        Message(role="assistant", content="Das Ergebnis ist 5."),
        Message(role="assistant", tool_calls=[ToolCall(id="alt_2", name="get_datetime", arguments={})]),
        Message(role="user", content="Wie spät ist es?"),
    ]
    print("\n--- Turn ---")
    provider.run_turn(
        TurnRequest(history=history, model=model, system_prompt="Antworte knapp auf Deutsch.", tools=tools), show
    )
    print()
    provider.close()
