"""Flask-Server von NoAssist: REST-API, Server-Sent Events, statische Dateien.

- ``create_app(service, *, port, dev, boot_id)`` baut die Flask-App mit allen
  Routen aus dem API-Schema (siehe Plan). ``service`` ist ein ``ChatService``
  (oder ein gleichwertiger Ersatz, z. B. für Tests).
- ``AppServer(port, dev, data_dir)`` baut ``ChatStore``, ``ConfigStore`` und
  ``ChatService`` aus ``data_dir``, prüft den Port und startet einen
  werkzeug-Server fest auf 127.0.0.1.

Sicherheitsregeln (gelten für alle Routen):

- ``Host`` muss ``127.0.0.1:<port>`` oder ``localhost:<port>`` sein, sonst 403
  (Schutz gegen DNS-Rebinding).
- Ein vorhandener ``Origin`` muss zum eigenen Server gehören, sonst 403.
- Alles außer GET/HEAD braucht ``Content-Type: application/json``, sonst 415
  (ein leerer Body gilt als ``{}``).
- Keine CORS-Header, kein Werkzeug-Debugger, Request-Bodies werden nie geloggt.
- Felder ``api_key`` werden aus allen JSON-Antworten entfernt (``has_api_key``
  statt dessen). Bekannte Geheimnisse ab 8 Zeichen (``storage.MIN_SECRET_LEN``)
  werden in Fehlermeldungen, Status- und Hinweistexten (Provider-Status,
  Modellliste, MCP-Fehler, SSE-Events ``error``/``status``) durch ``***``
  ersetzt. Chat-Inhalte (Titel, Nachrichten, Text-Deltas) werden nicht
  geschwärzt; kurze Platzhalter-Keys wie ``"ollama"`` verfälschen sie so nicht.

Wiederholen: ``POST /api/chats/<id>/messages`` mit ``{"retry": true,
"user_message_id": "<ID der letzten User-Nachricht>", "provider_id"?, "model"?}``.
Fehlt ``user_message_id``: 400. Läuft ein Turn: 409 ``busy``. Ist die Nachricht
nicht mehr die letzte oder schon fertig beantwortet: 409 ``retry_stale`` (der
Client lädt den Chat neu).

Demo (aus ``src``): ``python -m server`` startet den Server mit einem
Fake-ChatService (ohne Provider) auf http://127.0.0.1:8765/.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import mimetypes
import queue
import socket
import sys
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from flask import Flask, Response, request, send_from_directory
from werkzeug.exceptions import HTTPException
from werkzeug.serving import make_server

HOST = "127.0.0.1"
DEFAULT_PORT = 8765

SRC_DIR = Path(__file__).resolve().parent
STATIC_DIR = SRC_DIR / "web" / "static"
DEFAULT_DATA_DIR = SRC_DIR.parent / "data"

SSE_RETRY_MS = 2000
SSE_PING_S = 15.0
_SSE_POLL_S = 1.0  # so oft prüfen Streams, ob der Server herunterfährt

_SAFE_METHODS = ("GET", "HEAD")

_REDACTED_EVENT_FIELDS: dict[str, tuple[str, ...]] = {"error": ("message", "hint"), "status": ("text",)}
"""Textfelder von SSE-Events, in denen bekannte Geheimnisse geschwärzt werden."""

# Unter Windows liefert die Registry für .js teils "text/plain"; ES-Module brauchen JavaScript.
mimetypes.add_type("text/javascript", ".js")
mimetypes.add_type("text/javascript", ".mjs")
mimetypes.add_type("text/css", ".css")

log = logging.getLogger("noassist.server")


# --- Fehler ---------------------------------------------------------------------

class ApiError(Exception):
    """Fehler, der als ``{"error": {"code", "message"}}`` beantwortet wird."""

    def __init__(self, status: int, code: str, message: str, hint: str | None = None):
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.hint = hint


class PortInUseError(RuntimeError):
    """Der gewünschte Port ist bereits belegt."""


def _class_names(exc: BaseException) -> set[str]:
    return {cls.__name__ for cls in type(exc).__mro__}


def _map_service_error(exc: Exception) -> ApiError | None:
    """Ordnet Ausnahmen des ChatService HTTP-Fehlern zu (``None`` = unbekannt, also 500).

    Erkannt wird über den Klassennamen in der MRO, damit ``chat_service`` hier
    nicht importiert werden muss: ``NotFound``/``KeyError``/``LookupError`` → 404,
    ``RetryStale`` → 409 ``retry_stale``, ``Busy``/``Conflict`` → 409 ``busy``,
    ``ProviderError`` → 400 (mit dessen Code), ``ValueError``/``TypeError`` → 400.
    """
    names = _class_names(exc)
    message = str(exc.args[0]) if exc.args else ""
    if "ProviderError" in names:
        return ApiError(400, getattr(exc, "code", "provider_error") or "provider_error",
                        getattr(exc, "message", message) or "Provider-Fehler.", getattr(exc, "hint", None))
    if "RetryStale" in names:
        return ApiError(409, "retry_stale", message or "Die Nachricht kann nicht mehr wiederholt werden. "
                                                       "Bitte den Chat neu laden.")
    if names & {"Busy", "Conflict"}:
        return ApiError(409, "busy", message or "In diesem Chat läuft bereits eine Antwort.")
    if names & {"NotFound", "KeyError", "LookupError"}:
        return ApiError(404, "not_found", message if "NotFound" in names and message else "Nicht gefunden.")
    if names & {"ValueError", "TypeError"}:
        return ApiError(400, "bad_request", message or "Ungültige Anfrage.")
    return None


# --- Hilfsfunktionen ------------------------------------------------------------

def _jsonable(value: Any) -> Any:
    """Wandelt Service-Ergebnisse (Dicts, Listen, ``Message``, Dataclasses) in JSON-taugliche Werte."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _jsonable(to_dict())
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _jsonable(dataclasses.asdict(value))
    return value


def _strip_secrets(value: Any) -> Any:
    """Entfernt ``api_key``-Felder rekursiv; ein Dict mit ``api_key`` bekommt ``has_api_key``.

    Gilt für jede JSON-Antwort, also auch für Chat-Inhalte: Ein Tool-Argument
    oder Schema-Feld namens ``api_key`` (z. B. in ``tool_calls[].arguments``
    in ``GET /api/chats/<id>`` oder in ``GET /api/tools``) erscheint dort
    ebenfalls nicht, sondern als ``has_api_key``. Das ist gewollt (lieber zu
    viel als zu wenig entfernt); SSE-Events sind davon nicht betroffen.
    """
    if isinstance(value, dict):
        result = {k: _strip_secrets(v) for k, v in value.items() if k != "api_key"}
        if "api_key" in value:
            result.setdefault("has_api_key", bool(value.get("api_key")))
        return result
    if isinstance(value, list):
        return [_strip_secrets(v) for v in value]
    return value


def _sse(event: str | None, data: Any, event_id: int | None = None) -> str:
    """Formatiert ein SSE-Event (``data`` als JSON in einer Zeile)."""
    lines = []
    if event_id is not None:
        lines.append(f"id: {event_id}")
    if event:
        lines.append(f"event: {event}")
    lines.append("data: " + json.dumps(data, ensure_ascii=False, separators=(",", ":"), default=str))
    return "\n".join(lines) + "\n\n"


def _parse_seq(raw: str | None) -> int | None:
    if raw is None:
        return None
    try:
        value = int(str(raw).strip())
    except ValueError:
        return None
    return value if value >= 0 else None


class DevHub:
    """Verteilt ``reload``-Events des Dev-Modus an alle verbundenen ``/api/dev/events``-Streams."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._subscribers: list[queue.Queue[dict[str, Any]]] = []

    def subscribe(self) -> queue.Queue[dict[str, Any]]:
        q: queue.Queue[dict[str, Any]] = queue.Queue()
        with self._lock:
            self._subscribers.append(q)
        return q

    def unsubscribe(self, q: queue.Queue[dict[str, Any]]) -> None:
        with self._lock:
            if q in self._subscribers:
                self._subscribers.remove(q)

    def notify_reload(self, files: list[str], css_only: bool) -> None:
        """Callback für ``devreload.watch_static``."""
        data = {"files": list(files), "css_only": bool(css_only)}
        with self._lock:
            subscribers = list(self._subscribers)
        for q in subscribers:
            q.put(data)


# --- App ------------------------------------------------------------------------

def create_app(service: Any, *, port: int, dev: bool, boot_id: str) -> Flask:
    """Baut die Flask-App mit allen Routen aus dem API-Schema.

    ``service`` muss die Schnittstelle von ``ChatService`` bieten (siehe Plan).
    Zusätzlich genutzt, falls vorhanden: ``service.config`` (``ConfigStore``,
    für ``redact`` und ``cli_path`` beim Claude-Login) und ``service.redact(text)``
    (z. B. Keys aus Umgebungsvariablen über ``provider.redact``).

    In ``app.extensions["noassist"]`` liegen ``dev_hub`` (``DevHub``) und
    ``stop`` (``threading.Event``; gesetzt = laufende Streams beenden sich).
    """
    app = Flask(__name__, static_folder=None)
    app.debug = False
    app.config.update(PROPAGATE_EXCEPTIONS=False, TRAP_HTTP_EXCEPTIONS=False, JSON_SORT_KEYS=False)
    app.json.sort_keys = False  # type: ignore[attr-defined]
    app.json.ensure_ascii = False  # type: ignore[attr-defined]

    hub = DevHub()
    stop = threading.Event()
    app.extensions["noassist"] = {"dev_hub": hub, "stop": stop, "service": service}

    allowed_hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
    allowed_origins = {f"http://{h}" for h in allowed_hosts}

    # -- Schwärzen von Geheimnissen (nur Fehler-, Status- und Hinweistexte, nie Chat-Inhalte) --

    def redact(text: str) -> str:
        if not text or not isinstance(text, str):
            return text
        config = getattr(service, "config", None)
        config_redact: Callable[[str], str] | None = getattr(config, "redact", None)
        if callable(config_redact):
            try:
                text = config_redact(text)
            except Exception:  # noqa: BLE001 - Schwärzen darf nie selbst scheitern
                pass
        service_redact: Callable[[str], str] | None = getattr(service, "redact", None)
        if callable(service_redact):
            try:
                text = service_redact(text)
            except Exception:  # noqa: BLE001
                pass
        return text

    def redact_fields(data: Any, *keys: str) -> Any:
        """Kopie von ``data`` (Dict), in der die Textfelder ``keys`` geschwärzt sind."""
        if not isinstance(data, dict):
            return data
        out = dict(data)
        for key in keys:
            if isinstance(out.get(key), str):
                out[key] = redact(out[key])
        return out

    def error_response(status: int, code: str, message: str, hint: str | None = None) -> Response:
        err: dict[str, Any] = {"code": code, "message": redact(message)}
        if hint:
            err["hint"] = redact(hint)
        return app.response_class(
            json.dumps({"error": err}, ensure_ascii=False), status=status, mimetype="application/json"
        )

    def ok(data: Any, status: int = 200) -> Response:
        body = json.dumps(_strip_secrets(_jsonable(data)), ensure_ascii=False, default=str)
        return app.response_class(body, status=status, mimetype="application/json")

    def no_content() -> Response:
        return app.response_class(status=204)

    def body() -> dict[str, Any]:
        """JSON-Body der Anfrage; leerer Body = ``{}``. Kein Objekt → 400."""
        raw = request.get_data(cache=True)
        if not raw.strip():
            return {}
        try:
            data = json.loads(raw)
        except (ValueError, UnicodeDecodeError):
            raise ApiError(400, "bad_request", "Der Request-Body ist kein gültiges JSON.") from None
        if not isinstance(data, dict):
            raise ApiError(400, "bad_request", "Der Request-Body muss ein JSON-Objekt sein.")
        return data

    def call(func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Ruft eine Service-Methode auf und übersetzt erwartbare Ausnahmen in ``ApiError``."""
        try:
            return func(*args, **kwargs)
        except ApiError:
            raise
        except Exception as exc:
            mapped = _map_service_error(exc)
            if mapped is None:
                raise
            raise mapped from exc

    def opt_str(data: dict[str, Any], key: str) -> str | None:
        value = data.get(key)
        if value is not None and not isinstance(value, str):
            raise ApiError(400, "bad_request", f"Feld '{key}' muss ein String oder null sein.")
        return value or None

    # -- Sicherheit --

    @app.before_request
    def _security() -> Response | None:
        host = (request.headers.get("Host") or "").strip().lower()
        if host not in allowed_hosts:
            return error_response(403, "forbidden_host", "Unzulässiger Host-Header.")
        origin = request.headers.get("Origin")
        if origin is not None and origin.strip().lower().rstrip("/") not in allowed_origins:
            return error_response(403, "forbidden_origin", "Anfragen von fremden Seiten sind nicht erlaubt.")
        if request.method not in _SAFE_METHODS and request.mimetype != "application/json":
            return error_response(415, "unsupported_media_type",
                                  "Content-Type muss application/json sein.")
        return None

    @app.after_request
    def _after(response: Response) -> Response:
        # Kein pauschales Schwärzen des Bodys: das verfälschte Titel/Nachrichten (und mit
        # kurzen Keys sogar das JSON selbst). Geschwärzt wird gezielt (error_response, redact_fields, SSE).
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        return response

    # -- Fehlerbehandlung --

    @app.errorhandler(ApiError)
    def _api_error(exc: ApiError) -> Response:
        return error_response(exc.status, exc.code, exc.message, exc.hint)

    @app.errorhandler(HTTPException)
    def _http_error(exc: HTTPException) -> Response:
        status = exc.code or 500
        codes = {400: "bad_request", 403: "forbidden", 404: "not_found", 405: "method_not_allowed",
                 409: "conflict", 415: "unsupported_media_type"}
        messages = {404: "Nicht gefunden.", 405: "Methode nicht erlaubt."}
        response = error_response(status, codes.get(status, "http_error"),
                                  messages.get(status, exc.description or exc.name))
        valid = getattr(exc, "valid_methods", None)
        if status == 405 and valid:
            response.headers["Allow"] = ", ".join(valid)
        return response

    @app.errorhandler(Exception)
    def _internal_error(exc: Exception) -> Response:
        # Nur Typ und geschwärzte Meldung loggen, nie den Request-Body.
        log.error("Interner Fehler bei %s %s: %s: %s", request.method, request.path,
                  type(exc).__name__, redact(str(exc)))
        return error_response(500, "internal", f"Interner Fehler: {type(exc).__name__}: {exc}")

    # -- Statische Dateien --

    def static_response(path: str) -> Response:
        response = send_from_directory(STATIC_DIR, path, max_age=0)
        if dev:
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/")
    def index() -> Response:
        response = static_response("index.html")
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/static/<path:path>")
    def static_file(path: str) -> Response:
        return static_response(path)

    # -- Allgemein --

    @app.get("/api/health")
    def health() -> Response:
        return ok({"app": "noassist", "boot_id": boot_id, "dev": bool(dev)})

    @app.get("/api/settings")
    def get_settings() -> Response:
        return ok(call(service.get_settings))

    @app.put("/api/settings")
    def put_settings() -> Response:
        return ok(call(service.update_settings, body()))

    # -- Provider --

    @app.get("/api/providers")
    def list_providers() -> Response:
        return ok(call(service.list_providers))

    @app.post("/api/providers")
    def add_provider() -> Response:
        return ok(call(service.add_provider, body()), 201)

    @app.post("/api/providers/claude/login")
    def claude_login() -> Response:
        body()  # Body prüfen (leer oder {})
        try:
            from providers.claude_provider import start_login
        except ImportError as exc:
            raise ApiError(501, "not_implemented",
                           f"Die Claude-Anmeldung ist nicht verfügbar (claude_provider fehlt: {exc}).") from exc
        cli_path = None
        config = getattr(service, "config", None)
        for p in (getattr(config, "data", None) or {}).get("providers") or []:
            if isinstance(p, dict) and p.get("id") == "claude":
                cli_path = p.get("cli_path") or None
        call(start_login, cli_path)
        return ok({"started": True}, 202)

    @app.put("/api/providers/<pid>")
    def update_provider(pid: str) -> Response:
        return ok(call(service.update_provider, pid, body()))

    @app.delete("/api/providers/<pid>")
    def delete_provider(pid: str) -> Response:
        body()
        call(service.delete_provider, pid)
        return no_content()

    @app.get("/api/providers/<pid>/models")
    def provider_models(pid: str) -> Response:
        return ok(redact_fields(call(service.provider_models, pid), "error"))

    @app.get("/api/providers/<pid>/status")
    def provider_status(pid: str) -> Response:
        return ok(redact_fields(call(service.provider_status, pid), "message", "hint"))

    # -- Tools --

    @app.get("/api/tools")
    def list_tools() -> Response:
        data = call(service.list_tools)
        if isinstance(data, dict) and isinstance(data.get("mcp_servers"), list):
            data = {**data, "mcp_servers": [redact_fields(s, "error") for s in data["mcp_servers"]]}
        return ok(data)

    # -- Chats --

    def chat_out(chat: Any) -> dict[str, Any]:
        """Chat für die Antwort: ohne ``provider_state``; ``active_turn_id`` nur bei laufendem Turn."""
        data = _jsonable(chat)
        if not isinstance(data, dict):
            raise ApiError(500, "internal", "Ungültige Chat-Daten vom Service.")
        data.pop("provider_state", None)
        return data

    @app.get("/api/chats")
    def list_chats() -> Response:
        return ok([chat_out(c) for c in call(service.list_chats) or []])

    @app.post("/api/chats")
    def create_chat() -> Response:
        data = body()
        chat = call(service.create_chat, provider_id=opt_str(data, "provider_id"),
                    model=opt_str(data, "model"), title=opt_str(data, "title"))
        return ok(chat_out(chat), 201)

    @app.get("/api/chats/<chat_id>")
    def get_chat(chat_id: str) -> Response:
        chat = call(service.get_chat, chat_id)
        if chat is None:
            raise ApiError(404, "not_found", "Chat nicht gefunden.")
        data = chat_out(chat)
        tid = data.get("active_turn_id")
        if tid:
            turn = service.get_turn(tid)
            if turn is None or getattr(turn, "finished", False):
                tid = None
        data["active_turn_id"] = tid or None
        data.setdefault("messages", [])
        return ok(data)

    @app.patch("/api/chats/<chat_id>")
    def update_chat(chat_id: str) -> Response:
        data = body()
        allowed = {k: data[k] for k in ("title", "provider_id", "model") if k in data}
        for key in allowed:
            opt_str(allowed, key)
        chat = call(service.update_chat, chat_id, allowed)
        if chat is None:
            raise ApiError(404, "not_found", "Chat nicht gefunden.")
        return ok(chat_out(chat))

    @app.delete("/api/chats/<chat_id>")
    def delete_chat(chat_id: str) -> Response:
        body()
        if call(service.delete_chat, chat_id) is False:
            raise ApiError(404, "not_found", "Chat nicht gefunden.")
        return no_content()

    @app.post("/api/chats/<chat_id>/messages")
    def send_message(chat_id: str) -> Response:
        """Neue Nachricht (``content``) oder Wiederholung (``retry`` + ``user_message_id``)."""
        data = body()
        retry = data.get("retry", False)
        if not isinstance(retry, bool):
            raise ApiError(400, "bad_request", "Feld 'retry' muss true oder false sein.")
        content = data.get("content")
        if content is None and retry:
            content = ""
        if not isinstance(content, str):
            raise ApiError(400, "bad_request", "Feld 'content' muss ein String sein.")
        if not retry and not content.strip():
            raise ApiError(400, "bad_request", "Die Nachricht ist leer.")
        kwargs: dict[str, Any] = {"provider_id": opt_str(data, "provider_id"), "model": opt_str(data, "model"),
                                  "retry": retry}
        if retry:  # ohne retry wird user_message_id ignoriert
            user_message_id = opt_str(data, "user_message_id")
            if not user_message_id or not user_message_id.strip():
                raise ApiError(400, "bad_request", "Feld 'user_message_id' fehlt: Zum Wiederholen die ID der "
                                                   "User-Nachricht angeben, deren Antwort neu erzeugt werden soll.")
            kwargs["user_message_id"] = user_message_id.strip()
        turn, user_message = call(service.start_turn, chat_id, content, **kwargs)
        return ok({"turn_id": turn.id, "user_message": user_message}, 202)

    # -- Turns (SSE) --

    @app.get("/api/turns/<turn_id>/events")
    def turn_events(turn_id: str) -> Response:
        turn = service.get_turn(turn_id)
        if turn is None:
            raise ApiError(404, "not_found", "Turn nicht gefunden oder abgelaufen.")
        # Last-Event-ID (automatisches Wiederverbinden) hat Vorrang vor ?after=
        after = _parse_seq(request.headers.get("Last-Event-ID"))
        if after is None:
            after = _parse_seq(request.args.get("after")) or 0

        first, finished = turn.events_after(after, 0)
        if finished and not first:
            # Nichts mehr zu senden; 204 beendet EventSource endgültig (kein Dauer-Neuverbinden).
            return no_content()

        def stream() -> Iterator[str]:
            seq = after
            pending: list[tuple[int, dict[str, Any]]] = list(first)
            done = finished
            yield f"retry: {SSE_RETRY_MS}\n\n"
            last_write = time.monotonic()
            while True:
                for seq_no, event in pending:
                    seq = max(seq, seq_no)
                    data = {k: v for k, v in event.items() if k != "type"}
                    kind = str(event.get("type") or "message")
                    if kind in _REDACTED_EVENT_FIELDS:  # nur Fehler-/Statustexte, nie Deltas oder Nachrichten
                        data = redact_fields(data, *_REDACTED_EVENT_FIELDS[kind])
                    yield _sse(kind, data, seq_no)
                    last_write = time.monotonic()
                    if kind == "done":
                        return
                if done or stop.is_set():
                    return
                pending, done = turn.events_after(seq, _SSE_POLL_S)
                if not pending and time.monotonic() - last_write >= SSE_PING_S:
                    yield ": ping\n\n"
                    last_write = time.monotonic()

        return sse_response(stream())

    @app.post("/api/turns/<turn_id>/cancel")
    def cancel_turn(turn_id: str) -> Response:
        body()
        if service.get_turn(turn_id) is None:
            raise ApiError(404, "not_found", "Turn nicht gefunden oder abgelaufen.")
        cancelled = bool(call(service.cancel_turn, turn_id))
        return ok({"cancelled": cancelled}, 202)

    # -- Dev-Events (SSE) --

    @app.get("/api/dev/events")
    def dev_events() -> Response:
        if not dev:
            raise ApiError(404, "not_found", "Nur im Dev-Modus verfügbar.")
        q = hub.subscribe()

        def stream() -> Iterator[str]:
            try:
                yield f"retry: {SSE_RETRY_MS}\n\n"
                yield _sse("hello", {"boot_id": boot_id})
                last_write = time.monotonic()
                while not stop.is_set():
                    try:
                        data = q.get(timeout=_SSE_POLL_S)
                    except queue.Empty:
                        if time.monotonic() - last_write >= SSE_PING_S:
                            yield ": ping\n\n"
                            last_write = time.monotonic()
                        continue
                    yield _sse("reload", data)
                    last_write = time.monotonic()
            finally:
                hub.unsubscribe(q)

        return sse_response(stream())

    def sse_response(gen: Iterator[str]) -> Response:
        response = Response(gen, mimetype="text/event-stream")
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Accel-Buffering"] = "no"
        return response

    return app


# --- Server ---------------------------------------------------------------------

def port_is_free(port: int, host: str = HOST) -> bool:
    """Prüft per Bind-Test, ob ``host:port`` frei ist.

    Unter Windows ohne ``SO_REUSEADDR`` (schlägt fehl, solange ein Socket den
    Port belegt, auch ein werkzeug-Server mit ``SO_REUSEADDR``); unter POSIX mit.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        if sys.platform != "win32":
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


def _port_message(port: int) -> str:
    return (f"Port {port} auf {HOST} ist bereits belegt. Läuft NoAssist schon? "
            f"Anderen Port wählen (--port oder NOASSIST_PORT) oder den belegenden Prozess beenden.")


class AppServer:
    """Startet NoAssist auf ``127.0.0.1:<port>`` (werkzeug, mehrere Threads).

    Baut aus ``data_dir`` (Standard: ``<repo>/data``) ``ChatStore``
    (``noassist.db``), ``ConfigStore`` (``config.json``) und ``ChatService``.
    Für Tests kann statt dessen ein fertiger ``service`` übergeben werden.
    Erst ``start_background()`` bzw. ``serve_forever()`` prüft den Port,
    startet den Service (``service.start()``) und bindet den Socket.
    ``shutdown()`` beendet alles (idempotent, threadsicher).
    """

    def __init__(
        self,
        port: int = DEFAULT_PORT,
        dev: bool = False,
        data_dir: str | Path | None = None,
        *,
        service: Any = None,
    ) -> None:
        self.port = int(port)
        self.dev = bool(dev)
        self.data_dir = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
        self.boot_id = uuid.uuid4().hex
        self.url = f"http://{HOST}:{self.port}/"
        self.service = service
        self.app: Flask | None = None
        self._server: Any = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._started = False
        self._closed = threading.Event()
        self._watch_stop = threading.Event()
        self._service_started = False

    def _build_service(self) -> Any:
        from chat_service import ChatService
        from storage import ChatStore, ConfigStore

        self.data_dir.mkdir(parents=True, exist_ok=True)
        store = ChatStore(self.data_dir / "noassist.db")
        config = ConfigStore(self.data_dir / "config.json")
        config.load()
        return ChatService(store, config, self.data_dir)

    def _bind(self) -> None:
        """Port prüfen, Service starten, Server binden, im Dev-Modus Beobachtung starten."""
        with self._lock:
            if self._closed.is_set():
                raise RuntimeError("Der Server wurde bereits beendet.")
            if self._started:
                raise RuntimeError("Der Server läuft bereits.")
            if not port_is_free(self.port):
                raise PortInUseError(_port_message(self.port))
            if self.service is None:
                self.service = self._build_service()
            start = getattr(self.service, "start", None)
            if callable(start):
                start()
            self._service_started = True
            self.app = create_app(self.service, port=self.port, dev=self.dev, boot_id=self.boot_id)
            try:
                self._server = make_server(HOST, self.port, self.app, threaded=True)
            except OSError as exc:
                self._close_service()
                raise PortInUseError(_port_message(self.port)) from exc
            self._server.daemon_threads = True
            self._started = True
            if self.dev:
                self._start_static_watch()

    def _start_static_watch(self) -> None:
        assert self.app is not None
        hub: DevHub = self.app.extensions["noassist"]["dev_hub"]
        try:
            from devreload import watch_static
        except ImportError as exc:
            log.warning("Live-Reload für statische Dateien nicht verfügbar: %s", exc)
            return
        try:
            watch_static(STATIC_DIR, hub.notify_reload, self._watch_stop)
        except Exception as exc:  # noqa: BLE001 - Dev-Komfort darf den Start nicht verhindern
            log.warning("Beobachtung von %s fehlgeschlagen: %s: %s", STATIC_DIR, type(exc).__name__, exc)

    def start_background(self) -> None:
        """Startet den Server in einem Daemon-Thread und kehrt sofort zurück.

        Wirft ``PortInUseError``, wenn der Port belegt ist.
        """
        self._bind()
        self._thread = threading.Thread(target=self._server.serve_forever, name="noassist-server", daemon=True)
        self._thread.start()

    def serve_forever(self) -> None:
        """Startet den Server und blockiert bis ``shutdown()`` (oder Strg+C).

        Die Anfrage-Schleife läuft in einem eigenen Thread; der aufrufende
        Thread wartet nur. So kann ``shutdown()`` auch aus einem Signal-Handler
        im selben Thread aufgerufen werden, ohne zu blockieren.
        """
        self.start_background()
        log.info("NoAssist läuft auf %s", self.url)
        try:
            while not self._closed.wait(0.5):
                if self._thread is not None and not self._thread.is_alive():
                    break
        except KeyboardInterrupt:
            pass
        finally:
            self.shutdown()

    def shutdown(self, timeout: float = 5.0) -> None:
        """Beendet Server, Streams, Beobachtung und Service (idempotent, threadsicher)."""
        with self._lock:
            if self._closed.is_set():
                return
            self._closed.set()
            self._watch_stop.set()
            if self.app is not None:
                self.app.extensions["noassist"]["stop"].set()
            server, thread = self._server, self._thread
            if server is not None:
                if thread is not None and thread.is_alive() and thread is not threading.current_thread():
                    server.shutdown()
                    thread.join(timeout)
                server.server_close()
            self._close_service(timeout)

    def _close_service(self, timeout: float = 5.0) -> None:
        if not self._service_started:
            return
        self._service_started = False
        close = getattr(self.service, "close", None)
        if callable(close):
            try:
                close(timeout=timeout)
            except TypeError:
                close()

    def __enter__(self) -> AppServer:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.shutdown()


# --- Demo -----------------------------------------------------------------------

class _DemoTurn:
    """Einfacher Turn mit Event-Puffer (seq ab 1), wie ``chat_service.Turn``."""

    def __init__(self, chat_id: str):
        self.id = "t_" + uuid.uuid4().hex
        self.chat_id = chat_id
        self.cancel = threading.Event()
        self.finished = False
        self._events: list[dict[str, Any]] = []
        self._cond = threading.Condition()

    def publish(self, event: dict[str, Any]) -> int:
        with self._cond:
            self._events.append(event)
            if event.get("type") == "done":
                self.finished = True
            self._cond.notify_all()
            return len(self._events)

    def events_after(self, seq: int, timeout: float) -> tuple[list[tuple[int, dict[str, Any]]], bool]:
        with self._cond:
            if len(self._events) <= seq and not self.finished and timeout > 0:
                self._cond.wait(timeout)
            items = [(i + 1, e) for i, e in enumerate(self._events) if i + 1 > seq]
            return items, self.finished


class _DemoService:
    """Fake-``ChatService`` für die Demo: Echo-Antworten, Daten nur im Speicher."""

    def __init__(self) -> None:
        from chat_types import Message
        from storage import ConfigStore, default_config

        self._Message = Message
        self.config = ConfigStore(Path("demo-config.json"))  # wird nie gespeichert
        self.config.data = default_config()
        self.chats: dict[str, dict[str, Any]] = {}
        self.turns: dict[str, _DemoTurn] = {}

    def start(self) -> None: ...
    def close(self, timeout: float = 5) -> None: ...

    def get_settings(self) -> dict[str, Any]:
        return dict(self.config.data["settings"])

    def update_settings(self, data: dict[str, Any]) -> dict[str, Any]:
        self.config.data["settings"].update({k: v for k, v in data.items() if k in self.config.data["settings"]})
        return self.get_settings()

    def list_providers(self) -> list[dict[str, Any]]:
        return [dict(p, builtin=p["id"] in ("lmstudio", "claude")) for p in self.config.public()["providers"]]

    def add_provider(self, data: dict[str, Any]) -> dict[str, Any]:
        raise ValueError("In der Demo können keine Provider angelegt werden.")

    def update_provider(self, pid: str, data: dict[str, Any]) -> dict[str, Any]:
        raise KeyError(pid)

    def delete_provider(self, pid: str) -> None:
        raise ValueError("Eingebaute Provider können nicht gelöscht werden.")

    def provider_models(self, pid: str) -> dict[str, Any]:
        return {"models": [{"id": "echo", "label": "Echo"}]}

    def provider_status(self, pid: str) -> dict[str, Any]:
        return {"ok": True, "message": "Demo", "details": {}}

    def list_tools(self) -> dict[str, Any]:
        return {"tools": [], "mcp_servers": []}

    def list_chats(self) -> list[dict[str, Any]]:
        return sorted(self.chats.values(), key=lambda c: c["updated_at"], reverse=True)

    def create_chat(self, provider_id=None, model=None, title=None) -> dict[str, Any]:  # noqa: ANN001
        now = time.strftime("%Y-%m-%dT%H:%M:%S")
        cid = "c_" + uuid.uuid4().hex
        self.chats[cid] = {"id": cid, "title": title or "Neuer Chat", "provider_id": provider_id or "lmstudio",
                           "model": model, "created_at": now, "updated_at": now, "messages": [],
                           "active_turn_id": None}
        return {k: v for k, v in self.chats[cid].items() if k not in ("messages", "active_turn_id")}

    def get_chat(self, cid: str) -> dict[str, Any] | None:
        return dict(self.chats[cid]) if cid in self.chats else None

    def update_chat(self, cid: str, data: dict[str, Any]) -> dict[str, Any] | None:
        if cid not in self.chats:
            return None
        self.chats[cid].update(data)
        return self.get_chat(cid)

    def delete_chat(self, cid: str) -> bool:
        return self.chats.pop(cid, None) is not None

    def start_turn(self, chat_id, content, provider_id=None, model=None, retry=False,  # noqa: ANN001, ANN201
                   user_message_id=None):
        chat = self.chats.get(chat_id)
        if chat is None:
            raise KeyError(chat_id)
        user = self._Message(role="user", content=content)
        if retry:
            from chat_service import RetryStale

            msgs = chat["messages"]
            index = next((i for i in range(len(msgs) - 1, -1, -1) if msgs[i]["role"] == "user"), None)
            # Demo: abgebrochene Turns speichern nichts; steht etwas dahinter, ist es eine fertige Antwort.
            if index is None or msgs[index]["id"] != user_message_id or index < len(msgs) - 1:
                raise RetryStale("Diese Nachricht kann nicht mehr wiederholt werden. Bitte den Chat neu laden.")
            user = self._Message.from_dict(msgs[index])
        else:
            chat["messages"].append(user.to_dict())
        turn = _DemoTurn(chat_id)
        self.turns[turn.id] = turn
        chat["active_turn_id"] = turn.id

        def run() -> None:
            turn.publish({"type": "turn_start", "turn_id": turn.id, "chat_id": chat_id,
                          "provider_id": chat["provider_id"], "model": model})
            text = f"Echo: {user.content}"
            for word in text.split(" "):
                if turn.cancel.wait(0.2):
                    turn.publish({"type": "done", "status": "cancelled"})
                    chat["active_turn_id"] = None
                    return
                turn.publish({"type": "delta", "text": word + " "})
            msg = self._Message(role="assistant", content=text, provider_id=chat["provider_id"])
            chat["messages"].append(msg.to_dict())
            turn.publish({"type": "message", "message": msg.to_dict()})
            chat["active_turn_id"] = None
            turn.publish({"type": "done", "status": "completed"})

        threading.Thread(target=run, daemon=True).start()
        return turn, user

    def get_turn(self, turn_id: str) -> _DemoTurn | None:
        return self.turns.get(turn_id)

    def cancel_turn(self, turn_id: str) -> bool:
        turn = self.turns.get(turn_id)
        if turn is None or turn.finished:
            return False
        turn.cancel.set()
        return True


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(prog="python -m server", description="NoAssist-Server mit Fake-ChatService (Demo).")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--dev", action="store_true", help="Dev-Modus (Live-Reload der statischen Dateien)")
    parser.add_argument("--real", action="store_true", help="echten ChatService aus data/ verwenden")
    args = parser.parse_args()

    server = AppServer(args.port, args.dev, service=None if args.real else _DemoService())
    try:
        server.serve_forever()  # Strg+C beendet
    except PortInUseError as exc:
        print(f"Fehler: {exc}", file=sys.stderr)
        sys.exit(1)
