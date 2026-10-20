"""Einstieg von NoAssist: Server, Fenster (pywebview) und Dev-Modus.

Aufruf (aus dem Repo): ``python src\\main.py [--dev] [--no-window] [--port N]``

Modi:

- Standard: Der Server (``AppServer``) läuft in einem Thread dieses Prozesses,
  im Hauptthread öffnet sich das pywebview-Fenster. Schließen des Fensters
  beendet alles.
- ``--dev``: Zwei Prozesse. Dieser Prozess (Fenster) startet über den
  ``Supervisor`` einen Server-Prozess (``--serve --dev``) und startet ihn bei
  Änderungen an ``src/**/*.py`` neu. Statische Dateien werden ohne Neustart
  neu geladen. Änderungen an ``main.py`` und ``devreload.py`` brauchen einen
  manuellen Neustart.
- ``--no-window``: Kein Fenster, nur die URL ausgeben; läuft bis Strg+C.
- ``--serve``: Nur der Server im Hauptthread (Kindprozess des Dev-Modus,
  zusammen mit ``--dev``). Beendet sich sauber bei ``SIGBREAK``
  (``CTRL_BREAK_EVENT`` vom Supervisor) und ``SIGINT``.

Der Port ist 8765, änderbar über ``--port`` oder ``NOASSIST_PORT``. Gebunden
wird immer nur an 127.0.0.1.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import threading
import time
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

HOST = "127.0.0.1"
DEFAULT_PORT = 8765
PORT_ENV = "NOASSIST_PORT"
READY_TIMEOUT_S = 30.0

SRC_DIR = Path(__file__).resolve().parent
REPO_DIR = SRC_DIR.parent

WINDOW_TITLE = "NoAssist"

log = logging.getLogger("noassist.main")


# --- Ausgabe und Hilfsfunktionen -------------------------------------------------

def _say(message: str, *, error: bool = False) -> None:
    """Meldung für den Nutzer auf stdout bzw. stderr (robust gegen Kodierungsfehler)."""
    stream = sys.stderr if error else sys.stdout
    try:
        print(message, file=stream, flush=True)
    except UnicodeEncodeError:
        encoding = getattr(stream, "encoding", None) or "ascii"
        print(message.encode(encoding, "replace").decode(encoding), file=stream, flush=True)
    except (OSError, ValueError):  # Ausgabe geschlossen (z. B. pythonw.exe)
        pass


def _port_in_use_message(port: int) -> str:
    return (f"Fehler: Port {port} auf {HOST} ist bereits belegt. Läuft NoAssist schon? "
            f"Anderen Port wählen (--port oder {PORT_ENV}) oder den belegenden Prozess beenden.")


def _wait_health(url: str, timeout: float, stop: threading.Event | None = None) -> bool:
    """Wartet bis zu ``timeout`` Sekunden, bis ``GET <url>api/health`` mit 200 antwortet."""
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # nie über einen Proxy
    health = url.rstrip("/") + "/api/health"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if stop is not None and stop.is_set():
            return False
        try:
            with opener.open(health, timeout=3) as response:
                if response.status == 200 and json.loads(response.read(65536)).get("app") == "noassist":
                    return True
        except (OSError, ValueError):
            pass
        time.sleep(0.1)
    return False


def _setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    # Kein Log je Anfrage bzw. je Verbindung (werkzeug, httpx, lmstudio-Websockets)
    for name in ("werkzeug", "httpx", "httpcore", "lmstudio", "watchfiles",
                 "AsyncWebsocketThread", "AsyncWebsocketHandler", "SyncLMStudioWebsocket"):
        logging.getLogger(name).setLevel(logging.WARNING)
    # pywebview gibt selbst aus; ohne das erschiene jede Zeile doppelt
    logging.getLogger("pywebview").propagate = False
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(errors="replace")
            except (OSError, ValueError):
                pass


def _port_from_env() -> int | None:
    raw = os.environ.get(PORT_ENV, "").strip()
    return int(raw) if raw else None


def _parse_port(text: str) -> int:
    try:
        port = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"ungültiger Port: {text!r}") from None
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError(f"Port muss zwischen 1 und 65535 liegen: {port}")
    return port


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python src\\main.py",
        description="NoAssist: Chat-Oberfläche für lokale und Cloud-Modelle.",
    )
    parser.add_argument("--dev", action="store_true",
                        help="Dev-Modus: Server-Neustart bei Codeänderungen, Live-Reload der Web-Dateien")
    parser.add_argument("--no-window", action="store_true",
                        help="kein Fenster öffnen, nur die URL ausgeben (Bedienung im Browser)")
    parser.add_argument("--port", type=_parse_port, default=None,
                        help=f"Port auf {HOST} (Standard: {PORT_ENV} oder {DEFAULT_PORT})")
    parser.add_argument("--serve", action="store_true",
                        help="nur den Server im Vordergrund starten (intern für den Dev-Modus)")
    args = parser.parse_args(argv)
    if args.port is None:
        try:
            env_port = _port_from_env()
            args.port = _parse_port(str(env_port)) if env_port is not None else DEFAULT_PORT
        except (ValueError, argparse.ArgumentTypeError):
            parser.error(f"{PORT_ENV}={os.environ.get(PORT_ENV)!r} ist kein gültiger Port.")
    return args


# --- Beenden per Konsole ----------------------------------------------------------

def _install_console_handler(on_interrupt: Callable[[], None], on_close: Callable[[], None]) -> None:
    """Reagiert auf Strg+C/Strg+Pause sowie das Schließen der Konsole.

    Unter Windows über ``SetConsoleCtrlHandler`` (pywin32): Der Handler läuft
    in einem eigenen Thread und greift auch, solange der Hauptthread in der
    Nachrichtenschleife des Fensters steckt (dort käme ein Python-Signal-Handler
    nicht zum Zug). ``on_close`` läuft beim Schließen der Konsole, Abmelden und
    Herunterfahren; danach beendet Windows den Prozess.
    """
    if sys.platform == "win32":
        try:
            import win32api
            import win32con
        except ImportError:
            pass
        else:
            def handler(ctrl_type: int) -> bool:
                if ctrl_type in (win32con.CTRL_C_EVENT, win32con.CTRL_BREAK_EVENT):
                    on_interrupt()
                    return True
                on_close()
                return True

            win32api.SetConsoleCtrlHandler(handler, True)
            return

    def signal_handler(signum: int, frame: Any) -> None:
        on_interrupt()

    signal.signal(signal.SIGINT, signal_handler)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, signal_handler)


# --- Fenster -------------------------------------------------------------------------

class WindowError(RuntimeError):
    """Das Fenster ließ sich nicht öffnen (pywebview/WebView2 fehlt oder ist defekt)."""


class Window:
    """Das pywebview-Fenster; ``close()`` darf aus jedem Thread aufgerufen werden."""

    def __init__(self) -> None:
        self._window: Any = None
        self._shown = threading.Event()
        self._failure: str | None = None
        self._closed = threading.Event()

    def run(self, url: str, dev: bool) -> None:
        """Öffnet das Fenster und blockiert, bis es geschlossen wird (nur im Hauptthread).

        Wirft ``WindowError``, wenn es sich nicht öffnen ließ.
        """
        try:
            import webview
        except Exception as exc:  # ImportError, aber auch Fehler beim Laden von pythonnet
            raise WindowError(f"pywebview lässt sich nicht laden ({type(exc).__name__}: {exc}).") from exc

        webview.settings["OPEN_DEVTOOLS_IN_DEBUG"] = False
        webview.settings["ALLOW_FILE_URLS"] = False
        try:
            self._window = webview.create_window(
                WINDOW_TITLE, url, width=1200, height=800, min_size=(720, 500)
            )
            self._window.events.shown += self._on_shown
            if self._closed.is_set():  # schon vor dem Öffnen beendet (Strg+C)
                return
            webview.start(debug=dev)
        except Exception as exc:
            raise WindowError(f"{type(exc).__name__}: {exc}") from exc
        finally:
            self._closed.set()
        if self._failure:
            raise WindowError(self._failure)
        if not self._shown.is_set():
            raise WindowError("Das Fenster wurde nicht angezeigt.")

    def _on_shown(self) -> None:
        renderer = getattr(sys.modules.get("webview.platforms.winforms"), "renderer", None)
        if renderer == "mshtml":
            # Ohne WebView2 fällt pywebview auf den alten IE-Kern zurück; der kann die Oberfläche nicht.
            self._failure = "Microsoft Edge WebView2 Runtime nicht gefunden (pywebview fiel auf MSHTML zurück)."
            self.close()
            return
        self._shown.set()

    def close(self) -> None:
        """Schließt das Fenster (idempotent)."""
        self._closed.set()
        window = self._window
        if window is not None:
            try:
                window.destroy()
            except Exception:  # noqa: BLE001 - Fenster ist evtl. schon zu
                pass


def _window_hint(url: str, exc: WindowError) -> None:
    _say(f"Fehler: Das Fenster konnte nicht geöffnet werden: {exc}", error=True)
    _say("Hinweis: Microsoft Edge WebView2 Runtime installieren oder NoAssist mit --no-window starten "
         f"und im Browser öffnen: {url}", error=True)
    _say(f"Der Server läuft weiter unter {url} (Strg+C beendet).")


# --- Modi ------------------------------------------------------------------------------

def run_serve(port: int, dev: bool) -> int:
    """``--serve``: nur der Server, im Vordergrund bis SIGBREAK/SIGINT."""
    from server import AppServer, PortInUseError

    stop = threading.Event()

    def request_stop(signum: int, frame: Any) -> None:
        # Nur das Ereignis setzen; das Herunterfahren läuft im Hauptthread unten
        # (shutdown() im Signal-Handler könnte eine gerade gehaltene Sperre blockieren).
        stop.set()

    signal.signal(signal.SIGINT, request_stop)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, request_stop)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, request_stop)

    server = AppServer(port, dev)
    try:
        server.start_background()
    except PortInUseError:
        _say(_port_in_use_message(port), error=True)
        server.shutdown()
        return 1
    try:
        log.info("NoAssist-Server läuft auf %s (PID %d, dev=%s)", server.url, os.getpid(), dev)
        while not stop.wait(0.5):
            pass
        log.info("Server wird beendet ...")
    finally:
        server.shutdown()
    return 0


def run_app(port: int, no_window: bool) -> int:
    """Standard: Server im Thread dieses Prozesses, dazu Fenster (oder nur die URL)."""
    from server import AppServer, PortInUseError

    stop = threading.Event()
    window = Window()

    def on_interrupt() -> None:
        stop.set()
        window.close()

    def on_close() -> None:
        on_interrupt()
        server.shutdown()  # Konsole wird geschlossen: Kindprozesse (claude.exe, MCP) noch selbst beenden

    server = AppServer(port, False)
    try:
        server.start_background()
    except PortInUseError:
        _say(_port_in_use_message(port), error=True)
        server.shutdown()
        return 1
    _install_console_handler(on_interrupt, on_close)
    try:
        if not _wait_health(server.url, READY_TIMEOUT_S, stop):
            if stop.is_set():
                return 0
            _say(f"Fehler: Der Server antwortet nach {READY_TIMEOUT_S:.0f} s nicht auf {server.url}api/health.",
                 error=True)
            return 1
        if no_window:
            _say(f"NoAssist läuft auf {server.url} (Strg+C beendet).")
        else:
            _say(f"NoAssist läuft auf {server.url} (auch im Browser erreichbar).")
            try:
                window.run(server.url, dev=False)
                return 0
            except WindowError as exc:
                if stop.is_set():
                    return 0
                _window_hint(server.url, exc)
        while not stop.wait(0.5):
            pass
        return 0
    finally:
        _say("NoAssist wird beendet ...")
        server.shutdown()


def run_dev(port: int, no_window: bool) -> int:
    """``--dev``: Supervisor startet den Server als Kindprozess, dazu Fenster (oder nur die URL)."""
    from devreload import Supervisor, port_is_free

    if not port_is_free(port):
        _say(_port_in_use_message(port), error=True)
        return 1

    cmd = [sys.executable, str(SRC_DIR / "main.py"), "--serve", "--dev", "--port", str(port)]
    supervisor = Supervisor(cmd, SRC_DIR, port, cwd=REPO_DIR)
    stop = threading.Event()
    window = Window()
    sup_thread: threading.Thread | None = None

    def on_interrupt() -> None:
        stop.set()
        window.close()

    def on_close() -> None:
        on_interrupt()
        supervisor.stop()

    def watch() -> None:
        try:
            supervisor.run_forever()
        except Exception as exc:  # noqa: BLE001 - im Thread: melden und beenden
            _say(f"[dev] Supervisor beendet: {type(exc).__name__}: {exc}", error=True)
            on_interrupt()

    _install_console_handler(on_interrupt, on_close)
    try:
        try:
            supervisor.start()  # vor run_forever: kein Wettlauf mit wait_ready
        except (OSError, RuntimeError) as exc:
            _say(f"Fehler: Der Server konnte nicht gestartet werden: {exc}", error=True)
            return 1
        sup_thread = threading.Thread(target=watch, name="noassist-supervisor", daemon=True)
        sup_thread.start()

        if not supervisor.wait_ready(READY_TIMEOUT_S):
            if stop.is_set():
                return 0
            _say("[dev] Der Server ist noch nicht bereit (siehe Meldungen oben). "
                 "Warte, bis er nach einer Korrektur läuft ... (Strg+C beendet)")
            while not stop.is_set() and not supervisor.wait_ready(1.0):
                stop.wait(0.5)
            if stop.is_set():
                return 0

        url = supervisor.url
        if no_window:
            _say(f"NoAssist (Dev) läuft auf {url} (Strg+C beendet).")
        else:
            _say(f"NoAssist (Dev) läuft auf {url} (auch im Browser erreichbar).")
            try:
                window.run(url, dev=True)
                return 0
            except WindowError as exc:
                if stop.is_set():
                    return 0
                _window_hint(url, exc)
        while not stop.wait(0.5):
            pass
        return 0
    finally:
        _say("NoAssist wird beendet ...")
        supervisor.stop()
        if sup_thread is not None:
            sup_thread.join(5.0)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _setup_logging()
    if args.serve:
        return run_serve(args.port, args.dev)
    if args.dev:
        return run_dev(args.port, args.no_window)
    return run_app(args.port, args.no_window)


if __name__ == "__main__":
    sys.exit(main())
