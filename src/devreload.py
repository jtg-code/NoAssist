"""Dev-Modus von NoAssist: Live-Reload ohne Neustart des Fensters.

Zwei Bausteine:

* ``watch_static`` läuft im Server-Prozess und beobachtet die statischen
  Web-Dateien. Änderungen werden an einen Callback gemeldet, damit Fenster und
  Browser neu laden (bzw. bei reinen CSS-Änderungen nur das CSS austauschen).
* ``Supervisor`` läuft im Fenster-Prozess. Er startet den Server als
  Kindprozess und startet ihn neu, sobald sich eine ``*.py``-Datei unter
  ``src/`` ändert (ohne ``web/``, ``main.py`` und ``devreload.py``; bei den
  beiden letzten gibt es nur einen Hinweis, dass ein manueller Neustart nötig
  ist).

Windows-Fallen, die hier bewusst umgangen werden:

* ``watchfiles.run_process`` wird nicht verwendet. Es beendet Kindprozesse unter
  Windows hart (``TerminateProcess``); Enkelprozesse wie ``claude.exe`` oder
  MCP-Server blieben verwaist zurück.
* werkzeug setzt ``SO_REUSEADDR``. Unter Windows kann ein verwaister alter
  Server denselben Port dann doppelt belegen. Deshalb wird vor jedem Start
  geprüft, ob der Port wirklich frei ist (``port_is_free``).
* Der Server läuft in einer eigenen Prozessgruppe (damit ``CTRL_BREAK_EVENT``
  nur ihn trifft) und in einem Job-Object mit
  ``JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE``. Stirbt der Supervisor, auch per
  ``taskkill /F``, schließt Windows den Job-Handle und beendet den ganzen
  Server-Prozessbaum.

Demo (aus ``src``): ``python -m devreload`` startet einen eingebauten
Dummy-Server unter Aufsicht; ``python -m devreload -- <befehl ...>`` beaufsichtigt
einen beliebigen Befehl.
"""

from __future__ import annotations

import contextlib
import http.client
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

IS_WINDOWS = sys.platform == "win32"

HOST = "127.0.0.1"
HEALTH_PATH = "/api/health"

# Relativ zu ``watch_dir``: wird komplett ignoriert.
DEFAULT_IGNORE: tuple[str, ...] = ("web",)
# Relativ zu ``watch_dir``: kein Neustart, nur ein Hinweis (laufen im Fenster-Prozess).
DEFAULT_MANUAL_RESTART: tuple[str, ...] = ("main.py", "devreload.py")

_DEBOUNCE_MS = 300
_TICK_MS = 200               # Takt von run_forever: Absturzprüfung, erster Start nach Watcher-Start
_CONNECT_PROBE_S = 0.25      # kurz halten: ein abgelehnter Connect dauert unter Windows ~2 s
_HEALTH_HTTP_TIMEOUT_S = 3.0
_JOB_GRACE_S = 1.0           # Restprozesse dürfen sich nach dem Server noch kurz selbst beenden

# Windows-Konstanten, die weder ``subprocess`` noch ``win32con`` sicher anbieten.
_CREATE_SUSPENDED = 0x00000004
_PROCESS_TERMINATE = 0x0001
_PROCESS_SET_QUOTA = 0x0100
_PROCESS_SUSPEND_RESUME = 0x0800

Log = Callable[[str], None]


def _default_log(message: str) -> None:
    """Gibt eine Meldung mit Präfix ``[dev]`` auf stdout aus (robust gegen Kodierungsfehler)."""
    text = f"[dev] {message}"
    try:
        print(text, flush=True)
    except UnicodeEncodeError:
        encoding = getattr(sys.stdout, "encoding", None) or "ascii"
        print(text.encode(encoding, "replace").decode(encoding), flush=True)
    except (OSError, ValueError):  # stdout geschlossen oder ungültig
        pass


def _rel_posix(path: str | Path, root: Path) -> str:
    """Pfad relativ zu ``root`` mit ``/`` als Trenner (Fallback: Dateiname)."""
    try:
        rel = os.path.relpath(path, root)
    except ValueError:  # anderes Laufwerk
        return Path(path).name
    return rel.replace(os.sep, "/")


def _norm(path: str | Path) -> str:
    """Normalisierter absoluter Pfad für Vergleiche (unter Windows ohne Groß-/Kleinschreibung)."""
    return os.path.normcase(os.path.abspath(path))


def _fmt_exit_code(code: int | None) -> str:
    """Exit-Code lesbar machen; große Windows-Codes (NTSTATUS) zusätzlich hexadezimal."""
    if code is None:
        return "unbekannt"
    if IS_WINDOWS and not 0 <= code <= 255:
        return f"{code} / 0x{code & 0xFFFFFFFF:08X}"
    return str(code)


# --- Port-Prüfung ------------------------------------------------------------

def port_is_free(port: int, host: str = HOST) -> bool:
    """Prüft per Bind-Test, ob ``host:port`` frei ist.

    Unter Windows wird ohne ``SO_REUSEADDR`` gebunden: Das schlägt fehl,
    solange irgendein Socket den Port belegt (auch ein werkzeug-Server mit
    ``SO_REUSEADDR``), gelingt aber trotz Verbindungen in ``TIME_WAIT``. Unter
    POSIX ist es mit ``SO_REUSEADDR`` genau umgekehrt nötig.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        if not IS_WINDOWS:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


# --- Statische Dateien (läuft im Server-Prozess) ------------------------------

def watch_static(
    static_dir: str | Path,
    on_change: Callable[[list[str], bool], None],
    stop: threading.Event,
) -> threading.Thread:
    """Beobachtet ``static_dir`` in einem Daemon-Thread und meldet Änderungen.

    ``on_change(files, css_only)`` erhält die geänderten Pfade relativ zu
    ``static_dir`` (sortiert, mit ``/``) und ``css_only=True``, wenn
    ausschließlich ``.css``-Dateien betroffen sind. Der Callback läuft im
    Beobachter-Thread; Exceptions darin werden gemeldet, beenden die
    Beobachtung aber nicht. Änderungen werden 300 ms gebündelt.

    Der Thread endet kurz nach ``stop.set()``. Gibt den bereits gestarteten
    Thread zurück. Die Beobachtung ist erst kurz nach dem Aufruf aktiv
    (Größenordnung 0,1 s); Änderungen davor werden nicht gemeldet. Wirft
    ``FileNotFoundError``, wenn ``static_dir`` kein Verzeichnis ist.
    """
    from watchfiles import watch  # hier statt im Thread: Importfehler sofort beim Aufrufer

    root = Path(static_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Verzeichnis für statische Dateien nicht gefunden: {root}")

    def run() -> None:
        try:
            for changes in watch(root, debounce=_DEBOUNCE_MS, stop_event=stop, raise_interrupt=False):
                if stop.is_set():
                    break
                # Windows meldet bei Änderungen in Unterordnern auch den Ordner selbst
                # als geändert; Ordner würden sonst css_only verfälschen.
                files = sorted({_rel_posix(path, root) for _change, path in changes if not os.path.isdir(path)})
                if not files:
                    continue
                css_only = all(name.lower().endswith(".css") for name in files)
                try:
                    on_change(files, css_only)
                except Exception as exc:  # Callback-Fehler dürfen die Beobachtung nicht beenden
                    _default_log(f"Fehler im Reload-Callback: {type(exc).__name__}: {exc}")
        except Exception as exc:
            if not stop.is_set():
                _default_log(f"Beobachtung von {root} beendet: {type(exc).__name__}: {exc}")

    thread = threading.Thread(target=run, name="devreload-static", daemon=True)
    thread.start()
    return thread


# --- Job-Object (Windows, pywin32) --------------------------------------------

def _create_kill_on_close_job() -> Any:
    """Erzeugt ein Job-Object mit ``JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE``.

    Der Handle ist nicht vererbbar; er gehört allein diesem Prozess. Wird er
    geschlossen (auch beim Tod des Prozesses), beendet Windows alle Prozesse
    im Job.
    """
    import win32job

    job: Any = win32job.CreateJobObject(None, "")  # PyHANDLE (die Typ-Stubs sind hier ungenau)
    try:
        info = win32job.QueryInformationJobObject(job, win32job.JobObjectExtendedLimitInformation)
        info["BasicLimitInformation"]["LimitFlags"] |= win32job.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        win32job.SetInformationJobObject(job, win32job.JobObjectExtendedLimitInformation, info)
    except Exception:
        job.Close()
        raise
    return job


def _assign_to_job(job: Any, pid: int) -> None:
    """Ordnet den Prozess ``pid`` dem Job zu."""
    import win32api
    import win32job

    handle: Any = win32api.OpenProcess(_PROCESS_SET_QUOTA | _PROCESS_TERMINATE, False, pid)  # PyHANDLE
    try:
        win32job.AssignProcessToJobObject(job, handle)
    finally:
        handle.Close()


def _resume_process(pid: int) -> None:
    """Setzt einen mit ``CREATE_SUSPENDED`` gestarteten Prozess fort.

    ``subprocess.Popen`` gibt den Thread-Handle nicht heraus, daher
    ``NtResumeProcess`` (stabil seit Windows XP, auch von psutil genutzt).
    """
    import ctypes
    from ctypes import wintypes

    import win32api

    handle: Any = win32api.OpenProcess(_PROCESS_SUSPEND_RESUME, False, pid)  # PyHANDLE
    try:
        ntdll = ctypes.WinDLL("ntdll")
        ntdll.NtResumeProcess.argtypes = [wintypes.HANDLE]
        ntdll.NtResumeProcess.restype = ctypes.c_long
        status = ntdll.NtResumeProcess(int(handle))
    finally:
        handle.Close()
    if status != 0:
        raise OSError(f"NtResumeProcess fehlgeschlagen (NTSTATUS 0x{status & 0xFFFFFFFF:08X})")


def _job_active_processes(job: Any) -> int:
    """Anzahl der noch laufenden Prozesse im Job."""
    import win32job

    info = win32job.QueryInformationJobObject(job, win32job.JobObjectBasicAccountingInformation)
    return int(info["ActiveProcesses"])


def _has_console() -> bool:
    """True, wenn dieser Prozess eine Konsole hat (auch ohne sichtbares Fenster).

    Nur dann teilt der Server die Konsole und ist per ``CTRL_BREAK_EVENT``
    erreichbar. Ohne Konsole (z. B. ``pythonw.exe``) würde jeder Server-Start
    ein neues Konsolenfenster öffnen.
    """
    import ctypes

    return bool(ctypes.windll.kernel32.GetConsoleCP())


# --- Supervisor (läuft im Fenster-Prozess) ------------------------------------

class Supervisor:
    """Startet den Server als Kindprozess und startet ihn bei Codeänderungen neu.

    ``cmd`` ist der Server-Befehl (z. B. ``[sys.executable, "main.py",
    "--serve", "--port", "8765"]``), ``watch_dir`` das beobachtete Verzeichnis
    (``src``), ``port`` der Port des Servers auf 127.0.0.1. ``ignore`` sind
    Pfade relativ zu ``watch_dir`` (oder absolut), die keinen Neustart
    auslösen (Standard: ``web``). Pfade aus ``manual_restart`` (Standard:
    ``main.py``, ``devreload.py``) lösen ebenfalls keinen Neustart aus, es wird
    aber ein Hinweis ausgegeben, dass ein manueller Neustart nötig ist; das
    gilt auch, wenn sie zusätzlich in ``ignore`` stehen.

    Ablauf eines Neustarts: ``CTRL_BREAK_EVENT`` an die Prozessgruppe des
    Servers, bis zu ``stop_timeout`` Sekunden warten, danach
    ``taskkill /PID <pid> /T /F``; übrig gebliebene Prozesse im Job-Object
    werden ebenfalls beendet. Dann warten, bis der Port frei ist, neu starten
    und auf ``/api/health`` warten. Stürzt der Server ab (auch beim Start),
    wird nicht in einer Schleife neu gestartet, sondern bis zur nächsten
    Änderung gewartet. Der Server läuft mit ``PYTHONDONTWRITEBYTECODE=1``
    (sofern ``env`` nichts anderes vorgibt), damit nach schnellen Änderungen
    kein veralteter ``.pyc``-Code geladen wird.

    Alle öffentlichen Methoden sind threadsicher; ``stop()`` darf aus einem
    anderen Thread aufgerufen werden, während ``run_forever()`` läuft.

    Empfohlene Nutzung (Fenster im Hauptthread)::

        sup = Supervisor(cmd, src_dir, port)
        threading.Thread(target=sup.run_forever, daemon=True).start()
        if not sup.wait_ready(30): ...   # Server nicht bereit
        ...                              # Fenster öffnen, blockiert bis zum Schließen
        sup.stop()

    ``run_forever`` startet den Server selbst, sobald die Beobachtung aktiv
    ist, damit keine Änderung während des Starts verloren geht. Wird vorher
    ``start()`` aufgerufen, holt ``run_forever`` Änderungen seit dem Start
    über die Änderungszeit der Dateien nach.
    """

    def __init__(
        self,
        cmd: Sequence[str],
        watch_dir: str | Path,
        port: int,
        ignore: Iterable[str | Path] | None = None,
        *,
        manual_restart: Iterable[str | Path] | None = None,
        cwd: str | Path | None = None,
        env: Mapping[str, str] | None = None,
        stop_timeout: float = 5.0,
        ready_timeout: float = 30.0,
        port_timeout: float = 10.0,
        log: Log | None = None,
    ) -> None:
        from watchfiles import DefaultFilter

        if not cmd:
            raise ValueError("cmd darf nicht leer sein.")
        self.cmd: list[str] = [str(part) for part in cmd]
        self.watch_dir: Path = Path(watch_dir).resolve()
        if not self.watch_dir.is_dir():
            raise FileNotFoundError(f"Zu beobachtendes Verzeichnis nicht gefunden: {self.watch_dir}")
        self.port: int = int(port)
        self.cwd: str | None = str(cwd) if cwd is not None else None
        self.env: dict[str, str] | None = dict(env) if env is not None else None
        self.stop_timeout = stop_timeout
        self.ready_timeout = ready_timeout
        self.port_timeout = port_timeout
        self.url: str = f"http://{HOST}:{self.port}/"
        self.health_url: str = f"http://{HOST}:{self.port}{HEALTH_PATH}"
        # ``boot_id`` aus der letzten erfolgreichen /api/health-Antwort (falls geliefert).
        self.boot_id: str | None = None
        self.last_exit_code: int | None = None

        self._log: Log = log or _default_log
        self._ignored = [self._resolve(p) for p in (DEFAULT_IGNORE if ignore is None else ignore)]
        self._manual = {
            self._resolve(p) for p in (DEFAULT_MANUAL_RESTART if manual_restart is None else manual_restart)
        }
        self._base_filter = DefaultFilter()
        # Ohne Proxy: Anfragen an 127.0.0.1 dürfen nie über einen System-Proxy laufen.
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

        self._lock = threading.RLock()          # schützt Start-/Stopp-Übergänge und _proc/_job
        self._stop_event = threading.Event()    # gesetzt = Supervisor wird beendet
        self._proc: subprocess.Popen[bytes] | None = None
        self._job: Any = None
        self._restarting = False
        self._initial_pending = False           # run_forever wird den Server gleich erstmals starten
        self._started = False
        self._spawn_time: float | None = None   # Wanduhrzeit des letzten Starts (für mtime-Vergleich)
        self._proc_console = True               # teilt der laufende Server unsere Konsole?
        self._stale_boot_id: str | None = None
        self._job_warned = False
        self._console_warned = False

    # --- öffentliche API ------------------------------------------------------

    @property
    def running(self) -> bool:
        """True, solange der Server-Prozess läuft (sagt nichts über /api/health aus)."""
        proc = self._proc
        return proc is not None and proc.poll() is None

    def start(self) -> None:
        """Startet den Server-Prozess, ohne auf ``/api/health`` zu warten.

        Läuft er bereits, passiert nichts. Wartet bis zu ``port_timeout``
        Sekunden darauf, dass der Port frei wird. Wirft ``RuntimeError``, wenn
        der Port belegt bleibt oder der Supervisor schon gestoppt wurde, und
        ``OSError``, wenn der Befehl nicht startbar ist.
        """
        with self._lock:
            if self._stop_event.is_set():
                raise RuntimeError("Der Supervisor wurde bereits gestoppt.")
            if self.running:
                return
            self._stop_server()  # räumt einen beendeten Prozess samt Job auf
            if not self._wait_port_free(self.port_timeout):
                if self._stop_event.is_set():
                    raise RuntimeError("Der Supervisor wurde während des Starts gestoppt.")
                raise RuntimeError(
                    f"Port {self.port} ist belegt. Läuft NoAssist bereits? "
                    f"Anderen Port wählen oder den belegenden Prozess beenden."
                )
            self._spawn()

    def wait_ready(self, timeout: float = 30.0) -> bool:
        """Wartet, bis ``/api/health`` mit 200 antwortet.

        Gibt ``False`` zurück bei Zeitüberschreitung, wenn der Server-Prozess
        sich beendet hat (z. B. Syntaxfehler beim Start), wenn kein Server
        läuft und auch keiner gerade (neu) gestartet wird oder wenn ``stop()``
        aufgerufen wurde. Liefert der Server eine ``boot_id``, muss sie sich
        von der des vorherigen Servers unterscheiden, damit kein alter Prozess
        fälschlich als bereit gilt.
        """
        deadline = time.monotonic() + max(0.0, timeout)
        while not self._stop_event.is_set():
            proc = self._proc
            if proc is None:
                if not (self._restarting or self._initial_pending):
                    return False
            elif proc.poll() is not None:
                self._reap_if_dead(proc)
                return False
            elif self._probe_health():
                return True
            if time.monotonic() >= deadline:
                return False
            self._stop_event.wait(0.1)
        return False

    def restart(self, reason: str | None = None) -> bool:
        """Stoppt den Server (falls er läuft), startet ihn neu und wartet auf ``/api/health``.

        Gibt ``True`` zurück, wenn der neue Server bereit ist. Fehler werden
        gemeldet, nicht geworfen; danach wird bis zur nächsten Änderung gewartet.
        """
        started_at = time.monotonic()
        with self._lock:
            if self._stop_event.is_set():
                return False
            if reason:
                self._log(reason)
            self._restarting = True
            try:
                self._stop_server()
                if not self._wait_port_free(self.port_timeout):
                    if not self._stop_event.is_set():
                        self._log(
                            f"Port {self.port} ist nach {self.port_timeout:.0f} s noch belegt "
                            f"(verwaister Prozess?). Kein Neustart; warte auf die nächste Änderung ..."
                        )
                    return False
                self._spawn()
            except OSError as exc:
                self._log(f"Server konnte nicht gestartet werden: {exc}. Warte auf die nächste Änderung ...")
                return False
            finally:
                self._restarting = False
        ok = self.wait_ready(self.ready_timeout)
        self._report_ready(ok, started_at, "neu gestartet")
        return ok

    def stop(self) -> None:
        """Beendet die Beobachtung und den Server samt Prozessbaum (idempotent, threadsicher).

        Blockiert, bis der Server beendet ist (höchstens etwa ``stop_timeout``
        Sekunden plus ``taskkill``). Danach ist kein Start mehr möglich.
        """
        self._stop_event.set()
        with self._lock:
            if self._stop_server():
                self._log("Server gestoppt.")

    def run_forever(self) -> None:
        """Beobachtet ``watch_dir`` und startet den Server bei Änderungen neu, bis ``stop()``.

        Wurde der Server noch nie gestartet, startet ``run_forever`` ihn,
        sobald die Beobachtung aktiv ist (Fehler von ``start()`` werden dann
        weitergereicht). Strg+C beendet die Schleife sauber. Beim Verlassen
        wird immer ``stop()`` aufgerufen, damit kein Server zurückbleibt.
        """
        from watchfiles import watch

        initial = not self._started
        self._initial_pending = initial
        first_tick = True
        try:
            for changes in watch(
                self.watch_dir,
                watch_filter=self._watch_filter,
                debounce=_DEBOUNCE_MS,
                stop_event=self._stop_event,
                rust_timeout=_TICK_MS,
                yield_on_timeout=True,
                raise_interrupt=True,
            ):
                if self._stop_event.is_set():
                    break
                if first_tick:
                    # Erst jetzt ist die Beobachtung aktiv; Änderungen ab hier gehen nicht verloren.
                    first_tick = False
                    self._log(f"Beobachte *.py unter {self.watch_dir} ...")
                    if initial:
                        started_at = time.monotonic()
                        try:
                            self.start()
                        except RuntimeError:
                            if self._stop_event.is_set():
                                break  # während des Starts gestoppt: kein Fehler
                            raise
                        finally:
                            self._initial_pending = False
                        self._report_ready(self.wait_ready(self.ready_timeout), started_at, "gestartet")
                        continue  # Änderungen dieses ersten Pakets sind älter als der Start
                    if self._changed_since_spawn():
                        self.restart("Änderungen seit dem Start erkannt - starte Server neu ...")
                        continue
                self._reap_if_dead()
                if changes:
                    self._handle_changes(changes)
        except KeyboardInterrupt:
            self._log("Abbruch per Strg+C.")
        finally:
            self._initial_pending = False
            self.stop()

    # --- Dateiänderungen --------------------------------------------------------

    def _resolve(self, path: str | Path) -> str:
        """Normalisiert einen Pfad aus ``ignore``/``manual_restart`` (relativ zu ``watch_dir``)."""
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = self.watch_dir / candidate
        return _norm(candidate)

    def _is_manual(self, path: str) -> bool:
        return _norm(path) in self._manual

    def _is_ignored(self, path: str) -> bool:
        key = _norm(path)
        return any(key == entry or key.startswith(entry + os.sep) for entry in self._ignored)

    def _watch_filter(self, change: Any, path: str) -> bool:
        """watchfiles-Filter: nur ``*.py``, ohne ``__pycache__`` usw. und ohne ``ignore``."""
        if not path.lower().endswith(".py") or not self._base_filter(change, path):
            return False
        return self._is_manual(path) or not self._is_ignored(path)

    def _handle_changes(self, changes: set[tuple[Any, str]]) -> None:
        restart_files: set[str] = set()
        manual_files: set[str] = set()
        for _change, path in changes:
            rel = _rel_posix(path, self.watch_dir)
            (manual_files if self._is_manual(path) else restart_files).add(rel)
        for rel in sorted(manual_files):
            self._log(
                f"Hinweis: {rel} wurde geändert. Diese Datei läuft im Fenster-Prozess und wird "
                f"nicht neu geladen - bitte NoAssist manuell neu starten."
            )
        if restart_files:
            names = sorted(restart_files)
            shown = ", ".join(names[:5]) + (f" (+{len(names) - 5} weitere)" if len(names) > 5 else "")
            self.restart(f"Änderung erkannt: {shown} - starte Server neu ...")

    def _changed_since_spawn(self) -> bool:
        """True, wenn eine beobachtete ``*.py`` seit dem letzten Start geändert wurde.

        Deckt die Lücke ab, wenn ``start()`` vor ``run_forever()`` aufgerufen
        wurde und die Beobachtung erst danach aktiv wurde.
        """
        since = self._spawn_time
        if since is None or not self.running:
            return False
        for root, dirs, files in os.walk(self.watch_dir):
            dirs[:] = [d for d in dirs if d != "__pycache__" and not self._is_ignored(os.path.join(root, d))]
            for name in files:
                path = os.path.join(root, name)
                if not self._watch_filter(None, path) or self._is_manual(path):
                    continue
                try:
                    if os.stat(path).st_mtime >= since:
                        return True
                except OSError:
                    continue
        return False

    # --- Prozesssteuerung (nur mit gehaltenem self._lock) ------------------------

    def _spawn(self) -> None:
        """Startet den Server-Prozess (Windows: eigene Prozessgruppe + Job-Object)."""
        self._stale_boot_id = self.boot_id
        self.boot_id = None
        self._spawn_time = time.time()
        env = self._child_env()
        if sys.platform != "win32":  # POSIX-Fallback (ungetestet): eigene Sitzung = eigene Prozessgruppe
            self._proc = subprocess.Popen(self.cmd, cwd=self.cwd, env=env, start_new_session=True)
            self._started = True
            self._log(f"Server-Prozess gestartet (PID {self._proc.pid}).")
            return

        job = None
        try:
            job = _create_kill_on_close_job()
        except Exception as exc:
            if not self._job_warned:
                self._log(f"Warnung: Job-Object nicht verfügbar ({type(exc).__name__}: {exc}); "
                          f"der Server-Baum wird nur per taskkill beendet.")
                self._job_warned = True

        console = _has_console()
        flags = subprocess.CREATE_NEW_PROCESS_GROUP
        if not console:
            flags |= subprocess.CREATE_NO_WINDOW  # kein aufpoppendes Konsolenfenster je Neustart
        if job is not None:
            # Angehalten starten, damit der Prozess im Job ist, bevor er Kinder
            # erzeugen kann (der venv-Launcher startet sofort das echte python.exe).
            flags |= _CREATE_SUSPENDED
        try:
            proc = subprocess.Popen(self.cmd, cwd=self.cwd, env=env, creationflags=flags)
        except BaseException:
            if job is not None:
                job.Close()
            raise

        if job is not None:
            try:
                _assign_to_job(job, proc.pid)
            except Exception as exc:
                self._log(f"Warnung: Server konnte keinem Job-Object zugeordnet werden ({exc}).")
                job.Close()
                job = None
            try:
                _resume_process(proc.pid)
            except Exception as exc:
                proc.kill()
                proc.wait()
                if job is not None:
                    job.Close()
                raise OSError(f"Server-Prozess konnte nicht fortgesetzt werden: {exc}") from exc

        self._proc = proc
        self._job = job
        self._proc_console = console
        self._started = True
        self._log(f"Server-Prozess gestartet (PID {proc.pid}).")

    def _child_env(self) -> dict[str, str]:
        """Umgebung des Servers: ``env`` bzw. die eigene, plus ``PYTHONDONTWRITEBYTECODE=1``.

        Python prüft ``.pyc``-Dateien nur über Änderungszeit (ganze Sekunden)
        und Größe. Zwei gleich große Fassungen einer Datei innerhalb derselben
        Sekunde würden sonst nach dem Neustart den alten Code laden. Ohne
        geschriebene ``.pyc`` wird ``src`` bei jedem Start frisch übersetzt
        (site-packages sind bereits vorübersetzt). Ein vom Aufrufer gesetzter
        Wert hat Vorrang.
        """
        env = dict(os.environ if self.env is None else self.env)
        env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
        return env

    def _stop_server(self) -> bool:
        """Beendet den aktuellen Server samt Baum; True, wenn er noch lief."""
        proc, job = self._proc, self._job
        self._proc = None
        self._job = None
        if proc is None:
            if job is not None:
                self._cleanup_job(job, grace=0.0)
            return False
        was_running = proc.poll() is None
        self._terminate(proc, job, self._proc_console)
        self.last_exit_code = proc.returncode
        return was_running

    def _terminate(self, proc: subprocess.Popen[bytes], job: Any, console: bool) -> None:
        """CTRL_BREAK, bis ``stop_timeout`` warten, dann ``taskkill /T /F``; danach Job leeren."""
        pid = proc.pid
        if proc.poll() is None:
            if self._request_shutdown(proc, console) and not self._wait_exit(proc, self.stop_timeout):
                self._log(f"Server (PID {pid}) hat sich nach {self.stop_timeout:.0f} s nicht beendet; "
                          f"beende Prozessbaum hart.")
            if proc.poll() is None:
                self._kill_tree(proc)
                if not self._wait_exit(proc, 5.0):
                    self._log(f"Warnung: PID {pid} läuft nach dem harten Beenden noch.")
        if job is not None:
            self._cleanup_job(job, grace=_JOB_GRACE_S)

    def _request_shutdown(self, proc: subprocess.Popen[bytes], console: bool) -> bool:
        """Bittet den Server um sauberes Beenden (Windows: CTRL_BREAK_EVENT an seine Prozessgruppe).

        Gibt ``False`` zurück, wenn das nicht möglich ist; dann wird sofort hart beendet.
        """
        try:
            if sys.platform != "win32":
                os.killpg(proc.pid, signal.SIGTERM)
            elif console:
                os.kill(proc.pid, signal.CTRL_BREAK_EVENT)
            else:
                if not self._console_warned:
                    self._log("Keine Konsole vorhanden: CTRL_BREAK ist nicht möglich, "
                              "der Server wird bei jedem Neustart hart beendet.")
                    self._console_warned = True
                return False
        except OSError as exc:
            self._log(f"Sauberes Beenden von PID {proc.pid} nicht möglich ({exc}); beende hart.")
            return False
        return True

    @staticmethod
    def _wait_exit(proc: subprocess.Popen[bytes], timeout: float) -> bool:
        try:
            proc.wait(timeout)
        except subprocess.TimeoutExpired:
            return False
        return True

    def _kill_tree(self, proc: subprocess.Popen[bytes]) -> None:
        """``taskkill /PID <pid> /T /F`` (POSIX: SIGKILL an die Prozessgruppe)."""
        if sys.platform != "win32":
            with contextlib.suppress(OSError):
                os.killpg(proc.pid, signal.SIGKILL)
            return
        try:
            subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=15,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            self._log(f"taskkill fehlgeschlagen ({exc}); beende nur den Hauptprozess.")
            proc.kill()

    def _cleanup_job(self, job: Any, grace: float) -> None:
        """Gibt Restprozessen im Job ``grace`` Sekunden, beendet den Rest und schließt den Job."""
        import win32job

        try:
            deadline = time.monotonic() + grace
            remaining = _job_active_processes(job)
            while remaining and time.monotonic() < deadline:
                time.sleep(0.05)
                remaining = _job_active_processes(job)
            if remaining:
                win32job.TerminateJobObject(job, 1)
                self._log(f"{remaining} verbliebene(n) Prozess(e) des Servers über das Job-Object beendet.")
        except Exception as exc:
            self._log(f"Warnung: Job-Object konnte nicht geleert werden ({exc}).")
        finally:
            job.Close()  # KILL_ON_JOB_CLOSE: spätestens hier endet alles im Job

    def _wait_port_free(self, timeout: float) -> bool:
        """Wartet bis zu ``timeout`` Sekunden, bis der Port frei ist (bricht bei ``stop()`` ab)."""
        deadline = time.monotonic() + max(0.0, timeout)
        announced = False
        while not self._stop_event.is_set():
            if port_is_free(self.port):
                return True
            if not announced:
                self._log(f"Port {self.port} ist noch belegt, warte ...")
                announced = True
            if time.monotonic() >= deadline:
                return False
            self._stop_event.wait(0.1)
        return False

    # --- Zustand und Gesundheit ---------------------------------------------------

    def _reap_if_dead(self, expected: subprocess.Popen[bytes] | None = None) -> bool:
        """Meldet einen beendeten Server genau einmal und räumt seinen Job auf.

        Kein automatischer Neustart: Es wird bis zur nächsten Änderung gewartet.
        Hält gerade ein anderer Thread die Sperre (Neustart oder Stopp), passiert
        nichts; der kümmert sich dann um den Prozess.
        """
        if self._stop_event.is_set() or not self._lock.acquire(blocking=False):
            return False
        try:
            proc = self._proc
            if proc is None or (expected is not None and proc is not expected):
                return False
            code = proc.poll()
            if code is None:
                return False
            job = self._job
            self._proc = None
            self._job = None
            self.last_exit_code = code
            if job is not None:
                self._cleanup_job(job, grace=0.0)
            self._log(
                f"Server-Prozess beendet (Exit-Code {_fmt_exit_code(code)}). "
                f"Kein automatischer Neustart; warte auf die nächste Änderung ..."
            )
            return True
        finally:
            self._lock.release()

    def _probe_health(self) -> bool:
        """Eine Prüfung von ``/api/health``; True bei 200 (und neuer ``boot_id``)."""
        # Erst ein kurzer TCP-Connect: Ein abgelehnter Connect dauert unter Windows
        # ~2 s, mit kurzem Timeout wird stattdessen schnell neu versucht.
        try:
            with socket.create_connection((HOST, self.port), timeout=_CONNECT_PROBE_S):
                pass
        except OSError:
            return False
        try:
            with self._opener.open(self.health_url, timeout=_HEALTH_HTTP_TIMEOUT_S) as response:
                if response.status != 200:
                    return False
                raw = response.read(65536)
        except (OSError, http.client.HTTPException, ValueError):
            return False
        boot_id: str | None = None
        try:
            data = json.loads(raw)
        except ValueError:
            data = None
        if isinstance(data, dict) and isinstance(data.get("boot_id"), str):
            boot_id = data["boot_id"]
        if boot_id is not None and boot_id == self._stale_boot_id:
            return False  # antwortet noch der alte Server?
        self.boot_id = boot_id
        return True

    def _report_ready(self, ok: bool, started_at: float, verb: str) -> None:
        if ok:
            self._log(f"Server {verb} in {time.monotonic() - started_at:.1f} s: {self.url}")
        elif not self._stop_event.is_set() and self.running:
            self._log(
                f"Server antwortet nach {self.ready_timeout:.0f} s nicht auf {HEALTH_PATH}. "
                f"Warte auf die nächste Änderung ..."
            )
        # Prozess beendet: das meldet bereits _reap_if_dead.


# --- Demo -----------------------------------------------------------------------

def _dummy_server(port: int) -> None:
    """Minimaler Server mit ``/api/health`` (nur für die Demo)."""
    import http.server
    import uuid

    boot_id = uuid.uuid4().hex

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 (Name von http.server vorgegeben)
            if self.path != HEALTH_PATH:
                self.send_error(404)
                return
            body = json.dumps({"app": "noassist", "boot_id": boot_id, "dev": True}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            pass

    server = http.server.ThreadingHTTPServer((HOST, port), Handler)
    print(f"[demo-server] PID {os.getpid()} auf http://{HOST}:{port}{HEALTH_PATH} (boot_id {boot_id[:8]})",
          flush=True)
    server.serve_forever()


def _demo(argv: list[str] | None = None) -> None:
    import argparse

    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        prog="python -m devreload",
        description="Demo des Dev-Supervisors: startet einen Server und startet ihn bei "
                    "Änderungen an *.py neu. Ohne Befehl wird ein eingebauter Dummy-Server verwendet.",
    )
    parser.add_argument("--port", type=int, default=8765, help="Port des Servers (Standard: 8765)")
    parser.add_argument("--watch", default=str(here), help="beobachtetes Verzeichnis (Standard: src)")
    parser.add_argument("--dummy-server", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("cmd", nargs=argparse.REMAINDER, help="Server-Befehl nach '--'")
    args = parser.parse_args(argv)

    if args.dummy_server:
        _dummy_server(args.port)
        return

    cmd = args.cmd[1:] if args.cmd[:1] == ["--"] else args.cmd
    if not cmd:
        cmd = [sys.executable, str(Path(__file__).resolve()), "--dummy-server", "--port", str(args.port)]

    supervisor = Supervisor(cmd, args.watch, args.port)
    stop_static = threading.Event()
    static_dir = Path(args.watch) / "web" / "static"
    if static_dir.is_dir():
        watch_static(
            static_dir,
            lambda files, css_only: _default_log(
                f"Statische Dateien geändert (css_only={css_only}): {', '.join(files)}"
            ),
            stop_static,
        )
    _default_log("Speichere eine .py-Datei im beobachteten Verzeichnis, um einen Neustart auszulösen. "
                 "Strg+C beendet.")
    try:
        supervisor.run_forever()
    finally:
        stop_static.set()


if __name__ == "__main__":
    _demo()
