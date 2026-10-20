"""Browser-Sitzungen der Agenten (Playwright).

Jeder Chat (Haupt-Chat, Subagent, Assistent Agent) hat seinen eigenen Browser, der zwischen den
Tool-Aufrufen erhalten bleibt. Alle Browser laufen in einem Hintergrund-Thread mit eigener Event-Loop;
die Tools rufen ihn synchron über ``manager.call`` auf.

Standardmäßig läuft der Browser unsichtbar (headless). ``manager.show(chat_id, True)`` startet ihn als
sichtbares Fenster neu und übernimmt Cookies/Anmeldung und die aktuelle Adresse (Chromium kann die
Sichtbarkeit nicht im laufenden Betrieb umschalten). Schließt der Nutzer das Fenster, ist die Sitzung
beendet; der nächste Tool-Aufruf startet einen neuen, unsichtbaren Browser.
"""

from __future__ import annotations

import asyncio
import logging
import re
import tempfile
import threading
import time
from collections import deque
from collections.abc import Awaitable, Callable
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

DEFAULT_KEY = "default"
"""Sitzung für Aufrufe ohne Chat."""
IDLE_CLOSE_S = 1800
"""Unsichtbare Browser, die so lange unbenutzt sind, werden beendet."""
CALL_TIMEOUT_S = 90
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/119.0.0.0 Safari/537.36"
)


class NoBrowser(LookupError):
    """Für diesen Chat läuft kein Browser."""


@dataclass(eq=False)
class _Session:
    browser: Any
    context: Any
    page: Any
    visible: bool
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    url: str = ""
    last_used: float = field(default_factory=time.monotonic)
    closing: bool = False
    notes: deque = field(default_factory=lambda: deque(maxlen=20))
    """Meldungen, die der Nutzer im Browser bemerken würde (Dialoge, neue Tabs, Downloads)."""

    def note(self, text: str) -> None:
        self.notes.append(text)


class BrowserManager:
    """Verwaltet die Browser aller Chats (threadsicher)."""

    def __init__(self) -> None:
        self._sessions: dict[str, _Session] = {}
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._start_lock = threading.Lock()
        self._create_lock: asyncio.Lock | None = None
        self._playwright: Any = None
        self.download_dir: Path | None = None

    # -- Event-Loop --

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        with self._start_lock:
            if self._loop is None or self._thread is None or not self._thread.is_alive():
                self._ready.clear()
                self._sessions.clear()
                self._playwright = None
                self._thread = threading.Thread(target=self._run_loop, name="browser-loop", daemon=True)
                self._thread.start()
                if not self._ready.wait(10):
                    raise RuntimeError("Der Browser-Dienst konnte nicht gestartet werden.")
            assert self._loop is not None
            return self._loop

    def _run_loop(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        self._create_lock = asyncio.Lock()
        loop.create_task(self._reaper())
        self._ready.set()
        try:
            loop.run_forever()
        finally:
            loop.close()

    def _run(self, coro: Awaitable[Any], timeout: float) -> Any:
        loop = self._ensure_loop()
        future = asyncio.run_coroutine_threadsafe(coro, loop)  # type: ignore[arg-type]
        try:
            return future.result(timeout)
        except FutureTimeout:
            future.cancel()
            raise TimeoutError(f"Der Browser hat nicht innerhalb von {int(timeout)} s geantwortet.") from None

    # -- Start und Sitzungen (laufen in der Browser-Loop) --

    async def _launch(self, visible: bool, storage_state: Any = None) -> _Session:
        if self._playwright is None:
            from playwright.async_api import async_playwright

            self._playwright = await async_playwright().start()
        browser = await self._playwright.chromium.launch(headless=not visible)
        try:
            options: dict[str, Any] = {"user_agent": USER_AGENT}
            if visible:
                options["no_viewport"] = True  # Seite füllt das Fenster, auch beim Vergrößern
            if storage_state:
                options["storage_state"] = storage_state
            context = await browser.new_context(**options)
            page = await context.new_page()
        except BaseException:
            await browser.close()
            raise
        session = _Session(browser=browser, context=context, page=page, visible=visible)
        browser.on("disconnected", lambda *_: self._on_disconnected(session))
        self._wire_page(session, page)
        context.on("page", lambda new: self._on_new_page(session, new))
        return session

    def _wire_page(self, session: _Session, page: Any) -> None:
        page.on("dialog", lambda d: asyncio.ensure_future(self._on_dialog(session, d)))
        page.on("download", lambda d: asyncio.ensure_future(self._on_download(session, d)))

    def _on_new_page(self, session: _Session, page: Any) -> None:
        """Ein neuer Tab (z. B. Link mit target=_blank) wird aktiv, wie bei einem Nutzer."""
        self._wire_page(session, page)
        session.page = page
        session.note("Ein neuer Tab wurde geöffnet und ist jetzt aktiv.")

    async def _on_dialog(self, session: _Session, dialog: Any) -> None:
        kinds = {"alert": "Hinweis", "confirm": "Bestätigung", "prompt": "Eingabe", "beforeunload": "Verlassen-Warnung"}
        session.note(f"{kinds.get(dialog.type, dialog.type)}-Dialog der Seite automatisch bestätigt: {dialog.message[:300]!r}")
        try:
            await dialog.accept()
        except Exception:  # noqa: BLE001 - schon geschlossen
            pass

    async def _on_download(self, session: _Session, download: Any) -> None:
        try:
            folder = self.download_dir or Path(tempfile.gettempdir()) / "noassist-downloads"
            folder.mkdir(parents=True, exist_ok=True)
            name = Path(download.suggested_filename or "download").name or "download"
            target, n = folder / name, 2
            while target.exists():
                target = folder / f"{Path(name).stem}-{n}{Path(name).suffix}"
                n += 1
            await download.save_as(target)
            session.note(f"Download gespeichert: {target}")
        except Exception as exc:  # noqa: BLE001
            session.note(f"Download fehlgeschlagen: {exc}")

    def _on_disconnected(self, session: _Session) -> None:
        if session.closing:
            return
        for key, current in list(self._sessions.items()):
            if current is session:
                del self._sessions[key]

    async def _close(self, session: _Session) -> None:
        session.closing = True
        for closer in (session.context.close, session.browser.close):
            try:
                await closer()
            except Exception:  # noqa: BLE001 - schon beendet (z. B. Fenster vom Nutzer geschlossen)
                pass

    async def _get(self, key: str) -> _Session:
        assert self._create_lock is not None
        async with self._create_lock:
            session = self._sessions.get(key)
            if session is None:
                session = await self._launch(False)
                self._sessions[key] = session
            return session

    async def _use(self, key: str, op: Callable[[Any], Awaitable[Any]], with_session: bool = False) -> Any:
        session = await self._get(key)
        async with session.lock:
            if session.page.is_closed():
                pages = [p for p in session.context.pages if not p.is_closed()]
                session.page = pages[-1] if pages else await session.context.new_page()
            try:
                return await op(session if with_session else session.page)
            finally:
                session.last_used = time.monotonic()
                try:
                    session.url = session.page.url
                except Exception:  # noqa: BLE001
                    pass

    async def _set_visible(self, key: str, visible: bool) -> None:
        session = self._sessions.get(key)
        if session is None:
            raise NoBrowser("Für diesen Chat läuft kein Browser.")
        async with session.lock:
            if session.visible == visible or session.closing:
                return
            state = await session.context.storage_state()
            url = session.page.url if not session.page.is_closed() else session.url
            fresh = await self._launch(visible, state)
            try:
                if url and url != "about:blank":
                    await fresh.page.goto(url, wait_until="domcontentloaded", timeout=30000)
            except Exception as exc:  # noqa: BLE001 - die Seite bleibt erreichbar, nur ohne Adresse
                log.warning("Adresse %s nach dem Umschalten nicht geladen: %s", url, exc)
            fresh.url = fresh.page.url
            self._sessions[key] = fresh
            await self._close(session)

    async def _drop(self, key: str) -> None:
        session = self._sessions.pop(key, None)
        if session is not None:
            await self._close(session)

    async def _shutdown(self) -> None:
        for key in list(self._sessions):
            await self._drop(key)
        if self._playwright is not None:
            try:
                await self._playwright.stop()
            except Exception:  # noqa: BLE001
                pass
            self._playwright = None

    async def _reaper(self) -> None:
        while True:
            await asyncio.sleep(60)
            now = time.monotonic()
            for key, session in list(self._sessions.items()):
                if not session.visible and not session.lock.locked() and now - session.last_used > IDLE_CLOSE_S:
                    log.info("Browser von %s wegen Inaktivität beendet", key)
                    await self._drop(key)

    # -- API für Tools und Server (synchron) --

    def call(
        self, key: str | None, op: Callable[[Any], Awaitable[Any]], timeout: float = CALL_TIMEOUT_S, session: bool = False
    ) -> Any:
        """Führt ``await op(page)`` im Browser des Chats ``key`` aus (startet ihn bei Bedarf).

        Mit ``session=True`` bekommt ``op`` statt der Seite die Sitzung (``.page``, ``.context``) – für Tabs.
        """
        return self._run(self._use(key or DEFAULT_KEY, op, session), timeout)

    def pop_notes(self, key: str | None) -> list[str]:
        """Holt und leert die Meldungen der Sitzung (neue Tabs, Dialoge, Downloads)."""
        session = self._sessions.get(key or DEFAULT_KEY)
        notes: list[str] = []
        while session is not None and session.notes:
            try:
                notes.append(session.notes.popleft())
            except IndexError:
                break
        return notes

    def close_now(self, key: str | None) -> bool:
        """Beendet den Browser von ``key`` und wartet darauf. ``False``, wenn keiner lief."""
        key = key or DEFAULT_KEY
        if key not in self._sessions:
            return False
        self._run(self._drop(key), 15)
        return True

    def state(self, key: str | None) -> dict[str, Any]:
        """``{"active", "visible", "url"}`` des Browsers von ``key`` (ohne ihn zu starten)."""
        session = self._sessions.get(key or DEFAULT_KEY)
        if session is None or session.closing:
            return {"active": False, "visible": False, "url": ""}
        return {"active": True, "visible": session.visible, "url": session.url}

    def show(self, key: str, visible: bool) -> dict[str, Any]:
        """Öffnet den Browser als sichtbares Fenster (``True``) oder versteckt ihn wieder (``False``).

        Wirft ``NoBrowser``, wenn für ``key`` kein Browser läuft.
        """
        if self._loop is None or key not in self._sessions:
            raise NoBrowser("Für diesen Chat läuft kein Browser.")
        self._run(self._set_visible(key, bool(visible)), 60)
        return self.state(key)

    def close_chat(self, key: str) -> None:
        """Beendet den Browser eines Chats, ohne zu warten."""
        loop = self._loop
        if loop is not None and key in self._sessions and self._thread is not None and self._thread.is_alive():
            asyncio.run_coroutine_threadsafe(self._drop(key), loop)

    def shutdown(self, timeout: float = 8) -> None:
        """Beendet alle Browser und den Hintergrund-Thread."""
        loop, thread = self._loop, self._thread
        if loop is None or thread is None or not thread.is_alive():
            return
        try:
            self._run(self._shutdown(), timeout)
        except Exception:  # noqa: BLE001
            log.warning("Browser konnten nicht sauber beendet werden", exc_info=True)
        loop.call_soon_threadsafe(loop.stop)


manager = BrowserManager()
