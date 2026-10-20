"""MCP-Tools als ``ToolSpec``s, mit den Original-JSON-Schemas der Server.

``McpServer`` (mcp_client.py) legt MCP-Tools nur als ``lms.ToolFunctionDef``
ab; dabei gehen Parameterbeschreibungen und verschachtelte Schemas verloren.
``McpToolSource`` merkt sich zusätzlich das Original-``input_schema`` jedes
Tools und liefert über ``specs()`` vollständige ``ToolSpec``s für alle
Provider. ``McpServer`` selbst bleibt unverändert.

Außerdem macht ``McpToolSource`` den Server robust gegen Hänger:

- **Zeitlimit für Tool-Aufrufe** (``call_timeout``, Standard 120 s): Antwortet
  der Server nicht rechtzeitig, wird der Aufruf abgebrochen (der Server bekommt
  eine Abbruch-Benachrichtigung) und das Tool-Ergebnis lautet
  ``"Fehler: Zeitüberschreitung nach N s: ..."``. Ohne Limit würde ein hängender
  Server den Turn blockieren und beim LM-Studio-Provider, der während
  ``act()`` seinen Vorhersage-Lock hält, auch alle weiteren Turns.
- **Start mit Zeitlimit** (``start(timeout=...)`` → ``McpStartTimeout``),
  **Abbruch** aus einem anderen Thread (``abort()`` → ``McpStartAborted``) und
  **Aufräumen** auch nach einem gescheiterten Start (``dispose()``: Event-Loop
  und ihr Thread werden beendet). ``McpServer`` kann das nicht von sich aus.
"""

import asyncio
import logging
import re
import threading
from collections.abc import Coroutine, Iterable, Mapping
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from mcp_client import McpServer, _attr
from toolset import NAME_RE, ToolSpec, clean_tool_name, spec_from_lms_def

if TYPE_CHECKING:
    from concurrent.futures import Future

    import lmstudio as lms

log = logging.getLogger(__name__)

_PREFIX_MAX = 24

DEFAULT_CALL_TIMEOUT_S = 120.0
"""Standard-Zeitlimit (Sekunden) für einen einzelnen Tool-Aufruf."""
DISPOSE_TIMEOUT_S = 3.0
"""So lange wartet ``dispose()`` auf das Beenden der Verbindung (z. B. des Server-Prozesses)."""


class McpStartTimeout(TimeoutError):
    """Der MCP-Server hat sich nicht innerhalb des Zeitlimits von ``start()`` gemeldet."""


class McpStartAborted(RuntimeError):
    """Der Start wurde mit ``abort()`` abgebrochen."""


class _CallTimeout(Exception):
    """Intern: ein Tool-Aufruf hat das Zeitlimit überschritten."""


class _NotConnected(Exception):
    """Intern: die Event-Loop läuft nicht (mehr), der Aufruf ist nicht möglich."""


def server_prefix(name: str | None) -> str:
    """Kurzer, gültiger Präfix aus dem Servernamen (z. B. ``"my-server"``)."""
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", name or "").strip("_-")
    return cleaned[:_PREFIX_MAX].rstrip("_-") or "mcp"


class McpToolSource(McpServer):
    """``McpServer``, der zusätzlich die Original-Schemas der Tools behält.

    Nutzung wie ``McpServer``::

        src = McpToolSource(command="npx", args=["-y", "..."], name="files").start(timeout=30)
        for spec in src.specs(reserved=toolset.names()):
            toolset.add(spec)
        ...
        src.close()

    ``call_timeout``: Zeitlimit je Tool-Aufruf in Sekunden (``None`` = ohne Limit).
    """

    def __init__(
        self,
        command: str | None = None,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        url: str | None = None,
        name: str | None = None,
        *,
        call_timeout: float | None = DEFAULT_CALL_TIMEOUT_S,
    ):
        if call_timeout is not None and (
            isinstance(call_timeout, bool) or not isinstance(call_timeout, (int, float)) or call_timeout <= 0
        ):
            raise ValueError("call_timeout muss eine positive Zahl (Sekunden) oder None sein.")
        self.schemas: dict[str, dict[str, Any]] = {}
        """Original-``input_schema`` je MCP-Tool-Name (wie vom Server geliefert)."""
        self.call_timeout: float | None = float(call_timeout) if call_timeout is not None else None
        self._ctl_lock = threading.Lock()  # schützt _lifecycle_task, _disposed und den Abbruch
        self._aborted = threading.Event()
        self._disposed = False
        self._lifecycle_task: Future[None] | None = None
        super().__init__(command=command, args=args, env=env, url=url, name=name)
        self._thread.name = f"mcp-loop-{server_prefix(self.name)}"

    @property
    def source(self) -> str:
        """Quellangabe für ``ToolSpec.source`` und Events: ``"mcp:<server>"``."""
        return f"mcp:{self.name}"

    # -- Lebenszyklus --

    def start(self, timeout: float | None = None) -> "McpToolSource":
        """Verbindet mit dem Server und lädt die Tools (blockiert).

        ``timeout``: höchstens so viele Sekunden warten (``None`` = unbegrenzt).
        Scheitert der Start, wird alles abgeräumt (``dispose``) und geworfen:
        ``McpStartTimeout`` bei Zeitüberschreitung, ``McpStartAborted`` nach
        ``abort()``, sonst die ursprüngliche Ausnahme (z. B. ``FileNotFoundError``).
        ``start()`` ist nur einmal möglich.
        """
        with self._ctl_lock:
            if self._lifecycle_task is not None or self._disposed:
                raise RuntimeError(f"MCP-Server {self.name}: start() ist nur einmal möglich.")
            aborted = self._aborted.is_set()
            if not aborted:
                self._lifecycle_task = asyncio.run_coroutine_threadsafe(self._lifecycle(), self._loop)
            task = self._lifecycle_task
        if aborted:
            self.dispose()
            raise McpStartAborted(f"Der Start von MCP-Server {self.name} wurde abgebrochen.")

        ready = self._ready.wait(timeout)  # abort() weckt ebenfalls auf
        if self._aborted.is_set():
            self.dispose()
            raise McpStartAborted(f"Der Start von MCP-Server {self.name} wurde abgebrochen.")
        if not ready:
            self.dispose()
            assert timeout is not None
            raise McpStartTimeout(
                f"Der MCP-Server {self.name} hat sich nicht innerhalb von {timeout:g} s gemeldet."
            )
        if self._error:
            # _lifecycle setzt _ready schon im finally, also bevor der Task fertig ist.
            # Erst auf das Ende warten, sonst bricht dispose() den Task ab und aus der
            # ursprünglichen Ausnahme würde ein CancelledError.
            exc: BaseException | None = None
            assert task is not None
            try:
                task.result(timeout=DISPOSE_TIMEOUT_S)
            except BaseException as e:  # noqa: BLE001 - wird unten weitergeworfen
                exc = e if task.done() else None
            self.dispose()
            if exc is not None:
                raise exc
            raise RuntimeError(f"Der Start von MCP-Server {self.name} ist fehlgeschlagen.")
        return self

    def abort(self) -> None:
        """Bricht einen laufenden oder noch bevorstehenden Start ab (threadsicher, idempotent).

        ``start()`` wirft dann ``McpStartAborted`` und räumt selbst auf. Eine
        schon bestehende Verbindung wird hart getrennt; zum regulären Beenden
        dient ``close()``.
        """
        with self._ctl_lock:
            self._aborted.set()
            task = self._lifecycle_task
        if task is not None and not task.done():
            task.cancel()
        self._ready.set()

    def close(self) -> None:
        """Beendet die Verbindung regulär und räumt danach alles ab (idempotent).

        Lief kein erfolgreicher Start oder scheitert das reguläre Beenden,
        wird hart abgeräumt (``dispose``).
        """
        with self._ctl_lock:
            if self._disposed:
                return
            task = self._lifecycle_task
            connected = task is not None and self._ready.is_set() and not self._error and not self._aborted.is_set()
        if connected:
            assert task is not None
            # Wie McpServer.close(), aber die Loop bleibt für dispose() an (offene Aufrufe abräumen).
            try:
                self._loop.call_soon_threadsafe(self._stop.set)
                task.result(timeout=10)
            except Exception as e:  # noqa: BLE001 - danach wird hart abgeräumt
                log.warning("MCP-Server %s ließ sich nicht sauber schließen (%s: %s); wird hart beendet",
                            self.name, type(e).__name__, e)
        self.dispose()

    def dispose(self, timeout: float = DISPOSE_TIMEOUT_S) -> None:
        """Räumt alles ab, auch nach gescheitertem oder abgebrochenem Start (idempotent).

        Bricht die Verbindung ab, wartet bis zu ``timeout`` Sekunden auf offene
        Tasks (z. B. das Beenden des Server-Prozesses), bricht übrige ab, stoppt
        die Event-Loop, beendet ihren Thread und schließt sie. Nicht aus dem
        Loop-Thread selbst aufrufen.
        """
        with self._ctl_lock:
            if self._disposed:
                return
            self._disposed = True
            task = self._lifecycle_task
        if task is not None and not task.done():
            task.cancel()
        loop = self._loop

        async def settle() -> None:
            me = asyncio.current_task()
            pending = [t for t in asyncio.all_tasks() if t is not me]
            if not pending:
                return
            _, still = await asyncio.wait(pending, timeout=timeout)
            for t in still:
                t.cancel()
            if still:
                await asyncio.wait(still, timeout=0.5)

        try:
            if loop.is_running():
                asyncio.run_coroutine_threadsafe(settle(), loop).result(timeout + 2.0)
        except Exception:  # Aufräumen darf nicht scheitern
            log.debug("MCP-Server %s: Event-Loop ließ sich nicht leeren", self.name, exc_info=True)
        try:
            loop.call_soon_threadsafe(loop.stop)
        except RuntimeError:
            pass  # Loop schon geschlossen
        self._thread.join(2.0)
        if self._thread.is_alive():
            log.warning("MCP-Server %s: Event-Loop-Thread endet nicht", self.name)
            return
        try:
            loop.close()
        except Exception:
            log.debug("MCP-Server %s: Event-Loop ließ sich nicht schließen", self.name, exc_info=True)

    # -- Tool-Aufrufe --

    def _run(self, coro: Coroutine[Any, Any, Any]) -> Any:
        """Führt ``coro`` in der Event-Loop des Servers aus, höchstens ``call_timeout`` Sekunden.

        Bei Zeitüberschreitung wird der Aufruf abgebrochen (``_CallTimeout``);
        läuft die Loop nicht mehr, gibt es sofort ``_NotConnected``.
        """
        if self._disposed or not self._loop.is_running():
            coro.close()
            raise _NotConnected
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        try:
            return fut.result(self.call_timeout)
        except TimeoutError:
            if fut.done() and not fut.cancelled():
                raise  # Zeitüberschreitung im Transport selbst, nicht unser Limit
            fut.cancel()  # bricht den Task in der Loop ab; mcp schickt dem Server ein "cancelled"
            raise _CallTimeout from None

    def _call(self, tool_name: str, arguments: dict[str, Any]) -> str:
        try:
            return super()._call(tool_name, arguments)
        except _CallTimeout:
            limit = self.call_timeout or 0.0
            log.warning("MCP-Server %s: Tool %s antwortet nicht innerhalb von %g s; Aufruf abgebrochen",
                        self.name, tool_name, limit)
            return (
                f"Fehler: Zeitüberschreitung nach {limit:g} s: Das MCP-Tool {tool_name!r} "
                f"(Server {self.name}) hat nicht rechtzeitig geantwortet; der Aufruf wurde abgebrochen."
            )
        except _NotConnected:
            return f"Fehler: Der MCP-Server {self.name} ist nicht (mehr) verbunden."

    # -- Schemas --

    def _wrap(self, mcp_tool: Any) -> "lms.ToolFunctionDef":
        tool_def = super()._wrap(mcp_tool)
        schema = _attr(mcp_tool, "input_schema", "inputSchema")
        self.schemas[mcp_tool.name] = dict(schema) if isinstance(schema, Mapping) else {}
        return tool_def

    def specs(self, reserved: Iterable[str] | None = None) -> list[ToolSpec]:
        """Die Tools des Servers als ``ToolSpec``s (mit Original-Schema).

        Ein Tool bekommt den Präfix ``<server>_``, wenn sein Name ungültige
        Zeichen enthält (erlaubt: ``^[A-Za-z0-9_-]{1,48}$``) oder mit einem
        Namen aus ``reserved`` kollidiert. ``reserved`` sind die bereits
        vergebenen Namen, z. B. ``toolset.names()``; Standard sind die Namen der
        lokalen ``@tool``-Funktionen. Restliche Konflikte (etwa durch Kürzen)
        löst ``ToolSet.add`` mit ``_2``, ``_3`` … auf.

        Aufgerufen wird das MCP-Tool immer unter seinem Originalnamen.
        """
        if reserved is None:
            from tools import get_tools

            reserved = [f.__name__ for f in get_tools()]
        taken = set(reserved)
        prefix = server_prefix(self.name)
        result: list[ToolSpec] = []
        for tool_def in self.tools:
            spec = spec_from_lms_def(tool_def, self.source, self.schemas.get(tool_def.name))
            name = tool_def.name
            if not NAME_RE.fullmatch(name) or name in taken:
                name = clean_tool_name(f"{prefix}_{name}")
            taken.add(name)
            if name != spec.name:
                spec = replace(spec, name=name)
            result.append(spec)
        return result


def _run_demo_server() -> None:
    """Kleiner stdio-MCP-Server für die Demo (Konflikt ``add``, ungültiger Name ``text.upper``)."""
    from typing import Annotated

    from mcp.server.mcpserver import MCPServer
    from pydantic import Field

    server = MCPServer("demo")

    @server.tool()
    def add(
        a: Annotated[int, Field(description="Erste Zahl")],
        b: Annotated[int, Field(description="Zweite Zahl")],
    ) -> int:
        """Addiert zwei ganze Zahlen (kollidiert mit dem lokalen add)."""
        return a + b

    @server.tool(name="text.upper")
    def upper(text: Annotated[str, Field(description="Text, der groß geschrieben wird")]) -> str:
        """Schreibt einen Text in Großbuchstaben."""
        return text.upper()

    server.run()


if __name__ == "__main__":
    import json
    import sys

    from toolset import ToolSet

    if "--demo-server" in sys.argv:
        _run_demo_server()
        sys.exit()

    # Aufruf: python mcp_tools.py [<command> <args...>]  (ohne Angabe: eingebauter Demo-Server)
    cmd = sys.argv[1:] or [sys.executable, __file__, "--demo-server"]
    source = McpToolSource(command=cmd[0], args=cmd[1:], name="demo").start(timeout=30)
    try:
        tools = ToolSet.from_any(source.specs())
        for spec in tools:
            print(f"\n{spec.name} ({spec.source}): {spec.description}")
            print(json.dumps(spec.parameters, ensure_ascii=False, indent=2))
        if "demo_add" in tools:
            print("\n", tools.execute("demo_add", {"a": 2, "b": 3}, print))
            print("\n", tools.execute("demo_text_upper", {"text": "hallo äöü"}, print))
    finally:
        source.close()
