import asyncio
import threading
from concurrent.futures import Future
from contextlib import AsyncExitStack
from typing import Any

import lmstudio as lms
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client

_JSON_TYPES: dict[str, type] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
}


def _attr(obj, *names):
    """mcp 2.x nutzt snake_case, 1.x camelCase."""
    for n in names:
        if hasattr(obj, n):
            return getattr(obj, n)


class McpServer:
    """Verbindung zu einem externen MCP-Server (stdio oder HTTP).

    Die MCP-Tools werden als LM-Studio-Tools bereitgestellt (`tools`).
    Tool-Aufrufe von LM Studio sind synchron, MCP ist async, deshalb läuft
    die Verbindung in einem eigenen Event-Loop-Thread.
    """

    def __init__(
        self,
        command: str | None = None,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        url: str | None = None,
        name: str | None = None,
    ):
        if bool(command) == bool(url):
            raise ValueError("Entweder `command` (stdio) oder `url` (HTTP) angeben.")
        self.command, self.args, self.env, self.url = command, args or [], env, url
        self.name = name or command or url
        self.tools: list[lms.ToolFunctionDef] = []

        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()
        self._stack = AsyncExitStack()
        self._session: ClientSession | None = None
        self._ready = threading.Event()
        self._error = False

    def _run(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()

    def start(self) -> "McpServer":
        self._task: Future[None] = asyncio.run_coroutine_threadsafe(
            self._lifecycle(), self._loop
        )
        # Wartet, bis verbunden (oder der Verbindungsaufbau fehlgeschlagen ist).
        self._ready.wait()
        if self._error:
            self._task.result()  # wirft die ursprüngliche Exception
        return self

    async def _lifecycle(self) -> None:
        # Verbindung öffnen und schließen müssen im selben Task passieren (anyio).
        self._stop = asyncio.Event()
        try:
            async with self._stack:
                await self._connect()
                self._ready.set()
                await self._stop.wait()
        except BaseException:
            self._error = True
            raise
        finally:
            self._ready.set()

    async def _connect(self) -> None:
        if self.url:
            read, write, *_ = await self._stack.enter_async_context(
                streamable_http_client(self.url)
            )
        else:
            assert self.command
            params = StdioServerParameters(
                command=self.command, args=self.args, env=self.env
            )
            read, write = await self._stack.enter_async_context(stdio_client(params))
        self._session = await self._stack.enter_async_context(
            ClientSession(read, write)
        )
        await self._session.initialize()
        listed = await self._session.list_tools()
        self.tools = [self._wrap(t) for t in listed.tools]

    def _wrap(self, mcp_tool) -> "lms.ToolFunctionDef":
        schema = _attr(mcp_tool, "input_schema", "inputSchema") or {}
        required = set(schema.get("required", []))
        params: dict[str, Any] = {}
        for pname, pschema in schema.get("properties", {}).items():
            ptype = _JSON_TYPES.get(pschema.get("type"), str)
            params[pname] = ptype if pname in required else {"type": ptype, "default": None}

        def call(**kwargs) -> str:
            # Nicht gesetzte optionale Parameter nicht an den Server schicken.
            kwargs = {k: v for k, v in kwargs.items() if v is not None}
            return self._call(mcp_tool.name, kwargs)

        return lms.ToolFunctionDef(
            name=mcp_tool.name,
            description=mcp_tool.description or "",
            parameters=params,
            implementation=call,
        )

    def _call(self, tool_name: str, arguments: dict[str, Any]) -> str:
        session = self._session
        assert session, "Server nicht gestartet"
        result = self._run(session.call_tool(tool_name, arguments))
        text = "\n".join(
            getattr(part, "text", "")
            for part in result.content
            if getattr(part, "type", "") == "text"
        )
        if _attr(result, "is_error", "isError"):
            return f"Fehler: {text}"
        return text

    def close(self) -> None:
        self._loop.call_soon_threadsafe(self._stop.set)
        self._task.result(timeout=10)
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)
