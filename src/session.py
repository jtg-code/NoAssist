"""Provider-unabhängige Chat-Session mit Verlauf, Werkzeugen und MCP-Servern.

Grundnutzung (wie bisher)::

    s = Session("Antworte knapp.", tools=[...])
    print(s.send("Hallo"))

Gestreamt::

    for ev in s.stream("Wie spät ist es?"):
        ...  # turn_start, delta, tool_call, tool_result, message, error, done

Breaking Changes gegenüber der LM-Studio-gebundenen Fassung:

- ``self.chat`` (``lms.Chat``) entfällt. Der Verlauf liegt provider-neutral in
  ``self.messages`` (``list[Message]``).
- ``self.model`` ist jetzt ein Modellname (``str | None``) statt eines
  ``lms.LLM``-Handles; ``None`` = Standardmodell des Providers.
- ``self.tools`` ist ein ``ToolSet`` statt einer Liste.
- ``add_mcp_server`` liefert eine ``McpToolSource`` (Unterklasse von ``McpServer``).
- ``send`` wirft bei Provider-Fehlern ``ProviderError`` statt der rohen
  SDK-Exception.
"""

from __future__ import annotations

import logging
import queue
import threading
from collections.abc import Callable, Iterable, Iterator
from typing import TYPE_CHECKING, Any

from chat_types import (
    INTERNAL,
    Event,
    Message,
    ProviderError,
    TurnRequest,
    ev_done,
    ev_error,
    ev_turn_start,
)
from toolset import ToolSet, ToolSpec

if TYPE_CHECKING:
    import lmstudio as lms

    from mcp_tools import McpToolSource
    from providers.base import Provider

log = logging.getLogger(__name__)

POLL_S = 0.5
"""So oft (Sekunden) prüft der Generator, ob der Provider-Thread noch lebt."""
JOIN_TIMEOUT_S = 10.0
"""So lange wird bei vorzeitigem Ende des Generators auf den Provider-Thread gewartet."""

_DONE = object()  # Ende-Markierung in der Event-Queue


class Session:
    """Chat-Session mit Kontext (Verlauf) und optionalen Werkzeugen.

    Werkzeuge sind normale Python-Funktionen mit Typannotationen und Docstring,
    ``lms.ToolFunctionDef``s oder ``ToolSpec``s. Der Provider (standardmäßig
    LM Studio) führt die Turns aus; die Session hält Verlauf, Zustand und Tools.
    """

    def __init__(
        self,
        system_prompt: str = "",
        model: str | None = None,
        tools: Iterable[Callable[..., Any] | lms.ToolFunctionDef | ToolSpec] | ToolSet | None = None,
        provider: Provider | None = None,
        messages: Iterable[Message] | None = None,
        state: dict[str, Any] | None = None,
    ):
        """`tools=None` lädt alle mit @tool registrierten Tools aus tools.py,
        `tools=[]` startet ohne Werkzeuge. Ein übergebenes ``ToolSet`` wird
        kopiert (MCP-Server der Session verändern es nicht).

        `provider=None` erzeugt einen eigenen ``LMStudioProvider``; dieser wird
        von ``close()`` geschlossen. Ein übergebener Provider bleibt offen.

        `state` (provider-spezifischer Chat-Zustand) wird nicht kopiert, damit
        der Aufrufer ihn nach einem Turn speichern kann.
        """
        self.system_prompt = system_prompt
        self.model: str | None = model
        if tools is None:
            from tools import get_tools

            tools = get_tools()
        self.tools: ToolSet = ToolSet.from_any(tools)
        if provider is None:
            from providers.lmstudio_provider import LMStudioProvider

            provider = LMStudioProvider()
            self._owns_provider = True
        else:
            self._owns_provider = False
        self.provider: Provider = provider
        self.messages: list[Message] = list(messages) if messages is not None else []
        self.state: dict[str, Any] = state if state is not None else {}
        self.mcp_servers: list[McpToolSource] = []
        self._mcp_specs: dict[int, list[ToolSpec]] = {}  # id(server) -> hinzugefügte Specs

    # -- Werkzeuge und MCP --

    def add_mcp_server(self, **kwargs: Any) -> McpToolSource:
        """MCP-Server verbinden und dessen Tools hinzufügen (der Server gehört der Session).

        stdio: add_mcp_server(command="npx", args=["-y", "..."])
        HTTP:  add_mcp_server(url="http://localhost:8000/mcp")
        """
        from mcp_tools import McpToolSource

        server = McpToolSource(**kwargs).start()
        added: list[ToolSpec] = []
        try:
            for spec in server.specs(reserved=self.tools.names()):
                added.append(self.tools.add(spec))
        except BaseException:
            for spec in added:
                self.tools.remove(spec)
            server.close()
            raise
        self.mcp_servers.append(server)
        self._mcp_specs[id(server)] = added
        return server

    def add_tool(self, tool: Callable[..., Any] | lms.ToolFunctionDef | ToolSpec) -> None:
        """Werkzeug registrieren (gilt ab der nächsten Nachricht)."""
        self.tools.add(tool)

    def remove_tool(self, tool: Callable[..., Any] | lms.ToolFunctionDef | ToolSpec | str) -> None:
        """Werkzeug per Funktion, Definition, ``ToolSpec`` oder Name entfernen."""
        self.tools.remove(tool)

    # -- Provider --

    def set_provider(self, provider: Provider, model: str | None = None) -> None:
        """Provider wechseln (gilt ab der nächsten Nachricht); Verlauf bleibt erhalten.

        Ein von der Session selbst erzeugter Provider wird dabei geschlossen.
        """
        if provider is self.provider:
            self.model = model
            return
        old, owned = self.provider, self._owns_provider
        self.provider = provider
        self._owns_provider = False
        self.model = model
        if owned:
            _close_quietly(old, "Provider")

    # -- Turns --

    def stream(self, message: str | Message, cancel: threading.Event | None = None) -> Iterator[Event]:
        """Nutzernachricht senden und die Events des Turns liefern.

        Reihenfolge: ``turn_start``, dann Provider-Events (``delta``,
        ``reasoning``, ``status``, ``tool_call``, ``tool_result``,
        ``message``), bei einem Fehler ``error``, zuletzt ``done`` mit
        ``completed``, ``cancelled`` oder ``error``.

        Die User-Nachricht und alle ``message``-Nachrichten landen in
        ``self.messages``. Der Provider läuft in einem Hilfsthread; wird der
        Generator vorzeitig beendet, wird ``cancel`` gesetzt und auf den
        Thread gewartet.
        """
        user = message if isinstance(message, Message) else Message(role="user", content=message)
        cancel = cancel if cancel is not None else threading.Event()
        self.messages.append(user)
        provider = self.provider
        req = TurnRequest(
            history=list(self.messages),
            model=self.model,
            system_prompt=self.system_prompt,
            tools=self.tools,
            cancel=cancel,
            state=self.state,
        )
        events: queue.Queue[Any] = queue.Queue()

        def worker() -> None:
            try:
                provider.run_turn(req, events.put)
            except ProviderError as e:
                events.put(e.to_event())
            except BaseException as e:
                log.exception("Unerwarteter Fehler im Provider %r", provider)
                events.put(ev_error(INTERNAL, f"Unerwarteter Fehler: {type(e).__name__}: {e}"))
            finally:
                events.put(_DONE)

        thread = threading.Thread(target=worker, name=f"session-turn-{provider.id}", daemon=True)
        finished = False
        failed = False
        try:
            yield ev_turn_start(provider_id=provider.id, model=self.model)
            thread.start()
            while True:
                try:
                    ev = events.get(timeout=POLL_S)
                except queue.Empty:
                    if thread.is_alive():
                        continue
                    break  # Sicherheitsnetz: Thread weg ohne Ende-Markierung
                if ev is _DONE:
                    break
                if ev.get("type") == "message":
                    self._collect(ev)
                elif ev.get("type") == "error":
                    failed = True
                yield ev
            finished = True
            thread.join(timeout=JOIN_TIMEOUT_S)
            if failed:
                status = "error"
            elif cancel.is_set():
                status = "cancelled"
            else:
                status = "completed"
            yield ev_done(status)
        finally:
            if not finished:
                # Generator vorzeitig beendet (close() oder Exception beim Aufrufer).
                cancel.set()
                if thread.is_alive():
                    thread.join(timeout=JOIN_TIMEOUT_S)
                    if thread.is_alive():
                        log.warning("Provider-Thread %s endet nicht nach Abbruch", thread.name)
                self._drain(events)

    def _collect(self, ev: Event) -> None:
        try:
            self.messages.append(Message.from_dict(ev["message"]))
        except Exception:
            log.warning("Ungültige Nachricht im Event verworfen", exc_info=True)

    def _drain(self, events: queue.Queue[Any]) -> None:
        """Nach vorzeitigem Ende: noch gesendete Nachrichten in den Verlauf übernehmen."""
        while True:
            try:
                ev = events.get_nowait()
            except queue.Empty:
                return
            if ev is not _DONE and ev.get("type") == "message":
                self._collect(ev)

    def send(self, message: str | Message, cancel: threading.Event | None = None) -> str:
        """Nutzernachricht senden und den Text der letzten Assistentenantwort zurückgeben.

        Verlauf (inkl. Tool-Aufrufe und -Ergebnisse) bleibt in der Session.
        Bei einem Fehler wird ``ProviderError`` geworfen.
        """
        reply = ""
        for ev in self.stream(message, cancel):
            kind = ev["type"]
            if kind == "message":
                msg = ev["message"]
                if msg.get("role") == "assistant" and (msg.get("content") or "").strip():
                    reply = msg["content"]
            elif kind == "error":
                raise ProviderError(ev.get("message", ""), ev.get("code", INTERNAL), ev.get("hint"))
        return reply

    def reset(self, keep_system_prompt: bool = True) -> None:
        """Kontext (Verlauf und Provider-Zustand) verwerfen."""
        self.messages.clear()
        self.state.clear()
        if not keep_system_prompt:
            self.system_prompt = ""

    def close(self) -> None:
        """Eigene MCP-Verbindungen (und einen selbst erzeugten Provider) schließen."""
        for server in self.mcp_servers:
            for spec in self._mcp_specs.pop(id(server), []):
                self.tools.remove(spec)
            _close_quietly(server, "MCP-Server")
        self.mcp_servers.clear()
        if self._owns_provider:
            _close_quietly(self.provider, "Provider")


def _close_quietly(obj: Any, what: str) -> None:
    try:
        obj.close()
    except Exception:
        log.warning("%s %r konnte nicht geschlossen werden", what, obj, exc_info=True)


if __name__ == "__main__":
    import json
    import sys

    session = Session("Antworte knapp auf Deutsch.", model=sys.argv[1] if len(sys.argv) > 1 else None)
    print("Tools:", session.tools.names())
    try:
        for event in session.stream("Hallo! Wie spät ist es gerade?"):
            if event["type"] in ("delta", "reasoning"):
                print(event["text"], end="", flush=True)
            elif event["type"] != "message":
                print("\n  [event]", json.dumps(event, ensure_ascii=False)[:200])
        print("Nachrichten im Verlauf:", len(session.messages))
        if session.messages and session.messages[-1].role == "assistant":
            print("send():", session.send("Und welcher Wochentag ist heute?"))
    except ProviderError as e:
        print(f"ProviderError {e.code}: {e.message}")
    finally:
        session.close()
