from typing import Callable, Iterable

import lmstudio as lms

from mcp_client import McpServer
from tools import get_tools


class Session:
    """Chat-Session mit Kontext (Verlauf) und optionalen Werkzeugen.

    Werkzeuge sind normale Python-Funktionen mit Typannotationen und Docstring;
    LM Studio leitet daraus das Tool-Schema ab.
    """

    def __init__(
        self,
        system_prompt: str = "",
        model: str | None = None,
        tools: Iterable[Callable] | None = None,
    ):
        """`tools=None` lädt alle mit @tool registrierten Tools aus tools.py,
        `tools=[]` startet ohne Werkzeuge."""
        self.system_prompt = system_prompt
        self.model = lms.llm(model) if model else lms.llm()
        self.chat = lms.Chat(system_prompt) if system_prompt else lms.Chat()
        self.tools: list[Callable | lms.ToolFunctionDef] = list(get_tools() if tools is None else tools)
        self.mcp_servers: list[McpServer] = []

    def add_mcp_server(self, **kwargs) -> McpServer:
        """MCP-Server verbinden und dessen Tools hinzufügen.

        stdio: add_mcp_server(command="npx", args=["-y", "..."])
        HTTP:  add_mcp_server(url="http://localhost:8000/mcp")
        """
        server = McpServer(**kwargs).start()
        self.mcp_servers.append(server)
        self.tools.extend(server.tools)
        return server

    def close(self) -> None:
        """Alle MCP-Verbindungen schließen."""
        for server in self.mcp_servers:
            self.tools = [t for t in self.tools if t not in server.tools]
            server.close()
        self.mcp_servers.clear()

    def add_tool(self, tool: Callable) -> None:
        """Werkzeug registrieren (gilt ab der nächsten Nachricht)."""
        if tool not in self.tools:
            self.tools.append(tool)

    def remove_tool(self, tool: Callable) -> None:
        if tool in self.tools:
            self.tools.remove(tool)

    def send(self, message: str) -> str:
        """Nutzernachricht senden und die Antwort zurückgeben.

        Verlauf (inkl. Tool-Aufrufe und -Ergebnisse) bleibt in der Session.
        """
        self.chat.add_user_message(message)

        if not self.tools:
            result = self.model.respond(self.chat)
            self.chat.add_assistant_response(result)
            return result.content

        # act() führt die Tool-Schleife selbst aus; on_message hängt jede
        # Nachricht (Assistent, Tool-Aufruf, Tool-Ergebnis) an den Verlauf.
        reply: list[str] = []

        def on_message(msg) -> None:
            self.chat.append(msg)
            if getattr(msg, "role", None) == "assistant":
                text = "".join(
                    getattr(part, "text", "") for part in getattr(msg, "content", [])
                )
                if text:
                    reply.append(text)

        self.model.act(self.chat, self.tools, on_message=on_message)
        return reply[-1] if reply else ""

    def reset(self, keep_system_prompt: bool = True) -> None:
        """Kontext verwerfen."""
        prompt = self.system_prompt if keep_system_prompt else ""
        self.chat = lms.Chat(prompt) if prompt else lms.Chat()
