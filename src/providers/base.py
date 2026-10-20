"""Abstrakte Basisklasse für alle Chat-Provider."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

from chat_types import Emit, ModelInfo, ProviderStatus, TurnRequest


class Provider(ABC):
    """Gemeinsame Schnittstelle aller Provider.

    Unterklassen setzen ``type`` (z. B. ``"lmstudio"``, ``"openai"``,
    ``"claude_agent"``) und implementieren ``list_models``, ``status`` und
    ``run_turn``.

    Vertrag für ``run_turn``:

    - Die Methode blockiert, bis der Turn fertig, abgebrochen oder fehlgeschlagen ist.
    - Gestreamte Inhalte gehen per ``emit`` als ``delta``-, ``reasoning``- und
      ``status``-Events raus.
    - Jede fertige Assistenten- oder Tool-Nachricht geht der Reihe nach als
      ``ev_message`` raus.
    - Tools werden ausschließlich über ``req.tools`` ausgeführt (dort entstehen
      auch die ``tool_call``/``tool_result``-Events).
    - ``req.cancel`` wird regelmäßig geprüft; bei gesetztem Event wird der Turn
      zügig beendet.
    - Erwartbare Fehler werden als ``ProviderError`` geworfen, ohne Geheimnisse
      (API-Keys, Tokens) im Text.
    - Fremde Verläufe (z. B. von einem anderen Provider) müssen verkraftet
      werden. Tool-Aufrufe ohne Ergebnis bekommen ein synthetisches Ergebnis
      ``"Fehler: abgebrochen"``.
    - ``req.state`` darf verändert werden; er wird nach dem Turn gespeichert.
    - ``turn_start`` und ``done`` sendet nicht der Provider, sondern der Aufrufer.
    """

    type: ClassVar[str]

    def __init__(self, id: str, name: str):
        self.id = id
        self.name = name

    @abstractmethod
    def list_models(self) -> list[ModelInfo]:
        """Liefert die verfügbaren Modelle."""

    @abstractmethod
    def status(self) -> ProviderStatus:
        """Prüft Erreichbarkeit bzw. Anmeldung, ohne einen Turn auszuführen."""

    @abstractmethod
    def run_turn(self, req: TurnRequest, emit: Emit) -> None:
        """Führt einen Turn aus (siehe Vertrag in der Klassenbeschreibung)."""

    def close(self) -> None:
        """Gibt Ressourcen frei. Standard: nichts zu tun."""

    def __repr__(self) -> str:
        return f"<{type(self).__name__} id={self.id!r} name={self.name!r}>"


if __name__ == "__main__":
    import threading

    from chat_types import Message, ev_delta, ev_message

    class EchoProvider(Provider):
        type = "echo"

        def list_models(self) -> list[ModelInfo]:
            return [ModelInfo(id="echo", label="Echo")]

        def status(self) -> ProviderStatus:
            return ProviderStatus(ok=True, message="bereit")

        def run_turn(self, req: TurnRequest, emit: Emit) -> None:
            text = req.history[-1].content
            emit(ev_delta(text))
            emit(ev_message(Message(role="assistant", content=text, provider_id=self.id, model=req.model)))

    p = EchoProvider("echo", "Echo")
    print(p, p.status(), p.list_models())
    req = TurnRequest(
        history=[Message(role="user", content="Hallo")],
        model="echo",
        system_prompt="",
        tools=None,  # type: ignore[arg-type]  # ToolSet entsteht in toolset.py
        cancel=threading.Event(),
        state={},
    )
    p.run_turn(req, print)
    p.close()
