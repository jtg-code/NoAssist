from typing import Callable

_registry: dict[str, Callable] = {}


def tool(func: Callable) -> Callable:
    """Decorator: registriert eine Funktion als Werkzeug für die KI.

    Typannotationen und Docstring der Funktion beschreiben das Tool für das Modell.
    """
    _registry[func.__name__] = func
    return func


def get_tools(*names: str) -> list[Callable]:
    """Alle registrierten Tools, oder nur die mit den angegebenen Namen."""
    if not names:
        return list(_registry.values())
    return [_registry[n] for n in names]


# --- Beispiele ---------------------------------------------------------------

@tool
def add(a: float, b: float) -> float:
    """Addiert zwei Zahlen."""
    return a + b


@tool
def read_file(path: str) -> str:
    """Liest eine Textdatei und gibt ihren Inhalt zurück."""
    with open(path, encoding="utf-8") as f:
        return f.read()
