from datetime import datetime, timedelta
from typing import Callable
from zoneinfo import ZoneInfo

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


@tool
def get_datetime(
    format: str = "datetime",
    timezone: str = "local",
    offset_days: int = 0,
) -> str:
    """Gibt Datum und/oder Uhrzeit zurück.

    format: "time" (HH:MM:SS), "date" (YYYY-MM-DD), "datetime", "weekday"
        (Wochentag), "iso", "timestamp" (Unix-Sekunden) oder ein eigenes
        strftime-Muster wie "%d.%m.%Y".
    timezone: "local" oder ein IANA-Name wie "UTC" oder "Europe/Berlin".
    offset_days: Verschiebung in Tagen, z.B. 1 für morgen, -1 für gestern.
    """
    tz = None if timezone == "local" else ZoneInfo(timezone)
    now = datetime.now(tz).astimezone() + timedelta(days=offset_days)
    if tz:
        now = now.astimezone(tz)

    formats = {
        "time": "%H:%M:%S",
        "date": "%Y-%m-%d",
        "datetime": "%Y-%m-%d %H:%M:%S",
        "weekday": "%A",
    }
    if format == "iso":
        return now.isoformat()
    if format == "timestamp":
        return str(int(now.timestamp()))
    return now.strftime(formats.get(format, format))


@tool
def search_web(
    query: str,
    max_results: int = 5,
    region: str = "de-de",
    backend: str = "auto",
) -> list[dict]:
    """Sucht im Web (ohne API-Key) und gibt eine Liste von Treffern zurück.

    Jeder Treffer hat "title", "url" und "snippet" (Textauszug). Für den vollen
    Inhalt einer Seite danach read_webpage mit der "url" aufrufen.

    max_results: Anzahl der Ergebnisse.
    region: Suchregion, z.B. "de-de", "us-en" oder "wt-wt" (weltweit).
    backend: Suchanbieter: "brave", "duckduckgo", "google", "bing", "yahoo",
        "startpage", "mojeek", "yandex" oder "wikipedia". Mehrere kommagetrennt
        (z.B. "google,bing"), "auto" lässt die Auswahl automatisch treffen.
    """
    from ddgs import DDGS

    try:
        results = DDGS().text(
            query, region=region, max_results=max_results, backend=backend
        )
    except Exception as e:
        return [{"error": f"Suche fehlgeschlagen: {e}"}]
    return [
        {"title": r["title"], "url": r["href"], "snippet": r["body"]}
        for r in results
    ]


@tool
def read_webpage(url: str, max_chars: int = 8000, start: int = 0) -> str:
    """Lädt eine Webseite und gibt ihren Haupttext (ohne Menüs/Werbung) zurück.

    url: Adresse der Seite (http/https), z.B. eine "url" aus search_web.
    max_chars: maximale Länge der Antwort.
    start: Zeichenposition, ab der gelesen wird. Ist die Seite länger, steht am
        Ende der nächste Wert für "start", um weiterzulesen.
    """
    import trafilatura

    if not url.startswith(("http://", "https://")):
        return "Fehler: Nur http(s)-URLs werden unterstützt."
    html = trafilatura.fetch_url(url)
    if not html:
        return f"Fehler: Seite konnte nicht geladen werden: {url}"
    text = trafilatura.extract(html, include_comments=False, include_tables=True)
    if not text:
        return "Fehler: Kein lesbarer Text auf der Seite gefunden."

    chunk = text[start : start + max_chars]
    end = start + len(chunk)
    if end < len(text):
        chunk += f"\n\n[... gekürzt, {len(text) - end} Zeichen übrig. Weiterlesen mit start={end}]"
    return chunk


if __name__ == "__main__":
    print("Verfügbare Tools:")
    for name, func in _registry.items():
        print(f"- {name}: {(func.__doc__ or '').splitlines()[0]}")
    result = search_web("Python programming language", max_results=3)
    print("\nBeispiel-Suche:")
    print(result)