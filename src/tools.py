from datetime import datetime, timedelta
from typing import Callable, Optional, List, Self
from zoneinfo import ZoneInfo
import os
import base64
import subprocess
import json
import contextvars
import asyncio
from playwright.async_api import async_playwright, Page, BrowserContext, Browser, Playwright

# Kontext-Isolation mit expliziter Typannotation für den Typ-Checker
_current_page: contextvars.ContextVar[Optional[Page]] = contextvars.ContextVar("_current_page", default=None)
_current_browser: contextvars.ContextVar[Optional[Browser]] = contextvars.ContextVar("_current_browser", default=None)
_current_playwright: contextvars.ContextVar[Optional[Playwright]] = contextvars.ContextVar("_current_playwright", default=None)

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
def math(formula: str) -> float:
    """Berechnet eine mathematische Formel."""
    return eval(formula)

@tool
def read_file(path: str, startline: int = 0, endline: int | None = None) -> str:
    """Liest eine Textdatei und gibt ihren Inhalt zurück."""
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()
        if endline is None:
            endline = len(lines)
        return "".join(lines[startline:endline])
    
@tool
def write_file(path: str, content: str) -> str:
    """Schreibt Text in eine Datei."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return f"Datei '{path}' geschrieben."

import os

@tool
def append_file(path: str, content: str, startline: int = 0) -> str:
    """
    Fügt Text an einer bestimmten Zeile in eine Datei ein.
    
    Args:
        path: Pfad zur Datei.
        content: Der einzufügende Text.
        startline: Die Zeilennummer (0-basiert), ab der der Text eingefügt werden soll.
                   Wenn 0 oder größer als die Zeilenanzahl, wird der Text am Ende angehängt.
    """
    if not os.path.exists(path):
        # Falls die Datei nicht existiert, erstellen wir sie einfach mit dem Inhalt
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Datei '{path}' wurde neu erstellt mit Inhalt."

    try:
        # 1. Datei einlesen
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # 2. Inhalt einfügen
        new_content = content if content.endswith('\n') else content + '\n'
        
        # Wenn startline innerhalb der existierenden Zeilen liegt:
        if 0 <= startline < len(lines):
            lines.insert(startline, new_content)
        else:
            # Wenn startline außerhalb liegt (z.B. sehr groß), hängen wir es einfach an
            lines.append(new_content)

        with open(path, "w", encoding="utf-8") as f:
            f.writelines(lines)

        return f"Inhalt wurde in '{path}' ab Zeile {startline} eingefügt."

    except Exception as e:
        return f"Fehler beim Bearbeiten der Datei: {e}"




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

@tool
def create_workspace(name: str, git_repo: str | None = None) -> str:
    """Erstellt einen neuen Arbeitsbereich (Ordner) im 'data' Verzeichnis,
    welches parallel zum 'src' Verzeichnis liegt.
    """
    current_dir = os.getcwd()
    print(f"Aktuelles Verzeichnis: {current_dir}")
    print(f"Data-Verzeichnis: {os.path.join(current_dir, 'data')}")
    # 3. Definiere den Pfad zum 'data' Ordner auf dieser Ebene
    data_dir = os.path.join(current_dir, "data")
    # 4. Definiere den endgültigen Pfad für den neuen Arbeitsbereich innerhalb von 'data'
    target_path = os.path.join(data_dir, name)
    print(f"Zielpfad: {target_path}")
    if os.path.exists(target_path):
        return f"Arbeitsbereich '{name}' existiert bereits: {target_path}"
    if git_repo:
        # Optional: Klone das Git-Repository in den neuen Arbeitsbereich
        import subprocess
        try:
            subprocess.run(["git", "clone", git_repo, target_path], check=True)
            return f"Arbeitsbereich '{name}' erstellt und Git-Repository geklont: {target_path}"
        except subprocess.CalledProcessError as e:
            return f"Fehler beim Klonen des Git-Repositories: {e}"
    try:
        # Erstellt auch den 'data' Ordner, falls er noch nicht existiert
        os.makedirs(target_path, exist_ok=True)
        return f"Arbeitsbereich '{name}' erstellt: {target_path}"
    except Exception as e:
        return f"Fehler beim Erstellen des Arbeitsbereichs: {e}"
    
@tool
def list_workspaces() -> list[str]:
    """Listet alle Arbeitsbereiche (Ordner) im 'data' Verzeichnis auf."""
    current_dir = os.getcwd()
    print(f"Aktuelles Verzeichnis: {current_dir}")
    data_dir = os.path.join(current_dir, "data")

    if not os.path.exists(data_dir):
        return []

    return [name for name in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, name))]

@tool
def delete_workspace(name: str) -> str:
    """Löscht einen Arbeitsbereich (Ordner) im 'data' Verzeichnis."""
    import shutil
    print(f"Aktuelles Verzeichnis: {os.getcwd()}")

    current_dir = os.getcwd()
    print(f"Elternverzeichnis: {os.path.dirname(current_dir)}")
    data_dir = os.path.join(current_dir, "data")
    target_path = os.path.join(data_dir, name)

    if not os.path.exists(target_path):
        return f"Arbeitsbereich '{name}' existiert nicht."

    try:
        shutil.rmtree(target_path)
        return f"Arbeitsbereich '{name}' wurde gelöscht."
    except Exception as e:
        return f"Fehler beim Löschen des Arbeitsbereichs: {e}"
    
@tool
def rename_workspace(old_name: str, new_name: str) -> str:
    """Benennt einen Arbeitsbereich (Ordner) im 'data' Verzeichnis um."""
    current_dir = os.getcwd()
    data_dir = os.path.join(current_dir, "data")
    old_path = os.path.join(data_dir, old_name)
    new_path = os.path.join(data_dir, new_name)

    if not os.path.exists(old_path):
        return f"Arbeitsbereich '{old_name}' existiert nicht."
    if os.path.exists(new_path):
        return f"Arbeitsbereich '{new_name}' existiert bereits."

    try:
        os.rename(old_path, new_path)
        return f"Arbeitsbereich '{old_name}' wurde in '{new_name}' umbenannt."
    except Exception as e:
        return f"Fehler beim Umbenennen des Arbeitsbereichs: {e}"
    
@tool
def end_workspace(name: str) -> str:
    """
    Beendet einen Arbeitsbereich, indem Änderungen an das Git-Repository 
    gepusht und Änderungen vom Remote-Repository gepullt werden.
    """
    path = os.path.join(os.getcwd(), "data", name)
    
    if not os.path.exists(path):
        return f"Arbeitsbereich '{name}' existiert nicht."

    try:
        # 1. Git-Befehle ausführen (im Verzeichnis des Arbeitsbereichs)
        subprocess.run(["git", "add", "."], cwd=path, check=True, capture_output=True)
        
        # Wir versuchen einen Commit.
        try:
            subprocess.run(["git", "commit", "-m", "Auto-commit beim Beenden des Arbeitsbereichs"], 
                           cwd=path, check=True, capture_output=True)
        except subprocess.CalledProcessError:
            pass

        # Schritt B: Pull (um Konflikte vor dem Push zu minimieren)
        subprocess.run(["git", "pull", "--rebase"], cwd=path, check=True, capture_output=True)

        # Schritt C: Push
        subprocess.run(["git", "push"], cwd=path, check=True, capture_output=True)

        return f"Arbeitsbereich '{name}' wurde erfolgreich synchronisiert und beendet."

    except subprocess.CalledProcessError as e:
        # Falls Git einen Fehler zurückgibt (z.B. Merge-Konflikte oder fehlende Berechtigungen)
        error_msg = e.stderr.decode().strip()
        return f"Fehler beim Synchronisieren von '{name}': {error_msg}"
    except Exception as e:
        return f"Ein unerwarteter Fehler ist aufgetreten: {str(e)}"


@tool
def run_command(command: str) -> str:
    """Führt einen Shell-Befehl aus und gibt die Ausgabe zurück."""
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        return result.stdout.strip() or "Befehl erfolgreich ausgeführt, aber keine Ausgabe."
    except subprocess.CalledProcessError as e:
        return f"Fehler beim Ausführen des Befehls: {e.stderr.strip()}"
    
_current_page: contextvars.ContextVar[Optional[Page]] = contextvars.ContextVar("_current_page", default=None)
_current_browser: contextvars.ContextVar[Optional[Browser]] = contextvars.ContextVar("_current_browser", default=None)
_current_playwright = contextvars.ContextVar("_current_playwright", default=None)

async def _get_or_create_page() -> Page:
    """Initialisiert den Browser vollautomatisch im Hintergrund, falls noch nicht geschehen."""
    page = _current_page.get()
    if page is None or page.is_closed():
        playwright = await async_playwright().start()
        browser = await playwright.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
        )
        page = await context.new_page()
        
        _current_playwright.set(playwright)
        _current_browser.set(browser)
        _current_page.set(page)
    return page


@tool
async def navigate(url: str) -> str:
    """Navigiert zu einer URL."""
    try:
        page = await _get_or_create_page()
        await page.goto(url, wait_until="domcontentloaded", timeout=30000)
        return f"Erfolgreich zu {url} navigiert."
    except Exception as e:
        return f"Fehler beim Navigieren: {str(e)}"


@tool
async def click_element(selector: str) -> str:
    """Klickt auf ein Element (CSS Selector oder Text)."""
    try:
        page = await _get_or_create_page()
        await page.click(selector, timeout=5000)
        return f"Element '{selector}' wurde geklickt."
    except Exception as e:
        return f"Fehler beim Klicken: {str(e)}"


@tool
async def type_text(selector: str, text: str) -> str:
    """Schreibt Text in ein Eingabefeld."""
    try:
        page = await _get_or_create_page()
        await page.fill(selector, text)
        return f"Text '{text}' wurde in '{selector}' geschrieben."
    except Exception as e:
        return f"Fehler beim Tippen: {str(e)}"


@tool
async def get_page_content() -> str:
    """Gibt den sichtbaren Text der aktuellen Seite zurück (für die KI zum 'Sehen')."""
    try:
        page = await _get_or_create_page()
        content = await page.evaluate("() => document.body.innerText")
        return content[:5000] 
    except Exception as e:
        return f"Fehler beim Lesen des Inhalts: {str(e)}"


@tool
async def take_screenshot(filename: str = "screenshot.png") -> str:
    """Macht einen Screenshot und gibt die Bilddaten direkt als Base64-kodiertes JSON an die KI zurück."""
    try:
        page = await _get_or_create_page()
        screenshot_bytes = await page.screenshot(path=filename)
        
        base64_data = base64.b64encode(screenshot_bytes).decode("utf-8")
        
        result = {
            "status": "success",
            "message": f"Screenshot wurde erfolgreich erstellt (gespeichert unter {filename}).",
            "mime_type": "image/png",
            "image_base64": base64_data
        }
        return json.dumps(result)
        
    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"Fehler beim Screenshot: {str(e)}"
        })




if __name__ == "__main__":
    print("Verfügbare Tools:")
    for name, func in _registry.items():
        print(f"- {name}: {(func.__doc__ or '').splitlines()[0]}")
    print("\nBeispielaufrufe:")
    # create workspace
    print(create_workspace("test_workspace"))
    # list workspaces
    print(list_workspaces())
    # delete workspace
    #print(delete_workspace("test_workspace"))
