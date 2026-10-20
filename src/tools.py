from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Optional, List, Self
from zoneinfo import ZoneInfo
import os
import secrets
import threading
import time
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

# --- Agenten und Subagenten -----------------------------------------------------

_service = None
_subagents: dict[str, dict] = {}
_subagents_lock = threading.RLock()

from agents import ASSISTANT_EXCLUDED_TOOLS, SUBAGENT_EXCLUDED_TOOLS  # noqa: E402

_KINDS = {
    "subagent": {"prefix": "sa_", "label": "Subagent", "excluded": SUBAGENT_EXCLUDED_TOOLS},
    "assistant": {"prefix": "aa_", "label": "Assistent Agent", "excluded": ASSISTANT_EXCLUDED_TOOLS},
}
"""Arten von Hilfsagenten. Der Haupt-Chat startet Subagenten, Subagenten starten Assistent Agents;
Assistent Agents starten keine weiteren Agenten."""

_ERR = "Fehler:"


def bind_service(service) -> None:
    """Verbindet die Agenten-Tools mit dem ChatService (wird beim Laden der Tools aufgerufen)."""
    global _service
    _service = service
    _load_subagents()


def _svc():
    if _service is None:
        raise RuntimeError("Der ChatService ist nicht verfügbar.")
    return _service


def _subagents_file() -> Path:
    return Path(_svc().data_dir) / "subagents.json"


def _load_subagents() -> None:
    with _subagents_lock:
        _subagents.clear()
        try:
            data = json.loads(_subagents_file().read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if isinstance(data, dict):
            _subagents.update({k: v for k, v in data.items() if isinstance(v, dict) and v.get("chat_id")})


def _save_subagents() -> None:
    path = _subagents_file()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(_subagents, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, path)
    except OSError:
        pass  # nur in-memory weiter; der Subagent funktioniert bis zum Neustart


def _agents():
    return _svc().agents


@tool
def list_agents() -> list[dict]:
    """Listet alle vorgefertigten Agenten auf.

    Jeder Eintrag hat "id", "name", "description" und "assets" (Dateinamen).
    Mit dem Namen lässt sich ein Agent in create_subagent verwenden oder mit delete_agent löschen.
    """
    try:
        return [
            {"id": a["id"], "name": a["name"], "description": a["description"],
             "assets": [x["name"] for x in a["assets"]]}
            for a in _agents().list()
        ]
    except Exception as e:
        return [{"error": f"{_ERR} {e}"}]


@tool
def create_agent(
    name: str,
    description: str,
    prompt: str,
    assets: dict[str, str] | None = None,
    asset_paths: list[str] | None = None,
) -> str:
    """Erstellt einen neuen vorgefertigten Agenten und bestätigt das Ergebnis.

    name: eindeutiger Name des Agenten.
    description: kurze Beschreibung, wofür der Agent gedacht ist.
    prompt: Anweisung (System-Prompt) des Agenten.
    assets: optional, Dateiname -> Textinhalt; wird als Asset-Datei gespeichert.
    asset_paths: optional, Pfade vorhandener Dateien, die als Assets kopiert werden.
    """
    from agents import MAX_ASSET_BYTES

    try:
        store = _agents()
        name = (name or "").strip()
        if any(a["name"].casefold() == name.casefold() for a in store.list()):
            return f"{_ERR} Ein Agent mit dem Namen '{name}' existiert bereits."
        files: list[tuple[str, bytes]] = []
        for filename, content in (assets or {}).items():
            files.append((filename, str(content).encode("utf-8")))
        for raw_path in asset_paths or []:
            if not os.path.isfile(raw_path):
                return f"{_ERR} Asset-Datei nicht gefunden: {raw_path}"
            if os.path.getsize(raw_path) > MAX_ASSET_BYTES:
                return f"{_ERR} Asset-Datei ist zu groß: {raw_path}"
            with open(raw_path, "rb") as f:
                files.append((os.path.basename(raw_path), f.read()))
        agent = store.create({"name": name, "description": description, "prompt": prompt}, files)
    except Exception as e:
        return f"{_ERR} Agent konnte nicht erstellt werden: {e}"
    return f"Agent '{agent['name']}' wurde erstellt (ID: {agent['id']}, {len(agent['assets'])} Assets)."


@tool
def delete_agent(name: str) -> str:
    """Löscht einen vorgefertigten Agenten samt Assets und bestätigt das Ergebnis.

    name: Name (oder ID) des Agenten, siehe list_agents.
    """
    try:
        store = _agents()
        agent = store.find(name)
        store.delete(agent["id"])
    except Exception as e:
        return f"{_ERR} Agent konnte nicht gelöscht werden: {e}"
    return f"Agent '{agent['name']}' wurde gelöscht."


def _agent_system_prompt(agent: dict) -> str:
    prompt = agent["prompt"]
    if agent["assets"]:
        folder = _agents().assets_dir(agent["id"])
        lines = "\n".join(f"- {a['name']}" for a in agent["assets"])
        prompt += f"\n\nZu dir gehörende Asset-Dateien (Ordner: {folder}); lies sie bei Bedarf mit read_file:\n{lines}"
    return prompt


def _resolve_provider(ref: str) -> dict:
    ref = (ref or "").strip()
    providers = _svc().list_providers()
    found = [p for p in providers if p["id"] == ref] or [
        p for p in providers if str(p.get("name") or "").casefold() == ref.casefold()
    ]
    if not found:
        names = ", ".join(f"{p['id']} ({p.get('name')})" for p in providers) or "keine"
        raise ValueError(f"Provider '{ref}' existiert nicht. Verfügbar: {names}.")
    if found[0].get("enabled") is False:
        raise ValueError(f"Provider '{found[0].get('name') or found[0]['id']}' ist deaktiviert.")
    return found[0]


def _create_child(kind: str, start_prompt: str, provider: str | None, model: str | None, agent: str | None) -> str:
    cfg = _KINDS[kind]
    try:
        service = _svc()
        if not (start_prompt or "").strip():
            return f"{_ERR} Der Startprompt darf nicht leer sein."
        agent_info = _agents().find(agent) if agent else None
        default_provider, default_model = service.role_defaults(kind, with_agent=agent_info is not None)
        if (provider or "").strip():
            prov = _resolve_provider(provider)
            # Das Modell der Einstellungen gilt nur für deren Provider
            model = model or (default_model if prov["id"] == default_provider else None)
        else:
            if not default_provider:
                return f"{_ERR} Es ist kein Provider konfiguriert."
            prov = _resolve_provider(default_provider)
            model = model or default_model
        system_prompt = _agent_system_prompt(agent_info) if agent_info else None
        sid = cfg["prefix"] + secrets.token_hex(4)
        title = f"{cfg['label']} {sid}" + (f" ({agent_info['name']})" if agent_info else "")
        chat = service.create_chat(provider_id=prov["id"], model=model or None, title=title)
        try:
            service.start_turn(
                chat["id"], start_prompt, provider_id=prov["id"], model=model or None,
                system_prompt=system_prompt, exclude_tools=cfg["excluded"],
            )
        except BaseException:
            service.delete_chat(chat["id"])
            raise
        with _subagents_lock:
            _subagents[sid] = {
                "kind": kind, "chat_id": chat["id"], "provider_id": prov["id"], "model": model or None,
                "system_prompt": system_prompt, "agent": agent_info["name"] if agent_info else None,
            }
            _save_subagents()
    except Exception as e:
        return f"{_ERR} {cfg['label']} konnte nicht erstellt werden: {e}"
    return json.dumps({f"{kind}_id" if kind == "assistant" else "subagent_id": sid, "status": "gestartet"})


@tool
def create_subagent(
    start_prompt: str,
    provider: str | None = None,
    model: str | None = None,
    agent: str | None = None,
) -> str:
    """Erstellt einen Subagenten, der den Startprompt sofort im Hintergrund bearbeitet.

    Gibt die Subagenten-ID zurück. Die Antwort auf den Startprompt holt
    send_to_subagent mit leerer Nachricht und wait_for_answer=true ab.

    start_prompt: erste Aufgabe/Nachricht an den Subagenten.
    provider: ID oder Name des Providers (z.B. "lmstudio" oder "claude"); leer = Einstellung
        "Subagenten" (bzw. "Agenten", wenn ein vorgefertigter Agent angegeben ist).
    model: Modell des Providers; leer = Modell aus den Einstellungen bzw. Standardmodell.
    agent: optional, Name eines vorgefertigten Agenten (dessen Prompt wird der System-Prompt).
    Subagenten können selbst keine Subagenten erstellen, aber Assistent Agents (create_assistant_agent).
    """
    return _create_child("subagent", start_prompt, provider, model, agent)


@tool
def create_assistant_agent(
    start_prompt: str,
    provider: str | None = None,
    model: str | None = None,
    agent: str | None = None,
) -> str:
    """Erstellt einen Assistent Agent (Hilfsagent eines Subagenten), der den Startprompt sofort bearbeitet.

    Gibt die ID des Assistent Agents zurück. Die Antwort auf den Startprompt holt
    send_to_assistant_agent mit leerer Nachricht und wait_for_answer=true ab.

    start_prompt: erste Aufgabe/Nachricht an den Assistent Agent.
    provider: ID oder Name des Providers; leer = Einstellung "Assistent Agents"
        (bzw. "Agenten", wenn ein vorgefertigter Agent angegeben ist).
    model: Modell des Providers; leer = Modell aus den Einstellungen bzw. Standardmodell.
    agent: optional, Name eines vorgefertigten Agenten (dessen Prompt wird der System-Prompt).
    Assistent Agents können selbst keine weiteren Agenten erstellen.
    """
    return _create_child("assistant", start_prompt, provider, model, agent)


def _last_answer(chat_id: str) -> str:
    chat = _svc().get_chat(chat_id)
    for msg in reversed((chat or {}).get("messages") or []):
        if msg.get("role") == "assistant" and (msg.get("content") or "").strip():
            return msg["content"]
    return ""


def _wait_for_turn(turn, timeout_s: float, label: str = "Subagent") -> str:
    """Wartet auf das Ende des Turns und gibt die letzte Antwort (oder den Fehler) als Text zurück."""
    tool_name = "send_to_assistant_agent" if label != "Subagent" else "send_to_subagent"
    deadline = time.monotonic() + timeout_s
    seq, answer, error, status = 0, "", None, None
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return (f"{_ERR} Zeitüberschreitung nach {int(timeout_s)} s: {label} arbeitet noch. "
                    f"Mit {tool_name} (leere Nachricht, wait_for_answer=true) erneut warten.")
        events, finished = turn.events_after(seq, min(remaining, 5.0))
        for seq, ev in events:
            kind = ev.get("type")
            if kind == "message":
                msg = ev.get("message") or {}
                if msg.get("role") == "assistant" and (msg.get("content") or "").strip():
                    answer = msg["content"]
            elif kind == "error":
                error = ev.get("message") or "Unbekannter Fehler"
            elif kind == "done":
                status = ev.get("status")
        if finished:
            break
    if error:
        return f"{_ERR} {label} meldet einen Fehler: {error}"
    if status == "cancelled":
        return f"{_ERR} Die Antwort wurde abgebrochen ({label})."
    return answer or f"({label} hat keine Textantwort geliefert.)"


def _send_child(kind: str, child_id: str, message: str, wait_for_answer: bool, timeout_s: int) -> str:
    cfg = _KINDS[kind]
    label = cfg["label"]
    try:
        service = _svc()
        with _subagents_lock:
            info = dict(_subagents.get(child_id) or {})
        if not info or info.get("kind", "subagent") != kind:
            return f"{_ERR} Unbekannter {label}: {child_id}"
        chat_id = info["chat_id"]
        timeout = float(max(1, min(int(timeout_s), 3600)))
        running = service.active_turn(chat_id)
        if not (message or "").strip():
            if not wait_for_answer:
                return f"{_ERR} Nachricht fehlt (leer nur zusammen mit wait_for_answer=true)."
            if running is not None:
                return _wait_for_turn(running, timeout, label)
            return _last_answer(chat_id) or "(Noch keine Antwort vorhanden.)"
        if running is not None:
            return (f"{_ERR} {label} bearbeitet noch eine Aufgabe. Erst mit leerer Nachricht und "
                    "wait_for_answer=true auf die Antwort warten.")
        turn, _ = service.start_turn(
            chat_id, message, provider_id=info["provider_id"], model=info.get("model"),
            system_prompt=info.get("system_prompt"), exclude_tools=cfg["excluded"],
        )
        if not wait_for_answer:
            return f"Nachricht an {label} {child_id} gesendet; er arbeitet im Hintergrund."
        return _wait_for_turn(turn, timeout, label)
    except Exception as e:
        return f"{_ERR} {e}"


@tool
def send_to_subagent(
    subagent_id: str,
    message: str = "",
    wait_for_answer: bool = True,
    timeout_s: int = 300,
) -> str:
    """Sendet eine Nachricht an einen Subagenten.

    wait_for_answer=true: wartet auf die Antwort und gibt sie zurück.
    wait_for_answer=false: gibt nur die Bestätigung zurück, der Subagent arbeitet im Hintergrund.
    Läuft noch eine frühere Aufgabe (z.B. der Startprompt), holt eine leere Nachricht mit
    wait_for_answer=true deren Antwort ab; ohne laufende Aufgabe liefert sie die letzte Antwort.

    subagent_id: ID aus create_subagent.
    message: Nachricht an den Subagenten.
    timeout_s: längste Wartezeit in Sekunden bei wait_for_answer=true.
    """
    return _send_child("subagent", subagent_id, message, wait_for_answer, timeout_s)


@tool
def send_to_assistant_agent(
    assistant_id: str,
    message: str = "",
    wait_for_answer: bool = True,
    timeout_s: int = 300,
) -> str:
    """Sendet eine Nachricht an einen Assistent Agent.

    wait_for_answer=true: wartet auf die Antwort und gibt sie zurück.
    wait_for_answer=false: gibt nur die Bestätigung zurück, der Assistent Agent arbeitet im Hintergrund.
    Läuft noch eine frühere Aufgabe (z.B. der Startprompt), holt eine leere Nachricht mit
    wait_for_answer=true deren Antwort ab; ohne laufende Aufgabe liefert sie die letzte Antwort.

    assistant_id: ID aus create_assistant_agent.
    message: Nachricht an den Assistent Agent.
    timeout_s: längste Wartezeit in Sekunden bei wait_for_answer=true.
    """
    return _send_child("assistant", assistant_id, message, wait_for_answer, timeout_s)
