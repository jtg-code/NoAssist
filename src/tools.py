from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Optional, List, Self
from zoneinfo import ZoneInfo
import os
import re
import secrets
import threading
import time
import base64
import subprocess
import json
import contextvars
import asyncio

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


COMMAND_TIMEOUT_S = 600
"""Längste Laufzeit eines Befehls in ``run_command``."""


def _stop_process_tree(proc: subprocess.Popen) -> None:
    """Beendet den Befehl samt Kindprozessen."""
    if proc.poll() is not None:
        return
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True, timeout=10)
        else:
            os.killpg(proc.pid, 15)
    except Exception:
        proc.kill()


@tool
def run_command(command: str) -> str:
    """Führt einen Shell-Befehl aus und gibt die Ausgabe zurück.

    Die Ausgabe wird während der Ausführung live im Chat angezeigt. Ein Befehl läuft höchstens
    10 Minuten und wird beim Abbruch der Antwort beendet.
    """
    import contextvars
    import locale

    from toolset import stream_output, tool_cancelled

    # Unter Windows schreiben Programme in der Codepage der Konsole (UTF-8 oder z. B. cp850)
    encoding = "utf-8"
    if os.name == "nt":
        try:
            import ctypes

            page = ctypes.windll.kernel32.GetConsoleOutputCP() or ctypes.windll.kernel32.GetOEMCP()
            encoding = "utf-8" if page == 65001 else f"cp{page}"
        except Exception:
            encoding = locale.getpreferredencoding(False)
    env = {**os.environ, "PYTHONIOENCODING": encoding, "PYTHONUNBUFFERED": "1"}

    try:
        proc = subprocess.Popen(
            command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL, bufsize=0, env=env,
            start_new_session=(os.name != "nt"),
            creationflags=(subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0),
        )
    except Exception as e:
        return f"Fehler: Befehl konnte nicht gestartet werden: {e}"

    chunks: dict[str, list[str]] = {"stdout": [], "stderr": []}

    def pump(name: str, pipe) -> None:
        import codecs

        decoder = codecs.getincrementaldecoder(encoding)(errors="replace")
        while True:
            data = pipe.read1(4096) if hasattr(pipe, "read1") else pipe.read(4096)
            if not data:
                break
            text = decoder.decode(data)
            if text:
                chunks[name].append(text)
                stream_output(text, name)
        tail = decoder.decode(b"", final=True)
        if tail:
            chunks[name].append(tail)
            stream_output(tail, name)

    # Jeder Leser läuft im Kontext des Tool-Aufrufs, damit stream_output() das Event senden kann
    readers = [
        threading.Thread(target=contextvars.copy_context().run, args=(pump, n, p), daemon=True)
        for n, p in (("stdout", proc.stdout), ("stderr", proc.stderr))
    ]
    for r in readers:
        r.start()

    deadline = time.monotonic() + COMMAND_TIMEOUT_S
    reason = None
    while proc.poll() is None:
        if tool_cancelled():
            reason = "abgebrochen"
        elif time.monotonic() > deadline:
            reason = f"Zeitüberschreitung nach {COMMAND_TIMEOUT_S} s"
        if reason:
            _stop_process_tree(proc)
            break
        time.sleep(0.05)
    proc.wait()
    for r in readers:
        r.join(timeout=2)

    out, err = "".join(chunks["stdout"]).strip(), "".join(chunks["stderr"]).strip()
    if reason:
        partial = f"\n{out}" if out else ""
        return f"Fehler: Befehl {reason}.{partial}"
    if proc.returncode != 0:
        return f"Fehler: Befehl endete mit Exit-Code {proc.returncode}.\n{err or out}".rstrip()
    return out or "Befehl erfolgreich ausgeführt, aber keine Ausgabe."
    
def _browser_key() -> str | None:
    """Der Chat, dessen Browser die Browser-Tools nutzen (je Chat ein eigener Browser)."""
    return _caller_chat.get()


def _browser(op, timeout: float | None = None, session: bool = False):
    from browser import CALL_TIMEOUT_S, manager

    return manager.call(_browser_key(), op, timeout or CALL_TIMEOUT_S, session)


def _browser_notes() -> str:
    """Meldungen des Browsers (neuer Tab, Dialog, Download) als Anhang für das Tool-Ergebnis."""
    from browser import manager

    notes = manager.pop_notes(_browser_key())
    return "".join(f"\n[Hinweis: {n}]" for n in notes)


def _browser_action(label: str, op, timeout: float | None = None, session: bool = False) -> str:
    """Führt eine Aktion aus und meldet Erfolg oder Fehler samt Hinweisen des Browsers."""
    try:
        detail = _browser(op, timeout, session)
    except Exception as e:
        return f"Fehler: Nicht ausgeführt ({label}). Grund: {_short_error(e)}{_browser_notes()}"
    return f"{label}{f': {detail}' if detail else ''}.{_browser_notes()}"


def _short_error(exc: BaseException) -> str:
    """Erste Zeilen der Playwright-Fehlermeldung (der Rest ist ein langes Aufrufprotokoll)."""
    lines = [ln.strip() for ln in str(exc).splitlines() if ln.strip()]
    if not lines:
        return type(exc).__name__
    head = lines[0]
    # Aus dem Aufrufprotokoll nur die Zeilen, die erklären, warum es scheiterte
    hints = [
        ln for ln in lines[1:]
        if re.search(r"not visible|not enabled|not stable|intercepts|outside of the viewport|detached|waiting for", ln, re.I)
    ]
    return " | ".join([head, *dict.fromkeys(hints[-3:])])[:500]


_SNAPSHOT_JS = r"""
() => {
  document.querySelectorAll('[data-na-ref]').forEach(e => e.removeAttribute('data-na-ref'));
  const sel = 'a[href],button,input,select,textarea,summary,[role=button],[role=link],[role=checkbox],[role=radio],'
    + '[role=tab],[role=menuitem],[role=switch],[role=option],[role=combobox],[role=textbox],[role=searchbox],'
    + '[contenteditable=""],[contenteditable=true],[onclick],[tabindex]:not([tabindex="-1"])';
  const seen = new Set(), items = [];
  const vh = innerHeight, vw = innerWidth;
  const visible = (el) => {
    const r = el.getBoundingClientRect(), s = getComputedStyle(el);
    return r.width > 0 && r.height > 0 && s.visibility !== 'hidden' && s.display !== 'none' && s.opacity !== '0';
  };
  const label = (el) => {
    const aria = el.getAttribute('aria-label');
    if (aria) return aria;
    if (el.id) { const l = document.querySelector(`label[for="${CSS.escape(el.id)}"]`); if (l) return l.innerText; }
    const wrap = el.closest('label'); if (wrap && wrap !== el) return wrap.innerText;
    return (el.tagName === 'SELECT' ? '' : el.innerText) || el.value || el.getAttribute('placeholder') || el.getAttribute('title') || el.getAttribute('alt') || el.getAttribute('name') || '';
  };
  let n = 0;
  for (const el of document.querySelectorAll(sel)) {
    if (seen.has(el) || !visible(el) || el.disabled && el.tagName !== 'OPTION') continue;
    seen.add(el);
    const r = el.getBoundingClientRect();
    const tag = el.tagName.toLowerCase();
    const role = el.getAttribute('role') || '';
    let kind = tag === 'a' ? 'Link' : tag === 'button' || role === 'button' ? 'Button' : tag === 'select' ? 'Auswahl'
      : tag === 'textarea' ? 'Textfeld' : tag === 'summary' ? 'Aufklappbar' : tag === 'input' ? (el.type || 'text') : role || tag;
    if (tag === 'input') kind = ({checkbox:'Checkbox', radio:'Radio', submit:'Button', button:'Button', file:'Datei', password:'Passwort', search:'Suche', range:'Regler'})[el.type] || 'Eingabe:' + (el.type || 'text');
    const ref = ++n;
    el.setAttribute('data-na-ref', ref);
    let text = label(el).replace(/\s+/g, ' ').trim().slice(0, 80);
    const extra = [];
    if (tag === 'a' && el.getAttribute('href')) extra.push('→ ' + el.getAttribute('href').slice(0, 70));
    if (el.type === 'checkbox' || el.type === 'radio') extra.push(el.checked ? 'angehakt' : 'nicht angehakt');
    if (tag === 'select') extra.push('Wert: ' + (el.selectedOptions[0] ? el.selectedOptions[0].text : ''));
    else if (['input', 'textarea'].includes(tag) && el.type !== 'password' && el.type !== 'checkbox' && el.type !== 'radio' && el.value) extra.push('Inhalt: ' + el.value.slice(0, 60));
    if (el.getAttribute('aria-expanded')) extra.push(el.getAttribute('aria-expanded') === 'true' ? 'offen' : 'zu');
    if (el.readOnly) extra.push('schreibgeschützt');
    const inView = r.bottom > 0 && r.top < vh && r.right > 0 && r.left < vw;
    items.push(`[${ref}] ${kind}${text ? ' „' + text + '“' : ''}${extra.length ? ' (' + extra.join(', ') + ')' : ''}${inView ? '' : ' [außerhalb des Sichtbereichs]'}`);
    if (n >= 150) break;
  }
  const doc = document.scrollingElement || document.documentElement;
  const text = (document.body ? document.body.innerText : '').replace(/\n{3,}/g, '\n\n').trim();
  return {
    title: document.title, url: location.href,
    scrollY: Math.round(doc.scrollTop), scrollMax: Math.max(0, Math.round(doc.scrollHeight - vh)),
    items, more: n >= 150, text,
  };
}
"""


async def _target(page, ref: int | None = None, selector: str | None = None, text: str | None = None):
    """Locator für ein Element: per Nummer aus ``get_page_content``, per Selektor oder per sichtbarem Text."""
    if ref is not None:
        loc = page.locator(f'[data-na-ref="{int(ref)}"]')
        if await loc.count() == 0:
            raise ValueError(
                f"Element [{ref}] gibt es nicht (mehr). Mit get_page_content die aktuelle Seitenübersicht holen."
            )
        return loc.first
    if selector:
        return page.locator(selector).first
    if text:
        exact = page.get_by_text(text, exact=True)
        return (exact if await exact.count() else page.get_by_text(text, exact=False)).first
    raise ValueError("Angabe fehlt: ref (Nummer aus get_page_content), selector oder text.")


def _describe(ref: int | None, selector: str | None, text: str | None) -> str:
    return f"[{ref}]" if ref is not None else f"'{selector}'" if selector else f"'{text}'"


@tool
def navigate(url: str) -> str:
    """Öffnet eine Adresse im aktuellen Tab. Danach get_page_content aufrufen, um die Seite zu sehen."""
    async def op(page):
        await page.goto(url, wait_until="domcontentloaded", timeout=30000)
        return page.url

    return _browser_action(f"Zu {url} navigiert", lambda p: op(p), timeout=40)


@tool
def get_page_content(max_chars: int = 6000) -> str:
    """Zeigt die aktuelle Seite so, wie ein Nutzer sie sieht: Titel, Adresse, Scrollstand, den sichtbaren Text
    und eine nummerierte Liste aller bedienbaren Elemente (Links, Buttons, Felder, Auswahlen).

    Mit der Nummer in eckigen Klammern, z.B. [7], sprechen click_element, type_text, select_option,
    hover_element und die übrigen Tools das Element an. Nach jeder Änderung der Seite neu aufrufen,
    denn die Nummern gelten nur für die zuletzt gelieferte Übersicht.

    max_chars: maximale Länge des Seitentextes.
    """
    async def op(page):
        return await page.evaluate(_SNAPSHOT_JS)

    try:
        snap = _browser(op)
    except Exception as e:
        return f"Fehler beim Lesen der Seite: {_short_error(e)}{_browser_notes()}"
    text = snap["text"]
    if len(text) > max_chars:
        text = text[:max_chars] + f"\n[... gekürzt, {len(snap['text']) - max_chars} Zeichen übrig. Mit scroll weiterlesen]"
    lines = [
        f"Seite: {snap['title']}", f"Adresse: {snap['url']}",
        f"Scrollstand: {snap['scrollY']} von {snap['scrollMax']} Pixel"
        + (" (am Seitenende)" if snap["scrollY"] >= snap["scrollMax"] else ""),
        "", "Bedienbare Elemente:", *(snap["items"] or ["(keine)"]),
    ]
    if snap["more"]:
        lines.append("[... weitere Elemente nicht aufgeführt: scrollen oder gezielter suchen]")
    lines += ["", "Sichtbarer Text:", text or "(leer)"]
    return "\n".join(lines) + _browser_notes()


@tool
def click_element(
    ref: int | None = None, selector: str | None = None, text: str | None = None,
    button: str = "left", double_click: bool = False,
) -> str:
    """Klickt auf ein Element, wie es ein Nutzer mit der Maus tut (scrollt es bei Bedarf automatisch ins Bild).

    ref: Nummer aus get_page_content (bevorzugt). Alternativ selector (CSS) oder text (sichtbarer Text).
    button: "left", "right" oder "middle". double_click: Doppelklick.
    Öffnet der Klick einen neuen Tab, wird dieser aktiv.
    """
    async def op(page):
        loc = await _target(page, ref, selector, text)
        await loc.click(button=button, click_count=2 if double_click else 1, timeout=8000)
        await page.wait_for_timeout(300)

    return _browser_action(f"{_describe(ref, selector, text)} geklickt", op)


@tool
def type_text(
    text: str, ref: int | None = None, selector: str | None = None,
    clear: bool = True, press_enter: bool = False, slowly: bool = False,
) -> str:
    """Schreibt Text in ein Eingabefeld.

    ref: Nummer aus get_page_content (bevorzugt) oder selector (CSS). Ohne beides wird in das Element
        getippt, das gerade den Fokus hat.
    clear: vorhandenen Inhalt ersetzen (Standard) statt anhängen.
    press_enter: danach Enter drücken (z.B. Suche absenden oder Formular abschicken).
    slowly: Taste für Taste tippen (für Felder mit Autovervollständigung oder Eingabeprüfung).
    """
    async def op(page):
        if ref is None and not selector:
            if clear:
                await page.keyboard.press("Control+A")
                await page.keyboard.press("Delete")
            await page.keyboard.type(text, delay=30 if slowly else 0)
        else:
            loc = await _target(page, ref, selector)
            if slowly:
                await loc.click(timeout=8000)
                if clear:
                    await page.keyboard.press("Control+A")
                    await page.keyboard.press("Delete")
                await loc.press_sequentially(text, delay=40)
            elif clear:
                await loc.fill(text, timeout=8000)
            else:
                await loc.click(timeout=8000)
                await page.keyboard.type(text)
        if press_enter:
            await page.keyboard.press("Enter")
            await page.wait_for_timeout(300)

    target = _describe(ref, selector, None) if (ref is not None or selector) else "fokussiertes Feld"
    shown = text if len(text) <= 60 else text[:57] + "..."
    return _browser_action(f"Text '{shown}' in {target} geschrieben" + (" und Enter gedrückt" if press_enter else ""), op)


@tool
def press_key(key: str, ref: int | None = None) -> str:
    """Drückt eine Taste oder Tastenkombination, wie auf der Tastatur.

    key: z.B. "Enter", "Escape", "Tab", "Shift+Tab", "ArrowDown", "PageDown", "Home", "End",
        "Backspace", "Control+A", "Control+C", "Control+V", "F5".
    ref: optional Element, das vorher den Fokus bekommt (Nummer aus get_page_content).
    """
    async def op(page):
        if ref is not None:
            await (await _target(page, ref)).focus(timeout=5000)
        await page.keyboard.press(key)
        await page.wait_for_timeout(200)

    return _browser_action(f"Taste '{key}' gedrückt", op)


@tool
def scroll(direction: str = "down", amount: int = 600, ref: int | None = None) -> str:
    """Scrollt die Seite wie mit dem Mausrad und meldet den neuen Scrollstand.

    direction: "down", "up", "top" (Seitenanfang), "bottom" (Seitenende), "left" oder "right".
    amount: Pixel für down/up/left/right (Standard 600, etwa eine halbe Bildschirmhöhe).
    ref: optional ein scrollbarer Bereich (Nummer aus get_page_content); ohne Angabe scrollt die ganze Seite.
    Danach get_page_content aufrufen, um die neuen Inhalte zu sehen.
    """
    d = (direction or "down").lower()
    if d not in ("down", "up", "top", "bottom", "left", "right"):
        return f"Fehler: Unbekannte Richtung '{direction}'. Erlaubt: down, up, top, bottom, left, right."

    async def op(page):
        js = r"""([d, n, ref]) => {
          const el = ref ? document.querySelector(`[data-na-ref="${ref}"]`) : (document.scrollingElement || document.documentElement);
          if (!el) return null;
          const dx = d === 'left' ? -n : d === 'right' ? n : 0;
          const dy = d === 'up' ? -n : d === 'down' ? n : 0;
          if (d === 'top') el.scrollTo({top: 0, behavior: 'instant'});
          else if (d === 'bottom') el.scrollTo({top: el.scrollHeight, behavior: 'instant'});
          else el.scrollBy({left: dx, top: dy, behavior: 'instant'});
          return {y: Math.round(el.scrollTop), max: Math.max(0, Math.round(el.scrollHeight - el.clientHeight))};
        }"""
        res = await page.evaluate(js, [d, int(amount), ref])
        if res is None:
            raise ValueError(f"Element [{ref}] gibt es nicht (mehr).")
        await page.wait_for_timeout(250)  # nachladende Inhalte (Lazy Loading) abwarten
        return res

    try:
        res = _browser(op)
    except Exception as e:
        return f"Fehler: Scrollen nicht möglich: {_short_error(e)}"
    where = "am Seitenende" if res["y"] >= res["max"] and res["max"] > 0 else "am Seitenanfang" if res["y"] == 0 else "mittendrin"
    return f"Gescrollt ({d}). Scrollstand: {res['y']} von {res['max']} Pixel, {where}.{_browser_notes()}"


@tool
def hover_element(ref: int | None = None, selector: str | None = None, text: str | None = None) -> str:
    """Bewegt die Maus über ein Element (öffnet z.B. Menüs und Tooltips, die erst beim Darüberfahren erscheinen).

    ref: Nummer aus get_page_content (bevorzugt). Alternativ selector (CSS) oder text.
    Das Menü bleibt offen, solange die Maus darüber steht. Danach get_page_content aufrufen, um die neuen
    Einträge zu sehen. Verschwindet das Menü beim Klick auf einen Eintrag, nacheinander erst den Eintrag
    hovern und dann klicken oder alternativ mit press_key ("Enter") bedienen.
    """
    async def op(page):
        await (await _target(page, ref, selector, text)).hover(timeout=8000)
        await page.wait_for_timeout(300)

    return _browser_action(f"Maus über {_describe(ref, selector, text)} bewegt", op)


@tool
def select_option(value: str, ref: int | None = None, selector: str | None = None) -> str:
    """Wählt einen Eintrag in einem Auswahlfeld (Dropdown).

    value: sichtbarer Text oder interner Wert des Eintrags.
    ref: Nummer aus get_page_content (bevorzugt) oder selector (CSS).
    """
    async def op(page):
        loc = await _target(page, ref, selector)
        try:
            await loc.select_option(label=value, timeout=4000)
        except Exception:
            await loc.select_option(value=value, timeout=4000)

    return _browser_action(f"'{value}' in {_describe(ref, selector, None)} gewählt", op)


@tool
def set_checkbox(checked: bool, ref: int | None = None, selector: str | None = None) -> str:
    """Hakt eine Checkbox oder ein Radio-Feld an oder ab.

    checked: true = anhaken, false = Haken entfernen.
    ref: Nummer aus get_page_content (bevorzugt) oder selector (CSS).
    """
    async def op(page):
        await (await _target(page, ref, selector)).set_checked(checked, timeout=8000)

    return _browser_action(f"{_describe(ref, selector, None)} {'angehakt' if checked else 'abgehakt'}", op)


@tool
def upload_file(path: str, ref: int | None = None, selector: str | None = None) -> str:
    """Wählt eine lokale Datei in einem Datei-Auswahlfeld aus (Upload).

    path: Pfad der Datei auf diesem Rechner.
    ref: Nummer des Datei-Felds aus get_page_content (bevorzugt) oder selector (CSS).
    """
    if not os.path.isfile(path):
        return f"Fehler: Datei nicht gefunden: {path}"

    async def op(page):
        await (await _target(page, ref, selector)).set_input_files(path, timeout=8000)

    return _browser_action(f"Datei '{os.path.basename(path)}' ausgewählt", op)


@tool
def drag_and_drop(source_ref: int, target_ref: int) -> str:
    """Zieht ein Element mit der Maus auf ein anderes (Drag & Drop, z.B. Karten, Sortierlisten).

    source_ref: Nummer des Elements, das gezogen wird (aus get_page_content).
    target_ref: Nummer des Ziels.
    """
    async def op(page):
        src = await _target(page, source_ref)
        dst = await _target(page, target_ref)
        await src.drag_to(dst, timeout=8000)
        await page.wait_for_timeout(300)

    return _browser_action(f"[{source_ref}] auf [{target_ref}] gezogen", op)


@tool
def click_at(x: int, y: int, button: str = "left") -> str:
    """Klickt auf eine Bildschirmposition (Pixel im sichtbaren Bereich der Seite, von links oben gemessen).

    Nur für Stellen ohne bedienbares Element in get_page_content, z.B. Karten, Canvas oder Spiele.
    Die Position ergibt sich aus take_screenshot.
    """
    async def op(page):
        await page.mouse.click(x, y, button=button)
        await page.wait_for_timeout(300)

    return _browser_action(f"Auf Position ({x}, {y}) geklickt", op)


@tool
def wait_for(text: str | None = None, seconds: float = 0, timeout_s: int = 15) -> str:
    """Wartet, bis ein Text auf der Seite erscheint, oder eine feste Zeit (für Ladevorgänge und Animationen).

    text: sichtbarer Text, auf den gewartet wird.
    seconds: ohne text: so viele Sekunden warten (höchstens 30).
    timeout_s: längste Wartezeit auf den Text.
    """
    limit = max(1, min(int(timeout_s), 60))

    async def op(page):
        if text:
            await page.get_by_text(text, exact=False).first.wait_for(state="visible", timeout=limit * 1000)
        else:
            await page.wait_for_timeout(int(max(0.0, min(float(seconds), 30.0)) * 1000))

    label = f"Text '{text}' erschienen" if text else f"{max(0.0, min(float(seconds), 30.0)):g} s gewartet"
    return _browser_action(label, op, timeout=limit + 10)


@tool
def go_back() -> str:
    """Geht im Verlauf eine Seite zurück (wie der Zurück-Knopf des Browsers)."""
    async def op(page):
        response = await page.go_back(wait_until="domcontentloaded", timeout=15000)
        return page.url if response else "keine frühere Seite"

    return _browser_action("Zurück", op, timeout=25)


@tool
def go_forward() -> str:
    """Geht im Verlauf eine Seite vor (wie der Vorwärts-Knopf des Browsers)."""
    async def op(page):
        response = await page.go_forward(wait_until="domcontentloaded", timeout=15000)
        return page.url if response else "keine spätere Seite"

    return _browser_action("Vor", op, timeout=25)


@tool
def reload_page() -> str:
    """Lädt die aktuelle Seite neu."""
    async def op(page):
        await page.reload(wait_until="domcontentloaded", timeout=30000)

    return _browser_action("Seite neu geladen", op, timeout=40)


@tool
def list_tabs() -> str:
    """Listet alle offenen Tabs (Nummer, Titel, Adresse) und zeigt, welcher aktiv ist."""
    async def op(s):
        rows = []
        for i, p in enumerate(s.context.pages):
            if p.is_closed():
                continue
            try:
                title = await p.title()
            except Exception:
                title = ""
            rows.append(f"{'*' if p is s.page else ' '} Tab {i}: {title or '(ohne Titel)'} – {p.url}")
        return "\n".join(rows)

    try:
        return (_browser(op, session=True) or "(keine Tabs)") + _browser_notes()
    except Exception as e:
        return f"Fehler: Tabs nicht lesbar: {_short_error(e)}"


@tool
def switch_tab(index: int) -> str:
    """Wechselt zu einem anderen Tab (Nummer aus list_tabs)."""
    async def op(s):
        pages = [p for p in s.context.pages if not p.is_closed()]
        if not 0 <= int(index) < len(pages):
            raise ValueError(f"Tab {index} gibt es nicht (offen: {len(pages)}).")
        s.page = pages[int(index)]
        await s.page.bring_to_front()
        return s.page.url

    return _browser_action(f"Zu Tab {index} gewechselt", op, session=True)


@tool
def new_tab(url: str | None = None) -> str:
    """Öffnet einen neuen Tab (optional mit Adresse) und macht ihn aktiv."""
    async def op(s):
        page = await s.context.new_page()
        s.page = page
        if url:
            await page.goto(url, wait_until="domcontentloaded", timeout=30000)
        return page.url

    return _browser_action("Neuer Tab geöffnet", op, timeout=40, session=True)


@tool
def close_tab(index: int | None = None) -> str:
    """Schließt einen Tab (Nummer aus list_tabs) oder den aktiven, wenn keine Nummer angegeben ist.
    Der letzte Tab bleibt bestehen."""
    async def op(s):
        pages = [p for p in s.context.pages if not p.is_closed()]
        if len(pages) <= 1:
            raise ValueError("Der letzte Tab kann nicht geschlossen werden.")
        victim = s.page if index is None else pages[int(index)] if 0 <= int(index) < len(pages) else None
        if victim is None:
            raise ValueError(f"Tab {index} gibt es nicht (offen: {len(pages)}).")
        await victim.close()
        s.page = [p for p in s.context.pages if not p.is_closed()][-1]
        return f"aktiv ist jetzt {s.page.url}"

    return _browser_action("Tab geschlossen", op, session=True)


@tool
def take_screenshot(filename: str = "screenshot.png", full_page: bool = False) -> str:
    """Macht einen Screenshot der Seite und gibt die Bilddaten direkt als Base64-kodiertes JSON zurück.

    full_page: true = die ganze Seite statt nur des sichtbaren Bereichs.
    """
    async def op(page):
        return await page.screenshot(path=filename, full_page=full_page)

    try:
        screenshot_bytes = _browser(op)
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


@tool
def close_browser() -> str:
    """Beendet den Browser dieses Agenten (Tabs und Anmeldungen gehen verloren). Der nächste Browser-Befehl
    startet einen frischen."""
    from browser import manager

    try:
        closed = manager.close_now(_browser_key())
    except Exception as e:
        return f"Fehler: Browser nicht beendet: {_short_error(e)}"
    return "Browser beendet." if closed else "Es lief kein Browser."


BROWSER_TOOL_NAMES = (
    "navigate", "get_page_content", "click_element", "type_text", "press_key", "scroll", "hover_element",
    "select_option", "set_checkbox", "upload_file", "drag_and_drop", "click_at", "wait_for", "go_back",
    "go_forward", "reload_page", "list_tabs", "switch_tab", "new_tab", "close_tab", "take_screenshot",
    "close_browser",
)
"""Tools, die den Browser des aufrufenden Chats bedienen."""


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

_caller_chat: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("_caller_chat", default=None)
_CALLER_AWARE = ("create_subagent", "create_assistant_agent", *BROWSER_TOOL_NAMES)


def bind_caller(toolset, chat_id: str) -> None:
    """Gibt den Tools in ``toolset`` die ID des aufrufenden Chats mit (für die Zuordnung zum Ursprungs-Chat)."""
    import dataclasses
    import functools

    for name in _CALLER_AWARE:
        spec = toolset.get(name)
        if spec is None:
            continue

        def wrap(func, chat_id=chat_id):
            @functools.wraps(func)
            def run(*args, **kwargs):
                token = _caller_chat.set(chat_id)
                try:
                    return func(*args, **kwargs)
                finally:
                    _caller_chat.reset(token)

            return run

        toolset.remove(name)
        toolset.add(dataclasses.replace(spec, func=wrap(spec.func)))


def subagent_chat_ids(service) -> set[str]:
    """IDs aller Chats von Subagenten und Assistent Agents (für die Chat-Liste ausgeblendet)."""
    bind_service(service)
    with _subagents_lock:
        return {info["chat_id"] for info in _subagents.values()}


def subagent_tree(service, chat_id: str | None) -> list[dict]:
    """Subagenten des Chats ``chat_id`` mit ihren Assistent Agents (``children``).

    Ist ``chat_id`` der Chat eines Subagenten, steht dieser mit seinen Assistent Agents allein in der Liste.
    Einträge gelöschter Chats entfallen. Ältere Einträge ohne bekannten Ursprungs-Chat erscheinen nirgends.
    """
    bind_service(service)
    if not chat_id:
        return []
    with _subagents_lock:
        entries = {sid: dict(info) for sid, info in _subagents.items()}

    from browser import manager

    def item(sid: str, info: dict) -> dict | None:
        chat = service.store.get_chat(info["chat_id"])
        if chat is None:
            return None
        return {
            "browser": manager.state(info["chat_id"]),
            "id": sid, "kind": info.get("kind", "subagent"), "chat_id": info["chat_id"],
            "title": chat["title"], "agent": info.get("agent"), "provider_id": info.get("provider_id"),
            "model": info.get("model"), "created_at": chat["created_at"], "children": [],
            "running": service.active_turn(info["chat_id"]) is not None,
        }

    own = next((sid for sid, info in entries.items() if info.get("chat_id") == chat_id and info.get("kind", "subagent") == "subagent"), None)
    items = {sid: it for sid, info in entries.items() if (it := item(sid, info)) is not None}
    roots: list[dict] = []
    for sid, it in items.items():
        parent = items.get(entries[sid].get("parent") or "")
        if it["kind"] == "assistant" and parent is not None and parent["kind"] == "subagent":
            parent["children"].append(it)
        elif own is None and entries[sid].get("origin") == chat_id and it["kind"] == "subagent":
            roots.append(it)
    if own is not None and own in items:
        roots = [items[own]]
    for it in items.values():
        it["children"].sort(key=lambda c: c["created_at"])
    roots.sort(key=lambda c: c["created_at"], reverse=True)
    return roots


def bind_service(service) -> None:
    """Verbindet die Agenten-Tools mit dem ChatService (wird beim Laden der Tools aufgerufen)."""
    global _service
    if _service is service:
        return
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


def _create_child(
    kind: str, start_prompt: str, provider: str | None, model: str | None, agent: str | None,
    wait_for_answer: bool = False, timeout_s: int = 300,
) -> str:
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
        parent = None
        caller = _caller_chat.get()
        if kind == "assistant" and caller:
            with _subagents_lock:
                parent = next((s for s, i in _subagents.items() if i.get("chat_id") == caller), None)
        sid = cfg["prefix"] + secrets.token_hex(4)
        title = f"{cfg['label']} {sid}" + (f" ({agent_info['name']})" if agent_info else "")
        chat = service.create_chat(provider_id=prov["id"], model=model or None, title=title)
        try:
            turn, _ = service.start_turn(
                chat["id"], start_prompt, provider_id=prov["id"], model=model or None,
                system_prompt=system_prompt, exclude_tools=cfg["excluded"],
            )
        except BaseException:
            service.delete_chat(chat["id"])
            raise
        with _subagents_lock:
            _subagents[sid] = {
                "kind": kind, "parent": parent, "origin": caller, "chat_id": chat["id"], "provider_id": prov["id"], "model": model or None,
                "system_prompt": system_prompt, "agent": agent_info["name"] if agent_info else None,
            }
            _save_subagents()
    except Exception as e:
        return f"{_ERR} {cfg['label']} konnte nicht erstellt werden: {e}"
    result: dict = {f"{kind}_id" if kind == "assistant" else "subagent_id": sid, "status": "gestartet"}
    if wait_for_answer:
        answer = _wait_for_turn(turn, float(max(1, min(int(timeout_s), 3600))), cfg["label"])
        if answer.startswith(f"{_ERR} Zeitüberschreitung"):
            result["status"] = "arbeitet noch"
        else:
            result["status"] = "fehlgeschlagen" if answer.startswith(_ERR) else "abgeschlossen"
        result["answer"] = answer
    return json.dumps(result, ensure_ascii=False)


@tool
def create_subagent(
    start_prompt: str,
    provider: str | None = None,
    model: str | None = None,
    agent: str | None = None,
    wait_for_answer: bool = False,
    timeout_s: int = 300,
) -> str:
    """Erstellt einen Subagenten, der den Startprompt sofort bearbeitet.

    Gibt die Subagenten-ID zurück ("subagent_id"). Mit wait_for_answer=true wartet der Aufruf
    zusätzlich auf die Antwort auf den Startprompt und liefert sie als "answer" mit; sonst arbeitet
    der Subagent im Hintergrund und die Antwort holt send_to_subagent mit leerer Nachricht und
    wait_for_answer=true ab.

    start_prompt: erste Aufgabe/Nachricht an den Subagenten.
    provider: ID oder Name des Providers (z.B. "lmstudio" oder "claude"); leer = Einstellung
        "Subagenten" (bzw. "Agenten", wenn ein vorgefertigter Agent angegeben ist).
    model: Modell des Providers; leer = Modell aus den Einstellungen bzw. Standardmodell.
    agent: optional, Name eines vorgefertigten Agenten (dessen Prompt wird der System-Prompt).
    wait_for_answer: true = auf die Antwort des Subagenten warten (Standard: false).
    timeout_s: längste Wartezeit in Sekunden bei wait_for_answer=true. Danach kommt die ID mit einem
        Hinweis zurück; der Subagent arbeitet weiter.
    Subagenten können selbst keine Subagenten erstellen, aber Assistent Agents (create_assistant_agent).
    """
    return _create_child("subagent", start_prompt, provider, model, agent, wait_for_answer, timeout_s)


@tool
def create_assistant_agent(
    start_prompt: str,
    provider: str | None = None,
    model: str | None = None,
    agent: str | None = None,
    wait_for_answer: bool = False,
    timeout_s: int = 300,
) -> str:
    """Erstellt einen Assistent Agent (Hilfsagent eines Subagenten), der den Startprompt sofort bearbeitet.

    Gibt die ID des Assistent Agents zurück ("assistant_id"). Mit wait_for_answer=true wartet der Aufruf
    zusätzlich auf die Antwort auf den Startprompt und liefert sie als "answer" mit; sonst arbeitet der
    Assistent Agent im Hintergrund und die Antwort holt send_to_assistant_agent mit leerer Nachricht und
    wait_for_answer=true ab.

    start_prompt: erste Aufgabe/Nachricht an den Assistent Agent.
    provider: ID oder Name des Providers; leer = Einstellung "Assistent Agents"
        (bzw. "Agenten", wenn ein vorgefertigter Agent angegeben ist).
    model: Modell des Providers; leer = Modell aus den Einstellungen bzw. Standardmodell.
    agent: optional, Name eines vorgefertigten Agenten (dessen Prompt wird der System-Prompt).
    wait_for_answer: true = auf die Antwort des Assistent Agents warten (Standard: false).
    timeout_s: längste Wartezeit in Sekunden bei wait_for_answer=true. Danach kommt die ID mit einem
        Hinweis zurück; der Assistent Agent arbeitet weiter.
    Assistent Agents können selbst keine weiteren Agenten erstellen.
    """
    return _create_child("assistant", start_prompt, provider, model, agent, wait_for_answer, timeout_s)


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
