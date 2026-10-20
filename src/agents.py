"""Vorgefertigte Agenten: Name, Beschreibung, Prompt und Assets (Dateien).

Jeder Agent liegt als Ordner unter ``<data>/agents/<id>/``::

    agent.md        Front Matter (name, description) + Prompt als Markdown-Text
    assets/         beliebige Dateien (flach, ohne Unterordner)

Import: eine ``.md``-Datei (Front Matter oder ``# Titel`` als Name) oder eine
``.zip`` mit einer Markdown-Datei (bevorzugt ``agent.md``) und weiteren Dateien
als Assets. Fehler werden als ``ValueError`` (ungültige Eingabe) bzw.
``NotFound`` (unbekannter Agent oder Asset) gemeldet.
"""

from __future__ import annotations

import io
import json
import os
import re
import secrets
import shutil
import threading
import unicodedata
import zipfile
from pathlib import Path
from typing import Any

MAX_NAME_CHARS = 100
MAX_DESCRIPTION_CHARS = 2000
MAX_PROMPT_CHARS = 100_000
MAX_ASSET_BYTES = 10 * 1024 * 1024
MAX_ASSETS = 100
MAX_TOTAL_ASSET_BYTES = 100 * 1024 * 1024
MAX_IMPORT_BYTES = 30 * 1024 * 1024
MAX_ZIP_UNPACKED_BYTES = 100 * 1024 * 1024

_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,80}$")
_RESERVED_NAMES = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}
_MD_SUFFIXES = (".md", ".markdown")


ASSISTANT_TOOLS = ("create_assistant_agent", "send_to_assistant_agent")
SUBAGENT_TOOLS = ("create_subagent", "send_to_subagent")

MAIN_EXCLUDED_TOOLS = ASSISTANT_TOOLS
"""Der Haupt-Chat startet Subagenten, aber keine Assistent Agents."""
SUBAGENT_EXCLUDED_TOOLS = SUBAGENT_TOOLS
"""Subagenten starten Assistent Agents, aber keine weiteren Subagenten."""
ASSISTANT_EXCLUDED_TOOLS = SUBAGENT_TOOLS + ASSISTANT_TOOLS
"""Assistent Agents starten keine weiteren Agenten."""


class NotFound(LookupError):
    """Agent oder Asset existiert nicht."""


# --- Markdown-Format ---------------------------------------------------------------

def _meta_value(raw: str) -> str:
    value = raw.strip()
    if value.startswith('"'):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, str):
                return parsed
        except ValueError:
            pass
    if len(value) >= 2 and value[0] == value[-1] == "'":
        return value[1:-1]
    return value


def parse_agent_md(text: str, fallback_name: str = "") -> dict[str, str]:
    """Liest ``name``, ``description`` und ``prompt`` aus Markdown (mit oder ohne Front Matter)."""
    lines = text.lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n").split("\n")
    meta: dict[str, str] = {}
    body_start = 0
    first = next((i for i, line in enumerate(lines) if line.strip()), None)
    if first is not None and lines[first].strip() == "---":
        end = next((i for i in range(first + 1, len(lines)) if lines[i].strip() == "---"), None)
        if end is not None:
            for line in lines[first + 1:end]:
                key, sep, value = line.partition(":")
                if sep and key.strip():
                    meta[key.strip().lower()] = _meta_value(value)
            body_start = end + 1
    body = lines[body_start:]
    name = meta.get("name", "").strip()
    if not meta and not name:
        # Ohne Front Matter: erste Überschrift "# Titel" ist der Name
        idx = next((i for i, line in enumerate(body) if line.strip()), None)
        if idx is not None and re.match(r"^#\s+\S", body[idx]):
            name = body[idx].lstrip("#").strip()
            body = body[idx + 1:]
    return {
        "name": name or fallback_name,
        "description": meta.get("description", "").strip(),
        "prompt": "\n".join(body).strip(),
    }


def render_agent_md(name: str, description: str, prompt: str) -> str:
    dump = lambda s: json.dumps(s, ensure_ascii=False)  # noqa: E731 - Zeilenumbrüche bleiben als \n erhalten
    return f"---\nname: {dump(name)}\ndescription: {dump(description)}\n---\n\n{prompt.strip()}\n"


# --- Validierung ---------------------------------------------------------------------

def _text(data: dict[str, Any], key: str, label: str, max_chars: int, *, required: bool = False) -> str | None:
    if key not in data:
        return None
    value = data[key]
    if value is None:
        value = ""
    if not isinstance(value, str):
        raise ValueError(f"Feld '{key}' muss ein String sein.")
    value = value.strip()
    if required and not value:
        raise ValueError(f"Bitte {label} angeben.")
    if len(value) > max_chars:
        raise ValueError(f"{label[0].upper()}{label[1:]} ist zu lang (höchstens {max_chars} Zeichen).")
    return value


def safe_asset_name(raw: str) -> str:
    """Macht aus einem beliebigen Dateinamen einen sicheren, flachen Dateinamen."""
    name = str(raw or "").replace("\\", "/").rsplit("/", 1)[-1]
    name = re.sub(r'[\x00-\x1f<>:"|?*]', "_", name).strip().rstrip(". ")
    if not name or name in (".", ".."):
        raise ValueError("Ungültiger Dateiname.")
    if name.split(".")[0].upper() in _RESERVED_NAMES:
        name = "_" + name
    return name[:120].rstrip(". ") or "_"


def _slug(name: str) -> str:
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", ascii_name.lower()).strip("-")[:40] or "agent"


def _unique(name: str, taken: set[str]) -> str:
    """Hängt bei Namenskollision (ohne Beachtung der Groß-/Kleinschreibung) ``-2``, ``-3`` … an."""
    stem, dot, ext = name.rpartition(".")
    if not dot:
        stem, ext = name, ""
    candidate, n = name, 2
    while candidate.casefold() in taken:
        candidate = f"{stem}-{n}{'.' + ext if ext else ''}"
        n += 1
    return candidate


def _decode(raw: bytes, label: str) -> str:
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise ValueError(f"{label} ist keine UTF-8-Textdatei.") from None


def _check_asset_size(name: str, size: int) -> None:
    if size > MAX_ASSET_BYTES:
        raise ValueError(f"Die Datei „{name}“ ist zu groß (höchstens {MAX_ASSET_BYTES // (1024 * 1024)} MB).")


# --- ZIP-Import ----------------------------------------------------------------------

def _read_zip(raw: bytes, fallback_name: str) -> tuple[dict[str, str], list[tuple[str, bytes]]]:
    try:
        zf = zipfile.ZipFile(io.BytesIO(raw))
    except zipfile.BadZipFile:
        raise ValueError("Ungültige ZIP-Datei.") from None
    with zf:
        infos = []
        for info in zf.infolist():
            parts = [p for p in info.filename.replace("\\", "/").split("/") if p]
            if info.is_dir() or not parts or "__MACOSX" in parts or parts[-1].startswith("."):
                continue
            infos.append((info, parts))
        if len(infos) > MAX_ASSETS + 1:
            raise ValueError(f"Die ZIP-Datei enthält zu viele Dateien (höchstens {MAX_ASSETS} Assets).")
        if sum(i.file_size for i, _ in infos) > MAX_ZIP_UNPACKED_BYTES:
            raise ValueError("Die ZIP-Datei ist entpackt zu groß.")
        mds = [(i, p) for i, p in infos if p[-1].lower().endswith(_MD_SUFFIXES)]
        if not mds:
            raise ValueError("Die ZIP-Datei enthält keine Markdown-Datei (.md).")
        main, main_parts = min(mds, key=lambda e: (e[1][-1].lower() != "agent.md", len(e[1]), "/".join(e[1]).lower()))

        def read(info: zipfile.ZipInfo, limit: int) -> bytes:
            try:
                with zf.open(info) as f:
                    data = f.read(limit + 1)
            except Exception:  # noqa: BLE001 - verschlüsselt, defekt oder nicht unterstützt
                raise ValueError(f"Die Datei „{info.filename}“ in der ZIP-Datei kann nicht gelesen werden.") from None
            if len(data) > limit:
                raise ValueError(f"Die Datei „{info.filename}“ in der ZIP-Datei ist zu groß.")
            return data

        stem = main_parts[-1].rsplit(".", 1)[0]
        fields = parse_agent_md(_decode(read(main, MAX_PROMPT_CHARS * 4), "Die Markdown-Datei"),
                                fallback_name or stem)
        assets: list[tuple[str, bytes]] = []
        taken: set[str] = set()
        for info, parts in infos:
            if info is main:
                continue
            name = _unique(safe_asset_name(parts[-1]), taken)
            taken.add(name.casefold())
            assets.append((name, read(info, MAX_ASSET_BYTES)))
        return fields, assets


# --- Speicher ------------------------------------------------------------------------

class AgentStore:
    """Dateibasierter Speicher der Agenten (threadsicher)."""

    def __init__(self, root: str | os.PathLike[str]):
        self.root = Path(root)
        self._lock = threading.RLock()

    # -- intern --

    def _dir(self, agent_id: str) -> Path:
        if not isinstance(agent_id, str) or not _ID_RE.match(agent_id):
            raise NotFound("Agent nicht gefunden.")
        path = self.root / agent_id
        if not (path / "agent.md").is_file():
            raise NotFound("Agent nicht gefunden.")
        return path

    @staticmethod
    def _assets(path: Path) -> list[dict[str, Any]]:
        folder = path / "assets"
        if not folder.is_dir():
            return []
        files = [p for p in folder.iterdir() if p.is_file()]
        return [{"name": p.name, "size": p.stat().st_size} for p in sorted(files, key=lambda p: p.name.casefold())]

    def _read(self, path: Path) -> dict[str, Any]:
        text = (path / "agent.md").read_text(encoding="utf-8-sig", errors="replace")
        fields = parse_agent_md(text, path.name)
        return {"id": path.name, **fields, "assets": self._assets(path)}

    @staticmethod
    def _write_md(path: Path, name: str, description: str, prompt: str) -> None:
        tmp = path / "agent.md.tmp"
        tmp.write_text(render_agent_md(name, description, prompt), encoding="utf-8", newline="\n")
        os.replace(tmp, path / "agent.md")

    def _create(self, name: str, description: str, prompt: str, assets: list[tuple[str, bytes]]) -> dict[str, Any]:
        for asset_name, data in assets:
            _check_asset_size(asset_name, len(data))
        if sum(len(d) for _, d in assets) > MAX_TOTAL_ASSET_BYTES:
            raise ValueError("Die Assets sind zusammen zu groß.")
        with self._lock:
            self.root.mkdir(parents=True, exist_ok=True)
            while True:
                agent_id = f"{_slug(name)}-{secrets.token_hex(3)}"
                if not (self.root / agent_id).exists():
                    break
            tmp = self.root / f".tmp-{agent_id}"
            try:
                (tmp / "assets").mkdir(parents=True)
                self._write_md(tmp, name, description, prompt)
                for asset_name, data in assets:
                    (tmp / "assets" / asset_name).write_bytes(data)
                os.replace(tmp, self.root / agent_id)
            except BaseException:
                shutil.rmtree(tmp, ignore_errors=True)
                raise
            return self._read(self.root / agent_id)

    # -- Agenten --

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            if not self.root.is_dir():
                return []
            agents = [
                self._read(p) for p in self.root.iterdir()
                if p.is_dir() and _ID_RE.match(p.name) and (p / "agent.md").is_file()
            ]
        return sorted(agents, key=lambda a: (a["name"].casefold(), a["id"]))

    def get(self, agent_id: str) -> dict[str, Any]:
        with self._lock:
            return self._read(self._dir(agent_id))

    def find(self, ref: str) -> dict[str, Any]:
        """Agent per ID oder Name (ohne Beachtung der Groß-/Kleinschreibung).

        Wirft ``NotFound`` (kein Treffer) oder ``ValueError`` (mehrere Agenten mit diesem Namen).
        """
        ref = (ref or "").strip()
        agents = self.list()
        by_id = [a for a in agents if a["id"] == ref]
        matches = by_id or [a for a in agents if a["name"].casefold() == ref.casefold()]
        if not matches:
            raise NotFound(f"Agent „{ref}“ nicht gefunden.")
        if len(matches) > 1:
            ids = ", ".join(a["id"] for a in matches)
            raise ValueError(f"Der Name „{ref}“ ist nicht eindeutig (IDs: {ids}).")
        return matches[0]

    def create(self, data: dict[str, Any], assets: list[tuple[str, bytes]] | None = None) -> dict[str, Any]:
        name = _text(data, "name", "einen Namen", MAX_NAME_CHARS, required=True)
        if name is None:
            raise ValueError("Bitte einen Namen angeben.")
        description = _text(data, "description", "die Beschreibung", MAX_DESCRIPTION_CHARS) or ""
        prompt = _text(data, "prompt", "der Prompt", MAX_PROMPT_CHARS) or ""
        prepared: list[tuple[str, bytes]] = []
        taken: set[str] = set()
        for filename, content in assets or []:
            asset_name = _unique(safe_asset_name(filename), taken)
            taken.add(asset_name.casefold())
            prepared.append((asset_name, content))
        if len(prepared) > MAX_ASSETS:
            raise ValueError(f"Ein Agent kann höchstens {MAX_ASSETS} Assets haben.")
        return self._create(name, description, prompt, prepared)

    def assets_dir(self, agent_id: str) -> Path:
        """Ordner mit den Asset-Dateien eines Agenten."""
        with self._lock:
            return (self._dir(agent_id) / "assets").resolve()

    def update(self, agent_id: str, data: dict[str, Any]) -> dict[str, Any]:
        name = _text(data, "name", "einen Namen", MAX_NAME_CHARS, required=True)
        description = _text(data, "description", "die Beschreibung", MAX_DESCRIPTION_CHARS)
        prompt = _text(data, "prompt", "der Prompt", MAX_PROMPT_CHARS)
        with self._lock:
            path = self._dir(agent_id)
            current = self._read(path)
            self._write_md(
                path,
                name if name is not None else current["name"],
                description if description is not None else current["description"],
                prompt if prompt is not None else current["prompt"],
            )
            return self._read(path)

    def delete(self, agent_id: str) -> None:
        with self._lock:
            shutil.rmtree(self._dir(agent_id))

    def import_file(self, filename: str, raw: bytes) -> dict[str, Any]:
        """Legt aus einer ``.md`` oder ``.zip`` einen neuen Agenten an."""
        if len(raw) > MAX_IMPORT_BYTES:
            raise ValueError(f"Die Datei ist zu groß (höchstens {MAX_IMPORT_BYTES // (1024 * 1024)} MB).")
        base = safe_asset_name(filename or "agent.md")
        lower = base.lower()
        stem = base.rsplit(".", 1)[0]
        if lower.endswith(".zip") or (not lower.endswith(_MD_SUFFIXES) and raw[:4] == b"PK\x03\x04"):
            fields, assets = _read_zip(raw, stem if lower.endswith(".zip") else "")
        elif lower.endswith(_MD_SUFFIXES):
            fields, assets = parse_agent_md(_decode(raw, "Die Datei"), stem), []
        else:
            raise ValueError("Nur Markdown-Dateien (.md) und ZIP-Dateien (.zip) werden unterstützt.")
        return self._create(
            _text(fields, "name", "einen Namen", MAX_NAME_CHARS, required=True) or "Agent",
            _text(fields, "description", "die Beschreibung", MAX_DESCRIPTION_CHARS) or "",
            _text(fields, "prompt", "der Prompt", MAX_PROMPT_CHARS) or "",
            assets,
        )

    # -- Assets --

    def add_assets(self, agent_id: str, files: list[tuple[str, bytes]]) -> dict[str, Any]:
        """Fügt Dateien hinzu; gleichnamige Assets werden ersetzt."""
        prepared: list[tuple[str, bytes]] = []
        for filename, data in files:
            name = safe_asset_name(filename)
            _check_asset_size(name, len(data))
            prepared.append((name, data))
        with self._lock:
            path = self._dir(agent_id)
            folder = path / "assets"
            folder.mkdir(exist_ok=True)
            existing = {a["name"].casefold(): a for a in self._assets(path)}
            sizes = {k: a["size"] for k, a in existing.items()}
            for name, data in prepared:
                sizes[name.casefold()] = len(data)
            if len(sizes) > MAX_ASSETS:
                raise ValueError(f"Ein Agent kann höchstens {MAX_ASSETS} Assets haben.")
            if sum(sizes.values()) > MAX_TOTAL_ASSET_BYTES:
                raise ValueError("Die Assets sind zusammen zu groß.")
            for name, data in prepared:
                old = existing.get(name.casefold())
                target = folder / (old["name"] if old else name)
                tmp = folder / f".upload-{secrets.token_hex(4)}"
                try:
                    tmp.write_bytes(data)
                    os.replace(tmp, target)
                finally:
                    tmp.unlink(missing_ok=True)
            return self._read(path)

    def asset_path(self, agent_id: str, name: str) -> Path:
        with self._lock:
            path = self._dir(agent_id)
            for asset in self._assets(path):
                if asset["name"] == name:
                    return path / "assets" / name
        raise NotFound("Asset nicht gefunden.")

    def delete_asset(self, agent_id: str, name: str) -> dict[str, Any]:
        with self._lock:
            self.asset_path(agent_id, name).unlink()
            return self._read(self._dir(agent_id))
