"""Persistenz: Chats und Nachrichten in SQLite (``ChatStore``), Konfiguration als JSON (``ConfigStore``).

- ``ChatStore`` öffnet pro Aufruf eine eigene SQLite-Verbindung (WAL-Modus,
  ``foreign_keys=ON``) und ist damit ohne gemeinsamen Zustand threadsicher.
- ``ConfigStore`` liest und schreibt ``data/config.json`` (UTF-8, ``indent=2``)
  atomar über eine temporäre Datei und ``os.replace``.

Schema-Versionen (``PRAGMA user_version``):

- 1: Tabellen ``chats`` und ``messages``.
- 2: ``chats`` bekommt ``last_turn_id``, ``last_turn_user_id`` und
  ``last_turn_status``: der zuletzt gestartete Turn des Chats, die
  User-Nachricht, auf die er antwortet, und sein Status (``running``,
  ``completed``, ``cancelled``, ``error``). Damit erkennt der ChatService, ob
  ein "Erneut versuchen" noch passt (siehe ``chat_service``). Die Migration
  von 1 auf 2 ergänzt die Spalten (``ALTER TABLE``, alte Chats: ``NULL``).
"""

from __future__ import annotations

import copy
import json
import os
import sqlite3
import threading
import time
import uuid
from collections.abc import Iterable
from contextlib import closing
from pathlib import Path
from typing import Any

from chat_types import Message, iso_now

SCHEMA_VERSION = 2

# Felder, die ``update_chat`` ändern darf
CHAT_FIELDS = ("title", "provider_id", "model", "provider_state")

DEFAULT_CHAT_TITLE = "Neuer Chat"

TURN_STATUSES = ("running", "completed", "cancelled", "error")
"""Werte für ``last_turn_status``; ``running`` = gestartet, aber (noch) nicht beendet."""

MIN_SECRET_LEN = 8
"""Mindestlänge eines Geheimnisses, das in Texten durch ``***`` ersetzt wird.

Gilt für API-Keys (``configured_keys``/``redact``) und alle Geheimnisse, die
der ChatService schwärzt. Kürzere Keys (z. B. der Platzhalter ``"ollama"``)
sind erlaubt, werden aber nie als Schwärz-Muster benutzt: sonst würden normale
Wörter in Titeln, Nachrichten oder JSON (``"a"`` in ``false``) verfälscht.
"""

# Spalten, die mit Schema-Version 2 dazukamen (Migration per ALTER TABLE)
_V2_CHAT_COLUMNS = (
    ("last_turn_id", "TEXT"),
    ("last_turn_user_id", "TEXT"),
    ("last_turn_status", "TEXT"),
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS chats (
    id                TEXT PRIMARY KEY,
    title             TEXT NOT NULL,
    provider_id       TEXT,
    model             TEXT,
    provider_state    TEXT NOT NULL DEFAULT '{}',
    created_at        TEXT NOT NULL,
    updated_at        TEXT NOT NULL,
    last_turn_id      TEXT,
    last_turn_user_id TEXT,
    last_turn_status  TEXT
);
CREATE TABLE IF NOT EXISTS messages (
    id         TEXT PRIMARY KEY,
    chat_id    TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
    seq        INTEGER NOT NULL,
    role       TEXT NOT NULL,
    data       TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE (chat_id, seq)
);
CREATE INDEX IF NOT EXISTS idx_chats_updated ON chats(updated_at);
"""


def new_chat_id() -> str:
    """Erzeugt eine neue Chat-ID der Form ``c_<32 hex>``."""
    return "c_" + uuid.uuid4().hex


# --- Chats und Nachrichten (SQLite) ------------------------------------------

class ChatStore:
    """Speichert Chats und ihre Nachrichten in einer SQLite-Datenbank.

    Chats werden als Dicts zurückgegeben:
    ``{"id", "title", "provider_id", "model", "provider_state": dict, "created_at", "updated_at"}``.
    """

    # Wiederholungen beim Anhängen, falls die Datenbank gesperrt ist oder eine seq kollidiert
    APPEND_RETRIES = 20

    def __init__(self, db_path: str | os.PathLike[str]):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    # -- intern --

    def _connect(self) -> sqlite3.Connection:
        """Öffnet eine neue Verbindung (Autocommit, Transaktionen werden explizit gesteuert)."""
        conn = sqlite3.connect(self.db_path, timeout=30, isolation_level=None)
        try:
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("PRAGMA busy_timeout=30000")
        except BaseException:
            conn.close()
            raise
        return conn

    def _init_schema(self) -> None:
        """Legt fehlende Tabellen an und migriert ältere Schema-Versionen (idempotent).

        Die Migration läuft in einer schreibenden Transaktion; startet ein
        zweiter Prozess gleichzeitig, wartet er und findet die Spalten schon vor.
        """
        with closing(self._connect()) as conn:
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            if version > SCHEMA_VERSION:
                raise RuntimeError(
                    f"Datenbank {self.db_path} hat Schema-Version {version}, "
                    f"unterstützt wird höchstens {SCHEMA_VERSION}."
                )
            conn.executescript(_SCHEMA)
            if version == SCHEMA_VERSION:
                return
            conn.execute("BEGIN IMMEDIATE")
            try:
                existing = {row["name"] for row in conn.execute("PRAGMA table_info(chats)")}
                for column, decl in _V2_CHAT_COLUMNS:
                    if column not in existing:
                        conn.execute(f"ALTER TABLE chats ADD COLUMN {column} {decl}")
                conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
                conn.execute("COMMIT")
            except BaseException:
                conn.execute("ROLLBACK")
                raise

    @staticmethod
    def _row_to_chat(row: sqlite3.Row) -> dict[str, Any]:
        try:
            state = json.loads(row["provider_state"] or "{}")
        except json.JSONDecodeError:
            state = {}
        return {
            "id": row["id"],
            "title": row["title"],
            "provider_id": row["provider_id"],
            "model": row["model"],
            "provider_state": state if isinstance(state, dict) else {},
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    # -- Chats --

    def list_chats(self) -> list[dict[str, Any]]:
        """Alle Chats, zuletzt geänderte zuerst."""
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT * FROM chats ORDER BY updated_at DESC, created_at DESC, rowid DESC"
            ).fetchall()
        return [self._row_to_chat(r) for r in rows]

    def create_chat(
        self,
        title: str | None = None,
        provider_id: str | None = None,
        model: str | None = None,
    ) -> dict[str, Any]:
        """Legt einen leeren Chat an und gibt ihn zurück."""
        now = iso_now()
        chat_id = new_chat_id()
        with closing(self._connect()) as conn:
            conn.execute(
                "INSERT INTO chats (id, title, provider_id, model, provider_state, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, '{}', ?, ?)",
                (chat_id, title or DEFAULT_CHAT_TITLE, provider_id, model, now, now),
            )
            row = conn.execute("SELECT * FROM chats WHERE id = ?", (chat_id,)).fetchone()
        return self._row_to_chat(row)

    def get_chat(self, chat_id: str) -> dict[str, Any] | None:
        """Der Chat mit dieser ID oder ``None``."""
        with closing(self._connect()) as conn:
            row = conn.execute("SELECT * FROM chats WHERE id = ?", (chat_id,)).fetchone()
        return self._row_to_chat(row) if row else None

    def update_chat(self, chat_id: str, **fields: Any) -> dict[str, Any] | None:
        """Ändert ``title``, ``provider_id``, ``model`` und/oder ``provider_state``.

        Setzt ``updated_at`` neu. Gibt den geänderten Chat zurück oder ``None``,
        wenn es ihn nicht gibt. Unbekannte Felder führen zu ``ValueError``.
        """
        unknown = set(fields) - set(CHAT_FIELDS)
        if unknown:
            raise ValueError(f"Unbekannte Chat-Felder: {', '.join(sorted(unknown))}")
        values: dict[str, Any] = {}
        for key, value in fields.items():
            if key == "provider_state":
                values[key] = json.dumps(value or {}, ensure_ascii=False)
            elif key == "title":
                values[key] = (value or "").strip() or DEFAULT_CHAT_TITLE
            else:
                values[key] = value
        values["updated_at"] = iso_now()
        assignments = ", ".join(f"{k} = ?" for k in values)
        with closing(self._connect()) as conn:
            cur = conn.execute(
                f"UPDATE chats SET {assignments} WHERE id = ?", (*values.values(), chat_id)
            )
            if cur.rowcount == 0:
                return None
            row = conn.execute("SELECT * FROM chats WHERE id = ?", (chat_id,)).fetchone()
        return self._row_to_chat(row)

    def delete_chat(self, chat_id: str) -> bool:
        """Löscht den Chat samt Nachrichten. ``True``, wenn es ihn gab."""
        with closing(self._connect()) as conn:
            cur = conn.execute("DELETE FROM chats WHERE id = ?", (chat_id,))
            return cur.rowcount > 0

    # -- Status des letzten Turns --

    def mark_turn_started(self, chat_id: str, turn_id: str, user_message_id: str) -> None:
        """Merkt sich den gerade gestarteten Turn (Status ``running``) und seine User-Nachricht.

        Ändert ``updated_at`` nicht. Wirft ``KeyError``, wenn es den Chat nicht gibt.
        """
        with closing(self._connect()) as conn:
            cur = conn.execute(
                "UPDATE chats SET last_turn_id = ?, last_turn_user_id = ?, last_turn_status = 'running' "
                "WHERE id = ?",
                (turn_id, user_message_id, chat_id),
            )
            if cur.rowcount == 0:
                raise KeyError(chat_id)

    def mark_turn_finished(self, chat_id: str, turn_id: str, status: str) -> bool:
        """Setzt den Endstatus (``completed``, ``cancelled`` oder ``error``) des Turns ``turn_id``.

        Wirkt nur, solange ``turn_id`` der zuletzt gestartete Turn des Chats
        ist (ein verspäteter Worker überschreibt keinen neueren Turn). Gibt
        zurück, ob etwas geändert wurde. Ändert ``updated_at`` nicht.
        """
        if status not in TURN_STATUSES or status == "running":
            raise ValueError(f"Ungültiger Endstatus für einen Turn: {status!r}")
        with closing(self._connect()) as conn:
            cur = conn.execute(
                "UPDATE chats SET last_turn_status = ? WHERE id = ? AND last_turn_id = ?",
                (status, chat_id, turn_id),
            )
            return cur.rowcount > 0

    def get_last_turn(self, chat_id: str) -> dict[str, Any] | None:
        """Der zuletzt gestartete Turn: ``{"turn_id", "user_message_id", "status"}``.

        ``None``, wenn es den Chat nicht gibt oder noch kein Turn vermerkt ist
        (z. B. Chats aus Schema-Version 1).
        """
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT last_turn_id, last_turn_user_id, last_turn_status FROM chats WHERE id = ?", (chat_id,)
            ).fetchone()
        if row is None or row["last_turn_id"] is None:
            return None
        return {
            "turn_id": row["last_turn_id"],
            "user_message_id": row["last_turn_user_id"],
            "status": row["last_turn_status"],
        }

    # -- Nachrichten --

    def get_messages(self, chat_id: str) -> list[Message]:
        """Alle Nachrichten des Chats in gespeicherter Reihenfolge (leer, wenn es ihn nicht gibt)."""
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT data FROM messages WHERE chat_id = ? ORDER BY seq", (chat_id,)
            ).fetchall()
        return [Message.from_dict(json.loads(r["data"])) for r in rows]

    def append_messages(self, chat_id: str, msgs: Message | Iterable[Message]) -> list[int]:
        """Hängt Nachrichten in der übergebenen Reihenfolge an und gibt ihre ``seq`` zurück.

        Die seq-Vergabe geschieht atomar in einer schreibenden Transaktion
        (``BEGIN IMMEDIATE``); bei Sperre oder seq-Konflikt wird wiederholt.
        Setzt ``updated_at`` des Chats. Wirft ``KeyError``, wenn es den Chat nicht gibt.
        """
        items = [msgs] if isinstance(msgs, Message) else list(msgs)
        if not items:
            return []
        payload = [(m.id, m.role, json.dumps(m.to_dict(), ensure_ascii=False), m.created_at) for m in items]

        delay = 0.01
        for attempt in range(self.APPEND_RETRIES):
            try:
                return self._append_once(chat_id, payload)
            except sqlite3.IntegrityError as exc:
                # Nur seq-Kollisionen wiederholen; doppelte Nachrichten-IDs sind ein echter Fehler.
                if "seq" not in str(exc) or attempt == self.APPEND_RETRIES - 1:
                    raise
            except sqlite3.OperationalError as exc:
                text = str(exc).lower()
                if ("locked" not in text and "busy" not in text) or attempt == self.APPEND_RETRIES - 1:
                    raise
            time.sleep(delay)
            delay = min(delay * 2, 0.5)
        raise RuntimeError("unerreichbar")  # pragma: no cover

    def _append_once(self, chat_id: str, payload: list[tuple[str, str, str, str]]) -> list[int]:
        with closing(self._connect()) as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                if conn.execute("SELECT 1 FROM chats WHERE id = ?", (chat_id,)).fetchone() is None:
                    raise KeyError(chat_id)
                start = conn.execute(
                    "SELECT COALESCE(MAX(seq), 0) FROM messages WHERE chat_id = ?", (chat_id,)
                ).fetchone()[0]
                seqs = list(range(start + 1, start + 1 + len(payload)))
                conn.executemany(
                    "INSERT INTO messages (id, chat_id, seq, role, data, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                    [(mid, chat_id, seq, role, data, created) for seq, (mid, role, data, created) in zip(seqs, payload)],
                )
                conn.execute("UPDATE chats SET updated_at = ? WHERE id = ?", (iso_now(), chat_id))
                conn.execute("COMMIT")
            except BaseException:
                conn.execute("ROLLBACK")
                raise
        return seqs

    def delete_messages_after(self, chat_id: str, message_id: str) -> int:
        """Löscht alle Nachrichten des Chats, die hinter ``message_id`` gespeichert sind.

        ``message_id`` selbst bleibt erhalten. Läuft in einer schreibenden
        Transaktion; ``updated_at`` des Chats wird gesetzt, wenn etwas gelöscht
        wurde. Gibt die Zahl der gelöschten Nachrichten zurück. Wirft
        ``KeyError``, wenn es den Chat oder die Nachricht in diesem Chat nicht gibt.
        """
        with closing(self._connect()) as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                if conn.execute("SELECT 1 FROM chats WHERE id = ?", (chat_id,)).fetchone() is None:
                    raise KeyError(chat_id)
                row = conn.execute(
                    "SELECT seq FROM messages WHERE chat_id = ? AND id = ?", (chat_id, message_id)
                ).fetchone()
                if row is None:
                    raise KeyError(message_id)
                removed = conn.execute(
                    "DELETE FROM messages WHERE chat_id = ? AND seq > ?", (chat_id, row["seq"])
                ).rowcount
                if removed:
                    conn.execute("UPDATE chats SET updated_at = ? WHERE id = ?", (iso_now(), chat_id))
                conn.execute("COMMIT")
            except BaseException:
                conn.execute("ROLLBACK")
                raise
        return removed


# --- Konfiguration (JSON) ----------------------------------------------------

_DEFAULT_CONFIG: dict[str, Any] = {
    "version": 1,
    "settings": {
        "theme": "system",
        "system_prompt": "",
        "default_provider": "lmstudio",
        "default_model": None,
    },
    "providers": [
        {"id": "lmstudio", "type": "lmstudio", "name": "LM Studio", "enabled": True},
        {
            "id": "claude",
            "type": "claude_agent",
            "name": "Claude (Agent SDK)",
            "enabled": True,
            "auth_mode": "login",
            "api_key": None,
            "cli_path": None,
            "default_model": "default",
        },
    ],
    "mcp_servers": [],
    "tools": {"disabled": []},
}


def default_config() -> dict[str, Any]:
    """Eine frische Kopie der Standard-Konfiguration."""
    return copy.deepcopy(_DEFAULT_CONFIG)


def configured_keys(config: dict[str, Any]) -> list[str]:
    """Alle in ``providers`` konfigurierten API-Keys ab ``MIN_SECRET_LEN`` Zeichen, längste zuerst.

    Kürzere Keys (Platzhalter wie ``"ollama"``) werden bewusst ausgelassen,
    damit sie nie als Schwärz-Muster dienen.
    """
    keys = {
        p["api_key"]
        for p in config.get("providers") or []
        if isinstance(p, dict) and isinstance(p.get("api_key"), str) and len(p["api_key"]) >= MIN_SECRET_LEN
    }
    return sorted(keys, key=len, reverse=True)


def redact(text: str, config: dict[str, Any]) -> str:
    """Ersetzt alle konfigurierten API-Keys (ab ``MIN_SECRET_LEN`` Zeichen) in ``text`` durch ``***``."""
    for key in configured_keys(config):
        text = text.replace(key, "***")
    return text


class ConfigStore:
    """Liest und schreibt die Konfigurationsdatei (z. B. ``data/config.json``).

    ``load()`` legt die Datei mit der Standard-Konfiguration an, falls sie fehlt.
    Der geladene Inhalt liegt in ``data``; Änderungen daran werden mit ``save()``
    geschrieben. Zugriffe sind über ein internes Lock serialisiert.
    """

    REPLACE_RETRIES = 10

    def __init__(self, path: str | os.PathLike[str]):
        self.path = Path(path)
        self.data: dict[str, Any] = default_config()
        self._lock = threading.RLock()

    def load(self) -> dict[str, Any]:
        """Lädt die Datei (oder legt sie mit Standardinhalt an) und gibt ``data`` zurück.

        Fehlende Abschnitte werden mit Standardwerten ergänzt. Bei ungültigem
        JSON wird ``ValueError`` geworfen; die Datei bleibt dann unverändert.
        """
        with self._lock:
            if not self.path.exists():
                self.data = default_config()
                self.save()
                return self.data
            try:
                loaded = json.loads(self.path.read_text(encoding="utf-8-sig"))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Konfigurationsdatei {self.path} ist kein gültiges JSON: {exc}") from exc
            if not isinstance(loaded, dict):
                raise ValueError(f"Konfigurationsdatei {self.path} muss ein JSON-Objekt enthalten.")
            defaults = default_config()
            for key, value in defaults.items():
                loaded.setdefault(key, value)
            if isinstance(loaded.get("settings"), dict):
                for key, value in defaults["settings"].items():
                    loaded["settings"].setdefault(key, value)
            if isinstance(loaded.get("tools"), dict):
                loaded["tools"].setdefault("disabled", [])
            self.data = loaded
            return self.data

    def save(self, data: dict[str, Any] | None = None) -> None:
        """Schreibt ``data`` (oder den übergebenen Inhalt) atomar in die Datei.

        Erst in eine temporäre Datei im selben Verzeichnis, dann ``os.replace``.
        Unter Windows kann das Ersetzen kurz mit ``PermissionError`` scheitern
        (Virenscanner, Leser); dann wird mehrfach wiederholt.
        """
        with self._lock:
            if data is not None:
                self.data = data
            self.path.parent.mkdir(parents=True, exist_ok=True)
            text = json.dumps(self.data, ensure_ascii=False, indent=2) + "\n"
            tmp = self.path.with_name(f"{self.path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
            try:
                with open(tmp, "w", encoding="utf-8", newline="\n") as f:
                    f.write(text)
                    f.flush()
                    os.fsync(f.fileno())
                delay = 0.05
                for attempt in range(self.REPLACE_RETRIES):
                    try:
                        os.replace(tmp, self.path)
                        break
                    except PermissionError:
                        if attempt == self.REPLACE_RETRIES - 1:
                            raise
                        time.sleep(delay)
                        delay = min(delay * 2, 1.0)
            finally:
                if tmp.exists():
                    try:
                        tmp.unlink()
                    except OSError:
                        pass

    def public(self) -> dict[str, Any]:
        """Kopie von ``data`` ohne Geheimnisse: ``api_key`` wird zu ``has_api_key`` (bool)."""
        with self._lock:
            result = copy.deepcopy(self.data)
        for provider in result.get("providers") or []:
            if isinstance(provider, dict):
                provider["has_api_key"] = bool(provider.pop("api_key", None))
        return result

    def redact(self, text: str) -> str:
        """Ersetzt die konfigurierten API-Keys (ab ``MIN_SECRET_LEN`` Zeichen) in ``text`` durch ``***``."""
        with self._lock:
            return redact(text, self.data)


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmp_dir:
        data_dir = Path(tmp_dir) / "data"

        store = ChatStore(data_dir / "noassist.db")
        chat = store.create_chat("Demo äöü", "lmstudio", "qwen")
        store.append_messages(chat["id"], [
            Message(role="user", content="Hallo"),
            Message(role="assistant", content="Hallo! Wie kann ich helfen?"),
        ])
        store.update_chat(chat["id"], provider_state={"session_id": "abc"})
        print("Chats:", [(c["id"], c["title"], c["provider_state"]) for c in store.list_chats()])
        for m in store.get_messages(chat["id"]):
            print(f"  {m.role:9} {m.content}")
        first = store.get_messages(chat["id"])[0]
        store.mark_turn_started(chat["id"], "t_demo", first.id)
        store.mark_turn_finished(chat["id"], "t_demo", "error")
        print("Letzter Turn:", store.get_last_turn(chat["id"]))
        print("Hinter der ersten Nachricht gelöscht:", store.delete_messages_after(chat["id"], first.id))
        print("Gelöscht:", store.delete_chat(chat["id"]), "-> Nachrichten:", store.get_messages(chat["id"]))

        config = ConfigStore(data_dir / "config.json")
        config.load()
        config.data["providers"].append({
            "id": "oai-abc123", "type": "openai", "name": "Test", "enabled": True,
            "base_url": "https://api.example.com/v1", "api_key": "sk-geheim", "api_key_env": None,
            "default_model": None, "models": [], "supports_tools": True,
        })
        config.save()
        print(json.dumps(config.public(), ensure_ascii=False, indent=2))
        print(config.redact("Fehler 401 mit Key sk-geheim"))
