"""Gemeinsame Tool-Schicht für alle Provider.

Ein ``ToolSpec`` beschreibt ein Tool provider-neutral: Name, Beschreibung,
JSON-Schema der Parameter, Parameterangaben für LM Studio und die
Implementierung. Ein ``ToolSet`` verwaltet mehrere Tools, führt sie aus
(mit ``tool_call``-/``tool_result``-Events) und erzeugt daraus die Formate der
einzelnen Provider:

- ``for_openai()``   -> Liste für ``chat.completions.create(tools=...)``
- ``for_lmstudio()`` -> Liste von ``lms.ToolFunctionDef`` für ``LLM.act()``
- ``for_claude()``   -> In-Process-MCP-Server für die Claude Agent SDK

``lmstudio``, ``pydantic`` und ``claude_agent_sdk`` werden erst bei Bedarf
importiert.
"""

from __future__ import annotations

import asyncio
import copy
import functools
import inspect
import json
import logging
import re
import sys
import threading
import time
import uuid
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Optional, get_type_hints

from chat_types import Emit, Event, ev_tool_call, ev_tool_result

if TYPE_CHECKING:
    import lmstudio as lms
    from claude_agent_sdk import McpSdkServerConfig

log = logging.getLogger(__name__)

MAX_NAME_LEN = 48
"""Maximale Länge eines Tool-Namens (``mcp__noassist__`` + 48 Zeichen <= 64)."""
NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,48}$")
"""Erlaubte Tool-Namen (gemeinsamer Nenner von OpenAI, Claude und LM Studio)."""
MAX_EVENT_CHARS = 4000
"""So viele Zeichen des Ergebnisses stehen höchstens im ``tool_result``-Event."""
MAX_RESULT_CHARS = 50_000
"""So viele Zeichen des Ergebnisses gehen höchstens ans Modell."""
ERROR_PREFIX = "Fehler:"
"""Ergebnisse mit diesem Präfix gelten als Fehler (``is_error``)."""

_MISSING = object()
_VAR_KINDS = (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)


# --- Datentypen ----------------------------------------------------------------

@dataclass(frozen=True)
class ToolSpec:
    """Provider-neutrale Beschreibung eines Tools.

    - ``parameters``: JSON-Schema mit ``type: object`` (für OpenAI und Claude).
    - ``lms_parameters``: Parameterangaben im Format von ``lms.ToolFunctionDef``
      (Name -> Typ oder ``{"type": Typ, "default": Wert}``).
    - ``func``: wird mit den Argumenten als Schlüsselwortargumenten aufgerufen.
    - ``source``: ``"local"`` oder ``"mcp:<server>"``.
    """

    name: str
    description: str
    parameters: dict[str, Any] = field(hash=False)
    lms_parameters: Mapping[str, Any] = field(hash=False)
    func: Callable[..., Any]
    source: str = "local"


@dataclass
class ToolOutcome:
    """Ergebnis einer Tool-Ausführung, so wie es ans Modell geht."""

    call_id: str
    text: str  # höchstens MAX_RESULT_CHARS Zeichen
    is_error: bool
    duration_ms: int


ApprovalHook = Callable[[ToolSpec, dict[str, Any], str], bool | str | None]
"""Freigabe-Hook für ``ToolSet.execute``: ``hook(spec, arguments, call_id)``.

Rückgabe ``None`` oder ``True``: Aufruf wird ausgeführt.
``False``: abgelehnt. Ein String: abgelehnt mit diesem Grund.
Eine Exception im Hook zählt ebenfalls als Ablehnung.
"""


# --- Namen -----------------------------------------------------------------------

def clean_tool_name(name: str) -> str:
    """Macht aus ``name`` einen gültigen Tool-Namen (``^[A-Za-z0-9_-]{1,48}$``).

    Gültige Namen bleiben unverändert. Sonst werden ungültige Zeichen durch
    ``_`` ersetzt und der Name auf 48 Zeichen gekürzt.
    """
    name = str(name)
    if NAME_RE.fullmatch(name):
        return name
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_")
    return cleaned[:MAX_NAME_LEN] or "tool"


# --- Schema-Erzeugung ------------------------------------------------------------

def _is_json(value: Any) -> bool:
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return False
    return True


def _type_schema(hint: Any, defs: dict[str, Any]) -> dict[str, Any]:
    """JSON-Schema für einen Typ-Hinweis; ``$defs`` werden in ``defs`` gesammelt."""
    from pydantic import TypeAdapter

    try:
        schema = TypeAdapter(hint).json_schema(ref_template="#/$defs/{model}")
    except Exception as e:
        log.warning("Kein JSON-Schema für Typ %r (%s); Parameter ohne Typangabe.", hint, e)
        return {}
    sub_defs = schema.pop("$defs", None)
    if sub_defs:
        defs.update(sub_defs)
    return schema


def _object_schema(
    properties: dict[str, Any], required: list[str], defs: dict[str, Any], extra_allowed: bool
) -> dict[str, Any]:
    schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    if not extra_allowed:
        schema["additionalProperties"] = False
    if defs:
        schema["$defs"] = defs
    return schema


def _type_hints(func: Callable[..., Any]) -> dict[str, Any]:
    """Typ-Hinweise der Parameter; leer, wenn sie nicht auflösbar sind."""
    target: Any = func
    while isinstance(target, functools.partial):
        target = target.func
    if inspect.isclass(target):
        target = target.__init__
    elif callable(target) and not (
        inspect.isfunction(target) or inspect.ismethod(target) or inspect.isbuiltin(target)
    ):
        target = type(target).__call__  # aufrufbare Objekte
    try:
        return get_type_hints(target, include_extras=True)
    except Exception:
        return {}


def _signature_schema(sig: inspect.Signature, hints: Mapping[str, Any]) -> dict[str, Any]:
    properties: dict[str, Any] = {}
    required: list[str] = []
    defs: dict[str, Any] = {}
    extra_allowed = False
    for pname, param in sig.parameters.items():
        if param.kind is inspect.Parameter.VAR_KEYWORD:
            extra_allowed = True
            continue
        if param.kind is inspect.Parameter.VAR_POSITIONAL:
            continue
        prop = _type_schema(hints.get(pname, Any), defs)
        if param.default is inspect.Parameter.empty:
            required.append(pname)
        elif "default" not in prop and _is_json(param.default):
            prop["default"] = param.default
        properties[pname] = prop
    return _object_schema(properties, required, defs, extra_allowed)


def _lms_param_ok(pname: str, value: Any) -> bool:
    """Prüft, ob LM Studio (msgspec) aus der Parameterangabe ein Schema bilden kann.

    Nutzt dieselben öffentlichen msgspec-Schritte wie lmstudio intern
    (``defstruct`` + ``json.schema``); dabei fallen unbekannte Typen (z. B.
    pydantic-Modelle, ``Path``) und unkodierbare Standardwerte (z. B.
    ``Path(".")``, nicht leere Listen) auf.
    """
    import msgspec  # Abhängigkeit von lmstudio

    if isinstance(value, Mapping):
        hint = value.get("type", Any)
        field_def = (pname, hint, value["default"]) if "default" in value else (pname, hint)
    else:
        field_def = (pname, value)
    try:
        msgspec.json.schema(msgspec.defstruct("ParamCheck", [field_def], kw_only=True))
    except Exception:
        return False
    return True


def _lms_compatible(pname: str, value: Any) -> Any:
    """Erste von LM Studio verwendbare Variante der Parameterangabe.

    Reihenfolge: unverändert, Typ ``Any``, Standardwert ``None`` (die
    Funktion behält dann ihren eigenen Standardwert, siehe
    ``_substituted_defaults``).
    """
    if _lms_param_ok(pname, value):
        return value
    if isinstance(value, Mapping) and "default" in value:
        hint, default = value.get("type", Any), value["default"]
        candidates: list[Any] = [
            {"type": hint, "default": default},
            {"type": Any, "default": default},
            # Optional[...] statt "hint | None": funktioniert für jede Typangabe (auch Strings).
            {"type": Optional[hint], "default": None},  # noqa: UP045
            {"type": Any, "default": None},
        ]
    else:
        hint = value.get("type", Any) if isinstance(value, Mapping) else value
        candidates = [hint, Any]
    for candidate in candidates:
        if _lms_param_ok(pname, candidate):
            return candidate
    return candidates[-1]


def _lms_parameters(
    func: Callable[..., Any],
    name: str,
    description: str,
    sig: inspect.Signature,
    hints: Mapping[str, Any],
) -> dict[str, Any]:
    """Parameterangaben für ``lms.ToolFunctionDef``.

    Grundlage ist ``lms.ToolFunctionDef.from_callable(func).parameters``. Wenn
    das scheitert oder nicht alle Parameter erfasst (z. B. Parameter ohne
    Typangabe), wird dasselbe Format selbst aus Signatur und Typen gebaut.
    Angaben, die LM Studio nicht verarbeiten kann, werden ersetzt (siehe
    ``_lms_compatible``), damit ein einzelnes Tool nicht den ganzen Turn
    scheitern lässt.
    """
    import lmstudio as lms

    expected = [n for n, p in sig.parameters.items() if p.kind not in _VAR_KINDS]
    try:
        params: dict[str, Any] | None = dict(
            lms.ToolFunctionDef.from_callable(func, name=name, description=description).parameters
        )
    except Exception:
        params = None
    if params is None or set(params) != set(expected):
        params = {}
        for pname in expected:
            param = sig.parameters[pname]
            hint = hints.get(pname, Any)
            if param.default is inspect.Parameter.empty:
                params[pname] = hint
            else:
                params[pname] = {"type": hint, "default": param.default}

    for pname, value in list(params.items()):
        compatible = _lms_compatible(pname, value)
        if compatible is not value:
            log.warning("Tool %s: Parameter %r für LM Studio angepasst: %r", name, pname, compatible)
            params[pname] = compatible
    return params


def _substituted_defaults(spec: ToolSpec) -> set[str]:
    """Parameter, deren Standardwert für LM Studio durch ``None`` ersetzt wurde.

    Für diese Parameter wird ``None`` beim Aufruf weggelassen, damit der
    eigene Standardwert der Funktion greift.
    """
    try:
        sig = inspect.signature(spec.func)
    except (TypeError, ValueError):
        return set()
    names: set[str] = set()
    for pname, value in spec.lms_parameters.items():
        if not (isinstance(value, Mapping) and "default" in value and value["default"] is None):
            continue
        param = sig.parameters.get(pname)
        if param is not None and param.default is not inspect.Parameter.empty and param.default is not None:
            names.add(pname)
    return names


def normalize_schema(schema: Any) -> dict[str, Any]:
    """Kopiert ein JSON-Schema und stellt ``type`` (String) und ``properties`` sicher.

    Die Claude Agent SDK erkennt ein Dict nur dann als JSON-Schema, wenn beide
    Schlüssel vorhanden sind.
    """
    if not isinstance(schema, Mapping):
        return {"type": "object", "properties": {}}
    result = copy.deepcopy(dict(schema))
    if not isinstance(result.get("type"), str):
        result["type"] = "object"
    if not isinstance(result.get("properties"), Mapping):
        result["properties"] = {}
    return result


def spec_from_callable(
    func: Callable[..., Any],
    *,
    name: str | None = None,
    description: str | None = None,
    source: str = "local",
) -> ToolSpec:
    """Baut einen ``ToolSpec`` aus einer Python-Funktion.

    - Name: ``func.__name__`` (oder ``name``).
    - Beschreibung: bereinigter Docstring (oder ``description``).
    - Schema: aus ``inspect.signature``, ``get_type_hints`` und
      ``pydantic.TypeAdapter`` (je Parameter); Parameter ohne Standardwert sind
      Pflicht, ``Annotated[..., Field(description=...)]`` wird übernommen.
    - ``lms_parameters``: siehe ``_lms_parameters``.
    """
    if not callable(func):
        raise TypeError(f"Tool muss aufrufbar sein: {func!r}")
    tool_name: str = name if name is not None else (getattr(func, "__name__", None) or type(func).__name__)
    tool_desc: str = description if description is not None else (inspect.getdoc(func) or "")
    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError) as e:
        raise TypeError(f"Signatur von {func!r} nicht ermittelbar: {e}") from e
    hints = _type_hints(func)
    return ToolSpec(
        name=tool_name,
        description=tool_desc,
        parameters=_signature_schema(sig, hints),
        lms_parameters=_lms_parameters(func, tool_name, tool_desc, sig, hints),
        func=func,
        source=source,
    )


def _schema_from_lms_parameters(params: Mapping[str, Any]) -> dict[str, Any]:
    properties: dict[str, Any] = {}
    required: list[str] = []
    defs: dict[str, Any] = {}
    for pname, value in params.items():
        if isinstance(value, Mapping):
            hint, default = value.get("type", Any), value.get("default", _MISSING)
        else:
            hint, default = value, _MISSING
        prop = _type_schema(hint, defs)
        if default is _MISSING:
            required.append(pname)
        elif "default" not in prop and _is_json(default):
            prop["default"] = default
        properties[pname] = prop
    return _object_schema(properties, required, defs, extra_allowed=False)


def spec_from_lms_def(
    d: lms.ToolFunctionDef, source: str = "local", schema: dict[str, Any] | None = None
) -> ToolSpec:
    """Baut einen ``ToolSpec`` aus einer ``lms.ToolFunctionDef``.

    ``schema``: vollständiges JSON-Schema (z. B. das Original eines MCP-Tools).
    Ohne ``schema`` wird es aus ``d.parameters`` erzeugt.
    """
    params = dict(d.parameters)
    description = d.description or ""
    if description and description == getattr(d.implementation, "__doc__", None):
        description = inspect.cleandoc(description)  # roher Docstring aus from_callable
    parameters = normalize_schema(schema) if schema is not None else _schema_from_lms_parameters(params)
    return ToolSpec(
        name=d.name,
        description=description,
        parameters=parameters,
        lms_parameters=params,
        func=d.implementation,
        source=source,
    )


def _lms_def_type() -> type | None:
    """``lms.ToolFunctionDef``, falls lmstudio schon geladen ist (sonst kann es keine geben)."""
    lms_module = sys.modules.get("lmstudio")
    return getattr(lms_module, "ToolFunctionDef", None) if lms_module else None


def as_spec(item: Any) -> ToolSpec:
    """Wandelt Funktion, ``lms.ToolFunctionDef`` (auch als Dict) oder ``ToolSpec`` in einen ``ToolSpec``."""
    if isinstance(item, ToolSpec):
        return item
    def_type = _lms_def_type()
    if def_type is not None and isinstance(item, def_type):
        return spec_from_lms_def(item)
    if isinstance(item, Mapping) and "implementation" in item:
        import lmstudio as lms

        return spec_from_lms_def(lms.ToolFunctionDef(**item))
    if callable(item):
        return spec_from_callable(item)
    raise TypeError(f"Kein gültiges Tool: {item!r}")


# --- Ausführung ------------------------------------------------------------------

def _new_call_id() -> str:
    return "call_" + uuid.uuid4().hex[:24]


def _emit_safe(emit: Emit | None, event: Event) -> None:
    if emit is None:
        return
    try:
        emit(event)
    except Exception:
        log.exception("Event %r konnte nicht gesendet werden", event.get("type"))


def _json_safe(value: dict[str, Any]) -> dict[str, Any]:
    """JSON-taugliche Kopie der Argumente für das Event."""
    try:
        return json.loads(json.dumps(value, ensure_ascii=False, default=str))
    except (TypeError, ValueError):
        return {str(k): str(v) for k, v in value.items()}


def _parse_arguments(arguments: Any) -> tuple[dict[str, Any], str | None]:
    """Argumente als Dict; akzeptiert auch einen JSON-String. Liefert (args, Fehlertext)."""
    if arguments is None:
        return {}, None
    if isinstance(arguments, Mapping):
        return dict(arguments), None
    if isinstance(arguments, (str, bytes, bytearray)):
        raw = arguments.decode("utf-8", "replace") if not isinstance(arguments, str) else arguments
        if not raw.strip():
            return {}, None
        try:
            parsed = json.loads(raw)
        except ValueError as e:
            return {}, f"{ERROR_PREFIX} Ungültige Argumente (kein JSON): {e}"
        if isinstance(parsed, dict):
            return parsed, None
        return {}, f"{ERROR_PREFIX} Ungültige Argumente: erwartet ein JSON-Objekt, erhalten {type(parsed).__name__}"
    return {}, f"{ERROR_PREFIX} Ungültige Argumente: erwartet ein JSON-Objekt, erhalten {type(arguments).__name__}"


def _error_text(exc: BaseException) -> str:
    msg = str(exc).strip()
    return f"{ERROR_PREFIX} {type(exc).__name__}: {msg}" if msg else f"{ERROR_PREFIX} {type(exc).__name__}"


def _to_text(result: Any) -> str:
    if isinstance(result, str):
        return result
    try:
        return json.dumps(result, ensure_ascii=False, default=str)
    except (TypeError, ValueError):  # z. B. zyklische Strukturen
        return str(result)


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    marker = f"\n\n[... gekürzt: das Ergebnis hatte {len(text)} Zeichen, übergeben werden höchstens {limit}]"
    return text[: max(0, limit - len(marker))] + marker


async def _await(awaitable: Any) -> Any:
    return await awaitable


def _run_awaitable(awaitable: Any) -> Any:
    """Führt das Ergebnis eines async-Tools in einer eigenen Event-Loop aus."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_await(awaitable))
    if inspect.iscoroutine(awaitable):
        awaitable.close()
    raise RuntimeError("Asynchrones Tool kann nicht aus einer laufenden Event-Loop heraus ausgeführt werden")


def _same_func(a: Any, b: Any) -> bool:
    if a is b:
        return True
    if inspect.ismethod(a) and inspect.ismethod(b):
        return a == b
    return False


# --- ToolSet ---------------------------------------------------------------------

class ToolSet:
    """Geordnete Menge von Tools mit eindeutigen, gültigen Namen.

    Threadsicher: Hinzufügen, Entfernen und Ausführen dürfen aus verschiedenen
    Threads passieren. ``for_*`` liefern Momentaufnahmen.

    ``approval_hook`` (standardmäßig ``None`` = inaktiv) ist die
    Erweiterungsstelle für eine Freigabe pro Aufruf, siehe ``ApprovalHook``.
    """

    def __init__(self, specs: Iterable[Any] = ()):
        self._lock = threading.RLock()
        self._specs: dict[str, ToolSpec] = {}
        self.approval_hook: ApprovalHook | None = None
        for spec in specs:
            self.add(spec)

    @classmethod
    def from_any(cls, items: Iterable[Any] | None) -> ToolSet:
        """ToolSet aus Funktionen, ``lms.ToolFunctionDef``s, ``ToolSpec``s oder einem ToolSet.

        Bei einem ToolSet wird eine Kopie erzeugt (inklusive ``approval_hook``).
        """
        if items is None:
            return cls()
        result = cls(items)
        if isinstance(items, ToolSet):
            result.approval_hook = items.approval_hook
        return result

    # -- Verwaltung --

    def add(self, tool: Any) -> ToolSpec:
        """Fügt ein Tool hinzu und gibt den tatsächlich gespeicherten ``ToolSpec`` zurück.

        Ungültige Zeichen im Namen werden ersetzt; bei Namenskonflikten wird
        ``_2``, ``_3`` … angehängt. Ist dieselbe Funktion schon enthalten,
        passiert nichts und der vorhandene ``ToolSpec`` wird zurückgegeben.
        """
        spec = as_spec(tool)
        with self._lock:
            for existing in self._specs.values():
                if _same_func(existing.func, spec.func):
                    return existing
            name = self._unique_name(clean_tool_name(spec.name))
            if name != spec.name:
                log.info("Tool %r wird als %r geführt", spec.name, name)
                spec = replace(spec, name=name)
            self._specs[name] = spec
            return spec

    def _unique_name(self, name: str) -> str:
        if name not in self._specs:
            return name
        i = 2
        while True:
            suffix = f"_{i}"
            candidate = name[: MAX_NAME_LEN - len(suffix)] + suffix
            if candidate not in self._specs:
                return candidate
            i += 1

    def remove(self, name_or_func: Any) -> None:
        """Entfernt ein Tool per Name, Funktion, ``ToolSpec`` oder ``lms.ToolFunctionDef``.

        Unbekannte Tools werden ignoriert.
        """
        with self._lock:
            if isinstance(name_or_func, str):
                self._specs.pop(name_or_func, None)
                return
            if isinstance(name_or_func, ToolSpec):
                func = name_or_func.func
            else:
                def_type = _lms_def_type()
                if def_type is not None and isinstance(name_or_func, def_type):
                    func = name_or_func.implementation
                else:
                    func = name_or_func
            for name in [n for n, s in self._specs.items() if _same_func(s.func, func)]:
                del self._specs[name]

    def merged(self, other: Iterable[Any]) -> ToolSet:
        """Neues ToolSet: erst die eigenen Tools, dann die aus ``other`` (Konflikte werden umbenannt).

        Der ``approval_hook`` dieses ToolSets wird übernommen.
        """
        result = ToolSet(self)
        result.approval_hook = self.approval_hook
        for item in other:
            result.add(item)
        return result

    def get(self, name: str) -> ToolSpec | None:
        with self._lock:
            return self._specs.get(name)

    def names(self) -> list[str]:
        with self._lock:
            return list(self._specs)

    def describe(self) -> list[dict[str, Any]]:
        """JSON-taugliche Übersicht (``name``, ``description``, ``source``, ``parameters``)."""
        return [
            {
                "name": s.name,
                "description": s.description,
                "source": s.source,
                "parameters": copy.deepcopy(s.parameters),
            }
            for s in self
        ]

    def __iter__(self) -> Iterator[ToolSpec]:
        with self._lock:
            return iter(list(self._specs.values()))

    def __len__(self) -> int:
        with self._lock:
            return len(self._specs)

    def __contains__(self, name: object) -> bool:
        with self._lock:
            return name in self._specs

    def __repr__(self) -> str:
        return f"<ToolSet {self.names()}>"

    # -- Ausführung --

    def execute(
        self, name: str, arguments: Any, emit: Emit | None, call_id: str | None = None
    ) -> ToolOutcome:
        """Führt ein Tool aus. Wirft nie; Fehler stehen als ``"Fehler: ..."`` im Ergebnis.

        Ablauf:

        1. ``tool_call``-Event (mit ``call_id``, neu erzeugt falls ``None``).
        2. Freigabe über ``approval_hook`` (falls gesetzt).
        3. Aufruf ``func(**arguments)``; ``arguments`` darf auch ein JSON-String sein.
           Exceptions werden zu ``"Fehler: <Typ>: <Meldung>"``, Nicht-Strings zu JSON.
        4. ``tool_result``-Event (Ergebnis auf 4000 Zeichen gekürzt, ``truncated``).

        Das zurückgegebene ``ToolOutcome.text`` hat höchstens 50 000 Zeichen.
        """
        return self._execute(self.get(name), name, arguments, emit, call_id)

    def _execute(
        self,
        spec: ToolSpec | None,
        name: str,
        arguments: Any,
        emit: Emit | None,
        call_id: str | None,
    ) -> ToolOutcome:
        call_id = call_id or _new_call_id()
        args, arg_error = _parse_arguments(arguments)
        source = spec.source if spec else "unknown"
        _emit_safe(emit, ev_tool_call(call_id, name, _json_safe(args), source))

        duration_ms = 0
        if spec is None:
            text = f"{ERROR_PREFIX} Unbekanntes Tool: {name}"
        elif arg_error:
            text = arg_error
        else:
            denial = self._check_approval(spec, args, call_id)
            if denial is not None:
                text = denial
            else:
                start = time.perf_counter()
                text = self._invoke(spec, args)
                duration_ms = int((time.perf_counter() - start) * 1000)

        is_error = text.startswith(ERROR_PREFIX)
        text = _truncate(text, MAX_RESULT_CHARS)
        _emit_safe(
            emit,
            ev_tool_result(
                call_id,
                name,
                text[:MAX_EVENT_CHARS],
                is_error=is_error,
                duration_ms=duration_ms,
                truncated=len(text) > MAX_EVENT_CHARS,
            ),
        )
        return ToolOutcome(call_id=call_id, text=text, is_error=is_error, duration_ms=duration_ms)

    def _check_approval(self, spec: ToolSpec, args: dict[str, Any], call_id: str) -> str | None:
        """``None`` = freigegeben, sonst der Fehlertext für das Modell."""
        hook = self.approval_hook
        if hook is None:
            return None
        try:
            decision = hook(spec, dict(args), call_id)
        except Exception as e:
            log.warning("Freigabe-Hook für %s fehlgeschlagen", spec.name, exc_info=True)
            return f"{ERROR_PREFIX} Freigabe fehlgeschlagen: {type(e).__name__}: {e}"
        if decision is None or decision is True:
            return None
        if isinstance(decision, str) and decision.strip():
            return f"{ERROR_PREFIX} Aufruf nicht freigegeben: {decision.strip()}"
        return f"{ERROR_PREFIX} Aufruf nicht freigegeben."

    @staticmethod
    def _invoke(spec: ToolSpec, args: dict[str, Any]) -> str:
        try:
            result = spec.func(**args)
            if inspect.isawaitable(result):
                result = _run_awaitable(result)
        except Exception as e:
            log.debug("Tool %s fehlgeschlagen", spec.name, exc_info=True)
            return _error_text(e)
        return _to_text(result)

    # -- Provider-Formate --

    def for_openai(self) -> list[dict[str, Any]]:
        """Tool-Liste im Format der OpenAI Chat Completions API (Kopien der Schemas)."""
        return [
            {
                "type": "function",
                "function": {
                    "name": s.name,
                    "description": s.description,
                    "parameters": copy.deepcopy(s.parameters),
                },
            }
            for s in self
        ]

    def for_lmstudio(self, emit: Emit | None) -> list[lms.ToolFunctionDef]:
        """Tools als ``lms.ToolFunctionDef``; jeder Aufruf läuft über ``execute``.

        Hinweis: LM Studio kodiert den Rückgabewert selbst per ``json.dumps``;
        im Tool-Ergebnis von ``act()`` steht der Text daher als JSON-String.
        """
        import lmstudio as lms

        return [
            lms.ToolFunctionDef(
                name=s.name,
                description=s.description,
                parameters=dict(s.lms_parameters),
                implementation=self._lms_implementation(s, emit),
            )
            for s in self
        ]

    def _lms_implementation(self, spec: ToolSpec, emit: Emit | None) -> Callable[..., str]:
        omit_none = _substituted_defaults(spec)

        def implementation(**kwargs: Any) -> str:
            if omit_none:
                kwargs = {k: v for k, v in kwargs.items() if not (k in omit_none and v is None)}
            return self._execute(spec, spec.name, kwargs, emit, None).text

        implementation.__name__ = spec.name
        implementation.__doc__ = spec.description
        return implementation

    def for_claude(
        self, emit: Emit | None, server_name: str = "noassist"
    ) -> tuple[McpSdkServerConfig, list[str]]:
        """In-Process-MCP-Server für die Claude Agent SDK.

        Liefert ``(Server-Konfiguration, allowed_tools)`` mit
        ``allowed_tools = ["mcp__<server_name>__<name>", ...]``. Die Handler
        führen ``execute`` per ``asyncio.to_thread`` aus, damit die Event-Loop
        nicht blockiert.
        """
        from claude_agent_sdk import SdkMcpTool, ToolAnnotations, create_sdk_mcp_server

        if not re.fullmatch(r"[A-Za-z0-9_-]{1,32}", server_name) or "__" in server_name:
            raise ValueError(f"Ungültiger Servername: {server_name!r}")
        sdk_tools = []
        allowed: list[str] = []
        for spec in self:
            sdk_tools.append(
                SdkMcpTool(
                    name=spec.name,
                    description=spec.description,
                    input_schema=normalize_schema(spec.parameters),
                    handler=self._claude_handler(spec, emit),
                    # Ergebnisse bis zu unserer Grenze inline lassen (nicht in eine Datei auslagern).
                    annotations=ToolAnnotations(maxResultSizeChars=MAX_RESULT_CHARS + 1000),
                )
            )
            allowed.append(f"mcp__{server_name}__{spec.name}")
        return create_sdk_mcp_server(server_name, tools=sdk_tools), allowed

    def _claude_handler(self, spec: ToolSpec, emit: Emit | None) -> Callable[[dict[str, Any]], Any]:
        async def handler(args: dict[str, Any]) -> dict[str, Any]:
            outcome = await asyncio.to_thread(self._execute, spec, spec.name, args, emit, None)
            return {
                "content": [{"type": "text", "text": outcome.text}],
                "is_error": outcome.is_error,
            }

        return handler


if __name__ == "__main__":
    from tools import get_tools

    def show(ev: Event) -> None:
        print("   event:", json.dumps(ev, ensure_ascii=False)[:160])

    tools = ToolSet.from_any(get_tools())
    print("Tools:", tools.names())
    for spec in tools:
        print(f"\n{spec.name} ({spec.source}): {spec.description.splitlines()[0]}")
        print(json.dumps(spec.parameters, ensure_ascii=False, indent=2))
    get_datetime, search_web = tools.get("get_datetime"), tools.get("search_web")
    assert get_datetime and "required" not in get_datetime.parameters
    assert search_web and search_web.parameters["required"] == ["query"]

    print("\n--- execute ---")
    print(tools.execute("add", {"a": 2, "b": 3.5}, show))
    print(tools.execute("read_file", '{"path": "gibt-es-nicht.txt"}', show))

    def beispiel_liste(anzahl: int = 2) -> list[dict]:
        """Gibt eine Beispiel-Liste zurück."""
        return [{"nr": i, "text": "äöü"} for i in range(anzahl)]

    def add(x: int) -> int:  # Namenskonflikt mit tools.add
        """Konkurrierendes add."""
        return x

    print("Hinzugefügt:", tools.add(beispiel_liste).name, tools.add(add).name)
    print(tools.execute("beispiel_liste", {}, show))

    print("\n--- for_openai (erstes Tool) ---")
    print(json.dumps(tools.for_openai()[0], ensure_ascii=False))

    print("\n--- for_lmstudio ---")
    defs = tools.for_lmstudio(show)
    for d in defs:
        print(f"   {d.name}: {dict(d.parameters)}")
    print("   Aufruf add_2(x=7) ->", defs[-1].implementation(x=7))

    print("\n--- for_claude ---")
    try:
        config, allowed = tools.for_claude(show)
    except ImportError as e:
        print("   claude_agent_sdk nicht installiert:", e)
    else:
        print("   Server:", config["name"], config["type"], "| allowed_tools:", allowed)
