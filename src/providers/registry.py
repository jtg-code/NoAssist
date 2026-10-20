"""Erzeugt Provider-Instanzen aus Einträgen der Konfiguration (``data/config.json``).

``build_provider(cfg, data_dir)`` wählt die Provider-Klasse anhand von
``cfg["type"]``. Die Provider-Module (und damit ``lmstudio``, ``openai`` bzw.
``claude_agent_sdk``) werden erst importiert, wenn ein Provider dieses Typs
gebraucht wird.

Unterstützte Typen:

- ``"lmstudio"``     -> ``LMStudioProvider`` (optional ``api_host``, z. B. ``"127.0.0.1:1234"``)
- ``"claude_agent"`` -> ``ClaudeAgentProvider.from_config(cfg, data_dir)``
- ``"openai"``       -> ``OpenAIProvider.from_config(cfg)``

Demo (aus ``src``): ``python -m providers.registry``
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from providers.base import Provider

PROVIDER_TYPES: tuple[str, ...] = ("lmstudio", "claude_agent", "openai")
"""Alle Provider-Typen, die ``build_provider`` kennt."""

TYPE_LABELS: dict[str, str] = {
    "lmstudio": "LM Studio",
    "claude_agent": "Claude (Agent SDK)",
    "openai": "OpenAI-kompatibel",
}


def build_provider(cfg: Mapping[str, Any], data_dir: str | Path | None = None) -> Provider:
    """Baut einen Provider aus einem Konfigurationseintrag.

    ``cfg`` ist ein Eintrag aus ``config["providers"]`` (mindestens ``id`` und
    ``type``). Unbekannte Felder (z. B. ``enabled``) werden ignoriert.
    ``data_dir`` ist das Datenverzeichnis (für den Arbeitsordner von Claude;
    ``None`` = Standard des Providers).

    Wirft ``ValueError`` bei fehlender ID, unbekanntem Typ oder ungültiger
    Konfiguration (z. B. fehlende ``base_url``). Die Meldung enthält keine
    Konfigurationswerte, also auch keine Keys.
    """
    if not isinstance(cfg, Mapping):
        raise ValueError("Provider-Konfiguration muss ein Objekt sein.")  # noqa: TRY004 - Aufrufer erwarten ValueError
    pid = cfg.get("id")
    if not isinstance(pid, str) or not pid.strip():
        raise ValueError("Provider-Konfiguration ohne gültige 'id'.")
    ptype = cfg.get("type")

    if ptype == "lmstudio":
        from providers.lmstudio_provider import LMStudioProvider

        api_host = cfg.get("api_host")
        if api_host is not None and not isinstance(api_host, str):
            raise ValueError(f"Provider {pid}: 'api_host' muss ein String sein.")
        return LMStudioProvider(
            id=pid,
            name=str(cfg.get("name") or TYPE_LABELS["lmstudio"]),
            api_host=(api_host or "").strip() or None,
        )

    if ptype == "claude_agent":
        from providers.claude_provider import ClaudeAgentProvider

        return ClaudeAgentProvider.from_config(cfg, Path(data_dir) if data_dir is not None else None)

    if ptype == "openai":
        from providers.openai_provider import OpenAIProvider

        try:
            return OpenAIProvider.from_config(dict(cfg))
        except (TypeError, ValueError) as e:
            # Meldungen des Konstruktors enthalten keine Werte (z. B. "base_url fehlt").
            raise ValueError(f"Provider {pid}: ungültige Konfiguration ({e}).") from e

    raise ValueError(f"Provider {pid}: unbekannter Typ {ptype!r} (erlaubt: {', '.join(PROVIDER_TYPES)}).")


if __name__ == "__main__":
    import sys

    from storage import default_config

    configs: list[dict[str, Any]] = list(default_config()["providers"])
    configs.append({
        "id": "oai-demo01", "type": "openai", "name": "Demo (lokal)", "enabled": True,
        "base_url": "http://localhost:1234/v1", "api_key": None, "api_key_env": None,
        "default_model": None, "models": ["demo-modell"], "supports_tools": True, "timeout": 300,
    })
    for cfg in configs:
        provider = build_provider(cfg, Path(__file__).resolve().parents[2] / "data")
        print(f"{cfg['id']:12} -> {provider!r}")
        if "--status" in sys.argv:
            st = provider.status()
            print(f"{'':12}    Status: ok={st.ok} {st.message} {st.hint or ''}")
        provider.close()

    for bad in ({"type": "openai", "id": "oai-x"}, {"id": "x", "type": "gibtsnicht"}, {"type": "lmstudio"}):
        try:
            build_provider(bad)
        except ValueError as e:
            print("ValueError:", e)
