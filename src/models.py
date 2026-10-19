import lmstudio as lms


def list_models() -> dict[str, list[str]]:
    """Gibt geladene und nicht geladene (nur heruntergeladene) Modelle zurück."""
    downloaded = lms.list_downloaded_models()
    loaded = lms.list_loaded_models()

    loaded_keys = {m.identifier for m in loaded}
    loaded_names = sorted(loaded_keys)
    unloaded_names = sorted(
        m.model_key for m in downloaded if m.model_key not in loaded_keys
    )
    return {"loaded": loaded_names, "unloaded": unloaded_names}


if __name__ == "__main__":
    models = list_models()

    print("Geladen:")
    for name in models["loaded"] or ["(keine)"]:
        print(f"  - {name}")

    print("Nicht geladen:")
    for name in models["unloaded"] or ["(keine)"]:
        print(f"  - {name}")
