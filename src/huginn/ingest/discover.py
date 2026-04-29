from pathlib import Path


SUPPORTED_EXTENSIONS = {".pdf"}


def discover_supported_files(root: str | Path) -> list[Path]:
    root_path = Path(root)
    discovered: list[Path] = []

    for path in root_path.rglob("*"):
        if not path.is_file():
            continue
        if any(part.startswith(".") for part in path.relative_to(root_path).parts[:-1]):
            continue
        if path.suffix.lower() in SUPPORTED_EXTENSIONS:
            discovered.append(path)

    return sorted(discovered)
