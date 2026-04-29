from pathlib import Path

import yaml
from pydantic import ValidationError

from huginn.schemas import RuntimeConfig


def load_runtime_config(path: str | Path) -> RuntimeConfig:
    config_path = Path(path)
    raw_data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw_data, dict):
        raise ValueError(f"Invalid runtime config in {config_path}")

    try:
        return RuntimeConfig.model_validate(raw_data)
    except ValidationError as exc:
        raise ValueError(str(exc)) from exc


__all__ = ["RuntimeConfig", "load_runtime_config"]
