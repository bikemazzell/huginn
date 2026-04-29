from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def _load_preflight_script():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "preflight.py"
    spec = spec_from_file_location("preflight_script", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_preflight_script_imports_urllib_for_endpoint_check() -> None:
    module = _load_preflight_script()

    assert hasattr(module, "urllib")
