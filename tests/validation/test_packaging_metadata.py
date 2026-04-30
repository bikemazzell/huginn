from pathlib import Path
import tomllib


def test_runtime_dependencies_include_sqlite_vec() -> None:
    pyproject_path = Path(__file__).resolve().parents[2] / "pyproject.toml"
    pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))

    dependencies = pyproject["project"]["dependencies"]

    assert any(dependency.startswith("sqlite-vec") for dependency in dependencies)
