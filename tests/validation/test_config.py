from pathlib import Path

import pytest

from huginn.config import load_runtime_config


def test_load_runtime_config_parses_valid_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "runtime.yaml"
    config_path.write_text(
        "\n".join(
            [
                f"root_path: {tmp_path / 'docs'}",
                "local_only: true",
                "models:",
                "  chat:",
                "    base_url: http://127.0.0.1:11434/v1",
                "    api_key: ollama",
                "    model: qwen3:8b",
                "  embedding:",
                "    base_url: http://127.0.0.1:11434/v1",
                "    api_key: ollama",
                "    model: bge-m3",
                "indexing:",
                "  chunk_size: 800",
                "  chunk_overlap: 120",
                "  top_k: 6",
                "features:",
                "  ocr_fallback: true",
                "  query_rewrite: false",
                "  rerank: false",
                "  answer_validation: false",
            ]
        ),
        encoding="utf-8",
    )

    config = load_runtime_config(config_path)

    assert config.root_path == tmp_path / "docs"
    assert config.models.chat.model == "qwen3:8b"
    assert config.models.embedding.model == "bge-m3"
    assert config.indexing.chunk_size == 800
    assert config.features.ocr_fallback is True


def test_load_runtime_config_rejects_chunk_overlap_larger_than_size(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "runtime.yaml"
    config_path.write_text(
        "\n".join(
            [
                f"root_path: {tmp_path / 'docs'}",
                "local_only: true",
                "models:",
                "  chat:",
                "    base_url: http://127.0.0.1:11434/v1",
                "    api_key: ollama",
                "    model: qwen3:8b",
                "  embedding:",
                "    base_url: http://127.0.0.1:11434/v1",
                "    api_key: ollama",
                "    model: bge-m3",
                "indexing:",
                "  chunk_size: 100",
                "  chunk_overlap: 101",
                "  top_k: 6",
                "features:",
                "  ocr_fallback: true",
                "  query_rewrite: false",
                "  rerank: false",
                "  answer_validation: false",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="chunk_overlap"):
        load_runtime_config(config_path)


def test_load_runtime_config_rejects_remote_endpoints_when_local_only(tmp_path: Path) -> None:
    config_path = tmp_path / "runtime.yaml"
    config_path.write_text(
        "\n".join(
            [
                f"root_path: {tmp_path / 'docs'}",
                "local_only: true",
                "models:",
                "  chat:",
                "    base_url: https://api.openai.com/v1",
                "    api_key: test",
                "    model: gpt-test",
                "  embedding:",
                "    base_url: http://127.0.0.1:11434/v1",
                "    api_key: ollama",
                "    model: bge-m3",
                "indexing:",
                "  chunk_size: 800",
                "  chunk_overlap: 120",
                "  top_k: 6",
                "features:",
                "  ocr_fallback: true",
                "  query_rewrite: false",
                "  rerank: false",
                "  answer_validation: false",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="local_only"):
        load_runtime_config(config_path)
