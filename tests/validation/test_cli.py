import pytest

from huginn.cli import build_parser


def test_cli_accepts_ingest_subcommand() -> None:
    parser = build_parser()

    args = parser.parse_args(["ingest", "/tmp/docs"])

    assert args.command == "ingest"
    assert args.folder == "/tmp/docs"
    assert args.reindex is False


def test_cli_requires_question_for_ask_subcommand() -> None:
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["ask"])


def test_cli_accepts_status_subcommand() -> None:
    parser = build_parser()

    args = parser.parse_args(["status"])

    assert args.command == "status"


def test_cli_accepts_db_path_and_reindex_for_ingest() -> None:
    parser = build_parser()

    args = parser.parse_args(["--db-path", "/tmp/custom.db", "ingest", "--reindex", "/tmp/docs"])

    assert args.db_path == "/tmp/custom.db"
    assert args.command == "ingest"
    assert args.reindex is True
    assert args.folder == "/tmp/docs"
