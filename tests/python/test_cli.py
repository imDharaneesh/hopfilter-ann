import subprocess
import sys
from pathlib import Path

import pytest

import hopfilter
from hopfilter.cli import main
from hopfilter.data.download import DATASETS


def test_version_flag_prints_package_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])

    assert exit_info.value.code == 0
    assert capsys.readouterr().out.strip() == f"hopfilter {hopfilter.__version__}"


def test_no_arguments_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 0
    assert "usage: hopfilter" in capsys.readouterr().out


def test_unknown_flag_exits_with_usage_error() -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["--no-such-flag"])

    assert exit_info.value.code == 2


class RecordingFetcher:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.urls: list[str] = []

    def fetch(self, url: str, destination: Path) -> None:
        self.urls.append(url)
        destination.write_bytes(self.payload)


def test_download_command_reports_integrity_failure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fetcher = RecordingFetcher(b"not the real archive")

    exit_code = main(["download", "movielens-1m", "--data-dir", str(tmp_path)], fetcher=fetcher)

    assert exit_code == 1
    assert fetcher.urls == [DATASETS["movielens-1m"].url]
    assert "error: " in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == []


def test_download_command_prints_existing_dataset_dir(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "ml-1m").mkdir()
    fetcher = RecordingFetcher(b"")

    exit_code = main(["download", "movielens-1m", "--data-dir", str(tmp_path)], fetcher=fetcher)

    assert exit_code == 0
    assert fetcher.urls == []
    assert capsys.readouterr().out.strip() == str(tmp_path / "ml-1m")


def test_download_rejects_unknown_dataset() -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["download", "imagenet"])

    assert exit_info.value.code == 2


def test_module_entry_point_runs() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "hopfilter", "--version"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert hopfilter.__version__ in completed.stdout
