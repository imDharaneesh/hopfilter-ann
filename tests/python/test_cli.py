import subprocess
import sys

import pytest

import hopfilter
from hopfilter.cli import main


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


def test_module_entry_point_runs() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "hopfilter", "--version"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert hopfilter.__version__ in completed.stdout
