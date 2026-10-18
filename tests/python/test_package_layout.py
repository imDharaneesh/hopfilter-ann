import importlib

import pytest

MODULES_WITHOUT_OPTIONAL_DEPENDENCIES = [
    "hopfilter",
    "hopfilter.cli",
    "hopfilter.config",
    "hopfilter.errors",
    "hopfilter.data",
    "hopfilter.data.download",
    "hopfilter.model",
    "hopfilter.eval",
]


@pytest.mark.parametrize("module", MODULES_WITHOUT_OPTIONAL_DEPENDENCIES)
def test_installed_package_contains_module(module: str) -> None:
    assert importlib.import_module(module) is not None
