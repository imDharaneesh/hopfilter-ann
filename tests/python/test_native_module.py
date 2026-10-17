import re
from importlib.metadata import version as distribution_version

import hopfilter
from hopfilter import _core


def test_native_version_matches_distribution_metadata() -> None:
    assert _core.version() == distribution_version("hopfilter-ann")


def test_package_reexports_native_version() -> None:
    assert hopfilter.__version__ == _core.__version__


def test_version_is_semantic() -> None:
    assert re.fullmatch(r"\d+\.\d+\.\d+", hopfilter.__version__)
