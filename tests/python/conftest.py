from importlib.util import find_spec

TRAIN_EXTRA_MODULES = ("torch", "polars")
TRAIN_EXTRA_TEST_DIRECTORIES = ("data/*", "model/*", "eval/*")


def _train_extra_installed() -> bool:
    return all(find_spec(module) is not None for module in TRAIN_EXTRA_MODULES)


collect_ignore_glob = [] if _train_extra_installed() else list(TRAIN_EXTRA_TEST_DIRECTORIES)
