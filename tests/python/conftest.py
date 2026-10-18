from importlib.util import find_spec

OPTIONAL_TEST_DIRECTORIES = {
    "data/*": ("polars",),
    "model/*": ("torch",),
    "eval/*": ("torch",),
    "bench/*": ("h5py",),
}


def _installed(modules: tuple[str, ...]) -> bool:
    return all(find_spec(module) is not None for module in modules)


collect_ignore_glob = [
    pattern for pattern, modules in OPTIONAL_TEST_DIRECTORIES.items() if not _installed(modules)
]
