# Development

Everything needed to build, test and change hopfilter locally.

## Prerequisites

| Tool | Version | Notes |
|---|---|---|
| [uv](https://docs.astral.sh/uv/) | 0.11+ | Manages Python, the virtualenv and every dev tool |
| CMake | 3.24+ | uv installs one automatically if missing |
| C++20 compiler | see below | |
| Git | any recent | |
| Docker | optional | For the Linux/GCC parity image |

| Platform | Compiler |
|---|---|
| Windows | Visual Studio 2022 or 2026 with the **Desktop development with C++** workload |
| Linux | GCC 11+ or Clang 16+ |
| macOS | Xcode Command Line Tools (AppleClang 15+) |

You do not need to install Python yourself: uv downloads the version pinned in `.python-version` (3.14).

## First-time setup

```bash
git clone https://github.com/imDharaneesh/hopfilter-ann.git
cd hopfilter-ann
uv sync
uv run pre-commit install
```

`uv sync` creates `.venv`, builds the C++ extension in editable mode, and installs the `dev` dependency group (test, lint and docs tools) at the versions locked in `uv.lock`. `pre-commit install` makes every `git commit` run the same checks as CI.

Check that it worked:

```bash
uv run hopfilter --version
```

## Everyday commands

| Task | Command |
|---|---|
| Run Python tests | `uv run pytest` |
| Python tests with coverage | `uv run pytest --cov=hopfilter` |
| All lint and format checks | `uv run pre-commit run --all-files` |
| Lint only | `uv run ruff check python tests` |
| Format | `uv run ruff format python tests` |
| Type check | `uv run mypy` |
| Serve docs locally | `uv run mkdocs serve` |
| Build docs (strict) | `uv run mkdocs build --strict` |

### Rebuilding after C++ changes

`pyproject.toml` lists the C++ sources under `[tool.uv] cache-keys`, so the next `uv sync` or `uv run` rebuilds the extension after any change in `core/`, `bindings/` or the CMake files. To force a rebuild:

```bash
uv sync --reinstall-package hopfilter-ann
```

## C++ core on its own

The core and its GoogleTest suite build without Python:

```bash
cmake -S . -B build -DHOPFILTER_BUILD_TESTS=ON -DHOPFILTER_WARNINGS_AS_ERRORS=ON
cmake --build build --config Release --parallel
ctest --test-dir build -C Release --output-on-failure
```

GoogleTest is downloaded by CMake on the first configure.

### CMake options

| Option | Default | Effect |
|---|---|---|
| `HOPFILTER_BUILD_TESTS` | `OFF` | Build `tests/cpp` |
| `HOPFILTER_BUILD_PYTHON` | `ON` under scikit-build-core, otherwise `OFF` | Build the `_core` extension |
| `HOPFILTER_WARNINGS_AS_ERRORS` | `OFF` | `-Werror` / `/WX`; always `ON` in CI |
| `HOPFILTER_SANITIZER` | empty | `address`, `undefined` or `thread` (GCC/Clang); `address` only on MSVC |

### Sanitizers

Sanitizer builds need GCC or Clang, so on Windows run them in WSL2 or Docker:

```bash
cmake -S . -B build-asan -G Ninja -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_BUILD_TYPE=RelWithDebInfo -DHOPFILTER_BUILD_TESTS=ON -DHOPFILTER_SANITIZER=address
cmake --build build-asan
ctest --test-dir build-asan --output-on-failure
```

### clang-tidy

clang-tidy 18 is pinned in the `lint` group. It needs a `compile_commands.json`, which the Ninja and Makefile generators produce:

```bash
cmake -S . -B build-tidy -G Ninja -DHOPFILTER_BUILD_TESTS=ON
uv run clang-tidy -p build-tidy core/src/common/version.cpp
```

clang-tidy 18 cannot parse the libstdc++ headers shipped with GCC 15 and newer. Run it against GCC 13/14 headers, or rely on the CI job, which uses Ubuntu 24.04.

## Docker

```bash
docker build --target dev -t hopfilter-dev .
docker run --rm hopfilter-dev
```

The `dev` image builds the wheel with GCC on Debian and runs the Python test suite inside it. It is the quickest way to check Linux behaviour from a Windows machine.

## Repository map

| Path | Contents |
|---|---|
| `core/include/hopfilter/` | Public C++ headers, one directory per module |
| `core/src/` | C++ implementation, mirrors `include/` |
| `bindings/` | pybind11 module `hopfilter._core` |
| `python/hopfilter/` | Python package |
| `tests/cpp/`, `tests/python/` | Tests, mirroring the source layout |
| `cmake/` | Warning and sanitizer helpers |
| `benchmarks/` | Configs, runners, results, plots |
| `docs/` | This site |

## Conventions

The coding rules (naming, comment policy, function size, design patterns) are in [PLAN.md section 2](https://github.com/imDharaneesh/hopfilter-ann/blob/master/PLAN.md#2-working-rules). The contribution workflow (branches, commits, reviews) is in [CONTRIBUTING.md](https://github.com/imDharaneesh/hopfilter-ann/blob/master/CONTRIBUTING.md).

## Releasing

1. Bump `VERSION` in the `project()` call in `CMakeLists.txt`. The Python package version follows automatically.
2. Commit as `chore(release): vX.Y.Z` and merge to `master`.
3. Tag and push: `git tag vX.Y.Z && git push origin vX.Y.Z`.
4. The `release` workflow builds wheels and the sdist and publishes to PyPI through trusted publishing.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `uv sync` fails with "could not find any instance of Visual Studio" | Install the C++ workload in the Visual Studio Installer |
| Python still sees old C++ behaviour | `uv sync --reinstall-package hopfilter-ann` |
| `pre-commit` hook says `uv: command not found` | Install uv and make sure it is on `PATH` in the shell git uses |
| TSan fails with "unexpected memory mapping" on Linux | `sudo sysctl vm.mmap_rnd_bits=28` |
