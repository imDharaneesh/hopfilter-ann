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

## Datasets and the training extra

The model, data and evaluation code needs the `train` extra (PyTorch, Polars); `uv sync` installs it through the `dev` group. Plain `pip install hopfilter-ann` does not, and the tests under `tests/python/{data,model,eval}` are skipped when it is missing. On Linux, PyTorch comes from the CPU wheel index, so no CUDA download is needed.

Datasets are fetched, checksum-verified and extracted into `data/` (git-ignored):

```bash
uv run hopfilter download movielens-1m
```

Available names: `movielens-1m`, `movielens-25m`. Use `--data-dir` to put them elsewhere.

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

`tests/fixtures/index_format_v1.hopf` is an index saved in format version 1, and the test suite checks that it still loads. Regenerate it only when you deliberately start a new format version:

```bash
build/tests/cpp/Release/hopfilter_core_tests --gtest_also_run_disabled_tests --gtest_filter=*WriteVersionOneFixture
```

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

clang-tidy 22 is pinned in the `lint` group (clang-format stays at 18 so formatting never shifts). It needs a `compile_commands.json`, which the Ninja and Makefile generators produce. On Linux and macOS:

```bash
cmake -S . -B build-tidy -G Ninja -DHOPFILTER_BUILD_TESTS=ON
git ls-files 'core/src/*.cpp' 'tests/cpp/*.cpp' | xargs uv run clang-tidy -p build-tidy --quiet
```

On Windows, generate the database from a Visual Studio developer prompt (`vcvars64.bat`) with `-G Ninja -DCMAKE_CXX_COMPILER=cl`, and clang-tidy then checks against the MSVC standard library. Avoid a MinGW compile database: clang's parser lags the newest GCC libstdc++ headers.

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
