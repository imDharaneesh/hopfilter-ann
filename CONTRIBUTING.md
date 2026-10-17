# Contributing to hopfilter-ann

Thanks for your interest. This guide covers the workflow. Environment setup is in [docs/development.md](docs/development.md), and the coding rules are in [PLAN.md section 2](PLAN.md#2-working-rules).

## Before you start

- **Bugs:** open an issue with the bug form and a minimal reproduction. Synthetic data is fine.
- **Features:** open a feature request first. Changes that add a new interface, pattern or dependency should be agreed on before code is written.
- **Good first issues** are labelled [`good first issue`](https://github.com/imDharaneesh/hopfilter-ann/labels/good%20first%20issue).

## Workflow

1. Fork, then create a branch from `master`: `feat/heuristic-selector`, `fix/visited-set-wrap`, `docs/benchmarks-hardware`.
2. Set up with `uv sync` and `uv run pre-commit install`.
3. If the change adds or alters an interface, update [docs/design.md](docs/design.md) first.
4. Write the tests, then the implementation.
5. Run the [quality gates](#quality-gates) locally.
6. Open a pull request using the template. Keep it to one concern.

## Quality gates

A pull request is mergeable when all of these pass. CI runs every one of them.

```bash
uv run pre-commit run --all-files
uv run pytest --cov=hopfilter
cmake -S . -B build -DHOPFILTER_BUILD_TESTS=ON -DHOPFILTER_WARNINGS_AS_ERRORS=ON
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

These also apply:

- Every public class has unit tests. Python line coverage stays at or above 85%.
- Any index or search change is tested against brute-force ground truth.
- C++ changes are clean under ASan, UBSan and TSan (CI runs them; see [docs/development.md](docs/development.md#sanitizers) for running them locally).
- `perf:` changes include before and after numbers and the exact command that produced them.

## Code rules in short

| Rule | Detail |
|---|---|
| Indentation | 4 spaces, never tabs, in every file type |
| Comments | Minimal. Docstrings on public Python APIs, a short header on public C++ classes, and the occasional one-line *why*. No commented-out code, no TODOs |
| C++ | C++20, explicit `std::`, `#pragma once`, RAII ownership only, no exceptions in hot loops |
| Python | 3.13+, full type hints, `mypy --strict` clean, specific exception types |
| Size | Functions under 30 lines, files under 300, nesting depth at most 3 |
| Design | SOLID; inject dependencies through constructors; add a pattern only when it removes duplication or opens a real extension point |

C++ naming is enforced by clang-tidy:

| Element | Style | Example |
|---|---|---|
| Types | `PascalCase` | `HnswGraph` |
| Functions, methods, variables, parameters | `camelCase` | `selectNeighbors`, `entryPoint` |
| Private and protected members | `snake_case_` | `entry_point_` |
| Constants | `kPascalCase` | `kDefaultEfSearch` |
| Namespaces | `lower_case` | `hopfilter` |

## Commit messages

[Conventional Commits](https://www.conventionalcommits.org/), one logical change per commit:

```
<type>(<optional scope>): <what changed, imperative, lower case>

<why it changed and anything a reviewer would not guess>
```

| Type | Use for |
|---|---|
| `feat` | New user-visible capability |
| `fix` | Bug fix |
| `perf` | Performance change, with numbers in the body |
| `refactor` | Behaviour-preserving restructuring |
| `test` | Tests only |
| `docs` | Documentation only |
| `build` | CMake, packaging, Docker, dependencies |
| `ci` | Workflows |
| `chore` | Tooling and repo maintenance |

Examples:

```
feat(graph): heuristic neighbor selection from the hnsw paper
fix(search): stop ef growth at maxEf when every node is rejected
perf(distance): avx2 l2 kernel, 3.1x on sift1m single-thread
```

## Independence rule

hopfilter implements HNSW from Malkov and Yashunin and logQ correction from Yi et al. **Do not copy, translate or closely paraphrase code from hnswlib, FAISS or other ANN libraries.** Reading papers and their pseudocode is encouraged. Pull requests that appear derived from another implementation's code will be closed.

## Licensing

By contributing, you agree that your contributions are licensed under the [Apache License 2.0](LICENSE).
