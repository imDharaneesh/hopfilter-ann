# hopfilter-ann: Development Plan

| | |
|---|---|
| Package (PyPI) | `hopfilter-ann` |
| Import name | `hopfilter` |
| CLI | `hopfilter` |
| Languages | C++20 (core), Python 3.13+ (training, pipeline, tooling) |
| License | Apache-2.0 |
| Target | v0.1.0 in roughly 7 weeks at about 15 hours/week |

This document is the source of truth for **scope, phases and working rules**. Architecture lives in [`docs/design.md`](docs/design.md), benchmark methodology in [`docs/benchmarks.md`](docs/benchmarks.md), and day-to-day setup in [`docs/development.md`](docs/development.md).

---

## 1. What we are building and why

### The problem

Large platforms hold millions to billions of items and must return the best few for a user in tens of milliseconds. Scoring every item with a heavy model is impossible, so systems split the work:

1. **Retrieval** narrows millions of items to a few hundred candidates, fast.
2. **Ranking** re-scores those candidates with a heavier model.

Real queries also carry conditions ("in stock", "category X", "allowed in this region"). Most vector-search tools handle these filters badly:

- **Post-filtering** fetches the top-k and then drops non-matching items, so strict filters return too few results.
- **Pre-filtering** restricts the graph to matching items, which breaks graph connectivity and falls back to slow scans.

### The solution

An open-source retrieval pipeline with three parts:

1. **Two-tower model (PyTorch).** A user tower and an item tower map to a shared vector space, trained with in-batch negatives and **logQ correction** to remove the popularity bias of sampled negatives.
2. **HNSW index (C++20).** A layered proximity graph implemented from the original paper, exposed to Python through pybind11.
3. **Filtered search (the differentiator).** Evaluate the filter during graph traversal, adapt the search width to filter selectivity, and fall back to exact search over the matching set when the filter is extremely strict.

### The proof

A reproducible benchmark suite against FAISS and hnswlib: recall@10/100, QPS, p50/p99 latency, build time, memory, and a filter-selectivity sweep at 100%, 50%, 10%, 1% and 0.1%.

### Honest scope

We will not beat FAISS or production systems at Google or Meta. The goal is a correct, readable, well-measured implementation that is competitive on recall, with a measurable advantage on filtered queries at specific selectivities. We report where we trail.

### Out of scope for v0.1

IVF-PQ, multimodal embeddings, distributed sharding, GPU. The ranking stage is an optional example only.

---

## 2. Working rules

### 2.1 Code style

- **Indentation: 4 spaces, never tabs**, in every language (C++, Python, YAML, CMake, JSON, TOML). Enforced by `.editorconfig`, `clang-format`, `ruff format` and a pre-commit hook that rejects tab characters.
- **Comments: minimal.** Code documents itself through names and small functions.
    - No comments restating what the code does, no commented-out code, no TODO comments (open an issue instead).
    - Allowed: docstrings on public Python APIs, a short Doxygen header on public C++ classes, and a rare one-line comment explaining a *non-obvious why* (a SIMD alignment requirement, a detail from the paper). Target: fewer than one comment per 40 lines.
- **C++:** C++20. Explicit `std::` qualifiers, never `using namespace std;`. `#pragma once` in every header. Warnings as errors in CI (`-Wall -Wextra -Wpedantic -Wshadow -Wconversion ...` on GCC/Clang, `/W4 /permissive-` on MSVC).
- **Python:** 3.13+, full type hints, `ruff` for lint and format, `mypy --strict` on the package.
- **Naming** (enforced by clang-tidy and ruff):

    | Element | C++ | Python |
    |---|---|---|
    | Types | `PascalCase` | `PascalCase` |
    | Functions and methods | `camelCase` | `snake_case` |
    | Variables and parameters | `camelCase` | `snake_case` |
    | Private members | `snake_case_` | `_snake_case` |
    | Constants | `kPascalCase` | `UPPER_SNAKE_CASE` |
    | Namespaces and modules | `lower_case` | `lower_case` |

    Names express intent: `neighborSelector`, not `ns`.
- Functions do one thing and stay under 30 lines. Files stay under 300 lines. Maximum nesting depth is 3. Prefer early returns.
- No magic numbers: use named constants or config fields.

### 2.2 Design principles

- **SOLID**
    - **Single responsibility:** one reason to change per class. The index does not parse datasets or compute metrics.
    - **Open/closed:** new metrics, selectors or search strategies are added without editing existing classes.
    - **Liskov:** any `SearchStrategy` or `DistanceMetric` implementation is substitutable without breaking callers.
    - **Interface segregation:** small interfaces (`Searchable`, `Insertable`, `Serializable`) instead of one fat index interface.
    - **Dependency inversion:** high-level code depends on abstractions. Dependencies are injected through constructors and never created inside business logic.
- **Low-level design first.** Every module has a clear interface, a clear owner and no cyclic dependencies. The class diagram in `docs/design.md` is updated *before* a module is coded.
- **Design patterns, only where they earn their place:**
    - **Strategy:** `DistanceMetric`, `NeighborSelector`, `SearchStrategy`, `LossFunction`.
    - **Factory / Registry:** build metrics, strategies, datasets and models from config names.
    - **Builder:** `HnswIndexBuilder` validates `IndexConfig` before construction.
    - **Facade:** `RetrievalPipeline` exposes train, export, build, search and evaluate behind one API.
    - **Template Method:** the shared skeleton of benchmark runners (load, build, query, measure, report).
    - **Adapter:** FAISS and hnswlib behind the same `AnnIndex` interface for fair benchmarking.
    - **RAII:** every C++ resource is owned by `std::unique_ptr` or a container. No raw owning pointers, no manual `delete`.
- A pattern is added only when it removes duplication or enables a real extension point.
- **Errors:** C++ throws at API boundaries on contract violations and never inside hot loops. Python raises specific exceptions (`InvalidConfigError`, `IndexNotBuiltError`, all deriving from `HopfilterError`), never bare `Exception`.
- **Concurrency:** thread-safety guarantees are documented in each interface (reads are lock-free after build, inserts take a lock). No data races; ThreadSanitizer runs in CI.
- **Immutability:** `const` wherever possible in C++, frozen dataclasses for Python configs.

### 2.3 Quality gates

A change is done only when all of these pass:

- Unit tests for every public class. At least 85% line coverage for Python (enforced by `pytest --cov`), meaningful coverage for the C++ core.
- Index correctness is always tested against brute-force ground truth.
- ASan, UBSan and TSan clean.
- `uv run pre-commit run --all-files` is clean (ruff, ruff format, mypy, clang-format, whitespace, no tabs). clang-tidy is clean on the core.
- Every performance claim comes with a reproducible benchmark script and a recorded number.

### 2.4 Commits

- Conventional commits: `feat:`, `fix:`, `perf:`, `test:`, `docs:`, `build:`, `ci:`, `chore:`, `refactor:`, with an optional scope such as `feat(graph):`.
- One logical block per commit (an interface with its tests, a build step, a workflow), not one file per commit and not a whole phase per commit.
- The subject line says what changed. The body says why and anything a reviewer would not guess.

### 2.5 Independence rule

HNSW is implemented from Malkov and Yashunin (2016/2018). logQ correction is implemented from Yi et al. (2019). **No code is copied or translated from hnswlib or FAISS.** They are used only as external benchmark baselines.

---

## 3. Tech stack

| Layer | Choice |
|---|---|
| Training | PyTorch, NumPy, Polars, plain YAML configs |
| Index core | C++20, CMake 3.24+, OpenMP, SIMD distance kernels (AVX2/FMA with scalar fallback and runtime CPU dispatch) |
| Bindings | pybind11 3, built by scikit-build-core |
| Environment | uv (`uv.lock` pins every dev tool), `.python-version` = 3.14 |
| Benchmarks | faiss-cpu, hnswlib, h5py, matplotlib |
| Tests | GoogleTest (C++), pytest + hypothesis (Python), Google Benchmark (micro) |
| Quality | clang-format 18, clang-tidy 18, ruff, mypy, pre-commit, ASan/UBSan/TSan |
| CI/CD | GitHub Actions, cibuildwheel, PyPI trusted publishing |
| Container | Multi-stage Dockerfile on `python:3.14-slim-trixie` |
| Docs | MkDocs Material |
| Optional | FastAPI demo |

**Windows note.** The primary dev machine runs Windows with Visual Studio 2026 (MSVC). Local Python builds use MSVC through scikit-build-core. GCC/Clang parity comes from CI and from the Docker image. AVX2 code is compiled in its own translation unit with its own flags, and the kernel is chosen at runtime through CPU feature detection, so wheels never crash on older CPUs.

---

## 4. Repository layout

```
hopfilter-ann/
    CMakeLists.txt               root build, options, subdirectories
    pyproject.toml               packaging, dependency groups, tool config
    uv.lock                      locked dev environment
    cmake/                       warnings and sanitizer helpers
    core/
        include/hopfilter/
            common/              version, types, errors, visited set, thread pool
            distance/            DistanceMetric, L2, InnerProduct, Cosine, kernels
            storage/             VectorStore (aligned contiguous vectors)
            graph/               HnswGraph, LayerAssigner, NeighborSelector
            index/               HnswIndex, IndexConfig, HnswIndexBuilder
            filter/              FilterPredicate, AttributeStore, Selectivity
            search/              SearchStrategy, PostFilter, PreFilter, InGraph, Adaptive
            io/                  IndexSerializer
        src/                     mirrors include/
    bindings/                    pybind11 module (_core)
    python/hopfilter/
        data/                    loaders, splits, download scripts
        model/                   towers, losses, trainer
        export/                  embedding export
        index/                   Python facade over bindings, adapters
        eval/                    metrics, ground truth
        pipeline/                RetrievalPipeline facade
        cli.py
    benchmarks/
        configs/  runners/  results/  plots/
    tests/
        cpp/                     GoogleTest, mirrors core/
        python/                  pytest
    docs/                        MkDocs site (design, benchmarks, development)
    examples/
    .github/                     workflows, issue forms, PR template
```

Datasets are never committed. Download scripts fetch them to `data/`, which is git-ignored.

---

## 5. Low-level design

The full design, with class diagrams, interface signatures, ownership and thread-safety contracts, is in [`docs/design.md`](docs/design.md). Summary:

### 5.1 C++ core

| Component | Responsibility | Pattern |
|---|---|---|
| `DistanceMetric` | `float distance(std::span<const float>, std::span<const float>)` | Strategy |
| `L2Metric`, `InnerProductMetric`, `CosineMetric` | Concrete metrics with SIMD dispatch | Strategy |
| `VectorStore` | Contiguous, aligned vector storage | RAII |
| `HnswGraph` | Layered adjacency lists, entry point, level lookup | Single responsibility |
| `LayerAssigner` | Random level from an exponential distribution with `mL = 1/ln(M)` | Single responsibility |
| `NeighborSelector` | Picks M neighbors from candidates (simple or the paper's heuristic) | Strategy |
| `VisitedSet` | Epoch-stamped visited tracking with no per-query reallocation | Object reuse |
| `FilterPredicate` | `bool matches(NodeId)` over attribute columns or bitsets | Strategy |
| `AttributeStore` | Per-item metadata (categorical, numeric range) | Single responsibility |
| `SearchStrategy` | `search(query, k, filter) -> SearchResults` | Strategy |
| `AdaptiveStrategy` | Picks PostFilter, InGraph or exact search from estimated selectivity | Strategy + policy |
| `HnswIndex` | Orchestrates insert and search through injected collaborators | Facade |
| `HnswIndexBuilder` | Validates `IndexConfig` (M, efConstruction, metric, seed) and builds the index | Builder |
| `IndexSerializer` | Versioned binary save and load | Single responsibility |

Interfaces are segregated into `Insertable`, `Searchable` and `Serializable`. `HnswIndex` implements all three; benchmark adapters implement only `Searchable`.

### 5.2 Python package

| Component | Responsibility |
|---|---|
| `DatasetLoader` (protocol) | Yields interactions and item metadata: `MovieLensLoader`, `AmazonReviewsLoader` |
| `TemporalSplitter` | Time-based train/validation/test split with no leakage |
| `UserTower`, `ItemTower` | Embedding + MLP with L2-normalized output |
| `LossFunction` (protocol) | `InBatchSoftmaxLoss`, `LogQCorrectedLoss` |
| `ItemFrequencyEstimator` | Streaming estimate of item sampling probability for logQ |
| `Trainer` | Training loop with injected model, loss, optimizer and scheduler |
| `EmbeddingExporter` | Writes item/user vectors and attributes to disk |
| `AnnIndex` (protocol) | `build`, `search`: `HopfilterAdapter`, `FaissAdapter`, `HnswlibAdapter` |
| `GroundTruthBuilder` | Exact brute-force neighbors, with and without filters |
| `MetricsCalculator` | recall@k, QPS, latency percentiles |
| `RetrievalPipeline` | Facade running train, export, build, search and evaluate from one config |
| `cli.py` | `hopfilter train / build / search / bench` |

---

## 6. Phases

Each phase ends with a written summary of what was built and the measured results. The next phase starts only after confirmation.

### Phase 0: Foundation (days 1-3)

- Repository hygiene: license, `.editorconfig`, `.gitattributes`, `.clang-format`, `.clang-tidy`, ruff and mypy config, pre-commit.
- CMake project with `hopfilter_core` as a static library, GoogleTest wired in, a pybind11 `_core` module, scikit-build-core building it through `uv sync`.
- CI: lint, clang-tidy, Python tests (3 OSes x 2 Pythons), C++ tests (GCC, Clang, AppleClang, MSVC), sanitizers, Docker build. Release workflow with cibuildwheel.
- `docs/design.md` with the class diagrams from section 5.
- **Exit:** `uv sync` builds and installs the package locally, all gates pass, and CI is green.

### Phase 1: Two-tower baseline (week 1)

- Download script and loader for MovieLens (1M first, then 25M). Temporal split.
- Towers, in-batch softmax loss, trainer, evaluation against exact search.
- `LogQCorrectedLoss` with a frequency estimator. Ablation with and without logQ.
- **Exit:** recall@k reported for both losses, logQ plot saved, unit tests for loss math on hand-computed small cases.
- **Outcome:** done. logQ raised test recall@50 from 0.039 to 0.109 on MovieLens-1M, but *lowered* tail recall (0.049 to 0.033), the opposite of this plan's original expectation of tail gains. Neither model beats the most-popular baseline (0.132) yet. Details in [`docs/benchmarks.md`](docs/benchmarks.md#logq-ablation-phase-1).

### Phase 2: HNSW core (weeks 2-3)

- Implement in order: scalar `DistanceMetric`, `VectorStore`, `LayerAssigner`, `HnswGraph`, `VisitedSet`, the search-layer routine, heuristic `NeighborSelector`, insert, query.
- `HnswIndexBuilder`, `IndexSerializer`, bindings with zero-copy NumPy input.
- Tests: graph invariants (degree bounds, level consistency), recall against brute force on small random data, serialization round trip, edge cases (empty index, `k` larger than size, duplicate vectors).
- Download SIFT1M and GloVe from ANN-Benchmarks and tune `M`, `efConstruction`, `efSearch`.
- **Exit:** about 0.95 recall@10 on SIFT1M, all tests and sanitizers clean.

### Phase 3: Performance (week 4)

- AVX2/FMA kernels with runtime dispatch, aligned contiguous layout for vectors and level-0 neighbor lists, software prefetch, parallel build with fine-grained locks.
- Profile (`perf` on Linux; VTune or Windows Performance Analyzer locally). Record the speedup of each optimization separately in `docs/benchmarks.md`.
- **Exit:** QPS and memory table against hnswlib and FAISS HNSW at matched recall, with gaps explained honestly.

### Phase 4: Filtered search (week 5)

- `AttributeStore` and `FilterPredicate` (category equality, numeric range, boolean combinations).
- Baselines first: `PostFilterStrategy` (over-fetch, then drop) and `PreFilterStrategy` (brute force on the matching subset).
- `InGraphStrategy`: apply the predicate during traversal, keep non-matching nodes as routing waypoints, and widen `ef` when the rejection rate is high.
- `AdaptiveStrategy`: estimate selectivity (by sampling or bitset popcount), then pick PostFilter (high), InGraph (medium) or exact search (very low).
- Read ACORN and related filtered-ANN papers for ideas, then implement independently.
- Sweep selectivity at 100%, 50%, 10%, 1% and 0.1% on item attributes (MovieLens genres and years, Amazon categories).
- **Exit:** recall-versus-QPS curves per selectivity for all four strategies, with a clear statement of where Adaptive wins and loses.

### Phase 5: Pipeline and benchmarks (week 6)

- `RetrievalPipeline` facade and CLI: one config runs the whole flow.
- Benchmark runners built as a Template Method with FAISS and hnswlib adapters, so everything is measured the same way (same machine, same threads, warm-up, fixed seeds, several repeats).
- Auto-generated plots and a results table committed under `benchmarks/results`.
- Optional: a small FastAPI demo and a GBDT re-ranking example over the top-200 candidates.
- **Exit:** `hopfilter bench --config benchmarks/configs/sift1m.yaml` reproduces the published numbers.

### Phase 6: Release (week 7)

- cibuildwheel wheels for Linux, Windows and macOS, PyPI trusted publishing, tagged release v0.1.0.
- MkDocs site, a quickstart that runs in under 5 minutes, and a README with honest numbers and plots.
- One blog post on the design decisions (logQ, filtered traversal, SIMD dispatch).
- 5-10 `good first issue` items. Share with relevant communities.
- **Exit:** `pip install hopfilter-ann` works on a clean machine and the quickstart runs.

---

## 7. Testing strategy

| Level | What | Tool |
|---|---|---|
| Unit (C++) | Metrics, selectors, layer assigner, visited set, predicates | GoogleTest |
| Property | Results always satisfy the filter, are sorted by distance and contain no duplicates | pytest + hypothesis |
| Correctness | Recall against brute-force ground truth on random and real data | pytest |
| Safety | ASan, UBSan, TSan (concurrent build and search) | CI jobs |
| Regression | An index serialized by the previous version still loads | pytest fixtures |
| Performance | Distance-kernel micro-benchmarks, end-to-end QPS | Google Benchmark, runners |
| Model | Loss values on tiny hand-computed batches, deterministic-seed training smoke test | pytest |

---

## 8. Container

See [`Dockerfile`](Dockerfile). It has three stages:

| Target | Base | Contents | Default command |
|---|---|---|---|
| `builder` | `python:3.14-slim-trixie` + build-essential + uv | Builds the wheel with `uv build` | n/a |
| `dev` | `builder` | Wheel + locked `test` group + tests | `pytest -q tests/python` |
| `runtime` | `python:3.14-slim-trixie` | Wheel only, non-root `app` user, no compilers | `hopfilter --help` |

```bash
docker build --target dev -t hopfilter-dev .
docker run --rm hopfilter-dev
docker build --target runtime -t hopfilter-ann .
docker run --rm hopfilter-ann --version
```

A HEALTHCHECK is added only if the optional FastAPI server ships. The base image digest is pinned before release.

---

## 9. Continuous integration

See [`.github/workflows/ci.yml`](.github/workflows/ci.yml) and [`.github/workflows/release.yml`](.github/workflows/release.yml).

| Workflow | Job | What it checks |
|---|---|---|
| `ci` | `lint` | All pre-commit hooks on every file |
| `ci` | `clang-tidy` | clang-tidy 18 on the core and C++ tests |
| `ci` | `python-tests` | pytest + coverage on Linux/Windows/macOS x Python 3.13/3.14 |
| `ci` | `cpp-tests` | ctest with GCC, Clang, AppleClang and MSVC, warnings as errors |
| `ci` | `sanitizers` | ASan, UBSan and TSan builds of the C++ tests |
| `ci` | `docker` | Builds the `dev` image and runs the test suite inside it |
| `release` | `build-wheels` | cibuildwheel for cp313/cp314 on Linux x86_64 and aarch64, Windows amd64, macOS arm64 |
| `release` | `build-sdist` | `uv build --sdist` |
| `release` | `publish` | PyPI trusted publishing (OIDC, no stored tokens) |

Repository setup: create a `pypi` environment in the GitHub settings, register this repository and workflow as a trusted publisher on PyPI, and require the `ci` workflow to pass before merging to `master`.

---

## 10. Benchmark methodology

Summary; the full protocol is in [`docs/benchmarks.md`](docs/benchmarks.md).

- **Datasets:** SIFT1M, GloVe-100 and GIST1M (ANN-Benchmarks HDF5 with ground truth) for raw ANN; MovieLens-1M/25M and Amazon Reviews 2023 for the full pipeline and filters.
- **Metrics:** recall@10 and recall@100, QPS (single-thread and all threads), p50/p99 latency, build time, peak memory.
- **Fairness:** same machine, pinned threads, warm-up runs, fixed seeds, 5 repeats reported as median and spread, a parameter sweep per library to draw full recall-QPS curves, comparisons at matched recall.
- **Filtered:** attributes with controlled selectivity (100%, 50%, 10%, 1%, 0.1%); report recall, QPS and result count (to expose post-filter shortfalls).
- **Reporting:** raw results and the plotting script are committed. Hardware is always stated.

---

## 11. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Scope creep | v0.1 is HNSW + filtered search + pipeline only |
| Accidental copying | Implement from papers; baselines are used only as external libraries |
| Subtle graph bugs | Brute-force ground-truth tests, invariants, sanitizers |
| SIMD wheel crashes on old CPUs | Runtime dispatch, one translation unit per instruction set |
| Windows/Linux divergence | CI matrix across MSVC, GCC, Clang; Docker for GCC parity |
| Weak benchmark numbers | Publish anyway, with an analysis of the gap |
| Dataset licensing | Download scripts only, never redistribute data |
| Toolchain drift | Every dev tool pinned in `uv.lock`; Actions pinned to major versions |

---

## 12. Definition of done (v0.1.0)

- [ ] `pip install hopfilter-ann` works on Linux, Windows and macOS
- [ ] Quickstart runs end to end on MovieLens in under 5 minutes
- [ ] About 0.95 recall@10 on SIFT1M with documented QPS against hnswlib and FAISS
- [ ] Filtered-search sweep with plots and honest conclusions
- [ ] logQ ablation plot
- [ ] CI green (lint, tests, sanitizers, Docker), coverage reported
- [ ] Docs site, design doc, blog post
- [ ] Apache-2.0 license, CONTRIBUTING.md, issue templates, good first issues

---

## 13. Progress

| Phase | Status |
|---|---|
| 0. Foundation | Done |
| 1. Two-tower baseline | Done; follow-ups: early stopping, temperature sweep, MovieLens-25M run |
| 2. HNSW core | Not started |
| 3. Performance | Not started |
| 4. Filtered search | Not started |
| 5. Pipeline and benchmarks | Not started |
| 6. Release | Not started |
