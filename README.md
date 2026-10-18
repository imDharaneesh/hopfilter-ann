# hopfilter-ann

[![ci](https://github.com/imDharaneesh/hopfilter-ann/actions/workflows/ci.yml/badge.svg)](https://github.com/imDharaneesh/hopfilter-ann/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
![Python 3.13+](https://img.shields.io/badge/python-3.13%2B-blue.svg)
![C++20](https://img.shields.io/badge/C%2B%2B-20-blue.svg)

**Two-tower retrieval with a from-scratch HNSW index and filter-aware search.**

> **Status: pre-alpha, Phase 2 of 6 done.** The two-tower model trains on MovieLens ([logQ results](docs/benchmarks.md#logq-ablation-phase-1)) and the from-scratch HNSW index reaches 0.964 recall@10 on SIFT1M ([results](docs/benchmarks.md#raw-ann-scalar-core-phase-2)). It is still scalar and single-threaded; filtered search is not implemented yet. See the [roadmap](PLAN.md#6-phases).

## What it is

Recommendation and search systems retrieve a few hundred candidates out of millions in milliseconds, and real queries come with conditions such as "in stock", "this category" or "allowed in this region". hopfilter is an open-source retrieval stack built around that problem:

| Part | Stack | Role |
|---|---|---|
| **Two-tower model** | PyTorch | Learns user and item vectors with in-batch negatives and **logQ correction** for popularity bias |
| **HNSW index** | C++20 + pybind11 | Layered proximity graph, written from the Malkov and Yashunin paper with no code from other libraries |
| **Filtered search** | C++20 | Evaluates filters *during* graph traversal, adapts the search width to selectivity, and falls back to exact search when only a handful of items match |

### Why filtered search is the focus

| Approach | Breaks when |
|---|---|
| Post-filter (fetch top-k, then drop) | The filter is strict: most results get dropped, so fewer than `k` come back |
| Pre-filter (search only matching items) | The graph loses connectivity and search degrades to a slow scan |
| **hopfilter: in-graph + adaptive** | Non-matching nodes still route the search but never enter the results; the strategy is chosen per query from estimated selectivity |

The aim is to be competitive with FAISS and hnswlib on recall, and measurably better on filtered queries at specific selectivities. Results will be published in [docs/benchmarks.md](docs/benchmarks.md), including where hopfilter trails.

## Quick start (from source)

Requirements: [uv](https://docs.astral.sh/uv/) and a C++20 compiler (Visual Studio 2022+ on Windows, GCC 11+ or Clang 16+ on Linux, Xcode CLT on macOS).

```bash
git clone https://github.com/imDharaneesh/hopfilter-ann.git
cd hopfilter-ann
uv sync
uv run hopfilter --version
uv run pytest
```

`uv sync` builds the C++ extension and installs pinned dev tools. Details are in [docs/development.md](docs/development.md).

Once v0.1.0 is released:

```bash
pip install hopfilter-ann
```

## Roadmap

| Phase | Scope | Status |
|---|---|---|
| 0 | Foundation: build, packaging, CI, docs | Done |
| 1 | Two-tower baseline on MovieLens, logQ ablation | Done: logQ recall@50 0.109 vs 0.039 without |
| 2 | HNSW core, about 0.95 recall@10 on SIFT1M | Done: 0.964 at efSearch 64 |
| 3 | SIMD kernels, memory layout, parallel build | Next |
| 4 | Filtered search: post, pre, in-graph, adaptive | Planned |
| 5 | Pipeline, CLI, reproducible benchmarks vs FAISS and hnswlib | Planned |
| 6 | Wheels on PyPI, docs site, v0.1.0 | Planned |

## Repository layout

```
core/        C++20 index library (hopfilter_core)
bindings/    pybind11 extension (hopfilter._core)
python/      Python package (hopfilter)
tests/       GoogleTest and pytest suites
benchmarks/  configs, runners, results, plots
docs/        design, development, benchmarks (MkDocs)
```

## Documentation

- [PLAN.md](PLAN.md): scope, working rules, phases and exit criteria
- [docs/design.md](docs/design.md): architecture, class diagrams, contracts
- [docs/development.md](docs/development.md): setup, build, test, release
- [docs/benchmarks.md](docs/benchmarks.md): methodology and results
- [CONTRIBUTING.md](CONTRIBUTING.md): workflow, code rules, commit format

## References

- Y. A. Malkov and D. A. Yashunin. *Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs.* IEEE TPAMI, 2018. arXiv:1603.09320.
- X. Yi et al. *Sampling-bias-corrected neural modeling for large corpus item recommendations.* RecSys 2019.

## License

[Apache License 2.0](LICENSE)
