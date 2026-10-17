# Benchmarks

!!! note "No results yet"
    The tables below are templates. Numbers are filled in from Phase 2 onward, each with the exact command and commit that produced it. A number without a reproducible command is not published.

## Principles

1. **Same harness for every library.** FAISS, hnswlib and hopfilter all run through the same `AnnIndex` adapter and `BenchmarkRunner`, so timing, warm-up and recall code are shared ([design](design.md#102-index-evaluation-and-pipeline-planned-phases-2-and-5)).
2. **Compare at matched recall.** QPS means nothing without the recall it was achieved at. Every library gets a parameter sweep, and comparisons are read off the recall-QPS curves.
3. **Report where we lose.** Results we trail on are published alongside the ones we win.
4. **Raw data is committed.** Each run writes a JSON file under `benchmarks/results/`. Plots are generated from those files by a committed script.

## Datasets

| Dataset | Vectors | Dim | Metric | Source | Used for |
|---|---|---|---|---|---|
| SIFT1M | 1,000,000 | 128 | L2 | ANN-Benchmarks HDF5 | Raw ANN recall and QPS |
| GloVe-100 | 1,183,514 | 100 | Angular | ANN-Benchmarks HDF5 | Raw ANN, cosine path |
| GIST1M | 1,000,000 | 960 | L2 | ANN-Benchmarks HDF5 | High-dimension stress |
| MovieLens-1M / 25M | 3.7k / 62k items | model-defined | Inner product | GroupLens | Full pipeline, genre/year filters |
| Amazon Reviews 2023 | category subsets | model-defined | Inner product | McAuley Lab | Full pipeline, category filters |

Datasets are downloaded by scripts into `data/` and never committed or redistributed.

## Metrics

| Metric | Definition |
|---|---|
| recall@k | `|returned ∩ true top-k| / k`, averaged over queries. Ground truth is exact brute force with the same filter |
| QPS | Queries per second over the full query set, reported both single-threaded and with all cores |
| p50 / p99 latency | Per-query wall-clock latency percentiles, single-threaded |
| Build time | Wall-clock time to insert every vector |
| Peak memory | Peak resident set size during build and search |
| Result count | Mean number of results returned (filtered runs only; exposes post-filter shortfalls) |

## Protocol

- **Machine.** Same machine for all libraries, frequency scaling disabled where possible, no other heavy processes.
- **Threads.** Pinned with `OMP_NUM_THREADS` and each library's own thread setting. Single-threaded runs use one pinned core.
- **Warm-up.** One full pass over the query set before measuring.
- **Repeats.** 5 measured repeats per configuration, reported as median with min-max spread.
- **Seeds.** Fixed seeds for index construction and query sampling, recorded in the result file.
- **Parameter sweep.**

    | Library | Build parameters | Search parameter |
    |---|---|---|
    | hopfilter | `M` in {8, 16, 32, 48}, `efConstruction` in {100, 200, 400} | `efSearch` in {10 ... 800} |
    | hnswlib | same `M`, `ef_construction` grid | `ef` in {10 ... 800} |
    | FAISS `IndexHNSWFlat` | same `M`, `efConstruction` grid | `efSearch` in {10 ... 800} |

## Filtered search sweep

Each item gets synthetic attributes with controlled selectivity, so that a filter matches exactly 100%, 50%, 10%, 1% or 0.1% of items. Both of these are measured:

- **Uncorrelated filters**, where matching items are scattered at random relative to the query.
- **Correlated filters**, built from real attributes (genre, category), where matching items cluster in vector space.

Strategies compared at each selectivity: `PostFilter`, `PreFilter`, `InGraph`, `Adaptive`, plus FAISS and hnswlib with their own filtering support where available.

## Hardware record

Every result file and every published table states:

| Field | Example |
|---|---|
| CPU | model, cores/threads, base/boost clock |
| SIMD | AVX2/FMA/AVX-512 availability and the kernel actually used |
| Memory | size and speed |
| OS | name, version, kernel |
| Compiler | name, version, flags |
| Commit | `git rev-parse HEAD` |

## Results

### Raw ANN (Phase 2-3)

| Dataset | Library | Params | recall@10 | QPS (1 thread) | QPS (all) | p99 (ms) | Build (s) | Memory (MB) |
|---|---|---|---|---|---|---|---|---|
| SIFT1M | hopfilter | | | | | | | |
| SIFT1M | hnswlib | | | | | | | |
| SIFT1M | FAISS HNSW | | | | | | | |

### Optimization log (Phase 3)

Each optimization is measured on its own against the previous row, on SIFT1M with fixed parameters.

| Change | QPS | Speedup vs previous | Commit |
|---|---|---|---|
| Scalar baseline | | 1.00x | |
| AVX2/FMA kernels | | | |
| Aligned level-0 layout | | | |
| Software prefetch | | | |

### Filtered search (Phase 4)

| Selectivity | Strategy | recall@10 | QPS | Mean results returned |
|---|---|---|---|---|
| 100% | | | | |
| 50% | | | | |
| 10% | | | | |
| 1% | | | | |
| 0.1% | | | | |

## Reproducing

From Phase 5, every published number is reproducible with:

```bash
hopfilter bench --config benchmarks/configs/<name>.yaml
```
