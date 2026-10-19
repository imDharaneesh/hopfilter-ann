# Benchmarks

!!! note "Status"
    The [logQ ablation](#logq-ablation-phase-1), the [scalar HNSW core](#raw-ann-scalar-core-phase-2), the [optimization log](#optimization-log-phase-3) and the [comparison with hnswlib and FAISS](#raw-ann-comparison-phase-3) have results. The filtered-search tables are still templates, filled in from Phase 4 onward, each with the exact command and commit that produced it. A number without a reproducible command is not published.

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

### logQ ablation (Phase 1)

Does logQ correction help a two-tower retriever trained with in-batch negatives? Same data, model, optimizer and seeds; only the loss changes.

**Setup**

| | |
|---|---|
| Data | MovieLens-1M, ratings ≥ 4 as positives (575,281) |
| Split | Global temporal: train before the 80% time quantile, validation 80-90%, test after 90%. Held-out rows with a user or item unseen in train are dropped (58,088) |
| Sizes after split | 460,224 train, 13,632 validation, 43,337 test positives, 987 test users, 5,378 users and 3,468 items in train |
| Head / tail | Head = top 20% of items by train count (694 items, 72.9% of train interactions). 64.6% of test positives are head items |
| Model | ID embedding 64 → MLP (128) → 64, L2-normalized; item tower adds mean genre embedding |
| Training | AdamW, lr 1e-3, batch 1024, 20 epochs, temperature 0.05, logQ estimator α = 0.01 |
| Evaluation | Exact top-k over all items, excluding the user's train items; interaction-level recall (see [design](design.md#evaluation)) |
| Repeats | Seeds 0, 1, 2; mean ± standard deviation |
| Hardware | AMD Ryzen (Family 25 Model 68), 8 torch threads, CPU only, Windows 11, Python 3.14.3, torch 2.14.1+cpu, ~2.5 min per run |
| Commit | `89e3f45` (recorded in the results file) |

**Test recall (mean ± std over 3 seeds)**

| Model | recall@10 | recall@50 | recall@100 |
|---|---|---|---|
| In-batch softmax | 0.0078 ± 0.0007 | 0.0394 ± 0.0021 | 0.0782 ± 0.0025 |
| logQ corrected | 0.0258 ± 0.0005 | 0.1089 ± 0.0011 | 0.1968 ± 0.0009 |
| Most popular (baseline) | 0.0348 | 0.1318 | 0.2207 |

| Model | head recall@50 | tail recall@50 | head recall@100 | tail recall@100 |
|---|---|---|---|---|
| In-batch softmax | 0.0342 | **0.0488** | 0.0706 | **0.0922** |
| logQ corrected | 0.1506 | 0.0327 | 0.2682 | 0.0667 |
| Most popular (baseline) | **0.2041** | 0.0000 | **0.3418** | 0.0000 |

![logQ ablation: test recall@50 by segment and validation recall per epoch](https://raw.githubusercontent.com/imDharaneesh/hopfilter-ann/master/benchmarks/plots/logq_ablation_ml1m.png)

**What this shows**

1. **logQ correction is a large win overall:** recall@50 rises from 0.039 to 0.109 (2.8×), with tight spread across seeds.
2. **It does not help tail items; it hurts them.** Tail recall@50 falls from 0.049 to 0.033, while head recall@50 rises from 0.034 to 0.151. Without correction, popular items are penalized every time they appear as an in-batch negative, so the uncorrected model is pushed *away* from the head. That inflates its tail share but costs far more on head items. The original plan expected tail gains, and this measurement contradicts it.
3. **Neither model beats "most popular" overall yet** (0.109 vs 0.132 at k = 50). On a temporal split of MovieLens-1M, two-thirds of test positives are head items, so popularity is a strong baseline. The logQ model's advantage is that it retrieves tail items at all (popularity's top-100 never contains one).
4. **The logQ model peaks early.** Validation recall@50 is best at epoch 2 (≈ 0.138) and declines to ≈ 0.113 by epoch 20, so the test numbers above come from a model past its best. The uncorrected model plateaus at ≈ 0.038-0.039 from about epoch 10 on.

**Not done yet** (tracked as follow-ups, not silently tuned away): early stopping on validation recall, a temperature and regularization sweep, mixing in uniformly sampled negatives, and a repeat on MovieLens-25M.

**Reproduce**

```bash
uv run hopfilter download movielens-1m
uv run python benchmarks/runners/logq_ablation.py --config benchmarks/configs/logq_ablation_ml1m.yaml
uv run python benchmarks/plots/plot_logq_ablation.py --results benchmarks/results/logq_ablation_ml1m.json --output benchmarks/plots/logq_ablation_ml1m.png
```

Raw per-run results, including per-epoch losses and validation curves, are in `benchmarks/results/logq_ablation_ml1m.json`.

### Raw ANN: scalar core (Phase 2)

The first measurement of the from-scratch HNSW core: scalar distance kernels, single-threaded build and search, no comparison libraries yet. This row is the baseline that Phase 3's SIMD kernels, memory layout and parallel build are measured against.

**Setup:** SIFT1M (1,000,000 x 128, L2; 10,000 queries with ground truth from ANN-Benchmarks), M = 16, efConstruction = 200, seed 42. AMD Ryzen (Family 25 Model 68), 16 logical CPUs, 15.8 GiB RAM, Windows 11, MSVC Release build, Python 3.14.3, commit `61219e3`. Idle machine, one thread.

| Metric | Value |
|---|---|
| Build time | 1,309 s (about 760 inserts/s) |
| Saved index size | 577 MiB (vectors 488 MiB + graph) |
| Process RSS after build | 1.2 GiB (includes the 0.5 GiB dataset arrays held by the runner) |

| efSearch | recall@10 | QPS (1 thread) | p50 (ms) | p99 (ms) |
|---|---|---|---|---|
| 10 | 0.710 | 5,382 | 0.195 | 0.361 |
| 16 | 0.802 | 4,519 | 0.215 | 0.481 |
| 24 | 0.866 | 3,743 | 0.269 | 0.526 |
| 32 | 0.904 | 3,065 | 0.333 | 0.611 |
| 48 | 0.944 | 2,275 | 0.456 | 0.842 |
| **64** | **0.964** | **1,767** | **0.638** | **1.021** |
| 96 | 0.982 | 1,302 | 0.787 | 1.263 |
| 128 | 0.990 | 1,052 | 1.007 | 1.570 |
| 192 | 0.995 | 737 | 1.390 | 2.230 |
| 256 | 0.997 | 585 | 1.854 | 2.886 |
| 384 | 0.999 | 410 | 2.515 | 3.646 |
| 512 | 0.999 | 330 | 3.227 | 5.005 |

The Phase 2 target of about 0.95 recall@10 is reached at efSearch 48-64. QPS is from one batch call over all 10,000 queries; p50 and p99 come from 1,000 individual calls and include Python call overhead.

An earlier run at the same commit was discarded. It overlapped with compiler and clang-tidy jobs, inflating build time to 1,900 s, and its efSearch-512 batch stalled, reporting 1 QPS against a 4 ms median latency. Recall was identical in both runs, as expected from a deterministic build.

**Reproduce**

``bash
uv run python benchmarks/runners/ann_recall.py --config benchmarks/configs/sift1m.yaml
``

**GloVe-100 (angular), recall only.** 1,183,514 x 100 vectors, cosine metric, M = 24, efConstruction = 300, same machine and commit series. Recall depends only on the graph, so it is reported. This run's QPS and build time are **not**: compiler jobs ran on the same CPU for its whole duration. They are re-measured in Phase 3.

| efSearch | 10 | 32 | 64 | 128 | 256 | 512 | 768 | 1024 |
|---|---|---|---|---|---|---|---|---|
| recall@10 | 0.547 | 0.733 | 0.814 | 0.875 | 0.923 | 0.957 | 0.972 | 0.980 |

GloVe is far harder than SIFT for graph indexes, because its angular neighborhoods are less well separated. Reaching 0.95 takes an efSearch of about 500 rather than 50.

### Optimization log (Phase 3)

Every optimization was measured on its own against the commit before it, on the same saved SIFT1M graph (M = 16, efConstruction = 200), at efSearch 64, where recall@10 stays at 0.9636 throughout.

**Python runner, one commit per row.** Each commit was built in its own worktree; the median of 5 single-thread passes over 10,000 queries, all measured in one session:

| Change | Commit | QPS (1 thread) | vs previous |
|---|---|---|---|
| Scalar kernels (Phase 2 core) | `d3cce9f` | 1,833 | |
| AVX2/FMA kernels, runtime dispatch | `1680158` | 2,266 | +24% |
| Prefetch the next neighbor's first cache line | `c0567f9` | 2,712 | +20% |
| Lock-capable search path for parallel build | `b02565d` | 2,657 | -2% (noise) |

**C++ harness, paired A/B.** `hopfilter_search_bench` runs the same queries with no Python in the loop, and the two builds alternate run by run so machine drift cancels out:

| Change | Commit | QPS A -> B | vs previous | Kept |
|---|---|---|---|---|
| Prefetch every cache line of the next neighbor's vector | `cb6c717` | 2,727 -> 3,493 | +28% (B ahead in 3/3 rounds) | yes |
| Prefetch visited stamps and the next candidate's neighbor list | `49c7857` | 3,188 -> 3,539 | +11% (B ahead in 5/5 rounds) | yes |
| 16-bit visited stamps (2 MB instead of 4 MB) | none | 3,473 -> 3,507 | +1%, within noise | no, reverted |

The search is bound by memory latency more than by arithmetic. AVX2 alone gave +24%. The three prefetch steps, which load what the next step will touch, gave +20%, +28% and +11%: together worth far more than the SIMD kernels. Forcing the scalar kernels on the final build tells the same story, since SIMD is then worth about 20%.

**Parallel build** (SIFT1M, AVX2 kernels):

| Threads | Build time | Speedup | recall@10 at efSearch 64 |
|---|---|---|---|
| 1 | 958 s (Phase 2 scalar: 1,309 s) | 1.0x | 0.9636 |
| 16 | 143 s | 6.7x | 0.9634 |

The parallel build gives up determinism (edges depend on thread timing) but not quality. The first parallel version left up to 32 of 4,000 nodes unreachable; [the merge fix](design.md#7-concurrency-contract) brought that to 0-1.

**Profiling note.** No sampling profiler (VTune, WPA, perf) was available on this machine. Instead of profiles, each change was justified by an isolated paired measurement.

### Raw ANN: comparison (Phase 3)

**Setup:** SIFT1M, M = 16, efConstruction = 200, seed 42, the same 16 threads for every build. Each library runs in its own process. QPS is the median of 3 passes over 10,000 queries, interpolated (log-linear) to matched recall. AMD Ryzen (Family 25 Model 68), 16 logical CPUs, 15.8 GiB, Windows 11, commit `49c7857`.

| Library | Build | Memory growth | QPS @0.95, 1 thread | QPS @0.95, 16 threads | QPS @0.99, 1 thread | QPS @0.99, 16 threads |
|---|---|---|---|---|---|---|
| hopfilter (AVX2) | **107 s** | 710 MiB | 5,654 | **27,137** | 2,611 | **12,183** |
| hnswlib 0.8.0 (AVX2) | 113 s | 814 MiB | **5,991** | 23,338 | **2,798** | 10,823 |
| FAISS 1.15.1 `IndexHNSWFlat` | 128 s | **629 MiB** | 5,919 | 25,332 | 2,641 | 11,584 |
| hopfilter (scalar kernels) | 98 s | 710 MiB | 4,695 | 29,474 | 2,221 | 12,941 |

![Recall vs QPS on SIFT1M for hopfilter, hnswlib and FAISS](https://raw.githubusercontent.com/imDharaneesh/hopfilter-ann/master/benchmarks/plots/compare_sift1m.png)

**Where hopfilter wins.** It has the fastest build, and it leads all three libraries once all 16 threads search: +16% over hnswlib and +7% over FAISS at recall 0.95, and +13% / +5% at 0.99.

**Where it loses.** Single-threaded, it trails hnswlib by 6% at recall 0.95 and 7% at 0.99, and FAISS by 5% and 1%. The remaining gap is most likely layout: hopfilter keeps vectors and level-0 links in separate arrays, so expanding a node touches two memory regions where an interleaved layout would touch one. It also pays two indirect calls per distance (the metric strategy, then the kernel dispatch). Both are deliberate design choices for clarity and extensibility, and this gap is their measured cost.

**Caveats, stated plainly:**

- The FAISS wheel on PyPI for Windows runs without SIMD (`faiss.SIMDConfig.get_level_name()` returns `NONE`); on Linux the same package loads an AVX2 build. FAISS without SIMD still matches hopfilter with AVX2 single-threaded, so FAISS's search loop is the most efficient of the three. hopfilter with scalar kernels is included as a like-for-like row against it.
- hnswlib 0.8.0 ships only as source and was compiled with `CL=/arch:AVX2`. Without it, MSVC builds hnswlib with SSE only.
- At 16 threads the scalar build is slightly faster than the AVX2 build. With 16 threads the search appears to be memory-bandwidth bound, so the arithmetic saved by AVX2 no longer matters; the difference is within the run-to-run spread.
- Memory growth is the process RSS increase during the build, which includes allocator slack; the saved hopfilter index is 577 MiB.
- An earlier comparison ran all libraries in one process and was discarded: memory carried over between libraries, and thermal state made later runs slower.

**Reproduce**

```bash
uv sync --group compare
uv run python benchmarks/runners/ann_compare.py --config benchmarks/configs/compare_sift1m.yaml
uv run python benchmarks/plots/plot_ann_compare.py --results benchmarks/results/compare_sift1m.json --output benchmarks/plots/compare_sift1m.png
```

On Windows, set `CL=/arch:AVX2` before `uv sync` so hnswlib is compiled with AVX2. The optimization-log configs are `benchmarks/configs/sift1m_optlog*.yaml` and `sift1m_build_*.yaml`. The C++ harness is built with `-DHOPFILTER_BUILD_BENCHMARKS=ON` and fed by `benchmarks/runners/export_ann_queries.py`.

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
