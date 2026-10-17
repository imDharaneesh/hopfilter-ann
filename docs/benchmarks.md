# Benchmarks

!!! note "Status"
    The [logQ ablation](#logq-ablation-phase-1) has results. The ANN and filtered-search tables are still templates and are filled in from Phase 2 onward, each with the exact command and commit that produced it. A number without a reproducible command is not published.

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
