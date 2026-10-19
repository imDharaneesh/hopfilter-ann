# hopfilter-ann

**Two-tower retrieval with a from-scratch HNSW index and filter-aware search.**

hopfilter is an open-source retrieval stack in three parts:

| Part | Language | What it does |
|---|---|---|
| Two-tower model | Python, PyTorch | Learns user and item vectors with in-batch negatives and logQ correction |
| HNSW index | C++20 | Layered proximity graph written from the Malkov and Yashunin paper |
| Filtered search | C++20 | Applies attribute filters *during* graph traversal and adapts to filter selectivity |

!!! warning "Pre-alpha"
    Phases 0-4 are done: foundation, the two-tower baseline ([logQ ablation](benchmarks.md#logq-ablation-phase-1)), the HNSW core, its performance work ([comparison with hnswlib and FAISS](benchmarks.md#raw-ann-comparison-phase-3)) and filtered search ([selectivity sweep](benchmarks.md#filtered-search-phase-4)). The end-to-end pipeline comes next. See the [roadmap](https://github.com/imDharaneesh/hopfilter-ann/blob/master/PLAN.md#6-phases) for what comes next.

## Why filtered search

Production queries rarely ask only for the nearest vectors. They ask for the nearest vectors **that are in stock, in category X and allowed in region Y**. The two common approaches both break at the extremes:

- **Post-filtering** retrieves the top-k and then drops non-matching items. With a 1% filter, almost nothing survives.
- **Pre-filtering** restricts the search to matching items. The proximity graph loses connectivity and search falls back to a slow scan.

hopfilter keeps non-matching nodes as routing waypoints while admitting only matching nodes to the result set. It estimates filter selectivity per query and switches to exact search when only a handful of items match. The [design](design.md#56-search-phase-2-unfiltered-phase-4-filtered) covers the details, and the [benchmarks](benchmarks.md#filtered-search-phase-4) show where this wins and where it does not.

## Where to go next

- [Development](development.md): set up the environment, build, test.
- [Design](design.md): modules, interfaces, contracts.
- [Benchmarks](benchmarks.md): methodology and results.
