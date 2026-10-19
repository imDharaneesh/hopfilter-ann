"""Sweep filter selectivity on an ANN-Benchmarks dataset for every filtered strategy.

Each item gets two synthetic numeric attributes. ``random`` is uniform noise, so the
matching set is spread evenly through the vector space. ``clustered`` is the item's
normalized rank along a fixed random direction, so the matching set is one contiguous
slab of the space and most queries sit far from it. A filter ``attribute <= s`` then
matches a share ``s`` of the items. For every attribute, selectivity, strategy and
efSearch the runner records recall@k against exact filtered ground truth, all-thread
QPS (median of several repeats) and the mean number of results returned, which exposes
post-filter shortfalls. Adaptive rows also record how many queries went to each strategy.

Usage:
    uv run python benchmarks/runners/filter_sweep.py --config benchmarks/configs/filter_sift1m.yaml
"""

import argparse
import json
import platform
import statistics
import subprocess
import time
from collections import Counter
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt
import psutil

from hopfilter import Filter, HnswIndex, _core
from hopfilter.config import load_config
from hopfilter.data.ann_benchmarks import AnnBenchmarkDataset, load_ann_benchmark
from hopfilter.data.download import DATASETS, DatasetDownloader, HttpFetcher

GROUND_TRUTH_BLOCK = 50_000
EXACT_STRATEGY = "pre_filter"

type FloatMatrix = npt.NDArray[np.float32]
type IdMatrix = npt.NDArray[np.int64]
type Attribute = Literal["random", "clustered"]


@dataclass(frozen=True)
class FilterSweepConfig:
    dataset: Literal["sift-128-euclidean", "glove-100-angular"]
    index_cache: str
    results_path: str
    data_dir: str = "data"
    k: int = 10
    queries: int = 1000
    selectivities: tuple[float, ...] = (1.0, 0.5, 0.1, 0.01, 0.001)
    attributes: tuple[Attribute, ...] = ("random", "clustered")
    strategies: tuple[str, ...] = ("post_filter", "in_graph", "pre_filter", "adaptive")
    ef_search: tuple[int, ...] = (10, 16, 32, 64, 128, 256, 512)
    repeats: int = 3
    warmup_queries: int = 100
    time_budget_seconds: float = 30.0
    search_threads: int = 0
    seed: int = 7


@dataclass(frozen=True)
class SweepRow:
    attribute: str
    selectivity: float
    matching: int
    strategy: str
    ef_search: int
    recall: float
    mean_results: float
    qps: dict[str, float]
    plan: str
    plan_counts: dict[str, int]


def load_dataset(config: FilterSweepConfig) -> AnnBenchmarkDataset:
    spec = DATASETS[config.dataset]
    dataset_dir = DatasetDownloader(HttpFetcher(), Path(config.data_dir)).download(spec)
    return load_ann_benchmark(dataset_dir / spec.file_name)


def attribute_values(
    attribute: Attribute, train: FloatMatrix, seed: int
) -> npt.NDArray[np.float64]:
    generator = np.random.default_rng(seed)
    if attribute == "random":
        return generator.random(len(train))
    direction = generator.standard_normal(train.shape[1])
    ranks = np.argsort(np.argsort(train @ direction, kind="stable"), kind="stable")
    return ranks.astype(np.float64) / len(train)


def matching_mask(values: npt.NDArray[np.float64], selectivity: float) -> npt.NDArray[np.bool_]:
    threshold = np.quantile(values, selectivity) if selectivity < 1.0 else np.inf
    mask: npt.NDArray[np.bool_] = values <= threshold
    return mask


def exact_filtered_neighbors(
    train: FloatMatrix, queries: FloatMatrix, allowed: npt.NDArray[np.int64], k: int
) -> IdMatrix:
    best_ids = np.full((len(queries), k), -1, dtype=np.int64)
    best_distances = np.full((len(queries), k), np.inf, dtype=np.float32)
    query_norms = (queries * queries).sum(axis=1, keepdims=True)
    for start in range(0, len(allowed), GROUND_TRUTH_BLOCK):
        block = allowed[start : start + GROUND_TRUTH_BLOCK]
        vectors = train[block]
        distances = query_norms - 2.0 * (queries @ vectors.T) + (vectors * vectors).sum(axis=1)
        ids = np.broadcast_to(block, distances.shape)
        merged_distances = np.concatenate([best_distances, distances], axis=1)
        merged_ids = np.concatenate([best_ids, ids], axis=1)
        keep = np.argsort(merged_distances, axis=1, kind="stable")[:, :k]
        best_distances = np.take_along_axis(merged_distances, keep, axis=1)
        best_ids = np.take_along_axis(merged_ids, keep, axis=1)
    return best_ids


def recall_at_k(found: IdMatrix, truth: IdMatrix, k: int) -> float:
    hits = [
        np.intersect1d(row[row >= 0], expected).size
        for row, expected in zip(found, truth, strict=True)
    ]
    return float(np.mean(hits)) / k


@dataclass(frozen=True)
class SelectivityCase:
    attribute: str
    selectivity: float
    filter: Filter
    truth: IdMatrix
    matching: int


@dataclass(frozen=True)
class SearchCall:
    k: int
    ef: int
    threads: int
    filter: Filter
    strategy: str

    def run(self, index: HnswIndex, queries: FloatMatrix) -> IdMatrix:
        ids, _ = index.search(
            queries,
            self.k,
            ef=self.ef,
            threads=self.threads,
            filter=self.filter,
            strategy=self.strategy,
        )
        return ids


def timed_qps(
    index: HnswIndex, queries: FloatMatrix, call: SearchCall, config: FilterSweepConfig
) -> tuple[IdMatrix, dict[str, float]]:
    rates: list[float] = []
    found: IdMatrix | None = None
    while len(rates) < config.repeats:
        started = time.perf_counter()
        found = call.run(index, queries)
        elapsed = time.perf_counter() - started
        rates.append(len(queries) / elapsed)
        if elapsed * config.repeats > config.time_budget_seconds:
            break
    assert found is not None
    qps = {"median": statistics.median(rates), "min": min(rates), "max": max(rates)}
    return found, qps | {"runs": float(len(rates))}


def ef_values(config: FilterSweepConfig, strategy: str) -> tuple[int, ...]:
    return config.ef_search[:1] if strategy == EXACT_STRATEGY else config.ef_search


def measure(
    index: HnswIndex,
    queries: FloatMatrix,
    case: SelectivityCase,
    config: FilterSweepConfig,
    run: tuple[str, int],
) -> SweepRow:
    strategy, ef = run
    counts = (
        dict(Counter(index.plan_queries(queries, case.filter, ef)))
        if strategy == "adaptive"
        else {strategy: len(queries)}
    )
    call = SearchCall(config.k, ef, config.search_threads, case.filter, strategy)
    call.run(index, queries[: config.warmup_queries])
    found, qps = timed_qps(index, queries, call, config)
    return SweepRow(
        attribute=case.attribute,
        selectivity=case.selectivity,
        matching=case.matching,
        strategy=strategy,
        ef_search=ef,
        recall=recall_at_k(found, case.truth, config.k),
        mean_results=float((found >= 0).sum(axis=1).mean()),
        qps=qps,
        plan=max(counts, key=counts.__getitem__),
        plan_counts=counts,
    )


def run_attribute(
    index: HnswIndex, dataset: AnnBenchmarkDataset, config: FilterSweepConfig, attribute: Attribute
) -> list[SweepRow]:
    values = attribute_values(attribute, dataset.train, config.seed)
    index.set_attribute(attribute, values)
    queries = dataset.test[: config.queries]
    rows = []
    for selectivity in config.selectivities:
        mask = matching_mask(values, selectivity)
        allowed = np.flatnonzero(mask).astype(np.int64)
        case = SelectivityCase(
            attribute=attribute,
            selectivity=selectivity,
            filter=Filter.between(attribute, 0.0, float(values[mask].max())),
            truth=exact_filtered_neighbors(dataset.train, queries, allowed, config.k),
            matching=len(allowed),
        )
        for strategy in config.strategies:
            for ef in ef_values(config, strategy):
                rows.append(measure(index, queries, case, config, (strategy, ef)))
                print(report(rows[-1], config.k), flush=True)
    return rows


def environment(config: FilterSweepConfig) -> dict[str, object]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()
    return {
        "commit": commit,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpus": psutil.cpu_count(logical=True),
        "memory_gib": round(psutil.virtual_memory().total / (1 << 30), 1),
        "search_threads": config.search_threads,
        "simd": _core.simd_kernels(),
    }


def report(row: SweepRow, k: int) -> str:
    return (
        f"  {row.attribute:>9} s={row.selectivity:<6} {row.strategy:>11} ef={row.ef_search:>4}"
        f"  recall@{k}={row.recall:.4f}  results={row.mean_results:5.2f}"
        f"  qps={row.qps['median']:9.0f}  plans={row.plan_counts}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="override results_path from the config")
    arguments = parser.parse_args()
    config = load_config(arguments.config, FilterSweepConfig)
    if arguments.output is not None:
        config = replace(config, results_path=str(arguments.output))
    dataset = load_dataset(config)
    index = HnswIndex.load(Path(config.index_cache))
    print(f"{config.dataset}: {len(index):,} vectors, kernels {_core.simd_kernels()}", flush=True)
    rows = []
    for attribute in config.attributes:
        rows.extend(run_attribute(index, dataset, config, attribute))
    results = {
        "config": asdict(config),
        "environment": environment(config),
        "sweep": [asdict(row) for row in rows],
    }
    output = Path(config.results_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=4) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
