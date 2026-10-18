"""Build an HNSW index on an ANN-Benchmarks dataset and sweep efSearch.

Records recall@k against the dataset's ground truth, single-thread and all-thread QPS
(median of several repeats), per-query p50/p99 latency, build time and resident memory.
Works against older builds of hopfilter too: options a build does not support are
recorded as unavailable instead of failing.

Usage:
    uv run python benchmarks/runners/ann_recall.py --config benchmarks/configs/sift1m.yaml
"""

import argparse
import json
import platform
import statistics
import subprocess
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt
import psutil

from hopfilter import HnswIndex, _core
from hopfilter.config import load_config
from hopfilter.data.ann_benchmarks import AnnBenchmarkDataset, load_ann_benchmark
from hopfilter.data.download import DATASETS, DatasetDownloader, HttpFetcher

INSERT_CHUNK = 50_000
BYTES_PER_MEBIBYTE = 1 << 20
NANOSECONDS_PER_MILLISECOND = 1e6
LATENCY_PERCENTILES = (50, 99)
UNAVAILABLE = "unavailable in this build"

type SearchResult = tuple[npt.NDArray[np.int64], npt.NDArray[np.float32]]


@dataclass(frozen=True)
class AnnRecallConfig:
    dataset: Literal["sift-128-euclidean", "glove-100-angular"]
    results_path: str
    index_cache: str
    data_dir: str = "data"
    max_neighbors: int = 16
    ef_construction: int = 200
    seed: int = 42
    k: int = 10
    ef_search: tuple[int, ...] = (10, 32, 64, 128, 256)
    latency_queries: int = 1000
    repeats: int = 5
    build_threads: int = 1
    search_threads: int = 0
    kernels: Literal["default", "scalar", "avx2"] = "default"


@dataclass(frozen=True)
class SweepRow:
    ef_search: int
    recall: float
    qps_single_thread: dict[str, float]
    qps_all_threads: dict[str, float]
    p50_ms: float
    p99_ms: float


@dataclass(frozen=True)
class BuildReport:
    seconds: float | None
    loaded_from_cache: bool
    threads: int | str
    rss_mebibytes: float


def load_dataset(config: AnnRecallConfig) -> AnnBenchmarkDataset:
    spec = DATASETS[config.dataset]
    dataset_dir = DatasetDownloader(HttpFetcher(), Path(config.data_dir)).download(spec)
    return load_ann_benchmark(dataset_dir / spec.file_name)


def with_threads[Result](call: Callable[..., Result], threads: int, *args: object) -> Result:
    try:
        return call(*args, threads=threads)
    except TypeError:
        return call(*args)


def build_or_load(
    config: AnnRecallConfig, dataset: AnnBenchmarkDataset
) -> tuple[HnswIndex, BuildReport]:
    cache = Path(config.index_cache)
    if cache.is_file():
        return HnswIndex.load(cache), BuildReport(None, True, "n/a", resident_mebibytes())
    index = HnswIndex(
        dataset.train.shape[1],
        len(dataset.train),
        metric=dataset.metric,
        max_neighbors=config.max_neighbors,
        ef_construction=config.ef_construction,
        seed=config.seed,
    )
    started = time.perf_counter()
    for start in range(0, len(dataset.train), INSERT_CHUNK):
        with_threads(index.add, config.build_threads, dataset.train[start : start + INSERT_CHUNK])
        print(f"  inserted {len(index):>9,} ({time.perf_counter() - started:7.1f}s)", flush=True)
    seconds = time.perf_counter() - started
    cache.parent.mkdir(parents=True, exist_ok=True)
    index.save(cache)
    return index, BuildReport(seconds, False, config.build_threads, resident_mebibytes())


def resident_mebibytes() -> float:
    return float(psutil.Process().memory_info().rss) / BYTES_PER_MEBIBYTE


def recall_at_k(found: npt.NDArray[np.int64], truth: npt.NDArray[np.int64], k: int) -> float:
    hits = [
        np.intersect1d(row, expected).size
        for row, expected in zip(found, truth[:, :k], strict=True)
    ]
    return float(np.mean(hits)) / k


def timed_qps(search: Callable[[], SearchResult], queries: int, repeats: int) -> dict[str, float]:
    rates = []
    for _ in range(repeats):
        started = time.perf_counter()
        search()
        rates.append(queries / (time.perf_counter() - started))
    return {"median": statistics.median(rates), "min": min(rates), "max": max(rates)}


def latency_milliseconds(
    index: HnswIndex, queries: npt.NDArray[np.float32], k: int, ef: int
) -> list[float]:
    samples = []
    for query in queries:
        started = time.perf_counter_ns()
        with_threads(index.search, 1, query[np.newaxis, :], k, ef)
        samples.append((time.perf_counter_ns() - started) / NANOSECONDS_PER_MILLISECOND)
    return samples


def measure(
    index: HnswIndex, dataset: AnnBenchmarkDataset, config: AnnRecallConfig, ef: int
) -> SweepRow:
    queries = dataset.test
    found, _ = with_threads(index.search, 1, queries, config.k, ef)
    single = timed_qps(
        lambda: with_threads(index.search, 1, queries, config.k, ef), len(queries), config.repeats
    )
    parallel = timed_qps(
        lambda: with_threads(index.search, config.search_threads, queries, config.k, ef),
        len(queries),
        config.repeats,
    )
    latencies = latency_milliseconds(index, queries[: config.latency_queries], config.k, ef)
    p50, p99 = np.percentile(latencies, LATENCY_PERCENTILES)
    return SweepRow(
        ef_search=ef,
        recall=recall_at_k(found, dataset.neighbors, config.k),
        qps_single_thread=single,
        qps_all_threads=parallel,
        p50_ms=float(p50),
        p99_ms=float(p99),
    )


def select_kernels(kernels: str) -> str:
    use = getattr(_core, "use_simd_kernels", None)
    if kernels != "default":
        if use is None:
            raise SystemExit(f"kernel set '{kernels}' requested but this build cannot switch")
        use(kernels)
    active = getattr(_core, "simd_kernels", None)
    return str(active()) if active is not None else UNAVAILABLE


def environment(config: AnnRecallConfig, kernels: str) -> dict[str, object]:
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
        "build_threads": config.build_threads,
        "search_threads": config.search_threads,
        "simd": kernels,
    }


def report(row: SweepRow, k: int) -> str:
    return (
        f"  ef={row.ef_search:>4}  recall@{k}={row.recall:.4f}"
        f"  qps(1)={row.qps_single_thread['median']:8.0f}"
        f"  qps(all)={row.qps_all_threads['median']:8.0f}"
        f"  p50={row.p50_ms:.3f}ms  p99={row.p99_ms:.3f}ms"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="override results_path from the config")
    arguments = parser.parse_args()
    config = load_config(arguments.config, AnnRecallConfig)
    if arguments.output is not None:
        config = replace(config, results_path=str(arguments.output))
    kernels = select_kernels(config.kernels)
    dataset = load_dataset(config)
    print(f"{config.dataset}: train {dataset.train.shape}, kernels {kernels}", flush=True)
    index, build = build_or_load(config, dataset)
    print(f"build: {build}", flush=True)
    sweep = []
    for ef in config.ef_search:
        sweep.append(measure(index, dataset, config, ef))
        print(report(sweep[-1], config.k), flush=True)
    results = {
        "config": asdict(config),
        "environment": environment(config, kernels),
        "build": asdict(build),
        "sweep": [asdict(row) for row in sweep],
    }
    output = Path(config.results_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=4) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
