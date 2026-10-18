"""Build an HNSW index on an ANN-Benchmarks dataset and sweep efSearch.

Records recall@k against the dataset's ground truth, single-thread QPS, per-query
p50/p99 latency, build time and resident memory.

Usage:
    uv run python benchmarks/runners/ann_recall.py --config benchmarks/configs/sift1m.yaml
"""

import argparse
import json
import platform
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt
import psutil

from hopfilter import HnswIndex
from hopfilter.config import load_config
from hopfilter.data.ann_benchmarks import AnnBenchmarkDataset, load_ann_benchmark
from hopfilter.data.download import DATASETS, DatasetDownloader, HttpFetcher

INSERT_CHUNK = 50_000
BYTES_PER_MEBIBYTE = 1 << 20
NANOSECONDS_PER_MILLISECOND = 1e6
LATENCY_PERCENTILES = (50, 99)


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


@dataclass(frozen=True)
class BuildReport:
    seconds: float | None
    loaded_from_cache: bool
    rss_mebibytes: float


def load_dataset(config: AnnRecallConfig) -> AnnBenchmarkDataset:
    spec = DATASETS[config.dataset]
    dataset_dir = DatasetDownloader(HttpFetcher(), Path(config.data_dir)).download(spec)
    return load_ann_benchmark(dataset_dir / spec.file_name)


def build_or_load(
    config: AnnRecallConfig, dataset: AnnBenchmarkDataset
) -> tuple[HnswIndex, BuildReport]:
    cache = Path(config.index_cache)
    if cache.is_file():
        index = HnswIndex.load(cache)
        return index, BuildReport(None, True, resident_mebibytes())
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
        index.add(dataset.train[start : start + INSERT_CHUNK])
        print(f"  inserted {len(index):>9,} ({time.perf_counter() - started:7.1f}s)", flush=True)
    seconds = time.perf_counter() - started
    cache.parent.mkdir(parents=True, exist_ok=True)
    index.save(cache)
    return index, BuildReport(seconds, False, resident_mebibytes())


def resident_mebibytes() -> float:
    return float(psutil.Process().memory_info().rss) / BYTES_PER_MEBIBYTE


def recall_at_k(found: npt.NDArray[np.int64], truth: npt.NDArray[np.int64], k: int) -> float:
    hits = [
        np.intersect1d(row, expected).size
        for row, expected in zip(found, truth[:, :k], strict=True)
    ]
    return float(np.mean(hits)) / k


def latency_milliseconds(
    index: HnswIndex, queries: npt.NDArray[np.float32], k: int, ef: int
) -> list[float]:
    samples = []
    for query in queries:
        started = time.perf_counter_ns()
        index.search(query[np.newaxis, :], k, ef)
        samples.append((time.perf_counter_ns() - started) / NANOSECONDS_PER_MILLISECOND)
    return samples


def measure(
    index: HnswIndex, dataset: AnnBenchmarkDataset, config: AnnRecallConfig, ef: int
) -> dict[str, float]:
    started = time.perf_counter()
    found, _ = index.search(dataset.test, config.k, ef)
    seconds = time.perf_counter() - started
    latencies = latency_milliseconds(index, dataset.test[: config.latency_queries], config.k, ef)
    p50, p99 = np.percentile(latencies, LATENCY_PERCENTILES)
    return {
        "ef_search": ef,
        "recall": recall_at_k(found, dataset.neighbors, config.k),
        "qps": len(dataset.test) / seconds,
        "p50_ms": float(p50),
        "p99_ms": float(p99),
    }


def environment() -> dict[str, object]:
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
        "threads_used": 1,
        "simd": "none (scalar kernels, phase 2)",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    config = load_config(parser.parse_args().config, AnnRecallConfig)
    dataset = load_dataset(config)
    print(f"{config.dataset}: train {dataset.train.shape}, test {dataset.test.shape}", flush=True)
    index, build = build_or_load(config, dataset)
    print(f"build: {build}", flush=True)
    sweep = []
    for ef in config.ef_search:
        sweep.append(measure(index, dataset, config, ef))
        row = sweep[-1]
        print(
            f"  ef={ef:>4}  recall@{config.k}={row['recall']:.4f}  qps={row['qps']:8.0f}"
            f"  p50={row['p50_ms']:.3f}ms  p99={row['p99_ms']:.3f}ms",
            flush=True,
        )
    results = {
        "config": asdict(config),
        "environment": environment(),
        "build": asdict(build),
        "sweep": sweep,
    }
    output = Path(config.results_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=4) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
