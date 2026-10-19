"""Compare hopfilter, hnswlib and FAISS HNSW on one ANN-Benchmarks dataset.

Every library runs in its own process, so memory growth and thermal state do not carry
over between libraries, and is built with the same M, efConstruction and thread count,
then swept over efSearch. Recall@k, single-thread and all-thread QPS (median of
repeats), build time and the resident-memory growth of the build are recorded.

Requires the ``compare`` dependency group:
    uv sync --group compare
    uv run python benchmarks/runners/ann_compare.py --config benchmarks/configs/compare_sift1m.yaml
"""

import argparse
import json
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Protocol

import faiss
import hnswlib
import numpy as np
import numpy.typing as npt
import psutil

from hopfilter import HnswIndex, _core
from hopfilter.config import load_config
from hopfilter.data.ann_benchmarks import AnnBenchmarkDataset, load_ann_benchmark
from hopfilter.data.download import DATASETS, DatasetDownloader, HttpFetcher
from hopfilter.errors import IndexNotBuiltError

BYTES_PER_MEBIBYTE = 1 << 20

type Matrix = npt.NDArray[np.float32]
type Ids = npt.NDArray[np.int64]
type LibraryName = Literal["hopfilter", "hopfilter_scalar", "hnswlib", "faiss"]


@dataclass(frozen=True)
class CompareConfig:
    dataset: Literal["sift-128-euclidean", "glove-100-angular"]
    results_path: str
    libraries: tuple[LibraryName, ...] = ("hopfilter", "hnswlib", "faiss")
    data_dir: str = "data"
    max_neighbors: int = 16
    ef_construction: int = 200
    seed: int = 42
    k: int = 10
    ef_search: tuple[int, ...] = (16, 32, 64, 128, 256)
    repeats: int = 3
    threads: int = 0


def built[Index](index: Index | None) -> Index:
    if index is None:
        raise IndexNotBuiltError("call build() before search()")
    return index


class AnnLibrary(Protocol):
    def build(self, train: Matrix, threads: int) -> None: ...

    def search(self, queries: Matrix, k: int, ef: int, threads: int) -> Ids: ...


class HopfilterLibrary:
    def __init__(self, config: CompareConfig, metric: str) -> None:
        self._config = config
        self._metric = metric
        self._index: HnswIndex | None = None

    def build(self, train: Matrix, threads: int) -> None:
        self._index = HnswIndex(
            train.shape[1],
            len(train),
            metric=self._metric,
            max_neighbors=self._config.max_neighbors,
            ef_construction=self._config.ef_construction,
            seed=self._config.seed,
        )
        self._index.add(train, threads=threads)

    def search(self, queries: Matrix, k: int, ef: int, threads: int) -> Ids:
        return built(self._index).search(queries, k, ef, threads=threads)[0]


class HnswlibLibrary:
    def __init__(self, config: CompareConfig, metric: str) -> None:
        self._config = config
        self._space = "l2" if metric == "l2" else "cosine"
        self._index: hnswlib.Index | None = None

    def build(self, train: Matrix, threads: int) -> None:
        self._index = hnswlib.Index(space=self._space, dim=train.shape[1])
        self._index.init_index(
            max_elements=len(train),
            M=self._config.max_neighbors,
            ef_construction=self._config.ef_construction,
            random_seed=self._config.seed,
        )
        self._index.add_items(train, num_threads=threads)

    def search(self, queries: Matrix, k: int, ef: int, threads: int) -> Ids:
        index = built(self._index)
        index.set_ef(ef)
        labels, _ = index.knn_query(queries, k=k, num_threads=threads)
        return np.asarray(labels, dtype=np.int64)


class FaissLibrary:
    def __init__(self, config: CompareConfig, metric: str) -> None:
        self._config = config
        self._normalize = metric == "cosine"
        self._index: faiss.IndexHNSWFlat | None = None

    def _prepared(self, vectors: Matrix) -> Matrix:
        if not self._normalize:
            return vectors
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    def build(self, train: Matrix, threads: int) -> None:
        faiss.omp_set_num_threads(threads)
        metric = faiss.METRIC_INNER_PRODUCT if self._normalize else faiss.METRIC_L2
        self._index = faiss.IndexHNSWFlat(train.shape[1], self._config.max_neighbors, metric)
        self._index.hnsw.efConstruction = self._config.ef_construction
        self._index.add(self._prepared(train))

    def search(self, queries: Matrix, k: int, ef: int, threads: int) -> Ids:
        index = built(self._index)
        faiss.omp_set_num_threads(threads)
        index.hnsw.efSearch = ef
        _, labels = index.search(self._prepared(queries), k)
        return np.asarray(labels, dtype=np.int64)


class ScalarHopfilterLibrary(HopfilterLibrary):
    def build(self, train: Matrix, threads: int) -> None:
        with scalar_kernels():
            super().build(train, threads)

    def search(self, queries: Matrix, k: int, ef: int, threads: int) -> Ids:
        with scalar_kernels():
            return super().search(queries, k, ef, threads)


@contextmanager
def scalar_kernels() -> Iterator[None]:
    previous = _core.simd_kernels()
    _core.use_simd_kernels("scalar")
    try:
        yield
    finally:
        _core.use_simd_kernels(previous)


def make_library(name: LibraryName, config: CompareConfig, metric: str) -> AnnLibrary:
    if name == "hopfilter":
        return HopfilterLibrary(config, metric)
    if name == "hopfilter_scalar":
        return ScalarHopfilterLibrary(config, metric)
    if name == "hnswlib":
        return HnswlibLibrary(config, metric)
    return FaissLibrary(config, metric)


def resident_mebibytes() -> float:
    return float(psutil.Process().memory_info().rss) / BYTES_PER_MEBIBYTE


def recall_at_k(found: Ids, truth: Ids, k: int) -> float:
    hits = [
        np.intersect1d(row, expected).size
        for row, expected in zip(found, truth[:, :k], strict=True)
    ]
    return float(np.mean(hits)) / k


def median_qps(
    library: AnnLibrary, queries: Matrix, config: CompareConfig, ef: int, threads: int
) -> float:
    rates = []
    for _ in range(config.repeats):
        started = time.perf_counter()
        library.search(queries, config.k, ef, threads)
        rates.append(len(queries) / (time.perf_counter() - started))
    return statistics.median(rates)


def run_library(
    name: LibraryName, dataset: AnnBenchmarkDataset, config: CompareConfig
) -> dict[str, object]:
    threads = config.threads or int(psutil.cpu_count(logical=True) or 1)
    library = make_library(name, config, dataset.metric)
    memory_before = resident_mebibytes()
    started = time.perf_counter()
    library.build(dataset.train, threads)
    build_seconds = time.perf_counter() - started
    memory_growth = resident_mebibytes() - memory_before
    print(f"{name}: built in {build_seconds:.1f}s, +{memory_growth:.0f} MiB", flush=True)
    sweep = []
    for ef in config.ef_search:
        found = library.search(dataset.test, config.k, ef, 1)
        row = {
            "ef_search": ef,
            "recall": recall_at_k(found, dataset.neighbors, config.k),
            "qps_single_thread": median_qps(library, dataset.test, config, ef, 1),
            "qps_all_threads": median_qps(library, dataset.test, config, ef, threads),
        }
        print(
            f"  ef={ef:>4} recall={row['recall']:.4f} qps(1)={row['qps_single_thread']:8.0f}"
            f" qps({threads})={row['qps_all_threads']:8.0f}",
            flush=True,
        )
        sweep.append(row)
    return {
        "build_seconds": build_seconds,
        "memory_growth_mib": memory_growth,
        "threads": threads,
        "sweep": sweep,
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
        "hopfilter_simd": _core.simd_kernels(),
        "hnswlib": getattr(hnswlib, "__version__", "0.8.0"),
        "faiss": faiss.__version__,
        "faiss_simd_level": faiss.SIMDConfig.get_level_name(),
    }


def run_in_child(name: LibraryName, config_path: Path, scratch: Path) -> dict[str, object]:
    partial = scratch / f"{name}.json"
    command = [sys.executable, __file__, "--config", str(config_path), "--library", name]
    subprocess.run([*command, "--partial", str(partial)], check=True)
    result: dict[str, object] = json.loads(partial.read_text(encoding="utf-8"))
    return result


def run_single(config: CompareConfig, name: LibraryName, partial: Path) -> None:
    spec = DATASETS[config.dataset]
    dataset_dir = DatasetDownloader(HttpFetcher(), Path(config.data_dir)).download(spec)
    dataset = load_ann_benchmark(dataset_dir / spec.file_name)
    partial.write_text(json.dumps(run_library(name, dataset, config)), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--library", help="run one library in this process (internal)")
    parser.add_argument("--partial", type=Path, help="where --library writes its result")
    arguments = parser.parse_args()
    config = load_config(arguments.config, CompareConfig)
    if arguments.library is not None:
        run_single(config, arguments.library, arguments.partial)
        return
    with tempfile.TemporaryDirectory() as scratch:
        results = {
            name: run_in_child(name, arguments.config, Path(scratch)) for name in config.libraries
        }
    output = Path(config.results_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {"config": asdict(config), "environment": environment(), "libraries": results}
    output.write_text(json.dumps(payload, indent=4) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
