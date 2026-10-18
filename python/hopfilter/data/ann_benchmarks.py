from dataclasses import dataclass
from pathlib import Path
from typing import Final

import h5py
import numpy as np
import numpy.typing as npt

from hopfilter.errors import DatasetIntegrityError, DatasetNotFoundError

METRIC_BY_DISTANCE: Final = {"euclidean": "l2", "angular": "cosine"}
REQUIRED_KEYS: Final = ("train", "test", "neighbors")
DISTANCE_ATTRIBUTE: Final = "distance"


@dataclass(frozen=True)
class AnnBenchmarkDataset:
    """An ANN-Benchmarks dataset: base vectors, queries and exact ground truth.

    Attributes:
        train: ``(n, d)`` float32 vectors to index.
        test: ``(q, d)`` float32 queries.
        neighbors: ``(q, g)`` exact nearest-neighbor ids per query, closest first.
        metric: hopfilter metric name matching the dataset's distance.
    """

    train: npt.NDArray[np.float32]
    test: npt.NDArray[np.float32]
    neighbors: npt.NDArray[np.int64]
    metric: str


def load_ann_benchmark(path: Path) -> AnnBenchmarkDataset:
    """Read an ANN-Benchmarks HDF5 file such as ``sift-128-euclidean.hdf5``.

    Raises:
        DatasetNotFoundError: ``path`` does not exist.
        DatasetIntegrityError: Required datasets are missing or the distance is unknown.
    """
    if not path.is_file():
        raise DatasetNotFoundError(f"{path} not found; run 'hopfilter download' first")
    with h5py.File(path, "r") as source:
        missing = [key for key in REQUIRED_KEYS if key not in source]
        if missing:
            raise DatasetIntegrityError(f"{path.name} lacks datasets: {', '.join(missing)}")
        distance = str(source.attrs.get(DISTANCE_ATTRIBUTE, ""))
        if distance not in METRIC_BY_DISTANCE:
            raise DatasetIntegrityError(f"{path.name} has unsupported distance '{distance}'")
        return AnnBenchmarkDataset(
            train=np.asarray(source["train"], dtype=np.float32),
            test=np.asarray(source["test"], dtype=np.float32),
            neighbors=np.asarray(source["neighbors"], dtype=np.int64),
            metric=METRIC_BY_DISTANCE[distance],
        )
