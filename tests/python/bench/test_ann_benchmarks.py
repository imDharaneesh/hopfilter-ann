from pathlib import Path

import h5py
import numpy as np
import pytest

from hopfilter.data.ann_benchmarks import load_ann_benchmark
from hopfilter.errors import DatasetIntegrityError, DatasetNotFoundError


def write_dataset(
    path: Path, distance: str, keys: tuple[str, ...] = ("train", "test", "neighbors")
) -> None:
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as sink:
        sink.attrs["distance"] = distance
        if "train" in keys:
            sink["train"] = rng.standard_normal((20, 4))
        if "test" in keys:
            sink["test"] = rng.standard_normal((3, 4))
        if "neighbors" in keys:
            sink["neighbors"] = rng.integers(0, 20, size=(3, 5), dtype=np.int32)


@pytest.mark.parametrize(("distance", "metric"), [("euclidean", "l2"), ("angular", "cosine")])
def test_loads_arrays_with_hopfilter_dtypes_and_metric(
    tmp_path: Path, distance: str, metric: str
) -> None:
    path = tmp_path / "toy.hdf5"
    write_dataset(path, distance)

    dataset = load_ann_benchmark(path)

    assert dataset.train.shape == (20, 4)
    assert dataset.test.shape == (3, 4)
    assert dataset.neighbors.shape == (3, 5)
    assert dataset.train.dtype == dataset.test.dtype == np.float32
    assert dataset.neighbors.dtype == np.int64
    assert dataset.metric == metric


def test_missing_file_points_to_download(tmp_path: Path) -> None:
    with pytest.raises(DatasetNotFoundError, match="hopfilter download"):
        load_ann_benchmark(tmp_path / "absent.hdf5")


def test_missing_datasets_are_reported(tmp_path: Path) -> None:
    path = tmp_path / "partial.hdf5"
    write_dataset(path, "euclidean", keys=("train",))

    with pytest.raises(DatasetIntegrityError, match="lacks datasets: test, neighbors"):
        load_ann_benchmark(path)


def test_unknown_distance_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "jaccard.hdf5"
    write_dataset(path, "jaccard")

    with pytest.raises(DatasetIntegrityError, match="unsupported distance 'jaccard'"):
        load_ann_benchmark(path)
