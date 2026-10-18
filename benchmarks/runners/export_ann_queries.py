"""Export an ANN-Benchmarks dataset's queries and ground truth as raw binary files.

The C++ search harness reads them without an HDF5 dependency. Each file starts with
two little-endian uint32 values (rows, columns) followed by row-major data: float32
for ``<name>.queries.bin`` and int32 for ``<name>.truth.bin``.

Usage:
    uv run python benchmarks/runners/export_ann_queries.py --dataset sift-128-euclidean
"""

import argparse
from pathlib import Path

import numpy as np
import numpy.typing as npt

from hopfilter.data.ann_benchmarks import load_ann_benchmark
from hopfilter.data.download import DATASETS

HEADER_DTYPE = np.dtype("<u4")


def write_matrix(path: Path, matrix: npt.NDArray[np.generic]) -> None:
    header = np.array(matrix.shape, dtype=HEADER_DTYPE)
    with path.open("wb") as sink:
        sink.write(header.tobytes())
        sink.write(np.ascontiguousarray(matrix).tobytes())
    print(f"wrote {path} {matrix.shape} {matrix.dtype}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    arguments = parser.parse_args()
    spec = DATASETS[arguments.dataset]
    dataset_dir = arguments.data_dir / spec.directory_name
    dataset = load_ann_benchmark(dataset_dir / spec.file_name)
    write_matrix(dataset_dir / f"{spec.name}.queries.bin", dataset.test.astype("<f4"))
    write_matrix(dataset_dir / f"{spec.name}.truth.bin", dataset.neighbors.astype("<i4"))


if __name__ == "__main__":
    main()
