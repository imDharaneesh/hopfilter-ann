from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from hopfilter import HnswIndex
from hopfilter.errors import (
    HopfilterError,
    IndexStateError,
    InvalidArgumentError,
    SerializationError,
)

DIMENSION = 16
POINTS = 2000
K = 10

type FloatMatrix = npt.NDArray[np.float32]


def random_matrix(rows: int, seed: int, dimension: int = DIMENSION) -> FloatMatrix:
    return np.random.default_rng(seed).standard_normal((rows, dimension)).astype(np.float32)


def exact_top_k(
    data: FloatMatrix, queries: FloatMatrix, k: int, metric: str
) -> npt.NDArray[np.int64]:
    if metric == "cosine":
        data = data / np.linalg.norm(data, axis=1, keepdims=True)
        queries = queries / np.linalg.norm(queries, axis=1, keepdims=True)
    if metric == "l2":
        scores = ((queries[:, None, :] - data[None, :, :]) ** 2).sum(axis=2)
    else:
        scores = -(queries @ data.T)
    return np.argsort(scores, axis=1, kind="stable")[:, :k]


def build(data: FloatMatrix, metric: str = "l2") -> HnswIndex:
    index = HnswIndex(DIMENSION, len(data), metric=metric, max_neighbors=12, ef_construction=100)
    index.add(data)
    return index


@pytest.fixture(scope="module")
def l2_data() -> FloatMatrix:
    return random_matrix(POINTS, seed=1)


@pytest.fixture(scope="module")
def l2_index(l2_data: FloatMatrix) -> HnswIndex:
    return build(l2_data)


@pytest.mark.parametrize("metric", ["l2", "ip", "cosine"])
def test_recall_against_numpy_brute_force(metric: str) -> None:
    data, queries = random_matrix(POINTS, seed=2), random_matrix(100, seed=3)
    index = build(data, metric)

    found, _ = index.search(queries, K, ef=64)

    truth = exact_top_k(data, queries, K, metric)
    recall = np.mean([len(set(f) & set(t)) / K for f, t in zip(found, truth, strict=True)])
    assert recall >= 0.95


def test_add_returns_sequential_ids_and_properties(l2_index: HnswIndex) -> None:
    index = HnswIndex(DIMENSION, 5, metric="cosine", max_neighbors=8, ef_construction=32, seed=7)

    ids = index.add(random_matrix(3, seed=4))

    assert ids.tolist() == [0, 1, 2]
    assert ids.dtype == np.int64
    assert (len(index), index.dimension, index.capacity) == (3, DIMENSION, 5)
    assert (index.metric, index.max_neighbors, index.ef_construction) == ("cosine", 8, 32)
    assert len(l2_index) == POINTS


def test_search_returns_sorted_squared_distances(l2_data: FloatMatrix, l2_index: HnswIndex) -> None:
    ids, distances = l2_index.search(l2_data[:5], K)

    assert ids.shape == distances.shape == (5, K)
    assert distances.dtype == np.float32
    assert ids[:, 0].tolist() == [0, 1, 2, 3, 4]
    assert np.all(np.diff(distances, axis=1) >= 0)
    expected = ((l2_data[ids[0]] - l2_data[0]) ** 2).sum(axis=1)
    np.testing.assert_allclose(distances[0], expected, rtol=1e-5, atol=1e-5)


def test_k_larger_than_index_is_padded() -> None:
    index = HnswIndex(DIMENSION, 3)
    index.add(random_matrix(2, seed=5))

    ids, distances = index.search(random_matrix(1, seed=6), 4)

    assert ids[0, 2:].tolist() == [-1, -1]
    assert np.isinf(distances[0, 2:]).all()


def test_accepts_float64_and_non_contiguous_input(
    l2_data: FloatMatrix, l2_index: HnswIndex
) -> None:
    as_float64 = l2_data[:3].astype(np.float64)
    strided = np.asfortranarray(l2_data[:3])

    first, _ = l2_index.search(as_float64, 1)
    second, _ = l2_index.search(strided, 1)

    assert first[:, 0].tolist() == second[:, 0].tolist() == [0, 1, 2]


@pytest.mark.parametrize(
    "bad_input",
    [np.zeros(DIMENSION), np.zeros((2, DIMENSION + 1)), np.zeros((1, 2, DIMENSION))],
)
def test_wrong_shapes_raise_invalid_argument(
    l2_index: HnswIndex, bad_input: npt.NDArray[np.float64]
) -> None:
    with pytest.raises(InvalidArgumentError, match=r"must have shape \(n, 16\)"):
        l2_index.search(bad_input, 1)


def test_zero_k_raises(l2_index: HnswIndex) -> None:
    with pytest.raises(InvalidArgumentError, match="k must be positive"):
        l2_index.search(random_matrix(1, seed=7), 0)


def test_adding_past_capacity_raises_index_state_error() -> None:
    index = HnswIndex(DIMENSION, 2)

    with pytest.raises(IndexStateError, match="full"):
        index.add(random_matrix(3, seed=8))


def test_invalid_construction_raises() -> None:
    with pytest.raises(InvalidArgumentError, match="unknown metric 'hamming'"):
        HnswIndex(DIMENSION, 10, metric="hamming")
    with pytest.raises(InvalidArgumentError, match="maxNeighbors must be at least 2"):
        HnswIndex(DIMENSION, 10, max_neighbors=1)


def test_save_and_load_round_trip(
    tmp_path: Path, l2_data: FloatMatrix, l2_index: HnswIndex
) -> None:
    path = tmp_path / "index.hopf"

    l2_index.save(path)
    restored = HnswIndex.load(path)

    queries = l2_data[:20]
    np.testing.assert_array_equal(restored.search(queries, K)[0], l2_index.search(queries, K)[0])
    assert (len(restored), restored.metric) == (POINTS, "l2")


def test_loading_garbage_raises_serialization_error(tmp_path: Path) -> None:
    path = tmp_path / "garbage.hopf"
    path.write_bytes(b"definitely not an index")

    with pytest.raises(SerializationError, match="bad magic"):
        HnswIndex.load(path)
    with pytest.raises(SerializationError, match="cannot open"):
        HnswIndex.load(tmp_path / "missing.hopf")


def test_native_errors_share_the_library_base_class() -> None:
    with pytest.raises(HopfilterError):
        HnswIndex(0, 10)


@settings(max_examples=30, deadline=None)
@given(
    k=st.integers(min_value=1, max_value=40),
    seed=st.integers(min_value=0, max_value=2**16),
)
def test_results_are_sorted_unique_and_valid(l2_index: HnswIndex, k: int, seed: int) -> None:
    ids, distances = l2_index.search(random_matrix(3, seed=seed), k)

    for row_ids, row_distances in zip(ids, distances, strict=True):
        assert len(set(row_ids.tolist())) == k
        assert np.all((row_ids >= 0) & (row_ids < POINTS))
        assert np.all(np.diff(row_distances) >= 0)
