from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from hopfilter import Filter, HnswIndex
from hopfilter.errors import InvalidArgumentError

DIMENSION = 8
POINTS = 1500
BUCKETS = 20
K = 10
FILTERED_STRATEGIES = ["adaptive", "in_graph", "post_filter", "pre_filter"]

type FloatMatrix = npt.NDArray[np.float32]
type BoolVector = npt.NDArray[np.bool_]


def random_matrix(rows: int, seed: int) -> FloatMatrix:
    return np.random.default_rng(seed).standard_normal((rows, DIMENSION)).astype(np.float32)


DATA = random_matrix(POINTS, seed=5)
BUCKET_CODES = np.random.default_rng(6).integers(0, BUCKETS, POINTS)
SCORES = np.random.default_rng(7).random(POINTS)


@pytest.fixture(scope="module")
def data() -> FloatMatrix:
    return DATA


@pytest.fixture(scope="module")
def buckets() -> npt.NDArray[np.int64]:
    return BUCKET_CODES


@pytest.fixture(scope="module")
def index() -> HnswIndex:
    built = HnswIndex(DIMENSION, POINTS, max_neighbors=12, ef_construction=100)
    built.add(DATA)
    built.set_attribute("bucket", BUCKET_CODES)
    built.set_attribute("score", SCORES)
    return built


def exact_filtered(
    data: FloatMatrix, queries: FloatMatrix, mask: BoolVector, k: int
) -> list[set[int]]:
    allowed = np.flatnonzero(mask)
    distances = ((queries[:, None, :] - data[None, allowed, :]) ** 2).sum(axis=2)
    order = np.argsort(distances, axis=1, kind="stable")[:, :k]
    return [set(allowed[row].tolist()) for row in order]


def well_formed(
    ids: npt.NDArray[np.int64], distances: npt.NDArray[np.float32], mask: BoolVector
) -> bool:
    for row_ids, row_distances in zip(ids, distances, strict=True):
        found = row_ids[row_ids >= 0]
        if len(set(found.tolist())) != len(found) or not mask[found].all():
            return False
        if not np.all(np.diff(row_distances[: len(found)]) >= 0):
            return False
        if not np.all(row_ids[len(found) :] == -1):
            return False
    return True


@st.composite
def filters(draw: st.DrawFn) -> tuple[Filter, BoolVector]:
    bucket = draw(st.integers(0, BUCKETS - 1))
    low = draw(st.floats(0.0, 1.0))
    high = draw(st.floats(low, 1.0))
    equals, in_bucket = Filter.equals("bucket", bucket), np.equal(BUCKET_CODES, bucket)
    between, in_range = (
        Filter.between("score", low, high),
        np.greater_equal(SCORES, low) & np.less_equal(SCORES, high),
    )
    cases = [
        (equals, in_bucket),
        (between, in_range),
        (equals & between, in_bucket & in_range),
        (equals | between, in_bucket | in_range),
        (~equals, ~in_bucket),
    ]
    return draw(st.sampled_from(cases))


@settings(max_examples=40, deadline=None)
@given(case=filters(), strategy=st.sampled_from(FILTERED_STRATEGIES))
def test_results_satisfy_the_filter_and_are_sorted(
    index: HnswIndex, case: tuple[Filter, BoolVector], strategy: str
) -> None:
    filter_, mask = case
    ids, distances = index.search(DATA[:20], K, filter=filter_, strategy=strategy)

    assert well_formed(ids, distances, mask)
    assert index.count_matching(filter_) == int(mask.sum())


@pytest.mark.parametrize("strategy", ["adaptive", "in_graph", "pre_filter"])
def test_filtered_recall_against_numpy(
    index: HnswIndex, data: FloatMatrix, buckets: npt.NDArray[np.int64], strategy: str
) -> None:
    queries = random_matrix(50, seed=8)
    mask = (buckets == 1) | (buckets == 2)
    filter_ = Filter.equals("bucket", 1) | Filter.equals("bucket", 2)

    ids, _ = index.search(queries, K, filter=filter_, strategy=strategy)
    truth = exact_filtered(data, queries, mask, K)

    recall = np.mean(
        [len(set(row.tolist()) & expected) / K for row, expected in zip(ids, truth, strict=True)]
    )
    assert recall >= 0.95


def test_post_filter_comes_up_short_under_a_strict_filter(
    index: HnswIndex, data: FloatMatrix
) -> None:
    strict = Filter.mask(np.arange(POINTS) < 15)

    ids, _ = index.search(data[:20], K, ef=16, filter=strict, strategy="post_filter")

    assert (ids >= 0).sum() < K * 20 / 2


def test_adaptive_plans_follow_selectivity(index: HnswIndex) -> None:
    rows = np.arange(POINTS)

    assert index.plan_filter(Filter.mask(rows < POINTS)) == "post_filter"
    assert index.plan_filter(Filter.mask(rows < POINTS // 5)) == "in_graph"
    assert index.plan_filter(Filter.mask(rows < 5)) == "pre_filter"


def test_queries_far_from_the_matches_plan_exact_search() -> None:
    corner = np.full(DIMENSION, 2.0, dtype=np.float32)
    distance_to_corner = ((DATA - corner) ** 2).sum(axis=1).astype(np.float64)
    index = HnswIndex(DIMENSION, POINTS, max_neighbors=12, ef_construction=100)
    index.add(DATA)
    index.set_attribute("corner", distance_to_corner)
    near_corner = Filter.between("corner", 0.0, float(np.quantile(distance_to_corner, 0.1)))
    queries = np.stack([corner, -corner])

    assert index.plan_filter(near_corner) == "in_graph"
    assert index.plan_queries(queries, near_corner) == ["in_graph", "pre_filter"]
    ids, _ = index.search(queries, K, filter=near_corner)
    truth = exact_filtered(
        DATA, queries, distance_to_corner <= np.quantile(distance_to_corner, 0.1), K
    )
    assert [set(row.tolist()) for row in ids] == truth


def test_unfiltered_search_is_unchanged(index: HnswIndex, data: FloatMatrix) -> None:
    default_ids, _ = index.search(data[:10], K)
    explicit_ids, _ = index.search(data[:10], K, strategy="unfiltered")

    np.testing.assert_array_equal(default_ids, explicit_ids)


def test_filter_api_errors(index: HnswIndex, data: FloatMatrix) -> None:
    with pytest.raises(InvalidArgumentError, match="does not exist"):
        index.search(data[:1], K, filter=Filter.equals("missing", 1))
    with pytest.raises(InvalidArgumentError, match="is not numeric"):
        index.search(data[:1], K, filter=Filter.between("bucket", 0.0, 1.0))
    with pytest.raises(InvalidArgumentError, match="unknown search strategy"):
        index.search(data[:1], K, filter=Filter.equals("bucket", 1), strategy="brute")
    with pytest.raises(InvalidArgumentError, match="cannot apply a filter"):
        index.search(data[:1], K, filter=Filter.equals("bucket", 1), strategy="unfiltered")
    with pytest.raises(InvalidArgumentError, match="low <= high"):
        Filter.between("score", 1.0, 0.0)
    with pytest.raises(TypeError, match="no truth value"):
        bool(Filter.equals("bucket", 1))


def test_attribute_validation() -> None:
    index = HnswIndex(DIMENSION, 10)

    with pytest.raises(InvalidArgumentError, match="capacity"):
        index.set_attribute("big", np.zeros(11, dtype=np.int64))
    with pytest.raises(InvalidArgumentError, match="integers"):
        index.set_attribute("names", np.array(["a", "b"]))
    with pytest.raises(InvalidArgumentError, match=r"\[0, 2\*\*32 - 1\]"):
        index.set_attribute("negative", np.array([-1]))
    with pytest.raises(InvalidArgumentError, match="one-dimensional"):
        index.set_attribute("matrix", np.zeros((2, 2)))


def test_filter_repr_and_combinators() -> None:
    combined = Filter.all_of([Filter.equals("a", 1), Filter.between("b", 0.5, 2.0)])

    assert repr(combined) == "Filter(a == 1 & b in [0.5, 2])"
    assert repr(~Filter.equals("a", 1)) == "Filter(~a == 1)"
    assert repr(Filter.any_of([Filter.equals("a", 1), Filter.mask([True, False])])) == (
        "Filter(a == 1 | mask(1 of 2))"
    )


def test_attributes_survive_save_and_load(
    index: HnswIndex, data: FloatMatrix, tmp_path: Path
) -> None:
    path = tmp_path / "filtered.hopf"
    index.save(path)
    restored = HnswIndex.load(path)
    filter_ = Filter.equals("bucket", 3) & Filter.between("score", 0.2, 0.8)

    assert restored.attribute_names == ["bucket", "score"]
    assert restored.count_matching(filter_) == index.count_matching(filter_)
    np.testing.assert_array_equal(
        restored.search(data[:5], K, filter=filter_)[0],
        index.search(data[:5], K, filter=filter_)[0],
    )
