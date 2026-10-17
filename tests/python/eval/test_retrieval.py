import numpy as np
import pytest
import torch

from hopfilter.errors import InvalidConfigError
from hopfilter.eval.retrieval import (
    ExactRetriever,
    PopularitySegmenter,
    RetrievalEvaluator,
    group_by_query,
    mask_excluded,
    retrieval_ranks,
)

ITEMS = torch.tensor([[1.0, 0.0], [0.8, 0.6], [0.0, 1.0], [-1.0, 0.0]])
NOTHING: np.ndarray = np.empty(0, dtype=np.int64)


def ids(*values: int) -> np.ndarray:
    return np.array(values, dtype=np.int64)


def test_top_k_orders_by_inner_product() -> None:
    retriever = ExactRetriever(ITEMS)

    top = retriever.top_k(torch.tensor([[1.0, 0.0], [0.0, 1.0]]), k=3, excluded=[NOTHING, NOTHING])

    assert top.tolist() == [[0, 1, 2], [2, 1, 0]]


def test_top_k_skips_excluded_items() -> None:
    retriever = ExactRetriever(ITEMS)

    top = retriever.top_k(torch.tensor([[1.0, 0.0]]), k=2, excluded=[ids(0, 1)])

    assert top.tolist() == [[2, 3]]


def test_chunking_does_not_change_results() -> None:
    torch.manual_seed(0)
    items, queries = torch.randn(50, 4), torch.randn(37, 4)
    excluded = [ids(row % 50) for row in range(37)]

    whole = ExactRetriever(items, query_chunk_size=1000).top_k(queries, 5, excluded)
    chunked = ExactRetriever(items, query_chunk_size=4).top_k(queries, 5, excluded)

    np.testing.assert_array_equal(whole, chunked)


def test_brute_force_agrees_with_numpy_reference() -> None:
    generator = np.random.default_rng(1)
    items, queries = generator.normal(size=(40, 3)), generator.normal(size=(6, 3))

    top = ExactRetriever(torch.from_numpy(items)).top_k(torch.from_numpy(queries), 4, [NOTHING] * 6)

    reference = np.argsort(-(queries @ items.T), axis=1)[:, :4]
    np.testing.assert_array_equal(top, reference)


@pytest.mark.parametrize("k", [0, 5])
def test_k_outside_item_range_is_rejected(k: int) -> None:
    with pytest.raises(InvalidConfigError, match="k must be in"):
        ExactRetriever(ITEMS).top_k(torch.zeros(1, 2), k, [NOTHING])


def test_excluded_must_align_with_queries() -> None:
    with pytest.raises(InvalidConfigError, match="one entry per query"):
        ExactRetriever(ITEMS).top_k(torch.zeros(2, 2), 1, [NOTHING])


def test_mask_excluded_handles_empty_rows() -> None:
    scores = torch.zeros(2, 3)

    mask_excluded(scores, [NOTHING, ids(1)])

    assert torch.isinf(scores).nonzero().tolist() == [[1, 1]]


def test_segmenter_marks_most_interacted_items_as_head() -> None:
    train_items = ids(3, 3, 3, 1, 1, 0)

    segmenter = PopularitySegmenter(train_items, num_items=5, head_fraction=0.4)

    assert segmenter.is_head(ids(0, 1, 2, 3, 4)).tolist() == [False, True, False, True, False]
    assert segmenter.head_interaction_share == pytest.approx(5 / 6)


def test_retrieval_ranks_marks_misses_with_list_length() -> None:
    top = np.array([[5, 6, 7], [7, 8, 9]], dtype=np.int64)

    positives, ranks = retrieval_ranks(top, [ids(7, 1), ids(9)])

    assert positives.tolist() == [7, 1, 9]
    assert ranks.tolist() == [2, 3, 2]


def test_evaluator_reports_interaction_level_recall_by_segment() -> None:
    retriever = ExactRetriever(ITEMS)
    segmenter = PopularitySegmenter(ids(0, 0, 0, 1), num_items=4, head_fraction=0.25)
    evaluator = RetrievalEvaluator(retriever, segmenter, ks=(1, 2))
    queries = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    report = evaluator.evaluate(queries, relevant=[ids(0, 2), ids(1)], excluded=[NOTHING, NOTHING])

    assert report.recall == {1: pytest.approx(1 / 3), 2: pytest.approx(2 / 3)}
    assert report.head_recall == {1: 1.0, 2: 1.0}
    assert report.tail_recall == {1: 0.0, 2: 0.5}
    assert (report.num_queries, report.num_positives, report.num_head_positives) == (2, 3, 1)
    assert report.to_dict()["recall"] == {"1": pytest.approx(1 / 3), "2": pytest.approx(2 / 3)}


def test_empty_segment_reports_zero() -> None:
    segmenter = PopularitySegmenter(ids(0), num_items=4, head_fraction=0.25)
    evaluator = RetrievalEvaluator(ExactRetriever(ITEMS), segmenter, ks=(1,))

    report = evaluator.evaluate(torch.tensor([[1.0, 0.0]]), [ids(2)], [NOTHING])

    assert report.head_recall == {1: 0.0}
    assert report.num_head_positives == 0


def test_group_by_query_returns_sorted_unique_items_per_query() -> None:
    groups = group_by_query(ids(2, 1, 2, 2, 5), ids(9, 4, 3, 9, 7), queries=ids(1, 2, 3))

    assert [group.tolist() for group in groups] == [[4], [3, 9], []]


@pytest.mark.parametrize("ks", [(), (0, 10)])
def test_invalid_ks_are_rejected(ks: tuple[int, ...]) -> None:
    segmenter = PopularitySegmenter(ids(0), num_items=4)

    with pytest.raises(InvalidConfigError, match="ks must be"):
        RetrievalEvaluator(ExactRetriever(ITEMS), segmenter, ks=ks)


def test_invalid_head_fraction_is_rejected() -> None:
    with pytest.raises(InvalidConfigError, match="head_fraction"):
        PopularitySegmenter(ids(0), num_items=4, head_fraction=1.0)
