from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor

from hopfilter.errors import InvalidConfigError

type IndexArray = npt.NDArray[np.int64]
type BoolArray = npt.NDArray[np.bool_]

DEFAULT_QUERY_CHUNK_SIZE = 1024
DEFAULT_HEAD_FRACTION = 0.2
DEFAULT_KS = (10, 50, 100)


class ExactRetriever:
    """Brute-force top-k by inner product against every item."""

    def __init__(
        self, item_vectors: Tensor, query_chunk_size: int = DEFAULT_QUERY_CHUNK_SIZE
    ) -> None:
        self._item_vectors = item_vectors.detach()
        self._query_chunk_size = query_chunk_size

    @property
    def num_items(self) -> int:
        """Number of items searched."""
        return int(self._item_vectors.shape[0])

    def top_k(self, query_vectors: Tensor, k: int, excluded: Sequence[IndexArray]) -> IndexArray:
        """Return the ``(num_queries, k)`` best item indices per query, best first.

        Args:
            query_vectors: ``(num_queries, dim)`` query embeddings.
            k: Number of items to return per query.
            excluded: For each query, item indices that must not be returned, such as
                items the user already interacted with during training.
        """
        if not 0 < k <= self.num_items:
            raise InvalidConfigError(f"k must be in [1, {self.num_items}], got {k}")
        if len(excluded) != query_vectors.shape[0]:
            raise InvalidConfigError("excluded must have one entry per query")
        chunks = []
        with torch.no_grad():
            for start in range(0, query_vectors.shape[0], self._query_chunk_size):
                stop = start + self._query_chunk_size
                scores = query_vectors[start:stop].detach() @ self._item_vectors.T
                mask_excluded(scores, excluded[start:stop])
                chunks.append(scores.topk(k, dim=1).indices)
        return torch.cat(chunks).numpy().astype(np.int64)


def mask_excluded(scores: Tensor, excluded: Sequence[IndexArray]) -> None:
    """Set ``scores[row, item]`` to ``-inf`` in place for every excluded item of each row."""
    rows = np.repeat(np.arange(len(excluded)), [items.size for items in excluded])
    columns = np.concatenate(excluded) if excluded else np.empty(0, dtype=np.int64)
    scores[torch.from_numpy(rows), torch.from_numpy(columns.astype(np.int64))] = float("-inf")


class PopularitySegmenter:
    """Labels the ``head_fraction`` most-interacted items as head and the rest as tail."""

    def __init__(
        self,
        train_item_indices: IndexArray,
        num_items: int,
        head_fraction: float = DEFAULT_HEAD_FRACTION,
    ) -> None:
        if not 0.0 < head_fraction < 1.0:
            raise InvalidConfigError(f"head_fraction must be in (0, 1), got {head_fraction}")
        counts = np.bincount(train_item_indices, minlength=num_items)
        head_size = max(1, round(head_fraction * num_items))
        most_popular_first = np.argsort(-counts, kind="stable")
        self._is_head = np.zeros(num_items, dtype=np.bool_)
        self._is_head[most_popular_first[:head_size]] = True
        total_interactions = int(counts.sum())
        head_interactions = int(counts[self._is_head].sum())
        self._head_interaction_share = (
            head_interactions / total_interactions if total_interactions else 0.0
        )

    @property
    def head_interaction_share(self) -> float:
        """Fraction of training interactions that involve head items."""
        return self._head_interaction_share

    def is_head(self, item_indices: IndexArray) -> BoolArray:
        """Whether each item is in the head segment."""
        return self._is_head[item_indices]


@dataclass(frozen=True)
class RetrievalReport:
    """Interaction-level recall@k overall and per popularity segment.

    ``recall[k]`` is the fraction of held-out positives retrieved in the user's
    top-k. ``head_recall`` and ``tail_recall`` restrict both numerator and
    denominator to positives on head or tail items.
    """

    recall: Mapping[int, float]
    head_recall: Mapping[int, float]
    tail_recall: Mapping[int, float]
    num_queries: int
    num_positives: int
    num_head_positives: int

    def to_dict(self) -> dict[str, object]:
        """JSON-friendly representation with string keys."""
        return {
            "recall": {str(k): value for k, value in self.recall.items()},
            "head_recall": {str(k): value for k, value in self.head_recall.items()},
            "tail_recall": {str(k): value for k, value in self.tail_recall.items()},
            "num_queries": self.num_queries,
            "num_positives": self.num_positives,
            "num_head_positives": self.num_head_positives,
        }


class RetrievalEvaluator:
    """Scores query vectors against held-out positives with exact retrieval."""

    def __init__(
        self,
        retriever: ExactRetriever,
        segmenter: PopularitySegmenter,
        ks: tuple[int, ...] = DEFAULT_KS,
    ) -> None:
        if not ks or min(ks) <= 0:
            raise InvalidConfigError(f"ks must be non-empty and positive, got {ks}")
        self._retriever = retriever
        self._segmenter = segmenter
        self._ks = tuple(sorted(ks))

    def evaluate(
        self,
        query_vectors: Tensor,
        relevant: Sequence[IndexArray],
        excluded: Sequence[IndexArray],
    ) -> RetrievalReport:
        """Recall@k for each configured k, overall and for head and tail items."""
        top_items = self._retriever.top_k(query_vectors, max(self._ks), excluded)
        positives, ranks = retrieval_ranks(top_items, relevant)
        head = self._segmenter.is_head(positives)
        return RetrievalReport(
            recall={k: _hit_rate(ranks, k) for k in self._ks},
            head_recall={k: _hit_rate(ranks[head], k) for k in self._ks},
            tail_recall={k: _hit_rate(ranks[~head], k) for k in self._ks},
            num_queries=len(relevant),
            num_positives=int(positives.size),
            num_head_positives=int(head.sum()),
        )


def retrieval_ranks(
    top_items: IndexArray, relevant: Sequence[IndexArray]
) -> tuple[IndexArray, IndexArray]:
    """Flatten held-out positives and the 0-based rank each was retrieved at.

    Positives that are not in a row's ``top_items`` get rank ``top_items.shape[1]``.
    """
    missed_rank = top_items.shape[1]
    ranks = []
    for row, items in enumerate(relevant):
        matches = top_items[row][np.newaxis, :] == items[:, np.newaxis]
        found = matches.any(axis=1)
        ranks.append(np.where(found, matches.argmax(axis=1), missed_rank))
    positives = np.concatenate(relevant) if relevant else np.empty(0, dtype=np.int64)
    flat_ranks = np.concatenate(ranks) if ranks else np.empty(0, dtype=np.int64)
    return positives.astype(np.int64), flat_ranks.astype(np.int64)


def group_by_query(
    query_indices: IndexArray, item_indices: IndexArray, queries: IndexArray
) -> list[IndexArray]:
    """For each entry of ``queries``, the sorted unique items paired with it."""
    order = np.argsort(query_indices, kind="stable")
    sorted_queries, sorted_items = query_indices[order], item_indices[order]
    starts = np.searchsorted(sorted_queries, queries, side="left")
    stops = np.searchsorted(sorted_queries, queries, side="right")
    return [np.unique(sorted_items[start:stop]) for start, stop in zip(starts, stops, strict=True)]


def _hit_rate(ranks: IndexArray, k: int) -> float:
    return float((ranks < k).mean()) if ranks.size else 0.0
