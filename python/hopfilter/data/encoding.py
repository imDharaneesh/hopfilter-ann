from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import polars as pl

from hopfilter.data.dataset import GENRES, ITEM_ID
from hopfilter.errors import EmptyDatasetError, UnknownIdError

type IndexArray = npt.NDArray[np.int64]
type FeatureMatrix = npt.NDArray[np.float32]

MAX_REPORTED_UNKNOWN_IDS = 5


class IdEncoder:
    """Maps raw ids to contiguous indices ``0..size-1`` in ascending id order."""

    def __init__(self, sorted_unique_ids: IndexArray) -> None:
        if sorted_unique_ids.size == 0:
            raise EmptyDatasetError("cannot build an encoder from zero ids")
        self._ids = sorted_unique_ids

    @classmethod
    def fit(cls, values: pl.Series) -> "IdEncoder":
        """Build an encoder over the distinct values of ``values``."""
        return cls(np.unique(values.to_numpy().astype(np.int64)))

    @property
    def size(self) -> int:
        """Number of distinct ids."""
        return int(self._ids.size)

    @property
    def ids(self) -> IndexArray:
        """Raw ids in index order; ``ids[i]`` is the id encoded as ``i``."""
        return self._ids.copy()

    def encode(self, values: pl.Series) -> IndexArray:
        """Return the index of every value.

        Raises:
            UnknownIdError: A value was not present when the encoder was fitted.
        """
        raw = values.to_numpy().astype(np.int64)
        positions = np.searchsorted(self._ids, raw).clip(max=self._ids.size - 1)
        unknown = raw[self._ids[positions] != raw]
        if unknown.size:
            sample = ", ".join(str(value) for value in unknown[:MAX_REPORTED_UNKNOWN_IDS])
            raise UnknownIdError(f"{unknown.size} ids were not seen at fit time, e.g. {sample}")
        return positions.astype(np.int64)

    def decode(self, indices: IndexArray) -> IndexArray:
        """Return the raw ids for ``indices``."""
        return self._ids[indices]


@dataclass(frozen=True)
class GenreMatrix:
    """Multi-hot genre membership aligned with an item ``IdEncoder``.

    Attributes:
        values: ``(num_items, num_genres)`` matrix of 0.0 and 1.0.
        vocabulary: Genre name for each column, sorted alphabetically.
    """

    values: FeatureMatrix
    vocabulary: tuple[str, ...]


def build_genre_matrix(items: pl.DataFrame, item_encoder: IdEncoder) -> GenreMatrix:
    """Build the multi-hot genre matrix for every item known to ``item_encoder``.

    Items missing from ``items`` get an all-zero row; rows of ``items`` the encoder
    does not know are ignored.
    """
    known_items = items.filter(pl.col(ITEM_ID).is_in(pl.Series(item_encoder.ids).implode()))
    vocabulary = tuple(sorted(known_items[GENRES].explode().drop_nulls().unique().to_list()))
    column_of = {genre: column for column, genre in enumerate(vocabulary)}
    values = np.zeros((item_encoder.size, len(vocabulary)), dtype=np.float32)
    rows = item_encoder.encode(known_items[ITEM_ID])
    for row, genres in zip(rows, known_items[GENRES].to_list(), strict=True):
        values[row, [column_of[genre] for genre in genres]] = 1.0
    return GenreMatrix(values=values, vocabulary=vocabulary)
