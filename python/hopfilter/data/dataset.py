from dataclasses import dataclass
from typing import Protocol

import polars as pl

USER_ID = "user_id"
ITEM_ID = "item_id"
RATING = "rating"
TIMESTAMP = "timestamp"
TITLE = "title"
YEAR = "year"
GENRES = "genres"

INTERACTION_SCHEMA = pl.Schema(
    {USER_ID: pl.Int64(), ITEM_ID: pl.Int64(), RATING: pl.Float32(), TIMESTAMP: pl.Int64()}
)
ITEM_SCHEMA = pl.Schema(
    {ITEM_ID: pl.Int64(), TITLE: pl.String(), YEAR: pl.Int32(), GENRES: pl.List(pl.String())}
)
DEFAULT_MIN_POSITIVE_RATING = 4.0


@dataclass(frozen=True)
class InteractionDataset:
    """User-item interactions plus item metadata in hopfilter's common schema.

    Attributes:
        interactions: Columns ``user_id``, ``item_id``, ``rating``, ``timestamp``
            (seconds since the Unix epoch).
        items: Columns ``item_id``, ``title``, ``year`` (nullable), ``genres``
            (list of strings, possibly empty).
    """

    interactions: pl.DataFrame
    items: pl.DataFrame


class DatasetLoader(Protocol):
    """Reads one dataset from local files into an ``InteractionDataset``."""

    def load(self) -> InteractionDataset:
        """Load and normalize the dataset."""
        ...


def keep_positive(
    interactions: pl.DataFrame, min_rating: float = DEFAULT_MIN_POSITIVE_RATING
) -> pl.DataFrame:
    """Keep only interactions rated at or above ``min_rating``."""
    return interactions.filter(pl.col(RATING) >= min_rating)
