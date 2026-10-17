from dataclasses import dataclass
from typing import Final

import polars as pl

from hopfilter.data.dataset import ITEM_ID, TIMESTAMP, USER_ID
from hopfilter.errors import EmptyDatasetError, InvalidConfigError

DEFAULT_VALIDATION_FRACTION = 0.1
DEFAULT_TEST_FRACTION = 0.1
CUTOFF_INTERPOLATION: Final = "nearest"


@dataclass(frozen=True)
class SplitConfig:
    """Fractions of interactions, by time, held out for validation and test."""

    validation_fraction: float = DEFAULT_VALIDATION_FRACTION
    test_fraction: float = DEFAULT_TEST_FRACTION

    def __post_init__(self) -> None:
        for name in ("validation_fraction", "test_fraction"):
            if not 0.0 < getattr(self, name) < 1.0:
                raise InvalidConfigError(f"{name} must be in (0, 1), got {getattr(self, name)}")
        if self.validation_fraction + self.test_fraction >= 1.0:
            raise InvalidConfigError("validation_fraction + test_fraction must be below 1")


@dataclass(frozen=True)
class DataSplits:
    """Train, validation and test interactions with cold-start rows removed.

    Attributes:
        train: Every interaction strictly before the validation cutoff.
        validation: Interactions in ``[validation cutoff, test cutoff)`` whose user and
            item both appear in ``train``.
        test: Interactions at or after the test cutoff whose user and item both appear
            in ``train``.
        dropped_cold_start: Validation and test rows removed because their user or
            item never appears in ``train``.
    """

    train: pl.DataFrame
    validation: pl.DataFrame
    test: pl.DataFrame
    dropped_cold_start: int


class TemporalSplitter:
    """Splits interactions at two global timestamps so no split sees its future."""

    def __init__(self, config: SplitConfig) -> None:
        self._config = config

    def split(self, interactions: pl.DataFrame) -> DataSplits:
        """Cut ``interactions`` at the configured time quantiles.

        Raises:
            EmptyDatasetError: ``interactions`` is empty, or no interaction falls before
                the validation cutoff.
        """
        if interactions.is_empty():
            raise EmptyDatasetError("cannot split an empty interaction table")
        validation_start, test_start = self._cutoffs(interactions[TIMESTAMP])
        timestamp = pl.col(TIMESTAMP)
        train = interactions.filter(timestamp < validation_start)
        if train.is_empty():
            raise EmptyDatasetError("no interactions fall before the validation cutoff")
        validation = interactions.filter(timestamp.is_between(validation_start, test_start, "left"))
        test = interactions.filter(timestamp >= test_start)
        warm_validation = keep_seen(validation, train)
        warm_test = keep_seen(test, train)
        dropped = validation.height + test.height - warm_validation.height - warm_test.height
        return DataSplits(train, warm_validation, warm_test, dropped)

    def _cutoffs(self, timestamps: pl.Series) -> tuple[int, int]:
        test_quantile = 1.0 - self._config.test_fraction
        validation_quantile = test_quantile - self._config.validation_fraction
        return (
            _timestamp_quantile(timestamps, validation_quantile),
            _timestamp_quantile(timestamps, test_quantile),
        )


def keep_seen(rows: pl.DataFrame, reference: pl.DataFrame) -> pl.DataFrame:
    """Keep rows of ``rows`` whose user and item both occur in ``reference``."""
    return rows.filter(
        pl.col(USER_ID).is_in(reference[USER_ID].unique().implode())
        & pl.col(ITEM_ID).is_in(reference[ITEM_ID].unique().implode())
    )


def _timestamp_quantile(timestamps: pl.Series, quantile: float) -> int:
    value = timestamps.quantile(quantile, interpolation=CUTOFF_INTERPOLATION)
    if value is None:
        raise EmptyDatasetError("timestamps are all null")
    return int(value)
