import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from hopfilter.data.dataset import INTERACTION_SCHEMA
from hopfilter.data.splits import DataSplits, SplitConfig, TemporalSplitter, keep_seen
from hopfilter.errors import EmptyDatasetError, InvalidConfigError


def interactions_from(rows: list[tuple[int, int, int]]) -> pl.DataFrame:
    users, items, timestamps = zip(*rows, strict=True) if rows else ((), (), ())
    return pl.DataFrame(
        {
            "user_id": list(users),
            "item_id": list(items),
            "rating": [5.0] * len(rows),
            "timestamp": list(timestamps),
        }
    ).cast(INTERACTION_SCHEMA)


USER_ITEM_PAIRS_BY_TIME = [
    (1, 1),
    (2, 2),
    (1, 2),
    (2, 1),
    (1, 3),
    (2, 3),
    (3, 1),
    (1, 4),
    (2, 2),
    (9, 9),
]


def split_ten_steps() -> DataSplits:
    rows = [(user, item, step) for step, (user, item) in enumerate(USER_ITEM_PAIRS_BY_TIME)]
    config = SplitConfig(validation_fraction=0.2, test_fraction=0.2)
    return TemporalSplitter(config).split(interactions_from(rows))


def test_cuts_at_global_time_quantiles() -> None:
    splits = split_ten_steps()

    assert splits.train["timestamp"].to_list() == [0, 1, 2, 3, 4]
    assert splits.validation["timestamp"].to_list() == [5]
    assert splits.test["timestamp"].to_list() == [8]


def test_drops_cold_start_users_and_items_from_held_out_splits() -> None:
    splits = split_ten_steps()

    assert splits.dropped_cold_start == 3


def test_keep_seen_requires_both_user_and_item() -> None:
    reference = interactions_from([(1, 1, 0), (2, 2, 0)])
    rows = interactions_from([(1, 2, 5), (1, 3, 5), (3, 1, 5)])

    assert keep_seen(rows, reference)[["user_id", "item_id"]].rows() == [(1, 2)]


@settings(max_examples=60, deadline=None)
@given(
    timestamps=st.lists(st.integers(min_value=0, max_value=50), min_size=5, max_size=200),
    validation_fraction=st.floats(min_value=0.05, max_value=0.4),
    test_fraction=st.floats(min_value=0.05, max_value=0.4),
)
def test_no_split_contains_its_future(
    timestamps: list[int], validation_fraction: float, test_fraction: float
) -> None:
    rows = [(index % 7, index % 5, timestamp) for index, timestamp in enumerate(timestamps)]
    splitter = TemporalSplitter(SplitConfig(validation_fraction, test_fraction))

    try:
        splits = splitter.split(interactions_from(rows))
    except EmptyDatasetError:
        return

    train_times = splits.train["timestamp"].to_list()
    validation_times = splits.validation["timestamp"].to_list()
    test_times = splits.test["timestamp"].to_list()
    assert all(time > max(train_times) for time in validation_times + test_times)
    assert all(earlier < later for earlier in validation_times for later in test_times)


def test_empty_input_raises() -> None:
    with pytest.raises(EmptyDatasetError, match="empty"):
        TemporalSplitter(SplitConfig()).split(interactions_from([]))


def test_all_equal_timestamps_leave_no_training_data() -> None:
    rows = [(1, 1, 7), (2, 2, 7), (3, 3, 7)]

    with pytest.raises(EmptyDatasetError, match="before the validation cutoff"):
        TemporalSplitter(SplitConfig()).split(interactions_from(rows))


@pytest.mark.parametrize(
    ("validation_fraction", "test_fraction", "message"),
    [
        (0.0, 0.1, "validation_fraction must be in"),
        (0.1, 1.0, "test_fraction must be in"),
        (0.5, 0.5, "must be below 1"),
    ],
)
def test_invalid_fractions_are_rejected(
    validation_fraction: float, test_fraction: float, message: str
) -> None:
    with pytest.raises(InvalidConfigError, match=message):
        SplitConfig(validation_fraction, test_fraction)
