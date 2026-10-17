import numpy as np
import polars as pl
import pytest

from hopfilter.data.dataset import ITEM_SCHEMA
from hopfilter.data.encoding import IdEncoder, build_genre_matrix
from hopfilter.errors import EmptyDatasetError, UnknownIdError


def test_encodes_to_contiguous_indices_in_ascending_id_order() -> None:
    encoder = IdEncoder.fit(pl.Series([30, 10, 20, 10]))

    assert encoder.size == 3
    assert encoder.encode(pl.Series([10, 20, 30, 20])).tolist() == [0, 1, 2, 1]


def test_decode_inverts_encode() -> None:
    encoder = IdEncoder.fit(pl.Series([7, 3, 11]))
    raw = pl.Series([11, 3, 7])

    assert encoder.decode(encoder.encode(raw)).tolist() == [11, 3, 7]


def test_ids_are_a_defensive_copy_in_index_order() -> None:
    encoder = IdEncoder.fit(pl.Series([7, 3, 11]))

    ids = encoder.ids
    ids[0] = -1

    assert encoder.ids.tolist() == [3, 7, 11]


@pytest.mark.parametrize("unknown", [5, 25, 99])
def test_unknown_ids_raise(unknown: int) -> None:
    encoder = IdEncoder.fit(pl.Series([10, 20]))

    with pytest.raises(UnknownIdError, match=f"1 ids were not seen at fit time, e.g. {unknown}"):
        encoder.encode(pl.Series([10, unknown]))


def test_empty_fit_raises() -> None:
    with pytest.raises(EmptyDatasetError):
        IdEncoder.fit(pl.Series([], dtype=pl.Int64))


def test_genre_matrix_is_aligned_with_encoder() -> None:
    items = pl.DataFrame(
        {
            "item_id": [20, 10, 99],
            "title": ["b", "a", "unknown to encoder"],
            "year": [2000, None, 1990],
            "genres": [["Drama", "Action"], [], ["Western"]],
        }
    ).cast(ITEM_SCHEMA)
    encoder = IdEncoder.fit(pl.Series([10, 20, 30]))

    genres = build_genre_matrix(items, encoder)

    assert genres.vocabulary == ("Action", "Drama")
    np.testing.assert_array_equal(
        genres.values,
        np.array([[0, 0], [1, 1], [0, 0]], dtype=np.float32),
    )
    assert genres.values.dtype == np.float32
