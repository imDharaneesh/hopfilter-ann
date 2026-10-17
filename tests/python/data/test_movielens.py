from pathlib import Path

import polars as pl
import pytest

from hopfilter.data.dataset import (
    INTERACTION_SCHEMA,
    ITEM_SCHEMA,
    InteractionDataset,
    keep_positive,
)
from hopfilter.data.movielens import MovieLens1MLoader, MovieLens25MLoader
from hopfilter.errors import DatasetNotFoundError

ML1M_RATINGS = "1::10::5::978300760\n1::20::3::978302109\n2::10::4::978301968\n"
ML1M_MOVIES = (
    "10::Misérables, Les (1995)::Drama|Musical\n"
    "20::Star Wars: Episode IV - A New Hope (1977)::Action|Sci-Fi\n"
)
ML25M_RATINGS = "userId,movieId,rating,timestamp\n7,30,4.5,1147880044\n8,40,2.0,1147868817\n"
ML25M_MOVIES = (
    "movieId,title,genres\n"
    '30,"American President, The (1995)",Comedy|Drama|Romance\n'
    "40,Untitled Short Film,(no genres listed)\n"
)


@pytest.fixture
def ml1m_dir(tmp_path: Path) -> Path:
    (tmp_path / "ratings.dat").write_text(ML1M_RATINGS, encoding="latin-1")
    (tmp_path / "movies.dat").write_text(ML1M_MOVIES, encoding="latin-1")
    return tmp_path


@pytest.fixture
def ml25m_dir(tmp_path: Path) -> Path:
    (tmp_path / "ratings.csv").write_text(ML25M_RATINGS, encoding="utf-8")
    (tmp_path / "movies.csv").write_text(ML25M_MOVIES, encoding="utf-8")
    return tmp_path


def assert_common_schema(dataset: InteractionDataset) -> None:
    assert dataset.interactions.schema == INTERACTION_SCHEMA
    assert dataset.items.schema == ITEM_SCHEMA


def test_ml1m_ratings_are_parsed_into_common_schema(ml1m_dir: Path) -> None:
    dataset = MovieLens1MLoader(ml1m_dir).load()

    assert_common_schema(dataset)
    assert dataset.interactions.rows() == [
        (1, 10, 5.0, 978300760),
        (1, 20, 3.0, 978302109),
        (2, 10, 4.0, 978301968),
    ]


def test_ml1m_movies_handle_latin1_colons_and_years(ml1m_dir: Path) -> None:
    items = MovieLens1MLoader(ml1m_dir).load().items

    assert items.rows() == [
        (10, "Misérables, Les", 1995, ["Drama", "Musical"]),
        (20, "Star Wars: Episode IV - A New Hope", 1977, ["Action", "Sci-Fi"]),
    ]


def test_ml25m_is_normalized_to_the_same_schema(ml25m_dir: Path) -> None:
    dataset = MovieLens25MLoader(ml25m_dir).load()

    assert_common_schema(dataset)
    assert dataset.interactions.rows() == [(7, 30, 4.5, 1147880044), (8, 40, 2.0, 1147868817)]
    assert dataset.items.rows() == [
        (30, "American President, The", 1995, ["Comedy", "Drama", "Romance"]),
        (40, "Untitled Short Film", None, []),
    ]


@pytest.mark.parametrize(
    ("loader_type", "present_file", "missing_file", "dataset_name"),
    [
        (MovieLens1MLoader, "ratings.dat", "movies.dat", "movielens-1m"),
        (MovieLens25MLoader, "movies.csv", "ratings.csv", "movielens-25m"),
    ],
)
def test_missing_files_point_to_download_command(
    tmp_path: Path,
    loader_type: type[MovieLens1MLoader] | type[MovieLens25MLoader],
    present_file: str,
    missing_file: str,
    dataset_name: str,
) -> None:
    (tmp_path / present_file).write_text("", encoding="utf-8")

    with pytest.raises(
        DatasetNotFoundError, match=f"{missing_file}.*hopfilter download {dataset_name}"
    ):
        loader_type(tmp_path).load()


def test_keep_positive_uses_inclusive_threshold() -> None:
    interactions = pl.DataFrame(
        {
            "user_id": [1, 1, 1],
            "item_id": [1, 2, 3],
            "rating": [3.5, 4.0, 5.0],
            "timestamp": [1, 2, 3],
        }
    ).cast(INTERACTION_SCHEMA)

    positives = keep_positive(interactions, min_rating=4.0)

    assert positives["item_id"].to_list() == [2, 3]
