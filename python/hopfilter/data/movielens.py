from pathlib import Path

import polars as pl

from hopfilter.data.dataset import (
    GENRES,
    INTERACTION_SCHEMA,
    ITEM_ID,
    ITEM_SCHEMA,
    RATING,
    TIMESTAMP,
    TITLE,
    USER_ID,
    YEAR,
    InteractionDataset,
)
from hopfilter.errors import DatasetNotFoundError

ML1M_RATINGS_FILE = "ratings.dat"
ML1M_MOVIES_FILE = "movies.dat"
ML1M_FIELD_SEPARATOR = "::"
ML1M_ENCODING = "latin-1"
ML25M_RATINGS_FILE = "ratings.csv"
ML25M_MOVIES_FILE = "movies.csv"
GENRE_SEPARATOR = "|"
NO_GENRES_MARKER = "(no genres listed)"
TRAILING_YEAR_PATTERN = r"\((\d{4})\)\s*$"
TRAILING_YEAR_WITH_SPACE_PATTERN = r"\s*\(\d{4}\)\s*$"
RAW_TITLE = "raw_title"
RAW_GENRES = "raw_genres"


class MovieLens1MLoader:
    """Loads MovieLens-1M (``ratings.dat`` and ``movies.dat``) from ``dataset_dir``."""

    def __init__(self, dataset_dir: Path) -> None:
        self._dataset_dir = dataset_dir

    def load(self) -> InteractionDataset:
        """Read ratings and movies and normalize them to the common schema."""
        ratings_path, movies_path = _require_files(
            self._dataset_dir, (ML1M_RATINGS_FILE, ML1M_MOVIES_FILE), "movielens-1m"
        )
        return InteractionDataset(
            interactions=_read_ml1m_ratings(ratings_path),
            items=_normalize_items(_read_ml1m_movies(movies_path)),
        )


class MovieLens25MLoader:
    """Loads MovieLens-25M (``ratings.csv`` and ``movies.csv``) from ``dataset_dir``."""

    def __init__(self, dataset_dir: Path) -> None:
        self._dataset_dir = dataset_dir

    def load(self) -> InteractionDataset:
        """Read ratings and movies and normalize them to the common schema."""
        ratings_path, movies_path = _require_files(
            self._dataset_dir, (ML25M_RATINGS_FILE, ML25M_MOVIES_FILE), "movielens-25m"
        )
        ratings = pl.read_csv(ratings_path).rename(
            {"userId": USER_ID, "movieId": ITEM_ID, "rating": RATING, "timestamp": TIMESTAMP}
        )
        movies = pl.read_csv(movies_path).rename(
            {"movieId": ITEM_ID, "title": RAW_TITLE, "genres": RAW_GENRES}
        )
        return InteractionDataset(
            interactions=ratings.select(INTERACTION_SCHEMA.names()).cast(INTERACTION_SCHEMA),
            items=_normalize_items(movies),
        )


def _require_files(dataset_dir: Path, names: tuple[str, str], dataset: str) -> tuple[Path, Path]:
    paths = (dataset_dir / names[0], dataset_dir / names[1])
    missing = [path.name for path in paths if not path.is_file()]
    if missing:
        raise DatasetNotFoundError(
            f"{dataset_dir} is missing {', '.join(missing)}; run 'hopfilter download {dataset}'"
        )
    return paths


def _read_ml1m_ratings(path: Path) -> pl.DataFrame:
    # Polars needs a single-byte separator, so split on ':' and keep every other column.
    raw = pl.read_csv(path, separator=":", has_header=False, quote_char=None)
    value_columns = raw.columns[:: len(ML1M_FIELD_SEPARATOR)]
    renamed = raw.select(value_columns).rename(
        dict(zip(value_columns, INTERACTION_SCHEMA, strict=True))
    )
    return renamed.cast(INTERACTION_SCHEMA)


def _read_ml1m_movies(path: Path) -> pl.DataFrame:
    lines = path.read_text(encoding=ML1M_ENCODING).splitlines()
    rows = [line.split(ML1M_FIELD_SEPARATOR) for line in lines if line]
    return pl.DataFrame(
        {
            ITEM_ID: [int(row[0]) for row in rows],
            RAW_TITLE: [row[1] for row in rows],
            RAW_GENRES: [row[2] for row in rows],
        }
    )


def _normalize_items(movies: pl.DataFrame) -> pl.DataFrame:
    raw_title = pl.col(RAW_TITLE).str.strip_chars()
    genres = (
        pl.col(RAW_GENRES).str.split(GENRE_SEPARATOR).list.filter(pl.element() != NO_GENRES_MARKER)
    )
    return movies.select(
        pl.col(ITEM_ID),
        raw_title.str.replace(TRAILING_YEAR_WITH_SPACE_PATTERN, "").alias(TITLE),
        raw_title.str.extract(TRAILING_YEAR_PATTERN, 1).alias(YEAR),
        genres.alias(GENRES),
    ).cast(ITEM_SCHEMA)
