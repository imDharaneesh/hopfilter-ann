import hashlib
import shutil
import urllib.request
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Literal, Protocol
from urllib.parse import urlparse

from hopfilter.errors import DatasetIntegrityError, InvalidConfigError

CHUNK_SIZE_BYTES = 1 << 20
DEFAULT_TIMEOUT_SECONDS = 60.0
ALLOWED_URL_SCHEMES = ("https://",)
PARTIAL_SUFFIX = ".part"
STAGING_PREFIX = ".staging-"
EXTRACTED_DIRECTORY = "extracted"


@dataclass(frozen=True)
class DatasetSpec:
    """Where a dataset lives and how to verify it.

    Attributes:
        name: Registry name used on the command line, e.g. ``movielens-1m``.
        url: HTTPS URL of the download.
        sha256: Expected SHA-256 of the download, lower-case hex.
        directory_name: Directory the dataset ends up in under the data directory.
        archive: ``"zip"`` to extract an archive whose top-level folder is
            ``directory_name``; ``"file"`` to store a single file inside it.
    """

    name: str
    url: str
    sha256: str
    directory_name: str
    archive: Literal["zip", "file"] = "zip"

    @property
    def file_name(self) -> str:
        """Last path component of the URL, e.g. ``sift-128-euclidean.hdf5``."""
        return PurePosixPath(urlparse(self.url).path).name


class UrlFetcher(Protocol):
    """Copies the resource at ``url`` into the file ``destination``."""

    def fetch(self, url: str, destination: Path) -> None:
        """Download ``url`` to ``destination``, overwriting it."""
        ...


class HttpFetcher:
    """``UrlFetcher`` backed by the standard library, restricted to HTTPS."""

    def __init__(self, timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS) -> None:
        self._timeout_seconds = timeout_seconds

    def fetch(self, url: str, destination: Path) -> None:
        """Stream ``url`` to ``destination`` in fixed-size chunks."""
        if not url.startswith(ALLOWED_URL_SCHEMES):
            raise InvalidConfigError(f"only https downloads are allowed, got {url}")
        with (
            urllib.request.urlopen(url, timeout=self._timeout_seconds) as response,
            destination.open("wb") as sink,
        ):
            shutil.copyfileobj(response, sink, CHUNK_SIZE_BYTES)


class DatasetDownloader:
    """Downloads, verifies and unpacks datasets under ``data_dir``."""

    def __init__(self, fetcher: UrlFetcher, data_dir: Path) -> None:
        self._fetcher = fetcher
        self._data_dir = data_dir

    def dataset_dir(self, spec: DatasetSpec) -> Path:
        """Directory the dataset ends up in."""
        return self._data_dir / spec.directory_name

    def download(self, spec: DatasetSpec) -> Path:
        """Fetch ``spec`` unless already present, and return its directory.

        Everything happens in a staging directory that is only renamed into place once
        the checksum and layout are verified.

        Raises:
            DatasetIntegrityError: The checksum does not match or the archive layout
                is unexpected. Nothing is left behind in ``data_dir``.
        """
        target = self.dataset_dir(spec)
        if target.is_dir():
            return target
        staging = self._data_dir / f"{STAGING_PREFIX}{spec.directory_name}"
        try:
            _promote(self._stage(spec, staging), target, spec)
        finally:
            shutil.rmtree(staging, ignore_errors=True)
        return target

    def _stage(self, spec: DatasetSpec, staging: Path) -> Path:
        staging.mkdir(parents=True, exist_ok=True)
        payload = staging / f"{spec.file_name}{PARTIAL_SUFFIX}"
        self._fetcher.fetch(spec.url, payload)
        verify_sha256(payload, spec.sha256)
        if spec.archive == "zip":
            extract_zip(payload, staging / EXTRACTED_DIRECTORY)
            return staging / EXTRACTED_DIRECTORY / spec.directory_name
        folder = staging / spec.directory_name
        folder.mkdir()
        payload.rename(folder / spec.file_name)
        return folder


def verify_sha256(path: Path, expected: str) -> None:
    """Raise ``DatasetIntegrityError`` unless ``path`` hashes to ``expected``."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(CHUNK_SIZE_BYTES), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != expected:
        raise DatasetIntegrityError(f"{path.name}: sha256 {actual} does not match {expected}")


def extract_zip(archive: Path, destination: Path) -> None:
    """Extract ``archive`` into ``destination``, rejecting members that escape it."""
    root = destination.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.namelist():
            if not (root / member).resolve().is_relative_to(root):
                raise DatasetIntegrityError(f"{archive.name}: unsafe member path {member}")
        bundle.extractall(root)


def _promote(extracted: Path, target: Path, spec: DatasetSpec) -> None:
    if not extracted.is_dir():
        raise DatasetIntegrityError(f"{spec.name}: archive has no {spec.directory_name}/ folder")
    extracted.rename(target)


DATASETS: Mapping[str, DatasetSpec] = MappingProxyType(
    {
        spec.name: spec
        for spec in (
            DatasetSpec(
                name="movielens-1m",
                url="https://files.grouplens.org/datasets/movielens/ml-1m.zip",
                sha256="a6898adb50b9ca05aa231689da44c217cb524e7ebd39d264c56e2832f2c54e20",
                directory_name="ml-1m",
            ),
            DatasetSpec(
                name="movielens-25m",
                url="https://files.grouplens.org/datasets/movielens/ml-25m.zip",
                sha256="8b21cfb7eb1706b4ec0aac894368d90acf26ebdfb6aced3ebd4ad5bd1eb9c6aa",
                directory_name="ml-25m",
            ),
            DatasetSpec(
                name="sift-128-euclidean",
                url="https://ann-benchmarks.com/sift-128-euclidean.hdf5",
                sha256="dd6f0a6ed6b7ebb8934680f861a33ed01ff33991eaee4fd60914d854a0ca5984",
                directory_name="sift-128-euclidean",
                archive="file",
            ),
        )
    }
)
