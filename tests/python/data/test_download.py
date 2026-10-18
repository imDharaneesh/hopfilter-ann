import hashlib
import io
import zipfile
from pathlib import Path

import pytest

from hopfilter.data.download import (
    DATASETS,
    DatasetDownloader,
    DatasetSpec,
    HttpFetcher,
    extract_zip,
    verify_sha256,
)
from hopfilter.errors import DatasetIntegrityError, InvalidConfigError


def make_zip(members: dict[str, str]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as bundle:
        for name, content in members.items():
            bundle.writestr(name, content)
    return buffer.getvalue()


def spec_for(payload: bytes, directory_name: str = "toy") -> DatasetSpec:
    return DatasetSpec(
        name="toy-dataset",
        url="https://example.invalid/toy.zip",
        sha256=hashlib.sha256(payload).hexdigest(),
        directory_name=directory_name,
    )


class FakeFetcher:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.requested_urls: list[str] = []

    def fetch(self, url: str, destination: Path) -> None:
        self.requested_urls.append(url)
        destination.write_bytes(self.payload)


def test_download_extracts_and_cleans_up(tmp_path: Path) -> None:
    payload = make_zip({"toy/ratings.dat": "1::2::5::100\n"})
    fetcher = FakeFetcher(payload)
    downloader = DatasetDownloader(fetcher, tmp_path)

    dataset_dir = downloader.download(spec_for(payload))

    assert dataset_dir == tmp_path / "toy"
    assert (dataset_dir / "ratings.dat").read_text() == "1::2::5::100\n"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["toy"]


def test_download_skips_fetch_when_already_extracted(tmp_path: Path) -> None:
    payload = make_zip({"toy/ratings.dat": ""})
    fetcher = FakeFetcher(payload)
    downloader = DatasetDownloader(fetcher, tmp_path)

    downloader.download(spec_for(payload))
    downloader.download(spec_for(payload))

    assert len(fetcher.requested_urls) == 1


def test_single_file_dataset_is_stored_in_its_own_directory(tmp_path: Path) -> None:
    payload = b"\x89HDF\r\n\x1a\n fake hdf5 bytes"
    spec = DatasetSpec(
        name="toy-vectors",
        url="https://example.invalid/datasets/toy-vectors.hdf5",
        sha256=hashlib.sha256(payload).hexdigest(),
        directory_name="toy-vectors",
        archive="file",
    )

    dataset_dir = DatasetDownloader(FakeFetcher(payload), tmp_path).download(spec)

    assert spec.file_name == "toy-vectors.hdf5"
    assert (dataset_dir / "toy-vectors.hdf5").read_bytes() == payload
    assert sorted(path.name for path in tmp_path.iterdir()) == ["toy-vectors"]


def test_checksum_mismatch_raises_and_leaves_nothing(tmp_path: Path) -> None:
    payload = make_zip({"toy/ratings.dat": ""})
    tampered_spec = spec_for(b"different bytes")
    downloader = DatasetDownloader(FakeFetcher(payload), tmp_path)

    with pytest.raises(DatasetIntegrityError, match="does not match"):
        downloader.download(tampered_spec)

    assert list(tmp_path.iterdir()) == []


def test_archive_without_expected_directory_raises(tmp_path: Path) -> None:
    payload = make_zip({"other/ratings.dat": ""})
    downloader = DatasetDownloader(FakeFetcher(payload), tmp_path)

    with pytest.raises(DatasetIntegrityError, match="no toy/ folder"):
        downloader.download(spec_for(payload))

    assert list(tmp_path.iterdir()) == []


def test_extract_rejects_path_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "evil.zip"
    archive.write_bytes(make_zip({"../escaped.txt": "x"}))

    with pytest.raises(DatasetIntegrityError, match="unsafe member path"):
        extract_zip(archive, tmp_path / "out")

    assert not (tmp_path / "escaped.txt").exists()


def test_verify_sha256_accepts_matching_digest(tmp_path: Path) -> None:
    path = tmp_path / "file.bin"
    path.write_bytes(b"hopfilter")

    verify_sha256(path, hashlib.sha256(b"hopfilter").hexdigest())


def test_http_fetcher_refuses_non_https(tmp_path: Path) -> None:
    with pytest.raises(InvalidConfigError, match="only https"):
        HttpFetcher().fetch("http://example.invalid/data.zip", tmp_path / "data.zip")


@pytest.mark.parametrize("name", ["movielens-1m", "movielens-25m", "sift-128-euclidean"])
def test_registry_specs_are_https_with_full_sha256(name: str) -> None:
    spec = DATASETS[name]

    assert spec.name == name
    assert spec.url.startswith("https://")
    assert len(spec.sha256) == 64
    assert all(character in "0123456789abcdef" for character in spec.sha256)
