import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from hopfilter import __version__
from hopfilter.data.download import DATASETS, DatasetDownloader, HttpFetcher, UrlFetcher
from hopfilter.errors import HopfilterError

PROGRAM_NAME = "hopfilter"
DEFAULT_DATA_DIR = Path("data")
EXIT_SUCCESS = 0
EXIT_FAILURE = 1

type SubParsers = argparse._SubParsersAction[argparse.ArgumentParser]


def build_parser() -> argparse.ArgumentParser:
    """Create the top-level argument parser for the ``hopfilter`` command."""
    parser = argparse.ArgumentParser(
        prog=PROGRAM_NAME,
        description="Two-tower retrieval with a filter-aware HNSW index.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", metavar="COMMAND")
    _add_download_command(commands)
    return parser


def _add_download_command(commands: SubParsers) -> None:
    download = commands.add_parser("download", help="fetch and verify a dataset")
    download.add_argument("dataset", choices=sorted(DATASETS))
    download.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)


def run_download(dataset_name: str, data_dir: Path, fetcher: UrlFetcher) -> Path:
    """Download the registered dataset ``dataset_name`` and return its directory."""
    return DatasetDownloader(fetcher, data_dir).download(DATASETS[dataset_name])


def main(argv: Sequence[str] | None = None, fetcher: UrlFetcher | None = None) -> int:
    """Run the command line interface and return a process exit code."""
    parser = build_parser()
    arguments = parser.parse_args(argv)
    if arguments.command is None:
        parser.print_help()
        return EXIT_SUCCESS
    try:
        dataset_dir = run_download(arguments.dataset, arguments.data_dir, fetcher or HttpFetcher())
    except HopfilterError as error:
        print(f"error: {error}", file=sys.stderr)
        return EXIT_FAILURE
    print(dataset_dir)
    return EXIT_SUCCESS
