import argparse
from collections.abc import Sequence

from hopfilter import __version__

PROGRAM_NAME = "hopfilter"


def build_parser() -> argparse.ArgumentParser:
    """Create the top-level argument parser for the ``hopfilter`` command."""
    parser = argparse.ArgumentParser(
        prog=PROGRAM_NAME,
        description="Two-tower retrieval with a filter-aware HNSW index.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line interface and return a process exit code."""
    parser = build_parser()
    parser.parse_args(argv)
    parser.print_help()
    return 0
