from collections.abc import Sequence
from os import PathLike
from typing import ClassVar

import numpy as np
import numpy.typing as npt

__version__: str

def version() -> str: ...
def simd_kernels() -> str:
    """Name of the distance kernel set in use, e.g. ``avx2``."""

def available_simd_kernels() -> list[str]:
    """Kernel sets this machine supports, ``scalar`` first."""

def use_simd_kernels(name: str) -> None:
    """Switch every distance computation to the kernel set ``name``."""

class Filter:
    """Attribute filter for ``HnswIndex.search``, combined with ``&``, ``|`` and ``~``.

    A filter is an immutable expression; it is checked against the index's attribute
    columns when a search starts, so one filter can be reused across indexes.
    """

    @staticmethod
    def equals(column: str, value: int) -> Filter:
        """Rows whose categorical ``column`` holds the code ``value``."""

    @staticmethod
    def between(column: str, low: float, high: float) -> Filter:
        """Rows whose numeric ``column`` lies in the closed range ``[low, high]``."""

    @staticmethod
    def mask(mask: npt.ArrayLike) -> Filter:
        """Rows whose position is ``True`` in a one-dimensional boolean array."""

    @staticmethod
    def all_of(filters: Sequence[Filter]) -> Filter:
        """Rows matching every filter in ``filters``."""

    @staticmethod
    def any_of(filters: Sequence[Filter]) -> Filter:
        """Rows matching at least one filter in ``filters``."""

    def __and__(self, other: Filter) -> Filter: ...
    def __or__(self, other: Filter) -> Filter: ...
    def __invert__(self) -> Filter: ...
    def __bool__(self) -> bool:
        """Always raises ``TypeError``: use ``&``, ``|`` and ``~``, not ``and``, ``or``, ``not``."""

class HnswIndex:
    """HNSW index over float32 vectors, implemented in C++.

    ``add`` and ``search`` release the GIL and parallelize internally (``threads=0`` uses
    every core). Several Python threads may also search at once, or add at once;
    searching while another thread adds is not supported. With ``threads=1`` the graph is
    deterministic for a given seed and insertion order.
    """

    def __init__(
        self,
        dimension: int,
        capacity: int,
        metric: str = "l2",
        max_neighbors: int = 16,
        ef_construction: int = 200,
        seed: int = 42,
    ) -> None:
        """Create an empty index; ``metric`` is ``"l2"``, ``"ip"`` or ``"cosine"``."""

    def add(self, vectors: npt.ArrayLike, threads: int = 0) -> npt.NDArray[np.int64]:
        """Insert an ``(n, dimension)`` batch and return the new ids in row order.

        The whole batch is rejected with ``IndexStateError`` if it does not fit.
        """

    strategies: ClassVar[list[str]]

    def search(
        self,
        queries: npt.ArrayLike,
        k: int,
        ef: int = 64,
        threads: int = 0,
        filter: Filter | None = None,
        strategy: str = "adaptive",
    ) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float32]]:
        """Return ``(ids, distances)`` of shape ``(n, k)``, closest first.

        Rows are padded with id ``-1`` and distance ``inf`` when fewer than ``k`` vectors
        (or matching vectors) are found. Distances are squared L2, ``1 - dot`` or
        ``1 - cosine``. ``strategy`` is one of ``HnswIndex.strategies``; ``adaptive``
        picks post-filtering, in-graph filtering or exact search from the filter's
        estimated selectivity.
        """

    def set_attribute(self, name: str, values: npt.ArrayLike) -> None:
        """Attach one value per row: integers become a categorical column, floats numeric.

        Rows past the end of ``values`` match no filter on ``name``.
        """

    def plan_filter(self, filter: Filter, ef: int = 64) -> str:
        """Strategy ``adaptive`` picks for ``filter`` from global selectivity alone."""

    def plan_queries(self, queries: npt.ArrayLike, filter: Filter, ef: int = 64) -> list[str]:
        """Strategy ``adaptive`` would run for each query, after its neighbourhood check.

        Differs from ``plan_filter`` only for in-graph plans: a query whose neighbourhood
        holds far fewer matches than the filter overall is sent to ``pre_filter``.
        """

    def count_matching(self, filter: Filter) -> int:
        """Exact number of stored vectors that match ``filter``."""

    @property
    def attribute_names(self) -> list[str]: ...

    def save(self, path: str | PathLike[str]) -> None:
        """Write the index to ``path`` in hopfilter's binary format."""

    @staticmethod
    def load(path: str | PathLike[str]) -> HnswIndex:
        """Read an index written by ``save``."""

    def __len__(self) -> int: ...
    @property
    def dimension(self) -> int: ...
    @property
    def capacity(self) -> int: ...
    @property
    def metric(self) -> str: ...
    @property
    def max_neighbors(self) -> int: ...
    @property
    def ef_construction(self) -> int: ...
