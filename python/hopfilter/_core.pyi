from os import PathLike

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

    def search(
        self, queries: npt.ArrayLike, k: int, ef: int = 64, threads: int = 0
    ) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float32]]:
        """Return ``(ids, distances)`` of shape ``(n, k)``, closest first.

        Rows are padded with id ``-1`` and distance ``inf`` when the index holds fewer
        than ``k`` vectors. Distances are squared L2, ``1 - dot`` or ``1 - cosine``.
        """

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
