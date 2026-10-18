from os import PathLike

import numpy as np
import numpy.typing as npt

__version__: str

def version() -> str: ...

class HnswIndex:
    """HNSW index over float32 vectors, implemented in C++.

    Searches may run concurrently from several threads; ``add`` calls are serialized;
    searching while another thread adds is not supported. The GIL is released inside
    ``add`` and ``search``.
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

    def add(self, vectors: npt.ArrayLike) -> npt.NDArray[np.int64]:
        """Insert an ``(n, dimension)`` batch and return the new ids in order."""

    def search(
        self, queries: npt.ArrayLike, k: int, ef: int = 64
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
