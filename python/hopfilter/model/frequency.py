import torch
from torch import Tensor

from hopfilter.errors import InvalidConfigError

DEFAULT_LEARNING_RATE = 0.01
STATE_DTYPE = torch.float64
OUTPUT_DTYPE = torch.float32


class ItemFrequencyEstimator:
    """Streaming estimate of the probability that an item appears in a training batch.

    Follows Algorithm 2 of Yi et al. (RecSys 2019). For each item the estimator keeps
    the step it was last seen at and a moving average of the gap between sightings;
    the probability estimate is the inverse of that gap. Items are already encoded to
    contiguous indices, so the paper's hash function is the identity.

    An item repeated within one batch counts as a single sighting, because the
    quantity being estimated is "appears in the batch", not "appears once per row".
    """

    def __init__(
        self,
        num_items: int,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        initial_gap: float | None = None,
    ) -> None:
        if num_items <= 0:
            raise InvalidConfigError(f"num_items must be positive, got {num_items}")
        if not 0.0 < learning_rate <= 1.0:
            raise InvalidConfigError(f"learning_rate must be in (0, 1], got {learning_rate}")
        starting_gap = float(num_items) if initial_gap is None else initial_gap
        if starting_gap < 1.0:
            raise InvalidConfigError(f"initial_gap must be at least 1, got {starting_gap}")
        self._learning_rate = learning_rate
        self._last_seen_step = torch.zeros(num_items, dtype=STATE_DTYPE)
        self._average_gap = torch.full((num_items,), starting_gap, dtype=STATE_DTYPE)
        self._step = 0

    @property
    def step(self) -> int:
        """Number of batches observed so far."""
        return self._step

    def observe(self, item_indices: Tensor) -> Tensor:
        """Record one training step and return ``log q`` for ``item_indices``.

        The returned tensor is float32, on the same device as ``item_indices``, and
        already reflects this step's update.
        """
        self._step += 1
        seen = torch.unique(item_indices.detach().cpu())
        gap_since_last_seen = self._step - self._last_seen_step[seen]
        previous_gap = self._average_gap[seen]
        self._average_gap[seen] = previous_gap + self._learning_rate * (
            gap_since_last_seen - previous_gap
        )
        self._last_seen_step[seen] = float(self._step)
        return self.log_probabilities(item_indices)

    def log_probabilities(self, item_indices: Tensor) -> Tensor:
        """Current ``log q`` for ``item_indices`` without updating any state."""
        gaps = self._average_gap[item_indices.detach().cpu()]
        return (-torch.log(gaps)).to(dtype=OUTPUT_DTYPE, device=item_indices.device)

    def probabilities(self) -> Tensor:
        """Current ``q`` for every item, in index order."""
        return (1.0 / self._average_gap).to(OUTPUT_DTYPE)
