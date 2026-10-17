from typing import Protocol

import torch
from torch import Tensor
from torch.nn import functional

from hopfilter.errors import InvalidConfigError
from hopfilter.model.frequency import ItemFrequencyEstimator

DEFAULT_TEMPERATURE = 0.05


class LossFunction(Protocol):
    """Loss over a batch of positive (user, item) pairs using the batch as negatives."""

    def __call__(self, user_vectors: Tensor, item_vectors: Tensor, item_indices: Tensor) -> Tensor:
        """Return the mean loss for ``B`` aligned user and item vectors."""
        ...


class InBatchSoftmaxLoss:
    """Softmax cross-entropy where row ``i`` must pick item ``i`` among the batch items."""

    def __init__(self, temperature: float = DEFAULT_TEMPERATURE) -> None:
        self.temperature = require_positive_temperature(temperature)

    def __call__(self, user_vectors: Tensor, item_vectors: Tensor, item_indices: Tensor) -> Tensor:
        """Return the mean in-batch softmax loss."""
        logits = in_batch_logits(user_vectors, item_vectors, self.temperature)
        return diagonal_cross_entropy(mask_duplicate_items(logits, item_indices))


class LogQCorrectedLoss:
    """In-batch softmax with every logit corrected by ``-log q`` of its column's item.

    In-batch negatives are drawn in proportion to popularity. Subtracting the log of
    each item's in-batch probability (Yi et al., RecSys 2019) removes that bias, so a
    popular item is no longer penalized just for appearing as a negative more often.
    Each call advances ``estimator`` by one step.
    """

    def __init__(
        self, estimator: ItemFrequencyEstimator, temperature: float = DEFAULT_TEMPERATURE
    ) -> None:
        self.temperature = require_positive_temperature(temperature)
        self._estimator = estimator

    def __call__(self, user_vectors: Tensor, item_vectors: Tensor, item_indices: Tensor) -> Tensor:
        """Return the mean logQ-corrected in-batch softmax loss."""
        log_q = self._estimator.observe(item_indices)
        logits = in_batch_logits(user_vectors, item_vectors, self.temperature)
        corrected = logits - log_q.unsqueeze(0)
        return diagonal_cross_entropy(mask_duplicate_items(corrected, item_indices))


def require_positive_temperature(temperature: float) -> float:
    """Return ``temperature`` or raise ``InvalidConfigError`` if it is not positive."""
    if temperature <= 0.0:
        raise InvalidConfigError(f"temperature must be positive, got {temperature}")
    return temperature


def in_batch_logits(user_vectors: Tensor, item_vectors: Tensor, temperature: float) -> Tensor:
    """``(B, B)`` similarity matrix ``U Vᵀ / temperature``."""
    return user_vectors @ item_vectors.T / temperature


def mask_duplicate_items(logits: Tensor, item_indices: Tensor) -> Tensor:
    """Set ``logits[i, j]`` to ``-inf`` where ``j != i`` holds the same item as ``i``.

    Without this, a batch containing an item twice would treat each copy as a
    negative for the other row's positive.
    """
    same_item = item_indices.unsqueeze(0) == item_indices.unsqueeze(1)
    diagonal = torch.eye(logits.shape[0], dtype=torch.bool, device=logits.device)
    return logits.masked_fill(same_item & ~diagonal, float("-inf"))


def diagonal_cross_entropy(logits: Tensor) -> Tensor:
    """Mean cross-entropy where the correct class of row ``i`` is column ``i``."""
    targets = torch.arange(logits.shape[0], device=logits.device)
    return functional.cross_entropy(logits, targets)
