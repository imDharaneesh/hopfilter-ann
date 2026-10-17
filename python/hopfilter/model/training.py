from collections.abc import Callable, Iterator
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor

from hopfilter.errors import EmptyDatasetError, InvalidConfigError
from hopfilter.model.losses import LossFunction
from hopfilter.model.towers import TwoTowerModel

DEFAULT_EPOCHS = 10
DEFAULT_BATCH_SIZE = 1024
DEFAULT_LEARNING_RATE = 1e-3

type EpochCallback = Callable[[int, float], None]


@dataclass(frozen=True)
class TrainerConfig:
    """Optimization settings for one training run."""

    epochs: int = DEFAULT_EPOCHS
    batch_size: int = DEFAULT_BATCH_SIZE
    learning_rate: float = DEFAULT_LEARNING_RATE
    weight_decay: float = 0.0
    seed: int = 0

    def __post_init__(self) -> None:
        if self.epochs <= 0 or self.batch_size <= 0:
            raise InvalidConfigError("epochs and batch_size must be positive")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise InvalidConfigError("learning_rate must be positive and weight_decay >= 0")


@dataclass(frozen=True)
class Batch:
    """Aligned user and item indices of one batch of positive pairs."""

    user_indices: Tensor
    item_indices: Tensor


@dataclass(frozen=True)
class TrainingHistory:
    """Mean training loss of each epoch, in order."""

    epoch_losses: tuple[float, ...]


class InteractionBatcher:
    """Yields shuffled fixed-size batches of (user, item) pairs, one pass per iteration.

    The final partial batch is dropped so every step sees exactly ``batch_size`` pairs,
    which keeps in-batch frequency estimates comparable across steps.
    """

    def __init__(
        self,
        user_indices: npt.NDArray[np.int64],
        item_indices: npt.NDArray[np.int64],
        batch_size: int,
        seed: int,
    ) -> None:
        if user_indices.shape != item_indices.shape:
            raise InvalidConfigError("user_indices and item_indices must have the same shape")
        if user_indices.size < batch_size:
            raise EmptyDatasetError(
                f"{user_indices.size} pairs cannot fill one batch of {batch_size}"
            )
        self._users = torch.from_numpy(user_indices)
        self._items = torch.from_numpy(item_indices)
        self._batch_size = batch_size
        self._generator = torch.Generator().manual_seed(seed)

    def __len__(self) -> int:
        return int(self._users.numel()) // self._batch_size

    def __iter__(self) -> Iterator[Batch]:
        order = torch.randperm(self._users.numel(), generator=self._generator)
        for batch_number in range(len(self)):
            rows = order[batch_number * self._batch_size : (batch_number + 1) * self._batch_size]
            yield Batch(user_indices=self._users[rows], item_indices=self._items[rows])


def make_optimizer(model: TwoTowerModel, config: TrainerConfig) -> torch.optim.Optimizer:
    """AdamW over every model parameter with the configured learning rate and decay."""
    return torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )


class Trainer:
    """Runs epochs of gradient descent with an injected model, loss and optimizer."""

    def __init__(
        self,
        model: TwoTowerModel,
        loss: LossFunction,
        optimizer: torch.optim.Optimizer,
        epochs: int,
    ) -> None:
        self._model = model
        self._loss = loss
        self._optimizer = optimizer
        self._epochs = epochs

    def fit(
        self, batcher: InteractionBatcher, on_epoch_end: EpochCallback | None = None
    ) -> TrainingHistory:
        """Train for the configured number of epochs.

        Args:
            batcher: Source of batches; iterated once per epoch.
            on_epoch_end: Called with ``(epoch, mean_loss)`` after each epoch, with the
                model in eval mode.
        """
        losses: list[float] = []
        for epoch in range(self._epochs):
            self._model.train()
            losses.append(self._run_epoch(batcher))
            self._model.eval()
            if on_epoch_end is not None:
                on_epoch_end(epoch, losses[-1])
        return TrainingHistory(epoch_losses=tuple(losses))

    def _run_epoch(self, batcher: InteractionBatcher) -> float:
        total_loss = 0.0
        for batch in batcher:
            self._optimizer.zero_grad(set_to_none=True)
            user_vectors, item_vectors = self._model(batch.user_indices, batch.item_indices)
            loss = self._loss(user_vectors, item_vectors, batch.item_indices)
            torch.autograd.backward(loss)
            self._optimizer.step()
            total_loss += float(loss.detach())
        return total_loss / len(batcher)
