import numpy as np
import pytest
import torch

from hopfilter.errors import EmptyDatasetError, InvalidConfigError
from hopfilter.model.losses import InBatchSoftmaxLoss
from hopfilter.model.towers import ItemTower, TowerConfig, TwoTowerModel, UserTower
from hopfilter.model.training import (
    InteractionBatcher,
    Trainer,
    TrainerConfig,
    TrainingHistory,
    make_optimizer,
)

NUM_USERS = 8
NUM_ITEMS = 8
TOY_CONFIG = TowerConfig(embedding_dim=8, hidden_dims=(16,), output_dim=8)


def toy_pairs(repeats: int = 16) -> tuple[np.ndarray, np.ndarray]:
    users = np.tile(np.arange(NUM_USERS, dtype=np.int64), repeats)
    return users, users.copy()


def train_toy(seed: int, epochs: int = 15) -> tuple[TwoTowerModel, TrainingHistory]:
    torch.manual_seed(seed)
    model = TwoTowerModel(
        UserTower(NUM_USERS, TOY_CONFIG), ItemTower(torch.zeros(NUM_ITEMS, 1), TOY_CONFIG)
    )
    config = TrainerConfig(epochs=epochs, batch_size=8, learning_rate=0.05, seed=seed)
    users, items = toy_pairs()
    batcher = InteractionBatcher(users, items, config.batch_size, config.seed)
    trainer = Trainer(model, InBatchSoftmaxLoss(0.1), make_optimizer(model, config), epochs)
    return model, trainer.fit(batcher)


def test_training_reduces_loss_and_learns_the_toy_mapping() -> None:
    model, history = train_toy(seed=0)

    assert history.epoch_losses[-1] < 0.5 * history.epoch_losses[0]
    with torch.no_grad():
        scores = model.encode_users(torch.arange(NUM_USERS)) @ model.encode_all_items().T
    assert scores.argmax(dim=1).tolist() == list(range(NUM_ITEMS))


def test_same_seed_reproduces_the_loss_curve() -> None:
    _, first = train_toy(seed=4, epochs=3)
    _, second = train_toy(seed=4, epochs=3)

    assert first.epoch_losses == second.epoch_losses


def test_epoch_callback_sees_eval_mode_and_each_loss() -> None:
    torch.manual_seed(0)
    model = TwoTowerModel(
        UserTower(NUM_USERS, TOY_CONFIG), ItemTower(torch.zeros(NUM_ITEMS, 1), TOY_CONFIG)
    )
    users, items = toy_pairs()
    seen: list[tuple[int, float, bool]] = []
    config = TrainerConfig(epochs=2, batch_size=8)
    trainer = Trainer(model, InBatchSoftmaxLoss(), make_optimizer(model, config), config.epochs)

    history = trainer.fit(
        InteractionBatcher(users, items, 8, seed=0),
        on_epoch_end=lambda epoch, loss: seen.append((epoch, loss, model.training)),
    )

    assert seen == [(0, history.epoch_losses[0], False), (1, history.epoch_losses[1], False)]


def test_batcher_covers_full_batches_once_per_epoch_and_reshuffles() -> None:
    users = np.arange(10, dtype=np.int64)
    batcher = InteractionBatcher(users, users + 100, batch_size=3, seed=1)

    first_epoch = [batch.user_indices.tolist() for batch in batcher]
    second_epoch = [batch.user_indices.tolist() for batch in batcher]

    assert len(batcher) == 3
    assert all(len(rows) == 3 for rows in first_epoch)
    flattened = [row for rows in first_epoch for row in rows]
    assert len(set(flattened)) == 9
    assert first_epoch != second_epoch


def test_batcher_keeps_users_and_items_aligned() -> None:
    users = np.arange(6, dtype=np.int64)
    batcher = InteractionBatcher(users, users * 10, batch_size=2, seed=0)

    for batch in batcher:
        torch.testing.assert_close(batch.item_indices, batch.user_indices * 10)


def test_batcher_rejects_too_few_pairs_and_mismatched_shapes() -> None:
    users = np.arange(3, dtype=np.int64)

    with pytest.raises(EmptyDatasetError, match="cannot fill one batch"):
        InteractionBatcher(users, users, batch_size=4, seed=0)
    with pytest.raises(InvalidConfigError, match="same shape"):
        InteractionBatcher(users, users[:2], batch_size=1, seed=0)


@pytest.mark.parametrize(
    "kwargs",
    [{"epochs": 0}, {"batch_size": -1}, {"learning_rate": 0.0}, {"weight_decay": -0.1}],
)
def test_invalid_trainer_config_is_rejected(kwargs: dict[str, float]) -> None:
    with pytest.raises(InvalidConfigError):
        TrainerConfig(**kwargs)
