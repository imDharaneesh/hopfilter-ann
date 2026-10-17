import pytest
import torch

from hopfilter.errors import InvalidConfigError
from hopfilter.model.towers import ItemTower, TowerConfig, TwoTowerModel, UserTower, build_mlp

SMALL_CONFIG = TowerConfig(embedding_dim=8, hidden_dims=(16,), output_dim=4)
GENRES = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])


def make_model(seed: int = 0) -> TwoTowerModel:
    torch.manual_seed(seed)
    return TwoTowerModel(UserTower(5, SMALL_CONFIG), ItemTower(GENRES, SMALL_CONFIG))


def test_towers_produce_unit_vectors_of_output_dim() -> None:
    model = make_model()

    users, items = model(torch.tensor([0, 4]), torch.tensor([0, 2]))

    assert users.shape == items.shape == (2, 4)
    torch.testing.assert_close(users.norm(dim=-1), torch.ones(2))
    torch.testing.assert_close(items.norm(dim=-1), torch.ones(2))


def test_encode_all_items_matches_per_item_encoding() -> None:
    model = make_model()

    all_items = model.encode_all_items()

    assert all_items.shape == (3, 4)
    torch.testing.assert_close(all_items[1:2], model.item_tower(torch.tensor([1])))


def test_genre_weights_are_mean_pooled_and_empty_rows_stay_zero() -> None:
    tower = ItemTower(GENRES, SMALL_CONFIG)

    torch.testing.assert_close(
        tower.genre_weights,
        torch.tensor([[0.5, 0.0, 0.5], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]),
    )
    assert tower.num_items == 3


def test_items_differing_only_in_genre_get_different_vectors() -> None:
    tower = ItemTower(GENRES, SMALL_CONFIG)
    with torch.no_grad():
        tower.id_embedding.weight.zero_()

    vectors = tower(torch.tensor([0, 1]))

    assert not torch.allclose(vectors[0], vectors[1])


def test_same_seed_gives_identical_parameters() -> None:
    first, second = make_model(seed=3), make_model(seed=3)

    for left, right in zip(first.parameters(), second.parameters(), strict=True):
        torch.testing.assert_close(left, right)


def test_build_mlp_layer_shapes() -> None:
    mlp = build_mlp(10, TowerConfig(embedding_dim=8, hidden_dims=(32, 16), output_dim=4))

    linear_shapes = [
        tuple(layer.weight.shape) for layer in mlp if isinstance(layer, torch.nn.Linear)
    ]

    assert linear_shapes == [(32, 10), (16, 32), (4, 16)]


@pytest.mark.parametrize(
    "kwargs",
    [{"embedding_dim": 0}, {"output_dim": -1}, {"hidden_dims": (8, 0)}, {"dropout": 1.0}],
)
def test_invalid_tower_config_is_rejected(kwargs: dict[str, object]) -> None:
    with pytest.raises(InvalidConfigError):
        TowerConfig(**kwargs)
