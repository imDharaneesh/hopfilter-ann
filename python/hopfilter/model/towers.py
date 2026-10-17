from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional

from hopfilter.errors import InvalidConfigError

DEFAULT_EMBEDDING_DIM = 64
DEFAULT_HIDDEN_DIMS = (128,)
DEFAULT_OUTPUT_DIM = 64
FEATURE_DIM = -1
ITEM_FEATURE_GROUPS = 2


@dataclass(frozen=True)
class TowerConfig:
    """Shape of one tower: an embedding of ``embedding_dim`` followed by an MLP."""

    embedding_dim: int = DEFAULT_EMBEDDING_DIM
    hidden_dims: tuple[int, ...] = DEFAULT_HIDDEN_DIMS
    output_dim: int = DEFAULT_OUTPUT_DIM
    dropout: float = 0.0

    def __post_init__(self) -> None:
        sizes = (self.embedding_dim, self.output_dim, *self.hidden_dims)
        if any(size <= 0 for size in sizes):
            raise InvalidConfigError(f"tower dimensions must be positive, got {sizes}")
        if not 0.0 <= self.dropout < 1.0:
            raise InvalidConfigError(f"dropout must be in [0, 1), got {self.dropout}")


def build_mlp(input_dim: int, config: TowerConfig) -> nn.Sequential:
    """ReLU MLP from ``input_dim`` through ``config.hidden_dims`` to ``config.output_dim``."""
    layers: list[nn.Module] = []
    previous_dim = input_dim
    for hidden_dim in config.hidden_dims:
        layers += [nn.Linear(previous_dim, hidden_dim), nn.ReLU(), nn.Dropout(config.dropout)]
        previous_dim = hidden_dim
    layers.append(nn.Linear(previous_dim, config.output_dim))
    return nn.Sequential(*layers)


class UserTower(nn.Module):
    """Maps user indices to L2-normalized vectors."""

    def __init__(self, num_users: int, config: TowerConfig) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_users, config.embedding_dim)
        self.mlp = build_mlp(config.embedding_dim, config)

    def forward(self, user_indices: Tensor) -> Tensor:
        """Return a ``(batch, output_dim)`` unit-norm tensor."""
        hidden: Tensor = self.mlp(self.embedding(user_indices))
        return functional.normalize(hidden, dim=FEATURE_DIM)


class ItemTower(nn.Module):
    """Maps item indices to L2-normalized vectors from id and genre features.

    The genre feature is the mean of the item's genre embeddings, so rare items share
    signal with popular items of the same genres.
    """

    genre_weights: Tensor

    def __init__(self, genre_matrix: Tensor, config: TowerConfig) -> None:
        super().__init__()
        num_items, num_genres = genre_matrix.shape
        genre_counts = genre_matrix.sum(dim=FEATURE_DIM, keepdim=True).clamp(min=1.0)
        self.register_buffer("genre_weights", genre_matrix / genre_counts)
        self.id_embedding = nn.Embedding(num_items, config.embedding_dim)
        self.genre_embedding = nn.Linear(num_genres, config.embedding_dim, bias=False)
        self.mlp = build_mlp(ITEM_FEATURE_GROUPS * config.embedding_dim, config)

    @property
    def num_items(self) -> int:
        """Number of items the tower can encode."""
        return int(self.genre_weights.shape[0])

    def forward(self, item_indices: Tensor) -> Tensor:
        """Return a ``(batch, output_dim)`` unit-norm tensor."""
        genre_features = self.genre_embedding(self.genre_weights[item_indices])
        features = torch.cat([self.id_embedding(item_indices), genre_features], dim=FEATURE_DIM)
        hidden: Tensor = self.mlp(features)
        return functional.normalize(hidden, dim=FEATURE_DIM)


class TwoTowerModel(nn.Module):
    """A user tower and an item tower that share an output space."""

    def __init__(self, user_tower: UserTower, item_tower: ItemTower) -> None:
        super().__init__()
        self.user_tower = user_tower
        self.item_tower = item_tower

    def forward(self, user_indices: Tensor, item_indices: Tensor) -> tuple[Tensor, Tensor]:
        """Encode a batch of (user, item) pairs."""
        return self.encode_users(user_indices), self.item_tower(item_indices)

    def encode_users(self, user_indices: Tensor) -> Tensor:
        """Encode user indices with the user tower."""
        vectors: Tensor = self.user_tower(user_indices)
        return vectors

    def encode_all_items(self) -> Tensor:
        """Encode every item, in index order."""
        device = self.item_tower.genre_weights.device
        vectors: Tensor = self.item_tower(torch.arange(self.item_tower.num_items, device=device))
        return vectors
