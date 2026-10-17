import math

import pytest
import torch

from hopfilter.errors import InvalidConfigError
from hopfilter.model.losses import (
    InBatchSoftmaxLoss,
    diagonal_cross_entropy,
    in_batch_logits,
    mask_duplicate_items,
)

IDENTITY_2D = torch.eye(2)
DISTINCT_ITEMS = torch.tensor([0, 1])


def as_scalar(value: torch.Tensor) -> float:
    return float(value.item())


@pytest.mark.parametrize(
    ("temperature", "expected"),
    [
        (1.0, math.log(1 + math.exp(-1))),
        (0.5, math.log(1 + math.exp(-2))),
    ],
)
def test_orthogonal_pairs_match_hand_computed_loss(temperature: float, expected: float) -> None:
    loss = InBatchSoftmaxLoss(temperature)(IDENTITY_2D, IDENTITY_2D, DISTINCT_ITEMS)

    assert as_scalar(loss) == pytest.approx(expected, rel=1e-6)


def test_three_by_three_batch_matches_hand_computed_loss() -> None:
    users = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.6, 0.8]])
    items = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    logits = [[1.0, 0.0, 0.0], [0.0, 1.0, 1.0], [0.6, 0.8, 0.8]]
    expected = sum(
        -row[index] + math.log(sum(math.exp(value) for value in row))
        for index, row in enumerate(logits)
    ) / len(logits)

    loss = InBatchSoftmaxLoss(temperature=1.0)(users, items, torch.tensor([0, 1, 2]))

    assert as_scalar(loss) == pytest.approx(expected, rel=1e-6)


def test_duplicate_item_is_not_its_own_negative() -> None:
    users = IDENTITY_2D
    items = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    same_item_twice = torch.tensor([7, 7])

    loss = InBatchSoftmaxLoss(temperature=1.0)(users, items, same_item_twice)

    assert as_scalar(loss) == pytest.approx(0.0, abs=1e-7)


def test_mask_only_touches_off_diagonal_duplicates() -> None:
    logits = torch.zeros(3, 3)

    masked = mask_duplicate_items(logits, torch.tensor([4, 5, 4]))

    negative_infinity = float("-inf")
    expected = torch.tensor(
        [[0.0, 0.0, negative_infinity], [0.0, 0.0, 0.0], [negative_infinity, 0.0, 0.0]]
    )
    torch.testing.assert_close(masked, expected)


def test_logits_are_scaled_dot_products() -> None:
    users = torch.tensor([[1.0, 2.0]])
    items = torch.tensor([[3.0, 4.0], [5.0, 6.0]])

    torch.testing.assert_close(in_batch_logits(users, items, 0.5), torch.tensor([[22.0, 34.0]]))


def test_cross_entropy_targets_the_diagonal() -> None:
    confident = torch.tensor([[50.0, 0.0], [0.0, 50.0]])

    assert as_scalar(diagonal_cross_entropy(confident)) == pytest.approx(0.0, abs=1e-6)


def test_gradient_pulls_user_towards_its_positive() -> None:
    users = torch.tensor([[0.6, 0.8], [0.8, 0.6]], requires_grad=True)

    InBatchSoftmaxLoss(temperature=1.0)(users, IDENTITY_2D, DISTINCT_ITEMS).backward()

    assert users.grad is not None
    assert users.grad[0, 0] < 0
    assert users.grad[1, 1] < 0


@pytest.mark.parametrize("temperature", [0.0, -0.1])
def test_non_positive_temperature_is_rejected(temperature: float) -> None:
    with pytest.raises(InvalidConfigError, match="temperature must be positive"):
        InBatchSoftmaxLoss(temperature)
