import math

import pytest
import torch

from hopfilter.errors import InvalidConfigError
from hopfilter.model.frequency import ItemFrequencyEstimator
from hopfilter.model.losses import InBatchSoftmaxLoss, LogQCorrectedLoss

IDENTITY_2D = torch.eye(2)
DISTINCT_ITEMS = torch.tensor([0, 1])
EXACT_GAP_LEARNING_RATE = 1.0


def test_matches_hand_computed_corrected_loss() -> None:
    estimator = ItemFrequencyEstimator(num_items=2, learning_rate=EXACT_GAP_LEARNING_RATE)
    estimator.observe(torch.tensor([0]))
    loss_function = LogQCorrectedLoss(estimator, temperature=1.0)

    loss = loss_function(IDENTITY_2D, IDENTITY_2D, DISTINCT_ITEMS)

    # Item 0 was seen at steps 1 and 2 (q = 1); item 1 only at step 2 (gap 2, q = 1/2).
    row_zero = -1.0 + math.log(math.e + 2.0)
    row_one = -(1.0 + math.log(2.0)) + math.log(1.0 + 2.0 * math.e)
    assert float(loss.item()) == pytest.approx((row_zero + row_one) / 2, rel=1e-6)


def test_equal_frequencies_reduce_to_plain_in_batch_softmax() -> None:
    users = torch.tensor([[0.6, 0.8], [1.0, 0.0], [0.0, 1.0]])
    items = torch.tensor([[1.0, 0.0], [0.8, 0.6], [0.0, 1.0]])
    item_indices = torch.tensor([0, 1, 2])
    estimator = ItemFrequencyEstimator(num_items=3, learning_rate=EXACT_GAP_LEARNING_RATE)

    corrected = LogQCorrectedLoss(estimator, temperature=0.2)(users, items, item_indices)
    plain = InBatchSoftmaxLoss(temperature=0.2)(users, items, item_indices)

    torch.testing.assert_close(corrected, plain)


def test_popular_negative_is_penalized_less_than_without_correction() -> None:
    estimator = ItemFrequencyEstimator(num_items=2, learning_rate=EXACT_GAP_LEARNING_RATE)
    estimator.observe(torch.tensor([1]))
    users = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)

    LogQCorrectedLoss(estimator, temperature=1.0)(users, IDENTITY_2D, DISTINCT_ITEMS).backward()
    assert users.grad is not None
    corrected_push_from_popular = float(users.grad[0, 1])

    users.grad = None
    InBatchSoftmaxLoss(temperature=1.0)(users, IDENTITY_2D, DISTINCT_ITEMS).backward()
    assert users.grad is not None
    plain_push_from_popular = float(users.grad[0, 1])

    assert 0 < corrected_push_from_popular < plain_push_from_popular


def test_each_call_advances_the_estimator() -> None:
    estimator = ItemFrequencyEstimator(num_items=2)
    loss_function = LogQCorrectedLoss(estimator)

    loss_function(IDENTITY_2D, IDENTITY_2D, DISTINCT_ITEMS)
    loss_function(IDENTITY_2D, IDENTITY_2D, DISTINCT_ITEMS)

    assert estimator.step == 2


def test_duplicate_items_are_still_masked() -> None:
    estimator = ItemFrequencyEstimator(num_items=8)
    items = torch.tensor([[1.0, 0.0], [1.0, 0.0]])

    loss = LogQCorrectedLoss(estimator, temperature=1.0)(IDENTITY_2D, items, torch.tensor([7, 7]))

    assert float(loss.item()) == pytest.approx(0.0, abs=1e-6)


def test_non_positive_temperature_is_rejected() -> None:
    with pytest.raises(InvalidConfigError, match="temperature must be positive"):
        LogQCorrectedLoss(ItemFrequencyEstimator(num_items=2), temperature=0.0)
