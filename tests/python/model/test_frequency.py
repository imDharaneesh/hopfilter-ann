import math

import pytest
import torch

from hopfilter.errors import InvalidConfigError
from hopfilter.model.frequency import ItemFrequencyEstimator


def test_first_observation_matches_hand_computed_update() -> None:
    estimator = ItemFrequencyEstimator(num_items=3, learning_rate=0.1, initial_gap=3.0)

    log_q = estimator.observe(torch.tensor([0]))

    expected_gap = 0.9 * 3.0 + 0.1 * 1
    assert float(log_q.item()) == pytest.approx(-math.log(expected_gap))
    assert estimator.step == 1


def test_second_observation_uses_gap_since_last_seen() -> None:
    estimator = ItemFrequencyEstimator(num_items=2, learning_rate=0.5, initial_gap=2.0)
    estimator.observe(torch.tensor([0]))
    estimator.observe(torch.tensor([1]))

    log_q = estimator.observe(torch.tensor([0]))

    first_gap = 0.5 * 2.0 + 0.5 * 1
    second_gap = 0.5 * first_gap + 0.5 * 2
    assert float(log_q.item()) == pytest.approx(-math.log(second_gap))


@pytest.mark.parametrize("period", [1, 2, 4, 10])
def test_converges_to_inverse_of_sighting_period(period: int) -> None:
    estimator = ItemFrequencyEstimator(num_items=2, learning_rate=0.05)
    for step in range(1, 2000):
        estimator.observe(torch.tensor([0]) if step % period == 0 else torch.tensor([1]))

    assert float(estimator.probabilities()[0]) == pytest.approx(1.0 / period, rel=0.02)


def test_unseen_items_keep_initial_probability() -> None:
    estimator = ItemFrequencyEstimator(num_items=4)
    estimator.observe(torch.tensor([0, 1]))

    torch.testing.assert_close(estimator.probabilities()[2:], torch.full((2,), 0.25))


def test_duplicates_within_a_batch_count_once() -> None:
    once = ItemFrequencyEstimator(num_items=3, learning_rate=0.2)
    repeated = ItemFrequencyEstimator(num_items=3, learning_rate=0.2)

    once.observe(torch.tensor([1]))
    repeated.observe(torch.tensor([1, 1, 1]))

    torch.testing.assert_close(once.probabilities(), repeated.probabilities())


def test_observe_returns_log_q_per_row_in_float32() -> None:
    estimator = ItemFrequencyEstimator(num_items=3)

    log_q = estimator.observe(torch.tensor([2, 0, 2]))

    assert log_q.dtype == torch.float32
    assert log_q.shape == (3,)
    assert log_q[0] == log_q[2]


def test_log_probabilities_do_not_change_state() -> None:
    estimator = ItemFrequencyEstimator(num_items=3)
    before = estimator.probabilities().clone()

    estimator.log_probabilities(torch.tensor([0, 1]))

    torch.testing.assert_close(estimator.probabilities(), before)
    assert estimator.step == 0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"num_items": 0}, "num_items must be positive"),
        ({"num_items": 3, "learning_rate": 0.0}, "learning_rate must be in"),
        ({"num_items": 3, "learning_rate": 1.5}, "learning_rate must be in"),
        ({"num_items": 3, "initial_gap": 0.5}, "initial_gap must be at least 1"),
    ],
)
def test_invalid_arguments_are_rejected(kwargs: dict[str, float], message: str) -> None:
    with pytest.raises(InvalidConfigError, match=message):
        ItemFrequencyEstimator(**kwargs)
