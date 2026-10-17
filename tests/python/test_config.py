from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import pytest

from hopfilter.config import config_from_mapping, load_config
from hopfilter.errors import InvalidConfigError


@dataclass(frozen=True)
class OptimizerSection:
    learning_rate: float
    betas: tuple[float, ...] = (0.9, 0.999)


@dataclass(frozen=True)
class ExampleConfig:
    name: str
    epochs: int
    optimizer: OptimizerSection
    loss: Literal["softmax", "logq"] = "softmax"
    shuffle: bool = True
    hidden_dims: tuple[int, ...] = field(default=(128,))


VALID_RAW = {"name": "run", "epochs": 3, "optimizer": {"learning_rate": 0.01}}


def test_builds_nested_dataclass_with_defaults() -> None:
    config = config_from_mapping(VALID_RAW, ExampleConfig)

    assert config == ExampleConfig(
        name="run", epochs=3, optimizer=OptimizerSection(learning_rate=0.01)
    )


def test_converts_lists_to_tuples_and_ints_to_floats() -> None:
    raw = {**VALID_RAW, "hidden_dims": [64, 32], "optimizer": {"learning_rate": 1, "betas": [0, 1]}}

    config = config_from_mapping(raw, ExampleConfig)

    assert config.hidden_dims == (64, 32)
    assert config.optimizer == OptimizerSection(learning_rate=1.0, betas=(0.0, 1.0))
    assert isinstance(config.optimizer.learning_rate, float)


def test_accepts_literal_choice() -> None:
    assert config_from_mapping({**VALID_RAW, "loss": "logq"}, ExampleConfig).loss == "logq"


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ({**VALID_RAW, "epoch": 3}, "config has unknown keys: epoch"),
        ({"name": "run", "optimizer": {"learning_rate": 0.1}}, "missing required keys: epochs"),
        ({**VALID_RAW, "epochs": "3"}, r"config\.epochs must be int, got str"),
        ({**VALID_RAW, "epochs": True}, r"config\.epochs must be int, got bool"),
        ({**VALID_RAW, "shuffle": 1}, r"config\.shuffle must be bool, got int"),
        ({**VALID_RAW, "loss": "hinge"}, r"config\.loss must be one of 'softmax', 'logq'"),
        ({**VALID_RAW, "hidden_dims": 64}, r"config\.hidden_dims must be a list"),
        ({**VALID_RAW, "hidden_dims": [64, "x"]}, r"config\.hidden_dims\[1\] must be int"),
        ({**VALID_RAW, "optimizer": {"lr": 0.1}}, r"config\.optimizer has unknown keys: lr"),
        ({**VALID_RAW, "optimizer": 0.1}, r"config\.optimizer must be a mapping"),
    ],
)
def test_rejects_invalid_values_with_location(raw: dict[str, object], message: str) -> None:
    with pytest.raises(InvalidConfigError, match=message):
        config_from_mapping(raw, ExampleConfig)


def test_rejects_non_mapping_document() -> None:
    with pytest.raises(InvalidConfigError, match="config must be a mapping, got list"):
        config_from_mapping([1, 2], ExampleConfig)


def test_loads_yaml_file(tmp_path: Path) -> None:
    path = tmp_path / "run.yaml"
    path.write_text(
        "name: run\nepochs: 2\noptimizer:\n    learning_rate: 0.5\nloss: logq\n", encoding="utf-8"
    )

    config = load_config(path, ExampleConfig)

    assert config.epochs == 2
    assert config.loss == "logq"


def test_empty_yaml_file_reports_missing_keys(tmp_path: Path) -> None:
    path = tmp_path / "empty.yaml"
    path.write_text("", encoding="utf-8")

    with pytest.raises(InvalidConfigError, match="missing required keys: epochs, name, optimizer"):
        load_config(path, ExampleConfig)


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(InvalidConfigError, match="config file not found"):
        load_config(tmp_path / "absent.yaml", ExampleConfig)
