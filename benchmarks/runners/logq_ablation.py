"""Train the two-tower model with and without logQ correction and record recall@k.

Usage:
    uv run python benchmarks/runners/logq_ablation.py \
        --config benchmarks/configs/logq_ablation_ml1m.yaml
"""

import argparse
import json
import platform
import statistics
import subprocess
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt
import polars as pl
import torch

from hopfilter.config import load_config
from hopfilter.data.dataset import ITEM_ID, USER_ID, DatasetLoader, keep_positive
from hopfilter.data.download import DATASETS
from hopfilter.data.encoding import IdEncoder, build_genre_matrix
from hopfilter.data.movielens import MovieLens1MLoader, MovieLens25MLoader
from hopfilter.data.splits import SplitConfig, TemporalSplitter
from hopfilter.eval.retrieval import (
    ExactRetriever,
    PopularitySegmenter,
    RetrievalEvaluator,
    RetrievalReport,
    group_by_query,
)
from hopfilter.model.frequency import ItemFrequencyEstimator
from hopfilter.model.losses import InBatchSoftmaxLoss, LogQCorrectedLoss, LossFunction
from hopfilter.model.towers import ItemTower, TowerConfig, TwoTowerModel, UserTower
from hopfilter.model.training import InteractionBatcher, Trainer, TrainerConfig, make_optimizer

type IndexArray = npt.NDArray[np.int64]
type LossName = Literal["in_batch_softmax", "logq_corrected"]

LOSS_NAMES: tuple[LossName, ...] = ("in_batch_softmax", "logq_corrected")
LOADERS: Mapping[str, Callable[[Path], DatasetLoader]] = {
    "movielens-1m": MovieLens1MLoader,
    "movielens-25m": MovieLens25MLoader,
}


@dataclass(frozen=True)
class DataSection:
    dataset: Literal["movielens-1m", "movielens-25m"]
    data_dir: str = "data"
    min_rating: float = 4.0
    validation_fraction: float = 0.1
    test_fraction: float = 0.1


@dataclass(frozen=True)
class LossSection:
    temperature: float = 0.05
    frequency_learning_rate: float = 0.01


@dataclass(frozen=True)
class EvaluationSection:
    ks: tuple[int, ...] = (10, 50, 100)
    head_fraction: float = 0.2
    validation_k: int = 50


@dataclass(frozen=True)
class AblationConfig:
    data: DataSection
    results_path: str
    tower: TowerConfig = field(default_factory=TowerConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    loss: LossSection = field(default_factory=LossSection)
    evaluation: EvaluationSection = field(default_factory=EvaluationSection)
    seeds: tuple[int, ...] = (0,)


@dataclass(frozen=True)
class EvaluationSet:
    users: IndexArray
    relevant: list[IndexArray]
    excluded: list[IndexArray]


@dataclass(frozen=True)
class PreparedData:
    train_users: IndexArray
    train_items: IndexArray
    num_users: int
    genre_matrix: torch.Tensor
    validation: EvaluationSet
    test: EvaluationSet
    segmenter: PopularitySegmenter
    statistics: dict[str, object]


def prepare_data(config: AblationConfig) -> PreparedData:
    dataset_dir = Path(config.data.data_dir) / DATASETS[config.data.dataset].directory_name
    loader: DatasetLoader = LOADERS[config.data.dataset](dataset_dir)
    dataset = loader.load()
    positives = keep_positive(dataset.interactions, config.data.min_rating)
    split_config = SplitConfig(config.data.validation_fraction, config.data.test_fraction)
    splits = TemporalSplitter(split_config).split(positives)
    users, items = IdEncoder.fit(splits.train[USER_ID]), IdEncoder.fit(splits.train[ITEM_ID])
    train_users, train_items = (
        users.encode(splits.train[USER_ID]),
        items.encode(splits.train[ITEM_ID]),
    )
    segmenter = PopularitySegmenter(train_items, items.size, config.evaluation.head_fraction)
    return PreparedData(
        train_users=train_users,
        train_items=train_items,
        num_users=users.size,
        genre_matrix=torch.from_numpy(build_genre_matrix(dataset.items, items).values),
        validation=evaluation_set(splits.validation, users, items, train_users, train_items),
        test=evaluation_set(splits.test, users, items, train_users, train_items),
        segmenter=segmenter,
        statistics={
            "positives": positives.height,
            "train": splits.train.height,
            "validation": splits.validation.height,
            "test": splits.test.height,
            "dropped_cold_start": splits.dropped_cold_start,
            "users": users.size,
            "items": items.size,
            "head_interaction_share": segmenter.head_interaction_share,
        },
    )


def evaluation_set(
    held_out: pl.DataFrame,
    users: IdEncoder,
    items: IdEncoder,
    train_users: IndexArray,
    train_items: IndexArray,
) -> EvaluationSet:
    held_out_users, held_out_items = (
        users.encode(held_out[USER_ID]),
        items.encode(held_out[ITEM_ID]),
    )
    query_users = np.unique(held_out_users)
    return EvaluationSet(
        users=query_users,
        relevant=group_by_query(held_out_users, held_out_items, query_users),
        excluded=group_by_query(train_users, train_items, query_users),
    )


def build_loss(name: LossName, num_items: int, config: LossSection) -> LossFunction:
    if name == "in_batch_softmax":
        return InBatchSoftmaxLoss(config.temperature)
    estimator = ItemFrequencyEstimator(num_items, config.frequency_learning_rate)
    return LogQCorrectedLoss(estimator, config.temperature)


def evaluate(
    model: TwoTowerModel, data: PreparedData, split: EvaluationSet, ks: tuple[int, ...]
) -> RetrievalReport:
    with torch.no_grad():
        retriever = ExactRetriever(model.encode_all_items())
        query_vectors = model.encode_users(torch.from_numpy(split.users))
    return RetrievalEvaluator(retriever, data.segmenter, ks).evaluate(
        query_vectors, split.relevant, split.excluded
    )


def popularity_baseline(
    data: PreparedData, split: EvaluationSet, ks: tuple[int, ...]
) -> RetrievalReport:
    num_items = int(data.genre_matrix.shape[0])
    train_counts = np.bincount(data.train_items, minlength=num_items).astype(np.float32)
    retriever = ExactRetriever(torch.from_numpy(train_counts).unsqueeze(1))
    same_query_for_everyone = torch.ones(split.users.size, 1)
    return RetrievalEvaluator(retriever, data.segmenter, ks).evaluate(
        same_query_for_everyone, split.relevant, split.excluded
    )


@dataclass(frozen=True)
class RunResult:
    loss: LossName
    seed: int
    train_seconds: float
    epoch_losses: tuple[float, ...]
    validation_recall_curve: tuple[float, ...]
    test: RetrievalReport

    def to_dict(self) -> dict[str, object]:
        return {
            "loss": self.loss,
            "seed": self.seed,
            "train_seconds": self.train_seconds,
            "epoch_losses": list(self.epoch_losses),
            "validation_recall_curve": list(self.validation_recall_curve),
            "test": self.test.to_dict(),
        }


def run_once(name: LossName, seed: int, data: PreparedData, config: AblationConfig) -> RunResult:
    torch.manual_seed(seed)
    trainer_config = replace(config.trainer, seed=seed)
    model = TwoTowerModel(
        UserTower(data.num_users, config.tower), ItemTower(data.genre_matrix, config.tower)
    )
    loss = build_loss(name, int(data.genre_matrix.shape[0]), config.loss)
    trainer = Trainer(model, loss, make_optimizer(model, trainer_config), trainer_config.epochs)
    batcher = InteractionBatcher(
        data.train_users, data.train_items, trainer_config.batch_size, seed
    )
    validation_k = config.evaluation.validation_k
    validation_curve: list[float] = []

    def record_validation(_epoch: int, _loss: float) -> None:
        report = evaluate(model, data, data.validation, (validation_k,))
        validation_curve.append(report.recall[validation_k])

    started = time.perf_counter()
    history = trainer.fit(batcher, on_epoch_end=record_validation)
    seconds = time.perf_counter() - started
    test_report = evaluate(model, data, data.test, config.evaluation.ks)
    print(f"{name:>18} seed={seed} test recall={test_report.recall} ({seconds:.0f}s)")
    return RunResult(
        loss=name,
        seed=seed,
        train_seconds=seconds,
        epoch_losses=history.epoch_losses,
        validation_recall_curve=tuple(validation_curve),
        test=test_report,
    )


def summarize(runs: list[RunResult], ks: tuple[int, ...]) -> dict[str, object]:
    summary: dict[str, object] = {}
    for name in LOSS_NAMES:
        reports = [run.test for run in runs if run.loss == name]
        summary[name] = {
            "recall": _aggregate([report.recall for report in reports], ks),
            "head_recall": _aggregate([report.head_recall for report in reports], ks),
            "tail_recall": _aggregate([report.tail_recall for report in reports], ks),
        }
    return summary


def _aggregate(
    per_seed: list[Mapping[int, float]], ks: tuple[int, ...]
) -> dict[str, dict[str, float]]:
    return {str(k): _mean_and_std([values[k] for values in per_seed]) for k in ks}


def _mean_and_std(values: list[float]) -> dict[str, float]:
    spread = statistics.stdev(values) if len(values) > 1 else 0.0
    return {"mean": statistics.fmean(values), "std": spread}


def environment() -> dict[str, object]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()
    return {
        "commit": commit,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "torch_threads": torch.get_num_threads(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    arguments = parser.parse_args()
    config = load_config(arguments.config, AblationConfig)
    data = prepare_data(config)
    print(f"data: {data.statistics}")
    popularity = popularity_baseline(data, data.test, config.evaluation.ks)
    print(f"{'popularity':>18} test recall={popularity.recall}")
    runs = [run_once(name, seed, data, config) for name in LOSS_NAMES for seed in config.seeds]
    results = {
        "config": asdict(config),
        "environment": environment(),
        "data": data.statistics,
        "baselines": {"popularity": popularity.to_dict()},
        "runs": [run.to_dict() for run in runs],
        "summary": summarize(runs, config.evaluation.ks),
    }
    output = Path(config.results_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=4) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
