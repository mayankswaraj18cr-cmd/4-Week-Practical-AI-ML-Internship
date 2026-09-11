"""Experiment tracking primitives independent of a tracking service."""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List
import json


@dataclass(frozen=True)
class ExperimentResult:
    optimizer: str
    learning_rate: float
    epochs: int
    final_loss: float


class ExperimentLogger:
    """Collect and persist comparable optimizer experiments."""

    def __init__(self) -> None:
        self._results: List[ExperimentResult] = []

    def log(self, result: ExperimentResult) -> None:
        if result.optimizer not in {"Adam", "AdamW", "SGD"} or result.learning_rate <= 0 or result.epochs < 1:
            raise ValueError("invalid experiment configuration")
        self._results.append(result)

    def results(self) -> List[ExperimentResult]:
        return list(self._results)

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps([asdict(item) for item in self._results], indent=2), encoding="utf-8")
