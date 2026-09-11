"""Typed Python fundamentals used throughout the internship."""

from dataclasses import dataclass
from typing import Dict, Iterable, List


@dataclass(frozen=True)
class StudentRecord:
    """An immutable, validated learner record."""

    student_id: str
    name: str
    score: float

    def __post_init__(self) -> None:
        if not self.student_id.strip() or not self.name.strip():
            raise ValueError("student_id and name must be non-empty")
        if not 0 <= self.score <= 100:
            raise ValueError("score must be between 0 and 100")


class RecordManager:
    """In-memory record manager with explicit duplicate protection."""

    def __init__(self) -> None:
        self._records: Dict[str, StudentRecord] = {}

    def add(self, record: StudentRecord) -> None:
        if record.student_id in self._records:
            raise KeyError(f"record already exists: {record.student_id}")
        self._records[record.student_id] = record

    def get(self, student_id: str) -> StudentRecord:
        try:
            return self._records[student_id]
        except KeyError as exc:
            raise KeyError(f"record not found: {student_id}") from exc

    def average_score(self) -> float:
        if not self._records:
            raise ValueError("cannot calculate an average for an empty manager")
        return sum(record.score for record in self._records.values()) / len(self._records)

    def all(self) -> List[StudentRecord]:
        return list(self._records.values())


def summarize_scores(records: Iterable[StudentRecord]) -> Dict[str, float]:
    """Return count and mean for any iterable of records."""
    values = [record.score for record in records]
    if not values:
        raise ValueError("at least one record is required")
    return {"count": float(len(values)), "mean": sum(values) / len(values)}
