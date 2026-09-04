"""GPU-independent planning for reusable PaddleOCR-VL prepared runs."""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any


_PLAN_FIELDS = frozenset(
    {
        "task",
        "tasks",
        "prompt",
        "prompts",
        "model",
        "prepared_from",
        "sources",
        "source_runs",
        "train_samples",
        "validation_samples",
        "train_probabilities",
        "validation_probabilities",
        "prepared_from_runs",
        "prepared_weights",
        "prepared_weight_policy",
        "rejected",
    }
)


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return deepcopy(value)


@dataclass(frozen=True)
class PreparedRunPlan:
    """Validated, immutable inputs needed to construct a training config."""

    model: str | None
    prepared_from: str | None
    extras: Mapping[str, Any]
    tasks: tuple[str, ...]
    prompts: tuple[str, ...]
    sources: tuple[Mapping[str, Any], ...]
    source_runs: tuple[str, ...]
    train_samples: int
    validation_samples: int
    train_probabilities: tuple[float, ...]
    validation_probabilities: tuple[float, ...]
    prepared_from_runs: tuple[str, ...]
    prepared_weights: tuple[float, ...]
    prepared_weight_policy: str
    rejected: tuple[tuple[str, int], ...]

    @classmethod
    def from_summary(cls, summary: Mapping[str, Any]) -> "PreparedRunPlan":
        sources = tuple(
            _freeze(dict(source)) for source in summary.get("sources", ())
        )
        rejected = tuple(
            sorted(
                (str(reason), count)
                for reason, count in summary.get("rejected", {}).items()
                if isinstance(count, int)
            )
        )
        return cls(
            model=summary.get("model"),
            prepared_from=summary.get("prepared_from"),
            extras=_freeze(
                {
                    key: value
                    for key, value in summary.items()
                    if key not in _PLAN_FIELDS
                }
            ),
            tasks=tuple(summary.get("tasks", ())),
            prompts=tuple(summary.get("prompts", ())),
            sources=sources,
            source_runs=tuple(summary.get("source_runs", ())),
            train_samples=summary["train_samples"],
            validation_samples=summary["validation_samples"],
            train_probabilities=tuple(summary["train_probabilities"]),
            validation_probabilities=tuple(summary["validation_probabilities"]),
            prepared_from_runs=tuple(summary.get("prepared_from_runs", ())),
            prepared_weights=tuple(summary.get("prepared_weights", ())),
            prepared_weight_policy=summary.get(
                "prepared_weight_policy", "relative_normalized"
            ),
            rejected=rejected,
        )

    def to_summary(self) -> dict[str, Any]:
        summary: dict[str, Any] = _thaw(self.extras)
        summary.update(
            {
                "task": self.tasks[0] if len(self.tasks) == 1 else "mixed",
                "tasks": list(self.tasks),
                "prompts": list(self.prompts),
                "sources": [_thaw(source) for source in self.sources],
                "source_runs": list(self.source_runs),
                "train_samples": self.train_samples,
                "validation_samples": self.validation_samples,
                "train_probabilities": list(self.train_probabilities),
                "validation_probabilities": list(self.validation_probabilities),
                "prepared_from_runs": list(self.prepared_from_runs),
                "prepared_weights": list(self.prepared_weights),
                "prepared_weight_policy": self.prepared_weight_policy,
                "rejected": dict(self.rejected),
            }
        )
        if self.model is not None:
            summary["model"] = self.model
        if self.prepared_from is not None:
            summary["prepared_from"] = self.prepared_from
        if len(self.tasks) == 1:
            summary["prompt"] = self.prompts[0]
        return summary

    def write_summary(self, work_dir: Path) -> dict[str, Any]:
        work_dir.mkdir(parents=True, exist_ok=True)
        summary = self.to_summary()
        (work_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return summary


class PreparedRunPlanner:
    """Compose validated prepared-run summaries without touching training runtime."""

    def __init__(
        self,
        read_run: Callable[[Path], dict[str, Any]],
        normalize_weights: Callable[
            [Sequence[Path], Sequence[float] | None], list[float]
        ],
        prompt_for_task: Callable[[str], str],
    ) -> None:
        self._read_run = read_run
        self._normalize_weights = normalize_weights
        self._prompt_for_task = prompt_for_task

    def plan(
        self,
        prepared_from: Sequence[Path],
        prepared_weights: Sequence[float] | None,
    ) -> PreparedRunPlan:
        prepared_runs = list(prepared_from)
        normalized_weights = self._normalize_weights(
            prepared_runs, prepared_weights
        )
        run_summaries = [self._read_run(path) for path in prepared_runs]
        if len(run_summaries) == 1:
            summary = dict(run_summaries[0])
            summary["prepared_weights"] = normalized_weights
            summary["prepared_weight_policy"] = "relative_normalized"
            summary["source_runs"] = [
                summary["prepared_from"] for _ in summary["sources"]
            ]
            summary["prepared_from_runs"] = [summary["prepared_from"]]
            return PreparedRunPlan.from_summary(summary)

        models = [summary.get("model") for summary in run_summaries]
        if any(not isinstance(model, str) or not model for model in models):
            raise ValueError("Every prepared summary must record a non-empty model")
        if len(set(models)) != 1:
            raise ValueError(f"Prepared runs use different base models: {models}")

        sources: list[dict[str, Any]] = []
        source_runs: list[str] = []
        train_probabilities: list[float] = []
        validation_probabilities: list[float] = []
        tasks: set[str] = set()
        rejected: Counter[str] = Counter()
        for run_summary, run_weight in zip(
            run_summaries, normalized_weights, strict=True
        ):
            run_path = run_summary["prepared_from"]
            run_sources = run_summary["sources"]
            sources.extend(run_sources)
            source_runs.extend(run_path for _ in run_sources)
            train_probabilities.extend(
                run_weight * value
                for value in run_summary["train_probabilities"]
            )
            validation_probabilities.extend(
                run_weight * value
                for value in run_summary["validation_probabilities"]
            )
            tasks.update(run_summary["tasks"])
            run_rejected = run_summary.get("rejected", {})
            if isinstance(run_rejected, Mapping):
                rejected.update(
                    {
                        str(reason): count
                        for reason, count in run_rejected.items()
                        if isinstance(count, int)
                    }
                )

        for field, probabilities in (
            ("train_probabilities", train_probabilities),
            ("validation_probabilities", validation_probabilities),
        ):
            if not math.isclose(
                sum(probabilities), 1.0, rel_tol=1e-6, abs_tol=1e-6
            ):
                raise ValueError(f"Aggregated {field} does not sum to 1.0")

        ordered_tasks = sorted(tasks)
        summary = {
            "task": ordered_tasks[0] if len(ordered_tasks) == 1 else "mixed",
            "tasks": ordered_tasks,
            "prompts": [self._prompt_for_task(task) for task in ordered_tasks],
            "model": models[0],
            "sources": sources,
            "source_runs": source_runs,
            "train_samples": sum(
                summary["train_samples"] for summary in run_summaries
            ),
            "validation_samples": sum(
                summary["validation_samples"] for summary in run_summaries
            ),
            "train_probabilities": train_probabilities,
            "validation_probabilities": validation_probabilities,
            "prepared_from_runs": [
                summary["prepared_from"] for summary in run_summaries
            ],
            "prepared_weights": normalized_weights,
            "prepared_weight_policy": "relative_normalized",
            "rejected": dict(sorted(rejected.items())),
        }
        return PreparedRunPlan.from_summary(summary)
