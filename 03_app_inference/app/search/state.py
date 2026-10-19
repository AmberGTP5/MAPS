from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, Iterable, Optional, Tuple


@dataclass(frozen=True)
class SearchAction:
    robot: str
    action: str
    target: str
    details: str = ""

    def to_plan_step(self) -> Dict[str, str]:
        return {
            "robot": self.robot,
            "action": self.action,
            "target": self.target,
            "details": self.details,
        }

    @property
    def key(self) -> Tuple[str, str, str]:
        return (self.robot, self.action, self.target)


@dataclass(frozen=True)
class SearchState:
    chassis_location: str = "Robot_Station_5F"
    dog_location: str = "Dog_Station_4F"
    coffee_source: Optional[str] = None
    item_holder: str = "unpicked"
    delivered: bool = False
    chassis_returned: bool = False
    dog_returned: bool = False
    path_cost: float = 0.0
    history: Tuple[SearchAction, ...] = field(default_factory=tuple)
    notes: Tuple[str, ...] = field(default_factory=tuple)

    def append(self, action: SearchAction, **updates) -> "SearchState":
        next_history = self.history + (action,)
        next_notes = updates.pop("notes", self.notes)
        return replace(self, history=next_history, notes=next_notes, **updates)


@dataclass(frozen=True)
class SearchPrior:
    sequence: Tuple[Tuple[str, str, str], ...] = field(default_factory=tuple)
    exact_counts: Tuple[Tuple[Tuple[str, str, str], int], ...] = field(default_factory=tuple)
    target_counts: Tuple[Tuple[str, int], ...] = field(default_factory=tuple)

    @classmethod
    def from_plan(cls, plan_steps: Iterable[dict]) -> "SearchPrior":
        sequence = []
        exact: Dict[Tuple[str, str, str], int] = {}
        targets: Dict[str, int] = {}
        for step in plan_steps or []:
            if not isinstance(step, dict):
                continue
            key = (
                str(step.get("robot", "")),
                str(step.get("action", "")),
                str(step.get("target", "")),
            )
            if not all(key):
                continue
            sequence.append(key)
            exact[key] = exact.get(key, 0) + 1
            targets[key[2]] = targets.get(key[2], 0) + 1

        return cls(
            sequence=tuple(sequence),
            exact_counts=tuple(sorted(exact.items())),
            target_counts=tuple(sorted(targets.items())),
        )

    def score(self, action: SearchAction, depth: int) -> float:
        score = 0.0
        key = action.key
        exact_counts = dict(self.exact_counts)
        target_counts = dict(self.target_counts)

        if depth < len(self.sequence) and self.sequence[depth] == key:
            score += 3.0
        elif key in exact_counts:
            score += 1.2

        if action.target in target_counts:
            score += min(0.8, 0.2 * target_counts[action.target])

        return score


def history_to_plan(history: Iterable[SearchAction]) -> list[dict]:
    return [action.to_plan_step() for action in history]
