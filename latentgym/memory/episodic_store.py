"""Append-only episodic fact store with JSON serialization."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Optional

from latentgym.memory.types import EpisodicFact, validate_fact_constraints


class EpisodicStore:
    """In-memory append-only log of verified (or tagged) episodic facts."""

    def __init__(self) -> None:
        self._facts: List[EpisodicFact] = []
        self._ids: set[str] = set()

    def __len__(self) -> int:
        return len(self._facts)

    def fact_ids(self) -> set[str]:
        return set(self._ids)

    def all_facts(self) -> List[EpisodicFact]:
        return list(self._facts)

    def get(self, fact_id: str) -> Optional[EpisodicFact]:
        for fact in self._facts:
            if fact.fact_id == fact_id:
                return fact
        return None

    def append(self, fact: EpisodicFact, *, validate: bool = True) -> None:
        if fact.fact_id in self._ids:
            raise ValueError(f"Duplicate fact_id {fact.fact_id!r}; store is append-only.")
        if validate:
            validate_fact_constraints(fact)
        self._facts.append(fact)
        self._ids.add(fact.fact_id)

    def extend(self, facts: Iterable[EpisodicFact], *, validate: bool = True) -> None:
        for fact in facts:
            self.append(fact, validate=validate)

    def by_trajectory(self, trajectory_id: str) -> List[EpisodicFact]:
        return [f for f in self._facts if f.trajectory_id == trajectory_id]

    def by_episode(self, trajectory_id: str, episode_idx: int) -> List[EpisodicFact]:
        return [
            f
            for f in self._facts
            if f.trajectory_id == trajectory_id and f.episode_idx == episode_idx
        ]

    def to_list(self) -> List[dict]:
        return [f.to_dict() for f in self._facts]

    @classmethod
    def from_list(cls, rows: List[dict]) -> EpisodicStore:
        store = cls()
        for row in rows:
            store.append(EpisodicFact.from_dict(row), validate=True)
        return store

    def save_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_list(), indent=2, ensure_ascii=True) + "\n")

    @classmethod
    def load_json(cls, path: str | Path) -> EpisodicStore:
        rows = json.loads(Path(path).read_text())
        if not isinstance(rows, list):
            raise ValueError(f"Expected JSON list of facts in {path}")
        return cls.from_list(rows)
