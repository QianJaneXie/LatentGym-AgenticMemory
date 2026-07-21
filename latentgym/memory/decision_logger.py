"""Decision-trace logger with provenance validation."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Set

from latentgym.memory.types import DecisionTrace, validate_decision_provenance


class DecisionLogger:
    def __init__(
        self,
        *,
        known_fact_ids: Optional[Set[str]] = None,
        known_cognition_ids: Optional[Set[str]] = None,
    ) -> None:
        self._traces: List[DecisionTrace] = []
        self._known_fact_ids: Set[str] = set(known_fact_ids or [])
        self._known_cognition_ids: Set[str] = set(known_cognition_ids or [])

    def update_known_facts(self, fact_ids: Set[str]) -> None:
        self._known_fact_ids |= set(fact_ids)

    def update_known_cognitions(self, cognition_ids: Set[str]) -> None:
        self._known_cognition_ids |= set(cognition_ids)

    def log(self, trace: DecisionTrace, *, validate: bool = True) -> None:
        if validate:
            validate_decision_provenance(
                trace,
                fact_ids=self._known_fact_ids,
                cognition_ids=self._known_cognition_ids,
            )
        self._traces.append(trace)

    def all_traces(self) -> List[DecisionTrace]:
        return list(self._traces)

    def to_list(self) -> List[dict]:
        return [t.to_dict() for t in self._traces]

    def save_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_list(), indent=2, ensure_ascii=True) + "\n")
