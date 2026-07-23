"""Decision-time presentation of episodic facts (Phase 1 / Stage A0).

Stage A0 presents all prior-episode facts (read-all). Ranking / top-k budgets
are deferred until store size causes measurable interference.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

from latentgym.memory.episodic_store import EpisodicStore
from latentgym.memory.types import EpisodicFact


@dataclass
class RetrievalResult:
    facts: List[EpisodicFact]
    fact_ids: List[str]
    query: str


def _candidate_facts(
    store: EpisodicStore,
    *,
    trajectory_id: str,
    episode_idx: int,
    environment: str,
) -> List[EpisodicFact]:
    candidates: List[EpisodicFact] = []
    for fact in store.all_facts():
        if fact.trajectory_id != trajectory_id:
            continue
        if fact.episode_idx >= episode_idx:
            continue
        if (fact.context or {}).get("environment", environment) != environment:
            continue
        candidates.append(fact)
    # Stable chronological order — no relevance ranking in Stage A0.
    candidates.sort(
        key=lambda f: (
            f.episode_idx,
            f.decision_idx if f.decision_idx is not None else 10**9,
            f.fact_id,
        )
    )
    return candidates


def retrieve_episodic_facts(
    store: EpisodicStore,
    *,
    trajectory_id: str,
    episode_idx: int,
    decision_type: str = "first_guess",
    environment: str = "number_guessing",
    budget: Optional[int] = None,
) -> RetrievalResult:
    """Return prior-episode facts for the upcoming decision.

    Default (`budget=None`) is read-all: every eligible fact, chronological.
    A positive `budget` truncates after chronological order (legacy / later
    budget sweeps only). Cognition layer is omitted in Phase 1.
    """
    query = (
        f"environment={environment}; trajectory={trajectory_id}; "
        f"episode={episode_idx}; decision_type={decision_type}; "
        f"mode={'read_all' if budget is None else f'budget_{budget}'}"
    )
    candidates = _candidate_facts(
        store,
        trajectory_id=trajectory_id,
        episode_idx=episode_idx,
        environment=environment,
    )
    selected = candidates if budget is None else candidates[:budget]
    return RetrievalResult(
        facts=selected,
        fact_ids=[f.fact_id for f in selected],
        query=query,
    )


def format_facts_for_prompt(facts: Sequence[EpisodicFact]) -> str:
    """Render facts as fallible historical records for the task agent."""
    if not facts:
        return ""
    lines = [
        "Verified past records (fallible historical notes; current evidence overrides them):",
    ]
    for fact in facts:
        action_part = f"action={fact.action}; " if fact.action is not None else ""
        lines.append(
            f"- [{fact.fact_id}] episode={fact.episode_idx}; "
            f"{action_part}outcome={fact.outcome}"
        )
    return "\n".join(lines)
