"""Decision-time retrieval over episodic facts (Phase 1: no cognition layer)."""
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


def _rank_key(fact: EpisodicFact, *, episode_idx: int, decision_type: str) -> tuple:
    """Deterministic ranking features (higher is better → negate where needed)."""
    ctx = fact.context or {}
    same_decision = 1 if ctx.get("decision_type") == decision_type else 0
    is_outcome = 1 if ctx.get("decision_type") == "episode_outcome" else 0
    surprising = 0
    outcome = fact.outcome.lower()
    if "correct" in outcome or "solved=true" in outcome or "out of guesses" in outcome:
        surprising = 1
    if "invalid" in outcome or "rejected" in outcome:
        surprising = 1
    # Prefer same latent session already guaranteed by store filter; use recency.
    return (
        is_outcome,  # episode outcomes are high-information
        same_decision,
        surprising,
        fact.episode_idx,  # recency within prefix
        fact.decision_idx if fact.decision_idx is not None else -1,
    )


def retrieve_episodic_facts(
    store: EpisodicStore,
    *,
    trajectory_id: str,
    episode_idx: int,
    decision_type: str = "first_guess",
    environment: str = "number_guessing",
    budget: int = 10,
) -> RetrievalResult:
    """Retrieve up to `budget` high-information facts for the upcoming decision.

    Only returns facts from the same trajectory with episode_idx < current
    (no future leakage). Cognition layer is omitted in Phase 1.
    """
    query = (
        f"environment={environment}; trajectory={trajectory_id}; "
        f"episode={episode_idx}; decision_type={decision_type}"
    )
    candidates: List[EpisodicFact] = []
    for fact in store.all_facts():
        if fact.trajectory_id != trajectory_id:
            continue
        if fact.episode_idx >= episode_idx:
            continue
        if (fact.context or {}).get("environment", environment) != environment:
            continue
        candidates.append(fact)

    candidates.sort(
        key=lambda f: _rank_key(f, episode_idx=episode_idx, decision_type=decision_type),
        reverse=True,
    )
    selected = candidates[:budget]
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
