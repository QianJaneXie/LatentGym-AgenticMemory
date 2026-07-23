"""Decision-time presentation of episodic facts (Phase 1 / Stage A0).

Stage A0 presents all prior-episode facts (read-all). Ranking / top-k budgets
are deferred until store size causes measurable interference.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Sequence

from latentgym.memory.episodic_store import EpisodicStore
from latentgym.memory.types import EpisodicFact

_TARGET_REVEALED = re.compile(r"target revealed as (\d+)", re.IGNORECASE)


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


def select_outcome_only_facts(facts: Sequence[EpisodicFact]) -> List[EpisodicFact]:
    """Keep only episode-outcome facts (Pilot 1 outcome-only baseline)."""
    return [
        f
        for f in facts
        if (f.context or {}).get("decision_type") == "episode_outcome"
    ]


def format_skill_from_facts(facts: Sequence[EpisodicFact]) -> str:
    """Hermes-style procedural lesson from agent-visible outcomes (LatentGym adaptation).

    Deterministic template for Pilot 2 — not a full Hermes Agent integration and not
    a cognitive-memory schema with regression status.
    """
    outcomes = select_outcome_only_facts(facts)
    if not outcomes:
        return ""

    revealed: List[int] = []
    for fact in outcomes:
        m = _TARGET_REVEALED.search(fact.outcome or "")
        if m:
            revealed.append(int(m.group(1)))
    if not revealed:
        return (
            "Experience / skill note (fallible procedural advice; current evidence overrides it):\n"
            "- Past episodes did not reveal numeric targets in the visible feedback.\n"
            "- Suggested procedure: use ordinary binary search from the stated range."
        )

    uniq = sorted(set(revealed))
    lo, hi = min(uniq), max(uniq)
    return (
        "Experience / skill note (fallible procedural advice distilled from past episodes; "
        "current evidence overrides it):\n"
        f"- Previously revealed targets: {uniq}\n"
        f"- Observed span so far: [{lo}, {hi}]\n"
        "- Suggested procedure: before restarting a full-range binary search, first try values "
        "near the min and max of previously revealed targets, then search inside that observed "
        "span when feedback remains consistent with it."
    )


def format_oracle_summary_from_facts(facts: Sequence[EpisodicFact]) -> str:
    """Build a concise oracle summary from agent-visible outcome facts only.

    Uses revealed targets / solve status from prior episode outcomes. Never reads
    evaluator-only fields such as latent range_start or set_values.
    """
    outcomes = select_outcome_only_facts(facts)
    if not outcomes:
        return ""

    revealed: List[int] = []
    lines = [
        "Oracle factual summary (compact restatement of agent-visible episode outcomes only; "
        "current evidence overrides it):",
    ]
    for fact in outcomes:
        lines.append(f"- episode={fact.episode_idx}; {fact.outcome}")
        m = _TARGET_REVEALED.search(fact.outcome or "")
        if m:
            revealed.append(int(m.group(1)))

    if revealed:
        lo, hi = min(revealed), max(revealed)
        uniq = sorted(set(revealed))
        lines.append(
            f"- compact restatement: observed revealed targets so far = {uniq}; "
            f"observed min={lo}, max={hi}."
        )
    return "\n".join(lines)
