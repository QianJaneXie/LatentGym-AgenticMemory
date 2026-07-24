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
_BEST_REVEALED = re.compile(r"best revealed as (\w+)", re.IGNORECASE)


def _env_from_facts(facts: Sequence[EpisodicFact]) -> str:
    for f in facts:
        env = (f.context or {}).get("environment")
        if env:
            return str(env)
    return "number_guessing"


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


def build_atomic_flat_extraction_prompt(
    *,
    episode_idx: int,
    turn_lines: Sequence[str],
    end_feedback: str,
    environment: str = "number_guessing",
) -> str:
    """Prompt for Mem0-style flat memory extraction from one visible episode."""
    lines = [
        f"Extract short, flat, reusable memories from this {environment} episode.",
        "Use ONLY the agent-visible transcript below.",
        "Do not invent hidden latents or evaluator-only information.",
        "Prefer objective observations over advice or strategies.",
        "Return 1-6 bullet lines. Each bullet should be one short standalone fact.",
        "Plain text only. No JSON.",
        "",
        f"Episode index: {episode_idx}",
        "Visible transcript:",
    ]
    for line in turn_lines:
        lines.append(f"- {line}")
    if end_feedback.strip():
        lines.append(f"- episode_end: {end_feedback.strip()}")
    return "\n".join(lines)


def parse_atomic_flat_bullets(text: str) -> List[str]:
    """Parse LLM bullet lines into cleaned flat memory strings."""
    out: List[str] = []
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line[0] in "-*•":
            line = line[1:].strip()
        elif len(line) > 2 and line[0].isdigit() and line[1] in ".)":
            line = line[2:].strip()
        if line:
            out.append(line)
    # Light dedupe while preserving order
    seen = set()
    unique: List[str] = []
    for item in out:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def format_atomic_flat_memories(memories: Sequence[str]) -> str:
    """Render accumulated flat memories for the task agent (read-all)."""
    if not memories:
        return ""
    lines = [
        "Atomic flat memories (LLM-extracted from past visible episodes; fallible; "
        "current evidence overrides them):",
    ]
    for i, mem in enumerate(memories):
        lines.append(f"- [m{i}] {mem}")
    return "\n".join(lines)


def build_inline_skill_distillation_prompt() -> str:
    """User prompt appended to the live task conversation after an episode ends.

    Hermes-pattern: the task agent writes the skill in the same conversation that
    contains the rules and the play just completed (not a blind outcomes-only call).
    """
    return (
        "Before the next game, write a short reusable skill/lesson based on this "
        "conversation so far (the rules already above, plus the play you just completed).\n"
        "If an experience/skill note was already injected earlier in this conversation, "
        "revise it using what you just learned.\n"
        "Use only information visible in this conversation. Do not invent hidden latents "
        "or evaluator-only information that was never stated.\n"
        "Return 3-6 bullet lines of procedural advice for the next game.\n"
        "Plain text only. No JSON."
    )


def wrap_distilled_skill(skill_body: str) -> str:
    """Prefix LLM-distilled skill text for injection into the task agent."""
    body = (skill_body or "").strip()
    if not body:
        return ""
    return (
        "Experience / skill note (LLM-distilled procedural advice from past visible "
        "play; fallible; current evidence overrides it):\n"
        + body
    )


def format_skill_from_facts(facts: Sequence[EpisodicFact]) -> str:
    """Hermes-pattern proxy: deterministic procedural lesson from visible outcomes.

    Experimenter template for Pilot 2 plumbing — not agent distillation and not
    a cognitive-memory schema with regression status.
    """
    outcomes = select_outcome_only_facts(facts)
    if not outcomes:
        return ""

    env = _env_from_facts(facts)
    if env == "bandits":
        bests: List[str] = []
        for fact in outcomes:
            m = _BEST_REVEALED.search(fact.outcome or "")
            if m:
                bests.append(m.group(1).lower())
        if not bests:
            return (
                "Experience / skill note (fallible procedural advice; current evidence overrides it):\n"
                "- Past episodes did not reveal best-button labels in the visible feedback.\n"
                "- Suggested procedure: explore each button a few times, then select the "
                "empirical favorite."
            )
        latest = bests[-1]
        uniq = list(dict.fromkeys(bests))
        return (
            "Experience / skill note (fallible procedural advice distilled from past episodes; "
            "current evidence overrides it):\n"
            f"- Previously revealed best buttons (in order): {uniq}\n"
            f"- Latest revealed best: {latest}\n"
            "- Suggested procedure: begin by testing the latest revealed best, but treat older "
            "bests as historical events that may no longer be active if the session pattern drifts."
        )

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

    Uses revealed targets / best buttons from prior episode outcomes. Never reads
    evaluator-only latent fields beyond what the feedback already revealed.
    """
    outcomes = select_outcome_only_facts(facts)
    if not outcomes:
        return ""

    env = _env_from_facts(facts)
    lines = [
        "Oracle factual summary (compact restatement of agent-visible episode outcomes only; "
        "current evidence overrides it):",
    ]
    if env == "bandits":
        bests: List[str] = []
        for fact in outcomes:
            lines.append(f"- episode={fact.episode_idx}; {fact.outcome}")
            m = _BEST_REVEALED.search(fact.outcome or "")
            if m:
                bests.append(m.group(1).lower())
        if bests:
            lines.append(
                f"- compact restatement: revealed best buttons in order = {bests}; "
                f"latest={bests[-1]}."
            )
        return "\n".join(lines)

    revealed: List[int] = []
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
