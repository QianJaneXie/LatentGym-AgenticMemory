"""Deterministic fact reconciliation helpers (Pilot 3 / eng. Phase 2 MVP).

Starts with Bandits: keep append-only episode events, maintain a rebuildable
current view with supersession of the latest revealed-best-arm claim.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from latentgym.memory.types import EpisodicFact

_BEST_REVEALED = re.compile(r"best revealed as (\w+)", re.IGNORECASE)
_EXPLORE_REWARD = re.compile(r"explore (\w+); reward=([01])", re.IGNORECASE)


@dataclass
class FactClaim:
    claim_id: str
    subject_key: str
    predicate: str
    object_value: Any
    episode_idx: Optional[int]
    source_fact_ids: List[str]
    derivation_type: str  # observed, deterministic_aggregate
    verification_status: str  # verified, disputed, unresolved
    current_status: str  # active, superseded, historical
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FactRelation:
    relation_id: str
    source_claim_id: str
    target_claim_id: str
    relation_type: str  # supersedes, same_value_new_event, supports
    rationale: str
    evidence_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CurrentFactView:
    view_id: str
    active_claim_ids: List[str]
    historical_claim_ids: List[str]
    disputed_claim_ids: List[str]
    unresolved_relation_ids: List[str]
    claims: List[FactClaim]
    relations: List[FactRelation]
    render_text: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "view_id": self.view_id,
            "active_claim_ids": list(self.active_claim_ids),
            "historical_claim_ids": list(self.historical_claim_ids),
            "disputed_claim_ids": list(self.disputed_claim_ids),
            "unresolved_relation_ids": list(self.unresolved_relation_ids),
            "claims": [c.to_dict() for c in self.claims],
            "relations": [r.to_dict() for r in self.relations],
            "render_text": self.render_text,
        }


def _outcome_facts(facts: Sequence[EpisodicFact]) -> List[EpisodicFact]:
    return [
        f
        for f in facts
        if (f.context or {}).get("decision_type") == "episode_outcome"
    ]


def build_bandits_current_view(
    facts: Sequence[EpisodicFact],
    *,
    trajectory_id: str,
    as_of_episode_idx: int,
) -> CurrentFactView:
    """Build a current factual view from prior-episode bandit facts.

    Rules:
    - Episode outcome events are historical event claims (never merged across episodes).
    - Latest revealed best button is the sole active session-state claim; prior bests
      are superseded (not deleted).
    - Explore reward tallies are deterministic aggregates.
    """
    prior = [
        f
        for f in facts
        if f.trajectory_id == trajectory_id and f.episode_idx < as_of_episode_idx
    ]
    claims: List[FactClaim] = []
    relations: List[FactRelation] = []
    active: List[str] = []
    historical: List[str] = []
    disputed: List[str] = []

    # Per-episode outcome events
    last_best_claim_id: Optional[str] = None
    last_best_value: Optional[str] = None
    for fact in _outcome_facts(prior):
        cid = f"claim_{fact.fact_id}"
        best_m = _BEST_REVEALED.search(fact.outcome or "")
        best = best_m.group(1).lower() if best_m else (fact.context or {}).get("best")
        claims.append(
            FactClaim(
                claim_id=cid,
                subject_key=f"episode:{fact.episode_idx}",
                predicate="episode_outcome",
                object_value=fact.outcome,
                episode_idx=fact.episode_idx,
                source_fact_ids=[fact.fact_id],
                derivation_type="observed",
                verification_status="verified",
                current_status="historical",
                note="Immutable episode event; same best value in another episode is a new event.",
            )
        )
        historical.append(cid)

        if best:
            best_cid = f"claim_{fact.fact_id}_best"
            status = "active"
            claims.append(
                FactClaim(
                    claim_id=best_cid,
                    subject_key="session:revealed_best_button",
                    predicate="revealed_best_button",
                    object_value=best,
                    episode_idx=fact.episode_idx,
                    source_fact_ids=[fact.fact_id],
                    derivation_type="observed",
                    verification_status="verified",
                    current_status=status,
                )
            )
            if last_best_claim_id is not None:
                # Supersede previous active best claim
                for c in claims:
                    if c.claim_id == last_best_claim_id:
                        c.current_status = "superseded"
                        if c.claim_id in active:
                            active.remove(c.claim_id)
                        if c.claim_id not in historical:
                            historical.append(c.claim_id)
                rel_id = f"rel_{last_best_claim_id}_to_{best_cid}"
                relations.append(
                    FactRelation(
                        relation_id=rel_id,
                        source_claim_id=best_cid,
                        target_claim_id=last_best_claim_id,
                        relation_type="supersedes",
                        rationale=(
                            f"New revealed best '{best}' at episode {fact.episode_idx} "
                            f"replaces prior revealed best '{last_best_value}' as active "
                            "session state; prior claim kept as history."
                        ),
                        evidence_ids=[fact.fact_id],
                    )
                )
                if last_best_value == best:
                    relations.append(
                        FactRelation(
                            relation_id=f"rel_same_{best_cid}",
                            source_claim_id=best_cid,
                            target_claim_id=last_best_claim_id,
                            relation_type="same_value_new_event",
                            rationale=(
                                "Same button value revealed in a later episode is a new "
                                "event claim, not a duplicate record."
                            ),
                            evidence_ids=[fact.fact_id],
                        )
                    )
            active.append(best_cid)
            last_best_claim_id = best_cid
            last_best_value = best

    # Explore tallies
    tallies: Dict[str, Dict[str, int]] = {}
    explore_sources: Dict[str, List[str]] = {}
    for fact in prior:
        m = _EXPLORE_REWARD.search(fact.outcome or "")
        if not m:
            continue
        btn, rew = m.group(1).lower(), m.group(2)
        tallies.setdefault(btn, {"n": 0, "ones": 0})
        tallies[btn]["n"] += 1
        if rew == "1":
            tallies[btn]["ones"] += 1
        explore_sources.setdefault(btn, []).append(fact.fact_id)

    for btn, stats in sorted(tallies.items()):
        cid = f"claim_tally_{btn}_before_e{as_of_episode_idx}"
        claims.append(
            FactClaim(
                claim_id=cid,
                subject_key=f"button:{btn}",
                predicate="explore_reward_tally",
                object_value={"n": stats["n"], "reward_1_count": stats["ones"]},
                episode_idx=None,
                source_fact_ids=explore_sources.get(btn, []),
                derivation_type="deterministic_aggregate",
                verification_status="verified",
                current_status="active",
                note="Aggregate of prior explore observations only.",
            )
        )
        active.append(cid)

    render = format_current_fact_view_for_prompt(
        claims=claims,
        relations=relations,
        as_of_episode_idx=as_of_episode_idx,
    )
    return CurrentFactView(
        view_id=f"{trajectory_id}_view_before_e{as_of_episode_idx}",
        active_claim_ids=active,
        historical_claim_ids=historical,
        disputed_claim_ids=disputed,
        unresolved_relation_ids=[],
        claims=claims,
        relations=relations,
        render_text=render,
    )


def format_current_fact_view_for_prompt(
    *,
    claims: Sequence[FactClaim],
    relations: Sequence[FactRelation],
    as_of_episode_idx: int,
) -> str:
    if not claims:
        return ""
    lines = [
        "Reconciled current factual view (deterministic; rebuildable from append-only "
        "evidence; current evidence overrides it):",
        f"- view built for decisions in episode={as_of_episode_idx}",
    ]
    active_best = [
        c
        for c in claims
        if c.predicate == "revealed_best_button" and c.current_status == "active"
    ]
    if active_best:
        c = active_best[-1]
        lines.append(
            f"- ACTIVE session state: latest revealed best button = {c.object_value} "
            f"(from episode {c.episode_idx}; claim {c.claim_id})"
        )
    superseded = [
        c
        for c in claims
        if c.predicate == "revealed_best_button" and c.current_status == "superseded"
    ]
    for c in superseded:
        lines.append(
            f"- SUPERSEDED (kept for audit): revealed best button = {c.object_value} "
            f"at episode {c.episode_idx}"
        )
    for c in claims:
        if c.predicate == "episode_outcome":
            lines.append(
                f"- HISTORICAL event episode={c.episode_idx}: {c.object_value}"
            )
    for c in claims:
        if c.predicate == "explore_reward_tally":
            val = c.object_value or {}
            lines.append(
                f"- ACTIVE aggregate {c.subject_key}: explores={val.get('n')}, "
                f"reward_1_count={val.get('reward_1_count')}"
            )
    if relations:
        lines.append("- Relations:")
        for r in relations:
            lines.append(
                f"  - [{r.relation_type}] {r.source_claim_id} -> {r.target_claim_id}: "
                f"{r.rationale}"
            )
    return "\n".join(lines)
