"""Memory schemas for the agentic hierarchical memory pipeline.

Phase 1 uses EpisodicFact and DecisionTrace. CognitiveMemory / RegressionRun
are defined for provenance checks and later phases but are not promoted yet.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


# Words that suggest inference/advice — banned from verified fact outcome/action text
# unless the fact is explicitly tagged as an unverified agent_statement.
PROHIBITED_INFERENTIAL_TERMS = (
    "because",
    "therefore",
    "should",
    "always",
    "generally",
    "probably",
    "likely",
)


@dataclass
class EpisodicFact:
    fact_id: str
    trajectory_id: str
    episode_idx: int
    decision_idx: Optional[int]
    context: Dict[str, Any]
    action: Optional[str]
    outcome: str
    source_type: str
    source_ref: Dict[str, Any]
    verified: bool
    created_at: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> EpisodicFact:
        return cls(
            fact_id=d["fact_id"],
            trajectory_id=d["trajectory_id"],
            episode_idx=d["episode_idx"],
            decision_idx=d.get("decision_idx"),
            context=d.get("context", {}),
            action=d.get("action"),
            outcome=d["outcome"],
            source_type=d["source_type"],
            source_ref=d.get("source_ref", {}),
            verified=bool(d["verified"]),
            created_at=d["created_at"],
        )


@dataclass
class CognitiveMemory:
    cognition_id: str
    claim: str
    scope: Dict[str, Any]
    action_implication: str
    supporting_fact_ids: List[str]
    counterevidence_fact_ids: List[str] = field(default_factory=list)
    falsification_condition: str = ""
    status: str = "candidate"
    confidence: Optional[float] = None
    validation_run_ids: List[str] = field(default_factory=list)
    revision_parent_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> CognitiveMemory:
        return cls(
            cognition_id=d["cognition_id"],
            claim=d["claim"],
            scope=d.get("scope", {}),
            action_implication=d["action_implication"],
            supporting_fact_ids=list(d.get("supporting_fact_ids", [])),
            counterevidence_fact_ids=list(d.get("counterevidence_fact_ids", [])),
            falsification_condition=d.get("falsification_condition", ""),
            status=d.get("status", "candidate"),
            confidence=d.get("confidence"),
            validation_run_ids=list(d.get("validation_run_ids", [])),
            revision_parent_id=d.get("revision_parent_id"),
        )


@dataclass
class DecisionTrace:
    decision_id: str
    trajectory_id: str
    episode_idx: int
    decision_type: str
    query: str
    loaded_fact_ids: List[str]
    loaded_cognition_ids: List[str]
    cited_memory_ids: List[str]
    action: str
    outcome: str
    reward: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> DecisionTrace:
        return cls(
            decision_id=d["decision_id"],
            trajectory_id=d["trajectory_id"],
            episode_idx=d["episode_idx"],
            decision_type=d["decision_type"],
            query=d.get("query", ""),
            loaded_fact_ids=list(d.get("loaded_fact_ids", [])),
            loaded_cognition_ids=list(d.get("loaded_cognition_ids", [])),
            cited_memory_ids=list(d.get("cited_memory_ids", [])),
            action=d.get("action", ""),
            outcome=d.get("outcome", ""),
            reward=d.get("reward"),
        )


@dataclass
class RegressionRun:
    run_id: str
    cognition_id: str
    source_trajectory_id: str
    fork_episode_idx: int
    suffix_episode_indices: List[int]
    condition: str
    seed: int
    episode_rewards: List[float]
    episode_turns: List[int]
    first_guess_correct: List[bool]
    memory_token_count: int
    failure_tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> RegressionRun:
        return cls(
            run_id=d["run_id"],
            cognition_id=d["cognition_id"],
            source_trajectory_id=d["source_trajectory_id"],
            fork_episode_idx=d["fork_episode_idx"],
            suffix_episode_indices=list(d.get("suffix_episode_indices", [])),
            condition=d["condition"],
            seed=d["seed"],
            episode_rewards=list(d.get("episode_rewards", [])),
            episode_turns=list(d.get("episode_turns", [])),
            first_guess_correct=list(d.get("first_guess_correct", [])),
            memory_token_count=int(d.get("memory_token_count", 0)),
            failure_tags=list(d.get("failure_tags", [])),
        )


def fact_contains_prohibited_language(text: str) -> List[str]:
    """Return prohibited inferential terms found in text (case-insensitive)."""
    lowered = text.lower()
    return [term for term in PROHIBITED_INFERENTIAL_TERMS if term in lowered]


def validate_fact_constraints(fact: EpisodicFact) -> None:
    """Fail loudly if a verified fact contains inferential / advice language."""
    if not fact.verified:
        return
    blobs = [fact.outcome]
    if fact.action:
        blobs.append(fact.action)
    hits: List[str] = []
    for blob in blobs:
        hits.extend(fact_contains_prohibited_language(blob))
    if hits:
        raise ValueError(
            f"Verified fact {fact.fact_id} contains prohibited inferential terms "
            f"{sorted(set(hits))}: outcome/action must stay factual."
        )


def validate_cognition_provenance(
    cognition: CognitiveMemory,
    fact_ids: set[str],
) -> None:
    """Every supporting / counterevidence fact id must exist."""
    missing = [
        fid
        for fid in cognition.supporting_fact_ids + cognition.counterevidence_fact_ids
        if fid not in fact_ids
    ]
    if missing:
        raise ValueError(
            f"Cognition {cognition.cognition_id} references missing facts: {missing}"
        )
    if cognition.status not in {
        "candidate",
        "tentative",
        "validated",
        "validated_within_scope",
        "revised",
        "rejected",
        "stale",
    }:
        raise ValueError(
            f"Cognition {cognition.cognition_id} has invalid status {cognition.status!r}"
        )


def validate_decision_provenance(
    decision: DecisionTrace,
    fact_ids: set[str],
    cognition_ids: set[str],
) -> None:
    """Loaded memories on a decision must resolve to known store entries."""
    missing_facts = [fid for fid in decision.loaded_fact_ids if fid not in fact_ids]
    missing_cogs = [
        cid for cid in decision.loaded_cognition_ids if cid not in cognition_ids
    ]
    if missing_facts or missing_cogs:
        raise ValueError(
            f"Decision {decision.decision_id} has broken provenance: "
            f"missing_facts={missing_facts}, missing_cognitions={missing_cogs}"
        )
