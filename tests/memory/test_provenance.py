"""Provenance validation tests."""
from __future__ import annotations

import pytest

from latentgym.memory.types import (
    CognitiveMemory,
    DecisionTrace,
    validate_cognition_provenance,
    validate_decision_provenance,
)


def test_cognition_requires_existing_facts():
    cog = CognitiveMemory(
        cognition_id="h1",
        claim="Targets may recur.",
        scope={"environment": "number_guessing"},
        action_implication="Try prior targets first.",
        supporting_fact_ids=["f1", "missing"],
        falsification_condition="A later target outside the set.",
        status="candidate",
    )
    with pytest.raises(ValueError, match="missing facts"):
        validate_cognition_provenance(cog, fact_ids={"f1"})


def test_decision_requires_existing_memories():
    dec = DecisionTrace(
        decision_id="d1",
        trajectory_id="t0",
        episode_idx=1,
        decision_type="first_guess",
        query="q",
        loaded_fact_ids=["f1", "f9"],
        loaded_cognition_ids=["h1"],
        cited_memory_ids=[],
        action="[115]",
        outcome="correct",
    )
    with pytest.raises(ValueError, match="broken provenance"):
        validate_decision_provenance(dec, fact_ids={"f1"}, cognition_ids=set())


def test_valid_decision_passes():
    dec = DecisionTrace(
        decision_id="d1",
        trajectory_id="t0",
        episode_idx=1,
        decision_type="first_guess",
        query="q",
        loaded_fact_ids=["f1"],
        loaded_cognition_ids=[],
        cited_memory_ids=[],
        action="[115]",
        outcome="correct",
    )
    validate_decision_provenance(dec, fact_ids={"f1"}, cognition_ids=set())
