"""Fact constraint tests for Phase 1 episodic memory."""
from __future__ import annotations

import pytest

from latentgym.memory.fact_extractor import VisibleTurn, extract_number_guessing_facts
from latentgym.memory.types import EpisodicFact, validate_fact_constraints


def test_verified_fact_rejects_inferential_language():
    bad = EpisodicFact(
        fact_id="f_bad",
        trajectory_id="t0",
        episode_idx=0,
        decision_idx=0,
        context={"environment": "number_guessing"},
        action="500",
        outcome="target is probably always lower because of the pattern",
        source_type="environment_feedback",
        source_ref={},
        verified=True,
        created_at="2026-01-01T00:00:00+00:00",
    )
    with pytest.raises(ValueError, match="prohibited"):
        validate_fact_constraints(bad)


def test_unverified_agent_statement_allows_inferential_language():
    ok = EpisodicFact(
        fact_id="f_stmt",
        trajectory_id="t0",
        episode_idx=0,
        decision_idx=0,
        context={"environment": "number_guessing", "decision_type": "agent_statement"},
        action=None,
        outcome="agent said targets should always recur",
        source_type="agent_statement",
        source_ref={},
        verified=False,
        created_at="2026-01-01T00:00:00+00:00",
    )
    validate_fact_constraints(ok)


def test_extract_guess_and_comparison_facts():
    turns = [
        VisibleTurn(0, "[500]", "The number is less than 500."),
        VisibleTurn(1, "[250]", "The number is greater than 250."),
        VisibleTurn(2, "[300]", "Correct! You guessed the number 300 in 3 turns."),
    ]
    facts = extract_number_guessing_facts(
        trajectory_id="traj_0",
        episode_idx=0,
        turns=turns,
        end_feedback="Episode 1 finished. Score: 0.940. The number was 300.",
    )
    assert len(facts) == 4  # 3 turns + outcome
    assert facts[0].action == "500"
    assert "less than 500" in facts[0].outcome
    assert facts[2].verified is True
    assert "target revealed as 300" in facts[2].outcome
    assert facts[3].context["decision_type"] == "episode_outcome"
    assert "target revealed as 300" in facts[3].outcome
    for fact in facts:
        validate_fact_constraints(fact)


def test_extract_does_not_require_hidden_target_when_unrevealed():
    turns = [
        VisibleTurn(0, "[500]", "The number is less than 500.\nYou've run out of guesses."),
    ]
    facts = extract_number_guessing_facts(
        trajectory_id="traj_0",
        episode_idx=0,
        turns=turns,
        end_feedback="Episode 1 finished. You didn't find the number. Score: 0.000",
    )
    outcome = facts[-1]
    assert "solved=False" in outcome.outcome
    assert "target revealed" not in outcome.outcome
