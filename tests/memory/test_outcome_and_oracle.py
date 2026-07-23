"""Pilot 1 outcome-only and oracle summary formatting."""
from __future__ import annotations

from latentgym.memory.fact_extractor import VisibleTurn, extract_number_guessing_facts
from latentgym.memory.retriever import (
    build_skill_distillation_prompt,
    format_oracle_summary_from_facts,
    format_skill_from_facts,
    select_outcome_only_facts,
    wrap_distilled_skill,
)


def _facts_two_episodes():
    f0 = extract_number_guessing_facts(
        trajectory_id="traj_0",
        episode_idx=0,
        turns=[
            VisibleTurn(0, "[500]", "The number is greater than 500."),
            VisibleTurn(1, "[669]", "Correct! You guessed the number 669 in 2 turns."),
        ],
        end_feedback="Episode 1 finished. Score: 0.960. The number was 669.",
    )
    f1 = extract_number_guessing_facts(
        trajectory_id="traj_0",
        episode_idx=1,
        turns=[
            VisibleTurn(0, "[669]", "The number is less than 669."),
            VisibleTurn(1, "[658]", "Correct! You guessed the number 658 in 2 turns."),
        ],
        end_feedback="Episode 2 finished. Score: 0.960. The number was 658.",
    )
    return f0 + f1


def test_outcome_only_filters_turn_facts():
    facts = _facts_two_episodes()
    outcomes = select_outcome_only_facts(facts)
    assert len(outcomes) == 2
    assert all(f.decision_idx is None for f in outcomes)
    assert all((f.context or {}).get("decision_type") == "episode_outcome" for f in outcomes)


def test_oracle_summary_uses_visible_targets_only():
    facts = _facts_two_episodes()
    text = format_oracle_summary_from_facts(facts)
    assert "Oracle factual summary" in text
    assert "669" in text and "658" in text
    assert "observed min=658, max=669" in text
    assert "set_values" not in text
    assert "range_start" not in text
    assert "target_number" not in text


def test_skill_note_is_procedural_not_latent_dump():
    facts = _facts_two_episodes()
    text = format_skill_from_facts(facts)
    assert "Experience / skill note" in text
    assert "Suggested procedure" in text
    assert "658" in text and "669" in text
    assert "set_values" not in text
    assert "range_start" not in text


def test_skill_distillation_prompt_uses_outcomes_only():
    facts = _facts_two_episodes()
    prompt = build_skill_distillation_prompt(facts)
    assert "Verified episode outcomes" in prompt
    assert "669" in prompt and "658" in prompt
    assert "set_values" not in prompt
    wrapped = wrap_distilled_skill("- Prefer searching near previously revealed targets.")
    assert "LLM-distilled" in wrapped
    assert "Prefer searching" in wrapped
