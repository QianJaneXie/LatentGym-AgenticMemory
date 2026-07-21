"""Episodic store JSON round-trip tests."""
from __future__ import annotations

from latentgym.memory.episodic_store import EpisodicStore
from latentgym.memory.fact_extractor import VisibleTurn, extract_number_guessing_facts


def test_store_roundtrip(tmp_path):
    facts = extract_number_guessing_facts(
        trajectory_id="traj_0",
        episode_idx=0,
        turns=[VisibleTurn(0, "[115]", "Correct! You guessed the number 115 in 1 turns.")],
        end_feedback="Episode 1 finished. Score: 0.980. The number was 115.",
    )
    store = EpisodicStore()
    store.extend(facts)
    path = tmp_path / "facts.json"
    store.save_json(path)

    loaded = EpisodicStore.load_json(path)
    assert loaded.to_list() == store.to_list()
    assert len(loaded) == len(store)
    assert loaded.get(facts[0].fact_id) is not None


def test_store_rejects_duplicate_ids():
    facts = extract_number_guessing_facts(
        trajectory_id="traj_0",
        episode_idx=0,
        turns=[VisibleTurn(0, "[1]", "The number is greater than 1.")],
        end_feedback="Episode 1 finished. Score: 0.000",
    )
    store = EpisodicStore()
    store.append(facts[0])
    try:
        store.append(facts[0])
        assert False, "expected duplicate failure"
    except ValueError as e:
        assert "Duplicate" in str(e)
