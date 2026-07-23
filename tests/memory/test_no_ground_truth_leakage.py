"""Ensure memory prompts never receive evaluator-only ground truth."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

import latentgym.envs.number_guessing  # noqa: F401
from latentgym.core.env_config import FullyDefinedEnv
from latentgym.core.registry import make_env
from latentgym.eval.memory_agent.runner import MemoryAPIRunner, _assert_no_ground_truth_leakage
from latentgym.eval.model_interface import MockModel
from latentgym.memory.episodic_store import EpisodicStore
from latentgym.memory.fact_extractor import VisibleTurn, extract_number_guessing_facts
from latentgym.memory.retriever import format_facts_for_prompt, retrieve_episodic_facts


def test_assert_detects_config_dump():
    with pytest.raises(RuntimeError, match="target_number"):
        _assert_no_ground_truth_leakage(
            [{"role": "user", "content": '{"target_number": 115}'}]
        )


def test_retrieval_excludes_future_episodes():
    store = EpisodicStore()
    for ep in (0, 1, 2):
        store.extend(
            extract_number_guessing_facts(
                trajectory_id="traj_0",
                episode_idx=ep,
                turns=[VisibleTurn(0, f"[{100 + ep}]", f"Correct! You guessed the number {100 + ep} in 1 turns.")],
                end_feedback=f"Episode {ep + 1} finished. Score: 0.980. The number was {100 + ep}.",
            )
        )
    result = retrieve_episodic_facts(
        store, trajectory_id="traj_0", episode_idx=2
    )
    assert all(f.episode_idx < 2 for f in result.facts)
    # Stage A0 read-all: every prior-episode fact, no top-k truncation.
    assert len(result.facts) == sum(1 for f in store.all_facts() if f.episode_idx < 2)
    assert "read_all" in result.query
    prompt = format_facts_for_prompt(result.facts)
    assert "target_number" not in prompt
    assert '"set_values"' not in prompt


def test_memory_runner_conversation_has_no_config_dump():
    traj_dir = Path("latentgym/data/eval/number_guessing/set_of_2")
    traj_path = traj_dir / "traj_000.json"
    if not traj_path.exists():
        pytest.skip("Phase 0 trajectories not present")

    # Hidden targets from the trajectory file (evaluator-only).
    hidden = json.loads(traj_path.read_text())
    hidden_targets = [ep["target_number"] for ep in hidden["episodes"]]
    set_values = hidden.get("metadata", {}).get("context", {}).get("set_values", [])

    fd = FullyDefinedEnv(
        env_name="number_guessing",
        latent_id="set_of_2",
        prompt_id="full_info",
        feedback_id="information",
        num_episodes=5,
    )
    # Mock that emits valid number guesses cycling mid-range values.
    responses = [f"[{g}]" for g in (500, 250, 125, 115, 655) * 40]
    model = MockModel(name="mock:ng", responses=responses, default_response="[500]")

    async def _run(condition: str):
        env = make_env(fd, trajectory_path=str(traj_path))
        runner = MemoryAPIRunner(model, condition=condition, env_name="number_guessing")
        return await runner.run_trajectory(env, seed=0, trajectory_id="traj_000")

    for condition in ("no_memory", "full_history", "episodic_only"):
        model._call_count = 0
        result = asyncio.run(_run(condition))
        text = "\n".join(m["content"] for m in result.conversation)
        assert '"target_number"' not in text
        assert '"episode_configs"' not in text
        assert '"set_values"' not in text
        # Natural language may reveal targets via information feedback after play;
        # that is allowed. Config dumps / pre-reveal of the full latent set are not.
        if set_values:
            assert f"set_values: {set_values}" not in text
        mem = result.metadata["memory"]
        assert mem["condition"] == condition
        # Facts must not be copied from episode_configs blindly: every fact is extracted.
        for fact in mem["facts"]:
            assert "episode_configs" not in fact.get("source_ref", {})
        # Evaluator still has ground truth on the result object for metrics.
        assert result.episode_configs
        assert [c["target_number"] for c in result.episode_configs] == hidden_targets
