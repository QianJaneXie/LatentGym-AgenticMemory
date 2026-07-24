"""Hermes-pattern: skill distillation runs in the live task conversation."""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Dict, List

import pytest

import latentgym.envs.number_guessing  # noqa: F401
from latentgym.core.env_config import FullyDefinedEnv
from latentgym.core.registry import make_env
from latentgym.eval.memory_agent.runner import MemoryAPIRunner
from latentgym.eval.model_interface import MockModel, ModelResponse
from latentgym.memory.retriever import build_inline_skill_distillation_prompt


class RecordingMockModel(MockModel):
    """Mock that records every generate() message list."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.calls: List[List[Dict[str, str]]] = []

    async def generate(self, messages: List[Dict[str, str]], **kwargs) -> ModelResponse:
        self.calls.append([dict(m) for m in messages])
        return await super().generate(messages, **kwargs)


def _skill_calls(model: RecordingMockModel) -> List[List[Dict[str, str]]]:
    marker = build_inline_skill_distillation_prompt()
    out = []
    for msgs in model.calls:
        if any(m.get("content") == marker for m in msgs):
            out.append(msgs)
    return out


def test_skill_only_llm_distills_in_task_conversation():
    traj_dir = Path("latentgym/data/eval/number_guessing/set_of_2")
    traj_path = traj_dir / "traj_000.json"
    if not traj_path.exists():
        pytest.skip("Phase 0 trajectories not present")

    fd = FullyDefinedEnv(
        env_name="number_guessing",
        latent_id="set_of_2",
        prompt_id="full_info",
        feedback_id="information",
        num_episodes=3,
    )
    # Alternate plausible guesses with skill bullets (extra skill calls after episodes).
    guess_cycle = [f"[{g}]" for g in (500, 250, 125, 115, 655)]
    skill_line = "- Prefer mid-range probes then reuse revealed targets when allowed."
    responses: List[str] = []
    for _ in range(40):
        responses.extend(guess_cycle)
        responses.append(skill_line)
    model = RecordingMockModel(
        name="mock:inline-skill",
        responses=responses,
        default_response="[500]",
    )

    async def _run():
        env = make_env(fd, trajectory_path=str(traj_path))
        runner = MemoryAPIRunner(
            model, condition="skill_only_llm", env_name="number_guessing"
        )
        return await runner.run_trajectory(env, seed=0, trajectory_id="traj_000")

    result = asyncio.run(_run())
    skill_calls = _skill_calls(model)
    assert skill_calls, "expected at least one inline skill-distill generate() call"

    # Skill write must see the live task conversation, not a blind outcomes-only pair.
    first = skill_calls[0]
    assert first[0]["role"] == "system"
    joined = "\n".join(m["content"] for m in first)
    assert "Number Guessing" in joined or "number" in joined.lower()
    assert "Verified episode outcomes" not in joined
    assert "You distill short reusable skills from verified game outcomes" not in joined
    # full_info rules / episode framing should be present in the task system or user turns.
    assert "Episode" in joined or "Game" in joined or "contiguous" in joined.lower()

    mem = result.metadata["memory"]
    assert mem["condition"] == "skill_only_llm"
    assert mem["distilled_skill_history"], "expected distilled_skill_history entries"
    assert "Experience / skill note" in mem["distilled_skill_history"][0]["skill_text"]
