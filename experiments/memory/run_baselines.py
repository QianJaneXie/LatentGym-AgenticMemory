#!/usr/bin/env python3
"""Phase 1 baselines: no_memory / full_history / episodic_only on Number Guessing.

Reuses the same trajectory JSON files across conditions.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path

import latentgym.envs.number_guessing  # noqa: F401
from latentgym.core.env_config import FullyDefinedEnv
from latentgym.core.registry import make_env
from latentgym.core.trajectory_utils import load_manifest
from latentgym.eval.memory_agent import MemoryAPIRunner
from latentgym.eval.model_interface import MockModel
from latentgym.cli.run_eval import _parse_model_spec

logger = logging.getLogger(__name__)

CONDITIONS = ("no_memory", "full_history", "episodic_only")


def _load_traj_files(trajectory_dir: Path, env_name: str, latent_id: str) -> list[Path]:
    nested = trajectory_dir / env_name / latent_id
    flat = trajectory_dir / f"{env_name}_{latent_id}"
    subdir = nested if nested.exists() else flat if flat.exists() else trajectory_dir
    manifest = load_manifest(str(subdir))
    return [subdir / f for f in manifest.trajectory_files]


async def run_condition(
    *,
    model,
    condition: str,
    fd: FullyDefinedEnv,
    traj_files: list[Path],
    output_dir: Path,
    n_trajectories: int,
) -> list[dict]:
    runner = MemoryAPIRunner(
        model,
        condition=condition,  # type: ignore[arg-type]
        env_name=fd.env_name,
    )
    out = output_dir / condition
    traj_out = out / "trajectories"
    traj_out.mkdir(parents=True, exist_ok=True)
    memory_out = out / "memory"
    memory_out.mkdir(parents=True, exist_ok=True)

    summaries = []
    for i, traj_path in enumerate(traj_files[:n_trajectories]):
        # Fresh model call counter per trajectory for MockModel fairness across conditions
        if hasattr(model, "_call_count"):
            model._call_count = 0
        env = make_env(fd, trajectory_path=str(traj_path))
        result = await runner.run_trajectory(
            env,
            seed=i,
            trajectory_id=traj_path.stem,
        )
        result.benchmark_id = fd.benchmark_id
        result.prompt_id = fd.prompt_id
        result.feedback_id = fd.feedback_id

        path = traj_out / f"traj_{i:04d}.json"
        path.write_text(json.dumps(result.to_dict(), indent=2, default=str))

        mem = result.metadata.get("memory", {})
        (memory_out / f"traj_{i:04d}_facts.json").write_text(
            json.dumps(mem.get("facts", []), indent=2)
        )
        (memory_out / f"traj_{i:04d}_decisions.json").write_text(
            json.dumps(mem.get("decisions", []), indent=2)
        )

        summaries.append(
            {
                "trajectory_id": traj_path.stem,
                "condition": condition,
                "cumulative_reward": result.cumulative_reward,
                "episode_rewards": result.episode_rewards,
                "episode_turns": result.episode_turns,
                "n_facts": mem.get("n_facts", 0),
                "n_decisions": mem.get("n_decisions", 0),
                "conversation_messages": len(result.conversation),
            }
        )
        logger.info(
            "  %s %s reward=%.3f facts=%s msgs=%s",
            condition,
            traj_path.name,
            result.cumulative_reward,
            mem.get("n_facts"),
            len(result.conversation),
        )
    (out / "summary.json").write_text(json.dumps(summaries, indent=2))
    return summaries


async def main_async(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    fd = FullyDefinedEnv(
        env_name=args.env,
        latent_id=args.latent,
        prompt_id=args.prompt,
        feedback_id=args.feedback,
        num_episodes=args.num_episodes,
    )
    traj_files = _load_traj_files(Path(args.trajectory_dir), args.env, args.latent)
    if not traj_files:
        raise FileNotFoundError(f"No trajectories under {args.trajectory_dir}")

    model = _parse_model_spec(args.model)
    # For number guessing mock sanity, prefer numeric guesses if using mock:random
    if isinstance(model, MockModel) and model._default_response == "[red]":
        model = MockModel(
            name=model.name,
            responses=[f"[{g}]" for g in (500, 250, 125, 62, 115, 655) * 50],
            default_response="[500]",
        )

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_summaries = {}
    for condition in CONDITIONS:
        logger.info("Running condition=%s", condition)
        # Independent model instance state per condition
        if hasattr(model, "_call_count"):
            model._call_count = 0
        all_summaries[condition] = await run_condition(
            model=model,
            condition=condition,
            fd=fd,
            traj_files=traj_files,
            output_dir=output_dir,
            n_trajectories=args.n_trajectories,
        )

    (output_dir / "baselines_summary.json").write_text(
        json.dumps(all_summaries, indent=2)
    )
    print(f"Wrote {output_dir / 'baselines_summary.json'}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="mock:random")
    p.add_argument("--env", default="number_guessing")
    p.add_argument("--latent", default="set_of_2")
    p.add_argument("--prompt", default="full_info")
    p.add_argument("--feedback", default="information")
    p.add_argument("--num-episodes", type=int, default=7)
    p.add_argument("--n-trajectories", type=int, default=3)
    p.add_argument("--trajectory-dir", default="latentgym/data/eval/")
    p.add_argument("--output", default="latentgym/results/memory_phase1/")
    return p


def main() -> None:
    args = build_parser().parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
