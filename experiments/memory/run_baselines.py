#!/usr/bin/env python3
"""Memory baselines on Number Guessing or Bandits.

Reuses the same trajectory JSON files across conditions.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path

import latentgym.envs.bandits  # noqa: F401
import latentgym.envs.number_guessing  # noqa: F401
from latentgym.core.env_config import FullyDefinedEnv
from latentgym.core.registry import make_env
from latentgym.core.trajectory_utils import load_manifest
from latentgym.eval.memory_agent import MemoryAPIRunner
from latentgym.eval.model_interface import MockModel
from latentgym.cli.run_eval import _parse_model_spec

logger = logging.getLogger(__name__)

PILOT1_CONDITIONS = (
    "no_memory",
    "full_history",
    "outcome_only",
    "episodic_only",
    "oracle_summary",
)

PILOT2_CONDITIONS = (
    "skill_only",
    "facts_plus_skill",
    "skill_only_llm",
    "facts_plus_skill_llm",
    "atomic_flat_llm",
)

# Bandits Pilot 3: prior baselines + reconciled current view
BANDITS_PILOT_CONDITIONS = (
    "no_memory",
    "full_history",
    "outcome_only",
    "episodic_only",
    "oracle_summary",
    "reconciled_view",
    "atomic_flat_llm",
    "skill_only_llm",
    "facts_plus_skill_llm",
)


def _load_traj_files(trajectory_dir: Path, env_name: str, latent_id: str) -> list[Path]:
    nested = trajectory_dir / env_name / latent_id
    flat = trajectory_dir / f"{env_name}_{latent_id}"
    subdir = nested if nested.exists() else flat if flat.exists() else trajectory_dir
    manifest = load_manifest(str(subdir))
    return [subdir / f for f in manifest.trajectory_files]


def _configure_mock_for_env(model, env_name: str):
    """Give MockModel env-plausible scripted actions when using the default mock."""
    if not isinstance(model, MockModel):
        return model
    if env_name == "bandits":
        buttons = ["red", "blue", "green", "yellow", "purple"]
        responses = []
        for b in buttons:
            responses.append(f"[{b}]")
        responses.append("[select red]")
        # Repeat explore/select patterns for many episodes
        script = (responses * 40)
        return MockModel(name=model.name, responses=script, default_response="[select red]")
    if env_name == "number_guessing":
        return MockModel(
            name=model.name,
            responses=[f"[{g}]" for g in (500, 250, 125, 62, 115, 655) * 50],
            default_response="[500]",
        )
    return model


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
        if mem.get("reconciled_views"):
            (memory_out / f"traj_{i:04d}_reconciled_views.json").write_text(
                json.dumps(mem.get("reconciled_views", []), indent=2)
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

    model = _configure_mock_for_env(_parse_model_spec(args.model), args.env)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.conditions:
        conditions = tuple(args.conditions)
    elif args.env == "bandits":
        conditions = BANDITS_PILOT_CONDITIONS
    else:
        conditions = PILOT1_CONDITIONS

    all_summaries = {}
    summary_path = output_dir / "baselines_summary.json"
    if args.merge_existing and summary_path.exists():
        all_summaries = json.loads(summary_path.read_text())

    for condition in conditions:
        logger.info("Running condition=%s", condition)
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

    summary_path.write_text(json.dumps(all_summaries, indent=2))
    print(f"Wrote {summary_path}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="mock:random")
    p.add_argument("--env", default="number_guessing")
    p.add_argument("--latent", default="range_100")
    p.add_argument("--prompt", default="full_info")
    p.add_argument("--feedback", default="standard")
    p.add_argument(
        "--num-episodes",
        type=int,
        default=None,
        help="Default: 7 for number_guessing, 10 for bandits (env designer default).",
    )
    p.add_argument("--n-trajectories", type=int, default=1)
    p.add_argument("--trajectory-dir", default="latentgym/data/eval/")
    p.add_argument("--output", default="latentgym/results/memory_phase1/")
    p.add_argument(
        "--conditions",
        nargs="+",
        default=None,
        help=(
            f"Subset of conditions. Pilot1={list(PILOT1_CONDITIONS)}; "
            f"Pilot2={list(PILOT2_CONDITIONS)}; "
            f"Bandits default={list(BANDITS_PILOT_CONDITIONS)}."
        ),
    )
    p.add_argument(
        "--merge-existing",
        action="store_true",
        help="Merge into existing baselines_summary.json instead of replacing it",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.num_episodes is None:
        args.num_episodes = 10 if args.env == "bandits" else 7
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
