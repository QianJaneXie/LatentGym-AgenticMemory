# Agentic Memory — Phase 0 Note

**Status:** complete (local macOS reproduction)  
**Date:** 2026-07-21  
**Constraint:** no memory package code yet

---

## Acceptance checklist

| Criterion | Result |
|---|---|
| Install succeeds | Yes (macOS-minimal path; see below) |
| Mock-model sanity check | Yes — `mock:random` × `number_guessing/set_of_2` |
| One Number Guessing evaluation | Yes — 3 trajectories × 5 episodes |
| Trajectory JSON / viewer | Yes — results JSON + `trajectory_explorer.html` |
| Episode-boundary path documented | Yes — this note + comment in `api_runner.py` |

---

## Commands that worked

```bash
source .env
source "$VENV_DIR/bin/activate"
export PYTHONPATH="${PWD}:${PWD}/TextArena:${PYTHONPATH:-}"

python -m latentgym.cli.generate_data list --env number_guessing

# JSONs land under latentgym/data/eval/number_guessing/set_of_2/
# (parquet step may fail if latentgym.data.eval.generate is missing; APIRunner does not need parquet)
python -m latentgym.cli.generate_data eval \
  --env number_guessing --latent set_of_2 \
  --n-trajectories 3 --num-episodes 5 \
  --output latentgym/data/eval/

python -m latentgym.cli.run_eval single \
  --models mock:random \
  --env number_guessing --latent set_of_2 \
  --prompt full_info --feedback information \
  --num-episodes 5 --n-trajectories 3 \
  --trajectory-dir latentgym/data/eval/ \
  --output latentgym/results/memory_sanity/

python -m latentgym.cli.report \
  --data-dir latentgym/results/memory_sanity/ \
  --trajectories \
  --output latentgym/results/memory_sanity/report/
```

Artifacts:

- Input trajs: `latentgym/data/eval/number_guessing/set_of_2/traj_00{0,1,2}.json`
- Result trajs: `latentgym/results/memory_sanity/trajectories/mock_random/.../traj_000{0,1,2}.json`
- Viewer: `latentgym/results/memory_sanity/report/trajectory_explorer.html`

Mock reward is `0.0` because `MockModel` defaults to `[red]`, which is invalid for number guessing. That is expected for a no-cost pipeline sanity check.

---

## macOS install caveat

Official `bash setup.sh` runs `uv sync --active --extra vllm` and resolves Linux CUDA wheels (`flash-attn`, `torch+cu128`). That fails on Apple Silicon.

Phase 0 used a **minimal** venv at `$VENV_DIR` (`/Users/a1/scratch/latentgym`) with:

- editable `skyrl-gym`
- `omegaconf`, `pyyaml`, `numpy`, `tqdm`, `openai`, `anthropic`, `google-genai`, `python-dotenv`, `rich`, `nltk`, `chess`

Full GPU / vLLM stack is not required for agentic-memory Phase 1–2 mock and API evals.

---

## Episode-boundary code path

```text
NumberGuessingSingleEpisodeEnv.step
  → (raw_feedback, reward, episode_done, info)

MultiEpisodeEnv.step  (latentgym/core/multi_episode_env.py ~319–424)
  if episode_done:
    append episode_rewards / turns_per_episode
    if not last episode:
      format end feedback + transition message
      _current_episode += 1
      reset next episode_config → next_obs
      pack into ONE user observation
    else:
      end feedback only
  trajectory done iff len(episode_rewards) >= num_episodes
  metadata["episode"] == _current_episode   # already advanced on non-final ends

APIRunner.run_trajectory  (latentgym/eval/single_agent/api_runner.py ~89–127)
  new_episode = step_metadata["episode"]
  if new_episode != current_episode or done:
    flush EpisodeOutcome(s) from episode_rewards
    # ← MEMORY HOOK (future): extract facts for completed episode;
    #   retrieve for next first guess; do not inject episode_configs
```

### Signals to use

| Signal | Meaning |
|---|---|
| `step_metadata["episode"]` changed | Prior episode just finished; value is the **next** episode index |
| `step_result["done"]` | Entire trajectory finished (final episode just ended) |
| `step_metadata["episode_rewards"]` | Completed episode rewards so far |
| Conversation user message on boundary | Contains end-feedback **and** (if not last) next episode initial obs |

### Visibility reminder

- Agent-visible: conversation messages only (and feedback text inside them).
- Evaluator-only: `init_metadata["episode_configs"]`, `EpisodeOutcome.ground_truth`, trajectory JSON `episodes` / `metadata.context.set_values`.
- Never feed evaluator-only fields into task-agent / fact-extractor / hypothesis prompts.

---

## Next

Proceed to **Phase 1** (episodic facts + memory-aware runner; do not modify original `APIRunner` beyond the documentation comment above).
