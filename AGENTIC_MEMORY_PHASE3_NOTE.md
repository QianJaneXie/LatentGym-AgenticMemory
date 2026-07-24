# Agentic Memory — Pilot 3 / Bandits Note

**Status:** in progress (Bandits + reconciled_view MVP)  
**Depends on:** Pilot 1–2 (`AGENTIC_MEMORY_PHASE1_NOTE.md`, `AGENTIC_MEMORY_PHASE2_NOTE.md`)  
**Plan mapping:** Pilot 3 / engineering **Phase 2** (fact reconciliation)

---

## Setup

| Field | Value |
|---|---|
| Environment | `bandits` (LatentGym default **10 episodes**) |
| Latent | `hot_hand` (best arm often persists; flips create supersession) |
| Prompt / feedback | `full_info` / `information` (reveals best + probabilities — agent-visible) |
| Model | GPT-5.6 via LLMCenter |
| Trajectories | `latentgym/data/eval/bandits/hot_hand/` (3 files; pilot uses `traj_000000`) |

Why Bandits (not NG): session-state claims like “latest revealed best button” naturally **supersede** when the latent drifts; episode outcome events with the same button value stay distinct.

---

## Comparison matrix

| Condition | What is injected | Role |
|---|---|---|
| `no_memory` | nothing cross-episode | floor |
| `full_history` | raw chat | upper reference |
| `outcome_only` | episode outcomes only | sparse facts |
| `episodic_only` | dense explore/select + outcomes (append-only read-all) | **ours without reconcile** |
| `oracle_summary` | compact restatement of visible outcomes | oracle facts |
| `reconciled_view` | deterministic `CurrentFactView` (active best + superseded history + explore tallies) | **ours + reconciliation MVP** |
| `atomic_flat_llm` | Mem0-style flat LLM notes | representation baseline |
| `skill_only_llm` | same-conversation LLM skill only | Hermes-pattern inline distill |
| `facts_plus_skill_llm` | dense facts + same-conversation skill | facts vs skill interaction |

Reconciliation MVP rules (`latentgym/memory/reconcile.py`):

- episode outcomes are immutable historical events;
- same best button in two episodes → `same_value_new_event`, not a merge;
- latest revealed best is the only **active** session-state claim; priors are **superseded** (kept for audit);
- explore reward tallies are deterministic aggregates.

---

## How to run

```bash
source .env && source "$VENV_DIR/bin/activate"
export PYTHONPATH="${PWD}:${PWD}/TextArena:${PYTHONPATH:-}"

# data (JSON only; parquet step may fail in CLI)
python - <<'PY'
from latentgym.envs.bandits.trajectory_generator import generate_bandit_trajectories
generate_bandit_trajectories(
    latent_id="hot_hand", num_episodes=10, n_trajectories=3, seed=42,
    output_dir="latentgym/data/eval/bandits/hot_hand/",
)
PY

python -m pytest tests/memory/test_bandits_reconcile.py -q

python experiments/memory/run_baselines.py \
  --model llmcenter:gpt-5.6-sol \
  --env bandits --latent hot_hand \
  --prompt full_info --feedback information \
  --n-trajectories 1 \
  --trajectory-dir latentgym/data/eval/ \
  --output latentgym/results/memory_bandits_hot_hand_gpt56/ \
  --conditions no_memory full_history outcome_only episodic_only oracle_summary reconciled_view
```

LLM skill / flat extras:

```bash
python experiments/memory/run_baselines.py ... \
  --conditions atomic_flat_llm skill_only_llm facts_plus_skill_llm \
  --merge-existing
```

---

## Results

### Mock plumbing

Dir: `latentgym/results/memory_bandits_hot_hand_mock/` — scripted mock ignores memory (equal rewards); used to verify extractors + reconciled_view rendering.

### GPT-5.6 (`traj_000000`, 10 episodes)

Dir: `latentgym/results/memory_bandits_hot_hand_gpt56/`

| Condition | reward | turns |
|---|---|---|
| **atomic_flat_llm** | **7.265** | `[13, 2, 2, 16, 3, 2, 2, 9, 4, 2]` |
| **episodic_only** | **7.190** | `[6, 7, 8, 10, 3, 3, 10, 3, 3, 10]` |
| oracle_summary | 6.325 | `[13, 12, 7, 2, 6, 2, 2, 15, 4, 1]` |
| skill_only_llm | 6.070 | `[10, 8, 10, 9, 9, 10, 8, 9, 10, 9]` |
| no_memory | 5.980 | `[10, 9, 12, 8, 10, 10, 15, 8, 29, 11]` |
| full_history | 5.745 | `[12, 13, 1, 1, 1, 1, 1, 1, 1, 1]` |
| reconciled_view | 5.685 | `[16, 17, 1, 1, 1, 1, 1, 1, 1, 1]` |
| facts_plus_skill_llm | 5.520 | `[8, 14, 9, 18, 3, 4, 8, 9, 3, 4]` |
| outcome_only | 4.850 | `[11, 10, 2, 1, 2, 2, 13, 1, 2, 2]` |

Single-seed reading:

- Top tier: `atomic_flat_llm` ≈ `episodic_only` (flat slightly higher on this seed).
- `reconciled_view` did **not** beat dense/flat facts; after early episodes it often locked in on turn 1 (like `full_history`), suggesting over-trust in the ACTIVE latest-best claim when `hot_hand` flips.
- `skill_only_llm` beats no-memory but trails dense/flat facts; `facts_plus_skill_llm` is worse than facts alone (skill may interfere).
- `outcome_only` is weakest — missing explore tallies hurts.

Primary follow-up: refine reconciliation presentation (keep recent explore evidence beside ACTIVE best; avoid implying “select immediately”), and/or add `episodic_plus_reconciled`.
