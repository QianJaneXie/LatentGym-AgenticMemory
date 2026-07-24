# Agentic Memory — Phase 1 Note

**Status:** Pilot 1 complete on `range_100` (single trajectory, 7 episodes, GPT)  
**Depends on:** Phase 0 (`AGENTIC_MEMORY_PHASE0_NOTE.md`)  
**Scope:** single-layer factual memory baselines (no cognition / no drill-down / no reconciliation layer)

---

## Relation to `AGENTIC_MEMORY_PLAN.md`

Pilot 1 minimum set (plan §15 / Pilot 1 list):

| Plan baseline | Runner condition | Status |
|---|---|---|
| No cross-task memory | `no_memory` | Done |
| Full history | `full_history` | Done |
| Outcome-only factual memory | `outcome_only` | Done |
| Context–action–outcome factual memory | `episodic_only` (dense turn + outcome records) | Done |
| Oracle factual summary | `oracle_summary` | Done |

Defaults used for reporting: **7 episodes**, latent **`range_100`**, feedback **`standard`**, presentation **read-all**, model **GPT-5.6**.  
`set_of_2` remains optional plumbing smoke only.

---

## What landed

```text
latentgym/memory/
  types.py, episodic_store.py, fact_extractor.py,
  retriever.py, decision_logger.py, __init__.py

latentgym/eval/memory_agent/runner.py
experiments/memory/run_baselines.py
tests/memory/
```

Conditions: `no_memory`, `full_history`, `outcome_only`, `episodic_only`, `oracle_summary`.

---

## How to run Pilot 1

```bash
source .env && source "$VENV_DIR/bin/activate"
export PYTHONPATH="${PWD}:${PWD}/TextArena:${PYTHONPATH:-}"

python -m pytest tests/memory -q

python experiments/memory/run_baselines.py \
  --model llmcenter:gpt-5.6-sol \
  --env number_guessing --latent range_100 \
  --prompt full_info --feedback standard \
  --n-trajectories 1 --num-episodes 7 \
  --trajectory-dir latentgym/data/eval/ \
  --output latentgym/results/memory_phase1_gpt56_range100_standard/
```

Subset / merge:

```bash
python experiments/memory/run_baselines.py ... \
  --conditions outcome_only oracle_summary --merge-existing
```

---

## Pilot 1 findings (`range_100`, `traj_000`, 7 episodes)

Dir: `latentgym/results/memory_phase1_gpt56_range100_standard/`

| Condition | reward | turns |
|---|---|---|
| no_memory | 5.700 | `[9, 10, 10, 8, 10, 8, 10]` |
| full_history | 6.120 | `[9, 8, 6, 7, 5, 4, 5]` |
| outcome_only | 6.020 | `[9, 8, 6, 7, 7, 7, 5]` |
| episodic_only | **6.140** | `[9, 8, 6, 6, 5, 4, 5]` |
| oracle_summary | 6.120 | `[9, 8, 6, 7, 5, 4, 5]` |

Reading (single-seed pilot, not a multi-seed proof):

- Any factual memory beats no memory.
- Dense context–action–outcome (`episodic_only`) ≈ full history, slightly above outcome-only.
- Oracle summary matches full history here; does not beat dense facts on this seed.
- Single trajectory by design for this pilot; next work is Pilot 2, not more Pilot 1 seeds.

### Smoke: `set_of_2` + `information` (`traj_000`, 7 episodes)

Dir: `latentgym/results/memory_phase1_gpt56_setof2_info/` — plumbing only; too easy for representation headroom.

---

## Safety invariants

1. Fact extraction uses only agent-visible turn text + episode-end feedback.
2. Verified facts reject inferential terms.
3. Decision traces reference known fact IDs.
4. Presentation only uses `episode_idx < current`.
5. Oracle summary uses only agent-visible revealed targets / outcomes (no latent `range_start` / `set_values`).

---

## Next

- **Pilot 2** (representation + Hermes-pattern skill / Mem0-style flat extract): done — see `AGENTIC_MEMORY_PHASE2_NOTE.md`.
- **Pilot 3 / plan Phase 2** (fact reconciliation, `FactClaim` / relations / current view): not started. Faithful Mem0 top-k remains deferred until scale hurts.
