# Agentic Memory — Phase 1 Note

**Status:** Stage A0 plumbing + GPT pilots complete; Pilot 1 baseline matrix **not** fully closed  
**Depends on:** Phase 0 (`AGENTIC_MEMORY_PHASE0_NOTE.md`)  
**Scope:** single-layer episodic facts only (no cognition / no regression / no drill-down)

---

## Relation to `AGENTIC_MEMORY_PLAN.md`

Aligned with plan **Phase 1 / Stage A0** and §8.2 baseline families:

| Plan item | Status here |
|---|---|
| Single-layer facts + `source_ref` | Done (`EpisodicFact`) |
| Read-all prior facts (no top-k) | Done (`fact_budget=None`) |
| Horizon **7 episodes** (env default) | Done (reporting runs) |
| Primary latent `range_100` + `standard` | Done for focus; `set_of_2` is smoke only |
| no memory / full history | Done |
| Dense context–action–outcome records plus episode outcomes (`episodic_only`) | Done |
| **Outcome-only** facts condition | **Missing** |
| **Oracle** factual summary | **Missing** |
| Atomic flat facts / Hermes skill baselines | Later (Pilot 2) |
| Two-level facts / drill-down / Mem0 | Later phases |

Earlier 5-episode pilots were discarded / overwritten; do not mix them with 7-episode reporting.

---

## What landed

```text
latentgym/memory/
  types.py, episodic_store.py, fact_extractor.py,
  retriever.py, decision_logger.py, __init__.py

latentgym/eval/memory_agent/
  runner.py          # MemoryAPIRunner
  __init__.py

experiments/memory/run_baselines.py   # default --num-episodes 7
tests/memory/
  test_fact_constraints.py
  test_provenance.py
  test_store_roundtrip.py
  test_no_ground_truth_leakage.py
```

`APIRunner` was not modified beyond the Phase 0 boundary comment. Stores are append-only; budgets mean prompt injection limits, not disk deletion.

---

## Conditions (current runner)

| Condition | Cross-episode context | Memory injection |
|---|---|---|
| `full_history` | Full conversation retained | None |
| `no_memory` | Cleared at each episode boundary | None |
| `episodic_only` | Cleared at each episode boundary | **All** prior-episode facts (read-all; dense turn + outcome records) |

Memory logs: `TrajectoryResult.metadata["memory"]` (`presentation_mode: read_all`) and sidecars under `memory/`.

**Explorer caveat:** for `no_memory` / `episodic_only`, saved `conversation` is the final episode state. Earlier episodes live in `memory/*_facts.json` and `episode_turns`.

Episode 0 has no prior facts; memory differences start from episode 1 onward.

---

## How to run

```bash
source .env && source "$VENV_DIR/bin/activate"
export PYTHONPATH="${PWD}:${PWD}/TextArena:${PYTHONPATH:-}"

python -m pytest tests/memory -q

# Primary Stage A focus
python experiments/memory/run_baselines.py \
  --model llmcenter:gpt-5.6-sol \
  --env number_guessing --latent range_100 \
  --prompt full_info --feedback standard \
  --n-trajectories 1 --num-episodes 7 \
  --trajectory-dir latentgym/data/eval/ \
  --output latentgym/results/memory_phase1_gpt56_range100_standard/

# Optional plumbing smoke (not the main claim latent)
python experiments/memory/run_baselines.py \
  --model llmcenter:gpt-5.6-sol \
  --env number_guessing --latent set_of_2 \
  --prompt full_info --feedback information \
  --n-trajectories 1 --num-episodes 7 \
  --trajectory-dir latentgym/data/eval/ \
  --output latentgym/results/memory_phase1_gpt56_setof2_info/
```

Primary model: **GPT-5.6 via LLMCenter**. Horizon: **7 episodes**.

---

## Pilot findings (read-all, GPT-5.6, 7 episodes)

### Primary: `range_100` + `standard` (`traj_000`)

Dir: `latentgym/results/memory_phase1_gpt56_range100_standard/`

| Condition | reward | turns |
|---|---|---|
| no_memory | 5.700 | `[9, 10, 10, 8, 10, 8, 10]` |
| full_history | 6.120 | `[9, 8, 6, 7, 5, 4, 5]` |
| episodic_only | **6.140** | `[9, 8, 6, 6, 5, 4, 5]` |

Dense factual memory ≈ full history ≫ no memory; later episodes ~4–5 turns (not 1-shot).

### Smoke: `set_of_2` + `information` (`traj_000`)

Dir: `latentgym/results/memory_phase1_gpt56_setof2_info/`

| Condition | reward | turns |
|---|---|---|
| no_memory | 5.720 | `[9, 10, 9, 9, 9, 9, 9]` |
| full_history | **6.520** | `[9, 10, 1, 1, 1, 1, 1]` |
| episodic_only | **6.520** | `[9, 10, 1, 1, 1, 1, 1]` |

Useful for plumbing; too easy for representation / cognition headroom.

---

## Safety invariants enforced

1. Fact extraction uses only agent-visible turn text + episode-end feedback.
2. Verified facts reject inferential terms (`because`, `should`, `always`, …).
3. Decision traces must reference known fact IDs.
4. Presentation only considers facts with `episode_idx < current`.
5. Runner scans prompts for JSON-ish ground-truth key dumps before each generate.

---

## Next (to finish Pilot 1 on `range_100`)

1. **Add `outcome_only` condition** — inject only episode-outcome facts; compare to the current dense `episodic_only` store (plan: outcome-only vs context–action–outcome).
2. **Multi-seed** — run at least `traj_001` / `traj_002` (already generated under `data/eval/number_guessing/range_100/`) for the three (then four) conditions.
3. **Oracle factual summary** (handwritten or scripted from agent-visible outcomes, e.g. “targets fall in [655,755]”) as an upper-bound factual representation — not hidden latent dumps.
4. Defer atomic LLM-extracted facts, Hermes-style skill baselines, drill-down, and Mem0 until after the above.
