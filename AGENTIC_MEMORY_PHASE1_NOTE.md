# Agentic Memory — Phase 1 Note

**Status:** complete (NG Stage A0 / plan Phase 1 pilot)  
**Depends on:** Phase 0 (`AGENTIC_MEMORY_PHASE0_NOTE.md`)  
**Scope:** single-layer episodic facts only (no cognition / no regression / no drill-down)

---

## Relation to `AGENTIC_MEMORY_PLAN.md`

The current plan’s **Phase 1 / Stage A0** is a minimal single-layer factual baseline: write agent-visible facts, then test whether they change behavior vs no memory and full history. Two-level facts, drill-down, cognition, and learned top-k retrieval are later stages.

**This note matches that intent.** We landed a single-layer `EpisodicFact` with `source_ref` provenance and a three-way condition comparison.

Known deviation from the plan’s ideal A0 wording (“read-all, no top-k”):

- `episodic_only` still uses a deterministic ranker with **`fact_budget=10`**.
- On short `set_of_2` pilots this is often close to showing the useful outcomes; on denser stores (many per-turn facts) it is **not** a strict read-all of every fact.
- Treat true read-all (or a much higher budget) as a small follow-up if we need stricter A0 alignment; do not treat the current top-10 as the final retrieval design.

Use the plan for target architecture and RQ numbering; use this note for what actually ran.

---

## What landed

```text
latentgym/memory/
  types.py, episodic_store.py, fact_extractor.py,
  retriever.py, decision_logger.py, __init__.py

latentgym/eval/memory_agent/
  runner.py          # MemoryAPIRunner
  __init__.py

experiments/memory/run_baselines.py
tests/memory/
  test_fact_constraints.py
  test_provenance.py
  test_store_roundtrip.py
  test_no_ground_truth_leakage.py
```

`APIRunner` was not modified beyond the Phase 0 boundary comment. Stores are append-only; budgets here mean **prompt injection limits**, not disk deletion.

---

## Conditions

| Condition | Cross-episode context | Memory injection |
|---|---|---|
| `full_history` | Full conversation retained | None |
| `no_memory` | Cleared at each episode boundary | None |
| `episodic_only` | Cleared at each episode boundary | Up to 10 ranked facts before first guess |

Memory logs are stored in `TrajectoryResult.metadata["memory"]` and as sidecars under `memory/`.

**Explorer caveat:** for `no_memory` / `episodic_only`, saved `conversation` is the **final** episode state. Earlier episodes live in `memory/*_facts.json` and `episode_turns`, not the end-state chat. Open `full_history/report/` to see the full multi-episode transcript.

Episode 0 has no prior facts under any condition; memory differences start from episode 1 onward. First-episode turn variance across runs is mostly sampling noise, not a memory-condition effect.

---

## How to run

```bash
source .env && source "$VENV_DIR/bin/activate"
export PYTHONPATH="${PWD}:${PWD}/TextArena:${PYTHONPATH:-}"

python -m pytest tests/memory -q

python experiments/memory/run_baselines.py \
  --model mock:random \
  --env number_guessing --latent set_of_2 \
  --prompt full_info --feedback information \
  --n-trajectories 3 --num-episodes 5 \
  --trajectory-dir latentgym/data/eval/ \
  --output latentgym/results/memory_phase1/
```

Primary model for real-API pilots: **GPT-5.6 via LLMCenter** (`llmcenter:gpt-5.6-sol` or the current gateway id).

Result dirs include:

- `latentgym/results/memory_phase1_{gpt56,kimi,fable5,minicpm}/` — `set_of_2` + `information`
- `latentgym/results/memory_phase1_gpt56_standard/` — `set_of_2` + `standard`
- `latentgym/results/memory_phase1_gpt56_range100_standard/` — `range_100` + `standard`

---

## Pilot findings

### `set_of_2` + `information` (same `traj_000`)

| Model | no_memory | full_history | episodic_only |
|---|---|---|---|
| MiniCPM | 0 | 0 | 0 |
| Kimi | ~4.04 | ~4.50 | ~4.52 |
| GPT-5.6 | ~4.08 | ~4.54 | ~4.54 |
| Fable5 | ~4.00 | ~4.56 | ~4.34 |

On GPT: later episodes solve in **1–2 turns** under both `full_history` and `episodic_only`, while `no_memory` stays ~9–10 turns/episode. Supports plan **RQ1 / RQ3** on this easy setting (small factual set helps; can match full history). Pilot only, not multi-seed proof.

`set_of_2` + `standard` with GPT was essentially unchanged: successful episodes still reveal the target via `Correct! ...`, so the factual layer still gets the points.

### `range_100` + `standard` (GPT, `traj_000`)

| Condition | reward | turns |
|---|---|---|
| no_memory | ~4.06 | `[9, 9, 10, 10, 9]` |
| full_history | ~4.28 | `[9, 7, 7, 6, 7]` |
| episodic_only | ~4.28 | `[10, 7, 8, 5, 6]` |

Memory still helps and `episodic_only ≈ full_history`, but later episodes stay ~5–7 turns (interval search), not 1-shot. Better Stage A stress test than `set_of_2`.

Observed on mock Phase 0 trajectories (`traj_000` solves; others time out):

- `full_history` keeps far more messages
- `episodic_only` injects `Verified past records...` after episode 0
- no `"target_number"` / `"set_values"` / `"episode_configs"` keys appear in agent messages

---

## Safety invariants enforced

1. Fact extraction uses only agent-visible turn text + episode-end feedback.
2. Verified facts reject inferential terms (`because`, `should`, `always`, …).
3. Decision traces must reference known fact IDs.
4. Retrieval only considers facts with `episode_idx < current`.
5. Runner scans prompts for JSON-ish ground-truth key dumps before each generate.

---

## Next

- Prefer reporting Stage A claims with the three-way comparison; default treatment path remains `episodic_only`.
- Optional A0 tightening: read-all (or raise budget) so presentation matches the plan’s “no premature top-k” wording.
- Plan **Phase 2**: two-level facts / drill-down only if single-layer representation is clearly insufficient.
- Ranking / top-k / context budgets: only after store size causes measurable interference (plan priority order).
- Cognition / paired regression: later plan phases, after Stage A acceptance criteria you care about are met.
