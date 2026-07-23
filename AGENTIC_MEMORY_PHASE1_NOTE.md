# Agentic Memory — Phase 1 Note

**Status:** complete (Stage A0 read-all pilots with GPT)  
**Depends on:** Phase 0 (`AGENTIC_MEMORY_PHASE0_NOTE.md`)  
**Scope:** single-layer episodic facts only (no cognition / no regression / no drill-down)

---

## Relation to `AGENTIC_MEMORY_PLAN.md`

Matches the plan’s **Phase 1 / Stage A0**:

- single-layer `EpisodicFact` with `source_ref`
- **read-all** presentation of prior-episode facts (chronological; no ranking / top-k)
- three-way comparison: `no_memory` / `full_history` / `episodic_only`

Two-level facts, drill-down, cognition, and retrieval budgets are later stages.  
`fact_budget=None` is the default; a positive budget is reserved for later sweeps only.

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

`APIRunner` was not modified beyond the Phase 0 boundary comment. Stores are append-only; any future budget is a **prompt injection** limit, not disk deletion.

---

## Conditions

| Condition | Cross-episode context | Memory injection |
|---|---|---|
| `full_history` | Full conversation retained | None |
| `no_memory` | Cleared at each episode boundary | None |
| `episodic_only` | Cleared at each episode boundary | **All** prior-episode facts (read-all) |

Memory logs are stored in `TrajectoryResult.metadata["memory"]` (`presentation_mode: read_all`) and as sidecars under `memory/`.

**Explorer caveat:** for `no_memory` / `episodic_only`, saved `conversation` is the **final** episode state. Earlier episodes live in `memory/*_facts.json` and `episode_turns`.

Episode 0 has no prior facts under any condition; memory differences start from episode 1 onward.

---

## How to run

```bash
source .env && source "$VENV_DIR/bin/activate"
export PYTHONPATH="${PWD}:${PWD}/TextArena:${PYTHONPATH:-}"

python -m pytest tests/memory -q

python experiments/memory/run_baselines.py \
  --model llmcenter:gpt-5.6-sol \
  --env number_guessing --latent set_of_2 \
  --prompt full_info --feedback information \
  --n-trajectories 1 --num-episodes 7 \
  --trajectory-dir latentgym/data/eval/ \
  --output latentgym/results/memory_phase1_gpt56_setof2_info/
```

Harder stress (after generating `range_100` trajectories):

```bash
python experiments/memory/run_baselines.py \
  --model llmcenter:gpt-5.6-sol \
  --env number_guessing --latent range_100 \
  --prompt full_info --feedback standard \
  --n-trajectories 1 --num-episodes 7 \
  --trajectory-dir latentgym/data/eval/ \
  --output latentgym/results/memory_phase1_gpt56_range100_standard/
```

Primary model: **GPT-5.6 via LLMCenter**. Horizon matches env default: **7 episodes**.

---

## Pilot findings (read-all, GPT-5.6, 7 episodes)

### `set_of_2` + `information` (`traj_000`)

Dir: `latentgym/results/memory_phase1_gpt56_setof2_info/`

| Condition | reward | turns |
|---|---|---|
| no_memory | 5.720 | `[9, 10, 9, 9, 9, 9, 9]` |
| full_history | **6.520** | `[9, 10, 1, 1, 1, 1, 1]` |
| episodic_only | **6.520** | `[9, 10, 1, 1, 1, 1, 1]` |

Longer horizon widens the gap vs no_memory: memory conditions 1-shot from episode 2 onward; no_memory stays ~9–10 every episode.

### `range_100` + `standard` (`traj_000`)

Dir: `latentgym/results/memory_phase1_gpt56_range100_standard/` — 7-episode re-run in progress / see `baselines_summary.json` when complete.

---

## Safety invariants enforced

1. Fact extraction uses only agent-visible turn text + episode-end feedback.
2. Verified facts reject inferential terms (`because`, `should`, `always`, …).
3. Decision traces must reference known fact IDs.
4. Presentation only considers facts with `episode_idx < current`.
5. Runner scans prompts for JSON-ish ground-truth key dumps before each generate.

---

## Next

- Keep three-way comparison when claiming Stage A results.
- Plan Phase 2 (two-level / drill-down) only if single-layer representation is insufficient.
- Ranking / top-k budgets only after store size causes measurable interference.
- Cognition later, after Stage A acceptance criteria you care about are met.
