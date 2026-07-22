# Agentic Memory — Phase 1 Note

**Status:** complete (NG Stage A pilot)  
**Depends on:** Phase 0 (`AGENTIC_MEMORY_PHASE0_NOTE.md`)  
**Scope:** explicit episodic facts only (no cognition / no regression)

---

## Relation to `AGENTIC_MEMORY_PLAN.md`

The plan describes an ideal **two-layer** factual stack (`DetailedFact` → compact `FactualMemory`) plus hierarchical drill-down (`OPEN_DETAILED_FACT`).

**This Phase 1 delivery does not implement that stack.** For Number Guessing Stage A we keep a **single-layer** `EpisodicFact` with `source_ref` provenance. Full detailed/compact split and drill-down are **deferred** until a setting where compact summaries are actually lossy.

Plan section numbering also moved: handwritten cognition / paired regression is **Phase 3** in the current plan (not “Phase 2” as older notes said). Use the plan for target architecture; use this note for what actually landed.

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

`APIRunner` was not modified beyond the Phase 0 boundary comment.

---

## Conditions

| Condition | Cross-episode context | Memory injection |
|---|---|---|
| `full_history` | Full conversation retained | None |
| `no_memory` | Cleared at each episode boundary | None |
| `episodic_only` | Cleared at each episode boundary | Up to 10 retrieved facts before first guess |

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

Primary model for real-API pilots: **GPT-5.6 via LLMCenter** (`llmcenter:gpt-5.6-sol` or the current gateway id). Results dirs: `latentgym/results/memory_phase1_{gpt56,kimi,fable5,minicpm}/`.

---

## Pilot findings (same `traj_000`, `set_of_2`, information feedback)

| Model | no_memory | full_history | episodic_only |
|---|---|---|---|
| MiniCPM | 0 | 0 | 0 |
| Kimi | ~4.04 | ~4.50 | ~4.52 |
| GPT-5.6 | ~4.08 | ~4.54 | ~4.54 |
| Fable5 | ~4.00 | ~4.56 | ~4.34 |

On GPT (cleanest signal): later episodes solve in **1–2 turns** under both `full_history` and `episodic_only`, while `no_memory` stays ~9–10 turns/episode. That supports RQ1/RQ2 **on this trajectory and feedback**: compact episodic facts match full history for exploitation and beat no cross-task memory. It is a pilot signal, not a multi-seed proof.

Observed on mock Phase 0 trajectories (`traj_000` solves; others time out):

- `full_history` keeps far more messages (e.g. 47 vs 14 on `traj_000`)
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

- Broader Stage A pilots (more trajectories / latents) still use the three-way comparison when measuring factual memory.
- Default **treatment / default agent path** going forward: `episodic_only`.
- Keep `full_history` and `no_memory` as **baselines** whenever reporting Stage A claims; do not drop them from the eval harness.
- Dual-layer facts + drill-down: deferred (see plan Stage A RQ3).
- Cognition / paired regression: plan **Phase 3**, only after Stage A acceptance criteria you care about are met.
