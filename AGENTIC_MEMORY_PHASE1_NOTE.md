# Agentic Memory — Phase 1 Note

**Status:** complete  
**Depends on:** Phase 0 (`AGENTIC_MEMORY_PHASE0_NOTE.md`)  
**Scope:** explicit episodic facts only (no cognition / no regression)

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

## Next (Phase 2)

Paired regression harness with handwritten good vs toxic cognition; common-prefix fork via fresh envs from the same trajectory JSON.
