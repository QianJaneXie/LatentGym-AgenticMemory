# Agentic Memory — Phase / Pilot 2 Note

**Status:** core Pilot 2 conditions run on `traj_000`  
**Depends on:** Pilot 1 complete (`AGENTIC_MEMORY_PHASE1_NOTE.md`)  
**Focus latent:** `range_100`, 7 episodes, GPT-5.6, read-all

---

## Goal (plan §8.2 Pilot 2)

Compare factual representations and experience / skill paradigms:

| Item | Condition | Status |
|---|---|---|
| Provenance-grounded event facts | `episodic_only` | Done in Pilot 1 |
| Hermes-style skill only | `skill_only` | Done (`traj_000`) |
| Facts plus the same skill | `facts_plus_skill` | Done (`traj_000`) |
| Atomic flat LLM-extracted facts | TBD | Deferred (needs extractor); not labeled Mem0 |

This is a **LatentGym adaptation** of the Hermes skill pattern (procedural lesson without / with supporting facts), not a full Hermes Agent integration.

---

## Skill format

Deterministic template from agent-visible revealed targets only, e.g.:

```text
Experience / skill note (...):
- Previously revealed targets: [658, 669, ...]
- Observed span so far: [658, 749]
- Suggested procedure: try near min/max of revealed targets, then search inside that span...
```

---

## How to run

```bash
python experiments/memory/run_baselines.py \
  --model llmcenter:gpt-5.6-sol \
  --env number_guessing --latent range_100 \
  --prompt full_info --feedback standard \
  --n-trajectories 1 --num-episodes 7 \
  --trajectory-dir latentgym/data/eval/ \
  --output latentgym/results/memory_phase1_gpt56_range100_standard/ \
  --conditions skill_only facts_plus_skill \
  --merge-existing
```

Compare against Pilot 1 rows in the same `baselines_summary.json`, especially `episodic_only` and `no_memory`.

---

## Results (`range_100`, `traj_000`, merged into Pilot 1 summary)

Dir: `latentgym/results/memory_phase1_gpt56_range100_standard/baselines_summary.json`

| Condition | reward | turns |
|---|---|---|
| no_memory | 5.700 | `[9, 10, 10, 8, 10, 8, 10]` |
| full_history | 6.120 | `[9, 8, 6, 7, 5, 4, 5]` |
| episodic_only | **6.140** | `[9, 8, 6, 6, 5, 4, 5]` |
| skill_only | 6.080 | `[9, 8, 7, 3, 5, 7, 7]` |
| facts_plus_skill | 6.060 | `[9, 8, 7, 7, 5, 6, 5]` |

On this single seed, the procedural skill alone helps vs no memory but does **not** beat dense factual memory; adding the skill on top of facts also does not improve over `episodic_only`. Treat as suggestive only.

Still deferred: atomic LLM flat-fact extraction; faithful Mem0.
