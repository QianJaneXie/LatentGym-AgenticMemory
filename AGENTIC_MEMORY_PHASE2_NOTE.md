# Agentic Memory — Phase / Pilot 2 Note

**Status:** proxy + LLM skill done; Mem0-style flat extraction added  
**Depends on:** Pilot 1 complete (`AGENTIC_MEMORY_PHASE1_NOTE.md`)  
**Focus latent:** `range_100`, 7 episodes, GPT-5.6, one trajectory (`traj_000`)

---

## Hermes-style skill: two adaptations

| Condition | Who writes the skill | Meaning |
|---|---|---|
| `skill_only` / `facts_plus_skill` | Experimenter **template** | Proxy pattern only |
| `skill_only_llm` / `facts_plus_skill_llm` | **Same task LLM** distills a lesson after each episode from agent-visible outcomes | Closer Hermes adaptation |

Neither is a full Hermes Agent integration. Neither is toxic cognition.

LLM distillation flow:

```text
episode ends
  -> extract visible outcome facts
  -> separate generate() call: "write a short reusable skill from these outcomes"
  -> store distilled skill text
  -> next episode: inject skill only, or dense facts + skill
```

`facts_plus_skill*` uses the **same dense facts as `episodic_only`**.

---

## Mem0-style atomic flat extraction (`atomic_flat_llm`)

This is the baseline we may want to **beat** with dense context–action–outcome facts:

| Axis | This condition | Full Mem0 |
|---|---|---|
| Extraction | LLM writes short flat bullets from visible transcript | Same idea |
| Presentation | **read-all** accumulated notes | Usually query → top-k |
| Ranking / embeddings | None | Deferred until store size hurts |

Flow:

```text
episode ends
  -> separate generate(): extract 1-6 short flat facts from visible turns + end feedback
  -> append (light dedupe) to flat_memories
  -> next episode: inject all flat memories (no dense CAO block)
```

Extracted notes are stored in `memory.flat_memories` / `flat_memory_history` (not as `EpisodicFact` IDs).

---

## How to run Pilot 2 extras

```bash
python experiments/memory/run_baselines.py \
  --model llmcenter:gpt-5.6-sol \
  --env number_guessing --latent range_100 \
  --prompt full_info --feedback standard \
  --n-trajectories 1 --num-episodes 7 \
  --trajectory-dir latentgym/data/eval/ \
  --output latentgym/results/memory_phase1_gpt56_range100_standard/ \
  --conditions skill_only_llm facts_plus_skill_llm atomic_flat_llm \
  --merge-existing
```

---

## Results (`range_100`, `traj_000`)

### Proxy template (earlier)

| Condition | reward | turns |
|---|---|---|
| skill_only | 6.080 | `[9, 8, 7, 3, 5, 7, 7]` |
| facts_plus_skill | 6.060 | `[9, 8, 7, 7, 5, 6, 5]` |
| episodic_only (ref) | **6.140** | `[9, 8, 6, 6, 5, 4, 5]` |

### LLM-distilled skill

| Condition | reward | turns |
|---|---|---|
| skill_only_llm | 5.880 | `[9, 8, 6, 9, 9, 10, 5]` |
| facts_plus_skill_llm | 6.040 | `[9, 8, 6, 6, 6, 7, 6]` |

### Atomic flat LLM (Mem0-style extract, read-all)

| Condition | reward | turns |
|---|---|---|
| atomic_flat_llm | 6.000 | `[9, 8, 11, 6, 7, 4, 5]` |
| episodic_only (ref) | **6.140** | `[9, 8, 6, 6, 5, 4, 5]` |
| outcome_only (ref) | 6.020 | `[9, 8, 6, 7, 7, 7, 5]` |
| no_memory (ref) | 5.700 | `[9, 10, 10, 8, 10, 8, 10]` |

On this single seed, flat LLM notes beat `no_memory` but are **slightly worse** than dense context–action–outcome (`episodic_only`) and roughly match / slightly under `outcome_only`. Qualitatively, notes lose episode binding (e.g. “the number was less than 669” without which episode), so past targets can blur across games. Full Mem0 top-k retrieval is still deferred.

On this single seed, LLM-distilled skill **alone** is weaker than the proxy template and far below dense facts (`episodic_only` 6.14). Facts + LLM skill also does not beat dense facts alone. Distilled texts live in `memory.distilled_skill_history`; flat notes in `memory.flat_memories`.
