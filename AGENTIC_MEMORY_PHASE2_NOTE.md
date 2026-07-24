# Agentic Memory — Pilot 2 Note

**Status:** complete on `traj_000` (proxy skill, LLM skill, Mem0-style flat extract)  
**Depends on:** Pilot 1 complete (`AGENTIC_MEMORY_PHASE1_NOTE.md`)  
**Focus latent:** `range_100`, 7 episodes, GPT-5.6, one trajectory (`traj_000`)

> **Naming:** This note is **Pilot 2** (plan: “compare factual representations and experience paradigms”).  
> The main plan’s engineering **Phase 2** is now **fact reconciliation** (Pilot 3). Do not confuse the two.

---

## Hermes-pattern skill: two adaptations

| Condition | Who writes the skill | Meaning |
|---|---|---|
| `skill_only` / `facts_plus_skill` | Experimenter **template** | Proxy pattern only |
| `skill_only_llm` / `facts_plus_skill_llm` | **Same task agent conversation** writes/revises a skill after each episode | Hermes-pattern adaptation (inline) |

Neither is a full Hermes Agent integration (no `SKILL.md` / `skill_manage`).

**Superseded (blind distill):** An earlier Pilot 2 implementation used a separate `generate()` that saw only verified outcomes (no game rules). That blind arm produced soft-toxicity lessons and the numbers/explorer below. **Current code no longer does that** — skill write appends to the live task conversation (rules + play), then compacted modes rebuild context for the next episode. Do not treat the historical `skill_only_llm` reward table as measuring the current condition.

LLM distillation flow (current):

```text
episode ends (conversation still holds rules + this episode's play)
  -> append user turn: "write a short reusable skill from this conversation"
  -> generate() on that same conversation
  -> store distilled skill text
  -> next episode: rebuild compacted context; inject skill only, or dense facts + skill
```

`facts_plus_skill*` uses the **same dense facts as `episodic_only`**.

Explorer (updated for inline distill; includes historical blind quotes for contrast):  
`latentgym/results/memory_phase1_gpt56_range100_standard/skill_only_llm/hermes_soft_toxicity_explorer.html`

---

## Mem0-style atomic flat extraction (`atomic_flat_llm`)

Baseline we may want to **beat** with dense context–action–outcome facts:

| Axis | This condition | Full Mem0 |
|---|---|---|
| Extraction | LLM writes short flat bullets from visible transcript | Same idea |
| Presentation | **read-all** accumulated notes | Usually query → top-k |
| Ranking / embeddings | None | Deferred until store size hurts |

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

### Proxy template

| Condition | reward | turns |
|---|---|---|
| skill_only | 6.080 | `[9, 8, 7, 3, 5, 7, 7]` |
| facts_plus_skill | 6.060 | `[9, 8, 7, 7, 5, 6, 5]` |
| episodic_only (ref) | **6.140** | `[9, 8, 6, 6, 5, 4, 5]` |

### LLM-distilled skill (historical — blind outcomes-only distill; superseded)

| Condition | reward | turns |
|---|---|---|
| skill_only_llm | 5.880 | `[9, 8, 6, 9, 9, 10, 5]` |
| facts_plus_skill_llm | 6.040 | `[9, 8, 6, 6, 6, 7, 6]` |

These scores reflect the old blind distill, not inline same-conversation skill write. Re-run before comparing to other conditions under current code.

### Atomic flat LLM (Mem0-style extract, read-all)

| Condition | reward | turns |
|---|---|---|
| atomic_flat_llm | 6.000 | `[9, 8, 11, 6, 7, 4, 5]` |
| episodic_only (ref) | **6.140** | `[9, 8, 6, 6, 5, 4, 5]` |
| outcome_only (ref) | 6.020 | `[9, 8, 6, 7, 7, 7, 5]` |
| no_memory (ref) | 5.700 | `[9, 10, 10, 8, 10, 8, 10]` |

On this single seed (with historical blind skill distill): dense CAO > flat LLM notes ≈ outcome-only ≫ no memory; blind skill-only LLM was the weakest experience arm. Distilled texts: `memory.distilled_skill_history`. Flat notes: `memory.flat_memories`.

---

## Next → Pilot 3 / plan Phase 2 (reconciliation)

Plan now puts **fact reconciliation** before cognition/RL:

- `FactClaim`, `FactRelation`, rebuildable `CurrentFactView`
- deterministic identity / duplicate / conflict / correction / supersession
- controlled conflict & drift cases; optional unresolved-case drill-down

See plan §8 (reconciliation), Pilot 3 list, and engineering Phase 2.  
**Not started** in code yet. Phase 0 note needs no content change for this.
