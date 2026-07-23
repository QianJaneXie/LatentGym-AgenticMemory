# Agentic Memory — Phase / Pilot 2 Note

**Status:** proxy skill done; LLM-distilled Hermes-pattern skill added  
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

## Why atomic flat facts is lower priority right now

Atomic flat facts = another LLM extracts short unstructured notes from the transcript (representation ablation, **not** Mem0).

Deferred because:

1. Pilot 1 already compared outcome-only vs dense context–action–outcome vs oracle — that answers “what factual body helps” without a second extractor.
2. On `range_100`, dense facts already ≈ full history; a flatter LLM extract is unlikely to change the main story before skill/cognition questions.
3. It adds API cost and another failure mode (bad extraction) without addressing Hermes-style experience vs facts.
4. Mem0 (query retrieval) remains separately deferred until scale matters.

Revisit atomic flat facts only if we need a representation ablation against Mem0-like extraction, or dense facts become too noisy.

---

## How to run LLM-distilled skills

```bash
python experiments/memory/run_baselines.py \
  --model llmcenter:gpt-5.6-sol \
  --env number_guessing --latent range_100 \
  --prompt full_info --feedback standard \
  --n-trajectories 1 --num-episodes 7 \
  --trajectory-dir latentgym/data/eval/ \
  --output latentgym/results/memory_phase1_gpt56_range100_standard/ \
  --conditions skill_only_llm facts_plus_skill_llm \
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

### LLM-distilled

| Condition | reward | turns |
|---|---|---|
| skill_only_llm | 5.880 | `[9, 8, 6, 9, 9, 10, 5]` |
| facts_plus_skill_llm | 6.040 | `[9, 8, 6, 6, 6, 7, 6]` |

On this single seed, LLM-distilled skill **alone** is weaker than the proxy template and far below dense facts (`episodic_only` 6.14). Facts + LLM skill also does not beat dense facts alone. Distilled texts live in `memory.distilled_skill_history`.
