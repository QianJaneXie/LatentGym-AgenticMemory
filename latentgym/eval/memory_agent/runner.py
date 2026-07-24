"""MemoryAPIRunner — Phase 1 / Stage A0 memory-aware trajectory runner.

Isolated from latentgym.eval.single_agent.api_runner.APIRunner.
Pilot 1 conditions without cognition:
  - full_history: retain the full conversation (LatentGym default behavior)
  - no_memory: clear cross-episode context at each episode boundary
  - episodic_only: clear raw history; inject all prior facts (dense context-action-outcome)
  - outcome_only: clear raw history; inject episode-outcome facts only
  - oracle_summary: clear raw history; inject compact restatement of visible outcomes
  - skill_only: clear raw history; inject proxy (template) skill only
  - facts_plus_skill: clear raw history; inject dense facts plus proxy skill
  - skill_only_llm: clear raw history; inject skill distilled in the same task conversation
  - facts_plus_skill_llm: clear raw history; inject dense facts plus same-conversation skill
  - atomic_flat_llm: clear raw history; inject Mem0-style flat LLM memories (read-all)
  - reconciled_view: clear raw history; inject deterministic CurrentFactView (Bandits MVP)
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Literal, Optional

from latentgym.core.multi_episode_env import MultiEpisodeEnv
from latentgym.eval.model_interface import ModelInterface
from latentgym.eval.types import EpisodeOutcome, OutcomeType, TrajectoryResult
from latentgym.memory.bandits_extractor import extract_bandits_facts
from latentgym.memory.decision_logger import DecisionLogger
from latentgym.memory.episodic_store import EpisodicStore
from latentgym.memory.fact_extractor import (
    VisibleTurn,
    extract_number_guessing_facts,
    split_boundary_user_message,
)
from latentgym.memory.reconcile import build_bandits_current_view
from latentgym.memory.retriever import (
    build_atomic_flat_extraction_prompt,
    build_inline_skill_distillation_prompt,
    format_atomic_flat_memories,
    format_facts_for_prompt,
    format_oracle_summary_from_facts,
    format_skill_from_facts,
    parse_atomic_flat_bullets,
    retrieve_episodic_facts,
    select_outcome_only_facts,
    wrap_distilled_skill,
)
from latentgym.memory.types import DecisionTrace

logger = logging.getLogger(__name__)

MemoryCondition = Literal[
    "no_memory",
    "full_history",
    "episodic_only",
    "outcome_only",
    "oracle_summary",
    "skill_only",
    "facts_plus_skill",
    "skill_only_llm",
    "facts_plus_skill_llm",
    "atomic_flat_llm",
    "reconciled_view",
]
_COMPACTED_CONDITIONS = (
    "no_memory",
    "episodic_only",
    "outcome_only",
    "oracle_summary",
    "skill_only",
    "facts_plus_skill",
    "skill_only_llm",
    "facts_plus_skill_llm",
    "atomic_flat_llm",
    "reconciled_view",
)
_FACT_INJECT_CONDITIONS = (
    "episodic_only",
    "outcome_only",
    "oracle_summary",
    "skill_only",
    "facts_plus_skill",
    "skill_only_llm",
    "facts_plus_skill_llm",
    "atomic_flat_llm",
    "reconciled_view",
)
_LLM_SKILL_CONDITIONS = ("skill_only_llm", "facts_plus_skill_llm")

def _split_init_system(content: str) -> tuple[str, str]:
    """Split init system message into stable rules vs episode-1 header/obs."""
    for marker in ("\n\n--- Episode ", "\n\n--- Game "):
        idx = content.find(marker)
        if idx >= 0:
            return content[:idx].strip(), content[idx:].strip()
    return content.strip(), ""


def _assert_no_ground_truth_leakage(messages: List[Dict[str, str]]) -> None:
    """Best-effort guard: prompts must not embed evaluator config dumps."""
    joined = "\n".join(m.get("content", "") for m in messages)
    # Structured leakage markers (JSON-ish), not natural feedback like "target revealed".
    if re.search(r"['\"]target_number['\"]\s*:", joined):
        raise RuntimeError("Ground-truth leakage: target_number key found in agent messages")
    if re.search(r"['\"]set_values['\"]\s*:", joined):
        raise RuntimeError("Ground-truth leakage: set_values key found in agent messages")
    if re.search(r"['\"]episode_configs['\"]\s*:", joined):
        raise RuntimeError("Ground-truth leakage: episode_configs key found in agent messages")


class MemoryAPIRunner:
    """Run a model with explicit memory-condition control."""

    def __init__(
        self,
        model: ModelInterface,
        *,
        condition: MemoryCondition = "episodic_only",
        env_name: str = "number_guessing",
        fact_budget: Optional[int] = None,
    ):
        if condition not in (
            "no_memory",
            "full_history",
            "episodic_only",
            "outcome_only",
            "oracle_summary",
            "skill_only",
            "facts_plus_skill",
            "skill_only_llm",
            "facts_plus_skill_llm",
            "atomic_flat_llm",
            "reconciled_view",
        ):
            raise ValueError(f"Unknown memory condition: {condition}")
        self.model = model
        self.condition: MemoryCondition = condition
        self.env_name = env_name
        # None = Stage A0 read-all (plan default). Positive int = later budget sweep.
        self.fact_budget = fact_budget

    def _extract_env_facts(
        self,
        *,
        trajectory_id: str,
        episode_idx: int,
        turns: List[VisibleTurn],
        end_feedback: str,
    ):
        if self.env_name == "bandits":
            return extract_bandits_facts(
                trajectory_id=trajectory_id,
                episode_idx=episode_idx,
                turns=turns,
                end_feedback=end_feedback,
                environment=self.env_name,
            )
        if self.env_name == "number_guessing":
            return extract_number_guessing_facts(
                trajectory_id=trajectory_id,
                episode_idx=episode_idx,
                turns=turns,
                end_feedback=end_feedback,
                environment=self.env_name,
            )
        raise ValueError(f"No fact extractor for env_name={self.env_name!r}")

    async def run_trajectory(
        self,
        env: MultiEpisodeEnv,
        seed: int = 0,
        max_total_turns: int = 500,
        success_threshold: float = 0.01,
        trajectory_id: Optional[str] = None,
    ) -> TrajectoryResult:
        traj_id = trajectory_id or f"traj_{seed:04d}"
        store = EpisodicStore()
        decision_logger = DecisionLogger()
        distilled_skill = ""
        distilled_skill_history: List[Dict[str, Any]] = []
        flat_memories: List[str] = []
        flat_memory_history: List[Dict[str, Any]] = []
        reconciled_views: List[Dict[str, Any]] = []

        conversation, init_metadata = env.init([])
        # Evaluator-only: keep for TrajectoryResult / EpisodeOutcome, never inject.
        episode_configs = init_metadata.get("episode_configs", [])
        latent_id = init_metadata.get("latent_id", "")
        reward_type = init_metadata.get("reward_type", "")
        max_turns_per_ep = init_metadata.get("max_turns_per_episode", 0)
        env_params = init_metadata.get("env_params", {})

        if not conversation or conversation[0].get("role") != "system":
            raise RuntimeError("Expected env.init to return a system message first")

        rules, ep0_tail = _split_init_system(conversation[0]["content"])
        # For compacted modes, rebuild ep0 so memory can be injected cleanly.
        # Per-episode visible turns awaiting fact extraction
        pending_turns: List[VisibleTurn] = []
        # First-decision bookkeeping for DecisionTrace completion
        open_first_decision: Optional[Dict[str, Any]] = None

        # Always put episode-0 content in a user message. Some gateways (e.g.
        # MiniCPM via LLMCenter/vLLM) reject conversations that are system-only
        # ("No user query found in messages"). full_history still accumulates
        # across episodes; only the init packaging changes.
        conversation = [{"role": "system", "content": rules}]
        ep0_user = ep0_tail or "Begin episode 1."
        conversation.append({"role": "user", "content": ep0_user})
        if self.condition in _COMPACTED_CONDITIONS:
            open_first_decision = self._maybe_inject_memory(
                conversation,
                store=store,
                decision_logger=decision_logger,
                trajectory_id=traj_id,
                episode_idx=0,
                pending_first_decision=True,
                distilled_skill=distilled_skill,
                flat_memories=flat_memories,
                reconciled_views=reconciled_views,
            )
        else:
            self._log_retrieval_only(
                store=store,
                decision_logger=decision_logger,
                trajectory_id=traj_id,
                episode_idx=0,
            )
            open_first_decision = {
                "episode_idx": 0,
                "loaded_fact_ids": [],
                "query": "full_history episode=0 first_guess",
                "action": "",
            }

        episode_outcomes: List[EpisodeOutcome] = []
        reasoning_trace: List[Optional[str]] = []
        current_episode = 0
        episode_turn = 0
        final_metadata: Dict[str, Any] = {}

        done = False
        while not done and (episode_turn + sum(o.turns for o in episode_outcomes)) < max_total_turns:
            _assert_no_ground_truth_leakage(conversation)

            response = await self.model.generate(conversation)
            episode_turn += 1
            reasoning_trace.append(response.reasoning)
            conversation.append({"role": "assistant", "content": response.text})
            assistant_idx = len(conversation) - 1

            # Capture open first-decision action
            if episode_turn == 1 and open_first_decision is not None:
                open_first_decision["action"] = response.text
            elif episode_turn == 1 and open_first_decision is None:
                # full_history path: create a minimal first-decision shell
                open_first_decision = {
                    "episode_idx": current_episode,
                    "loaded_fact_ids": [],
                    "query": f"full_history episode={current_episode} first_guess",
                    "action": response.text,
                }

            step_result = env.step(response.text)
            obs = step_result["observations"]
            done = step_result["done"]
            step_metadata = step_result["metadata"]

            user_content = ""
            user_idx: Optional[int] = None
            if obs:
                conversation.extend(obs)
                user_idx = len(conversation) - 1
                user_content = obs[0].get("content", "") if obs else ""

            # Episode-boundary signal (same as APIRunner; see AGENTIC_MEMORY_PHASE0_NOTE.md)
            new_episode = step_metadata.get("episode", 0)
            ep_rewards = step_metadata.get("episode_rewards", [])
            ep_turns = step_metadata.get("turns_per_episode", [])
            boundary = new_episode != current_episode or done

            if boundary:
                parts = split_boundary_user_message(user_content) if user_content else {
                    "step_feedback": "",
                    "end_feedback": "",
                    "next_obs": "",
                }
                step_feedback = parts["step_feedback"] or user_content
                end_feedback = parts["end_feedback"]
                next_obs = parts["next_obs"]

                pending_turns.append(
                    VisibleTurn(
                        decision_idx=episode_turn - 1,
                        action=response.text,
                        feedback=step_feedback,
                        assistant_message_index=assistant_idx,
                        user_message_index=user_idx,
                    )
                )

                # Complete first-decision outcome if present
                if open_first_decision is not None:
                    self._finalize_first_decision(
                        open_first_decision,
                        decision_logger=decision_logger,
                        trajectory_id=traj_id,
                        outcome=step_feedback,
                        reward=ep_rewards[-1] if ep_rewards else None,
                    )
                    open_first_decision = None

                completed_ep = len(episode_outcomes)
                new_facts = self._extract_env_facts(
                    trajectory_id=traj_id,
                    episode_idx=completed_ep,
                    turns=pending_turns,
                    end_feedback=end_feedback,
                )
                store.extend(new_facts)
                decision_logger.update_known_facts(store.fact_ids())

                if self.condition in _LLM_SKILL_CONDITIONS:
                    distilled_skill = await self._distill_skill_from_conversation(
                        conversation
                    )
                    distilled_skill_history.append(
                        {
                            "after_episode_idx": completed_ep,
                            "skill_text": distilled_skill,
                        }
                    )

                if self.condition == "atomic_flat_llm":
                    extracted = await self._extract_atomic_flat_memories(
                        episode_idx=completed_ep,
                        turns=pending_turns,
                        end_feedback=end_feedback,
                    )
                    if extracted:
                        existing = {m.lower() for m in flat_memories}
                        added = [m for m in extracted if m.lower() not in existing]
                        flat_memories.extend(added)
                        flat_memory_history.append(
                            {
                                "after_episode_idx": completed_ep,
                                "extracted": extracted,
                                "added": added,
                                "n_total": len(flat_memories),
                            }
                        )

                pending_turns = []

                while len(episode_outcomes) < len(ep_rewards):
                    idx = len(episode_outcomes)
                    ep_reward = ep_rewards[idx]
                    ep_turn_count = ep_turns[idx] if idx < len(ep_turns) else episode_turn
                    if ep_reward >= success_threshold:
                        outcome_type = OutcomeType.WIN
                    elif ep_turn_count >= max_turns_per_ep > 0:
                        outcome_type = OutcomeType.TIMEOUT
                    elif ep_reward > 0:
                        outcome_type = OutcomeType.PARTIAL
                    else:
                        outcome_type = OutcomeType.LOSS

                    gt = episode_configs[idx] if idx < len(episode_configs) else {}
                    episode_outcomes.append(
                        EpisodeOutcome(
                            episode_idx=idx,
                            reward=ep_reward,
                            turns=ep_turn_count,
                            success=ep_reward >= success_threshold,
                            agent_name=self.model.name,
                            latent_id=step_metadata.get("latent_id", latent_id),
                            max_turns=max_turns_per_ep,
                            outcome_type=outcome_type,
                            ground_truth=gt,
                        )
                    )

                current_episode = new_episode
                episode_turn = 0

                # Rebuild context for compacted modes before the next first guess.
                if not done and self.condition in _COMPACTED_CONDITIONS:
                    conversation = [{"role": "system", "content": rules}]
                    start_user = next_obs or f"--- Episode {current_episode + 1} ---"
                    conversation.append({"role": "user", "content": start_user})
                    open_first_decision = self._maybe_inject_memory(
                        conversation,
                        store=store,
                        decision_logger=decision_logger,
                        trajectory_id=traj_id,
                        episode_idx=current_episode,
                        pending_first_decision=True,
                        distilled_skill=distilled_skill,
                        flat_memories=flat_memories,
                        reconciled_views=reconciled_views,
                    )
                elif not done and self.condition == "full_history":
                    open_first_decision = {
                        "episode_idx": current_episode,
                        "loaded_fact_ids": [],
                        "query": f"full_history episode={current_episode} first_guess",
                        "action": "",
                    }
            else:
                # Mid-episode observation
                pending_turns.append(
                    VisibleTurn(
                        decision_idx=episode_turn - 1,
                        action=response.text,
                        feedback=user_content,
                        assistant_message_index=assistant_idx,
                        user_message_index=user_idx,
                    )
                )
                if open_first_decision is not None and episode_turn == 1:
                    # Defer finalize until we know episode reward; store interim feedback
                    open_first_decision["outcome"] = user_content

            final_metadata = step_metadata

        env.close()

        result = TrajectoryResult(
            episode_outcomes=episode_outcomes,
            conversation=conversation,
            model_name=self.model.name,
            benchmark_id=init_metadata.get("benchmark_id", ""),
            seed=seed,
            env_name=self.env_name,
            latent_id=latent_id,
            prompt_id=init_metadata.get("prompt_id", ""),
            feedback_id=init_metadata.get("feedback_id", ""),
            reward_type=reward_type,
            max_turns_per_episode=max_turns_per_ep,
            env_params=env_params,
            agent_assignments=[self.model.name] * len(episode_outcomes),
            episode_configs=episode_configs,
            reasoning_trace=reasoning_trace,
            init_metadata=init_metadata,
            final_metadata=final_metadata,
            metadata={
                "memory": {
                    "condition": self.condition,
                    "trajectory_id": traj_id,
                    "fact_budget": self.fact_budget,
                    "presentation_mode": (
                        "read_all" if self.fact_budget is None else f"budget_{self.fact_budget}"
                    ),
                    "facts": store.to_list(),
                    "decisions": decision_logger.to_list(),
                    "n_facts": len(store),
                    "n_decisions": len(decision_logger.all_traces()),
                    "distilled_skill": distilled_skill,
                    "distilled_skill_history": distilled_skill_history,
                    "flat_memories": list(flat_memories),
                    "flat_memory_history": flat_memory_history,
                    "reconciled_views": reconciled_views,
                }
            },
        )
        return result

    async def _extract_atomic_flat_memories(
        self,
        *,
        episode_idx: int,
        turns: List[VisibleTurn],
        end_feedback: str,
    ) -> List[str]:
        """Mem0-style flat extraction from one episode's agent-visible transcript."""
        turn_lines = [
            f"guess={t.action.strip()} | feedback={t.feedback.strip()}" for t in turns
        ]
        prompt = build_atomic_flat_extraction_prompt(
            episode_idx=episode_idx,
            turn_lines=turn_lines,
            end_feedback=end_feedback,
            environment=self.env_name,
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "You extract short flat memories from a visible game transcript. "
                    "Never invent hidden latents or evaluator-only information."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        _assert_no_ground_truth_leakage(messages)
        response = await self.model.generate(messages)
        return parse_atomic_flat_bullets(response.text)

    async def _distill_skill_from_conversation(
        self,
        conversation: List[Dict[str, str]],
    ) -> str:
        """Ask the task model to write a skill in the live play conversation (Hermes-pattern).

        Appends a write-skill user turn to the current conversation (rules + play so far),
        then generates. Compacted modes rebuild conversation afterward, so this turn is
        not carried into the next episode.
        """
        prompt = build_inline_skill_distillation_prompt()
        conversation.append({"role": "user", "content": prompt})
        _assert_no_ground_truth_leakage(conversation)
        response = await self.model.generate(conversation)
        return wrap_distilled_skill(response.text)

    def _log_retrieval_only(
        self,
        *,
        store: EpisodicStore,
        decision_logger: DecisionLogger,
        trajectory_id: str,
        episode_idx: int,
    ) -> None:
        # full_history does not inject facts; still record empty retrieval intent.
        decision_logger.update_known_facts(store.fact_ids())
        _ = retrieve_episodic_facts(
            store,
            trajectory_id=trajectory_id,
            episode_idx=episode_idx,
            decision_type="first_guess",
            environment=self.env_name,
            budget=self.fact_budget,
        )

    def _maybe_inject_memory(
        self,
        conversation: List[Dict[str, str]],
        *,
        store: EpisodicStore,
        decision_logger: DecisionLogger,
        trajectory_id: str,
        episode_idx: int,
        pending_first_decision: bool,
        distilled_skill: str = "",
        flat_memories: Optional[List[str]] = None,
        reconciled_views: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Inject memory into the latest user message for compacted fact conditions."""
        decision_logger.update_known_facts(store.fact_ids())
        retrieval = retrieve_episodic_facts(
            store,
            trajectory_id=trajectory_id,
            episode_idx=episode_idx,
            decision_type="first_guess",
            environment=self.env_name,
            budget=self.fact_budget,
        )

        loaded_ids: List[str] = []
        block = ""
        if self.condition == "episodic_only" and retrieval.facts:
            block = format_facts_for_prompt(retrieval.facts)
            loaded_ids = list(retrieval.fact_ids)
        elif self.condition == "outcome_only" and retrieval.facts:
            outcomes = select_outcome_only_facts(retrieval.facts)
            block = format_facts_for_prompt(outcomes)
            loaded_ids = [f.fact_id for f in outcomes]
        elif self.condition == "oracle_summary" and retrieval.facts:
            block = format_oracle_summary_from_facts(retrieval.facts)
            loaded_ids = [f.fact_id for f in select_outcome_only_facts(retrieval.facts)]
        elif self.condition == "skill_only" and retrieval.facts:
            block = format_skill_from_facts(retrieval.facts)
            loaded_ids = [f.fact_id for f in select_outcome_only_facts(retrieval.facts)]
        elif self.condition == "facts_plus_skill" and retrieval.facts:
            facts_block = format_facts_for_prompt(retrieval.facts)
            skill_block = format_skill_from_facts(retrieval.facts)
            block = "\n\n".join(p for p in (facts_block, skill_block) if p)
            loaded_ids = list(retrieval.fact_ids)
        elif self.condition == "skill_only_llm" and distilled_skill:
            block = distilled_skill
            loaded_ids = [f.fact_id for f in select_outcome_only_facts(retrieval.facts)]
        elif self.condition == "facts_plus_skill_llm" and (retrieval.facts or distilled_skill):
            facts_block = format_facts_for_prompt(retrieval.facts) if retrieval.facts else ""
            block = "\n\n".join(p for p in (facts_block, distilled_skill) if p)
            loaded_ids = list(retrieval.fact_ids)
        elif self.condition == "atomic_flat_llm" and flat_memories:
            # Flat notes are not EpisodicFact IDs; keep decision provenance empty.
            block = format_atomic_flat_memories(flat_memories)
            loaded_ids = []
        elif self.condition == "reconciled_view":
            if self.env_name != "bandits":
                raise ValueError("reconciled_view MVP currently supports bandits only")
            view = build_bandits_current_view(
                store.all_facts(),
                trajectory_id=trajectory_id,
                as_of_episode_idx=episode_idx,
            )
            if reconciled_views is not None:
                reconciled_views.append(view.to_dict())
            block = view.render_text
            loaded_ids = list(retrieval.fact_ids)

        if block:
            for i in range(len(conversation) - 1, -1, -1):
                if conversation[i]["role"] == "user":
                    conversation[i] = {
                        "role": "user",
                        "content": block + "\n\n" + conversation[i]["content"],
                    }
                    break

        _assert_no_ground_truth_leakage(conversation)

        if not pending_first_decision:
            return None
        return {
            "episode_idx": episode_idx,
            "loaded_fact_ids": loaded_ids if self.condition in _FACT_INJECT_CONDITIONS else [],
            "query": f"{retrieval.query}; inject={self.condition}",
            "action": "",
            "outcome": "",
        }

    def _finalize_first_decision(
        self,
        shell: Dict[str, Any],
        *,
        decision_logger: DecisionLogger,
        trajectory_id: str,
        outcome: str,
        reward: Optional[float],
    ) -> None:
        ep = int(shell["episode_idx"])
        trace = DecisionTrace(
            decision_id=f"{trajectory_id}_e{ep}_first_guess",
            trajectory_id=trajectory_id,
            episode_idx=ep,
            decision_type="first_guess",
            query=str(shell.get("query", "")),
            loaded_fact_ids=list(shell.get("loaded_fact_ids", [])),
            loaded_cognition_ids=[],
            cited_memory_ids=[],
            action=str(shell.get("action", "")),
            outcome=outcome or str(shell.get("outcome", "")),
            reward=reward,
        )
        decision_logger.log(trace, validate=True)
