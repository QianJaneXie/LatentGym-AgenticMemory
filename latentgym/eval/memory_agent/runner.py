"""MemoryAPIRunner — Phase 1 memory-aware trajectory runner.

Isolated from latentgym.eval.single_agent.api_runner.APIRunner.
Supports three conditions without cognition:
  - full_history: retain the full conversation (LatentGym default behavior)
  - no_memory: clear cross-episode context at each episode boundary
  - episodic_only: clear cross-episode raw history; inject retrieved facts
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Literal, Optional

from latentgym.core.multi_episode_env import MultiEpisodeEnv
from latentgym.eval.model_interface import ModelInterface
from latentgym.eval.types import EpisodeOutcome, OutcomeType, TrajectoryResult
from latentgym.memory.decision_logger import DecisionLogger
from latentgym.memory.episodic_store import EpisodicStore
from latentgym.memory.fact_extractor import (
    VisibleTurn,
    extract_number_guessing_facts,
    split_boundary_user_message,
)
from latentgym.memory.retriever import format_facts_for_prompt, retrieve_episodic_facts
from latentgym.memory.types import DecisionTrace

logger = logging.getLogger(__name__)

MemoryCondition = Literal["no_memory", "full_history", "episodic_only"]

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
        fact_budget: int = 10,
    ):
        if condition not in ("no_memory", "full_history", "episodic_only"):
            raise ValueError(f"Unknown memory condition: {condition}")
        self.model = model
        self.condition: MemoryCondition = condition
        self.env_name = env_name
        self.fact_budget = fact_budget

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
        if self.condition in ("no_memory", "episodic_only"):
            open_first_decision = self._maybe_inject_memory(
                conversation,
                store=store,
                decision_logger=decision_logger,
                trajectory_id=traj_id,
                episode_idx=0,
                pending_first_decision=True,
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
                new_facts = extract_number_guessing_facts(
                    trajectory_id=traj_id,
                    episode_idx=completed_ep,
                    turns=pending_turns,
                    end_feedback=end_feedback,
                    environment=self.env_name,
                )
                store.extend(new_facts)
                decision_logger.update_known_facts(store.fact_ids())
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
                if not done and self.condition in ("no_memory", "episodic_only"):
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
                    "facts": store.to_list(),
                    "decisions": decision_logger.to_list(),
                    "n_facts": len(store),
                    "n_decisions": len(decision_logger.all_traces()),
                }
            },
        )
        return result

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
    ) -> Optional[Dict[str, Any]]:
        """Inject retrieved facts into the latest user message for episodic_only."""
        decision_logger.update_known_facts(store.fact_ids())
        retrieval = retrieve_episodic_facts(
            store,
            trajectory_id=trajectory_id,
            episode_idx=episode_idx,
            decision_type="first_guess",
            environment=self.env_name,
            budget=self.fact_budget,
        )

        if self.condition == "episodic_only" and retrieval.facts:
            block = format_facts_for_prompt(retrieval.facts)
            # Prepend to the latest user message (episode start).
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
            "loaded_fact_ids": list(retrieval.fact_ids) if self.condition == "episodic_only" else [],
            "query": retrieval.query,
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
