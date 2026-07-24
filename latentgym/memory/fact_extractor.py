"""Deterministic fact extraction from agent-visible Number Guessing transcripts.

Never reads evaluator-only fields (episode_configs, latent set_values, raw env info).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

from latentgym.memory.types import EpisodicFact, validate_fact_constraints

_GUESS_BRACKET = re.compile(r"\[(\d+)\]")
_GUESS_BARE = re.compile(r"\d+")
_GREATER = re.compile(r"The number is greater than (\d+)", re.IGNORECASE)
_LESS = re.compile(r"The number is less than (\d+)", re.IGNORECASE)
_CORRECT = re.compile(
    r"Correct! You guessed the number (\d+) in (\d+) turns?",
    re.IGNORECASE,
)
_REVEALED = re.compile(r"The number was (\d+)\.", re.IGNORECASE)
_SCORE = re.compile(r"Score:\s*([0-9.]+)", re.IGNORECASE)
_OUT_OF_GUESSES = re.compile(r"run out of guesses", re.IGNORECASE)
_INVALID = re.compile(r"Invalid format", re.IGNORECASE)
_OUT_OF_RANGE = re.compile(r"Your guess must be between", re.IGNORECASE)


@dataclass
class VisibleTurn:
    """One agent-visible decision within an episode."""

    decision_idx: int
    action: str
    feedback: str
    assistant_message_index: Optional[int] = None
    user_message_index: Optional[int] = None


def parse_guess(action: str) -> Optional[int]:
    bracketed = _GUESS_BRACKET.findall(action)
    if bracketed:
        return int(bracketed[-1])
    bare = _GUESS_BARE.findall(action)
    if bare:
        return int(bare[-1])
    return None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _make_fact(
    *,
    fact_id: str,
    trajectory_id: str,
    episode_idx: int,
    decision_idx: Optional[int],
    context: Dict[str, Any],
    action: Optional[str],
    outcome: str,
    source_type: str,
    source_ref: Dict[str, Any],
    verified: bool,
) -> EpisodicFact:
    fact = EpisodicFact(
        fact_id=fact_id,
        trajectory_id=trajectory_id,
        episode_idx=episode_idx,
        decision_idx=decision_idx,
        context=context,
        action=action,
        outcome=outcome,
        source_type=source_type,
        source_ref=source_ref,
        verified=verified,
        created_at=_now_iso(),
    )
    validate_fact_constraints(fact)
    return fact


def extract_number_guessing_facts(
    *,
    trajectory_id: str,
    episode_idx: int,
    turns: Sequence[VisibleTurn],
    end_feedback: str = "",
    environment: str = "number_guessing",
    id_prefix: Optional[str] = None,
) -> List[EpisodicFact]:
    """Extract append-only facts from agent-visible turns + episode-end text.

    Args:
        trajectory_id: Stable id for the multi-episode trajectory.
        episode_idx: Zero-based episode index that just completed.
        turns: Ordered visible (action, feedback) pairs for that episode only.
            Feedback strings must be the agent-visible step text, not raw env info.
        end_feedback: Agent-visible episode-end text (score / optional reveal).
        environment: Environment name stored in fact context.
        id_prefix: Optional prefix for fact ids (defaults to trajectory/episode).
    """
    prefix = id_prefix or f"{trajectory_id}_e{episode_idx}"
    facts: List[EpisodicFact] = []
    solved = False
    revealed_target: Optional[int] = None
    turns_used = len(turns)

    for turn in turns:
        guess = parse_guess(turn.action)
        action_str = str(guess) if guess is not None else None
        feedback = turn.feedback or ""

        if _INVALID.search(feedback):
            outcome = "invalid guess format; no comparison returned"
        elif _OUT_OF_RANGE.search(feedback) and not _GREATER.search(feedback) and not _LESS.search(feedback):
            outcome = "guess rejected as out of range"
        elif (m := _CORRECT.search(feedback)):
            solved = True
            revealed_target = int(m.group(1))
            turns_used = int(m.group(2))
            outcome = f"correct; target revealed as {revealed_target}; turns={turns_used}"
        elif (m := _GREATER.search(feedback)):
            outcome = f"target was greater than {m.group(1)}"
            if _OUT_OF_GUESSES.search(feedback):
                outcome += "; episode ended: out of guesses"
        elif (m := _LESS.search(feedback)):
            outcome = f"target was less than {m.group(1)}"
            if _OUT_OF_GUESSES.search(feedback):
                outcome += "; episode ended: out of guesses"
        elif _OUT_OF_GUESSES.search(feedback):
            outcome = "episode ended: out of guesses"
        else:
            # Keep a short literal snippet; do not invent interpretation.
            snippet = " ".join(feedback.split())[:160]
            outcome = f"environment feedback: {snippet}" if snippet else "environment feedback recorded"

        decision_type = "first_guess" if turn.decision_idx == 0 else "guess"
        facts.append(
            _make_fact(
                fact_id=f"{prefix}_d{turn.decision_idx}",
                trajectory_id=trajectory_id,
                episode_idx=episode_idx,
                decision_idx=turn.decision_idx,
                context={
                    "environment": environment,
                    "latent_session": trajectory_id,
                    "decision_type": decision_type,
                },
                action=action_str,
                outcome=outcome,
                source_type="environment_feedback",
                source_ref={
                    "assistant_message_index": turn.assistant_message_index,
                    "user_message_index": turn.user_message_index,
                    "decision_idx": turn.decision_idx,
                },
                verified=True,
            )
        )

    # Episode-end feedback (agent-visible only)
    end_text = end_feedback or ""
    if end_text:
        if (m := _REVEALED.search(end_text)):
            revealed_target = int(m.group(1))
        score_match = _SCORE.search(end_text)
        score = score_match.group(1) if score_match else None
        if "didn't find the number" in end_text.lower():
            solved = False
        elif "finished" in end_text.lower() and score is not None:
            try:
                solved = float(score) > 0
            except ValueError:
                pass

        parts = [f"solved={solved}"]
        if score is not None:
            parts.append(f"score={score}")
        parts.append(f"turns={turns_used}")
        if revealed_target is not None:
            parts.append(f"target revealed as {revealed_target}")
        outcome = "; ".join(parts)

        facts.append(
            _make_fact(
                fact_id=f"{prefix}_outcome",
                trajectory_id=trajectory_id,
                episode_idx=episode_idx,
                decision_idx=None,
                context={
                    "environment": environment,
                    "latent_session": trajectory_id,
                    "decision_type": "episode_outcome",
                },
                action=None,
                outcome=outcome,
                source_type="environment_feedback",
                source_ref={"kind": "episode_end_feedback", "text_len": len(end_text)},
                verified=True,
            )
        )

    return facts


def split_boundary_user_message(content: str) -> Dict[str, str]:
    """Split a MultiEpisodeEnv boundary user message into parts.

    Non-final episode boundaries pack:
      step_feedback + end_feedback + transition + next_obs
    into one user message. We recover pieces with conservative heuristics.
    """
    next_obs = ""
    # Number Guessing next-episode cue
    marker = "I'm thinking of a number between"
    idx = content.rfind(marker)
    if idx >= 0:
        # Include any preceding "--- Game/Episode ..." header on the same block.
        header_idx = content.rfind("---", 0, idx)
        start = header_idx if header_idx >= 0 else idx
        next_obs = content[start:].strip()
        prior = content[:start].strip()
    else:
        # Bandits / generic: next episode often starts at the last "--- Game N of M ---"
        game_headers = [
            m.start()
            for m in re.finditer(r"(?m)^--- Game \d+ of \d+ ---", content)
        ]
        if len(game_headers) >= 1 and (
            "finished" in content.lower() or "complete!" in content.lower()
        ):
            # Prefer a header that appears after an episode-finished line.
            start = game_headers[-1]
            finished_idx = content.lower().rfind("finished")
            if finished_idx >= 0 and start > finished_idx:
                next_obs = content[start:].strip()
                prior = content[:start].strip()
            else:
                prior = content.strip()
        else:
            prior = content.strip()

    end_feedback = ""
    for line in prior.splitlines():
        if line.strip().lower().startswith("episode ") and "finished" in line.lower():
            end_feedback = line.strip()
            break
    if not end_feedback and prior:
        # Fallback: last non-empty paragraph before next_obs
        paras = [p.strip() for p in prior.split("\n\n") if p.strip()]
        if paras:
            end_feedback = paras[-1]

    # Step feedback is everything before the episode-finished line when present.
    step_feedback = prior
    if end_feedback and end_feedback in prior:
        step_feedback = prior[: prior.rfind(end_feedback)].strip()

    return {
        "step_feedback": step_feedback,
        "end_feedback": end_feedback,
        "next_obs": next_obs,
        "prior": prior,
    }
