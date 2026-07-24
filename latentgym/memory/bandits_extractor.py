"""Deterministic fact extraction for Bandits (agent-visible transcript only)."""
from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

from latentgym.memory.fact_extractor import VisibleTurn
from latentgym.memory.types import EpisodicFact, validate_fact_constraints

_BUTTON = re.compile(r"\[(?:select\s+)?(\w+)\]", re.IGNORECASE)
_REWARD = re.compile(r"Reward:\s*([01])", re.IGNORECASE)
_SELECTED = re.compile(r"You selected '(\w+)'", re.IGNORECASE)
_CORRECT = re.compile(r"correct!", re.IGNORECASE)
_WRONG = re.compile(r"wrong", re.IGNORECASE)
_BEST = re.compile(r"The best was '(\w+)'", re.IGNORECASE)
_SCORE = re.compile(r"Score:\s*([0-9.]+)", re.IGNORECASE)
_PROBS = re.compile(r"Probabilities:\s*(.+)$", re.IGNORECASE)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_button_action(action: str) -> tuple[Optional[str], bool]:
    """Return (button, is_select) from an agent action string."""
    select = re.search(r"\[select\s+(\w+)\]", action or "", re.IGNORECASE)
    if select:
        return select.group(1).lower(), True
    m = _BUTTON.search(action or "")
    if m:
        return m.group(1).lower(), False
    return None, False


def _make_fact(
    *,
    fact_id: str,
    trajectory_id: str,
    episode_idx: int,
    decision_idx: Optional[int],
    context: Dict[str, Any],
    action: Optional[str],
    outcome: str,
    source_ref: Dict[str, Any],
) -> EpisodicFact:
    fact = EpisodicFact(
        fact_id=fact_id,
        trajectory_id=trajectory_id,
        episode_idx=episode_idx,
        decision_idx=decision_idx,
        context=context,
        action=action,
        outcome=outcome,
        source_type="environment_feedback",
        source_ref=source_ref,
        verified=True,
        created_at=_now_iso(),
    )
    validate_fact_constraints(fact)
    return fact


def extract_bandits_facts(
    *,
    trajectory_id: str,
    episode_idx: int,
    turns: Sequence[VisibleTurn],
    end_feedback: str = "",
    environment: str = "bandits",
    id_prefix: Optional[str] = None,
) -> List[EpisodicFact]:
    """Extract append-only explore/select/outcome facts from visible bandit turns."""
    prefix = id_prefix or f"{trajectory_id}_e{episode_idx}"
    facts: List[EpisodicFact] = []
    turns_used = len(turns)
    selected: Optional[str] = None
    correct: Optional[bool] = None
    best: Optional[str] = None

    for turn in turns:
        button, is_select = parse_button_action(turn.action)
        action_str = (
            f"select {button}" if is_select and button else (button if button else None)
        )
        feedback = turn.feedback or ""
        if is_select or _SELECTED.search(feedback):
            if button:
                selected = button
            sel_m = _SELECTED.search(feedback)
            if sel_m:
                selected = sel_m.group(1).lower()
            if _CORRECT.search(feedback):
                correct = True
                outcome = f"final select={selected}; result=correct"
            elif _WRONG.search(feedback):
                correct = False
                outcome = f"final select={selected}; result=wrong"
                if "was not" in feedback.lower():
                    outcome += "; env stated selection was not the highest-probability button"
            else:
                outcome = f"final select={selected or '?'}; result=recorded"
            decision_type = "select"
        else:
            rm = _REWARD.search(feedback)
            if rm:
                outcome = f"explore {button}; reward={rm.group(1)}"
            else:
                snippet = " ".join(feedback.split())[:160]
                outcome = f"explore {button}; feedback={snippet}" if button else (
                    f"environment feedback: {snippet}" if snippet else "environment feedback recorded"
                )
            decision_type = "first_explore" if turn.decision_idx == 0 else "explore"

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
                    "button": button,
                },
                action=action_str,
                outcome=outcome,
                source_ref={
                    "assistant_message_index": turn.assistant_message_index,
                    "user_message_index": turn.user_message_index,
                    "decision_idx": turn.decision_idx,
                },
            )
        )

    end_text = end_feedback or ""
    if end_text:
        sel_m = _SELECTED.search(end_text)
        if sel_m:
            selected = sel_m.group(1).lower()
        if _CORRECT.search(end_text):
            correct = True
        elif _WRONG.search(end_text):
            correct = False
        best_m = _BEST.search(end_text)
        if best_m:
            best = best_m.group(1).lower()
        # If correct and no explicit best, selected button is the revealed best.
        if correct and selected and best is None:
            best = selected
        score_m = _SCORE.search(end_text)
        score = score_m.group(1) if score_m else None
        probs_m = _PROBS.search(end_text)
        probs = probs_m.group(1).strip() if probs_m else None

        parts = [f"solved={bool(correct)}"]
        if selected:
            parts.append(f"selected={selected}")
        if correct is not None:
            parts.append(f"correct={correct}")
        if score is not None:
            parts.append(f"score={score}")
        parts.append(f"turns={turns_used}")
        if best:
            parts.append(f"best revealed as {best}")
        if probs:
            parts.append(f"probabilities={probs}")

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
                    "selected": selected,
                    "best": best,
                    "correct": correct,
                },
                action=None,
                outcome="; ".join(parts),
                source_ref={"kind": "episode_end_feedback", "text_len": len(end_text)},
            )
        )

    return facts
