"""Bandits fact extraction + deterministic reconciliation tests."""
from latentgym.memory.bandits_extractor import extract_bandits_facts, parse_button_action
from latentgym.memory.fact_extractor import VisibleTurn
from latentgym.memory.reconcile import build_bandits_current_view


def test_parse_button_action():
    assert parse_button_action("I try [red]") == ("red", False)
    assert parse_button_action("[select blue]") == ("blue", True)


def test_extract_bandits_facts_and_supersession():
    turns0 = [
        VisibleTurn(0, "[red]", "You pressed red. Reward: 1"),
        VisibleTurn(1, "[select red]", "You selected 'red' as your final answer on turn 2. Correct!"),
    ]
    end0 = (
        "Episode 1 finished. You selected 'red' — correct! Score: 0.970. "
        "Probabilities: red=0.80, blue=0.30, green=0.25, yellow=0.40, purple=0.35"
    )
    facts0 = extract_bandits_facts(
        trajectory_id="t",
        episode_idx=0,
        turns=turns0,
        end_feedback=end0,
    )
    assert any("best revealed as red" in f.outcome for f in facts0)

    turns1 = [
        VisibleTurn(0, "[red]", "You pressed red. Reward: 0"),
        VisibleTurn(1, "[select blue]", "You selected 'blue'. Wrong!"),
    ]
    end1 = (
        "Episode 2 finished. You selected 'blue' — wrong. The best was 'green'. "
        "Score: 0.000. Probabilities: green=0.75, red=0.40, blue=0.30, yellow=0.20, purple=0.25"
    )
    facts1 = extract_bandits_facts(
        trajectory_id="t",
        episode_idx=1,
        turns=turns1,
        end_feedback=end1,
    )
    all_facts = facts0 + facts1
    view = build_bandits_current_view(all_facts, trajectory_id="t", as_of_episode_idx=2)
    active_best = [
        c
        for c in view.claims
        if c.predicate == "revealed_best_button" and c.current_status == "active"
    ]
    superseded = [
        c
        for c in view.claims
        if c.predicate == "revealed_best_button" and c.current_status == "superseded"
    ]
    assert len(active_best) == 1 and active_best[0].object_value == "green"
    assert any(c.object_value == "red" for c in superseded)
    assert any(r.relation_type == "supersedes" for r in view.relations)
    assert "ACTIVE session state" in view.render_text
    assert "SUPERSEDED" in view.render_text
