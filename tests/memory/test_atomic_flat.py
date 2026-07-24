"""Unit tests for Mem0-style atomic flat memory helpers."""
from latentgym.memory.retriever import (
    format_atomic_flat_memories,
    parse_atomic_flat_bullets,
)


def test_parse_atomic_flat_bullets_dedupes_and_strips():
    text = """
- Target was 42
* Target was 42
1. Guess 50 was too high
2) Guess 50 was too high
not a bullet ignored if empty

- Range felt like 1-100
"""
    got = parse_atomic_flat_bullets(text)
    assert got == [
        "Target was 42",
        "Guess 50 was too high",
        "not a bullet ignored if empty",
        "Range felt like 1-100",
    ]


def test_format_atomic_flat_memories_empty():
    assert format_atomic_flat_memories([]) == ""


def test_format_atomic_flat_memories_lists():
    block = format_atomic_flat_memories(["a", "b"])
    assert "Atomic flat memories" in block
    assert "[m0] a" in block
    assert "[m1] b" in block
