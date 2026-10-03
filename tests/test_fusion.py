import subprocess
import sys
from pathlib import Path

import pytest

from thinkfuse import MockModelClient, ThinkFuse, select_aligned_prefix

BODY = "60 + 90 = 150 km in 2 hours, so the average is 150 / 2 = 75 km/h. " * 4
BODY = BODY[: len(BODY) // 8 * 8]
SCRIPT = "<think>\n" + BODY + "</think>" + "\n\nThe average speed is 75 km/h."
PROMPT = "Question: a train covers 60 km and then 90 km in two hours. Average speed?\nAnswer:"
THINKING_SEGMENTS = len("<think>\n" + BODY + "</think>") // 8


def run(**options):
    primary = MockModelClient("qwen3-4b", script=SCRIPT)
    auxiliary = MockModelClient("ministral-3b-reasoning")
    return ThinkFuse(primary, auxiliary, **options).run(PROMPT)


def test_run_is_deterministic():
    assert run() == run()


def test_primary_script_is_kept_when_the_primary_wins():
    assert run().text == SCRIPT


def test_only_thinking_segments_are_monitored():
    result = run()
    assert result.total_segments == THINKING_SEGMENTS
    assert 0 <= result.fused_segments <= result.total_segments
    assert result.fusion_ratio == pytest.approx(result.fused_segments / result.total_segments)


def test_threshold_out_of_reach_never_fuses():
    result = run(tau=1e9)
    assert result.fused_segments == 0
    assert result.text == SCRIPT


def test_threshold_below_every_segment_always_fuses():
    result = run(tau=-1e9, budget_lambda=0.0)
    assert result.fused_segments == result.total_segments == THINKING_SEGMENTS


def test_soft_budget_lowers_the_fusion_ratio():
    free = run(init_mu=0.0, init_sigma=0.0, tau=0.0, budget_lambda=0.0)
    budgeted = run(init_mu=0.0, init_sigma=0.0, tau=0.0, budget_lambda=50.0)
    assert budgeted.fused_segments <= free.fused_segments


def test_generation_stops_at_the_token_budget():
    result = run(max_new_tokens=16)
    assert result.text == SCRIPT[:16]


def test_loose_alignment_keeps_the_longest_shared_prefix():
    a, b = MockModelClient("a"), MockModelClient("b")
    assert select_aligned_prefix(a, [b], a.encode("hello wo")) == ("hello wo", 8)


def test_strict_alignment_keeps_the_shortest_shared_prefix():
    a, b = MockModelClient("a"), MockModelClient("b")
    assert select_aligned_prefix(a, [b], a.encode("hello wo"), mode="strict") == ("h", 1)


def test_alignment_of_nothing_is_empty():
    a, b = MockModelClient("a"), MockModelClient("b")
    assert select_aligned_prefix(a, [b], []) == ("", 0)


def test_example_script_runs():
    example = Path(__file__).resolve().parents[1] / "example.py"
    done = subprocess.run([sys.executable, str(example)], capture_output=True, text=True, cwd=example.parent)
    assert done.returncode == 0, done.stderr
    assert "fusion ratio" in done.stdout
