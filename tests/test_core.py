import math

import pytest

from thinkfuse import (
    AdaptiveThreshold,
    PhaseDetector,
    SoftFusionBudget,
    UncertaintyCalculator,
    adapt_think_tags_for_model,
    normalize_think_tags,
    select_by_primary_ppl,
)


def logprobs(probs):
    return {i: math.log(p) for i, p in enumerate(probs)}


def test_entropy_is_one_for_a_uniform_distribution():
    calc = UncertaintyCalculator()
    assert calc.token_uncertainty(logprobs([0.2] * 5)) == pytest.approx(1.0)


def test_entropy_matches_the_normalized_shannon_entropy():
    probs = [0.9, 0.025, 0.025, 0.025, 0.025]
    expected = -sum(p * math.log(p) for p in probs) / math.log(5)
    assert UncertaintyCalculator().token_uncertainty(logprobs(probs)) == pytest.approx(expected)


def test_entropy_renormalizes_the_top_k_mass():
    full = UncertaintyCalculator().token_uncertainty(logprobs([0.5, 0.3, 0.2]))
    truncated = UncertaintyCalculator().token_uncertainty(logprobs([0.25, 0.15, 0.1]))
    assert truncated == pytest.approx(full)


def test_missing_logprobs_count_as_fully_uncertain():
    assert UncertaintyCalculator().token_uncertainty({}) == 1.0


def test_confidence_gap_is_one_minus_the_top_two_margin():
    calc = UncertaintyCalculator(method="confidence_gap")
    assert calc.token_uncertainty(logprobs([0.7, 0.2, 0.1])) == pytest.approx(0.5)


def test_unknown_method_is_rejected():
    with pytest.raises(ValueError):
        UncertaintyCalculator(method="variance")


def test_ewca_matches_the_weighted_average():
    alpha, beta = 0.1, 0.5
    us = [0.1, 0.4, 0.2, 0.9]
    weights = []
    for i in range(len(us)):
        weights.append(math.exp(alpha * i) * math.prod(1.0 + beta * u for u in us[:i]))
    expected = sum(w * u for w, u in zip(weights, us)) / sum(weights)
    calc = UncertaintyCalculator(ewca_alpha=alpha, ewca_beta=beta)
    assert calc.segment_uncertainty(us) == pytest.approx(expected)


def test_ewca_emphasizes_later_tokens():
    calc = UncertaintyCalculator()
    assert calc.segment_uncertainty([0.0, 0.0, 0.0, 1.0]) > calc.segment_uncertainty([1.0, 0.0, 0.0, 0.0])


def test_ewca_of_a_constant_segment_is_that_constant():
    assert UncertaintyCalculator().segment_uncertainty([0.3] * 8) == pytest.approx(0.3)


def test_empty_segment_has_zero_uncertainty():
    assert UncertaintyCalculator().segment_uncertainty([]) == 0.0


def test_compute_returns_the_segment_and_token_values():
    calc = UncertaintyCalculator()
    segment, tokens = calc.compute([logprobs([0.2] * 5)] * 8)
    assert segment == pytest.approx(1.0)
    assert tokens == pytest.approx([1.0] * 8)


def test_initial_threshold_is_mu_plus_tau_sigma():
    assert AdaptiveThreshold().get_threshold(0.0) == pytest.approx(0.6)


def test_soft_budget_raises_the_threshold_quadratically():
    threshold = AdaptiveThreshold()
    assert threshold.get_threshold(0.1) == pytest.approx(0.6 * (1.0 + 50.0 * 0.01))
    assert threshold.get_threshold(0.2) == pytest.approx(0.6 * (1.0 + 50.0 * 0.04))


def test_ema_update_uses_the_previous_mean():
    threshold = AdaptiveThreshold()
    threshold.update(0.9)
    assert threshold.sigma == pytest.approx(0.95 * 0.1 + 0.05 * 0.4)
    assert threshold.mu == pytest.approx(0.95 * 0.5 + 0.05 * 0.9)


def test_fusion_triggers_at_or_above_the_threshold():
    threshold = AdaptiveThreshold()
    theta = threshold.get_threshold(0.0)
    assert threshold.should_fuse(theta, 0.0)
    assert not threshold.should_fuse(theta - 1e-6, 0.0)


def test_budget_tracks_the_cumulative_fusion_ratio():
    budget = SoftFusionBudget()
    assert budget.fusion_ratio == 0.0
    for fused in [True, False, False, True]:
        budget.record(fused)
    assert budget.fusion_ratio == pytest.approx(0.5)
    budget.reset()
    assert budget.total_segments == 0 and budget.fusion_ratio == 0.0


def test_reasoning_tags_round_trip_for_exaone_deep():
    text = "<thought>step</thought> done"
    normalized = normalize_think_tags(text, "EXAONE-Deep-2.4B")
    assert normalized == "<think>step</think> done"
    assert adapt_think_tags_for_model(normalized, "EXAONE-Deep-2.4B") == text


def test_reasoning_tags_are_untouched_for_other_models():
    text = "<think>step</think> done"
    assert normalize_think_tags(text, "qwen3-4b") == text
    assert adapt_think_tags_for_model(text, "qwen3-4b") == text


def test_phase_detector_follows_the_think_tags():
    detector = PhaseDetector()
    assert detector.detect("<think> working") == PhaseDetector.THINKING
    assert detector.detect("<think> working </think> answer") == PhaseDetector.ANSWER
    assert detector.detect("") == PhaseDetector.ANSWER


class FixedPerplexity:

    def __init__(self, table):
        self.table = table

    def perplexity(self, context_ids, segment_text):
        return self.table[segment_text]


def test_selection_takes_the_lowest_primary_perplexity():
    client = FixedPerplexity({"a": 3.0, "b": 1.5, "c": 2.0})
    best, ppls = select_by_primary_ppl(client, [], ["a", "b", "c"])
    assert best == 1
    assert ppls == [3.0, 1.5, 2.0]


def test_selection_needs_a_candidate():
    with pytest.raises(ValueError):
        select_by_primary_ppl(FixedPerplexity({}), [], [])
