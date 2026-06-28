from dataclasses import dataclass

from .alignment import select_aligned_prefix
from .scoring import select_by_primary_ppl
from .tags import adapt_think_tags_for_model, normalize_think_tags
from .threshold import AdaptiveThreshold, SoftFusionBudget
from .uncertainty import UncertaintyCalculator


@dataclass
class FusionResult:
    text: str
    total_segments: int
    fused_segments: int
    fusion_ratio: float


class ThinkFuse:

    def __init__(
        self,
        primary,
        auxiliary,
        preview_tokens=8,
        top_k=5,
        uncertainty_method="entropy",
        ewca_alpha=0.1,
        ewca_beta=0.5,
        ema_eta=0.05,
        tau=1.0,
        budget_lambda=50.0,
        init_mu=0.5,
        init_sigma=0.1,
        align_mode="loose",
        max_new_tokens=512,
    ):
        self.primary = primary
        self.auxiliary = auxiliary
        self.preview_tokens = preview_tokens
        self.top_k = top_k
        self.align_mode = align_mode
        self.max_new_tokens = max_new_tokens
        self.uncertainty = UncertaintyCalculator(
            method=uncertainty_method,
            top_k=top_k,
            ewca_alpha=ewca_alpha,
            ewca_beta=ewca_beta,
        )
        self.threshold = AdaptiveThreshold(
            eta=ema_eta,
            tau=tau,
            budget_lambda=budget_lambda,
            init_mu=init_mu,
            init_sigma=init_sigma,
        )
        self.budget = SoftFusionBudget()

    def run(self, prompt):
        primary, aux = self.primary, self.auxiliary
        self.budget.reset()

        c_p = primary.encode(prompt)
        c_a = aux.encode(prompt)
        generated = 0
        fused_text = ""
        thinking_ended = False

        for _ in range(self.max_new_tokens):
            primary_ids, primary_eos, primary_logprobs = primary.generate(
                c_p, self.preview_tokens, self.top_k
            )
            if not primary_ids:
                break

            primary_seg, primary_len = select_aligned_prefix(
                primary, [aux], primary_ids, self.align_mode
            )
            primary_seg = normalize_think_tags(primary_seg, primary.name)
            if not primary_seg:
                break

            chosen_seg = primary_seg
            winner = primary
            winner_ids = primary_ids[:primary_len]

            if not thinking_ended:
                seg_u, _ = self.uncertainty.compute(primary_logprobs)
                ratio = self.budget.fusion_ratio
                do_fuse = self.threshold.should_fuse(seg_u, ratio)
                self.threshold.update(seg_u)

                if do_fuse:
                    aux_ids, _, _ = aux.generate(c_a, self.preview_tokens, self.top_k)
                    aux_seg, aux_len = select_aligned_prefix(
                        aux, [primary], aux_ids, self.align_mode
                    )
                    aux_seg = normalize_think_tags(aux_seg, aux.name)

                    candidates = [primary_seg]
                    sources = [(primary, primary_ids[:primary_len])]
                    if aux_seg.strip():
                        candidates.append(aux_seg)
                        sources.append((aux, aux_ids[:aux_len]))

                    scored = [adapt_think_tags_for_model(c, primary.name) for c in candidates]
                    best, _ = select_by_primary_ppl(primary, c_p, scored)
                    chosen_seg = candidates[best]
                    winner, winner_ids = sources[best]

                self.budget.record(do_fuse)

            if not thinking_ended and "</think>" in chosen_seg:
                thinking_ended = True

            fused_text += chosen_seg
            generated += len(winner_ids)

            if winner is primary:
                c_p.extend(winner_ids)
                c_a.extend(aux.encode(chosen_seg))
            else:
                c_a.extend(winner_ids)
                c_p.extend(primary.encode(chosen_seg))

            if generated >= self.max_new_tokens:
                break
            if winner is primary and primary_eos:
                break

        return FusionResult(
            text=fused_text,
            total_segments=self.budget.total_segments,
            fused_segments=self.budget.fused_segments,
            fusion_ratio=self.budget.fusion_ratio,
        )
