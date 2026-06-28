import math

import numpy as np


class UncertaintyCalculator:

    VALID_METHODS = {"entropy", "confidence_gap"}

    def __init__(self, method="entropy", top_k=5, ewca_alpha=0.1, ewca_beta=0.5):
        if method not in self.VALID_METHODS:
            raise ValueError(f"Unknown method '{method}'. Choose from {self.VALID_METHODS}.")
        self.method = method
        self.top_k = max(top_k, 2)
        self.ewca_alpha = ewca_alpha
        self.ewca_beta = ewca_beta

    def token_uncertainty(self, token_logprobs):
        if not token_logprobs:
            return 1.0

        logprobs = np.array(list(token_logprobs.values()), dtype=np.float64)
        probs = np.exp(logprobs)
        probs = np.clip(probs, 1e-12, None)
        probs = probs / probs.sum()

        if self.method == "entropy":
            k = max(len(probs), 2)
            entropy = -np.sum(probs * np.log(probs))
            normaliser = math.log(k)
            if normaliser < 1e-12:
                return 0.0
            return float(np.clip(entropy / normaliser, 0.0, 1.0))

        sorted_probs = np.sort(probs)[::-1]
        p_top1 = sorted_probs[0]
        p_top2 = sorted_probs[1] if len(sorted_probs) > 1 else 0.0
        return float(np.clip(1.0 - (p_top1 - p_top2), 0.0, 1.0))

    def segment_uncertainty(self, token_uncertainties):
        if not token_uncertainties:
            return 0.0

        us = np.array(token_uncertainties, dtype=np.float64)
        T = len(us)
        weights = np.zeros(T, dtype=np.float64)

        for t in range(T):
            exp_term = math.exp(self.ewca_alpha * t)
            if t == 0:
                causal_term = 1.0
            else:
                causal_term = float(np.prod(1.0 + self.ewca_beta * us[:t]))
            weights[t] = exp_term * causal_term

        Z = weights.sum()
        if Z < 1e-12:
            return float(np.mean(us))

        u_seg = float(np.dot(weights, us) / Z)
        return float(np.clip(u_seg, 0.0, 1.0))

    def compute(self, segment_logprobs):
        token_us = [self.token_uncertainty(lp) for lp in segment_logprobs]
        seg_u = self.segment_uncertainty(token_us)
        return seg_u, token_us
