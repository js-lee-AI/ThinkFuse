class SoftFusionBudget:

    def __init__(self):
        self.total_segments = 0
        self.fused_segments = 0

    @property
    def fusion_ratio(self):
        if self.total_segments == 0:
            return 0.0
        return self.fused_segments / self.total_segments

    def record(self, was_fused):
        self.total_segments += 1
        if was_fused:
            self.fused_segments += 1

    def reset(self):
        self.total_segments = 0
        self.fused_segments = 0


class AdaptiveThreshold:

    def __init__(self, eta=0.05, tau=1.0, budget_lambda=50.0, init_mu=0.5, init_sigma=0.1):
        self.eta = eta
        self.tau = tau
        self.budget_lambda = budget_lambda
        self.mu = init_mu
        self.sigma = init_sigma

    def get_threshold(self, fusion_ratio):
        theta_base = self.mu + self.tau * self.sigma
        return theta_base * (1.0 + self.budget_lambda * fusion_ratio ** 2)

    def update(self, uncertainty):
        mu_old = self.mu
        self.sigma = (1.0 - self.eta) * self.sigma + self.eta * abs(uncertainty - mu_old)
        self.mu = (1.0 - self.eta) * mu_old + self.eta * uncertainty

    def should_fuse(self, uncertainty, fusion_ratio):
        return uncertainty >= self.get_threshold(fusion_ratio)
