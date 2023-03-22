import numpy as np

class Metropolis:
    def __init__(self, logTarget, initialState):
        self.logTarget = logTarget
        self.state = initialState
        self.step_size = 1.0
        self.accept_count = 0
        self.total_count = 0
        self.samples = []
        
    def _accept(self, proposal):
        log_ratio = self.logTarget(proposal) - self.logTarget(self.state)
        acceptance_prob = np.exp(min(0, log_ratio))
        accept = np.random.uniform() < acceptance_prob
        if accept:
            self.state = proposal
            self.accept_count += 1
        self.total_count += 1
        return accept
    
    def adapt(self, n_blocks=10):
        acceptance_rate = 0
        for _ in range(n_blocks):
            for _ in range(100):
                proposal = np.random.normal(self.state, self.step_size)
                self._accept(proposal)
            acceptance_rate += self.accept_count / self.total_count
            self.accept_count = 0
            self.total_count = 0
        acceptance_rate /= n_blocks
        target_rate = 0.4
        if acceptance_rate < target_rate / 2:
            self.step_size /= 2
        elif acceptance_rate > target_rate * 2:
            self.step_size *= 2
        return self
    
    def sample(self, n):
        for _ in range(n):
            proposal = np.random.normal(self.state, self.step_size)
            self._accept(proposal)
            self.samples.append(self.state)
        return self
    
    def summary(self):
        mean = np.mean(self.samples)
        interval = np.percentile(self.samples, [2.5, 97.5])
        return {'mean': mean, 'credible_interval': interval}