class Metropolis:
    def __init__(self, logTarget, initialState, stepSize):
        self.logTarget = logTarget
        self.state = initialState
        self.step_size = step_size=0.1
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
    
    def sample(self, n):
        for _ in range(n):
            proposal = np.random.normal(self.state, self.step_size)
            self._accept(proposal)
            self.samples.append(self.state)
        return self
    
    def adapt(self, blockLengths):
        acceptance_rate = 0
        for blockLength in blockLengths:
            for i in range(blockLength):
                proposal = np.random.normal(self.state, self.stepSize)
                if self._accept(proposal):
                    acceptance_rate += 1
            acceptance_rate /= blockLength
            if acceptance_rate > 0.4:
                self.stepSize *= 1.1
            else:
                self.stepSize /= 1.1
        return self

    def summary(self):
        mean = np.mean(self.samples)
        interval = np.percentile(self.samples, [2.5, 97.5])
        return {'mean': mean, 'credible_interval': interval}