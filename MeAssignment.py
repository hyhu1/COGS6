import numpy as np


class Metropolis:
    def __init__(self, logTarget, initialState, stepSize=0.1):
        self.logTarget = logTarget
        self.state = initialState
        self.stepSize = stepSize
        self.accept_count = 0
        self.total_count = 0
        self.samples = []
        self.acceptanceRate = 0.4
        
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
            proposal = np.random.normal(self.state, self.stepSize)
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
        return {
            'mean': np.mean(self.samples),
            'sd': np.std(self.samples),
            'se': np.std(self.samples) / np.sqrt(len(self.samples)),
            'c025': np.percentile(self.samples, 2.5),
            'c975': np.percentile(self.samples, 97.5),
            'acceptanceRate': self.acceptanceRate
        }
