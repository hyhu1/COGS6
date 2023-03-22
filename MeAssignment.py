import scipy.stats
import numpy as np
import matplotlib.pyplot as plt
import math

class Metropolis:
    def __init__(self, logTarget, initialState):
        self.logTarget= logTarget
        self.currentState= initialState
