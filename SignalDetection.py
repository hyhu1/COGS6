import numpy as np
from scipy.stats import norm
from scipy.special import ndtri
from scipy.optimize import minimize as mini
import matplotlib.pyplot as plt

class SignalDetection:

    def __init__(self, hits, misses, falseAlarms, correctRejections):
        self.hits = hits
        self.misses = misses
        self.falseAlarms = falseAlarms
        self.correctRejections = correctRejections
        self.hit_rate = hits/(hits + misses)
        self.fa_rate = falseAlarms/(falseAlarms + correctRejections)

    def d_prime(self):
        self.d_prime = ndtri(self.hit_rate) - ndtri(self.fa_rate)
        return self.d_prime
    
    def criterion (self):
        self.criterion = -0.5*(ndtri(self.hit_rate) + ndtri(self.fa_rate))
        return self.criterion

    def __add__ (self, other):
        return SignalDetection(self.hits + other.hits , self.misses + other.misses , self.falseAlarms + other.falseAlarms , self.correctRejections + other.correctRejections)

    def __mul__ (self, scalar):
        return SignalDetection(self.hits * scalar , self.misses * scalar , self.falseAlarms * scalar , self.correctRejections * scalar)

    def plot_sdt(self):
        d = SignalDetection.d_prime(self)
        criterion = self.criterion()
        x = np.linspace(-6, 6, 100) # assume that standard deviation is 1
        plt.plot(x, norm.pdf(x, 0 , 1),label = 'noise')
        plt.plot(x, norm.pdf(x, -d , 1),label = 'signal')
        plt.axvline(x = criterion, color = 'black', label = 'criterion')
        plt.title('Noise vs. Signal distribution')
        plt.legend()
        plt.show()

    @staticmethod
    def simulate(dprime, criteriaList, signalCount, noiseCount):
        sdtList = list()
        for i in range(len(criteriaList)):
            hit_r = norm.cdf(0.5*dprime - criteriaList[i]) # we know this relation from the two equations defining dprime and criterion
            fa_r = norm.cdf(-0.5*dprime - criteriaList[i])  # these rates need be calculated using the dprime and criterion values given
            hits, falseAlarms = np.random.binomial(n = [signalCount , noiseCount], p =[hit_r , fa_r]) # binomial rng
            misses = signalCount - hits
            correctRejections = noiseCount - falseAlarms
            sdt_object = SignalDetection(hits, misses, falseAlarms, correctRejections) # intermediate variable storing the class created
            sdtList.append(sdt_object)  
        return sdtList

    @staticmethod
    def plot_roc(sdtList):
        x = np.arange(start = 0.00, stop  = 1.00, step  = 0.01)
        y = SignalDetection.rocCurve(x, SignalDetection.fit_roc(sdtList))
        for j in range(len(sdtList)):
            plt.plot(sdtList[j].fa_rate, sdtList[j].hit_rate, marker = 'o', color = 'black') # original data points
        plt.plot(x, y, '-', color = 'red') # plotting the curve
        plt.plot([0,1], '--', color = 'b') # Performance by chance
        plt.xlim([0,1])
        plt.ylim([0,1])
        plt.title('Receiver Operating Characteristic')
        plt.xlabel('False Alarm Rate')
        plt.ylabel('Hit Rate')
        # plt.show()

    def nLogLikelihood(self,hit_r,fa_r):
        return - self.hits * np.log(hit_r) - self.misses * np.log(1- hit_r) - self.falseAlarms * np.log(fa_r) - self.correctRejections * np.log(1 - fa_r)

    @staticmethod
    def rocCurve(fa_r , a):
        return norm.cdf(a + ndtri(fa_r))

    @staticmethod
    def rocLoss(a, sdtList):
        cumulative_Likelihood = 0
        hit_r = []
        for i in range(len(sdtList)):
            hit_r.append(SignalDetection.rocCurve(sdtList[i].fa_rate, a))
            cumulative_Likelihood = cumulative_Likelihood + SignalDetection.nLogLikelihood(sdtList[i], hit_r[i], sdtList[i].fa_rate)
        return cumulative_Likelihood

    @staticmethod
    def fit_roc(sdtList):
        point = np.random.randn()
        result = mini(fun = SignalDetection.rocLoss, x0 = point, args = (sdtList))
        return result.x[0]