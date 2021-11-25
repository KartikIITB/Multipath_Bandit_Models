import numpy as np
from numpy import random

class MultiArmedBandit(object):
    '''
    Multi armed bandit class
    '''
    def __init__(self, k):
        self.k = k
        self.arms = np.arange(k)
        self.optimal_arm = 0 # Default

    def pull(self, arm):
        reward = 0
        return reward

class GaussianBandit(MultiArmedBandit):
    '''
    Gaussian bandit class
    '''
    def __init__(self, k, mu, sigma):
        super().__init__(k)
        self.mu = mu
        self.sigma = sigma
        self.optimal_arm = np.argmax(mu)

    def pull(self, arm):
        reward = self.sigma[arm]*random.randn() + self.mu[arm]
        return reward
class BernoulliBandit(MultiArmedBandit):
    '''
    Bernoulli bandit class
    '''
    def __init__(self, k, p):
        super().__init__(k)
        self.p = p

    def pull(self, arm):
        reward = random.binomial(1, self.p[arm])
        return reward
