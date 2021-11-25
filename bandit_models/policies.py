import numpy as np
from math import sqrt, log

class Policy(object):
    '''
    A generic policy based on history.
    '''
    def __str__(self):
        return 'generic policy'

    def choice(self, t):
        return None, 0

class Explore(Policy):
    '''
    Explores all the arms irrespective of reward.
    '''
    def __init__(self, bandit):
        self.bandit = bandit
        self.k = bandit.k # Number of arms

    def choice(self, t, T, empirical_mean):
        A_t = int(t%self.k) # Action
        X_t = self.bandit.pull(A_t) # Reward
        empirical_mean[A_t] = empirical_mean[A_t]*T[A_t] + X_t
        T[A_t] += 1 # Update number of times A_t has been played
        empirical_mean[A_t] = empirical_mean[A_t]/T[A_t] # Update mean of corresponding arm
        return A_t, X_t, T, empirical_mean

class Commit(Policy):
    '''
    Commits to a certain action irrespective of reward.
    '''
    def __init__(self, bandit):
        self.bandit = bandit

    def choice(self, A, T, empirical_mean):
        A_t = A # Action
        X_t = self.bandit.pull(A_t) # Reward
        empirical_mean[A_t] = empirical_mean[A_t]*T[A_t] + X_t
        T[A_t] += 1 # Update number of times A_t has been played
        empirical_mean[A_t] = empirical_mean[A_t]/T[A_t] # Update mean of corresponding arm
        return A_t, X_t, T, empirical_mean

class UCB1Policy(Policy):
    '''
    Picks arm corresponding to largest Upper Confidence Bound.
    '''
    def __init__(self, bandit, delta):
        self.bandit = bandit
        self.k = bandit.k # Number of arms
        self.delta = delta

    def choice(self, UCB, T, empirical_mean):
        A_t = np.argmax(UCB)
        X_t = self.bandit.pull(A_t)
        empirical_mean[A_t] = empirical_mean[A_t]*T[A_t] + X_t
        T[A_t] += 1 # Update number of times A_t has been played
        empirical_mean[A_t] = empirical_mean[A_t]/T[A_t] # Update mean of corresponding arm
        UCB[A_t] = empirical_mean[A_t] + sqrt((2*log(1/self.delta))/T[A_t])
        return A_t, X_t, T, empirical_mean, UCB
