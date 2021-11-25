from math import sqrt, log, ceil

import numpy as np
from numpy import random
import matplotlib.pyplot as plt
from bandit_models.bandits import GaussianBandit
from bandit_models.policies import UCB1Policy

def initializer(k, mu, sigma):
    '''
    Initializes the bandit and the sub-optimality gap of each arm.
    '''
    # Environment Class of the bandit
    bandit = GaussianBandit(k, mu, sigma)

    # Initializing Deltas
    Deltas = max(mu) - mu

    return bandit, Deltas

def UCB_1(n, k, bandit, delta):
    '''
    Performs the UCB(delta) algorithm. 
    '''
    # Initializing actions, rewards, number of times each action has been played, regrets each round, average rewards, the overall obtained regret and the expected regret
    A = np.zeros(n) # Actions
    X = np.zeros(n) # rewards
    T = np.zeros(k) # Number of times eac action is played
    empirical_mean = np.zeros(k) # average_rewards
    UCB = np.full_like(T, np.inf)
    R = 0 # regret

    # Implementing the algorithm:
    for t in range(n):
        A[t], X[t], T, empirical_mean, UCB = UCB1Policy(bandit, delta).choice(UCB, T, empirical_mean)

    R = max(bandit.mu) - sum(X) 

    return A, X, T, UCB, R

def regret_analysis(n, k, bandit, Deltas, delta, no_of_sims):
    '''
    Performs a Monte-Carlo simulation to analyze the regret.
    '''
    R_n = 0 # Initializing regret
    non_zero_Deltas = Deltas[Deltas != 0]
    R_upper_bound = 3*sum(Deltas) + 16*log(n)*sum(1./non_zero_Deltas) # Finding the upper bound.
    expted_T = np.zeros(k) # Initializing E[T_i] for all arms

    for sim in range(1, no_of_sims+1):
        _, _, T, _, _ = UCB_1(n, k, bandit, delta)
        expted_T += T
    
    expted_T = expted_T/no_of_sims
    # print(expted_T)

    R_n = np.dot(Deltas, expted_T)

    return R_n, R_upper_bound

# Results from UBC(delta).

# Values taken from Experiment 7.1 in the book.

n = 1000
k = 2
Delta_values = np.concatenate((np.arange(0, 0.3, 0.03), np.arange(0.3, 1, 0.07)), axis = None)

# Initializing R_n and R_upper_bound
R_n = np.zeros(20)
R_upper_bound = np.zeros(20)

i = 0
for Delta in Delta_values:
    bandit, Deltas = initializer(k, np.asarray([0, -Delta]), np.asarray([1, 1]))
    R_n[i], R_upper_bound[i] = regret_analysis(n, k, bandit, Deltas, (1/(n**2)), 1000)
    # print(R_n[i])
    i += 1

# Plotting and comparing results:
plt.plot(Delta_values, R_n)
# plt.scatter(Delta_values, R_upper_bound)
plt.title("R_{n} for Various Values of $\Delta$")
plt.xlabel('Values of Delta')
plt.ylabel('Regret Values')
plt.legend(['Expected', 'Upper Bound'])
plt.grid()
plt.show()


