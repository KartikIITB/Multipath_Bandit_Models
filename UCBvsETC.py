from math import sqrt, log, ceil

import numpy as np
import pickle
import matplotlib.pyplot as plt
from bandit_models.Algorithms import ETCAlgorithm, UCB1Algorithm
from bandit_models.bandits import GaussianBandit

def initializer(k, mu, sigma):
    '''
    Initializes the bandit and the sub-optimality gap of each arm.
    '''
    # Environment Class of the bandit
    bandit = GaussianBandit(k, mu, sigma)
    # Initializing Deltas
    Deltas = max(mu) - mu

    return bandit, Deltas

# Results from UBC(delta) and ETC for various values of m.

# Values taken from Experiment 7.1 in the book.

n = 1000
k = 2
Delta_values = np.concatenate((np.arange(0.01, 0.3, 0.03), np.arange(0.3, 1, 0.07)), axis = None)
delta = 1/(n**2)

# Initializing R_n and R_upper_bound
R_UCB = np.zeros(20)
R_ETC_optimal = np.zeros(20)
R_ETC_25 = np.zeros(20)
R_ETC_50 = np.zeros(20)
R_ETC_75 = np.zeros(20)
R_ETC_100 = np.zeros(20)

i = 0
for Delta in Delta_values:
    bandit, Deltas = initializer(k, np.asarray([0, -Delta]), np.asarray([1, 1]))
    optimal_m = ETCAlgorithm(n, 100, bandit).optimal_m(Delta)

    R_UCB[i], _ = UCB1Algorithm(n, bandit, delta).regret_analysis(2000, Deltas)
    R_ETC_optimal[i], _ = ETCAlgorithm(n, optimal_m, bandit).regret_analysis(2000, Deltas)
    R_ETC_25[i], _ = ETCAlgorithm(n, 25, bandit).regret_analysis(2000, Deltas)
    R_ETC_50[i], _ = ETCAlgorithm(n, 50, bandit).regret_analysis(2000, Deltas)
    R_ETC_75[i], _ = ETCAlgorithm(n, 75, bandit).regret_analysis(2000, Deltas)
    R_ETC_100[i], _ = ETCAlgorithm(n, 100, bandit).regret_analysis(2000, Deltas)
    print(R_UCB[i])
    i += 1

dataset = {'UCB': R_UCB, 'ETC_optimal': R_ETC_optimal, 'm = 25': R_ETC_25, 'm = 50': R_ETC_50, 'm = 75': R_ETC_75, 'm = 100': R_ETC_100}
DataFile = 'R_values.data'
fw = open(DataFile, 'wb')
pickle.dump(dataset, fw)
fw.close()

# Plotting and comparing results:
plt.plot(Delta_values, R_UCB, marker = 'o')
plt.plot(Delta_values, R_ETC_optimal, marker = 'o')
plt.plot(Delta_values, R_ETC_25, marker = 'o')
plt.plot(Delta_values, R_ETC_50, marker = 'o')
plt.plot(Delta_values, R_ETC_75, marker = 'o')
plt.plot(Delta_values, R_ETC_100, marker = 'o')
# plt.scatter(Delta_values, R_upper_bound)
plt.title("$R_n$ for Various Values of $\Delta$")
plt.xlabel('Values of Delta')
plt.ylabel('Regret Values')
plt.legend(['UCB', 'ETC, optimal', 'ETC, m = 25', 'ETC, m = 50', 'ETC, m = 75', 'ETC, m = 100'])
plt.grid()
plt.show()