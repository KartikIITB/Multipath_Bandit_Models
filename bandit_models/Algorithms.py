import numpy as np
from math import sqrt, log, ceil
from .policies import Explore, Commit, UCB1Policy

class Algorithm(object):
    '''
    Bandit algorithm for minimizing regret.
    '''
    def __init__(self, n, bandit):
        self.n = n
        self.bandit = bandit
        self.k = bandit.k

    def results(self):
        return 0, 0, 0, 0

# NEXT:

class ETCAlgorithm(Algorithm):
    '''
    Explore Then Commit algorithm.
    '''
    def __init__(self, n, m, bandit):
        super().__init__(n, bandit)
        self.m = m

    def __str__(self):
        return f"ETC algorithm. \n Horizon: {self.n} rounds \n End of exploration: {self.m} rounds \n"

    def optimal_m(self, delta):
        '''
        Optimal value of m.
        '''
        return max(1, ceil((4/(delta**2))*log(self.n*((delta**2)/4))))

    def optimal_upper_bound(self, delta):
        '''
        Upper bound for the regret for optimal m corresponding to a given delta.
        '''
        return min(self.n*delta, delta+(4/delta)*(1+max(0, log(self.n*((delta**2)/4)))))

    def results(self):
        '''
        Performs the ETC algorithm. 
        '''
        # Initializing actions, rewards, number of times each action has been played, regrets each round, average rewards, the overall obtained regret and the expected regret
        A = np.zeros(self.n) # Actions
        X = np.zeros(self.n) # rewards
        T = np.zeros(self.k) # Number of times eac action is played
        R = 0 # regret
        empirical_mean = np.zeros(self.k) # average_rewards

        # Implementing the algorithm:
        for t in range(self.n):
            if t <= self.m*self.k:
                A[t], X[t], T, empirical_mean =  Explore(self.bandit).choice(t, T, empirical_mean) # Explore and update
                A_max = np.argmax(empirical_mean) # Picking reward with maximum average reward
            else:
                A[t], X[t], T, _ = Commit(self.bandit).choice(A_max, T, empirical_mean) # Commit to A_max and update

        R = max(self.bandit.mu) - sum(X) 

        return A, X, T, R

    def regret_analysis(self, no_of_sims, Deltas):
        '''
        Performs a Monte-Carlo simulation to analyze the regret.
        '''
        R_n = 0 # Initializing regret
        R_upper_bound = self.m*sum(Deltas) + (self.n - self.m*self.m)*np.dot(Deltas, np.exp(-(self.m*(Deltas**2))/4)) # Finding the upper bound.
        expted_T = np.zeros(self.k) # Initializing E[T_i] for all arms

        for sim in range(0, no_of_sims):
            _, _, T, _ = self.results()
            expted_T += T
        
        expted_T = expted_T/no_of_sims

        R_n = np.dot(Deltas, expted_T)

        return R_n, R_upper_bound

# NEXT:

class UCB1Algorithm(Algorithm):
    '''
    The UBC(delta) algorithm.
    '''
    def __init__(self, n, bandit, delta):
        super().__init__(n, bandit)
        self.delta = delta

    def __str__(self):
        return f"UBC{self.delta} algorithm. \n Horizon: {self.n} rounds \n"

    def results(self):
        '''
        Performs the UCB(delta) algorithm. 
        '''
        # Initializing actions, rewards, number of times each action has been played, regrets each round, average rewards, the overall obtained regret and the expected regret
        A = np.zeros(self.n) # Actions
        X = np.zeros(self.n) # rewards
        T = np.zeros(self.k) # Number of times eac action is played
        empirical_mean = np.zeros(self.k) # average_rewards
        UCB = np.full_like(T, np.inf)
        R = 0 # regret

        # Implementing the algorithm:
        for t in range(self.n):
            A[t], X[t], T, empirical_mean, UCB = UCB1Policy(self.bandit, self.delta).choice(UCB, T, empirical_mean)

        R = max(self.bandit.mu) - sum(X) 

        return A, X, T, UCB, R

    def regret_analysis(self, no_of_sims, Deltas):
        '''
        Performs a Monte-Carlo simulation to analyze the regret.
        '''
        R_n = 0 # Initializing regret
        non_zero_Deltas = Deltas[Deltas != 0]
        R_upper_bound = 3*sum(Deltas) + 16*log(self.n)*sum(1./non_zero_Deltas)# Finding the upper bound.
        expted_T = np.zeros(self.k) # Initializing E[T_i] for all arms

        for sim in range(0, no_of_sims):
            _, _, T, _, _ = self.results()
            expted_T += T
        
        expted_T = expted_T/no_of_sims

        R_n = np.dot(Deltas, expted_T)

        return R_n, R_upper_bound
