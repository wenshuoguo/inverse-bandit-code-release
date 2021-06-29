"""Implements simple k-arm bandit with Gaussian rewards."""

from collections import defaultdict
import jax
import numpy as np


class SimpleBandit(object):
    def __init__(self, prior, seed=10):
        ## currently serves for K-arm bandit
        self.prior = prior
        self.means = prior[:, 0]
        self.sigma2 = prior[:, 1]
        
        ## all parameters for managing randomness goes here.
        self.seed = seed
        self.key = jax.random.PRNGKey(seed=seed)
        
        self.best_arm = np.argmax(self.means)
        self.num_arms = len(self.prior)

        ## reset this metadata 
        self.history = []
        self.regret = [0]
        self.rew = []
        self.rew_per_arm = [[] for arms in range(self.num_arms)]
        self.times_pulled = [0 for arm in range(self.num_arms)]

    def pull(self, arm, num_samples=1):
        """
        Pull given `arm` for `num_samples` rounds.
        """
        mu, sigma2 = self.prior[arm, :]
        sigma = np.sqrt(sigma2)
        self.key, subkey = jax.random.split(self.key)
        rew = mu + sigma * jax.random.normal(subkey, [num_samples])
        
        ## add arm/reward to history
        for s in range(num_samples):
            self.history.append(arm)
            self.times_pulled[arm] += 1
            self.regret.append(
                self.regret[-1] + self.means[self.best_arm] - self.means[arm])
    
        self.rew_per_arm[arm].extend(rew)
        self.rew.extend(rew)
        return rew
    
    def reset(self, increment_seed=None):
        seed = self.seed
        
        # the increment seed is to control random seeds across demonstrators
        if increment_seed:
            seed += increment_seed
        self.key = jax.random.PRNGKey(seed=seed)
        
        self.history = []
        self.regret = [0]
        self.rew = []
        self.times_pulled = [0 for arm in range(self.num_arms)]


        