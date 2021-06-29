from collections import defaultdict
import jax
import numpy as np
import ray
import time
from tqdm import tqdm


@ray.remote
def run_(obj):
    """
    Run this instance using UCB algorithm until T rounds
    """
    t_ast = 1
    S = [arm for arm in range(obj.num_arms)]

    ## initially pull arms once 
    for arm_to_pull in range(obj.num_arms):
        rew_best = obj.get_rew(obj.best_arm)
        rew_arm_to_pull = obj.get_rew(arm_to_pull)

        if arm_to_pull  != obj.best_arm:
            obj.regret = obj.regret  + rew_best - rew_arm_to_pull
            obj.gaps[arm_to_pull] = 1.0
            obj.rew_per_arm[arm_to_pull] = rew_arm_to_pull
        else:
            obj.rew_per_arm[arm_to_pull] = rew_best
            
        obj.times_pulled[arm_to_pull] = 1
        t_ast += 1    

    while t_ast <= obj.T:            

        confidence_bounds = []
        confidence_intervals = []
        #compute best reward if pulled the best arm
        rew_best = obj.get_rew(obj.best_arm)

        for arm in S:
            confidence_intervals.append(obj.ci_fn(obj.T, obj.times_pulled[arm]))
            confidence_bounds.append(obj.rew_per_arm[arm] + confidence_intervals[arm])

        arm_to_pull = np.argmax(confidence_bounds) 
        rew_arm_to_pull = obj.get_rew(arm_to_pull)

        # if suboptimal arm is pulled, estimate gap!
        if arm_to_pull != obj.best_arm:
            obj.regret = obj.regret  + rew_best - rew_arm_to_pull
            obj.gaps[arm_to_pull] = confidence_intervals[arm_to_pull] - confidence_intervals[obj.best_arm]
            obj.rew_per_arm[arm_to_pull] = float(rew_arm_to_pull + obj.rew_per_arm[arm_to_pull] * obj.times_pulled[arm_to_pull])/float(obj.times_pulled[arm_to_pull] + 1)
            obj.times_pulled[arm_to_pull] += 1
        # if the optimal arm is pulled, we incur no regret 
        else:
            obj.rew_per_arm[arm_to_pull] = float(rew_best + obj.rew_per_arm[arm_to_pull] * obj.times_pulled[arm_to_pull])/float(obj.times_pulled[arm_to_pull] + 1)
            obj.times_pulled[arm_to_pull] += 1
            
        t_ast += 1

    identified_best_arm = arm_to_pull  # last arm pulled
    pull_best_arm_last = identified_best_arm == obj.best_arm
    is_valid = pull_best_arm_last and obj.is_valid(obj.times_pulled, obj.gaps)
    return (is_valid, obj.gaps, obj)


    
class Demonstrator(object):
    '''
    Bandit instance performing the SAE algorithm for K arms
    
    '''
    def __init__(self, prior=None, T=2500, seed=88, ci_fn=None, **kwargs):
        
        '''
        Args:
        prior: nparray with shape K * [mean, variance] 
        T: total rounds to run
        
        '''

        # all parameters for managing randomness goes here.
        self.seed = seed
        self.key = jax.random.PRNGKey(seed=seed)
        
        self.prior = prior
        self.means = prior[:, 0]
        self.sigma2 = prior[:, 1]
        
        self.best_arm = np.argmax(self.means)
        self.num_arms = len(self.prior)
        
        self.T = T
        
        # history
        self.regret = 0
        self.regret_t = [0 for t in range(self.T)]
        
        self.rew_per_arm = [0 for arm in range(self.num_arms)]     # K running sample means 
        self.rew_t = [[0 for arm in range(self.num_arms)] for t in range(self.T)]
        self.times_pulled = [0 for arm in range(self.num_arms)]   # [N_1, ..., N_K]
        self.switch_time = [0 for arm in range(self.num_arms)]

        # confidence interval
        if ci_fn is None:
            def log_bound(T, times_pulled):
                return np.sqrt( np.log10(T)/times_pulled )
            ci_fn = log_bound
        self.ci_fn = ci_fn
        
        # gaps in mean estimates
        self.gaps = [0 for arm in range(self.num_arms)]    # the LAST round where an arm has not been eliminated
        self.gaps_t = [[0 for arm in range(self.num_arms)] for t in range(self.T)]

        self.mu_gap_t = [[0 for arm in range(self.num_arms)] for t in range(self.T)]
        self.mu_gap = [0 for arm in range(self.num_arms)]

        
    def get_rew(self, arm):
        """returns reward corresponding to an arm pull"""
        
        mu, sigma2 = self.prior[arm, :]
        sigma = np.sqrt(sigma2)
        self.key, subkey = jax.random.split(self.key)
        rew_arm = mu + sigma * jax.random.normal(subkey, [1])[0]
        return rew_arm
    

    def is_valid(self, times_pulled=None, gaps=None, min_pulls_per_arm=5):
        is_valid_ = ~np.isnan(gaps).any()
        is_valid_ &= (np.array(times_pulled) > min_pulls_per_arm).all()
        return is_valid_
        