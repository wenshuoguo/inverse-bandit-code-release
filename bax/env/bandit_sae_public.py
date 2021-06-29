from collections import defaultdict
import jax
import numpy as np
import ray
import time


@ray.remote 
def run_(obj):
    """
    Run this instance using SAE algorithm until T rounds

    """
    t = 1
    t_ast = 1
    S_t = [arm for arm in range(obj.num_arms)]

    
    num_arms_left = len(S_t)
    while num_arms_left > 1 and t_ast <= obj.T:
        print(num_arms_left, t_ast)
        #compute best reward if pulled the best arm
        rew_best = obj.get_rew(obj.best_arm)
        mu_max = -100
        for arm in S_t:
            rew_arm = obj.get_rew(arm)
            t_ast += 1

            if arm != obj.best_arm:
                obj.regret = obj.regret  + rew_best - rew_arm
                obj.rew_per_arm[arm] = float(obj.rew_per_arm[arm] * obj.times_pulled[arm] + rew_arm)/float(obj.times_pulled[arm] + 1)
                
            else:
                obj.rew_per_arm[arm] = float(obj.rew_per_arm[arm] * obj.times_pulled[arm] + rew_best)/float(obj.times_pulled[arm] + 1)
            
            if obj.rew_per_arm[arm] >= mu_max:
                mu_max = obj.rew_per_arm[arm]
            obj.times_pulled[arm] += 1

        # Elimination
        S_active = []
        num_arms_left = 0
        for arm in S_t:
            alpha_it = obj.ci_fn(obj.T, obj.times_pulled[arm])
            if obj.rew_per_arm[arm] > mu_max - 2*alpha_it:
                S_active.append(arm)
                num_arms_left += 1
            else: 
                obj.taus[arm] = t  #Eliminated
                obj.gaps[arm] = 2*alpha_it

        S_t = S_active
        t += 1

    if len(S_t) != 1:
        identified_best_arm = -1

    elif len(S_t) ==1:
        identified_best_arm = S_t[0]

    is_valid = obj.is_valid_demo(identified_best_arm)    
    return (is_valid, obj.gaps, obj)



class Demonstrator(object):
    '''
    Bandit instance performing the SAE algorithm for K arms
    
    '''
    def __init__(self, prior=None, T=2500, beta=0.05, delta=0.1, seed=88, scale_fn=None, ci_fn=None):
        
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
        self.best_mean = self.means[self.best_arm]
        self.num_arms = len(self.prior)
        
        self.beta = beta 
        self.delta = delta 
        self.T = T
        
        if ci_fn is None:
            def log_bound(T, times_pulled):
                return np.sqrt( np.log10(T)/times_pulled )
            ci_fn = log_bound
        self.ci_fn = ci_fn
         
        # history
        self.regret = 0
        self.rew_per_arm = [0 for arm in range(self.num_arms)]    # K running sample means 
        self.times_pulled = [0 for arm in range(self.num_arms)]   # [N_1, ..., N_K]
        self.taus = [T for arm in range(self.num_arms)]           # the LAST round where an arm has not been eliminated
        self.gaps = [0 for arm in range(self.num_arms)]
        
    def get_rew(self, arm):
        mu, sigma2 = self.prior[arm, :]
        sigma = np.sqrt(sigma2)
        self.key, subkey = jax.random.split(self.key)
        rew = mu + sigma * jax.random.normal(subkey, [1])[0]
        return rew 
        
    def is_valid_demo(self, id_best_arm):
        is_valid = id_best_arm == self.best_arm
        return is_valid
    