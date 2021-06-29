from collections import defaultdict
import jax
import numpy as np
import ray
import time
from tqdm import tqdm


@ray.remote
def run_(obj):
        """
        Run this instance using SAE algorithm until T rounds
        """
        t_ast = 1
        S = [arm for arm in range(obj.num_arms)]

        for arm_to_pull in range(obj.num_arms):
            rew_best = obj.get_rew(obj.best_arm)
            rew_arm_to_pull = obj.get_rew(arm_to_pull)

            if arm_to_pull  != obj.best_arm:
                obj.regret = obj.regret  + rew_best - rew_arm_to_pull
                obj.regret = obj.regret  + rew_best - rew_arm_to_pull
                obj.gaps[arm_to_pull] = 1.0
                obj.rew_per_arm[arm_to_pull] = float(obj.rew_per_arm[arm_to_pull] * obj.times_pulled[arm_to_pull] + rew_arm_to_pull)/float(obj.times_pulled[arm_to_pull] + 1)
                obj.times_pulled[arm_to_pull] += 1
            else:
                obj.rew_per_arm[arm_to_pull] = float(obj.rew_per_arm[arm_to_pull] * obj.times_pulled[arm_to_pull] + rew_best)/float(obj.times_pulled[arm_to_pull] + 1)
                obj.times_pulled[arm_to_pull] += 1

            for arm in S:
                obj.gaps_t[t_ast][arm] = obj.gaps[arm]
            obj.regret_t[t_ast] = obj.regret
            
            # compute the mean diff 
            obj.mu_gap[arm_to_pull] = obj.rew_per_arm[obj.best_arm] - obj.rew_per_arm[arm_to_pull]
            
            for arm in S:
                obj.mu_gap_t[t_ast][arm] = obj.rew_per_arm[obj.best_arm] - obj.rew_per_arm[arm]

            ## update the reward at current timestep
            obj.rew_t[t_ast] = obj.rew_per_arm
            t_ast += 1    

        while t_ast < obj.T:            
            # delta = 1/(t_ast**3)    
            # scale = obj.scale_fn(t_ast)

            confidence_bounds = []
            #compute best reward if pulled the best arm
            rew_best = obj.get_rew(obj.best_arm)

            for arm in S:
                if obj.times_pulled[arm] == 0:
                    confidence_bounds.append(np.inf)
                else:
                    confidence_bounds.append(obj.rew_per_arm[arm] + obj.ci_fn(t_ast, obj.times_pulled[arm]))

            arm_to_pull = np.argmax(confidence_bounds) 
            rew_arm_to_pull = obj.get_rew(arm_to_pull)

            # if suboptimal arm is pulled, estimate gap!
            if arm_to_pull != obj.best_arm:
                # we consider two possible estimators for the gap for suboptimal arm "arm_subopt"
                #     "after_switch" : compute difference in CI after switching from "arm_subopt"
                #     "before_switch": compute difference in CI before switching from "arm_subopt"
                if obj.estimate_at =='after_switch':
                    obj.regret = obj.regret  + rew_best - rew_arm_to_pull
                    obj.rew_per_arm[arm_to_pull] = float(obj.rew_per_arm[arm_to_pull] * obj.times_pulled[arm_to_pull] + rew_arm_to_pull)/float(obj.times_pulled[arm_to_pull] + 1)
                    obj.times_pulled[arm_to_pull] += 1
                    temp = obj.rew_per_arm[arm_to_pull] + obj.ci_fn(t_ast, obj.times_pulled[arm_to_pull])  # np.sqrt( (1/(2*obj.times_pulled[arm_to_pull])) * scale)
                    obj.gaps[arm_to_pull] = (temp- obj.rew_per_arm[arm_to_pull]) - (confidence_bounds[obj.best_arm] - obj.rew_per_arm[obj.best_arm])   
                    # compute the bias for the pulled arm here 
                    obj.bias[arm_to_pull] = obj.get_bias_estimate(t_ast, obj.times_pulled[arm_to_pull])
                elif obj.estimate_at == 'before_switch':
                    obj.regret = obj.regret  + rew_best - rew_arm_to_pull
                    obj.gaps[arm_to_pull] = (confidence_bounds[arm_to_pull] - obj.rew_per_arm[arm_to_pull]) - (confidence_bounds[obj.best_arm] - obj.rew_per_arm[obj.best_arm])
                    obj.rew_per_arm[arm_to_pull] = float(obj.rew_per_arm[arm_to_pull] * obj.times_pulled[arm_to_pull] + rew_arm_to_pull)/float(obj.times_pulled[arm_to_pull] + 1)
                    # compute the bias for the pulled arm here 
                    obj.bias[arm_to_pull] = obj.get_bias_estimate(t_ast, obj.times_pulled[arm_to_pull])
                    obj.times_pulled[arm_to_pull] += 1
            # if the optimal arm is pulled, we incur no regret 
            else:
                obj.rew_per_arm[arm_to_pull] = float(obj.rew_per_arm[arm_to_pull] * obj.times_pulled[arm_to_pull] + rew_best)/float(obj.times_pulled[arm_to_pull] + 1)
                obj.times_pulled[arm_to_pull] += 1
            for arm in S:
                obj.gaps_t[t_ast][arm] = obj.gaps[arm]
            obj.regret_t[t_ast] = obj.regret

            # compute the mean diff 
            obj.mu_gap[arm_to_pull] = obj.rew_per_arm[obj.best_arm] - obj.rew_per_arm[arm_to_pull]
            for arm in S:
                obj.mu_gap_t[t_ast][arm] = obj.rew_per_arm[obj.best_arm] - obj.rew_per_arm[arm]

            ## update the reward at current timestep
            obj.rew_t[t_ast] = obj.rew_per_arm
            t_ast += 1
                
            # compute the bias for the pulled arm here 
            # obj.bias[arm_to_pull] = obj.get_bias_estimate(t_ast, obj.times_pulled[arm_to_pull])
                
        identified_best_arm = arm_to_pull  # last arm pulled
        
        
        is_valid = obj.is_valid(obj.times_pulled, obj.gaps)
        return (is_valid, obj.gaps, obj)

    
    
class Demonstrator(object):
    '''
    Bandit instance performing the SAE algorithm for K arms
    
    '''
    def __init__(self, prior=None, T=2500, seed=88, scale_fn=None, ci_fn=None,
                 estimate_at='after_switch', **kwargs):
        
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

        # bias estimates
        self.bias = [0 for arm in range(self.num_arms)]
        
        # confidence interval
        if ci_fn is None:
            def log_bound(t, times_pulled):
                return np.sqrt( (1/(2*times_pulled)) * np.log(t**3))
            ci_fn = log_bound
        self.ci_fn = ci_fn
                                                                                
        # scaling fn
        if scale_fn is None:
            def logt3(t):
                return np.log(t**3)
            scale_fn = logt3
        self.scale_fn = scale_fn
        
        # when to compute bias
        self.estimate_at = estimate_at
        
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
        

    def get_bias_estimate(self, t_ast, T):
        """
        t_ast : total number of arm_pulls so far
        T : #times_pulled for given arm 
        """
        if t_ast <= 1 or T <= 1:
            return 0
        return np.sqrt( np.log((t_ast-1)**3) / (2*(T-1))) - np.sqrt( np.log(t_ast**3) / (2*T))