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
    delta = obj.delta

    while len(S_t) > 1 and t_ast < obj.T:

        #compute best reward if pulled the best arm
        rew_best = obj.get_rew(obj.best_arm)

        # rewards = []
        for arm in S_t:
            rew_arm = obj.get_rew(arm)
            # rewards.append(rew_arm)
            t_ast += 1

            if arm != obj.best_arm:
                obj.regret = obj.regret  + rew_best - rew_arm
                obj.rew_per_arm[arm] = float(obj.rew_per_arm[arm] * obj.times_pulled[arm] + rew_arm)/float(obj.times_pulled[arm] + 1)
                obj.times_pulled[arm] += 1
            else:
                obj.rew_per_arm[arm] = float(obj.rew_per_arm[arm] * obj.times_pulled[arm] + rew_best)/float(obj.times_pulled[arm] + 1)
                obj.times_pulled[arm] += 1

        for arm in range(obj.num_arms):
            obj.gap_t[t][arm] = obj.rew_per_arm[arm] - obj.rew_per_arm[obj.best_arm]

        # Elimination

        mu_tilde_max = -100
        for arm in S_t:
            if obj.rew_per_arm[arm] > mu_tilde_max:
                mu_tilde_max = obj.rew_per_arm[arm] 

        ## add different choices of delta here 
        # delta = 1/t**3
        alpha_t = obj.alpha(t, delta=delta)
        obj.ci_t[t] = alpha_t

        S_active = []
        for arm in S_t:

            if obj.rew_per_arm[arm] > mu_tilde_max - 2*alpha_t:
                S_active.append(arm)
            else: 
                obj.taus[arm] = t  #Eliminated
                obj.gaps[arm] = obj.get_gap_est(t)

        S_t = S_active

        t += 1

    if len(S_t) > 1:
        # print("DID NOT IDENTIFY BEST ARM!")
        identified_best_arm = -1

    elif len(S_t) ==1:
        identified_best_arm = S_t[0]
        obj.taus[identified_best_arm] = t-1
        obj.gaps[identified_best_arm] = obj.get_gap_Est(t-1)

    elif len(S_t) ==0:
        # print("ERROR: all arms are elinimated!")
        identified_best_arm = -1

        
    is_valid = obj.is_valid_demo(identified_best_arm)    
    return (is_valid, obj.gaps, obj)



class Demonstrator(object):
    '''
    Bandit instance performing the SAE algorithm for K arms
    
    '''
    def __init__(self, prior=None, T=2500, beta=0.05, delta=0.1, seed=88):
        
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
        
        self.beta = beta 
        self.delta = delta 
        self.T = T
        
        # history
        self.regret = 0
        self.rew_per_arm = [0 for arm in range(self.num_arms)]     # K running sample means 
        self.times_pulled = [0 for arm in range(self.num_arms)]   # [N_1, ..., N_K]
        self.taus = [T for arm in range(self.num_arms)]    # the LAST round where an arm has not been eliminated
        self.gaps = [0 for arm in ragne(self.num_arms)]
        self.ci_t = [0 for t in range(self.T)]
        self.gap_t = [[0 for arm in range(self.num_arms)] for t in range(self.T)]
        
    def get_rew(self, arm):
        mu, sigma2 = self.prior[arm, :]
        sigma = np.sqrt(sigma2)
        self.key, subkey = jax.random.split(self.key)
        rew = mu + sigma * jax.random.normal(subkey, [1])
        return rew 
    
    def alpha(self, num_iter, delta=None):
        """
        Calculate the confidence interval at t = num_iter
        
        \alpha(t) = \sqrt{\frac{ln(c*n*t^2/\delta)}{t}}, c*n = beta

        Args:
            ts : array of timestamps 
            beta : c*n, where n is number of demonstrators
            delta : SAE specific constant 
        Returns:
            alpha(t) for t = num_iter
        """
        if delta is None:
            delta = self.delta
        elif delta == 'cube':
            delta = 1/(num_iter**3)
        elif delta == 'square':
            delta == 1/(num_iter**2)
        coeff = self.beta / delta 
        if coeff * (num_iter**2) <= 1:
            alpha = 100
        else:
            alpha = np.sqrt(np.log(coeff * (num_iter**2))/num_iter)
        return alpha
    
    def is_valid_demo(self, id_best_arm):
        is_valid = id_best_arm == self.true_best_arm
        return is_valid
    
    
    def get_gap_est(self, tau=None, delta_est=0.1, beta_est=0.05):
        """

        Estimating the gap for one arm for SAE

        Estimator is 2 * \alpha(t): 

        \alpha(t) = \sqrt{\frac{ln(c*n*t^2/\delta)}{t}}, c*n = beta

        Args:
            Ellimination time for that arm, tau
            beta_est : c*n, where n is number of demonstrators, used by the demonstrators!
            delta_est : SAE specific constant, used by the demonstrators!

        Returns:
            alpha(t) for t = num_iter
        """
        if delta_est == 'cube':
            delta_est = 1/tau**3
        elif delta_est == 'square':
            delta_est = 1/tau**2
        coeff = beta_est / delta_est
        if coeff * (tau**2) <= 1:
            gap_est = 1
            print('WARNING: negative values encountered during estimation when computing alpha!')
        else:
            gap_est = 2*np.sqrt(np.log(coeff * (tau**2))/tau)
        return gap_est

