from bax.env import bandit_sae_public as bsae
from bax.env import bandit_ucb_public as bucb 

import numpy as np
import pickle
import ray
import time 
from tqdm import tqdm

#  

ALGO = {
    "sae" : bsae,
    "ucb" : bucb
}


def train_helper_for_horizon(prior=None, true_best_arm=1,
                             Ts=[2500], num_runs=2,
                             checkpoint_freq=5, checkpoint=None, 
                             alpha=0, algo='sae', ci_fn=None):
    '''
    Helper function for runing multiple demonstrators on multiple runs
    
    Return:
        results :  dict('gaps', 'regret', 'seed')
    '''
    
    algorithm = ALGO[algo]
    
    start_time = time.time()
    gaps_est = []
    seed_order = []
    regret = []
    tidx = 0
    seed = 88

    for T in tqdm(Ts):
        tidx += 1
        id_refs = []
        gaps_est_one_run = []
        num_good_demo = 0
        seed_one_run = []
        regret_one_run = []
        
        while num_good_demo < num_runs:
            demos_to_run = num_runs - num_good_demo
            for newseed in np.arange(demos_to_run):
                seed += 1
                demonstrator = algorithm.Demonstrator(prior=prior, T=T, seed=seed, ci_fn=ci_fn(alpha))
                id_ref = algorithm.run_.remote(demonstrator)
                id_refs.append(id_ref)

            while len(id_refs) > 0:
                ready, not_ready = ray.wait(id_refs)
                if len(ready) > 0:
                    is_valid, id_gaps, obj = ray.get(ready[0])
                    if is_valid:
                        gaps_est_one_run.append(np.array(id_gaps))
                        seed_one_run.append(obj.seed)
                        regret_one_run.append(np.array(obj.regret))
                        num_good_demo += 1
                id_refs = not_ready


        argsort = np.argsort(seed_one_run)
        # sort by some order to make sure the results are reproducible
        gaps_est.append(np.array(gaps_est_one_run)[argsort])
        seed_order.append(np.sort(seed_one_run))
        regret.append(np.array(regret_one_run)[argsort])
        
        if checkpoint and tidx%checkpoint_freq == 0:
            basepath='results/'
            with open('{}alpha{:.2f}_maxT{}_numrun{}_{}'.format(basepath, alpha, T, num_runs, checkpoint), 'wb') as f:
                pickle.dump(
                     {"Ts": Ts[:tidx],
                      "gaps": gaps_est,
                      "regret": regret,
                      "seed_order": seed_order}, f)
    gaps_est = np.array(gaps_est)
    seed_order = np.array(seed_order)
    regret = np.array(regret)

    return {
        "gaps": gaps_est,
        "regret": regret,
        "seed_order": seed_order,
    }
