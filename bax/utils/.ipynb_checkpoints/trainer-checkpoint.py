from bax.env import bandit_sae as bsae
from bax.env import bandit_ucb as bucb 

import numpy as np

import ray
import time 
from tqdm import tqdm

#  

ALGO = {
    "sae" : bsae,
    "ucb" : bucb
}


def gap_est_SAE(tau=None, delta_est=0.1, beta_est=0.05):
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




def train_helper(prior=None, true_best_arm=0,
                T=2500, beta=0.05,delta=0.1,
                num_demos_max=100, num_runs=2,
                algo='sae', 
                scale_fn=None, ci_fn=None,
                estimate_at='after_switch'):
    '''
    Helper function for runing multiple demonstrators in multiple runs
    
    Return:
    taus: nparray with shape num_runs * num_demos_max * (K-1)
    
    '''
    logging_freq = max(1, num_demos_max // 10)
    
    algorithm = ALGO[algo]
    
    start_time = time.time()
    gaps_est, bias_est = [], []
    mu_gap_est = []
    mu_gap_t_est = []
    seed_order = []
    regret_t = []
    gaps_t   = []

    seed = 88

    for num_run in tqdm(range(num_runs)):
        # print('Start number of run: ', num_run)
        gaps_est_one_run = []
        gaps_t_one_run = []
        bias_est_one_run = []
        mu_gap_est_one_run = []
        mu_gap_t_one_run = np.zeros((T, len(prior)))
        num_demo = 0
        num_good_demo = 0
        id_refs = []
        seed_one_run = []
        regret_one_run = []
        
        while num_good_demo < num_demos_max:
            num_demos_to_collect = num_demos_max - num_good_demo
            for demo in np.arange(num_demos_to_collect):
                seed += 1
                demonstrator = algorithm.Demonstrator(prior=prior, T=T,
                                                      beta=beta, delta=delta, seed=seed, 
                                                      estimate_at=estimate_at, ci_fn=ci_fn)
                id_ref = algorithm.run_.remote(demonstrator)
                id_refs.append(id_ref)

            while len(id_refs) > 0:
                ready, not_ready = ray.wait(id_refs)
                if len(ready) > 0:
                    is_valid, id_gaps, obj = ray.get(ready[0])
                    if is_valid:

                        gaps_est_one_run.append(np.array(id_gaps))
                        bias_est_one_run.append(np.array(obj.bias))
                        seed_one_run.append(obj.seed)
                        mu_gap_est_one_run.append(np.array(obj.mu_gap))
                        regret_one_run.append(np.array(obj.regret_t))
                        gaps_t_one_run.append(np.array(obj.gaps_t))
                         # compute the avg mu_gap 
                        mu_gap_t_one_run += obj.mu_gap_t
                        num_good_demo += 1
                id_refs = not_ready
                
                # if num_good_demo % logging_freq == 0:
                #     print('Collected {} demos so far in {}'.format(num_good_demo, time.time()-start_time))
        
        argsort = np.argsort(seed_one_run)
        
        # sort by some order to make sure the results are reproducible
        mu_gap_t_est.append(mu_gap_t_one_run/num_good_demo)
        mu_gap_est.append(np.array(mu_gap_est_one_run)[argsort])
        bias_est.append(np.array(bias_est_one_run)[argsort])
        gaps_est.append(np.array(gaps_est_one_run)[argsort])
        gaps_t.append(np.array(gaps_t_one_run)[argsort])
        seed_order.append(seed_one_run)
        regret_t.append(np.array(regret_one_run))
    mu_gap_est = np.array(mu_gap_est).squeeze(1)
    gaps_est = np.array(gaps_est).squeeze(1)
    bias_est = np.array(bias_est).squeeze(1)
    seed_order = np.array(seed_order).squeeze(1)
    regret_t = np.array(regret_t).squeeze(1)
    gaps_t = np.array(gaps_t).squeeze(1)

    return {
        "gaps": gaps_est,
        "bias": bias_est,
        "mu_gap": mu_gap_est,
        "mu_gap_t": mu_gap_t_est,
        "seed_order": seed_order,
        "regret_t": regret_t,
        "gaps_t": gaps_t
    }



def train_helper_single_demo(prior=None, true_best_arm=0,
                T=2500, beta=0.05,delta=0.1,
                num_demos_max=100, num_runs=2,
                algo='sae', 
                scale_fn=None, ci_fn=None,
                estimate_at='after_switch'):
    '''
    Helper function for runing multiple demonstrators in multiple runs
    
    Return:
    taus: nparray with shape num_runs * num_demos_max * (K-1)
    
    '''
    logging_freq = max(1, num_demos_max // 10)
    
    algorithm = ALGO[algo]
    
    start_time = time.time()
    gaps_est, bias_est = [], []
    mu_gap_est = []
    mu_gap_t_est = []
    seed_order = []
    regret_t = []
    gaps_t   = []

    seed = 88

    for num_run in tqdm(range(num_runs)):
        # print('Start number of run: ', num_run)
        gaps_est_one_run = []
        gaps_t_one_run = []
        bias_est_one_run = []
        mu_gap_est_one_run = []
        mu_gap_t_one_run = np.zeros((T, len(prior)))
        num_demo = 0
        num_good_demo = 0
        id_refs = []
        seed_one_run = []
        regret_one_run = []
        
        while num_good_demo < num_demos_max:
            num_demos_to_collect = num_demos_max - num_good_demo
            for demo in np.arange(num_demos_to_collect):
                seed += 1
                demonstrator = algorithm.Demonstrator(prior=prior, T=T,
                                                      beta=beta, delta=delta, seed=seed, 
                                                      estimate_at=estimate_at, ci_fn=ci_fn)
                id_ref = algorithm.run_.remote(demonstrator)
                id_refs.append(id_ref)

            while len(id_refs) > 0:
                ready, not_ready = ray.wait(id_refs)
                if len(ready) > 0:
                    is_valid, id_gaps, obj = ray.get(ready[0])
                    if is_valid:

                        gaps_est_one_run.append(np.array(id_gaps))
                        bias_est_one_run.append(np.array(obj.bias))
                        seed_one_run.append(obj.seed)
                        mu_gap_est_one_run.append(np.array(obj.mu_gap))
                        regret_one_run.append(np.array(obj.regret_t))
                        gaps_t_one_run.append(np.array(obj.gaps_t))
                         # compute the avg mu_gap 
                        mu_gap_t_one_run += obj.mu_gap_t
                        num_good_demo += 1
                id_refs = not_ready
                
                # if num_good_demo % logging_freq == 0:
                #     print('Collected {} demos so far in {}'.format(num_good_demo, time.time()-start_time))
        
        argsort = np.argsort(seed_one_run)
        
        # sort by some order to make sure the results are reproducible
        mu_gap_t_est.append(mu_gap_t_one_run/num_good_demo)
        mu_gap_est.append(np.array(mu_gap_est_one_run)[argsort])
        bias_est.append(np.array(bias_est_one_run)[argsort])
        gaps_est.append(np.array(gaps_est_one_run)[argsort])
        gaps_t.append(np.array(gaps_t_one_run)[argsort])
        seed_order.append(seed_one_run)
        regret_t.append(np.array(regret_one_run))
    mu_gap_est = np.array(mu_gap_est).squeeze(1)
    gaps_est = np.array(gaps_est).squeeze(1)
    bias_est = np.array(bias_est).squeeze(1)
    seed_order = np.array(seed_order).squeeze(1)
    regret_t = np.array(regret_t).squeeze(1)
    gaps_t = np.array(gaps_t).squeeze(1)

    return {
        "gaps": gaps_est,
        "bias": bias_est,
        "mu_gap": mu_gap_est,
        "mu_gap_t": mu_gap_t_est,
        "seed_order": seed_order,
        "regret_t": regret_t,
        "gaps_t": gaps_t
    }




def compute_gaps_with_n(taus_all=None, demo_checkpoint=10, 
                         delta_est=0.1, beta_est=0.05, demo_type = 'SAE',
                           true_gaps = [0.5]):
    '''
    Args:
    taus_matrix: nparray with shape num_runs * num_demos_max * (K-1)
    demo_checkpoints: list, checkpoints of different demo numbers that need an estimation error
    
    Returns:
    
    list of squared error (or squared inf norm) for each run, length = num_runs
    '''
    num_runs = taus_all.shape[0]
    max_num_demos = taus_all.shape[1]
    num_gaps = taus_all.shape[2]
    
    # print(num_runs, max_num_demos, num_gaps)
    
    if max_num_demos < demo_checkpoint:
        print('ERROR: demo_checkpoint exceeds available demos!')
        
    SE_list = []
    
    if demo_type == 'SAE':
        
        for run_iter in range(num_runs):
            
            # print('Start run_iter: ', run_iter)
            taus_one_run = taus_all[run_iter, :, :]
            
            taus_one_run_demos = taus_one_run[:demo_checkpoint]

            taus_one_run_est = []
            for arm in range(num_gaps):
                one_arm_est = 0
                for demo in range(demo_checkpoint):
                    
                    # if demo % 100==0:
                    #     print('Start upto demonstrator: ', demo)
                    
                    one_demo_one_arm_est = gap_est_SAE(tau=taus_one_run_demos[demo, arm], delta_est=delta_est, 
                                          beta_est=beta_est)
                    one_arm_est += one_demo_one_arm_est
                one_arm_est = float(one_arm_est)/float(demo_checkpoint)
                taus_one_run_est.append(one_arm_est)

            SE_one_run = (abs(np.array(taus_one_run_est) - np.array(true_gaps)).max())**2
            
            SE_list.append(SE_one_run)  
            
    else:
        print('DEMO type not supported yet!')
    return SE_list
            
            
def compute_all_checkpoints(taus_all=None, delta_est=0.1, beta_est=0.05, demo_type ='SAE',
                       true_gaps = [0.5], demo_checkpoints = [1, 10]):
    '''
    Return all squared errors (inf norm for multi arms): num_runs * num_checkpoints
    
    '''
    
    # print('Start init')
    #     print(demo_checkpoint)
        
    
    start_time = time.time()
    SE_all = []
    
    # print('Start')
    
   
    for demo_checkpoint in demo_checkpoints:
        
        # print('Start compute with demo_checkpoint: ', 
        #        demo_checkpoint, 'after time: ', time.time() - start_time)
        
        SE_list = compute_gaps_with_n(taus_all=taus_all, demo_checkpoint=demo_checkpoint, 
                         delta_est=delta_est, beta_est=beta_est, demo_type=demo_type,
                         true_gaps = true_gaps)
        
        SE_all.append(SE_list)
        
    SE_all = np.array(SE_all)
    SE_all = SE_all.T
    
    return SE_all    



def compute_all_gap_checkpoints(gaps_all=None, delta_est=0.1, beta_est=0.05, demo_type ='SAE',
                            true_gaps = [0.5], demo_checkpoints = [1, 10], ltype='max_abs_square'):
    '''
    Return all squared errors (inf norm for multi arms): num_runs * num_checkpoints
    
    '''

        
    
    start_time = time.time()
    SE_all = []
    
    for demo_checkpoint in demo_checkpoints:
        
        mean_across_demos = np.mean(gaps_all[:, :demo_checkpoint], axis=1)
        if ltype == 'max_abs_square':
            SE_list = np.max(np.abs(mean_across_demos - true_gaps.reshape(1, -1)), axis=-1)**2
        elif ltype == 'max':
            SE_list = np.max(mean_across_demos - true_gaps.reshape(1, -1), axis=-1)
        SE_all.append(SE_list)
        
    SE_all = np.array(SE_all)
    SE_all = SE_all.T
    
    return SE_all  # return error_all_sqr : (num_runs x num_checkpoints)



def compute_all_bias_checkpoints(bias_all=None, demo_checkpoints=[1, 10]):
    
    # bias_all : numruns x numdemos x numarms
    SE_all = []
    for demo_checkpoint in demo_checkpoints:
        # TODO fix this by setting the
        mean_across_demos = np.max(np.mean(bias_all[:, :demo_checkpoint, :], axis=1), axis=-1)**2
        SE_all.append(mean_across_demos)
        
    SE_all = np.array(SE_all)
    SE_all = SE_all.T
    
    return SE_all  # return error_all_sqr : (num_ x num_checkpoints)
