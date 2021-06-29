from collections import defaultdict


import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rc('font',family='Times New Roman')

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import numpy as np






def plot_one_delta(results_mean, results_CI, delta='delta = 0.1', fit=True, demo_checkpoints=None):
    
    plt.errorbar(demo_checkpoints, results_mean, results_CI, linewidth=3.0, label=delta)

    if fit == True:
        # find line of best fit 
        xs = np.log(demo_checkpoints)
        m, b = np.polyfit(np.log(demo_checkpoints), np.log(results_mean), 1)
        yhat = m* xs + b
        plt.plot(np.exp(xs), np.exp(yhat), '--', linewidth=2.0,
                 label=delta + ': best-fit with slope={:.4f}'.format(m))

        print(delta, ": LINE OF BEST FIT with slope{}".format(m))
        
        
        
def plot_per_beta(errors, demo_checkpoints=[1], num_runs=3):
    all_means = []
    all_CIs = []
    for delta, val in errors.items():
        all_means.append(np.mean(val, axis=0))
        all_CIs.append(np.std(val, axis=0)/float(np.sqrt(3)))

    fig = plt.figure(figsize=(10, 10))

    ax = plt.axes()
    ax.set_xscale("log")
    ax.set_yscale("log")

    delta_names = ['{:.2f}'.format(k) for k in list(errors.keys())]
    print(delta_names)
    for delta_iter in range(len(all_means)):
        plot_one_delta(all_means[delta_iter], all_CIs[delta_iter], delta='delta='+delta_names[delta_iter], 
                       demo_checkpoints=demo_checkpoints, fit=True)

    plt.title('2-arm SAE (Avg. over {} runs, multiple-delta'.format(num_runs))
    plt.xlabel('Number of demonstrators')
    plt.ylabel(r'MSE = $\| \Delta_n - \Delta \|^2$')
    plt.ylim([1E-6, 1])

    plt.tight_layout()
    plt.legend()

    
def plot_all_beta(errors, demo_checkpoints=[1], num_runs=3):
    all_means = []
    all_CIs = []
    for beta, val in errors.items():
        all_means.append(np.mean(val, axis=0))
        all_CIs.append(np.std(val, axis=0)/float(np.sqrt(num_runs)))

    fig = plt.figure(figsize=(10, 10))

    ax = plt.axes()
    ax.set_xscale("log")
    ax.set_yscale("log")

    beta_names = ['{:.2f}'.format(k) for k in list(errors.keys())]
    print(beta_names)
    for beta_iter in range(len(all_means)):
        plot_one_delta(all_means[beta_iter], all_CIs[beta_iter], delta='beta='+beta_names[beta_iter], 
                       demo_checkpoints=demo_checkpoints, fit=True)

    plt.title('2-arm SAE (Avg. over {} runs, multiple-beta'.format(num_runs))
    plt.xlabel('Number of demonstrators')
    plt.ylabel(r'MSE = $\| \Delta_n - \Delta \|^2$')
    plt.ylim([1E-6, 1])

    plt.tight_layout()
    plt.legend()

    
    
def plot_one_exp(results_mean, results_CI, bias_means=None, bias_CIs=None, label='delta = 0.1', fit=True, demo_checkpoints=None, ptype='bar'):
    """
    args:
        bias_means : dict of bias estimates (averaged across runs)
                     key is the name of estimator
        bias_CI : dict of bias_CI (averaged across runs)
    """
    
    if ptype == 'bar':
        plt.errorbar(demo_checkpoints, results_mean, results_CI, linewidth=3.0, label=label)
        for key, bmeans in bias_means.items():
            plt.errorbar(demo_checkpoints, bmeans, bias_CIs[key], linewidth=3.0, label=label + "_" + key)
    if ptype == 'fill':
        plt.plot(demo_checkpoints, results_mean, linewidth=2.0, label=label)
        plt.fill_between(demo_checkpoints, results_mean-results_CI, results_mean+results_CI, alpha=0.3)
        for key, bmeans in bias_means.items():
            plt.plot(demo_checkpoints, bmeans, linewidth=2.0, label=label+" "+key)
            plt.fill_between(demo_checkpoints, bmeans-bias_CIs[key], bmeans+bias_CIs[key], alpha=0.3)

    if fit == True:
        # find line of best fit 
        xs = np.log(demo_checkpoints)
        m, b = np.polyfit(np.log(demo_checkpoints), np.log(results_mean), 1)
        yhat = m* xs + b
        plt.plot(np.exp(xs), np.exp(yhat), '--', linewidth=2.0,
                 label=label + ': best-fit with slope={:.4f}'.format(m))

        print(label, ": LINE OF BEST FIT with slope{}".format(m))


def plot_all_exp(errors, bias=None, demo_checkpoints=[1], num_runs=3, label='delta', scale='log', ptype='bar', algo="SAE"):
    all_means = []
    all_CIs = []
    
    num_bias_estimators = len(bias.keys())
    
    bias_means = []
    bias_CIs = []

    for k, v in errors.items():
        num_runs = v.shape[0]
        all_means.append(np.mean(v, axis=0))
        all_CIs.append(np.std(v, axis=0)/float(np.sqrt(num_runs)))
        
        if bias: 
            _bias_mean_exp = {}
            _bias_CI_exp = {}
            bias_per_exp = bias[k]
            for bkey, bval in bias_per_exp.items():
                    num_runs = bval.shape[0]
                    _bias_mean_exp[bkey] = np.mean(bval, axis=0)
                    _bias_CI_exp[bkey] = np.std(bval, axis=0)/float(np.sqrt(num_runs))
            bias_means.append(_bias_mean_exp)
            bias_CIs.append(_bias_CI_exp)
            
    fig = plt.figure(figsize=(10, 10))

    ax = plt.axes()
    if scale == 'log':
        ax.set_xscale("log")
        ax.set_yscale("log")

    key_names = ['{:.2f}'.format(k) for k in list(errors.keys())]
    for itr in range(len(all_means)):
        plot_one_exp(all_means[itr], all_CIs[itr], 
                     bias_means=bias_means[itr], 
                     bias_CIs=bias_CIs[itr], 
                     label=label+'='+key_names[itr], 
                     demo_checkpoints=demo_checkpoints, 
                     fit=True, ptype=ptype)

    plt.title('2-arm {} (Avg. over {} runs, multiple-{}'.format(algo, num_runs, label))
    plt.xlabel('Number of demonstrators')
    plt.ylabel(r'MSE = $\| \Delta_n - \Delta \|^2$')

    plt.tight_layout()
    plt.legend()