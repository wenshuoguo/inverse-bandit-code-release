from collections import defaultdict
import matplotlib 
import matplotlib.pyplot as plt
import numpy as np

def getrdict(defaults=list, depth=2):
    depth -= 1
    if depth:
        return defaultdict(lambda: getrdict(defaults, depth))
    else:
        return defaultdict(defaults)


def process_analysis(analysis, scope=['config.num_demos'], metric='rew'):
    """
    Takes an analysis from tune, and creates a more readable dictionary.
    """
    
    rdict = getrdict(depth=len(scope))
    for idx, val in enumerate(analysis.results_df[metric]):
        tdict = rdict
        for s in scope:
            tdict = tdict[analysis.results_df[s][idx]]
        tdict.append(val)
    return rdict


def deepplot(rdict, filters, metric, baseline, label='', xlabel=None, ylabel=None, title=None):
    xs, ys, yerr = [], [], []
    if len(filters) == 1:
        print(label)
        print(rdict.keys())
        for key, val in rdict.items():
            if np.any(np.isneginf(val)):
                continue
            xs.append(key)
            loss = (np.array(val) - baseline)**2
            ys.append(np.mean(loss))
            yerr.append(np.std(loss))
        ys = np.array(ys)
        yerr = np.array(yerr)
        print(yerr)
#         plt.loglog(xs, ys, linewidth=3.0, label='%s'%label)
        print(xs, ys)
        plt.errorbar(xs, ys, yerr=yerr, linewidth=3.0, label='%s'%label)    
    else:
        fig = plt.figure(figsize=(10, 8))
        for key, val in rdict.items():
            deepplot(val, filters[1:], metric, baseline, label=str(key))   
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()


def plot(analysis, scope=['config.num_demos'], metric='rew', baseline=0.7, plot='loglog'):
    """
    Plot the metrics 
    !!! currently supports only mse
    """
    rdict = process_analysis(analysis, scope, metric)
    xs, ys = [], []
    yerr = []
    for key, val in rdict.items():
        xs.append(key)
        loss = (np.array(val) - baseline)**2
        ys.append(np.mean(loss))
        yerr.append(np.std(loss))
        
    ax = plt.figure().gca()
    ax.set_ylabel('log')    
    if plot == 'errorbar':
        plt.errorbar(xs, ys, yerr,label='$\mu_2$=%s'%baseline)
    elif plot == 'loglog':
        plt.loglog(xs, ys, label='$\mu_2$=%s'%baseline)
    plt.xlabel('# demonstrators')
    plt.ylabel('mse for $\mu_2$')
    plt.legend()
    plt.title('IRL with SAE agents for 2-arm bandit')
    plt.savefig('plots/irl_sae_UCB.jpeg')