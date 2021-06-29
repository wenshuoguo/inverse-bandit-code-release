import matplotlib.pyplot as plt
import numpy as np


fontstyles = {
              0.15 : ['dashdot', 'blue', 'darkblue'],
              0.25 : ['solid', 'gray', 'black'],
             }

def get_mean_ci(rdict, arm=None):
    mdict, cdict = {}, {}
    for k, v in rdict.items():
        if arm is not None:
            v = v[:, :, arm]
            print(v.shape)
        num_runs = np.array(v).shape[1]
        print(np.array(v).shape)
        mdict[k] = np.mean(v, axis=1)
        cdict[k] = np.std(v, axis=1) / np.sqrt(num_runs)
    return mdict, cdict 


def get_true_gaps(prior):
    means = prior[:, 0]
    max_mean = np.max(means)
    true_gaps = max_mean - means
    print(true_gaps)
    return true_gaps 


def get_mse(pred_gap, true_gap, norm="infty"):
    """
    Args:
        pred_gap : shape #runs x #horizon # arms 
        true_gap : #arms 
        norm = choice in ['infty']
    """
    errors_all = {}
    for k, est_gap in pred_gap.items():
        if k == 0.0:
            continue
        error_per_arm = est_gap - true_gap 
        sqerr_per_arm = error_per_arm ** 2
        print(sqerr_per_arm.shape)
        errors_all[k] = sqerr_per_arm
    return errors_all 


def plot_regret(Ts, results, yticks=[20, 50, 100]):
    figure = plt.figure(figsize=(24, 24))
    fontsize = 88
    
    regret = {k: v['regret'] for k, v in results.items()}
    mdict, cdict = get_mean_ci(regret)
    for alpha, mean in mdict.items():       
        linestyle, facecolor, linecolor = fontstyles[alpha]
        plt.plot(Ts, mean, linewidth=24,  linestyle=linestyle, color=linecolor,  label=r'$\alpha$={:.2f}'.format(alpha))
        plt.fill_between(Ts, mean-cdict[alpha], mean-cdict[alpha], alpha=0.2, facecolor=facecolor)

    plt.legend(fontsize=fontsize)

    plt.xticks([500, 2500, 5000], [500, 2500, 5000], fontsize=fontsize-8)
    plt.yticks(yticks, yticks, fontsize=fontsize-8)
    plt.xlabel(r'Horizon $T$', fontsize=fontsize)
    plt.ylabel(r'Regret $R_T$', fontsize=fontsize)
    plt.tight_layout()

    
def plot_mse(Ts, results, prior, ylabel=r'MSE  $\mathbb{E} | \hat{\mu}_2 - \mu_2 |^2$'):
    figure = plt.figure(figsize=(24, 24))
    fontsize = 88
    
    
    true_gap = get_true_gaps(prior)
    gaps_all = {k: v['gaps'] for k, v in results.items()}
    errors_all = get_mse(gaps_all, true_gap)
    emean, eci = get_mean_ci(errors_all, arm=0)

    for alpha, mean in emean.items():
        linestyle, facecolor, linecolor = fontstyles[alpha]
        plt.plot(Ts, mean, linewidth=24,  linestyle=linestyle, color=linecolor, label=r'$\alpha$={:.2f}'.format(alpha))
        plt.fill_between(Ts, mean-eci[alpha], mean+eci[alpha], alpha=0.2, facecolor=facecolor)
    
        plt.legend(fontsize=fontsize)

        m, b = np.polyfit(np.log(Ts), np.log(mean), 1)
        yhat = np.exp(m * np.log(Ts) + b)
        plt.plot(Ts, yhat, linewidth=24, linestyle='--', color='black')
        
        
    plt.xscale("log")
    plt.yscale("log")
    plt.xticks([500, 2500, 5000], [500, 2500, 5000], fontsize=fontsize-8)
    plt.yticks([4e-3, 6e-3, 1e-2, 5e-2], [r"$4x10^{-3}$", r"$6x10^{-3}$", r"$10^{-2}$", r"$5x10^{-2}$"], fontsize=fontsize-8)
    plt.xlabel(r'Horizon $T$', fontsize=fontsize)
    plt.ylabel(ylabel, fontsize=fontsize)
    plt.tight_layout()