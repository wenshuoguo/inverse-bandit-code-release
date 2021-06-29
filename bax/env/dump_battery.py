import os
import numpy as np
from sim_with_seed import sim


def get_parameter_space(policy_file):

    policies = np.genfromtxt(policy_file,
            delimiter=',', skip_header=0)

    return policies[:, :3]

env_dir = 'src/bax_release/bax/env'
policy_file = os.path.join(env_dir, 'policies_all.csv')
all_arms = list(get_parameter_space(policy_file))
sigma = 164

modes = ['lo', 'med', 'hi']
for mode in modes:
    lifetime_dists = []
    for arm in all_arms:
        mean_lifetime = sim(arm[0], arm[1], arm[2], mode=mode, variance=False)
        lifetime_dists.append([mean_lifetime, sigma])
    lifetime_dists = np.array(lifetime_dists)
    print(lifetime_dists.shape, lifetime_dists.mean(axis=0))
    save_file = os.path.join(env_dir, "battery_" + mode + '.npy')
    np.save(save_file, lifetime_dists)


