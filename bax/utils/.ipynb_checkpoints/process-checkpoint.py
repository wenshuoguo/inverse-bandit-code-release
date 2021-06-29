import numpy as np

def remove_nans(results):
    keys = results.keys()
    validkey = []
    for k in keys:
        hasnan = False
        for gkey, gval in results[k].items():
            if np.any(np.isnan(gval)):
                hasnan = True
                break
        if not hasnan:
            validkey.append(k)