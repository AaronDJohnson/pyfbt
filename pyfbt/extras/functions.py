import numpy as np
import matplotlib.pyplot as plt


def en_search(orbit, ell, em, en_min, en_max):
    res_list = []
    for en in range(en_min, en_max + 1):
        E_inf = orbit.energy_inf(ell=ell, em=em, en=en, tol=1e-7)
        results = [ell, em, en, E_inf]
        print(results)
        res_list.append(results)
    results = np.array(res_list)
    return results


def plot_modes(results, normed=True, enmax=True):
    """
    plot results from en_search
    """
    if normed:
        plt.bar(results[:, 2], results[:, 3]/ np.sum(results[:, 3]))
    else:
        plt.bar(results[:, 2], results[:, 3])
    plt.show()
