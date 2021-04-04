from numpy import sqrt
from numba import jit


@jit(nopython=True)
def find_epsilon(omega, M=1):
    return 2 * M * omega


@jit(nopython=True)
def find_kappa(aa):
    return sqrt(1 - aa**2)


@jit(nopython=True)
def find_tau(epsilon, kappa, em, aa):
    return (epsilon - em * aa) / kappa


@jit(nopython=True)
def f_const(aa, omega, em, M=1):
    """
    Convenience function to return useful quantities
    """
    epsilon = find_epsilon(omega, M=1)
    kappa = find_kappa(aa)
    tau = find_tau(epsilon, kappa, em, aa)
    return epsilon, kappa, tau