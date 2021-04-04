import numpy as np
from numpy import sin
from scipy.special import loggamma, factorial
from numba import jit

from .special_funcs import poch
from .fminsol import calc_min_sol
from .constants import f_const


def find_K(nu, omega, aa, em, kappa, tau, eigen, ess=-2, nmax=100, M=1):
    """
    Compute K(nu) to be used in asymptotic coefficients.
    """
    # It looks like there is a symmetry in f values for K_nu and K_neg_nu.
    # TODO (AJ): check if there is a symmetry in general
    # We may only need to compute f once instead of twice!

    nhalf = nmax // 2

    epsilon, kappa, tau = f_const(aa, omega, em)
    fneg, fpos, f = calc_min_sol(eigen, nu, omega, aa, em, nmax=nmax, ess=ess, M=M)

    a = [1 - ess - 1j * epsilon - 1j * tau,
         2 * nu + 2, 2 * nu + 1]
    b = [nu + 1 - ess + 1j * epsilon,
         nu + 1 - ess - 1j * epsilon,
         nu + 1 - 1j * tau]

    gamma_prod_num = np.exp(loggamma(a))
    gamma_prod_denom = np.exp(loggamma(b))

    gamma_prod = np.prod(gamma_prod_num) / np.prod(gamma_prod_denom)

    prefactor = (
        2**(-ess) * (2 * epsilon * kappa)**(ess - nu) *
        np.exp(1j * epsilon * kappa) * gamma_prod
    )

    # pos_pre = np.zeros(nhalf, dtype=complex)
    a = [2 * nu + 1,
         nu + 1 + ess + 1j * epsilon,
         nu + 1 + 1j * tau]
    b = [nu + 1 - ess - 1j * epsilon,
         nu + 1 - 1j * tau]

    k_pos = np.arange(nhalf)
    num_poch = poch(a[0], k_pos) * poch(a[1], k_pos) * poch(a[2], k_pos)
    den_poch = poch(b[0], k_pos) * poch(b[1], k_pos)
    pos_pre = (
        (-1)**k_pos / factorial(k_pos) * num_poch / den_poch
    )

    # neg_pre = np.zeros(nhalf, dtype=complex)
    k_neg = np.arange(-nhalf + 1, 1)
    a = nu + 1 + ess - 1j * epsilon
    b = [2 * nu + 2,
         nu + 1 - ess + 1j * epsilon]
    num_poch = poch(a, k_neg)
    den_poch = poch(b[0], k_neg) * poch(b[1], k_neg)
    neg_pre = (-1.)**k_neg / factorial(-k_neg) * num_poch / den_poch

    # for k in range(-nhalf + 1, 1):  # upper bound not inclusive
    #     print(k)
    #     neg_pre[-k] = (
    #         (-1)**k / factorial(-k) * poch(a, k) / np.prod(poch(b, k))
    #     )
    # neg_pre = neg_pre[::-1]

    A = np.dot(fpos, pos_pre)
    B = np.dot(fneg, neg_pre)

    K_nu = prefactor * A / B

    return f, K_nu


def find_Aplus(f, epsilon, nu, ess=-2):
    pi = np.pi
    a = [-pi * epsilon / 2, pi / 2 * 1j * (nu + 1 - ess)]
    b = nu + 1 - ess + 1j * epsilon
    c = nu + 1 + ess - 1j * epsilon

    gamma_terms = np.exp(loggamma(b) - loggamma(c))

    A_plus = (
        2**(-1 + ess - 1j * epsilon) * np.prod(np.exp(a)) * gamma_terms *
        np.sum(f)
    )
    return A_plus


def find_Bin(nu, aa, omega, em, eigen, M=1, ess=-2, nmax=100):
    """
    Finds assymptotic values for the homogeneous Teukolsky equation.
    """
    pi = np.pi
    epsilon, kappa, tau = f_const(aa, omega, em)
    f, K_nu = find_K(nu, omega, aa, em, kappa, tau, eigen, ess=ess, nmax=nmax, M=M)

    _, K_negnu = find_K(-nu - 1, omega, aa, em, kappa, tau, eigen, ess=ess, nmax=nmax, M=M)

    A_plus = find_Aplus(f, epsilon, nu, ess=ess)

    B_in = (
        1 / omega * (K_nu - 1j * np.exp(-1j * pi * nu) *
                     sin(pi * (nu - ess + 1j * epsilon)) /
                     sin(pi * (nu + ess - 1j * epsilon)) *
                     K_negnu) * A_plus *
        np.exp(-1j * epsilon * (np.log(epsilon) - (1 - kappa) / 2))
    )
    return f, B_in

