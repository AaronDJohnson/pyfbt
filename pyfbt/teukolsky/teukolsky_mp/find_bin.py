import numpy as np
from mpmath import *

from .contfrac import complex_cont_frac

mp.dps = 100

def abg_nu(nu, aa, omega, em, eigen, ess=-2, nmax=100):
    """
    Calculates the FT recurrence relations in prep to calculate renormalized nu
    """
    epsilon = 2 * omega
    epsilon2 = epsilon**2
    kappa = (1 - aa**2)**(1 / 2)
    tau = (epsilon - em * aa) / kappa
    k_vec = np.arange(-nmax, nmax + 1)

    eta = k_vec + nu
    epo = eta + 1

    alpha = (
            1j * epsilon * kappa * (epo + ess + 1j * epsilon) *
            (epo + ess - 1j * epsilon) *
            (epo + 1j * tau) / (epo * (2 * eta + 3)))

    beta = (
            - eigen - ess * (ess + 1) + (eta) * epo +
            epsilon2 + epsilon * (epsilon - em * aa) +
            epsilon * (epsilon - em * aa) * (ess**2 + epsilon2) /
            (eta * epo))

    gamma = (
            - 1j * epsilon * kappa * (eta - ess + 1j * epsilon) *
            (eta - ess - 1j * epsilon) * (eta - 1j * tau) /
            (eta * (2 * eta - 1))
        )
    return alpha, beta, gamma


def min_sol(nu, aa, omega, em, eigen, nmax=100):
    """
    Calculates the minimal solution f_n^{nu} and outputs it as a vector.
    This is used in several locations when finding the solution to the
    Teukolsky equation.
    """
    nhalf = nmax // 2
    alpha, beta, gamma = abg_nu(nu, aa, omega, em, eigen, nmax=nmax)

    alpha_pos = alpha[nmax:-1]
    gamma_pos = gamma[nmax + 1:]

    ar = -alpha_pos * gamma_pos
    br = beta[nmax + 1:]

    alpha_neg = alpha[:nmax]
    gamma_neg = gamma[1:nmax + 1]

    alpha_neg = alpha_neg[::-1]
    gamma_neg = gamma_neg[::-1]

    al = -alpha_neg * gamma_neg
    bl = beta[:nmax]
    bl = bl[::-1]

    # prepend zeros so that the first value seen by cont_frac is zero for
    # a or b vectors. b0 + K(a1 / b1...) where b0 and a0 = 0.
    ar = np.insert(ar, 0, mpc("0"), axis=0)
    br = np.insert(br, 0, mpc("0"), axis=0)

    R = np.zeros(nhalf, dtype=complex) * mpc("1")
    fpos = np.zeros(nhalf, dtype=complex) * mpc("1")
    fpos[0] = mpc("1")

    al = np.insert(al, 0, mpc("0"), axis=0)
    bl = np.insert(bl, 0, mpc("0"), axis=0)

    L = np.zeros(nhalf, dtype=complex) * mpc("1")
    fneg = np.zeros(nhalf, dtype=complex) * mpc("1")
    fneg[0] = mpc("1")

    for j in range(1, nhalf):
        # print(ar, br)
        R[j] = complex_cont_frac(ar, br)
        # print(R[j])
        # print(fpos[j - 1])
        fpos[j] = R[j] * fpos[j - 1] / alpha_pos[0]

        ar = np.delete(ar, 1)
        br = np.delete(br, 1)
        alpha_pos = np.delete(alpha_pos, 0)

        L[j] = complex_cont_frac(al, bl)
        fneg[j] = L[j] * fneg[j - 1] / gamma_neg[0]

        al = np.delete(al, 1)
        bl = np.delete(bl, 1)
        gamma_neg = np.delete(gamma_neg, 0)

    fn2 = np.delete(fneg, 0)
    fn2 = fn2[::-1]
    fneg = fneg[::-1]
    f = np.concatenate([fn2, fpos])
    return fneg, fpos, f


def find_K(nu, nmax, aa, omega, em, eigen, ess=-2):
    """
    It looks like there is a symmetry in f values for K_nu and K_neg_nu.
    #TODO: check this
    """
    nhalf = nmax // 2
    epsilon = 2 * omega + 0j
    kappa = sqrt(1 - aa**2)
    tau = (epsilon - em * aa) / kappa

    fneg, fpos, f = min_sol(nu, aa, omega, em, eigen, nmax=nmax)
    # print(fneg)
    # print(fpos)

    a = [1 - ess - 1j * epsilon - 1j * tau,
         2 * nu + 2, 2 * nu + 1]
    b = [nu + 1 - ess + 1j * epsilon,
         nu + 1 - ess - 1j * epsilon,
         nu + 1 - 1j * tau]
    prefactor = (
        2**(-ess) * (2 * epsilon * kappa)**(ess - nu) *
        exp(1j * epsilon * kappa) * gammaprod(a, b)
    )
    # prefactor = mpc(prefactor)
    # print('prefactor =' + str(prefactor))
    # print('nu = ' + str(nu))

    pos_pre = np.zeros(nhalf, dtype=complex) * mpc("1")
    for k in range(nhalf):
        pos_pre[k] = (
            (-1)**k / factorial(k) * rf(2 * nu + 1, k) *
            rf(nu + 1 + ess + 1j * epsilon, k) *
            rf(nu + 1 + 1j * tau, k) /
            (rf(nu + 1 - ess - 1j * epsilon, k) *
             rf(nu + 1 - 1j * tau, k))
        )
        pos_pre[k] = mpc(pos_pre[k])
    # print(pos_pre)

    neg_pre = np.zeros(nhalf, dtype=complex) * mpc("1")
    for k in range(-nhalf + 1, 1):  # upper bound not inclusive
        neg_pre[-k] = (
            (-1)**k / factorial(-k) * rf(nu + 1 + ess - 1j * epsilon, k) /
            (rf(2 * nu + 2, k) * rf(nu + 1 - ess + 1j * epsilon, k))
        )
    neg_pre = neg_pre[::-1]
    # print(fneg)

    A = np.dot(fpos, pos_pre)
    B = np.dot(fneg, neg_pre)

    K_nu = prefactor * A / B

    # print('A = ' + str(A))
    # print('B = ' + str(B))

    # print('K_nu = ' + str(K_nu))
    return f, K_nu


def find_Bin(nu, nmax, aa, omega, em, eigen, ess=-2):
    """
    Finds assymptotic values for the homogeneous Teukolsky equation.
    """
    epsilon = 2 * omega + mpc("0")
    kappa = sqrt(1 - aa**2)
    f, K_nu = find_K(nu, nmax, aa, omega, em, eigen)
    # print('K_nu = ' + str(K_nu))
    _, K_negnu = find_K(-nu - 1, nmax, aa, omega, em, eigen)
    # print('K_negnu = ' + str(K_negnu))
    A_plus = (
        2**(-1 + ess - 1j * epsilon) * exp(-pi * epsilon / 2) *
        exp(pi / 2 * 1j * (nu + 1 - ess)) *
        gammaprod([nu + 1 - ess + 1j * epsilon],
                  [nu + 1 + ess - 1j * epsilon]) *
        sum(f)
    )
    # A_plus = complex(A_plus)
    # print('f_sum = ', sum(f))
    # print('A_plus = ', A_plus)
    B_in = (
        1 / omega * (K_nu - 1j * exp(-1j * pi * nu) *
                     sin(pi * (nu - ess + 1j * epsilon)) /
                     sin(pi * (nu + ess - 1j * epsilon)) *
                     K_negnu) * A_plus *
        exp(-1j * epsilon * (log(epsilon) - (1 - kappa) / 2))
    )
    # B_in = complex(B_in)
    # print('B_in = ' + str(B_in))
    return f, B_in

