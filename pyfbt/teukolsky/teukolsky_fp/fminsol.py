import numpy as np
from scipy.special import gamma, factorial, poch
from numba import jit

from .constants import f_const

@jit(nopython=True)
def f_alpha(n, kappa, tau, epsilon, nu, ess=-2):
    return ((1j * epsilon * kappa*(1 - 1j*epsilon + nu + (n + ess))*
             (1 + 1j*epsilon + nu + (n + ess))*(1 + n + nu + 1j*tau))/
               ((1 + n + nu)*(3 + 2*n + 2*nu)))


@jit(nopython=True)
def f_beta(n, epsilon, nu, eigen, aa, em, ess=-2):
    epsilon2 = epsilon * epsilon
    ess2 = ess * ess

    return (epsilon2 - ess*(1 + ess) + (n + nu)*(1 + n + nu) +
            epsilon*(epsilon - em*aa) + (epsilon*(epsilon2 + ess2)*(epsilon - em*aa))/
            ((n + nu)*(1 + n + nu)) - eigen)


@jit(nopython=True)
def f_gamma(n, kappa, tau, epsilon, nu, ess=-2):
    return (-(1j*epsilon*kappa*(-1j*epsilon - ess + n + nu)*
             (1j*epsilon - ess + n + nu)*(n + nu - 1j*tau))/
               ((n + nu)*(-1 + 2*n + 2*nu)))


@jit(nopython=True)
def complex_cont_frac(a, b, tol=1e-15):
    n = len(a)
    A = np.zeros(n + 1, dtype=np.complex128)
    B = np.zeros(n + 1, dtype=np.complex128)
    x = np.zeros(n + 1, dtype=np.complex128)
    A[0] = 1
    A[1] = b[0]
    B[0] = 0
    B[1] = 1
    x[1] = A[1] / B[1]

    A[2] = b[1] * A[1] + a[1] * A[0]
    B[2] = b[1] * B[1] + a[1] * B[0]
    x[2] = A[2] / B[2]

    A[3] = b[2] * A[2] + a[2] * A[1]
    B[3] = b[2] * B[2] + a[2] * B[1]
    x[3] = A[3] / B[3]

    for i in range(3, n - 1):
        A[i + 1] = b[i] * A[i] + a[i] * A[i - 1]
        B[i + 1] = b[i] * B[i] + a[i] * B[i - 1]
        x[i + 1] = A[i + 1] / B[i + 1]

        re_err = abs(np.real(x[i + 1]) - np.real(x[i]))
        im_err = abs(np.imag(x[i + 1]) - np.imag(x[i]))
        if re_err < tol and im_err < tol:
            return x[i + 1]


@jit(nopython=True)
def abg_f(k_vec, kappa, tau, epsilon, nu, eigen, aa, em, ess=-2, nmax=100):
    """
    Calculates alpha, beta, gamma in prep to calculate f_n.
    """
    # k_vec = np.arange(-nmax, nmax + 1)

    alpha = f_alpha(k_vec, kappa, tau, epsilon, nu, ess=ess)
    beta = f_beta(k_vec, epsilon, nu, eigen, aa, em, ess=ess)
    gamma = f_gamma(k_vec, kappa, tau, epsilon, nu, ess=ess)

    return alpha, beta, gamma


@jit(nopython=True)
def min_sol(alpha, beta, gamma, nmax=100):
    """
    Calculates the minimal solution f_n^{nu} and outputs it as a vector.
    This is used in several locations when finding the solution to the
    Teukolsky equation.
    """
    nhalf = nmax // 2

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
    zero_arr = np.array([0])

    ar = np.concatenate((zero_arr.ravel(), ar.ravel()))
    br = np.concatenate((zero_arr.ravel(), br.ravel()))

    # TODO (AJ): consider changing to only using R (1 / R_{n} ~ L_{n - 1})
    # this would make potentially more readable code and might save some
    # memory. Speed will likely be the same.

    R = np.zeros(nhalf, dtype=np.complex128)
    fpos = np.zeros(nhalf, dtype=np.complex128)
    fpos[0] = 1

    al = np.concatenate((zero_arr.ravel(), al.ravel()))
    bl = np.concatenate((zero_arr.ravel(), bl.ravel()))

    L = np.zeros(nhalf, dtype=np.complex128)
    fneg = np.zeros(nhalf, dtype=np.complex128)
    fneg[0] = 1

    for j in range(1, nhalf):
        R[j] = complex_cont_frac(ar, br)
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
    f = np.concatenate((fn2.ravel(), fpos.ravel()))
    return fneg, fpos, f


@jit(nopython=True)
def calc_min_sol(eigen, nu, omega, aa, em, nmax=100, ess=-2, M=1):
    k_vec = np.arange(-nmax, nmax + 1)
    epsilon, kappa, tau = f_const(aa, omega, em, M=M)
    alpha, beta, gamma = abg_f(k_vec, kappa, tau, epsilon, nu, eigen, aa, em, ess=ess, nmax=nmax)
    fneg, fpos, f = min_sol(alpha, beta, gamma, nmax=nmax)
    return fneg, fpos, f





