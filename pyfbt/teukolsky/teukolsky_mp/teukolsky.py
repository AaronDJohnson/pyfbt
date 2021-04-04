import numpy as np
from time import time
from mpmath import *
from mpmath import hyp2f1 as hypgeo

from .find_bin import min_sol

mp.dps = 100
# TODO: find 2 new contiguous relations to remove calculation of hypgeo
#       from the derivative function. This should be easy to do:
#       the ones we need are F[-2, -2, 1] = F[-1, -1, 0] + F[-2, -2, 0]
#       and F[-1, -1, 1] = F[-1, -1, 0] + F[-2, -2, 0]


def F000(a, b, c, z, Fm1m10, Fm2m20):
    """
    Uses Gauss' contiguous relations to solve for 2F1(0, 0, 0)
    in terms of 2F1(-1, -1, 0) and 2F1(-2, -2, 0)
    """
    # a = a - 1
    # b = b - 1
    alpha = (-1 + a - c) * (-1 + b - c) * (-1 + a + b - c)
    beta = ((-2 + a + b - c) * (2 * (-1 + a) * (-1 + b) +
            (-1 + a) * (-1 + a) * z - (-1 + a) * c * (1 + z) +
            (-b + c) * (c + z - (-1 + b) * z)))
    gamma = ((-1 + a) * (-1 + b) * (-3 + a + b - c) * (-1 + z) * (-1 + z))
    # numer = (alpha * Fm2m20 - beta * Fm1m10)
    # denom = gamma
    # print('numer = ', alpha * Fm2m20)
    # print('denom = ', beta * Fm1m10)
    ratio = (Fm1m10 / Fm2m20)
    # print(ratio)
    prefactor = - 1 / gamma * (alpha - beta * ratio)
    # print('prefactor = ', prefactor)
    # print('Fm2m20 = ', Fm2m20)
    res = prefactor * Fm2m20
    return res


def Fm2m2(a, b, c, z, F, Fm1m10):
    alpha = (-1 + a - c) * (-1 + b - c) * (-1 + a + b - c)
    beta = ((-2 + a + b - c) * (2 * (-1 + a) * (-1 + b) +
            (-1 + a) * (-1 + a) * z - (-1 + a) * c * (1 + z) +
            (-b + c) * (c + z - (-1 + b) * z)))
    gamma = ((-1 + a) * (-1 + b) * (-3 + a + b - c) * (-1 + z) * (-1 + z))

    ratio = (F / Fm1m10)

    prefactor = 1 / alpha * (beta - gamma * ratio)

    res = prefactor * Fm1m10
    return res


def F001(a, b, c, z, Fm1m11, Fm2m21):
    """
    Uses Gauss' contiguous relations to solve for 2F1(0, 0, 0)
    in terms of 2F1(-1, -1, 0) and 2F1(-2, -2, 0)
    """
    res = ((-2 + a + b - c)*(((-2 + a - c)*(2 - b + c)*
             (Fm2m21))/(-4 + a + b - c) - 
          (3 + c + a*(-2 + z) - ((-1 + a)*(-1 + a - c)*(-1 + z))/(-2 + a + b - c) + 
             ((-2 + a)*(2 - a + c)*(-1 + z))/(-4 + a + b - c) - b*z)*
           Fm1m11))/((-1 + a)*(-1 + b)*(-1 + z)**2)
    return res


def Fm2m21(a, b, c, z, F00p1, Fm1m11):
    res = ((-4 + a + b - c)*((-3 - c - a*(-2 + z) + 
            ((-1 + a)*(-1 + a - c)*(-1 + z))/(-2 + a + b - c) - 
             ((-2 + a)*(2 - a + c)*(-1 + z))/(-4 + a + b - c) + b*z)*
           Fm1m11 - 
          ((-1 + a)*(-1 + b)*(-1 + z)**2*F00p1)/
           (-2 + a + b - c)))/((2 - a + c)*(2 - b + c))
    return res


def R_2F1(r, f, nu, aa, omega, em, k_vec, ess=-2, nmax=100):
    """
    B.2 in Throwe's thesis

    46 digits correct compared to BHPTK (fix it!)
    """
    nhalf = nmax // 2
    kappa = sqrt(1 - aa**2)
    # print('kappa = ', kappa)  # good
    rp = 1 + kappa
    # print('rp = ', rp)  # good
    epsilon = 2 * omega
    # print('epsilon = ', epsilon)  # good
    tau = (epsilon - em * aa) / kappa
    # print(em)  # good
    # print(aa)  # good
    # print('tau = ', tau)  # good
    x = omega * (rp - r) / (epsilon * kappa)
    # check = x / omega * epsilon / 2  # good

    # exponents:
    a1 = 1j * epsilon * kappa * x
    a2 = -ess - 1j * (epsilon + tau) / 2
    a3 = -nu - 1 + 1j * (epsilon + tau) / 2

    c = 1 - ess - 1j * epsilon - 1j * tau
    d = -x / (1 - x)
    prefactor = exp(a1) * (-x)**a2 * (1 - x)**a3
    # print(exp(a1))  # good!
    # print((-check)**a2)
    # print(prefactor)

    a0 = nu + 1 - 1j * tau
    b0 = nu + 1 - ess - 1j * epsilon
    a = k_vec + nu + 1 - 1j * tau
    b = k_vec + nu + 1 - ess - 1j * epsilon
    # print(a0, b0, c, d)
    # print(r, nu, epsilon, kappa, tau, omega)
    hypgeo_vec = np.zeros(nmax - 1) * mpc("1")

    hypgeo_vec[nhalf - 3] = hyp2f1(a0 - 2, b0 - 2, c, d)
    hypgeo_vec[nhalf - 2] = hyp2f1(a0 - 1, b0 - 1, c, d)

    # print(a0 - 2, b0 - 2, c, d)
    # print(a0 - 1, b0 - 1, c, d)
    # print(hypgeo_vec[nhalf - 3])
    # print(hypgeo_vec[nhalf - 2])
    # hypgeo_vec[nhalf - 1] = hyp2f1(a, b, c, z)
    # vector = np.arange(-(nhalf - 1), nhalf)


    for i in range(nhalf):
        hypgeo_vec[nhalf - 1 + i] = F000(a0 + i, b0 + i, c, d,
                          hypgeo_vec[nhalf - 2 + i], hypgeo_vec[nhalf - 3 + i])

        if nhalf - 3 - i >= 0:
            hypgeo_vec[nhalf - 3 - i] = Fm2m2(a0 - i, b0 - i, c, d,
                          hypgeo_vec[nhalf - 1 - i], hypgeo_vec[nhalf - 2 - i])

    # for i in range(len(k_vec)):
    #     hypgeo_vec[i] = hypgeo(a[i], b[i], c, d)
    sum_pre = np.multiply(np.power(1 - x, -k_vec), hypgeo_vec)
    y = np.dot(f, sum_pre)

    Rin = prefactor * y
    return Rin, y, hypgeo_vec


def dR_2F1dr(r, f, nu, aa, omega, em, k_vec, ess=-2, nmax=100):
    """
    analytic derivative of B.2 in Throwe's thesis

    46 digits correct compared to BHPTK (fix it!)
    """
    nhalf = nmax // 2
    kappa = sqrt(1 - aa**2)
    rp = 1 + kappa
    epsilon = 2 * omega + 1j * 0
    tau = (epsilon - em * aa) / kappa
    x = omega * (rp - r) / (epsilon * kappa)

    Rin, y, hypgeo_vec = R_2F1(r, f, nu, aa, omega, em, k_vec, nmax=nmax)

    # exponents:
    a1 = 1j * epsilon * kappa
    a2 = -ess - 1j * (epsilon + tau) / 2
    a3 = -nu - 1 + 1j * (epsilon + tau) / 2
    c = 1 - ess - 1j * epsilon - 1j * tau
    d = -x / (1 - x)

    a0 = nu + 1 - 1j * tau
    b0 = nu + 1 - ess - 1j * epsilon
    a = k_vec + nu + 1 - 1j * tau  
    b = k_vec + nu + 1 - ess - 1j * epsilon
    hypgeo_vec2 = np.zeros(len(k_vec), dtype=complex) * mpc("1")
    # hypgeo_vec3 = np.zeros(len(k_vec), dtype=complex) * mpc("1")

    hypgeo_vec2[nhalf - 2] = hyp2f1(a0, b0, c + 1, d)
    hypgeo_vec2[nhalf - 3] = hyp2f1(a0 - 1, b0 - 1, c + 1, d)

    for i in range(nhalf):
        hypgeo_vec2[nhalf - 1 + i] = F001(a0 + 1 + i, b0 + 1 + i, c, d,
                         hypgeo_vec2[nhalf - 2 + i], hypgeo_vec2[nhalf - 3 + i])

        if nhalf - 3 - i >= 0:
            hypgeo_vec2[nhalf - 3 - i] = Fm2m21(a0 - i + 1, b0 - i + 1, c, d,
                         hypgeo_vec2[nhalf - 1 - i], hypgeo_vec2[nhalf - 2 - i])

    # for i in range(len(k_vec)):
    #     hypgeo_vec3[i] = hypgeo(1 + a[i], 1 + b[i], 1 + c, d)

    # print(np.divide(np.abs(hypgeo_vec3 - hypgeo_vec2), np.abs(hypgeo_vec3)))
    sum_pre = (-1 / c * (1 - x)**(-k_vec - 2) *
                (c * (x - 1) * (k_vec) * hypgeo_vec +
                 a * b * hypgeo_vec2))

    yd = np.dot(f, sum_pre)
    # print(yd)

    logR_prime = a1 + (a2 / x) + (a3 / (x - 1)) + (yd / y)

    Rdin = -omega / (epsilon * kappa) * logR_prime * Rin

    return Rin, Rdin


def potential(r, aa, em, omega, eigen, ess=-2, M=1):
    delta = r**2 - 2 * M * r + aa**2
    K = -aa * em + (aa**2 + r**2) * omega
    V = (eigen -
         4 * 1j * r * ess * omega -
         (-2 * 1j * (-M + r) * ess * K + K**2) / delta
         )
    return V


def dR_2F1dr2(r, f, nu, aa, omega, em, eigen, k_vec, ess=-2, nmax=100, M=1):
    """
    Uses the Teukolsky equation to find the second derivative
    """
    delta = r**2 - 2 * M * r + aa**2
    delta_prime = 2 * r - 2 * M
    V = potential(r, aa, em, omega, eigen)
    R, Rd = dR_2F1dr(r, f, nu, aa, omega, em, k_vec, nmax=nmax)
    Rdd = (R * V + Rd * delta_prime) / delta
    return R, Rd, Rdd


def find_Rin(r, f, nu, aa, omega, em, eigen, k_vec, nmax=100):
    Rin, Rd_in, Rdd_in = dR_2F1dr2(r, f, nu, aa, omega, em, eigen, k_vec, nmax=nmax)
    # print("r = ", r)
    # print("Rin = ", Rin)
    # print("Rd_in = ", Rd_in)
    # print("Rdd_in = ", Rdd_in)
    return Rin, Rd_in, Rdd_in


def find_R(r, nu, aa, omega, em, eigen, nmax=100):
    nhalf = nmax // 2
    k_vec = np.arange(-nhalf + 1, nhalf)
    fneg, fpos, f = min_sol(nu, aa, omega, em, eigen, nmax=nmax)
    Rin, Rd_in, Rdd_in = find_Rin(r, f, nu, aa, omega, em, eigen, k_vec, nmax=nmax)
    return Rin, Rd_in, Rdd_in


