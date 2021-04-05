import numpy as np
from mpmath import re, im

from .teukolsky_fp.teukolsky import calc_R
from .teukolsky_mp.teukolsky import find_R
# from .teukolsky_fp.asymptotic import find_Bin as find_Bin_fp
from .teukolsky_mp.find_bin import find_Bin as find_Bin_mp


# def teukolsky_soln_fp(r, nu, eigen, aa, omega, em, ess=-2, M=1, tol=1e-12):
#     nmax = 100
#     Rin, Rdin, Rddin = calc_R(r, nu, eigen, aa, omega, em, nmax=nmax, ess=ess, M=M)
#     R0 = np.array([Rin, Rdin, Rddin])
#     R0_re = np.real(R0)
#     R0_im = np.imag(R0)

#     nmax = 200
#     Rin, Rdin, Rddin = calc_R(r, nu, eigen, aa, omega, em, nmax=nmax, ess=ess, M=M)
#     R1 = np.array([Rin, Rdin, Rddin])
#     R1_re = np.real(R1)
#     R1_im = np.imag(R1)

#     re_err = np.max(R1_re - R0_re)
#     im_err = np.max(R1_im - R0_im)
#     while re_err > tol or im_err > tol:
#         nmax = 2 * nmax
#         R0 = R1
#         R0_re = np.real(R0)
#         R0_im = np.imag(R0)

#         Rin, Rdin, Rddin = calc_R(r, nu, eigen, aa, omega, em, nmax=nmax, ess=ess, M=M)
#         R1 = np.array([Rin, Rdin, Rddin])
#         R1_re = np.real(R1)
#         R1_im = np.imag(R1)

#         re_err = np.max(R1_re - R0_re)
#         im_err = np.max(R1_im - R0_im)

#     return R1[0], R1[1], R1[2]


# def teukolsky_soln_mp(r, nu, eigen, aa, omega, em, ess=-2, M=1, tol=1e-12):
#     nmax = 100
#     Rin, Rdin, Rddin = find_R(r, nu, aa, omega, em, eigen, nmax=nmax)
#     R0 = np.array([Rin, Rdin, Rddin])
#     R0_re = [re(R0[0]), re(R0[1]), re(R0[2])]
#     R0_im = [im(R0[0]), im(R0[1]), im(R0[2])]

#     nmax = 200
#     Rin, Rdin, Rddin = find_R(r, nu, aa, omega, em, eigen, nmax=nmax)
#     R1 = np.array([Rin, Rdin, Rddin])
#     R1_re = np.array([re(R1[0]), re(R1[1]), re(R1[2])])
#     R1_im = np.array([im(R1[0]), im(R1[1]), im(R1[2])])

#     # print(R1_re)
#     re_err = np.max(R1_re - R0_re)
#     im_err = np.max(R1_im - R0_im)
#     while re_err > tol or im_err > tol:
#         nmax = 2 * nmax
#         R0 = R1
#         R0_re = np.array([re(R0[0]), re(R0[1]), re(R0[2])])
#         R0_im = np.array([im(R0[0]), im(R0[1]), im(R0[2])])

#         Rin, Rdin, Rddin = find_R(r, nu, aa, omega, em, eigen, nmax=nmax)
#         R1 = np.array([Rin, Rdin, Rddin])
#         R1_re = np.array([re(R1[0]), re(R1[1]), re(R1[2])])
#         R1_im = np.array([im(R1[0]), im(R1[1]), im(R1[2])])

#         re_err = np.max(R1_re - R0_re)
#         im_err = np.max(R1_im - R0_im)

#     return complex(R1[0]), complex(R1[1]), complex(R1[2])


def teukolsky_soln(r, nu, eigen, aa, omega, em, ess=-2, M=1, nmax=100):
    Rin, Rdin, Rddin = calc_R(r, nu, eigen, aa, omega, em, nmax=nmax, ess=ess, M=M)
    if np.isnan(Rin) or np.isnan(Rdin) or np.isnan(Rddin):
        Rin, Rdin, Rddin = find_R(r, nu, aa, omega, em, eigen, nmax=nmax)
        Rin = complex(Rin)
        Rdin = complex(Rdin)
        Rddin = complex(Rddin)
    return Rin, Rdin, Rddin


# def calc_Bin_fp(nu, aa, omega, em, eigen, M=1, ess=-2, tol=1e-12):
#     nmax = 100
#     Bin0 = 1e30
#     __, Bin1 = find_Bin_fp(nu, aa, omega, em, eigen, M=M, ess=ess, nmax=nmax)
#     re_err = np.real(Bin1) - Bin0
#     im_err = np.imag(Bin1)
#     while re_err > tol or im_err>tol:
#         nmax *= 2
#         Bin0 = Bin1
#         print(Bin1)
#         __, Bin1 = find_Bin_fp(nu, aa, omega, em, eigen, M=M, ess=ess, nmax=nmax)
#         delta_Bin = Bin1 - Bin0
#         re_err = np.real(delta_Bin)
#         im_err = np.imag(delta_Bin)
#     return Bin1
    

def calc_Bin_mp(nu, aa, omega, em, eigen, ess=-2, tol=1e-12):
    nmax = 100
    Bin0 = 1e30
    __, Bin1 = find_Bin_mp(nu, nmax, aa, omega, em, eigen, ess=ess)
    re_err = re(Bin1) - Bin0
    im_err = im(Bin1)
    while re_err > tol or im_err>tol:
        nmax *= 2
        Bin0 = Bin1
        __, Bin1 = find_Bin_mp(nu, nmax, aa, omega, em, eigen, ess=ess)
        delta_Bin = Bin1 - Bin0
        re_err = re(delta_Bin)
        im_err = im(delta_Bin)
    return nmax, Bin1


# def calc_Bin(nu, aa, omega, em, eigen, M=1, ess=-2, tol=1e-12):
#     Bin = calc_Bin_fp(nu, aa, omega, em, eigen, M=M, ess=ess, tol=tol)
#     if np.isnan(Bin):
#         Bin = calc_Bin_mp(nu, aa, omega, em, eigen, ess=ess, tol=tol)
#         Bin = float(Bin[0]) + 1j * float(Bin[1])
#     return Bin


