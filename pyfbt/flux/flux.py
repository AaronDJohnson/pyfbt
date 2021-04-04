import os, sys, inspect
import functools
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from geodesic.coordinates.coords_gen import calc_equatorial_coords
from geodesic.coordinates.coords_circ_eq import calc_circular_eq_coords
from source.tsource import eq_source
from teukolsky.functions import teukolsky_soln, calc_Bin_mp

from scipy.integrate import quad

import numpy as np

from mpmath import mp

# TODO (aaron): set up special case of circular orbits (no integral in this case)


def trapezoidal_rule(f, a, b, tol=1e-8):
    """
    The trapezoidal rule is known to be very accurate for
    oscillatory integrals integrated over their period.

    See papers on spectral integration (it's just the composite trapezoidal rule....)

    TODO (aaron): f is memoized to get the already computed points quickly.
    Ideally, we should put this into a C++ function and call it with Cython. (Maybe someday)
    """
    # endpoints first:
    num = 2
    dx = b - a
    res0 = 1e30
    res1 = 0.5 * dx * (f(b) + f(a))
    delta_res = res0 - res1
    re_err = np.abs(np.real(delta_res))
    im_err = np.abs(np.imag(delta_res))
    while re_err > tol or im_err > tol:
        res0 = res1
        num = 2 * num - 1
        # print(num)
        x = np.linspace(a, b, num=num)
        res = 0
        dx = (x[1] - x[0])
        res += f(x[0])
        for i in range(1, len(x) - 1):
            res += 2 * f(x[i])
        res += f(x[-1]) 
        res1 = 0.5 * dx * res
        delta_res = res1 - res0
        re_err = np.abs(np.real(delta_res))
        im_err = np.abs(np.imag(delta_res))
        if num > 100000:
            print('Integral failed to converge with', num, 'points.')
            return np.nan, np.nan, np.nan
    return res1, re_err, im_err


def eq_find_z(nu, eigen, slr, ecc, aa, x, ups_r, ups_theta, ups_phi, gamma,
              omega, em, Lz, En, Slm, Slmd, Slmdd, omega_r, r1, r2, r3, r4, zp,
              zm, ess=-2, M=1, tol=1e-12):

    # print(type(nu))
    # print(type(Bin))
    # print(type(eigen))
    # print(type(slr))
    # print(type(ecc))
    # print(type(aa))
    # print(type(omega))
    # print(type(Lz))
    # print(type(En))
    # print(type(omega_r))
    # print(type(r1))
    # print(type(zp))

    @functools.lru_cache()
    def find_psi_integrand(psi):
        t, r, __, phi = calc_equatorial_coords(
                                                psi,
                                                ups_r,
                                                ups_theta,
                                                ups_phi,
                                                gamma,
                                                r1,
                                                r2,
                                                r3,
                                                r4,
                                                zp,
                                                zm,
                                                En,
                                                Lz,
                                                aa,
                                                slr,
                                                ecc,)
        # print(r, nu, eigen, aa, omega, em)
        Rin, dRdr, dRdr2 = teukolsky_soln(r, nu, eigen, aa, omega, em, ess=ess, M=M, tol=tol)
        # if np.isnan(np.real(Rin)):
        #     print('r =', r)
        #     print('Rin =', Rin)
        #     print('dRdr =', dRdr)
        #     print('dRdr2 =', dRdr2)
        J, V_t, V_r, I_plus = eq_source(psi, 1, slr, ecc, aa, omega, em, Lz,
                                        En, Slm, Slmd, Slmdd, Rin, dRdr, dRdr2)
        _, _, _, I_minus = eq_source(psi, -1, slr, ecc, aa, omega, em, Lz,
                                     En, Slm, Slmd, Slmdd, Rin, dRdr, dRdr2)
        result = (
            V_t / (J * np.sqrt(V_r)) *
            (I_plus * np.exp(1j * omega * t - 1j * em * phi) +
             I_minus * np.exp(-1j * omega * t + 1j * em * phi))
        )
        return result


    # @functools.lru_cache()
    def find_integrand_ce(psi):
        t, r, __, phi = calc_circular_eq_coords(psi, En, Lz, aa, slr, M=M)
        # Rin = R_vec[0]
        # dRdr = R_vec[1]
        # dRdr2 = R_vec[2]
        # Rin, dRdr, dRdr2 = teukolsky_soln(r, nu, eigen, aa, omega, em, ess=ess, M=M, tol=tol)
        # if np.isnan(np.real(Rin)):
        #     print('r =', r)
        #     print('Rin =', Rin)
        #     print('dRdr =', dRdr)
        #     print('dRdr2 =', dRdr2)
        J, V_t, V_r, I_plus = eq_source(psi, 1, slr, ecc, aa, omega, em, Lz,
                                        En, Slm, Slmd, Slmdd, Rin, dRdr, dRdr2)
        _, _, _, I_minus = eq_source(psi, -1, slr, ecc, aa, omega, em, Lz,
                                     En, Slm, Slmd, Slmdd, Rin, dRdr, dRdr2)
        result = (
            V_t / (J * np.sqrt(V_r)) *
            (I_plus * np.exp(1j * omega * t - 1j * em * phi) +
             I_minus * np.exp(-1j * omega * t + 1j * em * phi))
        )
        return result

    

    # coeff = abs(find_psi_integrand(0)) / abs(find_psi_integrand(np.pi))
    # print(coeff)

    # def integrand(zeta):
    #     psi = coeff * np.log(1 + zeta / coeff)
    #     dchi_dzeta = np.exp(-psi / coeff)

    #     t, r, __, phi = py_calc_equatorial_coords(psi, ups_r, ups_theta,
    #                                               ups_phi, gamma, r1,
    #                                               r2, r3, r4, zp, zm, En, Lz,
    #                                               aa, slr, ecc)

    #     re_nu = np.real(nu)
    #     im_nu = np.imag(nu)
    #     Rin, dRdr, dRdr2 = py_find_R(r, re_nu, im_nu, aa, omega, em, eigen)
    #     J, V_t, V_r, I_plus = eq_source(psi, 1, slr, ecc, aa, omega, em, Lz,
    #                                     En, Slm, Slmd, Slmdd, Rin, dRdr, dRdr2)
    #     _, _, _, I_minus = eq_source(psi, -1, slr, ecc, aa, omega, em, Lz,
    #                                  En, Slm, Slmd, Slmdd, Rin, dRdr, dRdr2)

    #     result = (
    #         V_t / (J * np.sqrt(V_r)) *
    #         (I_plus * np.exp(1j * omega * t - 1j * em * phi) +
    #          I_minus * np.exp(-1j * omega * t + 1j * em * phi))
    #     ) * dchi_dzeta
    #     # print(result)
    #     return result

    # def integrand_re(zeta):
    #     return np.real(integrand(zeta))

    # def integrand_im(zeta):
    #     return np.imag(integrand(zeta))

    def ce_re(zeta):
        return np.real(find_integrand_ce(zeta))

    def ce_im(zeta):
        return np.imag(find_integrand_ce(zeta))

    # a = 0
    # mp.dps += 50
    # TODO: fix overflow in the following line
    # b = float((np.exp(np.pi / coeff) - 1) * coeff)
    # print(b)
    # print("b = ", b)
    # re_res, re_err = quad(integrand_re, a, b)
    # im_res, im_err = quad(integrand_im, a, b)

    
    if ecc == 0 and x**2 == 1:
        r = slr
        Rin, dRdr, dRdr2 = teukolsky_soln(r, nu, eigen, aa, omega, em, ess=ess, M=M, tol=tol)
        re_res, re_err = quad(ce_re, 0, np.pi)
        im_res, im_err = quad(ce_im, 0, np.pi)
        # res2, __, __ = trapezoidal_rule(find_integrand_ce, 0, np.pi)
        # print(res2)
        res = re_res + 1j * im_res
        # print(res)

    else:
        res, re_err, im_err = trapezoidal_rule(find_psi_integrand, 0, np.pi)

    # print('Error in real part of integral:', re_err)
    # print('Error in imag part of integral:', im_err)

    # re_res = romberg(integrand_re, a, b, divmax=20)
    # im_res = romberg(integrand_im, a, b, divmax=20)
    Bin = calc_Bin_mp(nu, aa, omega, em, eigen)
    Bin = complex(Bin)
    # print(Bin)
    # Z = omega_r / (2 * 1j * omega * Bin) * (re_res + 1j * im_res)
    Z = omega_r / (2 * 1j * omega * Bin) * res
    return Z


def flux_inf(nu, eigen, slr, ecc, aa, x, ups_r, ups_theta, ups_phi,
             gamma, omega, em, Lz, En, Slm, Slmd, Slmdd, omega_r, r1, r2,
             r3, r4, zp, zm, ess=-2):
    Z = eq_find_z(nu, eigen, slr, ecc, aa, x, ups_r, ups_theta, ups_phi, gamma,
                    omega, em, Lz, En, Slm, Slmd, Slmdd, omega_r, r1, r2, r3, r4, zp,
                    zm, ess=ess)
    energy = abs(Z)**2 / (4 * np.pi * omega**2)
    return energy, Z
