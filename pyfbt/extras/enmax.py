import numpy as np

from geodesic.geodesic import calc_boyer_freqs, calc_consts


def radius(psi, slr, ecc):
    return slr / (1 + ecc * np.cos(psi))


def find_en_max(aa, slr, ecc, x, em):
    """
    The following is from Oohara 1984. It will calculate kmax near the separatrix.
    It relies on the average location of the radius being close to the
    black hole when near the separatrix.
    """
    En, Lz, Q = calc_consts(aa, slr, ecc, x)
    rp = radius(0, slr, ecc)  # peribothron
    omega_r, omega_theta, omega_phi = calc_boyer_freqs(aa, slr, ecc, x, M=1)


    T = En * (rp**2 + aa**2) - Lz * aa
    Delta = rp**2 - 2 * rp + aa**2
    omega_0 = ((-(aa * En - Lz) + aa * T / Delta) /
               (-aa * (aa * En - Lz) + (rp**2 + aa**2) * T / Delta))

    en = (1 / omega_r) * (em * omega_0 - em * omega_phi)
    en = int(en)

    return en

