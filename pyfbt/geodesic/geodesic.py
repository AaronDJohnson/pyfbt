from numpy import cos

try:
    from geodesic.constants import calc_constants
    from geodesic.geo_roots import radial_roots, polar_roots
    from geodesic.frequencies import mino_freqs, find_omega, mino_freqs, boyer_freqs
    from geodesic.coordinates.coords import calc_coords
    from geodesic.coordinates.coords_gen import calc_gen_coords_mino
except:
    from .constants.constants import calc_constants
    from .geo_roots import radial_roots, polar_roots
    from .frequencies import mino_freqs, find_omega, mino_freqs, boyer_freqs
    from .coordinates.coords import calc_coords
    from .coordinates.coords_gen import calc_gen_coords_mino


def calc_consts(aa, slr, ecc, x):
    """
    Compute adiabatic constants.

    Parameters:
        aa (float): SMBH spin
        slr (float): semi-latus rectum
        ecc (float): eccentricity
        x (float): cos of the inclination

    Returns:
        En (float): energy
        Lz (float): angular momentum
        Q (float): Carter constant
    """
    En, Lz, Q = calc_constants(aa, slr, ecc, x)
    return En, Lz, Q


def calc_radial_roots(aa, slr, ecc, x):
    """
    Compute radial roots.

    Parameters:
        aa (float): SMBH spin
        slr (float): semi-latus rectum
        ecc (float): eccentricity
        x (float): cos of the inclination

    Returns:
        r1 (float): apastron
        r2 (float): periastron
        r3 (float): radial root 3
        r4 (float): radial root 4
    """
    En, Lz, Q = calc_constants(aa, slr, ecc, x)
    r1, r2, r3, r4 = radial_roots(En, Q, aa, slr, ecc)
    return r1, r2, r3, r4


def calc_polar_roots(aa, slr, ecc, x):
    """
    Compute polar roots.

    Parameters:
        aa (float): SMBH spin
        slr (float): semi-latus rectum
        ecc (float): eccentricity
        x (float): cos of the inclination

    Returns:
        zp (float): polar root
        zm (float): polar root
    """
    En, Lz, Q = calc_constants(aa, slr, ecc, x)
    zp, zm = polar_roots(En, Lz, aa, slr, x)
    return zp, zm


def calc_mino_freqs(aa, slr, ecc, x, M=1):
    """
    Compute Mino frequencies.

    Parameters:
        aa (float): SMBH spin
        slr (float): semi-latus rectum
        ecc (float): eccentricity
        x (float): cos of the inclination
        M (float) [1]: stellar black hole mass

    Returns:
        ups_r (float): radial Mino frequency
        ups_theta (float): polar Mino frequency
        ups_phi (float): azimuthal Mino frequency
        gamma (float): temporal Mino frequency
    """
    En, Lz, Q = calc_constants(aa, slr, ecc, x)
    r1, r2, r3, r4 = radial_roots(En, Q, aa, slr, ecc, M)
    ups_r, ups_theta, ups_phi, gamma = mino_freqs(
        r1, r2, r3, r4, En, Lz, Q, aa, slr, ecc, x
    )
    return ups_r, ups_theta, ups_phi, gamma


def calc_boyer_freqs(aa, slr, ecc, x, M=1):
    """
    Compute Boyer-Lindquist frequencies.

    Parameters:
        aa (float): SMBH spin
        slr (float): semi-latus rectum
        ecc (float): eccentricity
        x (float): cos of the inclination

    Returns:
        Omega_r (float): radial Boyer-Lindquist frequency
        Omega_theta (float): polar Boyer-Lindquist frequency
        Omega_phi (float): azimuthal Boyer-Lindquist frequency
    """
    En, Lz, Q = calc_constants(aa, slr, ecc, x)
    r1, r2, r3, r4 = radial_roots(En, Q, aa, slr, ecc, M)
    ups_r, ups_theta, ups_phi, gamma = mino_freqs(
        r1, r2, r3, r4, En, Lz, Q, aa, slr, ecc, x
    )
    omega_r, omega_theta, omega_phi = boyer_freqs(
        ups_r, ups_theta, ups_phi, gamma, aa, slr, ecc, x, M
    )
    return omega_r, omega_theta, omega_phi


def find_omega(en, em, kay, aa, slr, ecc, x, M=1):
    """
    Compute gravitational wave frequency omega.

    Parameters:
        en (int): radial mode
        em (int): azimuthal mode
        kay (int): polar mode
        aa (float): SMBH spin
        slr (float): semi-latus rectum
        ecc (float): eccentricity
        x (float): cos of the inclination

    Returns:
        omega (float): gravitational wave frequency
    """
    En, Lz, Q = calc_constants(aa, slr, ecc, x)
    r1, r2, r3, r4 = radial_roots(En, Q, aa, slr, ecc, M)
    ups_r, ups_theta, ups_phi, gamma = mino_freqs(
        r1, r2, r3, r4, En, Lz, Q, aa, slr, ecc, x
    )
    omega_r, omega_theta, omega_phi = boyer_freqs(
        ups_r, ups_theta, ups_phi, gamma, aa, slr, ecc, x, M
    )
    omega = en * omega_r + em * omega_phi + kay * omega_theta
    return omega


def coordinates(psi, aa, slr, ecc, x):
    """
    Compute coordinates of the orbit given radial angle psi.

    Parameters:
        psi (float): radial angle
        chi (float): polar angle
        aa (float): SMBH spin
        slr (float): semi-latus rectum
        ecc (float): eccentricity
        x (float): cos of the inclination

    Returns:
        t (float): time coordinate
        r (float): radial coordinate
        theta (float): theta coordinate
        phi (float): phi coordinate
    """
    En, Lz, Q = calc_constants(aa, slr, ecc, x)
    r1, r2, r3, r4 = radial_roots(En, Q, aa, slr, ecc, M=1)
    zp, zm = polar_roots(En, Lz, aa, slr, x)
    ups_r, ups_theta, ups_phi, gamma = mino_freqs(
        r1, r2, r3, r4, En, Lz, Q, aa, slr, ecc, x
    )
    # print(psi)
    # print(chi)
    t, r, theta, phi = calc_coords(
        psi,
        # chi,
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
        Q,
        aa,
        slr,
        ecc,
        x,
    )
    return t, r, theta, phi


def mino_coords(mino_t, aa, slr, ecc, x):
    """
    Compute coordinates of the orbit given two angles psi and chi.

    Parameters:
        mino_t (float): mino time coordinate lambda
        aa (float): SMBH spin
        slr (float): semi-latus rectum
        ecc (float): eccentricity
        x (float): cos of the inclination

    Returns:
        t (float): time coordinate
        r (float): radial coordinate
        theta (float): theta coordinate
        phi (float): phi coordinate
    """
    En, Lz, Q = calc_constants(aa, slr, ecc, x)
    r1, r2, r3, r4 = radial_roots(En, Q, aa, slr, ecc, M=1)
    zp, zm = polar_roots(En, Lz, aa, slr, x)
    ups_r, ups_theta, ups_phi, gamma = mino_freqs(
        r1, r2, r3, r4, En, Lz, Q, aa, slr, ecc, x
    )
    t, r, theta, phi = calc_gen_coords_mino(
        mino_t,
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
        Q,
        aa,
    )
    return t, r, theta, phi
