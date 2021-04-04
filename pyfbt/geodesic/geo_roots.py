from numpy import sqrt


def radial_roots(En, Q, aa, slr, ecc, M=1):
    """
    Roots of the radial geodesic eq, these correspond to turning points of the
    orbits.

    Parameters:
        En (float): energy
        Q (float): Carter constant
        aa (float): spin parameter (0, 1)
        slr (float): semi-latus rectum [6, inf)
        ecc (float): eccentricity [0, 1)

    Keyword Args:
        M (float) [1]: mass of the large body

    Returns:
        r1 (float): apastron
        r2 (float): periastron
        r3 (float)
        r4 (float)
    """
    En2 = En * En

    r1 = slr / (1 - ecc)
    r2 = slr / (1 + ecc)

    AplusB = (2 * M) / (1 - En2) - (r1 + r2)
    AB = (aa * aa * Q) / ((1 - En2) * r1 * r2)
    r3 = (AplusB + sqrt((AplusB * AplusB - 4 * AB))) / 2
    r4 = 0
    if r3 != 0:
        r4 = AB / r3
    return r1, r2, r3, r4


def polar_roots(En, Lz, aa, slr, x):
    """
    Roots of the polar geodesic eq, these correspond to turning points of the
    orbits.

    Parameters:
        En (float): energy
        Lz (float): angular momentum
        aa (float): spin parameter (0, 1)
        slr (float): semi-latus rectum [6, inf)
        x (float): inclination value given by cos(theta_inc) (0, 1]
                   negative x -> retrograde
                   positive x -> prograde

    Returns:
        zp (float)
        zm (float)
    """
    L2 = Lz * Lz

    zm = sqrt(1 - x * x)
    if abs(zm) == 1:
        zp = 0
    else:
        zp = sqrt(aa * aa * (1 - En * En) + L2 / (1 - zm * zm))
    return zp, zm
