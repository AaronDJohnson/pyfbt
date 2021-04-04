from numpy import sqrt
from sys import exit

# ------------------------------------------------------------------------------
#  SC orbits (a = 0)
# ------------------------------------------------------------------------------


def calc_sc_energy(slr, ecc, x):
    """
    Compute energy for SC case (a = 0).

    Parameters:
        slr (float): semi-latus rectum [6, inf)
        ecc (float): eccentricity [0, 1)
        x (float): inclination value given by cos(theta_inc) (0, 1]
                    negative x -> retrograde
                    positive x -> prograde
    
    Returns:
        En (float): constant energy
    """
    ecc2 = ecc * ecc
    slr2 = slr * slr
    x2 = x * x

    En = sqrt((-4 * ecc2 + (-2 + slr) ** 2) / (slr * (-3 - ecc2 + slr)))
    return En


def calc_sc_ang_momentum(slr, ecc, x):
    """
    Compute angular momentum for SC case (a = 0).

    Parameters:
        slr (float): semi-latus rectum [6, inf)
        ecc (float): eccentricity [0, 1)
        x (float): inclination value given by cos(theta_inc) (0, 1]
                   negative x -> retrograde
                   positive x -> prograde

    Returns:
        Lz (float): angular momentum constant
    """
    ecc2 = ecc * ecc
    slr2 = slr * slr
    x2 = x * x

    Lz = (slr * x) / sqrt(-3 - ecc2 + slr)
    return Lz


def calc_sc_carter(slr, ecc, x):
    """
    Compute carter constant for the SC case (a = 0).

    Parameters:
        slr (float): semi-latus rectum [6, inf)
        ecc (float): eccentricity [0, 1)
        x (float): inclination value given by cos(theta_inc) (0, 1]
                   negative x -> retrograde
                   positive x -> prograde
    
    Returns:
        Q (float): Carter constant
    """
    ecc2 = ecc * ecc
    slr2 = slr * slr
    x2 = x * x

    Q = (slr2 * (-1 + x2)) / (3 + ecc2 - slr)
    return Q


def calc_sc_constants(slr, ecc, x):
    """
    Energy, angular momentum, and carter constant calculation.

    Schwarzschild case (spin parameter a = 0)

    Parameters:
        slr (float): semi-latus rectum [6, inf)
        ecc (float): eccentricity [0, 1)
        x (float): inclination value given by cos(theta_inc) (0, 1]
                   negative x -> retrograde
                   positive x -> prograde

    Returns:
        En (float): energy
        Lz (float): angular momentum
        Q (float): Carter constant
    """
    ecc2 = ecc * ecc
    slr2 = slr * slr
    x2 = x * x

    En = sqrt((-4 * ecc2 + (-2 + slr) ** 2) / (slr * (-3 - ecc2 + slr)))
    Lz = (slr * x) / sqrt(-3 - ecc2 + slr)
    Q = (slr2 * (-1 + x2)) / (3 + ecc2 - slr)

    return En, Lz, Q
