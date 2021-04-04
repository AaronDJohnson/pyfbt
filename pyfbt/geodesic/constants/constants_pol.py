from numpy import sqrt
from sys import exit

# ------------------------------------------------------------------------------
#  Kerr polar orbits (x = 0)
# ------------------------------------------------------------------------------


def pol_energy(aa, slr, ecc):
    """
    Calculate energy for polar Kerr case.

    Parameters:
        aa (float): spin parameter (0, 1)
        slr (float): semi-latus rectum [6, inf)
        ecc (float): eccentricity

    Returns:
        En (float): energy
    """
    aa2 = aa * aa
    aa4 = aa2 * aa2
    ecc2 = ecc * ecc
    ecc4 = ecc2 * ecc2
    slr2 = slr * slr
    slr4 = slr2 * slr2
    return sqrt(
        -(
            (
                slr
                * (
                    aa4 * (-1 + ecc2) ** 2
                    + (-4 * ecc2 + (-2 + slr) ** 2) * slr2
                    + 2 * aa2 * slr * (-2 + slr + ecc2 * (2 + slr))
                )
            )
            / (
                aa4 * (-1 + ecc2) ** 2 * (-1 + ecc2 - slr)
                + (3 + ecc2 - slr) * slr4
                - 2 * aa2 * slr2 * (-1 - ecc4 + slr + ecc2 * (2 + slr))
            )
        )
    )


def pol_carter_const(aa, slr, ecc):
    """
    Calculate energy for polar Kerr case.

    Parameters:
        aa (float): spin parameter (0, 1)
        slr (float): semi-latus rectum [6, inf)
        ecc (float): eccentricity

    Returns:
        Q (float): Carter constant
    """
    aa2 = aa * aa
    aa4 = aa2 * aa2
    ecc2 = ecc * ecc
    ecc4 = ecc2 * ecc2
    slr2 = slr * slr
    slr4 = slr2 * slr2
    return -(
        (
            slr2
            * (
                aa4 * (-1 + ecc2) ** 2
                + slr4
                + 2 * aa2 * slr * (-2 + slr + ecc2 * (2 + slr))
            )
        )
        / (
            aa4 * (-1 + ecc2) ** 2 * (-1 + ecc2 - slr)
            + (3 + ecc2 - slr) * slr4
            - 2 * aa2 * slr2 * (-1 - ecc4 + slr + ecc2 * (2 + slr))
        )
    )


def calc_pol_constants(aa, slr, ecc):
    """
    Call Kerr polar orbit (x = 0) functions.

    Parameters:
        aa (float): spin parameter [0, 1)
        slr (float): semi-latus rectum
        ecc (float): eccentricity [0, 1)

    Returns:
        En (float): energy
        Lz (float): angular momentum
        Q (float): Carter constant
    """
    Lz = 0

    En = pol_energy(aa, slr, ecc)
    Q = pol_carter_const(aa, slr, ecc)

    return En, Lz, Q
