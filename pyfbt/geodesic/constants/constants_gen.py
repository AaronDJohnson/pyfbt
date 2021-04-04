from numpy import sqrt
from sys import exit

# ------------------------------------------------------------------------------
#  Generic orbit constant calculation
# ------------------------------------------------------------------------------


def calc_delta(r, aa):
    """
    Calculate ubiquitous function on Kerr spacetimes.

    Parameters:
        r (float): radius
        aa (float): spin parameter (0, 1)

    Returns:
        delta (float)
    """
    # return r * r - 2 * r + aa * aa
    return r * (r - 2) + aa * aa


def calc_f(r, zm, aa):
    """
    DESCRIPTION TBD (FUNCTION USED IN GENERAL CONSTANTS).

    Parameters:
        r (float): radius
        zm (float): 1 - x * x
        aa (float): spin parameter (0, 1)

    Returns:
        f (float)
    """
    r2 = r * r
    r4 = r2 * r2
    aa2 = aa * aa
    zm2 = zm * zm

    delta = calc_delta(r, aa)
    return 2 * aa2 * r + aa2 * r2 + r4 + aa2 * zm2 * delta


def calc_g(r, aa):
    """
    DESCRIPTION TBD (FUNCTION USED IN GENERAL CONSTANTS).

    Parameters:
        r (float): radius
        aa (float): spin parameter (0, 1)

    Returns:
        g (float)
    """
    return 2 * aa * r


def calc_h(r, zm, aa):
    """
    DESCRIPTION TBD (FUNCTION USED IN GENERAL CONSTANTS).

    Parameters:
        r (float): radius
        zm (float): 1 - x * x
        aa (float): spin parameter (0, 1)

    Returns:
        h (float)
    """
    zm2 = zm * zm
    delta = calc_delta(r, aa)
    return (-2 + r) * r + (zm2 * delta) / (1 - zm2)


def calc_d(r, zm, aa):
    """
    DESCRIPTION TBD (FUNCTION USED IN GENERAL CONSTANTS).

    Parameters:
        r (float): radius
        zm (float): 1 - x * x
        aa (float): spin parameter (0, 1)

    Returns:
        d (float)
    """
    r2 = r * r
    aa2 = aa * aa
    zm2 = zm * zm

    delta = calc_delta(r, aa)
    return (r2 + aa2 * zm2) * delta


def gen_energy(zm, aa, slr, ecc, x):
    """
    Compute energy for generic orbit case.

    Parameters:
        zm (float): 1 - x * x
        aa (float): spin parameter (0, 1)
        slr (float): semi-latus rectum [~2, inf)
        ecc (float): eccentricity [0, 1)
        x (float): inclination value given by cos(theta_inc)
                   x < 0 -> retrograde
                   x > 0 -> prograde

    Returns:
        En (float): energy
    """
    r1 = slr / (1 - ecc)
    r2 = slr / (1 + ecc)

    dr1 = calc_d(r1, zm, aa)
    dr2 = calc_d(r2, zm, aa)
    gr1 = calc_g(r1, aa)
    gr2 = calc_g(r2, aa)
    hr1 = calc_h(r1, zm, aa)
    hr2 = calc_h(r2, zm, aa)
    fr1 = calc_f(r1, zm, aa)
    fr2 = calc_f(r2, zm, aa)

    kappa = dr1 * hr2 - hr1 * dr2
    epsilon = dr1 * gr2 - gr1 * dr2
    rho = fr1 * hr2 - hr1 * fr2
    eta = fr1 * gr2 - gr1 * fr2
    sigma = gr1 * hr2 - hr1 * gr2

    kappa2 = kappa * kappa
    epsilon2 = epsilon * epsilon
    rho2 = rho * rho
    x2 = x * x

    En = sqrt(
        (
            kappa * rho
            + 2 * epsilon * sigma
            - 2
            * sqrt(
                (
                    sigma
                    * (-(eta * kappa2) + epsilon * kappa * rho + epsilon2 * sigma)
                )
                / x2
            )
            * x
        )
        / (rho2 + 4 * eta * sigma)
    )

    return En


def gen_ang_momentum(En, aa, slr, ecc, x):
    """
    Compute angular momentum for generic orbit case.

    Parameters:
        En (float): energy
        aa (float): spin parameter (0, 1)
        slr (float): semi-latus rectum [6, inf)
        ecc (float): eccentricity [0, 1)
        x (float): inclination value given by cos(theta_inc)
                   x < 0 -> retrograde
                   x > 0 -> prograde

    Returns:
        z_ang_mom (float): angular momentum
    """
    r1 = slr / (1 - ecc)
    zm = sqrt(1 - x * x)

    # // delta_r1 = calc_delta(r1, aa)
    fr1 = calc_f(r1, zm, aa)
    gr1 = calc_g(r1, aa)
    hr1 = calc_h(r1, zm, aa)
    dr1 = calc_d(r1, zm, aa)

    En2 = En * En
    x2 = x * x
    Lz = (
        -(En * gr1) + x * sqrt((-(dr1 * hr1) + En2 * (gr1 * gr1 + fr1 * hr1)) / x2)
    ) / hr1
    return Lz


def gen_carter_const(En, Lz, aa, slr, ecc, x):
    """
    Compute Carter constant for generic orbit case.

    Parameters:
        En (float): energy
        Lz (float): angular momentum
        aa (float): spin parameter (0, 1)
        slr (float): semi-latus rectum [6, inf)
        ecc (float): eccentricity [0, 1)
        x (float): inclination value given by cos(theta_inc)
                   x < 0 -> retrograde
                   x > 0 -> prograde

    Returns:
        Q (float): Carter constant
    """
    zm = sqrt(1 - x * x)
    zm2 = zm * zm
    return zm2 * (aa * aa * (1 - En * En) + Lz * Lz / (1 - zm2))


def calc_gen_constants(aa, slr, ecc, x):
    """
    Call generic orbit constant calculating functions in one function.

    Parameters:
        aa (float): spin parameter (0, 1)
        slr (float): semi-latus rectum [6, inf)
        ecc (float): eccentricity [0, 1)
        x (float): inclination value given by cos(theta_inc)
                   x < 0 -> retrograde
                   x > 0 -> prograde

    Returns:
        En (float): energy
        Lz (float): angular momentum
        Q (float): Carter constant
    """
    zm = sqrt(1 - x * x)

    En = gen_energy(zm, aa, slr, ecc, x)
    Lz = gen_ang_momentum(En, aa, slr, ecc, x)
    Q = gen_carter_const(En, Lz, aa, slr, ecc, x)
    return En, Lz, Q
