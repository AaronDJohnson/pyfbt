try:
    from geodesic.constants.constants_eq import calc_eq_constants
    from geodesic.constants.constants_gen import calc_gen_constants
    from geodesic.constants.constants_pol import calc_pol_constants
    from geodesic.constants.constants_sc import calc_sc_constants
    from geodesic.constants.constants_sph import calc_sph_constants
except:
    from .constants_eq import calc_eq_constants
    from .constants_gen import calc_gen_constants
    from .constants_pol import calc_pol_constants
    from .constants_sc import calc_sc_constants
    from .constants_sph import calc_sph_constants


def calc_constants(aa, slr, ecc, x):
    """
    Choose which function to call based on input parameters.

    This version uses mpmath for extended precision.

    Parameters:
        aa (mpf): spin parameter (0, 1)
        slr (mpf): semi-latus rectum [6, inf)
        ecc (mpf): eccentricity [0, 1)
        x (mpf): inclination value given by cos(theta_inc)
                   x < 0 -> retrograde
                   x > 0 -> prograde

    Returns:
        En (mpf): energy
        Lz (mpf): angular momentum
        Q (mpf): Carter constant
    """
    if aa == 0:
        if slr < 6:
            print("slr value is too small for the SC case.")
            exit("Error in input values.")
        return calc_sc_constants(slr, ecc, x)
    elif x == 0:
        return calc_pol_constants(aa, slr, ecc)
    elif x ** 2 == 1:
        return calc_eq_constants(aa, slr, ecc, x)
    elif ecc == 0:
        return calc_sph_constants(aa, slr, x)
    else:
        return calc_gen_constants(aa, slr, ecc, x)
