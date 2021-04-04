from numpy import sin, cos
from numpy import arccos
from numpy import arcsinh
from numpy import arcsin
from numpy import sqrt, floor, pi, tan, real
from scipy.special import ellipj, ellipk, ellipe
from scipy.special import ellipkinc
from scipy.special import ellipeinc
from mpmath import ellippi  # elliptic integral of third kind not implemented in scipy yet


def calc_radius(psi, slr, ecc):
    """
    r coordinate in terms of radial angle, psi

    Parameters:
        psi (float): radial angle
        slr (float): semi-latus rectum
        ecc (float): eccentricity

    Returns:
        radius (float)
    """
    return slr / (1 + ecc * cos(psi))


def calc_rq(qr, r1, r2, r3, r4):
    """
    function used in computing radial geodesic coordinates

    Parameters:
        qr (float)
        r1 (float): radial root
        r2 (float): radial root
        r3 (float): radial root
        r4 (float): radial root

    Returns:
        rq (float)
    """
    kr = ((r1 - r2) * (r3 - r4)) / ((r1 - r3) * (r2 - r4))

    u = (qr * ellipk(kr)) / pi
    m = kr
    sn, __, __, __ = ellipj(u, m)

    return (
        -(r2 * (r1 - r3)) + (r1 - r2) * r3 * sn ** 2
    ) / (-r1 + r3 + (r1 - r2) * sn ** 2)


def calc_zq(qz, zp, zm, En, aa):
    """
    function used in computing polar geodesic coordinates

    Parameters:
        qz (float)
        zp (float): polar root
        zm (float): polar root
        En (float): energy
        aa (float): spin

    Returns:
        zq (float)
    """
    ktheta = (aa ** 2 * (1 - En ** 2) * zm ** 2) / zp ** 2
    u = (2 * (pi / 2.0 + qz) * ellipk(ktheta)) / pi
    m = ktheta
    sn, __, __, __ = ellipj(u, m)

    return zm * sn


def calc_psi_r(qr, r1, r2, r3, r4):
    """
    radial geodesic angle

    Parameters:
        qr (float)
        r1 (float): radial root
        r2 (float): radial root
        r3 (float): radial root
        r4 (float): radial root

    Returns:
        psi_r (float)
    """
    kr = ((r1 - r2) * (r3 - r4)) / ((r1 - r3) * (r2 - r4))
    u = (qr * ellipk(kr)) / pi
    m = kr
    __, __, __, ph = ellipj(u, m)
    return ph


def calc_t_r(qr, r1, r2, r3, r4, En, Lz, aa, M=1):
    """
    delta_t_r in Drasco and Hughes (2005)

    Parameters:
        qr (float)
        r1 (float): radial root
        r2 (float): radial root
        r3 (float): radial root
        r4 (float): radial root
        En (float): energy
        Lz (float): angular momentum
        aa (float): spin

    Keyword Args:
        M (float): mass

    Returns:
        t_r (float)
    """
    psi_r = calc_psi_r(qr, r1, r2, r3, r4)

    kr = ((r1 - r2) * (r3 - r4)) / ((r1 - r3) * (r2 - r4))

    rp = M + sqrt(-(aa ** 2) + M ** 2)
    rm = M - sqrt(-(aa ** 2) + M ** 2)

    hr = (r1 - r2) / (r1 - r3)
    hp = ((r1 - r2) * (r3 - rp)) / ((r1 - r3) * (r2 - rp))
    hm = ((r1 - r2) * (r3 - rm)) / ((r1 - r3) * (r2 - rm))

    return -(
        (
            En
            * (
                (
                    -4
                    * (r2 - r3)
                    * (
                        -(
                            (
                                (-2 * aa ** 2 + (4 - (aa * Lz) / En) * rm)
                                * ((qr * float(ellippi(hm, kr))) / pi - float(ellippi(hm, psi_r, kr)))
                            )
                            / ((r2 - rm) * (r3 - rm))
                        )
                        + (
                            (-2 * aa ** 2 + (4 - (aa * Lz) / En) * rp)
                            * ((qr * float(ellippi(hp, kr))) / pi - float(ellippi(hp, psi_r, kr)))
                        )
                        / ((r2 - rp) * (r3 - rp))
                    )
                )
                / (-rm + rp)
                + 4 * (r2 - r3) * ((qr * float(ellippi(hr, kr))) / pi - float(ellippi(hr, psi_r, kr)))
                + (r2 - r3)
                * (r1 + r2 + r3 + r4)
                * ((qr * float(ellippi(hr, kr))) / pi - float(ellippi(hr, psi_r, kr)))
                + (r1 - r3)
                * (r2 - r4)
                * (
                    (qr * ellipe(kr)) / pi
                    - ellipeinc(psi_r, kr)
                    + (hr * cos(psi_r) * sin(psi_r) * sqrt(1 - kr * sin(psi_r) ** 2))
                    / (1 - hr * sin(psi_r) ** 2)
                )
            )
        )
        / sqrt((1 - En ** 2) * (r1 - r3) * (r2 - r4))
    )


def calc_phi_r(qr, r1, r2, r3, r4, En, Lz, aa, M=1):
    """
    delta_phi_r in Drasco and Hughes (2005)

    Parameters:
        qr (float)
        r1 (float): radial root
        r2 (float): radial root
        r3 (float): radial root
        r4 (float): radial root
        En (float): energy
        Lz (float): angular momentum
        aa (float): spin

    Keyword Args:
        M (float): mass

    Returns:
        phi_r (float)
    """

    psi_r = calc_psi_r(qr, r1, r2, r3, r4)
    kr = ((r1 - r2) * (r3 - r4)) / ((r1 - r3) * (r2 - r4))
    rp = M + sqrt(-(aa ** 2) + M ** 2)
    rm = M - sqrt(-(aa ** 2) + M ** 2)

    hp = ((r1 - r2) * (r3 - rp)) / ((r1 - r3) * (r2 - rp))
    hm = ((r1 - r2) * (r3 - rm)) / ((r1 - r3) * (r2 - rm))

    return (
        2
        * aa
        * En
        * (
            -(
                (
                    (r2 - r3)
                    * (-((aa * Lz) / En) + 2 * rm)
                    * ((qr * float(ellippi(hm, kr))) / pi - float(ellippi(hm, psi_r, kr)))
                )
                / ((r2 - rm) * (r3 - rm))
            )
            + (
                (r2 - r3)
                * (-((aa * Lz) / En) + 2 * rp)
                * ((qr * float(ellippi(hp, kr))) / pi - float(ellippi(hp, psi_r, kr)))
            )
            / ((r2 - rp) * (r3 - rp))
        )
    ) / (sqrt((1 - En ** 2) * (r1 - r3) * (r2 - r4)) * (-rm + rp))


def calc_psi_z(qz, zp, zm, En, aa):
    """
    angle used in polar geodesic calculations

    Parameters:
        qz (float)
        zp (float): polar root
        zm (float): polar root
        En (float): energy
        aa (float): spin

    Returns:
        psi_z (float)
    """
    ktheta = (aa ** 2 * (1 - En ** 2) * zm ** 2) / zp ** 2
    u = (2 * (pi / 2.0 + qz) * ellipk(ktheta)) / pi
    m = ktheta
    __, __, __, ph = ellipj(u, m)
    return ph


def calc_t_z(qz, zp, zm, En, aa):
    """
    delta_t_theta in Drasco and Hughes (2003?)

    Parameters:
        qz (float)
        zp (float): polar root
        zm (float): polar root
        En (float): energy
        aa (float): spin

    Returns:
        t_z (float)
    """
    psi_z = calc_psi_z(qz, zp, zm, En, aa)
    ktheta = (aa ** 2 * (1 - En ** 2) * zm ** 2) / zp ** 2
    return (
        En * zp * ((2 * (pi / 2.0 + qz) * ellipe(ktheta)) / pi - ellipeinc(psi_z, ktheta))
    ) / (1 - En ** 2)


def calc_phi_z(qz, zp, zm, En, Lz, aa):
    """
    delta_phi_theta in Drasco and Hughes (2003?)

    Parameters:
        qz (float)
        zp (float): polar root
        zm (float): polar root
        En (float): energy
        Lz (float): angular momentum
        aa (float): spin

    Returns:
        phi_z (float)
    """
    psi_z = calc_psi_z(qz, zp, zm, En, aa)
    ktheta = (aa ** 2 * (1 - En ** 2) * zm ** 2) / zp ** 2
    return -(
        (
            Lz
            * (
                (2 * (pi / 2.0 + qz) * float(ellippi(zm ** 2, ktheta))) / pi
                - float(ellippi(zm ** 2, psi_z, ktheta))
            )
        )
        / zp
    )


def calc_Ct(qr0, qz0, r1, r2, r3, r4, zp, zm, En, Lz, aa):
    """
    phase constant so that Mino time starts at 0

    Parameters:
        qr0 (float): initial radial phase
        qz0 (float): initial polar phase
        r1 (float): radial root
        r2 (float): radial root
        r3 (float): radial root
        r4 (float): radial root
        zp (float): polar root
        zm (float): polar root
        En (float): energy
        Lz (float): angular momentum
        aa (float): spin

    Returns:
        Ct (float)
    """
    t_r = calc_t_r(qr0, r1, r2, r3, r4, En, Lz, aa)
    t_z = calc_t_z(qz0, zp, zm, En, aa)
    return t_r + t_z


def calc_Cz(qr0, qz0, r1, r2, r3, r4, zp, zm, En, Lz, aa):
    """
    phase constant so that Mino time starts at 0

    Parameters:
        qr0 (float): initial radial phase
        qz0 (float): initial polar phase
        r1 (float): radial root
        r2 (float): radial root
        r3 (float): radial root
        r4 (float): radial root
        zp (float): polar root
        zm (float): polar root
        En (float): energy
        Lz (float): angular momentum
        aa (float): spin

    Returns:
        Cz (float)
    """
    phi_r = calc_phi_r(qr0, r1, r2, r3, r4, En, Lz, aa)
    phi_z = calc_phi_z(qz0, zp, zm, En, Lz, aa)
    return phi_r + phi_z


def calc_t(
    mino_t, ups_r, ups_theta, gamma, qt0, qr0, qz0, r1, r2, r3, r4, zp, zm, En, Lz, aa
):
    """
    time geodesic coordinate

    Parameters:
        mino_t (float): Mino time
        ups_r (float): radial Mino frequency
        ups_theta (float): theta Mino frequency
        gamma (float): time Mino frequency
        qt0 (float): initial time phase
        qr0 (float): initial radial phase
        qz0 (float): initial theta phase
        r1 (float): radial root
        r2 (float): radial root
        r3 (float): radial root
        r4 (float): radial root
        zp (float): polar root
        zm (float): polar root
        En (float): energy
        Lz (float): angular momentum
        aa (float): spin

    Returns:
        t (float)
    """
    eta_t = qt0 + gamma * mino_t
    eta_r = qr0 + ups_r * mino_t
    eta_z = qz0 + ups_theta * mino_t
    if r1 == r2:
        t_r = 0
    else:
        t_r = calc_t_r(eta_r, r1, r2, r3, r4, En, Lz, aa)
    if zm == 0:
        t_z = 0
    else:
        t_z = calc_t_z(eta_z, zp, zm, En, aa)
    if qr0 == 0 and qz0 == 0:
        Ct = 0
    else:
        Ct = calc_Ct(qr0, qz0, r1, r2, r3, r4, zp, zm, En, Lz, aa)
    return eta_t + t_r + t_z - Ct


def calc_r(mino_t, ups_r, qr0, r1, r2, r3, r4):
    """
    radius in terms of Mino time

    Parameters:
        mino_t (float): Mino time
        ups_r (float): Mino radial frequency
        qr0 (float): inital radial phase
        r1 (float): radial root
        r2 (float): radial root
        r3 (float): radial root
        r4 (float): radial root

    Returns:
        r (float)
    """
    eta = ups_r * mino_t + qr0
    return calc_rq(eta, r1, r2, r3, r4)


def calc_theta(mino_t, ups_theta, qz0, zp, zm, En, aa):
    """
    theta in terms of Mino time

    Parameters:
        mino_t (float): Mino time
        ups_theta (float): Mino theta frequency
        qz0 (float): inital polar phase
        zp (float): polar root
        zm (float): polar root
        En (float): energy
        aa (float): spin

    Returns:
        theta (float)
    """
    eta = ups_theta * mino_t + qz0
    return arccos(calc_zq(eta, zp, zm, En, aa))


def calc_phi(
    mino_t,
    ups_r,
    ups_theta,
    ups_phi,
    qphi0,
    qr0,
    qz0,
    r1,
    r2,
    r3,
    r4,
    zp,
    zm,
    En,
    Lz,
    aa,
):
    """
    phi in terms of Mino time

    Parameters:
        mino_t (float): Mino time
        ups_r (float): Mino radial frequency
        ups_theta (float): Mino theta frequency
        ups_phi (float): Mino phi frequency
        qphi0 (float): initial phi phase
        qr0 (float): initial radial phase
        qz0 (float): initial polar phase
        r1 (float): radial root
        r2 (float): radial root
        r3 (float): radial root
        r4 (float): radial root
        zp (float): polar root
        zm (float): polar root
        En (float): energy
        Lz (float): angular momentum
        aa (float): spin

    Returns:
        phi (float)
    """
    eta_phi = ups_phi * mino_t + qphi0
    eta_r = ups_r * mino_t + qr0
    eta_theta = ups_theta * mino_t + qz0
    if r1 == r2:
        phi_r = 0
    else:
        phi_r = calc_phi_r(eta_r, r1, r2, r3, r4, En, Lz, aa)
    if zm == 0:
        phi_z = 0
    else:
        phi_z = calc_phi_z(eta_theta, zp, zm, En, Lz, aa)
    if qr0 == 0 and qz0 == 0:
        Cz = 0
    else:
        Cz = calc_Cz(qr0, qz0, r1, r2, r3, r4, zp, zm, En, Lz, aa)
    return eta_phi + phi_r + phi_z - Cz


def calc_lambda_r(r, r1, r2, r3, r4, En):
    """
    Mino time as a function of r (which in turn is a function of psi)

    Parameters:
        r (float): radius
        r1 (float): radial root
        r2 (float): radial root
        r3 (float): radial root
        r4 (float): radial root
        En (float): energy

    Returns:
        lambda (float)
    """
    kr = ((r1 - r2) * (r3 - r4)) / ((r1 - r3) * (r2 - r4))
    # if r1 == r2:
    #     # circular orbit
    #     print('Circular orbits currently do not work.')
    #     return 0
    yr = sqrt(((r - r2) * (r1 - r3)) / ((r1 - r2) * (r - r3)))
    F_asin = ellipkinc(arcsin(yr), kr)
    return (2 * F_asin) / (sqrt(1 - En * En) * sqrt((r1 - r3) * (r2 - r4)))


def calc_lambda_psi(psi, ups_r, r1, r2, r3, r4, En, slr, ecc):
    """
    changes lambda(r) -> lambda(psi) by computing lambda(r(psi))

    Parameters:
        psi (float): radial angle
        ups_r (float): radial Mino frequency
        r1 (float): radial root
        r2 (float): radial root
        r3 (float): radial root
        r4 (float): radial root
        En (float): energy
        slr (float): semi-latus rectum
        ecc (float): eccentricity

    Returns:
        r (float): radius
        lambda_psi (float)
    """
    r = calc_radius(psi, slr, ecc)
    lam_r = 2 * pi / ups_r  # radial period
    lam_r1 = calc_lambda_r(r2, r1, r2, r3, r4, En)
    turns = floor(psi / (2 * pi))
    if (psi % (2 * pi)) <= pi:
        res = calc_lambda_r(r, r1, r2, r3, r4, En) - lam_r1
    else:
        res = lam_r1 - calc_lambda_r(r, r1, r2, r3, r4, En)

    return r, lam_r * turns + res


def calc_lambda_0(chi, zp, zm, En, Lz, aa, slr, x):
    """
    Mino time as a function of polar angle, chi

    Parameters:
        chi (float): polar angle
        zp (float): polar root
        zm (float): polar root
        En (float): energy
        Lz (float): angular momentum
        aa (float): spin
        slr (float): semi-latus rectum
        x (float): inclination

    Returns:
        lambda_0 (float)

    """
    beta = aa * aa * (1 - En * En)
    k = sqrt(zm / zp)
    k2 = k * k
    prefactor = 1 / sqrt(beta * zp)
    ellipticK_k = ellipk(k2)
    ellipticF = ellipkinc(pi / 2 - chi, k2)

    return prefactor * (ellipticK_k - ellipticF)


def calc_wtheta(chi, ups_theta, zp, zm, En, Lz, aa, slr, x):
    """
    w_theta = ups_theta * lambda as a function of polar angle chi

    Parameters:
        chi (float): polar angle
        ups_theta (float): theta mino frequency
        En (float): energy
        Lz (float): angular momentum
        aa (float): spin
        slr (float): semi-latus rectum
        x (float): inclination

    Returns:
        w_theta (float)
    """
    if chi >= 0 and chi <= pi / 2:
        return ups_theta * calc_lambda_0(chi, zp, zm, En, Lz, aa, slr, x)
    elif chi > pi / 2 and chi <= pi:
        return pi - ups_theta * calc_lambda_0(pi - chi, zp, zm, En, Lz, aa, slr, x)
    elif chi > pi and chi <= 3 * pi / 2:
        return pi + ups_theta * calc_lambda_0(chi - pi, zp, zm, En, Lz, aa, slr, x)
    elif chi > 3 * pi / 2 and chi <= 2 * pi:
        return 2 * pi - ups_theta * calc_lambda_0(
            2 * pi - chi, zp, zm, En, Lz, aa, slr, x
        )
    else:
        print("Something went wrong in calc_wtheta!")
        return 0.0  # this case should not occur, but is required by C++


def calc_dwtheta_dchi(chi, zp, zm):
    """
    derivative of w_theta

    Parameters:
        chi (float): polar angle
        zp (float): polar root
        zm (float): polar root

    Returns:
        dw_dtheta (float)
    """
    k = sqrt(zm / zp)
    ellipticK_k = ellipk(k ** 2)
    return pi / (2 * ellipticK_k) * (1 / (1 - k * k * cos(chi) ** 2))


def calc_wr(psi, ups_r, En, Lz, Q, aa, slr, ecc, x):
    """
    Computes wr by analytic evaluation of the integral in Drasco and Hughes (2005)

    Note that this currently only works for [0, pi], but could be extended with
    a little bit of work.
    """
    a1 = (1 - ecc**2)*(1 - En**2)
    b1 = 2*(1 - En**2 - (1 - ecc**2)/slr)
    c1 = (((3 + ecc**2)*(1 - En**2))/(1 - ecc**2) - 4/slr + 
          ((1 - ecc**2)*(aa**2*(1 - En**2) + Lz**2 + Q))/slr**2)

    if psi == pi:
        # the closed form function has a singularity at psi = pi
        # but it can be evaluated in integral form to be pi
        return pi
    else:
        return ((-2j*(1 - ecc**2)*ups_r*cos(psi/2.)**2*
            ellipkinc(1j*arcsinh(sqrt((a1 - (-1 + ecc)*(b1 + c1 - c1*ecc))/
                (a1 + b1 + c1 - c1*ecc**2 + sqrt((b1**2 - 4*a1*c1)*ecc**2)))*tan(psi/2.)),
            (a1 + b1 + c1 - c1*ecc**2 + sqrt((b1**2 - 4*a1*c1)*ecc**2))/
            (a1 + b1 + c1 - c1*ecc**2 - sqrt((b1**2 - 4*a1*c1)*ecc**2)))*
            sqrt(2 + (2*(a1 - (-1 + ecc)*(b1 + c1 - c1*ecc))*tan(psi/2.)**2)/
            (a1 + b1 + c1 - c1*ecc**2 - sqrt((b1**2 - 4*a1*c1)*ecc**2)))*
            sqrt(1 + ((a1 - (-1 + ecc)*(b1 + c1 - c1*ecc))*tan(psi/2.)**2)/
            (a1 + b1 + c1 - c1*ecc**2 + sqrt((b1**2 - 4*a1*c1)*ecc**2))))/
        (sqrt((a1 - (-1 + ecc)*(b1 + c1 - c1*ecc))/
            (a1 + b1 + c1 - c1*ecc**2 + sqrt((b1**2 - 4*a1*c1)*ecc**2)))*slr*
            sqrt(2*a1 + 2*b1 + 2*c1 + c1*ecc**2 + 2*(b1 + 2*c1)*ecc*cos(psi) + c1*ecc**2*cos(2*psi))))


def calc_J(chi, En, Lz, Q, aa, slr, ecc):
    """
    Schmidt's J function.

    Parameters:
        chi (float): radial angle
        En (float): energy
        Lz (float): angular momentum
        Q (float): Carter constant
        aa (float): spin
        slr (float): semi-latus rectum
        ecc (float): eccentricity

    Returns:
        J (float)
    """
    En2 = En * En
    ecc2 = ecc * ecc
    aa2 = aa * aa
    Lz2 = Lz * Lz
    slr2 = slr * slr

    eta = 1 + ecc * cos(chi)
    eta2 = eta * eta

    J = (
        (1 - ecc2) * (1 - En2)
        + 2 * (1 - En2 - (1 - ecc2) / slr) * eta
        + (
            ((3 + ecc2) * (1 - En2)) / (1 - ecc2)
            + ((1 - ecc2) * (aa2 * (1 - En2) + Lz2 + Q)) / slr2
            - 4 / slr
        )
        * eta2
    )

    return J


def calc_equatorial_coords(
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
    ecc,
    qt0=0,
    qr0=0,
    qz0=0,
    qphi0=0,
):
    """
    Computes all equatorial coordinates in a convenient function

    Parameters:
        psi (float): radial angle
        ups_r (float): radial Mino frequency
        ups_theta (float): theta Mino frequency
        ups_phi (float): phi Mino frequency
        gamma (float): time Mino frequency
        r1 (float): radial root
        r2 (float): radial root
        r3 (float): radial root
        r4 (float): radial root
        zp (float): polar root
        zm (float): polar root
        En (float): energy
        Lz (float): angular momentum
        aa (float): spin
        slr (float): semi-latus rectum
        ecc (float): eccentricity

    Keyword Args:
        qt0 (float): initial time phase
        qr0 (float): initial radial phase
        qz0 (float): initial theta phase
        qphi0 (float): initial phi phase

    Returns:
        t (float): time coordinate
        r (float): radial coordinate
        theta (float): polar coordinate
        phi (float): azimuthal coordinate
    """
    if zm != 0:
        print("The orbit specified is not equatorial.")
    r, lam_psi = calc_lambda_psi(psi, ups_r, r1, r2, r3, r4, En, slr, ecc)
    t = calc_t(
        lam_psi,
        ups_r,
        ups_theta,
        gamma,
        qt0,
        qr0,
        qz0,
        r1,
        r2,
        r3,
        r4,
        zp,
        zm,
        En,
        Lz,
        aa,
    )
    theta = pi / 2
    phi = calc_phi(
        lam_psi,
        ups_r,
        ups_theta,
        ups_phi,
        qphi0,
        qr0,
        qz0,
        r1,
        r2,
        r3,
        r4,
        zp,
        zm,
        En,
        Lz,
        aa,
    )
    return t, r, theta, phi


def calc_dwr_dpsi(psi, ups_r, En, Lz, Q, aa, slr, ecc):
    J = calc_J(psi, En, Lz, Q, aa, slr, ecc)
    return (1 - ecc ** 2) / slr * ups_r / sqrt(J)


def calc_gen_coords_mino(
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
    aa,
    qphi0=0,
    qr0=0,
    qz0=0,
    qt0=0,
):
    t = calc_t(
        mino_t,
        ups_r,
        ups_theta,
        gamma,
        qt0,
        qr0,
        qz0,
        r1,
        r2,
        r3,
        r4,
        zp,
        zm,
        En,
        Lz,
        aa,
    )
    r = calc_r(mino_t, ups_r, qr0, r1, r2, r3, r4)
    theta = calc_theta(mino_t, ups_theta, qz0, zp, zm, En, aa)
    phi = calc_phi(
        mino_t,
        ups_r,
        ups_theta,
        ups_phi,
        qphi0,
        qr0,
        qz0,
        r1,
        r2,
        r3,
        r4,
        zp,
        zm,
        En,
        Lz,
        aa,
    )
    return t, r, theta, phi


def calc_gen_coords(
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
    Q,
    aa,
    slr,
    ecc,
    x,
    qphi0=0,
    qr0=0,
    qz0=0,
    qt0=0,
):
    wr = calc_wr(psi, ups_r, En, Lz, Q, aa, slr, ecc, x)
    wr = real(wr)  # this should be a real value
    # wtheta = calc_wtheta(chi, ups_theta, zp, zm, En, Lz, aa, slr, x)
    # when evaluating the orbit, we do not separate theta and r directions
    # when evaluating the flux integral, we will
    wtheta = (ups_theta / ups_r) * wr
    # TODO (aaron): check that the following are equal
    # print('lambdar =', wr / ups_r)
    # print('wr =', wr)
    # print('wtheta =', wtheta)
    # print('lambdatheta =', wtheta / ups_theta)
    mino_t = wr / ups_r
    t = calc_t(
        mino_t,
        ups_r,
        ups_theta,
        gamma,
        qt0,
        qr0,
        qz0,
        r1,
        r2,
        r3,
        r4,
        zp,
        zm,
        En,
        Lz,
        aa,
    )
    r = calc_r(mino_t, ups_r, qr0, r1, r2, r3, r4)
    theta = calc_theta(mino_t, ups_theta, qz0, zp, zm, En, aa)
    phi = calc_phi(
        mino_t,
        ups_r,
        ups_theta,
        ups_phi,
        qphi0,
        qr0,
        qz0,
        r1,
        r2,
        r3,
        r4,
        zp,
        zm,
        En,
        Lz,
        aa,
    )
    return t, r, theta, phi
