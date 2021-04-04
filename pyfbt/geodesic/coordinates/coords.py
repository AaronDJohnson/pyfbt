try:
    from geodesic.coordinates.coords_gen import calc_equatorial_coords, calc_gen_coords
    from geodesic.coordinates.coords_circ_eq import calc_circular_eq_coords
except:
    from .coords_gen import calc_equatorial_coords, calc_gen_coords
    from .coords_circ_eq import calc_circular_eq_coords

def calc_coords(
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
    qt0=0,
    qr0=0,
    qz0=0,
    qphi0=0,
    M=1
):
    if x ** 2 == 1 and ecc == 0:
        # print('detected circ_eq orbit')
        t, r, theta, phi = calc_circular_eq_coords(psi, En, Lz, aa, slr, M)
        return t, r, theta, phi
    elif x ** 2 == 1:
        # print('detected equatorial orbit')
        t, r, theta, phi = calc_equatorial_coords(
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
            qt0,
            qr0,
            qz0,
            qphi0
        )
        return t, r, theta, phi
    else:
        # print('detected generic orbit')
        t, r, theta, phi = calc_gen_coords(
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
