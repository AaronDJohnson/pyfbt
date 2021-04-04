import numpy as np
import matplotlib.pyplot as plt

from geodesic.geodesic import calc_consts, calc_radial_roots, calc_polar_roots
from geodesic.geodesic import calc_mino_freqs, calc_boyer_freqs, find_omega

from swsh.swsh import calc_swsh_eq

from renormnu.renormnu import find_nu

from flux.flux import flux_inf

# from swsh.swsh import calc_swsh_eq

# from renormnu.functions import calc_nu
# from teukolsky.homteuk import py_find_Bin

# from flux.flux import flux_inf
# from data import make_folder, save_data, save_wave_data

class Orbit:
    def __init__(self, aa=None, slr=None, ecc=None, x=None, double=True):
        """
        Initialize the Orbit object

        Input:
            aa (float): massive black hole spin parameter
            slr (float): semi-latus rectum
            ecc (float): eccentricity
            x (float): cos of the inclination angle
            double (bool): if True the energy values are doubled to account for neg. em values
        """
        self.double = double

        if aa is None:
            print('aa was not given a value and has defaulted to 0 (SC black hole).')
            self.aa = 0
        else:
            self.aa = aa

        if slr is None:
            print('slr was not given a value and has defaulted to 6.')
            self.slr = 6
        else:
            self.slr = slr

        if ecc is None:
            print('ecc was not given a value and has defaulted to 0 (circular orbit).')
            self.ecc = 0
        else:
            self.ecc = ecc

        if x is None:
            print('x was not given a value and has defaulted to 1 (equatorial orbit).')
            self.x = 1
        else:
            self.x = x

        # compute anything that doesn't require mode info (this doesn't take long):

        self.params = {'aa':self.aa, 'slr':self.slr, 'ecc':self.ecc, 'x':self.x}
        self.En, self.Lz, self.Q = calc_consts(self.aa, self.slr, self.ecc, self.x)
        self.constants = {'En':self.En, 'Lz':self.Lz, 'Q':self.Q}
        self.r1, self.r2, self.r3, self.r4 = calc_radial_roots(self.aa, self.slr, self.ecc, self.x)
        self.radial_roots = {'r1':self.r1, 'r2':self.r2, 'r3':self.r3, 'r4':self.r4}
        self.zp, self.zm = calc_polar_roots(self.aa, self.slr, self.ecc, self.x)
        self.polar_roots = {'zp':self.zp, 'zm':self.zm}
        self.ups_r, self.ups_theta, self.ups_phi, self.gamma = calc_mino_freqs(self.aa, self.slr, self.ecc, self.x)
        self.mino_freqs = {'ups_r':self.ups_r, 'ups_theta':self.ups_theta, 'ups_phi':self.ups_phi, 'gamma':self.gamma}
        self.omega_r, self.omega_theta, self.omega_phi = calc_boyer_freqs(self.aa, self.slr, self.ecc, self.x)
        self.boyer_freqs = {'omega_r':self.omega_r, 'omega_theta':self.omega_theta, 'omega_phi':self.omega_phi}


    def energy_inf(self, ell=None, em=None, en=None, kay=None):
        """
        Set the (ell, em) mode that we want to compute and compute E_inf associated with it.

        Inputs:
            ell (int): harmonic
            em (int): harmonic
            kay (int): polar mode
            en (int): radial mode
        """
        # deal with ell, em, en, kay as None:
        if ell is None:
            self.ell = 2
        else:
            self.ell = ell
        
        if em is None:
            self.em = self.ell
        elif em == 0:
            print('DC modes are not available yet.')  # nu throws an error with em = 0
            self.em = 0
        else:
            self.em = em
        
        if en is None:
            self.en = 0
        else:
            self.en = en
        
        if kay is None:
            self.kay = 0
        else:
            self.kay = kay


        self.omega = self.en * self.omega_r + self.em * self.omega_phi + self.kay * self.omega_theta
        self.eigen, self.Slm, self.Slmd, self.Slmdd = calc_swsh_eq(0, self.aa, self.omega, self.ell, self.em)
        self.nu = find_nu(self.aa, self.omega, self.eigen, self.ell, self.em, tol=1e-20)
        self.nu = float(self.nu[0]) + 1j * float(self.nu[1])
        self.E_inf, __ = flux_inf(self.nu, self.eigen, self.slr, self.ecc, self.aa,
                                  self.x, self.ups_r, self.ups_theta, self.ups_phi,
                                  self.gamma, self.omega, self.em, self.Lz, self.En,
                                  self.Slm, self.Slmd, self.Slmdd, self.omega_r,
                                  self.r1, self.r2, self.r3, self.r4, self.zp, self.zm)
        if self.double:
            self.E_inf = 2 * self.E_inf

        return self.E_inf
        





    # def energy_inf(self):
    #     """
    #     Computes the energy emitted to an observer at infinity for a given mode.
        
    #     WARNING: currently this only works for equatorial orbits (x**2 = 1)
    #     """

    #     if self.x**2 != 1:
    #         print('Off axis orbits are not currently supported.')
    #     elif self.harmonic is None:
    #         print('Unable to compute energy values without setting modes.')
    #     # else:
    #         # self.harmonic_key = tuple((self.harmonic['ell'], self.harmonic['em']))
    #         # self.mode_key = tuple((self.mode['kay'], self.mode['en']))
    #         # print(self.harmonic_key)
    #         # print(self.mode_key)
    #         # if self.harmonic_key in self.mode_content.keys():
    #         #     if self.mode_key in self.mode_content[self.harmonic_key].keys():
    #         #         print('Mode has already been computed and has been skipped.')
    #     # TODO (aaron): add some logic to skip modes that have already been computed
    #     else:
    #         self.omega = fp_find_omega(self.omega_r, self.omega_theta, self.omega_phi, self.em, self.kay, self.en)
    #         self.re_nu, self.im_nu = calc_nu(self.aa, self.slr, self.ecc, self.x, self.ell, self.en, self.em, self.kay)

    #         self.eigen, self.Slm, self.Slmd, self.Slmdd = calc_swsh_eq(self.aa, self.omega, self.ell, self.em, -2)

    #         self.nu = self.re_nu + 1j * self.im_nu
    #         self.Bin = py_find_Bin(self.re_nu, self.im_nu, self.eigen, self.aa, self.omega, self.em)

    #         self.mode_dependent = {'gw_freq':self.omega, 'eigen':self.eigen, 'nu':self.nu, 'Bin':self.Bin}

    #         self.e_inf, self.Z = flux_inf(self.nu, self.Bin, self.eigen, self.slr, self.ecc, self.aa, self.ups_r, self.ups_theta,
    #                                         self.ups_phi, self.gamma, self.omega, self.em, self.Lz, self.En, self.Slm, self.Slmd,
    #                                         self.Slmdd, self.omega_r, self.r1, self.r2, self.r3, self.r4, self.zp, self.zm)
    #         if np.isnan(self.e_inf):  # TODO (aaron): investigate why this can be nan
    #             print('Some value is NaN - this needs to be investigated.')
    #             self.e_inf = 0
    #             print('Energy at infinity stored as zero:', self.e_inf)

    #         elif self.double:
    #             self.e_inf = 2 * self.e_inf
    #             print('Energy at infinity:', self.e_inf)

    #         # put everything in a dict to save later as a json/hdf5
    #         self.harmonic_key = tuple((self.harmonic['ell'], self.harmonic['em']))
    #         self.mode_key = tuple((self.mode['kay'], self.mode['en']))

    #         if self.harmonic_key in self.mode_content.keys():
    #             self.mode_content[self.harmonic_key][self.mode_key] = self.e_inf
    #         else:
    #             self.mode_content[self.harmonic_key] = {self.mode_key: self.e_inf}


    # def mode_search(self, ells=None, kays=None, ens=None, ellem_same=False):
    #     """
    #     Search through modes in given vector ranges

    #     WARNING: Currently only computes the kay = 0 mode

    #     Inputs:
    #         ells (int): ell values to compute
    #         ems (int): em values to compute
    #         kays (int): kay values to compute
    #         ens (int): en values to compute
    #     """
    #     for ell in ells:
    #         if ellem_same:
    #             em = ell
    #             for kay in kays:
    #                 for en in ens:
    #                     self.set_modes(ell=ell, em=em, en=en, kay=kay)
    #                     print(self.harmonic)
    #                     print(self.mode)
    #                     self.energy_inf()
    #                     print('Einf =', self.e_inf)
    #         else:
    #             for em in np.arange(ell + 1):
    #                 for kay in kays:
    #                     for en in ens:
    #                         self.set_modes(ell=ell, em=em, en=en, kay=kay)
    #                         print(self.harmonic)
    #                         print(self.mode)
    #                         if em == 0 and en == 0 and kay == 0:
    #                             print('Finding the DC component (m, k, n) = (0, 0, 0) is not supported currently.')
    #                             continue
    #                         self.energy_inf()
    #                         print('Einf =', self.e_inf)


    # # def to_numpy(self, ell, em):
    # #     self.x = []
    # #     self.y = []
    # #     lengths = []
    # #     for key in low_ecc.mode_content.keys():
    # #         lengths.append(self.mode_content[key]
    # #     for i in range(-20, 21):
    # #         x.append(i)
    # #         y.append(low_ecc.mode_content[2,2][0, i])
    # #         tot += low_ecc.mode_content[2,2][0, i]
    # #     x = np.array(x)
    # #     y = np.array(y)
    # #     tot = np.array(tot)
    # #     y = y / tot
    # #     return x, y
    # # def plot_ens(ell=None, em=None):
    # #     """
    # #     Use matplotlib to plot the radial modes stored in the data
    # #     """
    # #     self.mode_content


    # def geodesic_eq(self, num_pts=10000, end=2*np.pi):
    #     self.psi = np.linspace(0, end, num=num_pts)
    #     self.geodesic_array = np.zeros((len(self.psi), 4))
    #     for i in range(len(self.psi)):
    #         self.zeta = self.psi[i]
    #         self.t, self.r, self.theta, self.phi = py_calc_equatorial_coords(self.zeta, self.ups_r, self.ups_theta,
    #                                                                          self.ups_phi, self.gamma, self.r1,
    #                                                                          self.r2, self.r3,
    #                                                                          self.r4, self.zp, self.zm, self.En,
    #                                                                          self.Lz, self.aa, self.slr, self.ecc)
    #         self.geodesic_array[i,:] = self.t, self.r, self.theta, self.phi
    #     return self.geodesic_array

    # def geodesic_psi(self, psi):
    #     self.t, self.r, self.theta, self.phi = py_calc_equatorial_coords(psi, self.ups_r, self.ups_theta,
    #                                                                          self.ups_phi, self.gamma, self.r1,
    #                                                                          self.r2, self.r3,
    #                                                                          self.r4, self.zp, self.zm, self.En,
    #                                                                          self.Lz, self.aa, self.slr, self.ecc)
    #     return self.t, self.r, self.theta, self.phi














