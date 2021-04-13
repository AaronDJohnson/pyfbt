# pyfbt
 Frequency based Teukolsky black hole perturbation code written in Python


## Installation
 Run `pip install git+https://github.com/AaronDJohnson/pyfbt.git` in the environment that you want to install the package in.

## Example usage
 Basic functions are currently implemented in the `Orbit` class. Instantiate an orbit with `orbit = Orbit(aa, slr, ecc, x)`.

 To see adiabatic constants of the orbit (for example), we use
 `orbit.constants`.

 For the energy emitted to infinity, use `orbit.energy_inf(ell, em, en, kay)`.

 See the .ipynb file included for some basic examples.

## Contribute
 This code is a work in progress. If you find it useful, then I encourage you to fork it and help me add features to it. Tests and documentation coming (hopefully) soon.

## Acknowledgements
 Some parts of this code are directly translated from the Mathematica code in the Black Hole Perturbation Toolkit [bhptoolkit.org](bhptoolkit.org).