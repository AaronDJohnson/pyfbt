import numpy as np
from scipy.special import loggamma

def poch(z, m):
    return np.exp(loggamma(z + m) - loggamma(z))