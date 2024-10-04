"""
Tools for Extended DMD (EDMD) analysis. Adapted from https://github.com/fnueske/KoopmanLib
"""
import numpy as np


def sample_rff_gaussian(d, p, sigma, seed=None):
    """ Draw a sample from the spectral density for a Gaussian kernel.
    Parameters:
    -----------
    d, int:         dimension of the state space for Gaussian kernel
    p, int:         number of samples from spectral density
    sigma, float:   bandwidth of the Gaussian kernel
    Returns:
    --------
    Omega (d, p):   p samples drawn from d-dimensional spectral density
    """

    if seed is not None:
        np.random.seed(seed)

    return (1.0 / sigma) * np.random.randn(d, p)


class RFF_Finite():
    def __init__(self, omega):
        """ Basis Class for Random Fourier features centered at specific
            frequencies.
            Parameters:
            -----------
            omega, (d, p): p d-dimensional frequency vectors defining the basis set
        """
        # Extract dimension and number of random features:
        self.d, self.p = omega.shape
        self.omega = omega

    def __call__(self, x):
        """ Evaluation of the basis set at data points x.
            Parameters:
            -----------
            x, nd-array (d, m):
                m positions in d-dimensional Euclidean space.
            Returns:
            --------
            PhiX, nd-array (p, m):
                Values of all RFF basis functions at all data points.
        """
        array = np.vstack([np.cos(np.dot(self.omega.T, x)),
                           np.sin(np.dot(self.omega.T, x))])

        return array