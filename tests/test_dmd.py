"""
Test the standard DMD algorithm, first-order based on ./examples/example01-moving-gaussian.py, Gaussian with moving mean and standard deviation.
"""

import numpy as np
import pytest
from dmd import dmd, plot_tools, generate_data
from dmd.tools.utils import sort_complex_array


def test_moving_gaussian():
    """
    Test the function to generate data
    """

    gauss, x_grid, time_grid = generate_data.moving_gaussian(nspace=1000, ntime=200, time_step=0.1)

    # First 5 snapshots
    gauss_5_desired = np.load('./refs/dmd_Gaus_data_5.npy')

    np.testing.assert_allclose(gauss[:, :5], gauss_5_desired, atol=1e-10)


def test_dmd():

    gauss, x_grid, time_grid = generate_data.moving_gaussian(nspace=1000, ntime=200, time_step=0.1)

    ndmd = 145
    dmd_run = dmd.dmd(gauss[:, :ndmd])

    dmd_run.sig_threshold = 1e-10
    dmd_run.rank = None
    dmd_run.time_step = 0.1
    dmd_run.verbose = False
    dmd_run.nsnap_extrap = 200

    dmd_run.compute_modes()
    gauss_extrap = dmd_run.extrapolate().real

    omega_array_desired = np.load('./refs/dmd_Gaus_omega_array.npy')
    sigma_full_array_desired = np.load('./refs/dmd_Gaus_sigma_full_array.npy')
    gauss_extrap_desired = np.load('./refs/dmd_Gaus_extrap_10_100_200.npy')

    omega_array_desired, idx = sort_complex_array(omega_array_desired)
    omega_array, idx = sort_complex_array(dmd_run.omega_array)

    np.testing.assert_allclose(omega_array[:10], omega_array_desired[:10], atol=1e-3, rtol=1e-4)
    np.testing.assert_allclose(dmd_run.sigma_full_array, sigma_full_array_desired, atol=1e-10, rtol=1e-8)
    np.testing.assert_allclose(gauss_extrap[[10, 100, 200], :], gauss_extrap_desired, atol=1e-6, rtol=1e-4)
