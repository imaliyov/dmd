"""
Test the standard DMD algorithm, first-order based on ./examples/example01-moving-gaussian.py, Gaussian with moving mean and standard deviation.
"""

import numpy as np
import pytest
from dmd import dmd, plot_tools, generate_data


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
    gauss_extrap = dmd_run.extrapolate()

    idx_sort = np.argsort(np.abs(dmd_run.mode_ampl_array))[::-1]
    idx_sort = idx_sort[np.argsort(dmd_run.mode_ampl_array[idx_sort].real)[::-1]]

    mode_array_10_desired = np.load('./refs/dmd_Gaus_mode_array_10.npy')
    omega_array_desired = np.load('./refs/dmd_Gaus_omega_array.npy')
    mode_ampl_array_desired = np.load('./refs/dmd_Gaus_mode_ampl_array.npy')
    sigma_full_array_desired = np.load('./refs/dmd_Gaus_sigma_full_array.npy')
    gauss_extrap_999_desired = np.load('./refs/dmd_Gaus_extrap_999.npy')

    #np.testing.assert_allclose(dmd_run.mode_array[:, idx_sort[:10]], mode_array_10_desired, atol=1e-10)
    #np.testing.assert_allclose(dmd_run.omega_array[idx_sort], omega_array_desired, atol=1e-10)
    #np.testing.assert_allclose(dmd_run.mode_ampl_array[idx_sort], mode_ampl_array_desired, atol=1e-10)
    np.testing.assert_allclose(dmd_run.sigma_full_array, sigma_full_array_desired, atol=1e-10)
    np.testing.assert_allclose(gauss_extrap[999, :], gauss_extrap_999_desired, atol=1e-10)
