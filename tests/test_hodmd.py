"""
Test DMD based on the RT-TDDFT data from the H2 molecule.
"""

import h5py
import numpy as np
import pytest
from dmd import dmd, plot_tools, generate_data
from dmd.tools.utils import sort_complex_array, read_rt_tddft_data
import matplotlib.pyplot as plt


@pytest.fixture(scope="module")
def hodmd_run():
    """
    Run DMD once for the H2 data, return the DMD object for further testing.
    """

    time_data, time_step = read_rt_tddft_data('./refs/H2_rt_tddft.h5')

    # Setup a DMD object
    idir = 0
    idmd_first = 15
    ndmd = 110
    nsnap_extrap = 1500
    hodmd_run_loc = dmd.dmd(time_data[idir, :, idmd_first:idmd_first + ndmd])

    # Setup DMD parameters
    hodmd_run_loc.time_step = time_step
    hodmd_run_loc.verbose = False
    # HODMD: order = 2
    hodmd_run_loc.order = 2
    hodmd_run_loc.ntshift = 1
    hodmd_run_loc.nsnap_extrap = nsnap_extrap
    hodmd_run_loc.compute_modes()

    return hodmd_run_loc


def test_shapes(hodmd_run):

    np.testing.assert_equal(hodmd_run.rank, 30)
    np.testing.assert_equal(hodmd_run.snap_array.shape, (181, 110))
    np.testing.assert_equal(hodmd_run.mode_array.shape, (181, 30))


def test_singular_values(hodmd_run):

    sigma_full_array_disred = np.load('./refs/hodmd_H2_sigma_full_array.npy')
    np.testing.assert_allclose(hodmd_run.sigma_full_array, sigma_full_array_disred, atol=1e-8)


def test_mode_frequencies(hodmd_run):

    omega_array = hodmd_run.omega_array.copy()
    omega_array_desired = np.load('./refs/hodmd_H2_omega_array.npy')
    omega_array, _ = sort_complex_array(omega_array)
    omega_array_desired, _ = sort_complex_array(omega_array_desired)
    np.testing.assert_allclose(omega_array, omega_array_desired, atol=1e-8)


def test_hodmd_modes(hodmd_run):

    mode_ampl_array = hodmd_run.mode_ampl_array.copy()
    mode_ampl_array_desired = np.load('./refs/hodmd_H2_mode_ampl_array.npy')
    mode_array = hodmd_run.mode_array.copy()
    mode_array_desired = np.load('./refs/hodmd_H2_mode_array.npy')

    # Sort the complex mode amplitudes
    mode_ampl_array_sorted, idx_sort1 = sort_complex_array(mode_ampl_array)
    mode_ampl_array_desired_sorted, idx_sort2 = sort_complex_array(mode_ampl_array_desired)

    # Take the absolute value of the complex mode amplitudes because for degenrate modes the sign can be different
    np.testing.assert_allclose(np.abs(mode_ampl_array_sorted), np.abs(mode_ampl_array_desired_sorted), atol=1e-8)

    # Multiply the modes by their amplitudes
    Phi_b = np.einsum('il,l->il', mode_array, mode_ampl_array)
    Phi_b_desired = np.einsum('il,l->il', mode_array_desired, mode_ampl_array_desired)
    np.testing.assert_allclose(Phi_b, Phi_b_desired, atol=1e-8)

    mode_array = mode_array[:, idx_sort1]
    mode_array_desired = mode_array_desired[:, idx_sort2]
    np.testing.assert_allclose(mode_array, mode_array_desired, atol=1e-8)


def test_extrapolation(hodmd_run):

    time_data_extrap = hodmd_run.extrapolate().real
    # Extrapolation array is a dipole moment, sum over the states of PD matrix
    time_data_extrap = np.sum(time_data_extrap, axis=0)
    time_data_extrap_desired = np.load('./refs/hodmd_H2_extrap_array.npy')
    np.testing.assert_allclose(time_data_extrap, time_data_extrap_desired, atol=1e-8)