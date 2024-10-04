"""
Test extended DMD (EDMD) based on the RT-TDDFT data from the H2 molecule.
"""

import h5py
import numpy as np
import pytest
from dmd import dmd, plot_tools, generate_data
from dmd.tools.utils import sort_complex_array, read_rt_tddft_data


@pytest.fixture(scope="module")
def edmd_run():
    """
    Run DMD once for the H2 data, return the DMD object for further testing.
    """

    time_data, time_step = read_rt_tddft_data('./refs/H2_rt_tddft.h5')

    # Setup a DMD object
    idir = 0
    idmd_first = 15
    ndmd = 110
    edmd_run_loc = dmd.dmd(time_data[idir, :, idmd_first:idmd_first + ndmd])

    # Setup DMD parameters
    edmd_run_loc.time_step = time_step
    edmd_run_loc.verbose = False
    # hodmd: order = 2
    edmd_run_loc.order = 2
    edmd_run_loc.ntshift = 1

    # EDMD
    edmd_run_loc.EDMD = True
    edmd_run_loc.EDMD_nfreq = 1000
    edmd_run_loc.EDMD_sigma = 10.0
    # For reproducibility, set the random seed
    edmd_run_loc.EDMD_random_seed = 123

    edmd_run_loc.nsnap_extrap = 1500
    edmd_run_loc.compute_modes()

    return edmd_run_loc


def test_edmd_shapes(edmd_run):

    np.testing.assert_equal(edmd_run.rank, 35)
    np.testing.assert_equal(edmd_run.snap_array.shape, (181, 110))
    np.testing.assert_equal(edmd_run.mode_array.shape, (181, 35))


def test_edmd_singular_values(edmd_run):

    sigma_full_array_disred = np.load('./refs/edmd_H2_sigma_full_array.npy')
    np.testing.assert_allclose(edmd_run.sigma_full_array, sigma_full_array_disred, atol=1e-8)


def test_edmd_mode_frequencies(edmd_run):

    omega_array = edmd_run.omega_array.copy()
    omega_array_desired = np.load('./refs/edmd_H2_omega_array.npy')
    omega_array, _ = sort_complex_array(omega_array)
    omega_array_desired, _ = sort_complex_array(omega_array_desired)
    np.testing.assert_allclose(omega_array, omega_array_desired, atol=1e-4)


def test_edmd_mode_amplitudes(edmd_run):

    mode_ampl_array = edmd_run.mode_ampl_array.copy()
    mode_ampl_array_desired = np.load('./refs/edmd_H2_mode_ampl_array.npy')

    # Sort the complex mode amplitudes
    mode_ampl_array_sorted, _ = sort_complex_array(mode_ampl_array)
    mode_ampl_array_desired_sorted, _ = sort_complex_array(mode_ampl_array_desired)
    # Take the absolute value because for degenerate modes the sign can be different
    np.testing.assert_allclose(np.abs(mode_ampl_array_sorted), np.abs(mode_ampl_array_desired_sorted), atol=1e-8)


def test_edmd_modes(edmd_run):

    mode_array = edmd_run.mode_array.copy()
    mode_array_desired = np.load('./refs/edmd_H2_mode_array.npy')

    # Sort based on absolute value
    mode_array_sorted, _ = sort_complex_array(np.sum(mode_array, axis=0))
    mode_array_desired_sorted, _ = sort_complex_array(np.sum(mode_array_desired, axis=0))
    # Take the absolute value because for degenerate modes the sign can be different
    np.testing.assert_allclose(np.abs(mode_array_sorted), np.abs(mode_array_desired_sorted), atol=1e-4)


def test_edmd_extrapolation(edmd_run):

    time_data_extrap = edmd_run.extrapolate().real
    # Extrapolation array is a dipole moment, sum over the states of PD matrix
    time_data_extrap = np.sum(time_data_extrap, axis=0)
    time_data_extrap_desired = np.load('./refs/edmd_H2_extrap_array.npy')
    np.testing.assert_allclose(time_data_extrap, time_data_extrap_desired, atol=1e-8)