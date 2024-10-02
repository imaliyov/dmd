"""
Test DMD and HODMD based on the RT-TDDFT data from the H2 molecule.
"""

import h5py
import numpy as np
import pytest
from dmd import dmd, plot_tools, generate_data
from dmd.tools.utils import sort_complex_array


# Helper functions, will not be tested
def compute_dipole_mol(dipole_ao, c_matrix_complete):
    """
    Compute the dipole moment in the molecular basis
    """

    nstate = c_matrix_complete.shape[1]

    dipole_mol = np.zeros((nstate, nstate, 3), dtype=float)
    for idir in range(3):
        dipole_mol[:, :, idir] = c_matrix_complete.T @ dipole_ao[:, :, idir] @ c_matrix_complete

    return dipole_mol


def compute_dipole_from_PD_MO(PD_MO_time):
    """
    Compute the dipole moment from the P x D matrix
    """

    nsnap = PD_MO_time.shape[2]
    dipole_time = np.zeros((3, nsnap), dtype=float)

    for idir in range(3):
        dipole_time[idir, :] = - np.sum(PD_MO_time[idir, :, :], axis=0)

    return dipole_time


def read_rt_tddft(filepath):
    """
    Read the RT-TDDFT HDF5 file

    Parameters
    ----------

    filepath : str
        Path to the HDF5 file

    Returns
    -------

    time_data : np.ndarray
        Array with the time-dependent data from the RT-TDDFT calculation.
        Shape is (3, nocc*nvirt, nsnap)

    time_step : float
        Time step in atomic units.

    nocc : int
        Number of occupied orbitals.

    nvirt : int
        Number of virtual orbitals.
    """

    cdyna_file = h5py.File(filepath, 'r')

    # Number of time snapshots
    nsnap = cdyna_file['nsnap'][()]

    # Number of states
    nstate = cdyna_file['nstate'][()]

    # Time step
    time_step = cdyna_file['time_step'][()]

    # Dipole moment in AO basis
    dipole_ao = cdyna_file['dipole_ao'][()]
    dipole_ao = np.transpose(dipole_ao, axes=(2, 1, 0))

    # Wavefunction coefficients at t=0
    c_matrix_complete = cdyna_file['c_matrix_complete_0_real'][()][0, :, :].T

    # Dipole moment in MO basis
    dipole_mol = compute_dipole_mol(dipole_ao, c_matrix_complete)

    # PD in MO basis
    p_matrix_MO_block_0 = cdyna_file['p_matrix_MO_block']['snap_0']

    nspin, nvirt, nocc = p_matrix_MO_block_0.shape
    p_matrix_MO_block_time = np.zeros((nocc, nvirt, nsnap), dtype=float)

    for isnap in range(nsnap):
        p_matrix_MO_block_time[:, :, isnap] = cdyna_file['p_matrix_MO_block'][f'snap_{isnap}'][()][0, :, :].T

    dipole_mol = compute_dipole_mol(dipole_ao, c_matrix_complete)
    # Multiply the off-diagonal elements by 2
    mask = ~np.eye(nstate, dtype=bool)[:, :]
    dipole_mol2 = dipole_mol.copy()
    dipole_mol2[mask] *= 2.0
    # Reshape the dipole moment similarly, but without time
    dipole_mol_block = dipole_mol2[:nocc, nocc:, :]
    dipole_mol_block = np.reshape(dipole_mol_block, (nocc * nvirt, 3))

    # Compute the PD matrix
    time_data = np.zeros((3, nocc * nvirt, nsnap), dtype=float)
    au_debye = 2.54174623105

    for idir in range(3):
        time_data[idir, :, :] = p_matrix_MO_block_time * dipole_mol_block[:, idir, np.newaxis] * au_debye

    cdyna_file.close()

    print(f'{time_data.shape=}')
    print(f'{time_data.shape[0]} directions, {time_data.shape[1]} "spatial" points, and {time_data.shape[2]} time snapshots')

    return time_data, time_step, nocc, nvirt


@pytest.fixture(scope="module")
def data():
    return read_rt_tddft('./refs/H2_rt_tddft.h5')


def test_dmd_rt_tddft(data):

    time_data, time_step, nocc, nvirt = data
    idir = 0

    idmd_first = 15
    ndmd = 110
    nsnap_extrap = 1500

    dmd_obj = dmd.dmd(time_data[idir, :, idmd_first:idmd_first + ndmd])

    dmd_obj.time_step = time_step
    dmd_obj.verbose = False

    dmd_obj.order = 1
    dmd_obj.ntshift = 1

    dmd_obj.compute_modes()

    dmd_obj.nsnap_extrap = nsnap_extrap

    time_data_extrap = dmd_obj.extrapolate()

    time_data_extrap_desired = np.load('./refs/dmd_H2_extrap_array.npy')

    # DMD Frequencies
    omega_array = dmd_obj.omega_array.copy()
    omega_array_desired = np.load('./refs/dmd_H2_omega_array.npy')
    omega_array, idx = sort_complex_array(omega_array)
    omega_array_desired, idx = sort_complex_array(omega_array_desired)
    np.testing.assert_allclose(omega_array, omega_array_desired, atol=1e-8)

    # Mode amplitudes
    mode_ampl_array = dmd_obj.mode_ampl_array.copy()
    mode_ampl_array_desired = np.load('./refs/dmd_H2_mode_ampl_array.npy')
    mode_ampl_array, idx = sort_complex_array(mode_ampl_array)
    mode_ampl_array_desired, idx = sort_complex_array(mode_ampl_array_desired)
    np.testing.assert_allclose(mode_ampl_array, mode_ampl_array_desired, atol=1e-8)

    np.testing.assert_allclose(time_data_extrap[:10, :], time_data_extrap_desired, atol=1e-8)