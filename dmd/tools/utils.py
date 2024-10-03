"""
Utilies for molDMD
"""
import sys
import h5py
import numpy as np


def symmetrize_array(nrow, array_triu):
    """
    Return the full symmetrized array from the upper triangle
    Shape: ntriu, nsnap
    """

    triu_idx = np.triu_indices(nrow)
    row_idx, col_idx = triu_idx

    nsnap = array_triu.shape[-1]

    array = np.zeros((nrow, nrow, nsnap), dtype=array_triu.dtype)
    array[row_idx, col_idx, :] = array_triu
    array[col_idx, row_idx, :] = array_triu

    return array


def assign_attributes(attr_name_list, attr_dict, obj):
    """
    Assign attributes to an object

    Parameters
    ----------

    attr_name_list : list
        List of attribute names

    attr_dict : dict
        Dictionary of attributes, it must contain the keys
        corresponding to the attribute names

    obj : object
        Object to assign attributes to
    """

    for attr_name in attr_name_list:

        if attr_name not in attr_dict.keys():
            raise ValueError(f'Attribute {attr_name} not found in the attribute dictionary')

        setattr(obj, attr_name, attr_dict[attr_name])

    return


def get_size(array, name='array', dump=True):
    """
    Get size of an array in MB

    Parameters
    ----------
    array : numpy array
        Array.

    name :
        Name of array, optional.

    dump :
        Specifies whether to print out or not the size

    Returns
    -------
    size : float
        Size in MB.
    """

    # Check if the array is a numpy array and calculate size accordingly
    if isinstance(array, np.ndarray):
        # Calculate size in bytes for numpy array
        size_bytes = array.size * array.itemsize
    else:
        # Use sys.getsizeof() for other types of arrays
        size_bytes = sys.getsizeof(array)

    size_kb = size_bytes / 1024.0

    if size_kb < 1024.0:

        size = size_kb
        unit = 'KB'

    elif size_kb / 1024.0 < 1024.0:

        size_mb = size_kb / 1024.0
        size = size_mb
        unit = 'MB'

    else:

        size_gb = size_kb / 1024.0**2
        size = size_gb
        unit = 'GB'

    if dump:
        print(f'===Size of {name:<10} {str(array.shape):<12} {str(array.dtype):<8}: {size:6.3f} {unit}')

    return size, unit


def sort_complex_array(array):
    sorted_indices = np.lexsort((np.angle(array), -np.abs(array)))
    #sorted_indices = np.lexsort((np.real(array), np.imag(array), -np.abs(array)))
    return array[sorted_indices], sorted_indices


# Utils for reading RT-TDDFT data from HDF5 files
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


def read_rt_tddft_data(filepath):
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

    return time_data, time_step

