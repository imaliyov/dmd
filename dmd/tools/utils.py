"""
Utilies for molDMD
"""
import sys
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

    size_kb = sys.getsizeof(array)/1024.0

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
