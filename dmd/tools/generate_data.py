"""
Functions to generate sample data for tests and examples.
"""
import numpy as np


def moving_gaussian(nspace, ntime, time_step, speed=0.05):
    """
    A Gaussian moving in time.
    Normalized Gaussian form: f(x) = exp(-(x - mu) ** 2 / (2 * sigma ** 2)) / (sqrt(2 * pi) * sigma)
    where mu is the mean and sigma is the standard deviation.

    Parameters
    ----------

    nspace : int
        Number of spatial points.

    ntime : int
        Number of time points.

    time_step : float
        Time step.

    speed : float, optional
        Speed of the shift of the Gaussian.

    plot : bool, optional
        If True, plot the Gaussian.

    Returns
    -------

    gauss : ndarray
        Array containing the Gaussian, first index is space and second is time.

    x_grid : ndarray
        Spatial grid.

    time_grid : ndarray
        Time grid.
    """

    gauss = np.zeros((nspace, ntime), dtype=float)

    # Spatial and time grids
    x_grid = np.linspace(0.0, 1.0, nspace)
    time_grid = np.linspace(0.0, time_step * ntime, ntime)

    # Gaussian parameters
    mu_start = 0.1
    sigma_start = 0.01
    sigma_end = 0.08

    # Laws of motion for mean and standard deviation
    mu_array = mu_start + speed * time_grid
    sigma_array = sigma_start + (sigma_end - sigma_start) * time_grid / (time_step * ntime)

    # Gaussian with moving mean and standard deviation
    gauss[:, :] = np.exp(-(x_grid[:, None] - mu_array[None, :]) ** 2 / (2 * sigma_array[None, :] ** 2)) / (np.sqrt(2 * np.pi) * sigma_array[None, :])

    return gauss, x_grid, time_grid


def sin_array(nspace, ntime, time_step, wmin=0.5):
    """
    A sin function with increasing frequency in space.

    Parameters
    ----------

    nspace : int
        Number of spatial points.

    ntime : int
        Number of time points.

    time_step : float
        Time step.

    wmin : float, optional
        Minimum frequency.

    Returns
    -------

    sin_array : ndarray
        Array containing the sin function, first index is space and second is time.

    x_grid : ndarray
        Spatial grid.

    time_grid : ndarray
        Time grid.
    """

    # Spatial and time grids
    x_grid = np.linspace(0.0, 1.0, nspace)
    time_grid = np.linspace(0.0, time_step * ntime, ntime)

    # The function: sin(2*pi*x*t)
    # The frequency of sin increases with x
    sin_array = np.sin(2 * np.pi * (x_grid[:, np.newaxis] + wmin) * time_grid[np.newaxis, :])

    return sin_array, x_grid, time_grid
