#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Basic example of the DMD usage.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from dmd import dmd, plot_tools
from matplotlib import colormaps


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


def plot_gaussian(gauss, x_grid, time_step, gauss_extrap=None):
    """
    Plot the moving Gaussian.

    Parameters
    ----------

    gauss : ndarray
        Array containing the Gaussian.

    time_step : float
        Time step.
    """

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    cmap = colormaps['coolwarm']

    ntime = gauss.shape[1]
    for i in range(ntime):
        if i % 10 != 0:
            continue

        t = i * time_step
        color = cmap(i / (ntime - 1))  # Calculate color based on time index
        ax.plot(x_grid, gauss[:, i], color=color)

        if gauss_extrap is not None:
            ax.plot(x_grid, gauss_extrap[:, i], color='gray', linestyle='--')

    ax.set_xlabel("coordinate x", fontsize=24)
    ax.set_ylabel("f(x)", fontsize=24)

    if gauss_extrap is not None:
        ax.legend(handles=[Line2D([0], [0], color='gray', linestyle='--', label='Extrapolated')])

    # Plot colormap on the right side
    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm.set_array(time_step * np.arange(ntime))
    fig.colorbar(sm, ax=ax, label='Time')

    plt.show()


def main():
    
    # Number of spatial points
    nspace = 10000
    # Number of time points
    ntime = 200
    # Time step
    time_step = 0.1
    
    # Create a sample data, the first dimension is space and second is time
    gauss, x_grid, time_grid = moving_gaussian(nspace, ntime, time_step)

    # Plot the moving Gaussian
    plot_gaussian(gauss, x_grid, time_step)
    plot_tools.plot_x_t_heatmap(gauss, time_step, title='Original')

    # Number of snapshots for DMD
    ndmd = 145

    # Enforce the rank
    # enforce_rank = 10

    # OR setup a threshold for the singular values
    enforce_rank = None
    sig_threshold = 1e-10

    # Create a DMD object
    dmd_run = dmd.dmd(gauss[:, :ndmd])

    # Setup the DMD parameters
    dmd_run.sig_threshold = sig_threshold
    dmd_run.rank = enforce_rank
    dmd_run.time_step = time_step
    dmd_run.verbose = True

    # Print the information
    print(dmd_run)

    # Run the DMD algorithm
    dmd_run.compute_modes()

    # Plot the singular values
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    plot_tools.plot_singular(ax, dmd_run.sigma_full_array, dmd_run.sig_threshold, dmd_run.rank, lw=2.5, ls='-', color='gray')
    plt.show()

    # Plot the DMD frequencies
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    plot_tools.plot_omegas_on_plane(ax, dmd_run.omega_array, mode_ampl=dmd_run.mode_ampl_array, fsize=30, title=False)
    plt.show()

    # Plot DMD modes
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    plot_tools.plot_modes(ax, dmd_run.mode_array, nfirst=5, mode_ampl=dmd_run.mode_ampl_array, leg_loc='upper left')
    plt.show()

    # Save the DMD info to YAML
    dmd_run.info_to_yaml(path='./')

    # Print the timings
    print(dmd_run.timings)

    # Number of snapshots for prediction
    dmd_run.nsnap_extrap = 200

    # Extrapolate the dynamics
    gauss_extrap = dmd_run.extrapolate()

    # Take the real part
    gauss_extrap = np.real(gauss_extrap)

    plot_gaussian(gauss, x_grid, time_step, gauss_extrap=gauss_extrap)
    plot_tools.plot_x_t_heatmap(gauss_extrap, time_step, ndmd=ndmd, title='DMD extrapolation')

    itraj_list = [500, 5000, 9999]
    plot_tools.plot_trajectories(gauss, time_step, itraj_list, data_extrap=gauss_extrap, ndmd=ndmd)


if __name__ == "__main__":
    main()
