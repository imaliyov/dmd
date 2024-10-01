#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Basic example of the DMD usage.
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from dmd import dmd, plot_tools, generate_data
from matplotlib import colormaps

plt.rcParams.update(plot_tools.plotparams)


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

    # Save data for tests reference
    save_data = True

    # Number of spatial points
    nspace = 1000
    # Number of time points
    ntime = 200
    # Time step
    time_step = 0.1

    # Create a sample data, the first dimension is space and second is time
    gauss, x_grid, time_grid = generate_data.moving_gaussian(nspace, ntime, time_step)

    # Plot the moving Gaussian
    plot_gaussian(gauss, x_grid, time_step)

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    plot_tools.plot_x_t_heatmap(fig, ax, gauss, time_step, title='Original')
    plt.show()

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
    ax.set_title('DMD frequencies')
    plt.show()

    # Plot DMD modes
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    plot_tools.plot_modes(ax, dmd_run.mode_array, nfirst=5, mode_ampl=dmd_run.mode_ampl_array, leg_loc='upper left')
    ax.set_title('DMD modes')
    ax.set_xlabel('coordinate x')
    ax.set_ylabel('DMD mode')
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

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    plot_tools.plot_x_t_heatmap(fig, ax, gauss_extrap, time_step, ndmd=ndmd, title='DMD extrapolation')
    plt.show()

    #itraj_list = [500, 5000, 9999]
    itraj_list = [50, 500, 999]
    plot_tools.plot_trajectories(gauss, time_step, itraj_list, data_extrap=gauss_extrap, ndmd=ndmd)

    # Save data to numpy binary files
    if save_data:

        # Initial data, first 5 snapshots
        with open('dmd_Gaus_data_5.npy', 'wb') as f:
            np.save(f, gauss[:, :5])

        # Sort based on mode amplitudes and then based on the real part of amplitude
        idx_sort = np.argsort(np.abs(dmd_run.mode_ampl_array))[::-1]
        idx_sort = idx_sort[np.argsort(dmd_run.mode_ampl_array[idx_sort].real)[::-1]]
        #idx_sort = np.argsort(np.abs(dmd_run.mode_ampl_array))[::-1]

        # First 10 DMD modes
        with open('dmd_Gaus_mode_array_10.npy', 'wb') as f:
            np.save(f, dmd_run.mode_array[:, idx_sort[:10]])

        # Mode amplitudes
        with open('dmd_Gaus_mode_ampl_array.npy', 'wb') as f:
            np.save(f, dmd_run.mode_ampl_array[idx_sort])

        # DMD frequencies
        with open('dmd_Gaus_omega_array.npy', 'wb') as f:
            np.save(f, dmd_run.omega_array[idx_sort])

        # Singular values
        with open('dmd_Gaus_sigma_full_array.npy', 'wb') as f:
            np.save(f, dmd_run.sigma_full_array)

        # Extrapolated data, last snapshot
        with open('dmd_Gaus_extrap_999.npy', 'wb') as f:
            np.save(f, gauss_extrap[999, :])


if __name__ == "__main__":
    main()
