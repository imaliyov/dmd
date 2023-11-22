#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example of the HODMD usage.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from dmd import dmd, plot_tools


def main():

    # Setup a data array where each spatial component has a different frequency

    nspace = 50
    ntime = 10000
    time_step = 0.01

    # Spatial and time grids
    x_grid = np.linspace(0.0, 1.0, nspace)
    time_grid = np.linspace(0.0, time_step * ntime, ntime)

    # The function: sin(2*pi*x*t)
    # The frequency of sin increases with x
    wmin = 0.5
    sin_array = np.sin(2 * np.pi * (x_grid[:, np.newaxis] + wmin) * time_grid[np.newaxis, :])

    # constant frequency
    #sin_array = np.zeros((nspace, ntime), dtype=float)
    #sin_array[:, :] = np.sin(2 * np.pi * time_grid[np.newaxis, :])
    print(sin_array.shape)

    #itraj_list = [0, 100, 500, 999]
    itraj_list = [0, 24, 49]
    plot_tools.plot_trajectories(sin_array, time_step, itraj_list)

    plot_tools.plot_x_t_heatmap(sin_array, time_step, title='Original')

    # First, apply the standard DMD
    ndmd = 5000

    dmd_run = dmd.dmd(sin_array[:, :ndmd])

    dmd_run.sig_threshold = 1e-13
    dmd_run.time_step = time_step
    dmd_run.verbose = False

    print(dmd_run)
    dmd_run.compute_modes()

    # Plot the singular values
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    plt.subplots_adjust(left=0.15)
    plot_tools.plot_singular(ax, dmd_run.sigma_full_array, dmd_run.sig_threshold, dmd_run.rank, lw=2.5, ls='-', color='gray')
    plt.show()

    # Plot the DMD frequencies
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    plt.subplots_adjust(left=0.15)
    plot_tools.plot_omegas_on_plane(ax, dmd_run.omega_array, mode_ampl=dmd_run.mode_ampl_array, fsize=30, title=False)
    plt.show()

    # Number of snapshots for prediction
    dmd_run.nsnap_extrap = 1000

    # Extrapolate the dynamics
    sin_array_extrap = np.real(dmd_run.extrapolate())
    print(dmd_run.timings)

    plot_tools.plot_trajectories(sin_array, time_step, itraj_list, data_extrap=sin_array_extrap, ndmd=ndmd)

    # Now, apply the HODMD
    HODMD_order = 2
    HODMD_shift = 2

    ndmd = 5000
    hodmd_run = dmd.dmd(sin_array[:, :ndmd])

    hodmd_run.order = HODMD_order
    hodmd_run.ntshift = HODMD_shift

    hodmd_run.sig_threshold = 1e-13
    hodmd_run.time_step = time_step
    hodmd_run.verbose = False

    print(hodmd_run)
    hodmd_run.compute_modes()

    # Plot the singular values
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    plt.subplots_adjust(left=0.15)
    plot_tools.plot_singular(ax, hodmd_run.sigma_full_array, hodmd_run.sig_threshold, hodmd_run.rank, lw=2.5, ls='-', color='gray')
    plt.show()

    # Plot the DMD frequencies
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    plt.subplots_adjust(left=0.15)
    plot_tools.plot_omegas_on_plane(ax, hodmd_run.omega_array, mode_ampl=hodmd_run.mode_ampl_array, fsize=30, title=False)
    plt.show()

    # Number of snapshots for prediction
    hodmd_run.nsnap_extrap = 10000

    # Extrapolate the dynamics
    sin_array_extrap = np.real(hodmd_run.extrapolate())

    print(hodmd_run.timings)

    plot_tools.plot_trajectories(sin_array, time_step, itraj_list, data_extrap=sin_array_extrap, ndmd=ndmd)


if __name__ == "__main__":
    main()