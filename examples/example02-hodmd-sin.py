#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example of the HODMD usage.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from dmd import dmd, plot_tools, generate_data


def main():

    # Setup a data array where each spatial component has a different frequency

    nspace = 50
    ntime = 10000
    time_step = 0.01

    sin_array, x_grid, time_grid = generate_data.sin_array(nspace, ntime, time_step)

    itraj_list = [0, 24, 49]
    plot_tools.plot_trajectories(sin_array, time_step, itraj_list, xlim=(0, 20))

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    plot_tools.plot_x_t_heatmap(fig, ax, sin_array, time_step, title='Original')
    ax.set_xlim(0, 20)
    plt.show()

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
    ax.set_title('Standard DMD: Singular values')
    plt.show()

    # Plot the DMD frequencies
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    plt.subplots_adjust(left=0.15)
    plot_tools.plot_omegas_on_plane(ax, dmd_run.omega_array, mode_ampl=dmd_run.mode_ampl_array, fsize=30, title=False)
    ax.set_title('Standard DMD: DMD frequencies')
    plt.show()

    # Number of snapshots for prediction
    dmd_run.nsnap_extrap = 1000

    # Extrapolate the dynamics
    sin_array_extrap = np.real(dmd_run.extrapolate())
    print(dmd_run.timings)

    plot_tools.plot_trajectories(sin_array, time_step, itraj_list, data_extrap=sin_array_extrap, ndmd=ndmd, title='Standard DMD extrapolation')

    # Standard DMD does not work well for this example, therefore we will use HODMD
    HODMD_order = 2
    HODMD_shift = 2

    hodmd_run = dmd.dmd(sin_array[:, :ndmd])

    hodmd_run.HODMD_order = HODMD_order
    hodmd_run.HODMD_ntshift = HODMD_shift

    hodmd_run.sig_threshold = 1e-13
    hodmd_run.time_step = time_step
    hodmd_run.verbose = False

    print(hodmd_run)
    hodmd_run.compute_modes()

    # Plot the singular values
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    plt.subplots_adjust(left=0.15)
    plot_tools.plot_singular(ax, hodmd_run.sigma_full_array, hodmd_run.sig_threshold, hodmd_run.rank, lw=2.5, ls='-', color='gray')
    ax.set_title('HODMD: Singular values')
    plt.show()

    # Plot the DMD frequencies
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    plt.subplots_adjust(left=0.15)
    plot_tools.plot_omegas_on_plane(ax, hodmd_run.omega_array, mode_ampl=hodmd_run.mode_ampl_array, fsize=30, title=False)
    ax.set_title('HODMD: DMD frequencies')
    plt.show()

    # Number of snapshots for prediction
    hodmd_run.nsnap_extrap = 10000

    # Extrapolate the dynamics
    sin_array_extrap = np.real(hodmd_run.extrapolate())

    print(hodmd_run.timings)

    plot_tools.plot_trajectories(sin_array, time_step, itraj_list, data_extrap=sin_array_extrap, ndmd=ndmd, title='HODMD extrapolation')


if __name__ == "__main__":
    main()