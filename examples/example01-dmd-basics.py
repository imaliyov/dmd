#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Basic example of the DMD usage.
"""

import numpy as np
import matplotlib.pyplot as plt
from dmd import dmd


def moving_gaussian(nspace, ntime, time_step, speed=0.05, plot=False):
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

    if plot:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        cmap = plt.cm.get_cmap('coolwarm')  # Colormap for changing color from blue to red

        for i in range(ntime):
            if i % 10 != 0:
                continue

            t = i * time_step
            label = f"Time: {t:.2f}"
            color = cmap(i / (ntime - 1))  # Calculate color based on time index
            ax.plot(x_grid, gauss[:, i], label=label, color=color)

        ax.set_xlabel("coordinate x", fontsize=16)
        ax.set_ylabel("f(x)", fontsize=16)

        ax.legend()
        plt.show()

        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        im = ax.imshow(gauss, cmap=cmap, origin='lower', aspect='auto', extent=[0, ntime * time_step, 0, 1])
        ax.set_xlabel("Time", fontsize=16)
        ax.set_ylabel("Position", fontsize=16)
        ax.set_title("Moving Gaussian Heatmap", fontsize=16)
        fig.colorbar(im, ax=ax, label='Gaussion value')
        plt.show()

    return gauss


def main():
    
    # Number of spatial points
    nspace = 1000
    # Number of time points
    ntime = 100
    # Time step
    time_step = 0.1
    
    # Create a sample data, the first dimension is space and second is time
    gauss = moving_gaussian(nspace, ntime, time_step, plot=True)

    # Number of snapshots for DMD
    ndmd = 30

    # Enforce the rank
    enforce_rank = 10
    # OR setup a threshold for the singular values
    #sig_threshold = 1e-10

    # Create a DMD object
    dmd_run = dmd.dmd(gauss[:, :ndmd])

    # Setup the DMD parameters
    #dmd_run.sig_threshold = sig_threshold
    dmd_run.enforce_rank = enforce_rank
    dmd_run.time_step = time_step

    # Number of snapshots for prediction
    dmd_run.nsnap_extrap = 200

    # Run the DMD algorithm
    dmd_run.compute_modes()

    # Save the DMD info to YAML
    dmd_run.info_to_yaml(path='./')

    # Print the timings
    print(dmd_run.timings)




if __name__ == "__main__":
    main()
