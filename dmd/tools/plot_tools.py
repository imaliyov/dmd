#!/usr/bin/env python3
"""
Tools for plotting
"""

import sys
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator,
                               FormatStrFormatter,
                               AutoMinorLocator,
                               StrMethodFormatter)

# Set common figure parameters
plotparams = {'figure.figsize': (10, 8),
                  'axes.grid': False,
                  'lines.linewidth': 2.8,
                  'axes.linewidth': 1.1,
                  'lines.markersize': 10,
                  'xtick.bottom': True,
                  'xtick.top': True,
                  'xtick.direction': 'in',
                  'xtick.minor.visible': True,
                  'ytick.left': True,
                  'ytick.right': True,
                  'ytick.direction': 'in',
                  'ytick.minor.visible': True,
                  'figure.autolayout': False,
                  'mathtext.fontset': 'dejavusans', # 'cm' 'stix'
                  'mathtext.default' : 'it',
                  'xtick.major.size': 4.5,
                  'ytick.major.size': 4.5,
                  'xtick.minor.size': 2.5,
                  'ytick.minor.size': 2.5,
                  'legend.handlelength': 3.0,
                  'legend.shadow'     : False,
                  'legend.markerscale': 1.0 ,
                  'font.size': 20}

plt.rcParams.update(plotparams)


def plot_singular(ax, sigma, sig_threshold, rank, **kwargs):
    """
    Plot the singluar values on an axis ax
    """

    ax.semilogy(sigma, 'o-')
    fsize = 25
    ax.set_xlabel(r'index of value $j$', fontsize=fsize)
    ax.set_ylabel(r'Singular value $\sigma_j$', fontsize=fsize)
    ax.axhline(sig_threshold, **kwargs)
    ax.axvline(rank - 0.5, **kwargs)

    # grid
    ax.xaxis.grid(True, which='major')

    # x axis format
    ax.tick_params(axis='x', which='minor', bottom=False, top=False)
    #ax.minorticks_off()
    #ax.xaxis.set_major_locator(MultipleLocator(1.0))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))


def plot_omegas_on_plane(ax, omega_array, mode_ampl=None, sort=True, fsize=20, title=True, plot_first=None, index_modes=False, s=100, **kwargs):
    """
    Plot the real and imaginary parts of DMD frequencies on a 2D plane
    """

    if mode_ampl is not None:
        if sort:
            idx_sort = np.argsort(np.abs(mode_ampl))[::-1]

            omega_array = omega_array[idx_sort]
            mode_ampl = mode_ampl[idx_sort]

            #print(f'{omega_array[0] = }')

    if plot_first is not None:
         omega_array = omega_array[:plot_first]
         mode_ampl = mode_ampl[:plot_first]

    ax.grid(True, which='major', zorder=-5, alpha=0.5)

    # scale according to mode_ample
    if mode_ampl is None:
        size = s

    else:
        smin = 1.0
        smax = 150.0

        vmin = np.min(np.abs(mode_ampl)) + 1e-11
        vmax = np.max(np.abs(mode_ampl)) + 1e-11

        norm = LogNorm(vmin=vmin, vmax=vmax)

        size = norm(np.abs(mode_ampl)+1e-11) * smax

        size[size < smin] = smin

     #ax.scatter(omega_array.real * 1e3, omega_array.imag * 1e3, zorder=3.0, s=size, **kwargs)

    if index_modes:
        for imode in range(omega_array.shape[0]):
            ax.plot(omega_array[imode].real * 1e3, omega_array[imode].imag * 1e3, marker='o', ls='', ms=10, label=imode+1, **kwargs)
    else:
        ax.scatter(omega_array.real, omega_array.imag, s=size, **kwargs)

    if index_modes:
        ax.legend()

    if title:
        ax.set_title(r'$\mathrm{Im}\{\omega_0\} = $' + f'{omega_array[0].imag:.3e}')

    ax.set_xlabel(r'Re($\omega^{\mathrm{DMD}}$)', fontsize=fsize)
    ax.set_ylabel(r'Im($\omega^{\mathrm{DMD}}$)', fontsize=fsize)


def apply_bar_range(ax, xmin, xmax, lines=False):
    """
    Draw a bar range
    """

    ax.axvspan(xmin, xmax, facecolor='gray', alpha=0.5)

    if lines:
        ax.axvline(xmin, color='green', ls='-', lw=1.0, zorder=3)
        ax.axvline(xmax, color='green', ls='-', lw=1.0, zorder=3)


def plot_modes(ax, mode_array, nfirst=None, mode_ampl=None, leg_loc=None, **kwargs):
    """Plot DMD modes

    Parameters
    ----------

    ax : matplotlib axis
         Axis to plot on

    mode_array : ndarray
         Array containing the DMD modes

    nfirst : int
         Number of modes to plot

    mode_ampl : ndarray
         Array containing the mode amplitudes. If provided, the modes are sorted according to the amplitude.
    """

    mode_array_plot = mode_array.copy()

    if mode_ampl is not None:
        idx_sort = np.argsort(np.abs(mode_ampl))[::-1]

        mode_array_plot = mode_array[:, idx_sort]
        mode_ampl = mode_ampl[idx_sort]

    nplot = nfirst if nfirst is not None else mode_array_plot.shape[1]

    for imode in range(nplot):
        # Plot the real part
        ax.plot(mode_array_plot[:, imode].real, label=f'{imode+1} real', **kwargs)
        # Plot the imaginary part
        ax.plot(mode_array_plot[:, imode].imag, label=f'{imode+1} imag', ls='--', **kwargs)

    loc = 'best' if leg_loc is None else leg_loc
    ax.legend(loc=loc)


def plot_x_t_heatmap(fig, ax, data, time_step, ndmd=None, title=None):
    """
    Plot the heatmap of spatial-temporal data

    Parameters
    ----------

    fig : matplotlib figure
        Figure to plot on

    ax : matplotlib axis
        Axis to plot on

    data : ndarray
        Array of data to plot

    time_step : float
        Time step

    ndmd : int, optional
        Number of snapshots used for DMD

    title : str, optional
        Title of the plot
    """

    cmap = plt.cm.get_cmap('coolwarm')  # Colormap for changing color from blue to red
    ntime = data.shape[1]
    im = ax.imshow(data, cmap=cmap, origin='lower', aspect='auto', extent=[0, ntime * time_step, 0, 1])
    ax.set_xlabel("Time", fontsize=24)
    ax.set_ylabel("Spatial coordinate", fontsize=24)
    fig.colorbar(im, ax=ax, label='Value')

    if ndmd is not None:
        ax.axvline(x=ndmd * time_step, color='black', linestyle='--')

    if title is not None:
        ax.set_title(title)


def plot_trajectories(data, time_step, itraj_list, data_extrap=None, ndmd=None, xlim=None, ylim=None, title=None):
    """Plot selected trajectories.

    Parameters
    ----------

    data : ndarray
        Array of data to plot. First dimension is spatial, second is temporal.

    data_extrap : ndarray
        Array of extrapolated data to plot

    time_step : float
        Time step

    ndmd : int
        Number of snapshots used for DMD

    itraj_list : list
        List of trajectories to plot (spatial indices for data)

    xlim : tuple, optional
        Limits for x axis

    ylim : tuple, optional
        Limits for y axis

    """

    ntime = data.shape[1]
    time_grid = np.linspace(0.0, time_step * ntime, ntime)

    if data_extrap is not None:
        ntime_extrap = data_extrap.shape[1]
        time_grid_extrap = np.linspace(0.0, time_step * ntime_extrap, ntime_extrap)

    for itraj in itraj_list:
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.1)
        ax.plot(time_grid, data[itraj, :], label='Original')

        if data_extrap is not None:
            ax.plot(time_grid_extrap, data_extrap[itraj, :], label='Extrapolated', linestyle='--')

        ax.set_xlabel("Time", fontsize=24)
        ax.set_ylabel("Value", fontsize=24)
        ax.set_title(f"Trajectory {itraj}", fontsize=24)

        if ndmd is not None:
            apply_bar_range(ax, 0, ndmd * time_step, lines=True)

        if data_extrap is not None:
            ax.legend()

        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)

        if title is not None:
            ax.set_title(title)

        plt.show()


