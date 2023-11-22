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


