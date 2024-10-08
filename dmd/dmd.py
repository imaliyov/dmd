#!/usr/bin/env python3
"""
Dynamical Mode Decomposition (DMD) functions
Object-oriented
"""

import os
import sys
from scipy.linalg import schur
import pickle
import numpy as np
import yaml
import matplotlib.pyplot as plt

from dmd.tools.utils import get_size
from dmd.tools.timing import TimingGroup, measure_runtime_and_calls
from dmd.tools.edmd_tools import sample_rff_gaussian, RFF_Finite


class dmd():
    """
    DMD class

    Attributes
    ----------

    HODMD_order : int
        HODMD_order of DMD. High-HODMD_order DMD (HODMD) is HODMD_order > 1.

    sig_threshold : float
        Threshold for the singular values.

    verbose : bool
        Output info and timing about the DMD procedure.

    nspace : int
        Num of points in space

    nsnap : int
        Number of time snapshots used for DMD.

    nsnap_extrap : int
        Number of time snapshots for the extrapolation.

    time_step : float
        Time step between the snapshots.

    save_svd_matrices : bool
        Save the full SVD matrices.

    rank : int
        Rank of the SVD of the X1 matrix. Dimension of the reduced space.

    snap_array : array
        The array of snapshots, complex or real.

    sigma_array : array
        Singular values of the SVD.

    mode_array : array
        DMD modes.

    mode_ampl_array : array
        Mode amplitudes.

    omega_array : array
        DMD frequencies.

    Methods
    -------

    compute_modes() : Compute the DMD modes and frequencies

    extrapolate() : Extrapolate the DMD dynamics
    """

    def __init__(self, snap_array, apply_scaling=True):
        """
        Initialize the DMD class

        Parameters
        ----------

        snap_array : array
            The array of snapshots, complex or real
        """

        self.timings = TimingGroup('DMD')

        self.name = 'dmd'

        # Scale factor to keep the data within the range
        self.scale_factor = 1.0
        self.apply_scaling = apply_scaling

        # High-HODMD_order DMD (HODMD) parameters
        self.HODMD_order = 1
        self.HODMD_ntshift = 1

        # Extended DMD (EDMD) parameters
        self.EDMD = False
        self.EDMD_sigma = 10
        self.EDMD_nfreq = 500
        self.EDMD_random_seed = None

        # Measure-preserving DMD (mpDMD)
        self.mpDMD = False

        self.sig_threshold = 1e-7
        self.rank = None

        self.snap_array = snap_array
        # self.snap_array = snap_array.astype(np.complex128)
        if self.apply_scaling:
            self.scale_factor = np.max(np.abs(self.snap_array))
            snap_array_copy = self.snap_array.copy()
            self.snap_array = snap_array_copy / self.scale_factor

        self.time_step = None

        self.save_svd_matrices = False

        self.verbose = False
        self.nspace, self.nsnap = snap_array.shape

        self.nsnap_extrap = self.nsnap + 1

        self.sigma_array = None
        self.mode_array = None
        self.mode_ampl_array = None

        self.omega_array = None

    def __str__(self):
        """
        Output the information about the instance
        """

        info = '\n'
        info += f"{'=' * 60}\n"
        info += f"{' DMD parameters ':=^60}\n"
        info += f"{'=' * 60}\n"
        info += "\n"

        # Basic
        info += f'{"Number of spatial points":>30}: {self.nspace}\n'
        info += f'{"Number of time snapshots":>30}: {self.nsnap}\n'

        info += f'{"Scale factor (internal)":>30}: {self.scale_factor:.3e}\n'

        if self.rank is not None:
            info += f'{"Max number of singular values":>30}: {self.rank}\n'
        else:
            info += f'{"Singular values threshold":>30}: {self.sig_threshold}\n'

        # HODMD
        info += f'{"HODMD_order":>30}: {self.HODMD_order}\n'

        if self.HODMD_order > 1:
            info += f'{"HODMD_ntshift":>30}: {self.HODMD_ntshift}\n'

        # EDMD
        info += f'{"EDMD":>30}: {self.EDMD}\n'
        if self.EDMD:
            info += f'{"(EDMD) Number of frequencies":>30}: {self.EDMD_nfreq}\n'
            info += f'{"(EDMD) Sigma":>30}: {self.EDMD_sigma}\n'
            info += f'{"(EDMD) Random seed":>30}: {self.EDMD_random_seed}\n'

        # mpDMD
        info += f'{"mpDMD":>30}: {self.HODMD_ntshift}\n'

        return info

    def vprint(self, message):
        """
        Print if self.verbose
        """

        if self.verbose:
            print(message)

    def check_attributes(self, names):
        for name in names:
            if getattr(self, name) is None:
                raise ValueError(f'{name} must be initialized')

    def get_info_dict(self):
        """
        Get a dictionary with the DMD info, no large arrays
        """

        info_dict = {}
        for attr in dir(self):
            value = getattr(self, attr)
            if not callable(value) \
                and isinstance(value, (int, str, float)) \
                    and not attr.startswith('_'):

                if isinstance(value, float):
                    value = float(value)

                info_dict[attr] = value

        info_dict['timings'] = self.timings.to_dict()

        return info_dict

    def get_attribute_sizes(self):
        """
        Get sizes of all the attributes
        """
        attribute_sizes = {}
        for attr_name, attr_value in self.__dict__.items():
            if isinstance(attr_value, np.ndarray):
                # Calculate size in bytes for numpy array
                size_bytes = attr_value.size * attr_value.itemsize
            else:
                # Use sys.getsizeof() for other types of arrays
                size_bytes = sys.getsizeof(attr_value)

            attribute_sizes[attr_name] = size_bytes

        # Convert sizes to MB
        attribute_sizes_mb = {attr_name: size / (1024 ** 2) for attr_name, size in attribute_sizes.items()}

        # Sort attribute sizes by descending HODMD_order
        sorted_sizes = sorted(attribute_sizes_mb.items(), key=lambda x: x[1], reverse=True)

        print('\nAttribute sizes (MB):')
        for attr_name, size_mb in sorted_sizes[:3]:
            print(f"{attr_name:>30}: {size_mb:.3f}")
        print('\n')

        return attribute_sizes

    def info_to_yaml(self, path='./'):
        """
        Dump the DMD info into a YAML file
        """

        filepath = os.path.join(path, f'{self.name}.yml')

        info_dict = self.get_info_dict()

        with open(filepath, 'w') as file:
            yaml.dump(info_dict, file)

    def to_pickle(self, path='./'):
        """
        Dumpe the instance of the Class into a pickle file
        """

        filepath = os.path.join(path, f'{self.name}.pkl')

        self.vprint(f'Writing DMD run to {filepath}')

        if self.verbose:
            self.get_attribute_sizes()

        with open(filepath, "wb") as file:
            pickle.dump(self, file)

    @measure_runtime_and_calls
    def setup_snap_array_slow(self):

        """
        Test
        """

        nk = self.nspace
        ns = self.HODMD_order
        ng = self.HODMD_ntshift
        ncut = self.nsnap

        R0 = self.snap_array

        nh = nk * ns # the height of Y
        nl = np.floor((ncut-ns-1)/ng)+1 # the length of Y
        nl = int(nl)

        Y = np.zeros((nh, nl))
        Yt = np.zeros((nh, nl))

        for l in range(nl):
            for s in range(ns):
                Y[s*nk:(s+1)*nk, l] = R0[:, l*ng+s]
                Yt[s*nk:(s+1)*nk, l] = R0[:, l*ng+s+1]

        return Y, Yt

    @measure_runtime_and_calls
    def setup_snap_array(self):
        """
        Setup the X1 and X2 matrices for DMD or HODMD
        """

        if self.HODMD_order > 1:
            # Number of snaps for one of the shifted
            # X matrices
            nsnap_HODMD = self.nsnap - self.HODMD_order + 1

            # Spatial dimention of extended array
            nspace_HODMD = self.nspace * self.HODMD_order

            # Construct the extended array of snaps
            snap_array_HODMD = np.zeros((nspace_HODMD, nsnap_HODMD), dtype=self.snap_array.dtype)

            for iorder in range(self.HODMD_order):
                imin = self.nspace * iorder
                imax = self.nspace * (iorder + 1)
                snap_array_HODMD[imin:imax, :] = \
                    self.snap_array[:, iorder:self.nsnap-self.HODMD_order+iorder + 1]

            x1_array = snap_array_HODMD[:, :-1]
            x2_array = snap_array_HODMD[:, 1:]

            if self.HODMD_ntshift > 1:
                x1_array = x1_array[:, ::self.HODMD_ntshift]
                x2_array = x2_array[:, ::self.HODMD_ntshift]

            if self.verbose:
                get_size(snap_array_HODMD, 'snap HODMD')

        else:
            x1_array = self.snap_array[:, : - 1]
            x2_array = self.snap_array[:, 1:]

        return x1_array, x2_array

    def compute_modes(self):
        """
        Compute the DMD modes and frequencies
        """

        self.vprint('\nComputing DMD modes...\n')

        self.check_attributes(['time_step'])

        if self.verbose:
            get_size(self.snap_array, 'snap matrix')

        # Step 0: setup the X1 ans X2 snapshot matrices
        x1_array, x2_array = self.setup_snap_array()

        # For EDMD, x1_array will be psi1_array and x2_array will be psi2_array
        if self.EDMD:
            x2_array_original = x2_array.copy()
            freq_array = sample_rff_gaussian(x1_array.shape[0], self.EDMD_nfreq, self.EDMD_sigma, seed=self.EDMD_random_seed)
            Phi_array = RFF_Finite(freq_array)
            x1_array = Phi_array(x1_array)
            x2_array = Phi_array(x2_array)

            if self.verbose:
                print('\n   EDMD random Fourier features arrays')
                get_size(x1_array, 'x1_array'); get_size(x2_array, 'x2_array'); print()

        # For debug purposes
        compute_full_a = False
        #compute_full_a = True
        if compute_full_a:
            print('Computing the full A matrix...')
            a_full_array = x2_array @ np.linalg.pinv(x1_array)

            # plot the full A matrix
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            # log scale
            sm = ax.imshow(np.log10(np.abs(a_full_array)), cmap='coolwarm')
            fig.colorbar(sm, ax=ax)
            plt.show()
            exit()

        if self.mpDMD:
            self.compute_modes_mpDMD(x1_array, x2_array)
        else:
            self.compute_modes_standard(x1_array, x2_array)

        # Step 6: For EDMD, we need to compute the matrix B,
        # which is the best approximation of the original X as linear combination
        if self.EDMD:
            self.vprint('EDMD: Compute the matrix B...\n')
            with self.timings.add('pinv(mode_array_inv@Psi)') as t:
                inv_array = np.linalg.pinv(self.mode_array_inv @ x2_array)

            with self.timings.add('compute matrix B') as t:
                self.mode_array = x2_array_original @ inv_array

        # Reduce the mode_array to the actual number
        # of spatial points
        self.mode_array = self.mode_array[:self.nspace, :]

    @measure_runtime_and_calls
    def compute_modes_standard(self, x1_array, x2_array):
        """
        Core function of DMD. Compute the DMD spatial modes
        and frequencies.

        Parameters
        ----------
        self : object
            The instance of the class
        """

        # Step 1: SVD on X1
        self.vprint('SVD on X1...')
        with self.timings.add('SVD') as t:
            u_array, sigma_array, v_array = \
                np.linalg.svd(x1_array, full_matrices=False)
            v_array = v_array.conj().T

        # Keep only the significant singular values
        if self.rank is None:
            self.rank = sigma_array[sigma_array > self.sig_threshold].shape[0]

        self.vprint(f'Rank: {self.rank}')
        self.vprint(f'Done: {t.t_delta:.3f} s\n')

        # Save the SVD full matrices
        if self.save_svd_matrices:
            self.u_full_array = u_array.copy()
            self.v_full_array = v_array.copy()

        v_array = v_array[:, :self.rank]
        u_array = u_array[:, :self.rank]

        self.sigma_full_array = sigma_array
        self.sigma_array = sigma_array[:self.rank]

        # Inverse of the diagonal Sigma matrix
        sigma_inv = np.diag(1.0 / self.sigma_array)

        # Step 2: Compute the A matrix in the reduces space
        self.vprint('Compute the A matrix...\n')
        with self.timings.add('compute matrix A') as t:
            a_array = u_array.conj().T @ x2_array \
                @ v_array @ sigma_inv

        if self.verbose:
            get_size(a_array, name='matrix A'); get_size(u_array, name='matrix U'); get_size(v_array, name='matrix V'); print()

        # Step 3: Diago of the Koopman matrix A
        self.vprint('Diago of the A matrix...\n')
        with self.timings.add('diago of matrix A') as t:
            eig_val, eig_vec = np.linalg.eig(a_array)

        # DMD frequencies, physicists' notation: exp(i omega t)
        self.omega_array = -1j * np.log(eig_val) / self.time_step

        # To avoid divergence: set the imag part of omega to zero if negative
        self.omega_array[np.imag(self.omega_array) < 0] = \
            np.real(self.omega_array[np.imag(self.omega_array) < 0])

        # Step 4: Compute the DMD modes
        self.vprint('Compute the DMD modes Psi...\n')
        with self.timings.add('compute Psi') as t:
            self.mode_array = x2_array @ v_array \
                @ sigma_inv @ eig_vec

        if self.verbose:
            get_size(self.mode_array, name='mode_array')

        # if self.HODMD_order > 1:
        #     self.mode_array = self.mode_array[:self.nspace, :]
        #     x1_array = x1_array[:self.nspace, :]

        # Step 5: Compute mode amplitudes
        self.vprint('Compute mode amplitudes b...\n')
        with self.timings.add('mode ampl (pinv)') as t:

            self.mode_array_inv = np.linalg.pinv(self.mode_array)
            self.mode_ampl_array = self.mode_array_inv @ x1_array[:, 0]

    @measure_runtime_and_calls
    def compute_modes_mpDMD(self, x1_array, x2_array):
        """
        Measure-preserving DMD (mpDMD) method.
        Adapted from Colbrook, SIAM Journal on Numerical Analysis, 61, 3, 1585-1608, 2023.
        """

        # Step 0: Prerequisites
        # uniform sampling
        weight = 1.0 / x1_array.shape[1]
        weight_array = np.eye(x1_array.shape[1]) * weight

        # M x N, according to the paper
        Psi1_array = x1_array.conj().T
        Psi2_array = x2_array.conj().T

        # Step 1: Compute G and A

        #G_array = Psi1_array.conj().T @ weight_array @ Psi1_array
        G_array = Psi1_array.conj().T @ Psi1_array

        #A_array = Psi1_array.conj().T @ weight_array @ Psi2_array
        A_array = Psi1_array.conj().T @ Psi2_array

        # Step 2: Compute the SVD of G^(-1/2) A^H G^(-1/2)
        # G is Hermitian

        # 2.1: Compute G^(-1/2)
        #G_eigval, G_eigvec = np.linalg.eigh(G_array)
        #G_inv_sqrt = G_eigvec @ np.diag(1.0 / np.sqrt(G_eigval)) @ G_eigvec.conj().T

        U_G_array, S_G_array, Vh_G_array = np.linalg.svd(G_array, hermitian=True)
        rank_G = S_G_array[S_G_array > 1e-10].shape[0]
        U_G_array = U_G_array[:, :rank_G]
        S_G_array = S_G_array[:rank_G]
        Vh_G_array = Vh_G_array[:rank_G, :]

        self.vprint(f'Rank of G: {rank_G}')

        S_G_inv_sqrt = np.diag(1.0 / np.sqrt(S_G_array))

        G_inv_sqrt = U_G_array @ S_G_inv_sqrt @ U_G_array.conj().T

        # 2.2: Compute the matrix product G^(-1/2) A^H G^(-1/2)
        GAG_array = G_inv_sqrt @ A_array.conj().T @ G_inv_sqrt

        # 2.3: Compute the SVD of GAG, notations according to Colbrook
        U1_GAG_array, S_GAG_array, U2h_GAG_array = np.linalg.svd(GAG_array)

        self.sigma_full_array = S_GAG_array
        self.rank = self.sigma_full_array.shape[0]

        # Step 3: Diago of U2 U1h
        U2_U1h_array = U2h_GAG_array.conj().T @ U1_GAG_array.conj().T

        eigval_Lambda, eigvec_V = np.linalg.eig(U2_U1h_array)
        # Schur to enforce the orthonormal basis
        #T_array, Z_array = schur(U2_U1h_array, output='complex')
        #eigvec_V = Z_array
        #eigval_Lambda = np.diag(T_array)

        # DMD frequencies, physicists' notation: exp(i omega t)
        self.omega_array = -1j * np.log(eigval_Lambda) / self.time_step

        # To avoid divergence: set the imag part of omega to zero if negative
        self.omega_array[np.imag(self.omega_array) < 0] = \
            np.real(self.omega_array[np.imag(self.omega_array) < 0])

        # Compute the DMD modes
        self.mode_array = G_inv_sqrt @ eigvec_V

        if self.verbose:
            get_size(self.mode_array, name='mode_array')

        # Compute the mode amplitudes
        self.mode_array_inv = np.linalg.pinv(self.mode_array)
        self.mode_ampl_array = self.mode_array_inv @ x1_array[:, 0]

    @measure_runtime_and_calls
    def extrapolate(self):
        """
        Extrapolate the DMD dynamics. compute_modes method
        must be ran first.

        .. math::
            x(t) = \sum_l^r b_l \phi_l e^{i \omega_l t}
        """

        self.vprint(f'Extrapolating dynamics for {self.nsnap_extrap} snaps...')
        self.check_attributes(['mode_array', 'omega_array', 'mode_ampl_array'])
        self.vprint(f'Memory required for DMD_traj: {self.nspace * self.nsnap_extrap * 16 / 1024**3:.5f} GB')

        # Step 1: Compute the time array t
        time_array = self.time_step * np.arange(self.nsnap_extrap)

        # Step 2: Compute i * omega_l * t
        with self.timings.add('i x omega x t') as t:
            omega_time = \
                np.einsum('r,t->rt', 1j * self.omega_array, time_array)

        # Step 3: Compute b_l * exp(i * omega_l * t)
        with self.timings.add('b_l x exp(i x omega x t)') as t:
            omega_time = self.mode_ampl_array[:, np.newaxis] * np.exp(omega_time)

        # Step 4: Compute the DMD trajectory: sum_l^r b_l * exp(i * omega_l * t)
        with self.timings.add('DMD traj') as t:
            # mode_array: nspace x nmode
            # omega_time: nmode x nsnap_extrap
            DMD_traj = self.mode_array @ omega_time

        # Step 5: Normalize the DMD trajectory with initial data
        with self.timings.add('normalize') as t:
            norm_ratio_array = \
                np.linalg.norm(self.snap_array[:, :], axis=1) * self.scale_factor / \
                np.linalg.norm(DMD_traj[:, :self.nsnap - self.HODMD_order + 1], axis=1)

            DMD_traj[:, :] *= norm_ratio_array[:, np.newaxis]

        if self.verbose:
            get_size(DMD_traj, 'DMD_traj')

        return DMD_traj

    @measure_runtime_and_calls
    def extrapolate_space_sum(self):
        """
        Extrapolate the DMD dynamics summed over spatial coordinate.
        compute_modes() method must be ran first.

        .. math::
            x(t) = \sum_l^r b_l \phi_l e^{i \omega_l t}

        Return \sum_k x_k(t)
        """

        self.vprint(f'Extrapolating dynamics for {self.nsnap_extrap} snaps...')
        self.check_attributes(['mode_array', 'omega_array', 'mode_ampl_array'])
        self.vprint(f'Memory required for DMD_traj: {self.nspace * self.nsnap_extrap * 16 / 1024**3:.5f} GB')

        if self.mode_array is None:
            raise ValueError('compute_modes() must be ran before extrapolation')

        sum_mode_array = np.sum(self.mode_array, axis=0)

        # Step 1: Compute the time array t
        time_array = self.time_step * np.arange(self.nsnap_extrap)

        # Step 2: Compute i * omega_l * t
        with self.timings.add('i x omega x t') as t:
            omega_time = \
                np.einsum('r,t->rt', 1j * self.omega_array, time_array)

        # Step 3: Compute b_l * exp(i * omega_l * t)
        with self.timings.add('b_l x exp(i x omega x t)') as t:
            omega_time = np.einsum('r,rt->rt', self.mode_ampl_array, np.exp(omega_time))

        # Step 4: Compute the DMD trajectory: sum_l^r b_l * exp(i * omega_l * t)
        with self.timings.add('DMD traj') as t:
            # mode_array: nspace x nmode
            # omega_time: nmode x nsnap_extrap
            DMD_traj_nsnap = self.mode_array @ omega_time[:, :self.nsnap - self.HODMD_order + 1]

        # Step 5: Normalize the DMD trajectory with initial data
        with self.timings.add('normalize') as t:
            # Negligible time, dimension: nspace
            # snap_array: nspace x nsnap
            # DMD_traj_nsnap: nspace x nsnap
            norm_ratio_array = \
                np.linalg.norm(self.snap_array[:, :], axis=1) * self.scale_factor / \
                np.linalg.norm(DMD_traj_nsnap[:, :], axis=1)

        # Step 6: Compute the DMD modes summed over spatial coordinate
        with self.timings.add('modes sum') as t:
            # mode_array: nspace x nmode
            # norm_ratio_array: nspace
            # sum_mode_array: nmode
            sum_mode_array = np.sum(self.mode_array * norm_ratio_array[:, np.newaxis], axis=0)

        # Step 7: Compute the DMD trajectory summed over spatial coordinate
        with self.timings.add('DMD_traj_sum') as t:
            # omega_time: nmode x nsnap_extrap
            # sum_mode_array: nmode
            # sum over modes
            DMD_traj_sum = np.sum(omega_time[:, :] * sum_mode_array[:, np.newaxis], axis=0)

        if self.verbose:
            get_size(DMD_traj_sum, 'DMD_traj_sum')
            get_size(norm_ratio_array, 'norm_ratio_array')
            get_size(sum_mode_array, 'sum_mode_array')
            get_size(DMD_traj_nsnap, 'DMD_traj_nsnap')
            get_size(omega_time, 'omega_time')

        return DMD_traj_sum, self.mode_ampl_array[:] * sum_mode_array[:]
