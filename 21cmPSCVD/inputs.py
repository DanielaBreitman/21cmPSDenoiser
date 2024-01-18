"""Module containing functionality for handling denoiser inputs."""

import numpy as np

from properties import denoiser_csts as csts

class DenoiserInput:
    """Class for handling denoiser input."""

    def format_input_ps(
            self,
            noisy_ps: np.ndarray,
            kperp: np.ndarray,
            kpar: np.ndarray,
            normed: bool = True,
            h_little: bool = False,
            delta_sq: bool = False,
        ) -> np.ndarray:
        """Format the input 2D PS into a 3D numpy array with shape (Nsamples, Nkperp, nkpar).


        Parameters
        ----------
        noisy_ps : np.ndarray
            Cylindrical 21-cm power spectrum sample(s) P(kperp, kpar) to be denoised 
            in units of mK^2 Mpc^3.
        kperp : np.ndarray
            K perpendicular modes of the cylindrical power spectrum in units of Mpc^{-1}
            by default.
        kpar : np.ndarray
            K parallel modes of the cylindrical power spectrum in units of Mpc^{-1}
            by default.
        h_little : bool, optional
            If True, we assume that the units of both input kperp and kpar are in hMpc^{-1},
            and the unit of the power spectrum is then mK^2h^{-3}Mpc^3.
        delta_sq : bool, optional
            If True, we assume that the input power spectrum is \Delta^2 in units of mK^2.

        """
        # Maybe eventually add warning if k values are far outside training range.

        if np.any(kpar) < 0:
            m = kpar > 1e-10
            kpar = kpar[m]
            noisy_ps = noisy_ps[:,m]

        if len(noisy_ps.shape) == 2:
            noisy_ps = noisy_ps[np.newaxis,...]
        if noisy_ps.shape[1] != len(kperp):
            raise ValueError('You supplied the wrong kperp bins: %s kperp bins in PS vs %s kperp bins in kperp array supplied.' % (noisy_ps.shape[1], len(kperp)))
        if noisy_ps.shape[2] != len(kpar):
            raise ValueError('You supplied the wrong kpar bins: %s kpar bins in PS vs %s kpar bins in kpar array supplied.' % (noisy_ps.shape[2], len(kpar)))
        if len(noisy_ps.shape) > 3:
            raise ValueError('The shape of the input noisy PS should be (Nsamples, Nkperp, Nkpar).')

        if delta_sq:
            kperp_grid, kpar_grid = np.meshgrid(kperp,kpar)
            noisy_ps /= kperp_grid**2 * kpar_grid / (2 * np.pi**2)

        if h_little:
            kperp /= csts.h
            kpar /= csts.h
            if not delta_sq:
                noisy_ps *= csts.h**3

        if np.sum(np.isnan(noisy_ps)) > 0:
            m = np.any(~np.isnan(noisy_ps), axis = -1)
            noisy_ps = noisy_ps[m]
            kperp = kperp[m]

        # From here on, units are:
        # PS: mK^2 Mpc^3, no more NaNs.
        # kperp and kpar: Mpc^{-1}, same shape as PS

        return self.normalize(noisy_ps), kperp, kpar

    def normalize(self, ps) -> tuple:
            """Normalize the parameters.

            Parameters
            ----------
            theta : np.ndarray
                Input parameters, strictly in 2D array format, with shape
                (n_batch, n_params).

            Returns
            -------
            np.ndarray
                Normalized parameters, with shape (n_batch, n_params).
            """

            return (np.log10(ps) - csts.noisy_PS_mean) / csts.noisy_PS_std