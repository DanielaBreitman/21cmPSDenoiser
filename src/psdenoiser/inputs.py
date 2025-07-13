"""Module containing functionality for handling denoiser inputs."""

import warnings

import astropy.units as un
import numpy as np
import torch

from psdenoiser.properties import denoiser_csts as csts
from psdenoiser.utils import transform


class DenoiserInput:
    """Class for handling denoiser input."""

    def format_input(
        self,
        ps_realisations: un.Quantity,
        kperp: un.Quantity,
        kpar: un.Quantity,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Format the input 2D PS into a 3D numpy array with shape (Nsamples, Nkperp, nkpar).

        Parameters
        ----------
        ps_realisations : un.Quantity
            Cylindrical 21-cm power spectrum sample(s) P(kperp, kpar) to be denoised
            in units of mK^2.
        kperp : un.Quantity
            K perpendicular modes of the cylindrical power spectrum in units of Mpc^{-1}
            by default.
        kpar : un.Quantity
            K parallel modes of the cylindrical power spectrum in units of Mpc^{-1}
            by default.
        """
        if not (np.allclose(kperp, csts.kperp) and np.allclose(kpar, csts.kpar)):
            raise NotImplementedError(
                "Interpolation of input kperp and kpar bins is not supported yet."
            )

        if ps_realisations.shape[1] != len(kperp):
            raise ValueError(
                "You supplied the wrong kperp bins: %s kperp bins in PS vs %s kperp bins in kperp array supplied."
                % (ps_realisations.shape[1], len(kperp))
            )
        if ps_realisations.shape[2] != len(kpar):
            raise ValueError(
                "You supplied the wrong kpar bins: %s kpar bins in PS vs %s kpar bins in kpar array supplied."
                % (ps_realisations.shape[2], len(kpar))
            )
        if len(ps_realisations.shape) > 3:
            raise ValueError(
                "The shape of the input ps_realisations PS should be (Nsamples, Nkperp, Nkpar)."
            )
        if np.sum(np.isnan(ps_realisations)):
            raise ValueError("There should not be any NaNs in the PS array!")

        # Check that input kperp and kpar are the same as the ones of the autoencoder data
        # In the future, this could be replaced by an interpolation step.

        if not np.allclose(kperp, csts.kperp):
            warnings.warn(
                "Input kperp bins are not the same as the expected kperp bins! Proceeding anyways..."
            )
        if not np.allclose(kpar, csts.kpar):
            warnings.warn(
                "Input kpar bins are not the same as the expected kpar bins! Proceeding anyways..."
            )
        if ps_realisations.unit != un.mK**2:
            raise ValueError(
                "Input PS should be in mK^2, but is in %s." % ps_realisations.unit
            )
        if kperp.unit != un.Mpc ** (-1):
            raise ValueError(
                "Input kperp should be in Mpc^{-1}, but is in %s." % kperp.unit
            )
        if kpar.unit != un.Mpc ** (-1):
            raise ValueError(
                "Input kpar should be in Mpc^{-1}, but is in %s." % kpar.unit
            )

        return self.normalize_noisy(ps_realisations.value), kperp.value, kpar.value

    def normalize_noisy(self, noisy: np.ndarray) -> np.ndarray:
        """Normalize the input PS to be in [-1,1]."""
        return (
            transform(
                torch.Tensor(noisy),
                torch.Tensor(csts.noisy_scale),
                torch.Tensor(csts.noisy_bias),
            )
            .cpu()
            .detach()
            .numpy()
        )
