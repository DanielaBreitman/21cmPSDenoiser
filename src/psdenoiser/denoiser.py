"""Module that interacts with the Denoiser PyTorch model."""

from __future__ import annotations

import gc
import logging
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
from astropy import units as un
from tqdm.auto import tqdm

from psdenoiser import utils
from psdenoiser.inputs import DenoiserInput
from psdenoiser.model import UNet
from psdenoiser.outputs import DenoiserOutput
from psdenoiser.properties import denoiser_csts
from psdenoiser.sample_pytorch import GetODESampler
from psdenoiser.sde import VPSDE

log = logging.getLogger(__name__)


class Denoiser:
    r"""A class that loads 21cmPSDenoiser model and runs it on the input PS samples.

    Parameters
    ----------
    device : torch.device, optional
        The device on which to store the model.
        Default will use GPU is available, otherwise CPU.
    nsamples : int, optional
        Number of diffusion samples, default is 200.
    sampler_denoise : bool, optional
        If `True`, add one-step denoising to final samples, default is `True`.
    sampler_rtol : float, optional
        The relative tolerance level of the probability flow ODE solver,
        default is 1e-5.
    sampler_atol : float, optional
        The absolute tolerance level of the probability flow ODE solver,
        default is 1e-5.
    """

    def __init__(
        self,
        device: torch.device | None = None,
        nsamples: int = 200,
        sampler_denoise: bool = True,
        sampler_rtol: float = 1e-5,
        sampler_atol: float = 1e-5,
    ):
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.csts = denoiser_csts
        model = UNet(
            dim=(len(self.csts.kperp), len(self.csts.kpar)),
            channels=2,
            dim_mults=(
                1,
                2,
                4,
                8,
            ),
            cdn_len=None,
        )
        here = Path(__file__).parent
        model.load_state_dict(
            torch.load(here / "denoiser_model.pt", map_location=device)
        )
        model.to(device)
        model.eval()
        self.model = model

        self.nsamples = nsamples
        sde = VPSDE(beta_min=0.1, beta_max=20.0)  # Like Ho+20
        self.sample = GetODESampler(
            sde,
            (nsamples, 1, len(self.csts.kperp), len(self.csts.kpar)),
            device=device,
            denoise=sampler_denoise,
            rtol=sampler_rtol,
            atol=sampler_atol,
        ).get_ode_sampler()

    def __getattr__(self, name: str) -> Any:
        """Allow access to denoiser properties directly from the denoiser object."""
        return getattr(self.csts, name)

    @torch.no_grad()
    def get_pred_single(self, noisy_sample: torch.Tensor) -> np.ndarray:
        r"""Get the mean 21-cm PS for a single noisy PS sample.

        Parameters
        ----------
        noisy_sample : torch.Tensor
            A single noisy PS sample of shape [len(kperp), len(kpar)].

        Returns
        -------
        samples_w_units : np.ndarray
            The denoised PS sample in mK^2,
            with shape [len(kperp), len(kpar)].
        """
        samples = (
            self.sample(self.model, x_cdn=noisy_sample[np.newaxis, ...], progress=True)
            .squeeze()
            .cpu()
            .detach()
        )

        samples_w_units = (
            utils.reverse_transform(samples, self.csts.mean_scale, self.csts.mean_bias)
            .cpu()
            .detach()
            .numpy()
        )

        noisy_sample.cpu()
        del noisy_sample
        return samples_w_units

    @torch.no_grad()
    def get_pred(
        self, noisy_samples: np.ndarray, progress: bool = True
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        r"""Get the mean 21-cm PS for multiple noisy PS realisations.

        Parameters
        ----------
        noisy_samples : np.ndarray
            Noisy 21-cm PS realisations
            of shape [Nrealisations, len(kperp), len(kpar)] in mK^2.
        progress : bool, optional
            Whether to show a progress bar.
            Useful when passing a large batch of realisations.

        Returns
        -------
        all_samples : np.ndarray
            Full output with all denoised PS samples,
            shape [Nrealisations, Nsamples, len(kperp), len(kpar)].
        all_preds : np.ndarray
            The median of the denoiser PS
            taken over the diffusion samples `Nsamples`,
            shape [Nrealisations, len(kperp), len(kpar)].
        all_stds : np.ndarray
            The standard deviation of the denoised PS samples
            taken over the diffusion samples`Nsamples`,
            shape [Nrealisations, len(kperp), len(kpar)].
        """
        all_preds = []
        all_stds = []
        all_samples = []
        if progress:
            pbar = tqdm(
                range(noisy_samples.shape[0]),
                total=noisy_samples.shape[0],
                desc="Sampling ",
            )
        else:
            pbar = range(noisy_samples.shape[0])
        for i in pbar:
            samples = self.get_pred_single(torch.Tensor(noisy_samples[i]))
            all_preds.append(np.median(samples, axis=0))
            all_stds.append(np.std(samples, axis=0))
            all_samples.append(samples)
        return np.array(all_samples), np.array(all_preds), np.array(all_stds)

    def predict(
        self,
        ps_realisations: un.Quantity,
        kperp: un.Quantity,
        kpar: un.Quantity,
    ) -> DenoiserOutput:
        r"""Call 21cmPSDenoiser and predict the mean 21-cm PS for the given PS samples.

        Parameters
        ----------
        ps_realisations : un.Quantity
            cylindrical 21-cm PS in mK^2 of shape [N, len(kperp), len(kpar)]
            No NaNs or Infs allowed.
            If the mean of the PS realisation is < 1e-2mK^2,
            the denoiser will not be applied.
        kperp : un.Quantity
            kperp bin center values of the cylindrical PS
        kpar : un.Quantity
            kpar bin center values of the cylindrical PS
        N : int, optional
            Number of diffusion samples to take the median over to obtain the
            denoised result, default is 250.


        Returns
        -------
        DenoiserOutput
            See the class definition for more information.
        """
        if len(ps_realisations.shape) == 2:
            ps_realisations = ps_realisations[np.newaxis, ...]

        mask = np.mean(ps_realisations.value, axis=(-1, -2)) > self.csts.min_PS_mean

        if np.sum(mask) > 0:
            normed_ps_realisations, kperp, kpar = DenoiserInput().format_input(
                ps_realisations[mask], kperp, kpar
            )
            if np.sum(np.isnan(normed_ps_realisations)) > 0:
                raise ValueError("There are NaNs in the normalised input PS!!")
            samples_pred, med_pred, std_pred = self.get_pred(normed_ps_realisations)
            if np.sum(mask) < len(mask):
                warnings.warn(
                    f"Mean of PS is too low, skipping denoising"
                    f" for {len(mask) - np.sum(mask)} samples...",
                    stacklevel=2,
                )
                final_med = np.zeros_like(ps_realisations)
                final_std = np.zeros_like(ps_realisations)
                final_samples = np.zeros(
                    (
                        len(mask),
                        self.Nsamples,
                    )
                    + ps_realisations.shape[1:]
                )
                final_samples[mask] = samples_pred
                final_samples[~mask] = ps_realisations[:, None, ...][~mask]
                final_med[mask] = med_pred
                final_med[~mask] = ps_realisations[~mask]
                final_std[mask] = std_pred
                final_std[~mask] = np.nan
            else:
                final_med = med_pred
                final_std = std_pred
                final_samples = samples_pred
        else:
            final_med = ps_realisations
            final_std = np.ones_like(ps_realisations) + np.nan
            final_samples = ps_realisations
        gc.collect()
        return DenoiserOutput(
            final_samples.squeeze() * un.mK**2,
            final_med.squeeze() * un.mK**2,
            final_std.squeeze() * un.mK**2,
            kperp / un.Mpc,
            kpar / un.Mpc,
        )
