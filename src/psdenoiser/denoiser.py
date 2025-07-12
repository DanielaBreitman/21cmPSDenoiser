"""Module that interacts with the Denoiser PyTorch model."""
from __future__ import annotations

import torch
import gc
import numpy as np
from astropy import units as un

import logging
from typing import Any
import warnings
from pathlib import Path

#from .get_denoiser import download_denoiser

from inputs import DenoiserInput
from model import UNet
from inputs import DenoiserInput
from outputs import DenoiserOutput
from properties import denoiser_csts
from tqdm.auto import tqdm

from model import UNet
import utils
from sde import VPSDE

from sample_pytorch import GetODESampler

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

log = logging.getLogger(__name__)


class Denoiser:
    r"""A class that loads 21cmPSDenoiser model and runs it on the input PS realisations.

    Parameters
    ----------
    Nsamples : int, optional
        Number of diffusion samples, default is 200.
    sampler_denoise : bool, optional
        If `True`, add one-step denoising to final samples, default is `True`.
    sampler_rtol : float, optional
        The relative tolerance level of the probability flow ODE solver, default is 1e-5.
    sampler_atol : float, optional
        The absolute tolerance level of the probability flow ODE solver, default is 1e-
    """

    def __init__(self,
                 Nsamples=200, 
                 sampler_denoise=True,
                 sampler_rtol=1e-5, 
                 sampler_atol=1e-5):
        #download_denoiser()
        self.csts = denoiser_csts
        model = UNet(dim=(len(self.csts.kperp),len(self.csts.kpar)),
                channels=2,
                dim_mults=(1, 2, 4, 8,),
                cdn_len=None,
            )
        here = Path(__file__).parent
        model.load_state_dict(torch.load(here / 'denoiser_model.pt', map_location=device))
        model.to(device)
        model.eval()
        self.model = model
        
        self.Nsamples = Nsamples
        sde = VPSDE(beta_min=0.1, beta_max=20.) #Like Ho+20
        self.sample = GetODESampler(sde, (Nsamples,1, len(self.csts.kperp),len(self.csts.kpar)), 
                                      device=device,
                                      denoise=sampler_denoise,
                                      rtol=sampler_rtol, atol=sampler_atol).get_ode_sampler()

    def __getattr__(self, name: str) -> Any:
        """Allow access to denoiser properties directly from the denoiser object."""
        return getattr(self.csts, name)

    
    @torch.no_grad()
    def get_pred_single(self, noisy_sample):
        samples = self.sample(self.model, 
                        x_cdn = noisy_sample[np.newaxis,...], 
                        progress=True).squeeze().cpu().detach()
       
        samples_w_units = utils.reverse_transform(samples, self.csts.mean_scale, self.csts.mean_bias).cpu().detach().numpy()
        
        noisy_sample.cpu()
        del noisy_sample  
        return samples_w_units
    
    @torch.no_grad()
    def get_pred(self, noisy_samples, progress=True):
        all_preds = []
        all_stds = []
        all_samples = []
        if progress:
            pbar = tqdm(range(noisy_samples.shape[0]),
                                  total=noisy_samples.shape[0],
                                  desc ="Sampling ")
        else:
            pbar = range(noisy_samples.shape[0])
        for i in pbar:
            samples = self.get_pred_single(torch.Tensor(noisy_samples[i]))
            all_preds.append(np.median(samples, axis = 0))
            all_stds.append(np.std(samples, axis=0))
            all_samples.append(samples)
        return np.array(all_samples), np.array(all_preds), np.array(all_stds)

    def predict(self, 
                ps_realisations: un.Quantity,
                kperp: un.Quantity,
                kpar: un.Quantity,
                ) -> tuple[np.ndarray, DenoiserOutput, dict[str, np.ndarray]]:
        r"""Call the emulator, evaluate it at the given parameters, restore dimensions.

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
        mask = np.squeeze(np.mean(ps_realisations, axis = (-1,-2)) > self.csts.min_PS_mean)
        if np.sum(mask) > 0:
            normed_ps_realisations, kperp, kpar = DenoiserInput().format_input(ps_realisations[mask], 
                                                                           kperp, kpar
                                                                           )
            if np.sum(np.isnan(normed_ps_realisations)) > 0:
                raise ValueError('There are NaNs in the normalised input PS!!')
            else:
                samples_pred, med_pred, std_pred = self.get_pred(normed_ps_realisations)
            if np.sum(mask) < len(mask):
                warnings.warn(f"Mean of PS is too low, skipping denoising for {len(mask) - np.sum(mask)} samples...")
                final_med = np.zeros_like(ps_realisations)
                final_std = np.zeros_like(ps_realisations)
                final_samples = np.zeros((len(mask),self.Nsamples,) + ps_realisations.shape[1:])
                final_samples[mask] = samples_pred
                final_samples[~mask] = ps_realisations[:,None,...][~mask]
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
            final_std = np.ones_like(ps_realisations)+np.nan
            final_samples = ps_realisations
        gc.collect()
        return DenoiserOutput(final_samples.squeeze()*un.mK**2,
                              final_med.squeeze()*un.mK**2,
                              final_std.squeeze()*un.mK**2, 
                              kperp/un.Mpc, 
                              kpar/un.Mpc)
