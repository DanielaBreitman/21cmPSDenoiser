"""Module that interacts with the Denoiser PyTorch model."""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

import torch
from torch import nn

#from .config import CONFIG
#from .get_emulator import get_emu_data
from inputs import DenoiserInput
from model import Autoencoder
from outputs import DenoiserOutput
from outputs import NNOutput
from properties import denoiser_csts


log = logging.getLogger(__name__)


class Denoiser:
    r"""A class that loads an emulator and uses it to obtain 21cmFAST summaries.

    Parameters
    ----------
    version : str, optional
        Emulator version to use/download, default is 'latest'.
    """

    def __init__(self, version: str = "latest"):
        #get_emu_data(version=version)

        model = Autoencoder(in_ch=1)
        model.load_state_dict(torch.load('/home/dbreitman/CV_PS_denoising/Dec_models/Dec23_KL2'))
        model.eval()

        self.model = model
        self.inputs = DenoiserInput()
        self.csts = denoiser_csts

    def __getattr__(self, name: str) -> Any:
        """Allow access to emulator properties directly from the emulator object."""
        return getattr(self.csts, name)

    def predict(self, 
                noisy_ps: np.ndarray,
                kperp: np.ndarray,
                kpar: np.ndarray,
                h_little: bool = False,
                delta_sq: bool = False,) -> tuple[np.ndarray, EmulatorOutput, dict[str, np.ndarray]]:
        r"""Call the emulator, evaluate it at the given parameters, restore dimensions.

        Parameters
        ----------
        astro_params : np.ndarray or dict
            An array with the nine astro_params input all $\in [0,1]$ OR in the
            21cmFAST AstroParams input units. Dicts (e.g. p21.AstroParams.defining_dict)
            are also accepted formats. Arrays of only dicts are accepted as well
            (for batch evaluation).
        verbose : bool, optional
            If True, prints the emulator prediction.

        Returns
        -------
        theta : np.ndarray
            The normalized cylindrical PS used to evaluate the denoiser.
        emu : DenoiserOutput
            The denoiser output, with dimensions restored.
        errors : dict
            The mean error on the test set (i.e. independent of input).
        """
        ps, kperp, kpar = self.inputs.format_input_ps(noisy_ps, kperp, kpar, 
                                             h_little=h_little,
                                             delta_sq=delta_sq, 
                                             normed=True)
        out = self.model(torch.Tensor(ps[:,np.newaxis,...])).detach().cpu().numpy()
        out = NNOutput(out, kperp, kpar)
        out = out.get_renormalized()

        #errors = self.get_errors()

        return noisy_ps, out#, errors

    def get_errors(self) -> dict[str, np.ndarray]:
        """Calculate the emulator error on its outputs.

        Parameters
        ----------
        emu : dict
            Dict containing the emulator predictions, defined in Emulator.predict
            
        Returns
        -------
        The mean error on the test set (i.e. independent of theta) with all units
        restored and logs removed.
        """
        # For now, we return the mean emulator error (obtained from the test set) for
        # each summary. All errors are the median absolute difference between test set
        # and prediction AFTER units have been restored AND log has been removed.
        return {
            "PS_err": self.PS_err,
            "Var_err": self.Var_err,
        }