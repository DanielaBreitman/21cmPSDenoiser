from pathlib import Path

import numpy as np
from astropy import units as un


class DenoiserConstants:
    """A class that contains the constants of the denoiser."""

    def __init__(self):
        here = Path(__file__).parent
        with np.load(here / "constants.npz") as f:
            self.noisy_bias = f["noisy_mean"]
            self.noisy_scale = f["noisy_std"]
            self.mean_bias = f["mean_mean"]
            self.mean_scale = f["mean_std"]
            self.param_labels = f["param_labels"]
            self.modes = f["Nmodes"]
            self.kperp = f["kperp"] / un.Mpc
            self.kpar = f["kpar"] / un.Mpc
            self.min_PS_mean = 1e-2
            self.denoiser_median_on_test_mean_percent = f["median_err_on_mean_2D"]
            self.h = f["h"]


denoiser_csts = DenoiserConstants()
