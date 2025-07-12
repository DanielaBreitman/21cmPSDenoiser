"""From Yang Song's score_sde_pytorch/sampling.py
ODE sampler for probability flow models."""
import functools

import torch
import numpy as np
import abc

from utils import from_flattened_numpy, to_flattened_numpy 
from model_utils import get_score_fn
from scipy import integrate
import gc

class Predictor(abc.ABC):
    """The abstract class for a predictor algorithm."""

    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__()
        self.sde = sde
        # Compute the reverse SDE/ODE
        self.rsde = sde.reverse(score_fn, probability_flow)
        self.score_fn = score_fn

    @abc.abstractmethod
    def update_fn(self, x, t, x_cdn=None, cdn=None):
        """One update of the predictor.

        Args:
          x: A PyTorch tensor representing the current state
          t: A Pytorch tensor representing the current time step.

        Returns:
          x: A PyTorch tensor of the next state.
          x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
        """
        pass

class ReverseDiffusionPredictor(Predictor):
    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__(sde, score_fn, probability_flow)

    def update_fn(self, x, t, x_cdn=None, cdn=None):
        f, G = self.rsde.discretize(x, t,x_cdn=x_cdn, cdn=cdn)
        z = torch.randn_like(x)
        x_mean = x - f
        x = x_mean + G[:, None, None, None] * z
        return x, x_mean

class GetODESampler:
    def __init__(self,sde, shape, inverse_scaler=None,
                    denoise=False, rtol=1e-5, atol=1e-5,
                    method='RK45', eps=1e-3, device='cuda'):
        
        """Probability flow ODE sampler with the black-box ODE solver.

        Args: 
            sde: An `sde_lib.SDE` object that represents the forward SDE.
            shape: A sequence of integers. The expected shape of a single sample.
            inverse_scaler: The inverse data normalizer.
            denoise: If `True`, add one-step denoising to final samples.
            rtol: A `float` number. The relative tolerance level of the ODE solver.
            atol: A `float` number. The absolute tolerance level of the ODE solver.
            method: A `str`. The algorithm used for the black-box ODE solver.
              See the documentation of `scipy.integrate.solve_ivp`.
            eps: A `float` number. The reverse-time SDE/ODE will be integrated to `eps` for numerical stability.
            device: PyTorch device.

        Returns:
            A sampling function that returns samples and the number of function evaluations during sampling.
          """
        self.sde = sde
        self.shape = shape
        self.inverse_scaler = inverse_scaler
        self.denoise = denoise
        self.rtol = rtol
        self.atol = atol
        self.method = method
        self.eps = eps
        self.device = device
    
    def denoise_update_fn(self, model, x, x_cdn=None, cdn=None):
        score_fn = get_score_fn(self.sde, model, train=False, continuous=True)
        # Reverse diffusion predictor for denoising
        predictor_obj = ReverseDiffusionPredictor(self.sde, score_fn, probability_flow=False)
        vec_eps = torch.ones(x.shape[0], device=x.device) * self.eps
        _, x = predictor_obj.update_fn(x, vec_eps, x_cdn=x_cdn, cdn=cdn)
        return x

    def drift_fn(self, model, x, t, x_cdn=None, cdn=None):
        """Get the drift function of the reverse-time SDE."""
        score_fn = get_score_fn(self.sde, model, train=False, continuous=True)
        rsde = self.sde.reverse(score_fn, probability_flow=True)
        return rsde.sde(x, t, x_cdn=x_cdn, cdn=cdn)[0]

    def ode_sampler(self, model, z=None, x_cdn=None, cdn=None, progress=False, return_nfe=False):
        """The probability flow ODE sampler with black-box ODE solver.

        Args: All inputs are on CPU except model that's on the GPU already
          model: A score model.
          z: If present, generate samples from latent code `z`.
        Returns:
          samples, number of function evaluations.
        """
        with torch.no_grad():
          # Initial sample
            if z is None:
                # If not represent, sample the latent code from the prior distibution of the SDE.
                x = self.sde.prior_sampling(self.shape).to(self.device)
            else:
                x = z.to(self.device)
            if x_cdn is not None:
                x_cdn = x_cdn.repeat((self.shape[0],)+(1,)*len(self.shape[1:])).reshape(self.shape).to(self.device)
            def ode_func(t, x):
                x = from_flattened_numpy(x, self.shape).to(self.device).type(torch.float32)
                vec_t = torch.ones(self.shape[0], device=x.device) * t
                drift = self.drift_fn(model, x, vec_t, x_cdn=x_cdn, 
                                 cdn=cdn if cdn is not None else None)
                x.cpu()
                vec_t.cpu()
                return to_flattened_numpy(drift)
            # Black-box ODE solver for the probability flow ODE
            solution = integrate.solve_ivp(ode_func, (self.sde.T, self.eps), to_flattened_numpy(x),
                                         rtol=self.rtol, atol=self.atol, method=self.method)
            x.cpu()
            nfe = solution.nfev
            sln = torch.tensor(solution.y[:, -1]).reshape(self.shape).to(self.device).type(torch.float32)
            # Denoising is equivalent to running one predictor step without adding noise
            if self.denoise:
                sln = self.denoise_update_fn(model, sln, x_cdn=x_cdn, cdn=cdn)
            if self.inverse_scaler is not None:
                sln = self.inverse_scaler(sln)
            
            if x_cdn is not None:
                x_cdn.cpu()
            sln.cpu()
            gc.collect()
            torch.cuda.empty_cache()
            if return_nfe:
                return sln, nfe
            else:
                return sln
    
    def get_ode_sampler(self):
        return self.ode_sampler
