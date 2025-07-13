"""Based on Yang Song's score_sde_pytorch/blob/main/sde_lib.py ."""

import abc

import numpy as np
import torch


class SDE(abc.ABC):
    r"""SDE abstract class. Functions are designed for a mini-batch of inputs."""

    def __init__(self, n):
        r"""Construct an SDE.

        Parameters
        ----------
          N: number of discretization time steps.
        """
        super().__init__()
        self.N = n

    @property
    @abc.abstractmethod
    def T(self):
        r"""End time of the SDE."""

    @abc.abstractmethod
    def sde(self, x, t):
        r"""Drift and diffusion functions for the forward SDE."""

    @abc.abstractmethod
    def marginal_prob(self, x, t):
        r"""Parameters to determine the marginal distribution of the SDE, $p_t(x)$."""

    @abc.abstractmethod
    def prior_sampling(self, shape):
        r"""Generate one sample from the prior distribution, $p_T(x)$."""

    @abc.abstractmethod
    def prior_logp(self, z):
        r"""Compute log-density of the prior distribution.

        Useful for computing the log-likelihood via probability flow ODE.

        Parameters
        ----------
          z: latent code

        Returns
        -------
            torch.Tensor
                Log probability density of the prior distribution.
                The shape is (batch_size,).
        """

    def discretize(self, x, t):
        r"""Discretize the SDE in the form: x_{i+1} = x_i + f_i(x_i) + G_i z_i.

        Useful for reverse diffusion sampling and probabiliy flow sampling.
        Defaults to Euler-Maruyama discretization.

        Args:
            x: a torch tensor
            t: a torch float representing the time step (from 0 to `self.T`)

        Returns
        -------
            f, G
        """
        dt = 1 / self.N
        drift, diffusion = self.sde(x, t)
        f = drift * dt
        G = diffusion * torch.sqrt(torch.tensor(dt, device=t.device))
        return f, G

    def reverse(self, score_fn: torch.Tensor, probability_flow: bool = False):
        r"""Create the reverse-time SDE/ODE.

        Parameters
        ----------
            score_fn:
                A time-dependent score-based model that
                takes x and t and returns the score.
            probability_flow:
                If `True`, create the reverse-time ODE
                used for probability flow sampling.
        """
        N = self.N
        T = self.T
        sde_fn = self.sde
        discretize_fn = self.discretize

        class RSDE(self.__class__):
            def __init__(self):
                r"""Class for the reverse-time SDE."""
                self.N = N
                self.probability_flow = probability_flow

            @property
            def T(self):
                r"""End time of the SDE."""
                return T

            def sde(
                self,
                x: torch.Tensor,
                t: torch.Tensor,
                x_cdn: torch.Tensor | None = None,
                cdn: torch.Tensor | None = None,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                r"""Drift and diffusion functions for the reverse SDE/ODE."""
                drift, diffusion = sde_fn(x, t)
                score = score_fn(x, t, x_cdn=x_cdn, cdn=cdn)
                drift = drift - diffusion[:, None, None, None] ** 2 * score * (
                    0.5 if self.probability_flow else 1.0
                )
                # Set the diffusion function to zero for ODEs.
                diffusion = 0.0 if self.probability_flow else diffusion
                return drift, diffusion

            def discretize(
                self,
                x: torch.Tensor,
                t: torch.Tensor,
                x_cdn: torch.Tensor | None = None,
                cdn: torch.Tensor | None = None,
            ):
                r"""Discretized iteration rules for the reverse diffusion sampler."""
                f, G = discretize_fn(x, t)
                rev_f = f - G[:, None, None, None] ** 2 * score_fn(
                    x, t, x_cdn=x_cdn, cdn=cdn
                ) * (0.5 if self.probability_flow else 1.0)
                rev_G = torch.zeros_like(G) if self.probability_flow else G
                return rev_f, rev_G

        return RSDE()


class VPSDE(SDE):
    def __init__(self, beta_min: float = 0.1, beta_max: float = 20, n: float = 1000):
        r"""Construct a Variance Preserving SDE.

        Args:
          beta_min: value of beta(0)
          beta_max: value of beta(1)
          n: number of discretization steps
        """
        super().__init__(n)
        self.beta_0 = beta_min
        self.beta_1 = beta_max
        self.N = n
        self.discrete_betas = torch.linspace(beta_min / n, beta_max / n, n)
        self.alphas = 1.0 - self.discrete_betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    @property
    def T(self):
        r"""End time of the SDE."""
        return 1

    def sde(self, x, t):
        r"""Drift and diffusion functions for the forward SDE."""
        beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
        drift = -0.5 * beta_t[:, None, None, None] * x
        diffusion = torch.sqrt(beta_t)
        return drift, diffusion

    def marginal_prob(self, x, t):
        r"""Parameters to determine the marginal distribution of the SDE, $p_t(x)$."""
        log_mean_coeff = (
            -0.25 * t**2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        )
        mean = torch.exp(log_mean_coeff[:, None, None, None]) * x
        std = torch.sqrt(1.0 - torch.exp(2.0 * log_mean_coeff))
        return mean, std

    def prior_sampling(self, shape):
        r"""Generate one sample from the prior distribution, $p_T(x)$."""
        return torch.randn(*shape)

    def prior_logp(self, z):
        r"""Compute log-density of the prior distribution."""
        shape = z.shape
        N = np.prod(shape[1:])
        return -N / 2.0 * np.log(2 * np.pi) - torch.sum(z**2, dim=(1, 2, 3)) / 2.0

    def discretize(self, x, t):
        """DDPM discretization."""
        timestep = (t * (self.N - 1) / self.T).long()
        beta = self.discrete_betas.to(x.device)[timestep]
        alpha = self.alphas.to(x.device)[timestep]
        sqrt_beta = torch.sqrt(beta)
        f = torch.sqrt(alpha)[:, None, None, None] * x - x
        G = sqrt_beta
        return f, G


class SubVPSDE(SDE):
    def __init__(self, beta_min: float = 0.1, beta_max: float = 20, n: int = 1000):
        r"""Construct the sub-VP SDE that excels at likelihoods.

        Parameters
        ----------
            beta_min: float
                Value of beta(0).
            beta_max: float
                Value of beta(1).
            n: int
                Number of discretization steps.
        """
        super().__init__(n)
        self.beta_0 = beta_min
        self.beta_1 = beta_max
        self.N = n

    @property
    def T(self):
        r"""End time of the SDE."""
        return 1

    def sde(self, x, t):
        r"""Drift and diffusion functions for the forward SDE."""
        beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
        drift = -0.5 * beta_t[:, None, None, None] * x
        discount = 1.0 - torch.exp(
            -2 * self.beta_0 * t - (self.beta_1 - self.beta_0) * t**2
        )
        diffusion = torch.sqrt(beta_t * discount)
        return drift, diffusion

    def marginal_prob(self, x, t):
        r"""Parameters to determine the marginal distribution of the SDE, $p_t(x)$."""
        log_mean_coeff = (
            -0.25 * t**2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        )
        mean = torch.exp(log_mean_coeff)[:, None, None, None] * x
        std = 1 - torch.exp(2.0 * log_mean_coeff)
        return mean, std

    def prior_sampling(self, shape):
        r"""Generate one sample from the prior distribution, $p_T(x)$."""
        return torch.randn(*shape)

    def prior_logp(self, z):
        r"""Compute log-density of the prior distribution."""
        shape = z.shape
        N = np.prod(shape[1:])
        return -N / 2.0 * np.log(2 * np.pi) - torch.sum(z**2, dim=(1, 2, 3)) / 2.0


class VESDE(SDE):
    def __init__(self, sigma_min: float = 0.01, sigma_max: float = 50, n: float = 1000):
        r"""Construct a Variance Exploding SDE.

        Parameters
        ----------
            sigma_min: float
                Smallest sigma, default is 0.01.
            sigma_max: float
                Largest sigma, default is 50.
            n: int
                Number of discretization steps, default is 1000.
        """
        super().__init__(n)
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.discrete_sigmas = torch.exp(
            torch.linspace(np.log(self.sigma_min), np.log(self.sigma_max), n)
        )
        self.N = n

    @property
    def T(self):
        r"""End time of the SDE."""
        return 1

    def sde(self, x, t):
        r"""Drift and diffusion functions for the forward SDE."""
        sigma = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        drift = torch.zeros_like(x)
        diffusion = sigma * torch.sqrt(
            torch.tensor(
                2 * (np.log(self.sigma_max) - np.log(self.sigma_min)), device=t.device
            )
        )
        return drift, diffusion

    def marginal_prob(self, x, t):
        r"""Parameters to determine the marginal distribution of the SDE, $p_t(x)$."""
        std = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        mean = x
        return mean, std

    def prior_sampling(self, shape):
        r"""Generate one sample from the prior distribution, $p_T(x)$."""
        return torch.randn(*shape) * self.sigma_max

    def prior_logp(self, z):
        r"""Compute log-density of the prior distribution."""
        shape = z.shape
        N = np.prod(shape[1:])
        return -N / 2.0 * np.log(2 * np.pi * self.sigma_max**2) - torch.sum(
            z**2, dim=(1, 2, 3)
        ) / (2 * self.sigma_max**2)

    def discretize(self, x, t):
        """SMLD(NCSN) discretization."""
        timestep = (t * (self.N - 1) / self.T).long()
        sigma = self.discrete_sigmas.to(t.device)[timestep]
        adjacent_sigma = torch.where(
            timestep == 0,
            torch.zeros_like(t),
            self.discrete_sigmas[timestep - 1].to(t.device),
        )
        f = torch.zeros_like(x)
        G = torch.sqrt(sigma**2 - adjacent_sigma**2)
        return f, G
