"""Based on Yang Song's code score_sde_pytorch/blob/main/losses.py ."""

from collections.abc import Callable

import torch

import psdenoiser.model_utils as mutils
from psdenoiser.sde import SDE
from psdenoiser.utils import extract

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def forward_diffusion(
    sde: SDE, x_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None
) -> torch.Tensor:
    """Forward diffusion (using Sohl-Dickstein+15)."""
    if noise is None:
        noise = torch.randn_like(x_0)

    sqrt_alphas_cumprod_t = extract(sde.sqrt_alphas_cumprod, t, x_0.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sde.sqrt_one_minus_alphas_cumprod, t, x_0.shape
    )

    return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise


def get_sde_loss_fnc(
    sde: SDE,
    train: bool,
    reduce_mean: bool = True,
    continuous: bool = True,
    likelihood_weighting: bool = False,
    eps: float = 1e-5,
) -> Callable:
    r"""Create a loss function for training with arbirary SDEs.

    Parameters
    ----------
    sde: sde.SDE
        An `sde.SDE` object that represents the forward SDE.
    train: bool
        `True` for training loss and `False` for evaluation loss.
    reduce_mean:
        If `True`, average the loss across data dimensions.
        Otherwise sum the loss across data dimensions.
    continuous:
        `True` indicates that the model is defined to take
        continuous time steps. Otherwise it requires
        ad-hoc interpolation to take continuous time steps.
    likelihood_weighting:
        If `True`, weigh the mixture of score matching losses
        according to https://arxiv.org/abs/2101.09258; otherwise use the weighting
        recommended in Song+20.
    eps: float, optional
        The smallest time step from which to sample.

    Returns
    -------
        A loss function.
    """
    reduce_op = (
        torch.mean
        if reduce_mean
        else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)
    )

    def loss_fn(
        model: torch.nn.Module,
        batch: torch.Tensor,
        cdn: torch.Tensor | None = None,
        x_cdn: torch.Tensor | None = None,
    ):
        """Compute the loss function.

        Parameters
        ----------
        model: torch.nn.Module
            A score model.
        batch: torch.Tensor
            A mini-batch of training data.
        cdn: torch.Tensor, optional
            A mini-batch of conditioning variables
            for the fully-connected network (similar to diffusion time).
        x_cdn: torch.Tensor, optional
            A mini-batch of conditioning variables
            for the U-Net (i.e. 2D image data).

        Returns
        -------
            loss: Scalar that represents the average loss value across the mini-batch.
        """
        score_fn = mutils.get_score_fn(sde, model, train=train, continuous=continuous)
        t = torch.rand(batch.shape[0], device=batch.device) * (sde.T - eps) + eps
        z = torch.randn_like(batch)
        # Calculate mean and std of p_t(x)
        mean, std = sde.marginal_prob(batch, t)
        perturbed_data = mean + std[:, None, None, None] * z
        # this returns -score/std
        score = score_fn(perturbed_data, t, x_cdn=x_cdn, cdn=cdn)

        if not likelihood_weighting:
            # this cancels the std, so this is -score + z
            losses = torch.square(score * std[:, None, None, None] + z)
            losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
        else:
            # This keeps the std: -score/std + z/std
            g2 = sde.sde(torch.zeros_like(batch), t)[1] ** 2
            losses = torch.square(score + z / std[:, None, None, None])
            # Then weighted by diffusion coefficient squared
            losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1) * g2

        return torch.mean(losses)

    return loss_fn
