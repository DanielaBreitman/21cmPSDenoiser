"""Based on Yang Song's code https://github.com/yang-song/score_sde_pytorch/blob/main/losses.py"""
import torch
import torch.nn.functional as F
from utils import extract
import model_utils as mutils

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def forward_diffusion(sde, x_0, t, noise=None):
    """Forward diffusion (using the nice property in Sohl-Dickstein+15)
    """
    if noise is None:
        noise = torch.randn_like(x_0)

    sqrt_alphas_cumprod_t = extract(sde.sqrt_alphas_cumprod, t, x_0.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sde.sqrt_one_minus_alphas_cumprod, t, x_0.shape
    )

    return sqrt_alphas_cumprod_t * x_0 + \
           sqrt_one_minus_alphas_cumprod_t * noise

def get_sde_loss_fnc(sde, train, reduce_mean=True, continuous=True, likelihood_weighting=False, eps=1e-5):
    """Create a loss function for training with arbirary SDEs.

    Args:
      sde: An `sde_lib.SDE` object that represents the forward SDE.
      train: `True` for training loss and `False` for evaluation loss.
      reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
      continuous: `True` indicates that the model is defined to take continuous time steps. Otherwise it requires
          ad-hoc interpolation to take continuous time steps.
      likelihood_weighting: If `True`, weight the mixture of score matching losses
          according to https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended in our paper.
      eps: A `float` number. The smallest time step to sample from.

    Returns:
        A loss function.
    """
    reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

    def loss_fn(model, batch,cdn=None, x_cdn=None):
        """Compute the loss function.

        Args:
            model: A score model.
            batch: A mini-batch of training data.

        Returns:
            loss: A scalar that represents the average loss value across the mini-batch.
        """
        score_fn = mutils.get_score_fn(sde, model, train=train, continuous=continuous)
        t = torch.rand(batch.shape[0], device=batch.device) * (sde.T - eps) + eps
        z = torch.randn_like(batch)
        # Calculate mean and std of p_t(x)
        mean, std = sde.marginal_prob(batch, t)
        perturbed_data = mean + std[:, None, None, None] * z
        #returns -score/std
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

        loss = torch.mean(losses)
        return loss

    return loss_fn
def loss(sde, model, x_start, t, x_cdn=None, cdn=None, noise=None, loss_type="l1"):
    if noise is None:
        noise = torch.randn_like(x_start)

    x_noisy = forward_diffusion(sde, x_0=x_start, t=t, noise=noise)
    score_pred = model(x_noisy, t, x_cdn = x_cdn, cdn=cdn)

    if loss_type == 'l1':
        loss_val = F.l1_loss(noise, score_pred)
    elif loss_type == 'l2':
        loss_val = F.mse_loss(noise, score_pred)
    elif loss_type == "huber":
        loss_val = F.smooth_l1_loss(noise, score_pred)
    else:
        raise NotImplementedError()

    return loss_val
