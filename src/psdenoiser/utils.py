import torch


def extract(a: torch.Tensor, t: torch.Tensor, x_shape: tuple) -> torch.Tensor:
    r"""Extract the value of a tensor `a` at time `t` for a batch with shape `x_shape`.

    Parameters
    ----------
    a : torch.Tensor
        The tensor from which to extract values.
    t : torch.Tensor
        The tensor containing the time steps at which to extract values.
    x_shape : tuple
        The shape of the data for which to extract values.

    Returns
    -------
      torch.Tensor
        A tensor containing the extracted values,
        reshaped to match the batch size of `t`.
    """
    batch_size = t.shape[0]
    out = a.cpu().gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


def transform(x: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    r"""Transform input x to be in [-1, 1] by first taking log10.

    Parameters
    ----------
    x : torch.Tensor
        The input tensor to be transformed.
    scale : torch.Tensor
        The scale factor to be applied.
    bias : torch.Tensor
        The bias to be applied.

    Returns
    -------
    torch.Tensor
        The transformed tensor in the range [-1, 1].
    """
    d = torch.log10(x)
    unit = (d - bias) / scale
    return unit * 2 - 1  # scale to [-1, 1]


def reverse_transform(
    y: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    r"""Reverse the transformation from [-1, 1] to the original scale.

    Parameters
    ----------
    y : torch.Tensor
        The input tensor in the range [-1, 1].
    scale : torch.Tensor
        The scale factor used in the original transformation.
    bias : torch.Tensor
        The bias used in the original transformation.

    Returns
    -------
    torch.Tensor
        The tensor transformed back to the original scale.
    """
    unit = (y + 1) / 2
    d = unit * scale + bias
    return 10**d


def to_flattened_numpy(x):
    """Flatten a torch tensor `x` and convert it to numpy."""
    return x.detach().cpu().numpy().reshape((-1,))


def from_flattened_numpy(x, shape):
    """Form a torch tensor with the given `shape` from a flattened numpy array `x`."""
    return torch.from_numpy(x.reshape(shape))
