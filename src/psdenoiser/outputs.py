"""Module to organise the denoiser output."""

from __future__ import annotations

import dataclasses as dc
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path

import astropy.units as un
import numpy as np

from psdenoiser.properties import denoiser_csts


@dataclass(frozen=True)
class DenoiserOutput:
    """A simple class that makes it easier to access the denoiser output."""

    deltasq_samples: un.Quantity
    deltasq_median: un.Quantity
    deltasq_std: un.Quantity
    kperp: un.Quantity
    kpar: un.Quantity

    csts = denoiser_csts

    def keys(self) -> Generator[str, None, None]:
        """Yield the keys of the main data products."""
        for k in dc.fields(self):
            yield k.name

    def items(self) -> Generator[tuple[str, np.ndarray], None, None]:
        """Yield the keys and values of the main data products, like a dict."""
        for k in self.keys():
            yield k, getattr(self, k)

    def __getitem__(self, key: str) -> np.ndarray:
        """Allow access to attributes as items."""
        return getattr(self, key)

    def squeeze(self):
        """Return a new EmulatorOutput with all dimensions of length 1 removed."""
        return DenoiserOutput(**{k: np.squeeze(v) for k, v in self.items()})

    @property
    def denoiser_median_on_test_mean(self):
        """Return the denoiser median error on the test set."""
        return (
            self.csts.denoiser_median_on_test_mean_percent / 100.0 * self.deltasq_median
        )

    def write(
        self,
        fname: str | Path,
        noisy_ps: np.ndarray | None = None,
        store: list[str] | None = None,
        clobber: bool = False,
    ):
        """Write this instance's data to a file.

        This saves the output as a numpy .npz file. The output is saved as a dictionary
        with the keys being the names of the attributes of this class and the values
        being the corresponding values of those attributes. If theta is not None, then
        the inputs are also saved under the key "inputs".

        Parameters
        ----------
        fname : str or Path
            The filename to write to.
        noisy_ps : np.ndarray or dict or None, optional
            The input noisy cylindrical PS associated with this output data to write to the file.
            If None, the inputs are not written.
        store : list of str or None, optional
            The names of the attributes to write to the file. If None, all attributes
            are written.
        clobber : bool, optional
            Whether to overwrite the file if it already exists.
        """
        if store is None:
            store = list(self.__dict__.keys())

        pth = Path(fname)
        if pth.exists() and not clobber:
            raise ValueError(f"File {pth} exists and clobber=False.")

        out = {k: getattr(self, k) for k in store}
        if noisy_ps is not None:
            out["inputs"] = noisy_ps

        np.savez(fname, out)
