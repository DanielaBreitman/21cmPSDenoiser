"""Module whose functionality is to organise the denoiser output."""
from __future__ import annotations

import dataclasses as dc
from dataclasses import dataclass
from pathlib import Path
from typing import Generator

import numpy as np

from properties import denoiser_csts
    
    
    
class NNOutput:

    def __init__(self, output: np.ndarray,
                 kperp: np.ndarray,
                 kpar: np.ndarray,):
        self.output = output
        self.kperp = kperp
        self.kpar = kpar
        self.csts = denoiser_csts

    @property
    def mean_PS(self) -> np.ndarray:
        """Denoised cylindrical PS"""
        return self.output[:,0,...]
    
    @property
    def std_PS(self) -> np.ndarray:
        """Denoised cylindrical PS"""
        #return self.output[:,1,...]
        return 10**(np.log10(self.renormalize('mean_PS') / np.sqrt(self.csts.modes)) - self.output[:,1,...])
    
    def renormalize(self, name: str):
        """Renormalize a normalized quantity.

        This ajudsts the quantity (as it exists in this class) back to its native
        range by adding the emulator data mean and multiplying by the emulator data
        standard deviation.
        """
        if name not in self.csts.normalized_quantities:
            raise ValueError(
                f"Cannot renormalize {name}. It is not a normalized quantity."
            )
        return 10**(getattr(self.csts, f"{name}_mean") + getattr(
            self.csts, f"{name}_std"
        ) * getattr(self, name))

    def get_renormalized(self) -> DenoiserOutput:
        """Get the output with normalized quantities re-normalized.

        Returns
        -------
        EmulatorOutput
            The emulator output with normalized quantities re-normalized back to
            physical units. Nothing is in log except UV LFs.
        """
        # Restore dimensions
        # Renormalize stuff that needs renormalization

        renorm = {k: self.renormalize(k) for k in self.csts.normalized_quantities}

        other = {
            k.name: getattr(self, k.name)
            for k in dc.fields(DenoiserOutput)
            if k.name not in renorm
        }

        out = {**renorm, **other}


        return DenoiserOutput(**out).squeeze()
    

@dataclass(frozen=True)
class DenoiserOutput:
    """A simple class that makes it easier to access the corrected denoiser output."""

    mean_PS: np.ndarray
    std_PS: np.ndarray
    kperp: np.ndarray
    kpar: np.ndarray

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