"""Download and install the emulator data."""

from __future__ import annotations

import logging
from warnings import warn

import git

from .config import CONFIG


log = logging.getLogger(__name__)


def download_denoiser() -> None:
    """Download the py21cmPSDenoiser model from Hugging Face.

    """
    if (CONFIG.data_path / "21cmEMU").exists():
        repo = git.Repo(CONFIG.data_path / "21cmEMU")
    elif not CONFIG["disable-network"]:
        URL = "https://huggingface.co/DanielaBreitman/21cmEMU"
        repo = git.Repo.clone_from(URL, CONFIG.data_path / "21cmEMU")

    # Check download
    p = CONFIG.data_path / "PSDenoiser" / "PSDenoiser" / "denoiser_model.pt"
    if not p.exists() or p.stat().st_size < 1e6:
        raise RuntimeError(
            "The emulator huggingface repo was not cloned properly.\n"
            "Check that git-lfs is installed properly on your system.\n"
            "If git-lfs cannot be installed or internet "
            "connection is not available, "
            "manually clone the repo on another machine with git-lfs"
            "and internet using\n"
            "git clone -v -- https://huggingface.co/DanielaBreitman/21cmEMU\n"
            "Then, ensure that it downloaded fully by running: "
            "du -sh 21cmEMU \n"
            "The folder should be about 500M: 250M for the model, 250M for .git. "
            "Now copy this folder and its contents "
            "over to your other machine and put it in "
            " ~/.local/share/py21cmEMU/21cmEMU "
        )

