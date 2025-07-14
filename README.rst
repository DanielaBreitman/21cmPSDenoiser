================================================================================
21cmPSDenoiser - A score-based diffusion model that denoises 21-cm power spectra
================================================================================

``21cmPSDenoiser`` is a package that provides a score-based diffusion model trained on 21cmFAST simulations that is capabale of significantly reducing the effect of sample variance on individual 21-cm power spectrum (PS) realisations.
In Breitman+25, we find that it's better to reproduce the 21-cm PS calculation as closely as possible to the training set for optimal performance. This is especially true when applying ``21cmPSDenoiser`` on 21-cm PS from different simulators and / or physical models.
The 21-cm power spectra in the training set have been computed with `tuesday <https://github.com/21cmfast/tuesday>`_, a wrapper around `powerbox <https://github.com/steven-murray/powerbox>`_.
In the near future, we will provide a script to calculate the 21-cm PS from a lightcone in the exact same way as was done in the training set.

The package can be installed with pip via ``pip install 21cmPSDenoiser`` and tutorials are in ``docs/tutorials``.

If you use this code in your research, please cite Breitman+25.
