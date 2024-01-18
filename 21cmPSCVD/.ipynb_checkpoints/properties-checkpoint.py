from pathlib import Path
import numpy as np

class DenoiserConstants:
    """A class that contains the constants of the denoiser."""
    
    def __init__(self):
        
        here = Path(__file__).parent
        all_csts = np.load('/home/dbreitman/CV_PS_denoising/Dec_models/norms.npz')#np.load(here / "denoiser_constants.npz")
        
        self.noisy_PS_mean = all_csts['noisy_mean']
        self.noisy_PS_std = all_csts['noisy_std']
        
        self.mean_PS_mean = all_csts['mean_mean']
        self.mean_PS_std = all_csts['mean_std']
        
        #self.std_mean = all_csts['std_mean']
        #self.std_std = all_csts['std_std']
        f = np.load('/projects/cosmo_database/dbreitman/CV_PS/Full_May2023_DB/dec_db_50_thetas_nointerp_nolog_lesszs.npz')
        
        self.modes = f['modes']
        
        self.h = 0.6727
        self.normalized_quantities = ['mean_PS']
        
denoiser_csts = DenoiserConstants()