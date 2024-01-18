from denoiser import Denoiser
import numpy as np

path = '/projects/cosmo_database/dbreitman/CV_PS/Full_May2023_DB/'
f = np.load(path + 'dec_db_50_thetas_nointerp_nolog_lesszs.npz')
seeds = f['PS_2D_seeds']
means = f['PS_2D_means']
kperp = f['kperp']
kpar = f['kpar']
std_means = f['PS_2D_std_means']

shape = (means.shape[0] * means.shape[1], means.shape[2], means.shape[3])
means = means.reshape(shape)
std_means = std_means.reshape(shape)
normed_in, out = Denoiser().predict(seeds.reshape(shape), kperp, kpar)
print('MEAN', np.mean(abs((out.mean_PS - means) / means)))
print('STD', np.mean(abs((out.std_PS - std_means) / std_means)))
print(out.mean_PS.shape, std_mean.shape)