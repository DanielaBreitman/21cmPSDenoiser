#Imports
import numpy as np
import os, datetime
from glob import glob
import re

from torch import nn
import torch

from tqdm import tqdm
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_default_device('cuda')

import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from scipy.stats import norm
import matplotlib.pyplot as plt
import h5py, corner
from scipy.interpolate import interp1d

from matplotlib import rcParams
rcParams.update({'font.size': 12})

import os
import urllib.request
from urllib.error import HTTPError


import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm.notebook import tqdm
import gc 



# Path to the folder where the datasets are/should be downloaded (e.g. CIFAR10)
DATASET_PATH = os.environ.get("PATH_DATASETS", "data")
# Path to the folder where the pretrained models are saved
CHECKPOINT_PATH = os.environ.get("PATH_CHECKPOINT", "saved_models/tutorial9")

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("Device:", device)

from pathlib import Path
import os
import sys
sys.path.insert(1, '/home/dbreitman')
from dani_utils.plotting import imshow_error_2D, plot_hist
from dani_utils.test_funcs import get_FE, diag_cov

path = '/projects/cosmo_database/dbreitman/CV_PS/Full_May2023_DB/'
model_path = '/home/dbreitman/CV_PS_denoising/Dec_models/'

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--model_path", "-mp", type = str, default = model_path)
parser.add_argument("--model_name", "-mn", type = str)
parser.add_argument("--debug", "-debug", type = bool, default = False)

inputs = parser.parse_args()
model_path = inputs.model_path
model_name = inputs.model_name
debug = inputs.debug

with np.load(model_path + 'norms.npz') as f:
    noisy_mean = f['noisy_mean']
    noisy_std = f['noisy_std']
    mean_mean = f['mean_mean']
    mean_std = f['mean_std']

f = np.load(path + 'dec_db_50_thetas_nointerp_nolog_lesszs.npz')
seeds = f['PS_2D_seeds']
means = f['PS_2D_means']

zs = f['redshifts']
kperp = f['kperp']
kpar = f['kpar']
modes = f['modes']
std_means = f['PS_2D_std_means']
param_idx = f['param_idx']

param_idx_test = np.array([[pi] * len(zs) for pi in param_idx]).ravel()

std_poisson = f['poisson_stds']
std_diff = np.log10(std_poisson) - np.log10(std_means)
if debug:
    print(seeds.max(), means.max())
    print(std_diff.min(), std_diff.max())

N_samples = seeds.shape[0]
zs_test = list(zs) * N_samples
shape1 = (seeds.shape[0] * seeds.shape[1], seeds.shape[2], seeds.shape[3])
std_diff = std_diff.reshape(shape1)

std_means = std_means.reshape(shape1)
zs_test = np.reshape(zs_test, shape1[0])
#std_poisson = std_poisson.reshape(shape1[:-1])

seeds = seeds.reshape(shape1)
means = means.reshape(shape1)
noisy = (np.log10(seeds) - noisy_mean) / noisy_std

mean = (np.log10(means) - mean_mean) / mean_std

X_test = noisy[:,np.newaxis,...]
Y_test = np.append(mean[...,np.newaxis], std_diff[...,np.newaxis], axis = -1)

if debug:
    print(X_test.shape)
    
from model import Autoencoder

model = Autoencoder(in_ch=1)

model.load_state_dict(torch.load(model_path + model_name))
model.eval()
Y_pred = []
nbatches = 600
chunk_size = X_test.shape[0] // nbatches

for i in range(nbatches):
    if i < nbatches - 1:
        Y_pred.extend(model(torch.Tensor(X_test[i*chunk_size:(i+1)*chunk_size, ...]).to(device)).detach().cpu().numpy())
    else:
        Y_pred.extend(model(torch.Tensor(X_test[i*chunk_size:, ...]).to(device)).detach().cpu().numpy())
        
Y_pred = np.array(Y_pred)

diff_pred = (torch.Tensor(Y_pred[:,1,...]))
mean_pred = 10**(((torch.Tensor(Y_pred[:,0,...]) * mean_std) + mean_mean)).cpu().numpy()

std_pred = 10**(torch.log10(mean_pred / torch.sqrt(torch.Tensor(modes))) - diff_pred).cpu().numpy()
m = mean_pred < 1e-2
mean_pred[m] = 1e-2

m = std_pred < 1e-2
std_pred[m] = 1e-2

mu_fe, std_fe = get_FE(Y_pred, Y_test, mean_mean, mean_std, modes)

print('Mean FE mean: ', np.nanmean(mu_fe), ', Mean FE STD: ', np.nanmean(std_fe))

fe_mean = abs((mean_pred - means) / means)
if debug:
    print('Mean FE mean: ', np.nanmean(fe_mean))

fe_std = abs((std_pred - std_means) / std_means)
if debug:
    print('Mean FE STD:', np.nanmean(fe_std))

### MEAN ERROR VS K PLOTS ###

# FnP   
f = np.load(path + 'FnP_db.npz')
seeds = f['PS_2D_seeds']

zs = f['redshifts']
kperp = f['kperp']
kpar = f['kpar']
param_idx_fnp = f['param_idx']

#std_poisson = std_poisson.reshape(shape1[:-1])

s = (seeds.shape[0]*seeds.shape[1], seeds.shape[2], seeds.shape[3], seeds.shape[4])
s1 = seeds.shape
#fnp_seeds = (np.log10(seeds.reshape(s)) - noisy_mean) / noisy_std
fnp_seeds = seeds.reshape(s1)

if debug:
    print('FnP seeds: ', seeds.min(), seeds.max())
    print(seeds.shape)
    print(fnp_seeds.min(), fnp_seeds.max())
    
    
floor = 1e-2

z_idxs = np.arange(len(zs))
thetas = np.arange(50) + 200

# for violin plot
denoiser_fe_vs_z = np.zeros((len(zs),32*32))
pair1_fe_vs_z = np.zeros((len(zs),32*32))
pair2_fe_vs_z = np.zeros((len(zs),32*32))
pair3_fe_vs_z = np.zeros((len(zs),32*32))

for ii, z_idx in enumerate(z_idxs):
    print('Redshift: ', zs[z_idx])
    m8 = zs_test == zs[z_idx]
    avg_over_pairs = np.zeros((len(thetas), len(kperp), len(kpar)))
    avg_over_pairs2 = np.zeros((len(thetas), len(kperp), len(kpar)))
    avg_over_pairs3 = np.zeros((len(thetas), len(kperp), len(kpar)))
    for theta in thetas:
        # Compare w FnP
        mtest = param_idx_test == theta
        mfnp = param_idx_fnp == theta

        test_fe = mu_fe[mtest]

        fnp_fe = []
        fnp_fe2 = []
        fnp_fe3 = []
        fnp_fe4 = []
        fnp_all = []

        fnp_seeds_theta = fnp_seeds[mfnp]

        mc_mean = means[mtest][zs_test[mtest] == zs[z_idx]][0]#Y_test[...,0][mtest][0]
        mc_mean[mc_mean < floor] = floor
        npairs = fnp_seeds_theta.shape[0]
        for i in range(npairs):
            m1 = np.mean(fnp_seeds_theta[i,...], axis = 0)[z_idx]
            this_fe = abs((m1 - mc_mean) / mc_mean)
            fnp_fe.append(this_fe)
            
            pick_seeds = np.arange(fnp_seeds_theta.shape[0])
            np.random.shuffle(pick_seeds)
            
            two_seeds = pick_seeds[:2]
            s3 = (fnp_seeds_theta[two_seeds,...].shape[0]*2, len(zs), 32, 32)
            m1 = np.mean(fnp_seeds_theta[two_seeds,...].reshape(s3), axis = 0)[z_idx]
            fnp_fe2.append(abs((m1 - mc_mean) / mc_mean))

            three_seeds = pick_seeds[:3]
            s3 = (fnp_seeds_theta[three_seeds,...].shape[0]*2, len(zs), 32, 32)
            m1 = np.mean(fnp_seeds_theta[three_seeds,...].reshape(s3), axis = 0)[z_idx]
            
            fnp_fe3.append(abs((m1 - mc_mean) / mc_mean))

            #four_seeds = pick_seeds[:4]
            #s3 = (fnp_seeds_theta[four_seeds,...].shape[0]*2, len(zs), 32, 32)
            #m1 = 10**(((torch.Tensor(np.mean(fnp_seeds_theta[four_seeds,...].reshape(s3), axis = 0) ) * mean_std) + mean_mean)).cpu().numpy()

            #fnp_fe4.append(abs((m1 - mc_mean) / mc_mean))
        s3 = (fnp_seeds_theta.shape[0]*2, len(zs), 32, 32)
        m1 = np.mean(fnp_seeds_theta.reshape(s3), axis = 0)

        fnp_feall = abs((m1 - mc_mean) / mc_mean)
        fnp_fe = np.array(fnp_fe)
        fnp_fe2 = np.array(fnp_fe2)
        fnp_fe3 = np.array(fnp_fe3)
        fnp_fe4 = np.array(fnp_fe4)
        fnp_feall = np.array(fnp_feall)
        #print('SHAPE', fnp_fe.shape)
        avg_over_pairs[theta-200] = np.mean(fnp_fe, axis = 0)
        avg_over_pairs2[theta-200] = np.mean(fnp_fe2, axis = 0)
        avg_over_pairs3[theta-200] = np.mean(fnp_fe3, axis = 0)
    avg_over_thetas = np.mean(avg_over_pairs, axis = 0)
    avg_over_thetas2 = np.mean(avg_over_pairs2, axis = 0)
    avg_over_thetas3 = np.mean(avg_over_pairs3, axis = 0)
    
    denoiser_fe_vs_z[ii,...] = np.mean(fe_mean[m8], axis = 0).ravel()*100.
    pair1_fe_vs_z[ii,...] = avg_over_thetas.ravel()*100.
    pair2_fe_vs_z[ii,...] = avg_over_thetas2.ravel()*100.
    pair3_fe_vs_z[ii,...] = avg_over_thetas3.ravel()*100.

    x = np.linspace(0,50,50)
    plt.hist(np.mean(fe_mean[m8], axis = 0).ravel()*100., bins = x, color = 'k', label = 'Denoiser', density=True, histtype='step')
    plt.hist(avg_over_thetas.ravel()*100., bins = x, color = 'r', label = '1 Pair', density = True, histtype='step')
    plt.hist(avg_over_thetas2.ravel()*100., bins = x, color = 'lime', label = '2 Pairs', density = True, histtype='step')
    plt.hist(avg_over_thetas3.ravel()*100., bins = x, color = 'b', label = '3 Pairs', density = True, histtype='step')
    #plt.hist(fnp_feall[z_idx,...].ravel()*100., bins = x, color = 'lime', label = 'All 9 Pairs', density = True, histtype='step')
    plt.legend()
    plt.title('z ~ ' + str(round(zs[z_idx],1)))
    plt.xlabel('FE on the mean (%)')
    plt.tight_layout()
    plt.savefig(model_path+'Plots/'+model_name+'FnP_z_' + str(ii))
    plt.clf()

#Do violin plot 
plt.figure(figsize = (12,8))
rcParams.update({'font.size': 25})
cs = ['k', 'b', 'lime']
quants = np.array([0.16, 0.5, 0.84] * len(zs)).reshape((len(zs),3)).T
print(quants)
#plt.plot(zs, avg_over_dims[:, b], marker = '.', color = cs[j], label = 'z ~ ' + str(np.round(zs[b], 1)), alpha = 0.2)
parts = plt.violinplot(denoiser_fe_vs_z.T, zs, widths = 0.5, quantiles = quants)
#plt.errorbar(nseeds, avg_over_dims[:, b], std_over_dims[:, b], ls = None, alpha = 0.5, color = 'k')
for pc in parts['bodies']:
    pc.set_facecolor('k')
    pc.set_edgecolor('k')
    pc.set_alpha(0.5)
for partname in ('cbars','cmins','cmaxes'):
    vp = parts[partname]
    vp.set_edgecolor('k')
    vp.set_linewidth(2)
    
parts = plt.violinplot(pair1_fe_vs_z.T, zs, widths = 0.5, quantiles = quants)
#plt.errorbar(nseeds, avg_over_dims[:, b], std_over_dims[:, b], ls = None, alpha = 0.5, color = 'k')
for pc in parts['bodies']:
    pc.set_facecolor('r')
    pc.set_edgecolor('r')
    pc.set_alpha(0.2)
for partname in ('cbars','cmins','cmaxes'):
    vp = parts[partname]
    vp.set_edgecolor('r')
    vp.set_linewidth(2)
    
parts = plt.violinplot(pair2_fe_vs_z.T, zs, widths = 0.5, quantiles = quants)
#plt.errorbar(nseeds, avg_over_dims[:, b], std_over_dims[:, b], ls = None, alpha = 0.5, color = 'k')
for pc in parts['bodies']:
    pc.set_facecolor('lime')
    pc.set_edgecolor('lime')
    pc.set_alpha(0.2)
for partname in ('cbars','cmins','cmaxes'):
    vp = parts[partname]
    vp.set_edgecolor('lime')
    vp.set_linewidth(2)

parts = plt.violinplot(pair3_fe_vs_z.T, zs, widths = 0.5, quantiles = quants)
#plt.errorbar(nseeds, avg_over_dims[:, b], std_over_dims[:, b], ls = None, alpha = 0.5, color = 'k')
for pc in parts['bodies']:
    pc.set_facecolor('b')
    pc.set_edgecolor('b')
    pc.set_alpha(0.2)
for partname in ('cbars','cmins','cmaxes'):
    vp = parts[partname]
    vp.set_edgecolor('b')
    vp.set_linewidth(2)
    #
#plt.axvline(100, color = 'r', ls = '--', alpha = 0.5, label = '100 samples')
#plt.axhline(2.8, color = 'b', ls = '-.', alpha = 0.5)
#plt.axhline(0, color = 'k', ls = '-', alpha = 0.3)
plt.xlabel('Redshift')
plt.ylabel('FE (%)')

handles = [Line2D([], [], color='k', linewidth = 3, ls = '-'),
           Line2D([], [], color='r', linewidth = 3, ls = '-'),
           Line2D([], [], color='lime', linewidth = 3, ls = '-'),
           Line2D([], [], color='b', linewidth = 3, ls = '-')
            ]

plt.legend(handles = handles, loc = (0.7,0.7), frameon = False, labels = ['Denoiser', '1 Pair', '2 Pairs', '3 Pairs'], fontsize = 20)

plt.savefig(model_path+'Plots/'+model_name+'fe_vs_z.png', dpi = 150)
plt.show()
    