"""Plotting routines for plots in Breitman+25a"""

from astropy import units as u
from astropy.cosmology.units import littleh
from astropy.cosmology import Planck15 as cosmo
from scipy.constants import c
import matplotlib.colors as colors
import numpy as np
speed_of_light = c * u.m / u.s
from matplotlib import rcParams
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline, interp1d

def freq2z(f):
    return 1420.4 / f  - 1.

def z2freq(z):
    return 1420.4 / (z + 1.)

def alpha(z):
    nu_21 = z2freq(z) *u.MHz
    speed_of_light = c * u.m / u.s
    return (speed_of_light * (1+z) / (nu_21 * cosmo.H(z))).to(u.Mpc / u.MHz)

def delay2kpar(tau, freq = None, z = None):
    if z is None:
        z = freq2z(freq.value)
    if freq is None:
        freq = z2freq(z)*u.MHz
    kpar = 2*np.pi * np.abs(tau) / alpha(z)
    return kpar.to(1/u.Mpc)

def kperp2baseline(kperp, freq = None, z = None):
    if z is None:
        z = freq2z(freq.value)
    if freq is None:
        freq = z2freq(z)*u.MHz
    baseline = np.abs(kperp * cosmo.comoving_distance(z) * speed_of_light / (2 * np.pi * freq))
    return baseline.to(u.m)

def horizon_limit(kperp, freq = None, z = None, buffer=300*u.ns):
    if z is None:
        z = freq2z(freq.value)
    if freq is None:
        freq = z2freq(z)*u.MHz
    baseline = kperp2baseline(kperp, freq, z)
    tau_wedge = baseline / speed_of_light # Eqn 8 Josh's paper
    return delay2kpar(tau_wedge + buffer, freq, z)

def get_hera_var_at_z(redshift):
    with np.load('HERA_mock_sensitivities_nosample.npz', allow_pickle=True) as f:
        freq = z2freq(redshift)
        bands = f['band_MHz']
        my_band = np.argmin(abs(bands - freq))
        print("HERA redshift", np.round(freq2z(bands[my_band]),1), "vs input redshift", np.round(redshift,1))
        this_band = str(int(bands[my_band]))+'MHz'
        kperp_centers = np.exp((np.log(f['kperp_edges'][1:]) + np.log(f['kperp_edges'][:-1])) / 2)
        kpar_centers = (f['kpar_edges'][1:] + f['kpar_edges'][:-1])/2
        return freq2z(bands[my_band]), kperp_centers, kpar_centers, f[this_band].item()['sample_2D_mK2']**2 +  f[this_band].item()['thermal_2D_mK2']**2

def results_fig4(kperp,kpar, noisy_sample, mean, pred, redshift, 
                 cmap='viridis', cmap2='cividis', fname=None, 
                 vminfe=None, vmaxfe=None, fontsize=30, figsize=(18,16), 
                 label3 = r'$\frac{|\Delta^2_{21} - \Delta^2_{21,test}|}{\sigma_{\rm HERA}}$'):
    r"""Plot Figure 4 in Breitman+25a."""
    rcParams.update({'font.size': fontsize})
    fig, ax = plt.subplots(nrows=3, ncols=2,sharex=True, sharey=True, figsize = figsize, layout='constrained')
    vmin = np.percentile(mean, 2.5)
    vmax = np.percentile(mean,97.5)
    ax[0,0].pcolormesh(kperp, kpar,noisy_sample.T, vmin = vmin, vmax = vmax)
    ax[0,0].text(0.25,0.7,'Sample', color = 'k')
    ax[0,0].text(0.25,0.42,'z ~ ' + str(np.round(redshift,1)), color = 'k')

    im = ax[0,1].pcolormesh(kperp, kpar, pred.T, vmin = vmin, vmax = vmax)
#     ax[0,1].text(0.5,0.7,'Mean', color = 'k')
#     im = ax[0,2].pcolormesh(kperp, kpar, pred.T, vmin = vmin, vmax = vmax)
    ax[0,1].text(0.043,0.7,'py21cmDenoiser', color = 'k')
    fig.colorbar(im, label=r'$\Delta^2_{21}$ [mK$^2$]')
    mu_fe = abs((noisy_sample - mean)/mean)*100.#, axis = (-1,-2))
    #abs_err = abs(pred - mean)
    if vminfe is None:
        vminfe = np.nanpercentile(mu_fe, 2.5)
    if vmaxfe is None:
        vmaxfe = np.nanpercentile(mu_fe,97.5)
    ax[1,0].pcolormesh(kperp, kpar,mu_fe.T, norm=colors.LogNorm(vmin=vminfe, vmax=vmaxfe),cmap = cmap)
    mu_fe = abs((pred - mean)/mean)*100.
    im = ax[1,1].pcolormesh(kperp, kpar,mu_fe.T, norm=colors.LogNorm(vmin=vminfe, vmax=vmaxfe),cmap = cmap)
    fig.colorbar(im, label=r'FE(%)')
    
#     im = ax[1,1].pcolormesh(kperp, kpar,abs_err.T, cmap = 'YlGnBu')
#     plt.colorbar(im, label='Abs Diff [mK$^2$]')
    # Mock
    hera_z, hera_kperp, hera_kpar, hera_var = get_hera_var_at_z(redshift)
    hera_var[np.isinf(hera_var)] = np.nan
    interped_pred = RectBivariateSpline(kperp, kpar, pred)(hera_kperp,hera_kpar)
    interped_inp = RectBivariateSpline(kperp, kpar, noisy_sample)(hera_kperp,hera_kpar)
    interped_mean = RectBivariateSpline(kperp, kpar, mean)(hera_kperp,hera_kpar)
    #interp pred and mean to hera grid
    hera_err = abs((interped_pred - interped_mean)/np.sqrt(hera_var))
    hera_err_noisy = abs((interped_inp - interped_mean)/np.sqrt(hera_var))
    vmin = 1e-2
    vmax = 1.
    ax[2,0].pcolormesh(kperp, kpar, hera_err_noisy.T, 
                       norm=colors.LogNorm(vmin=vmin, vmax=vmax),
                       cmap = cmap2)
    im = ax[2,1].pcolormesh(kperp, kpar,hera_err.T, 
                            norm=colors.LogNorm(vmin=vmin, vmax=vmax),
                            cmap = cmap2)
    cbar = fig.colorbar(im, ticks = [1, 0.1, 0.01, 0.001],
                 label=label3)
    
    min_hera_kperp = 9e-3 #hMpc^{-1}
    try:
        kperp = kperp.value
    except:
        pass
    try:
        hlb = hlb.value
    except:
        pass
    hera_c = 'b'
    
    kperpp = np.linspace(4e-3, 3.,1000)/u.Mpc
    hl = horizon_limit(kperpp, z2freq(redshift)*u.MHz, buffer = 0*u.ns).value
    hlb = horizon_limit(kperpp, z2freq(redshift)*u.MHz, buffer = 300*u.ns).value

    m = kperpp.value >= min_hera_kperp * cosmo.h
    ax[2,0].plot(kperpp[m], hlb[m], color = 'k', lw=2, label = 'HERA 300 ns buffer')
    ax[2,1].plot(kperpp[m], hlb[m], color = 'k', lw=2)

    ax[2,0].fill_between(kperpp[m].value, np.zeros(len(hlb[m])),hlb[m], color = 'k', alpha = 0.3, zorder=1)
    ax[2,1].fill_between(kperpp[m].value, np.zeros(len(hlb[m])),hlb[m], color = 'k', alpha = 0.3, zorder=1)

    ax[2,0].vlines(kperpp[m][0].value, hlb[m][0], hlb[-1], lw = 2, color = 'k')
    ax[2,1].vlines(kperpp[m][0].value, hlb[m][0], hlb[-1], lw = 2, color = 'k')
    #ax[1,2].set_xlim(kperpp[m][0].value/cosmo.h,kperp[-1]/cosmo.h)
    ff = interp1d(kperpp.value, hl, bounds_error=False)
    xx = np.logspace(np.log10(kperpp[0].value), np.log10(kperp[-1]))
    ax[2,0].plot(xx, ff(xx), color = 'cyan', ls = '--', zorder = 23,lw = 2, alpha = 0.8, label = 'Horizon limit')
    ax[2,1].plot(xx, ff(xx), color = 'cyan', ls = '--', zorder = 23,lw = 2, alpha = 0.8)

    xvals = np.linspace(kperpp[0].value, kperpp[-1].value, 100)
    mus = [0.97]
    cs = ['r', 'orange']
    r = np.logspace(-10,1., 10000)
    for ii, muval in enumerate(mus):
        theta = np.arccos(muval)
        x = r*np.sin(theta)
        y = r*muval
        m = x > kperpp[0].value
        ax[2,0].plot(x[m],y[m],color = cs[ii], lw = 2, zorder = 23,label = r'$\mu_{\rm min}$ = ' + str(muval))
        ax[2,1].plot(x[m],y[m],color = cs[ii], lw = 2, zorder = 23)

    ax[2,0].set_xlabel(r'k$_\perp$ [Mpc$^{-1}$]')
    ax[2,1].set_xlabel(r'k$_\perp$ [Mpc$^{-1}$]')
    ax[2,0].set_ylabel(r'k$_\parallel$ [Mpc$^{-1}$]')
    ax[0,0].set_ylabel(r'k$_\parallel$ [Mpc$^{-1}$]')
    ax[1,0].set_ylabel(r'k$_\parallel$ [Mpc$^{-1}$]')
    ax[2,0].legend(frameon=False, loc = 4, fontsize = np.min([fontsize, 18]))

    for a in ax.ravel():
        a.set_xlim(kperp[0], kperp[-1])
        a.set_ylim(kpar[0], kpar[-1])
    plt.loglog()
    if fname is not None:
        plt.savefig(fname)
    plt.show()
    
def results_fig5(kperp, kpar,all_mu_fes, all_sv_fe, all_zs, cmap = 'cividis', z_range=[8,9], qt = '50', path=None, fontsize = 30, figsize=(18,7)):
    r"""Plot Figure 5 in Breitman+25a."""
    rcParams.update({'font.size': fontsize})
    qt = str(qt)
    m = np.logical_and(all_zs >= z_range[0], all_zs<=z_range[1])
    print("There are",sum(m),"samples in the redshift range requested.")
    fe_percs_sv = np.percentile((all_sv_fe)[m], [2.5,16,50,84,97.5], axis = 0)
    fe_percs = np.percentile((all_mu_fes)[m], [2.5,16,50,84,97.5], axis = 0)

    #abs_err_percs = np.percentile((all_sv_abserr)[m], [2.5,16,50,84,97.5], axis = 0)

    fig, ax = plt.subplots(nrows = 1, ncols = 2,sharex=True, sharey=True, figsize = figsize, layout='constrained')
    ax = np.atleast_2d(ax)
    if qt == '50':
        qt_sv = fe_percs_sv[2]
        qt_ml = fe_percs[2]
        label = 'Median '
    elif qt == '68':
        qt_sv = (fe_percs_sv[3] - fe_percs_sv[1])
        qt_ml = (fe_percs[3] - fe_percs[1])
        label = r'1$\sigma$ '
    elif qt == '95':
        qt_sv = (fe_percs_sv[4] - fe_percs_sv[0])
        qt_ml = (fe_percs[4] - fe_percs[0])
        label = r'2$\sigma$ '
    else:
        raise ValueError('qt is 50, 68 or 95.')
    vmin = np.min([0.8,np.nanpercentile(qt_sv, 2.5)])
    vmax = np.nanpercentile(qt_sv, 97.5)
    im = ax[0,1].pcolormesh(kperp, kpar, qt_ml.T, norm = colors.LogNorm(vmin=vmin, vmax=vmax) if vmin > 0 else colors.SymLogNorm(vmin=vmin, vmax=vmax, linthresh=0.1), cmap = cmap,)
    plt.colorbar(im, label=label+'FE (%)')
    ax[0,0].pcolormesh(kperp, kpar, qt_sv.T,norm = colors.LogNorm(vmin=vmin, vmax=vmax) if vmin > 0 else colors.SymLogNorm(vmin=vmin, vmax=vmax, linthresh=0.1),  cmap = cmap,)

#     vmin = np.nanpercentile(fe_percs_sv[3] - fe_percs_sv[1], 2.5)
#     vmax = np.nanpercentile(fe_percs_sv[3] - fe_percs_sv[1], 97.5)

#     im = ax[1,1].pcolormesh(kperp, kpar, (fe_percs[3] - fe_percs[1]).T, norm = colors.LogNorm(vmin=vmin, vmax=vmax), cmap = cmap,)
#     plt.colorbar(im, label='FE(%) 68% CL')

#     ax[1,0].pcolormesh(kperp, kpar, (fe_percs_sv[3] - fe_percs_sv[1]).T, norm = colors.LogNorm(vmin=vmin, vmax=vmax), cmap = cmap,)

    plt.loglog()    
    
    ax[0,0].text(0.2,0.7,'Sample', color = 'white')
    ax[0,0].text(0.2,0.42,'z ~ ' + str(np.round(np.mean(all_zs[m]),1)), color = 'white')
    ax[0,1].text(0.03,0.7,'py21cmDenoiser', color = 'white')
    
    ax[0,0].set_xlabel(r'k$_\perp$ [Mpc$^{-1}$]')
    ax[0,1].set_xlabel(r'k$_\perp$ [Mpc$^{-1}$]')
    ax[0,0].set_ylabel(r'k$_\parallel$ [Mpc$^{-1}$]')
    #ax[1,0].set_ylabel(r'k$_\parallel$ [Mpc$^{-1}$]')
    if path is not None:
        plt.savefig(path+'NN_'+qt+'_overtest_z'+str(int(np.round(np.mean(all_zs[m]))))+'_sv.png')
    plt.show()