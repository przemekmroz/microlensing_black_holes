import numpy as np
import emcee
import argparse
import yaml

import parsubs
from galactic_model import rho_bulge, rho_disk
from imf import IMF

"""
Estimate the mass and distance of a dark lens using Galactic priors

This code calculates the posterior distribution of the lens mass and distance
given the light curve data and Gaia proper motion of the source star.

My model has eight parameters: (t0, tE, u0, piEN, piEE, mu, mu_s_N, mu_s_E).
The first five of them describe the shape of the light curve: t0 - time of the
minimal separation between the lens and the source, u0 - lens-source angular
separation at t0 (expressed in thetaE units), tE - Einstein timescale, piEN,
piEE - North and East components of the microlens parallax vector. Following
Gould (2004), these quantities are measured in the geocentric frame that is
moving with a velocity equal to that of the Earth at fiducial time t_0,par. The
sixth parameter, mu, is the relative lens-source proper motion measured relative
to the geocentric frame. The last two parameters, mu_s_N and mu_s_E are the
North and East components of the source proper motion vector (in the barycentric
frame).

The first five parameters are constrained by the light curve data, the
lens-source proper motion is constrained from the Galactic prior, and (mu_s_N,
mu_s_E) may be taken from Gaia or OGLE Uranus project.

Usage:
------
python3 single_parallax_fit_geocentric.py FILENAME

where FILENAME is a YAML file containing the following parameters:
* photfile : name of file containing photometric data
* alpha : right ascension (deg)
* delta : declination (deg)
* t0_par : reference parameter time
* l : Galactic longitude (deg)
* b : Galactic latitude (deg)
* vN : projected Earth velocity in N direction (km/s)
* vE : projected Earth velocity in E direction (km/s)
* mu_min : minimal value of mu (mas/yr)
* mu_max : maximal value of mu (mas/yr)
* mu_E_source : proper motion of source in RA (mas/yr)
* mu_N_source : proper motion of source in Decl (mas/yr)
* mu_E_err_source : uncertainty of mu_E_source (mas/yr)
* mu_N_err_source : uncertainty of mu_N_source (mas/yr)
* pm_corr_source : correlation between mu_E_source and mu_N_source
* output : MCMC results with be written to $output.h5
* nwalkers : number of MCMC walkers
* nburnin : number of MCMC burn-in steps
* nsamples : number of MCMC steps (note that the actual number of MCMC
    steps will be nburnin + nsamples)
* t0 : initial t_0
* tE : initial t_E (d)
* u0 : initial u_0
* piEN : initial pi_EN
* piEE : initial pi_EE
* sigma_t0 : initial uncertainty of t_0
* sigma_tE : initial uncertainty of t_E
* sigma_u0 : initial uncertainty of u_0
* sigma_piEN : initial uncertainty of pi_EN
* sigma_piEE : initial uncertainty of pi_EE

Returns:
--------

This code saves the MCMC chain as $output.h5. The following data are saved:

reader = emcee.backends.HDFBackend(hdf5_file)
samples = reader.get_chain(discard=nburnin,flat=True)
log_prob = reader.get_log_prob(discard=nburnin,flat=True)
blobs = reader.get_blobs(discard=nburnin,flat=True)

samples[:, 0] : t_0
samples[:, 1] : t_E
samples[:, 2] : u_0
samples[:, 3] : pi_EN
samples[:, 4] : pi_EE
samples[:, 5] : mu
samples[:, 6] : mu_s_E
samples[:, 7] : mu_s_N

log_prob : total log-likelihood

blobs[:, 0] : F_s (source flux)
blobs[:, 1] : F_b (blend flux)
blobs[:, 2] : theta_E (angular Einstein radius, mas)
blobs[:, 3] : M (lens mass, M_Solar)
blobs[:, 4] : Dl (lens distance, kpc)
blobs[:, 5] : mu_l (heliocentric proper motion of lens in l, mas/yr)
blobs[:, 6] : mu_b (heliocentric proper motion of lens in b, mas/yr)
blobs[:, 7] : lnlike (-0.5*chi^2)
blobs[:, 8] : lp (log of Galactic prior)
blobs[:, 9] : lp_flux (log of flux prior)

P. Mroz @ Caltech, Feb 23, 2021
"""


def magnification(time, t0, tE, u0, piEN, piEE, q_n, q_e):

    """
    Returns magnification for a point-lens microlensing event

    Parameters:
    -----------
    time : times of observations
    t0   : moment of the closest approach between lens and source
    tE   : Einstein timescale
    u0   : impact parameter
    piEN : microlens parallax, North component
    piEE : microlens parallax, East component
    q_n  : projected position of the Sun, North component, Gould (2004)
    q_e  : projected position of the Sun, East component, Gould (2004)

    Returns:
    --------
    A    : magnification
    """

    tau = (time-t0) / tE
    dtau = piEN*q_n + piEE*q_e
    dbeta = piEN*q_e - piEE*q_n

    taup = tau + dtau
    betap = u0 + dbeta

    u = np.sqrt(taup**2 + betap**2)

    A = (u**2+2.0)/np.sqrt(u**2+4.0)/u

    return A


def get_blending(pars, time, flux, ferr, q_n, q_e):

    """
    Calculating best-fitting flux parameters

    Parameters:
    -----------
    pars  : model parameters (8-element vector)
    time  : times of observations
    flux  : observed flux
    ferr  : flux uncertainty
    q_n   : projected position of the Sun, North component, Gould (2004)
    q_e   : projected position of the Sun, East component, Gould (2004)

    Returns:
    --------
    Fs    : source flux
    Fb    : blend flux
    resid : residuals from the model (in standard deviations)
    """

    A = magnification(
        time, pars[0], pars[1], pars[2], pars[3], pars[4], q_n, q_e)

    weight = 1.0/(ferr*ferr)
    sum1 = np.sum(A*A*weight)
    sum2 = np.sum(A*weight)
    sum3 = np.sum(weight)
    sum4 = np.sum(flux*A*weight)
    sum5 = np.sum(flux*weight)

    W = sum1*sum3 - sum2*sum2
    Ws = sum3*sum4 - sum2*sum5
    Wb = sum1*sum5 - sum2*sum4

    if W == 0.0:
        Fs = 0.0
        Fb = -1.0
    else:
        Fs = Ws/W
        Fb = Wb/W

    resid = (flux - Fs*A - Fb)/ferr

    return Fs, Fb, resid


def chi2_photometry(pars, time, flux, ferr, q_n, q_e):

    """
    Calculates chi^2 for the light curve fit

    Parameters:
    -----------
    pars  : model parameters (8-element vector)
    time  : times of observations
    flux  : observed flux
    ferr  : flux uncertainty
    q_n   : projected position of the Sun, North component, Gould (2004)
    q_e   : projected position of the Sun, East component, Gould (2004)

    Returns:
    --------
    chi2_phot : chi^2 for the light curve fit
    (Fs, Fb)  : flux parameters
    """

    Fs, Fb, resid = get_blending(pars, time, flux, ferr, q_n, q_e)

    chi2_phot = np.sum(resid**2)

    return chi2_phot, (Fs, Fb)


def ln_prior(params, aux):

    """
    Returns log-likelihood of prior

    Parameters:
    -----------
    params : model parameters (8-element vector)
    aux    : dictionary (containing a number of constants)

    This dictionary contains the following parameters:
    * mu_min       : minimal value of mu (mas/yr)
    * mu_max       : maximal value of mu (mas/yr)
    * vN           : North component of Earth's velocity vector (km/s)
    * vE           : East component of Earth's velocity vector (km/s)
    * mu_N_source  : proper motion of the source in declination (mas/yr)
    * mu_E_source  : proper motion of the source in right ascension (mas/yr)
    * ra           : right ascension (radians)
    * dec          : declination (radians)
    * l            : Galactic longitude (radians)
    * b            : Galactic latitude (radians)
    * ra_g         : right ascension of the north galactic pole (radians)
    * dec_g        : declination of the north galactic pole (radians)
    * V_Sun        : velocity of the Sun in V direction (km/s)
    * W_Sun        : velocity of the Sun in W direction (km/s)
    * v_y_disk     : mean velocity of disk stars in y direction
    * v_z_disk     : mean velocity of disk stars in z direction
    * sigma_y_disk : dispersion of velocities of disk stars in y axis
    * sigma_z_disk : dispersion of velocities of disk stars in z axis
    * sigma_y_blg  : dispersion of velocities of bulge stars in y axis
    * sigma_z_blg  : dispersion of velocities of bulge stars in z axis
    * Ds           : source distance (kpc)
    * R0           : distance to the Galactic center (kpc)
    * massfunction : mass function exponent

    Returns:
    --------
    lp     : logarithm of prior probability

    Velocity of the Sun relative to the local standard of rest equals
    to (U_Sun,V_Sun,W_Sun) = (11.1, 12.2, 7.3) km/s
    """

    tE_geo = params[1]  # geocentric timescale
    pi_EN = params[3]  # microlensing parallax, North component
    pi_EE = params[4]  # microlensing parallax, East component
    pi_E = np.sqrt(pi_EN**2 + pi_EE**2)
    mu = params[5]  # GEOCENTRIC relative lens-source proper motion
    mu_src_E = params[6]  # heliocentric proper motion of the source
    mu_src_N = params[7]  # heliocentric proper motion of the source

    # Return -inf if mu is outside the allowed range

    if mu > aux['mu_max'] or mu < aux['mu_min']:
        return -np.inf, (0.0, 0.0, 0.0, 0.0, 0.0)

    # Calculate Einstein radius, lens mass and distance

    theta_E = mu * tE_geo / 365.25

    M = theta_E / (8.144 * pi_E)
    pi_rel = theta_E * pi_E
    Dl = 1.0 / (pi_rel + 1.0/aux['Ds'])

    # Calculate heliocentric proper motion vector of the lens

    mu_N = mu * pi_EN / pi_E + mu_src_N + aux['vN'] * pi_rel / 4.74
    mu_E = mu * pi_EE / pi_E + mu_src_E + aux['vE'] * pi_rel / 4.74

    # Transform it to Galactic coordinates

    mu_l = aux['C1'] * mu_E + aux['C2'] * mu_N
    mu_b = -aux['C2'] * mu_E + aux['C1'] * mu_N

    # Calculate tangential velocity

    V_l = 4.74 * mu_l * Dl + aux['V_Sun']
    V_b = 4.74 * mu_b * Dl + aux['W_Sun']

    # Calculate lens density

    x = Dl * aux['cosb'] * aux['sinl']
    y = aux['R0'] - Dl * aux['cosb'] * aux['cosl']
    z = Dl * aux['sinb']

    rho_b = rho_bulge(x, y, z)  # density of lenses in the bulge

    rho_d = rho_disk(x, y, z)  # density of lenses in the disk

    # Calculate velocity distributions

    prob_disk = np.exp(-0.5*((V_b-aux['v_z_disk'])/aux['sigma_z_disk'])**2)
    prob_disk /= aux['sigma_z_disk']
    prob_disk *= np.exp(-0.5*((V_l-aux['v_y_disk'])/aux['sigma_y_disk'])**2)
    prob_disk /= aux['sigma_y_disk']

    prob_bulge = np.exp(-0.5*(V_b/aux['sigma_z_blg'])**2)/aux['sigma_z_blg']
    prob_bulge *= np.exp(-0.5*(V_l/aux['sigma_y_blg'])**2)/aux['sigma_y_blg']

    # Mass function

    gM = aux['massfunction'].get_imf(M)

    # Calculate log prior

    lp = np.log(tE_geo/pi_E)
    lp += 4.0 * np.log(Dl*mu)
    lp += np.log(M*gM)
    # this comes from transforming proper motion to velocity:
    lp += 2.0 * np.log(Dl)
    lp += np.log(rho_d*prob_disk+rho_b*prob_bulge)

    w = 1.0/(1-aux['pm_corr_source']**2)
    w1 = (mu_src_N-aux['mu_N_source'])/aux['mu_N_err_source']
    w2 = (mu_src_E-aux['mu_E_source'])/aux['mu_E_err_source']

    # prior on source proper motion - take into account correlation
    # between pmra and pmdec
    lp += -0.5 * w * w1 * w1
    lp += -0.5 * w * w2 * w2
    lp += aux['pm_corr_source']*w*w1*w2

    return lp, (theta_E, M, Dl, mu_l, mu_b)


def ln_prior_flux(flux):

    """
    Returns log of prior likelihood on flux

    Parameters:
    -----------
    flux : tuple

    Returns:
    --------
    lp   : log of prior probability
    """

    if flux[1] >= 0.0:
        lp = 0.0
    else:
        lp = -0.5*(flux[1]/0.1)**2

    return lp


def ln_prob(pars, time_phot, flux, ferr, qn_phot, qe_phot, aux):

    """
    Likelihood function

    Parameters:
    -----------
    pars      : model parameters (11-element vector)
    time_phot : times of photometric observations
    flux      : observed flux
    ferr      : flux uncertainty
    qn_phot   : projected position of the Sun, North component
    qe_phot   : projected position of the Sun, East component
    aux       : dictionary (containing a number of parameters)

    Returns:
    --------
    lnlike    : log-likelihood
    """

    chi2_phot, flux_pars = chi2_photometry(
        pars, time_phot, flux, ferr, qn_phot, qe_phot)

    lnlike = -0.5*chi2_phot

    lp, phys_pars = ln_prior(pars, aux)

    lp_flux = ln_prior_flux(flux_pars)

    return lnlike + lp + lp_flux, flux_pars + phys_pars + (lnlike, lp, lp_flux)


if __name__ == '__main__':

    # Reading input parameters from a YAML file

    parser = argparse.ArgumentParser()
    parser.add_argument('yaml_file')
    args = parser.parse_args()

    with open(args.yaml_file, 'r') as fp:
        try:
            pars = yaml.safe_load(fp)
        except yaml.YAMLError as exc:
            exit(exc)

    # Loading photometric data

    hjd, mag, magerr = np.loadtxt(pars['photfile'], unpack=True)
    N_phot = len(hjd)

    if hjd[0] > 2450000.0:
        hjd -= 2450000.0
    flux = pow(10.0, 0.4*(18.0-mag))
    ferr = 0.4*np.log(10.0)*flux*magerr

    qn, qe = parsubs.geta(hjd, pars['alpha'], pars['delta'], pars['t0_par'])

    # Reading parameters

    deg = np.pi/180.0
    aux = dict()
    aux['ra'] = pars['alpha']*deg  # Right Ascension
    aux['dec'] = pars['delta']*deg  # Declination
    aux['l'] = pars['l']*deg  # Galactic longitude
    aux['b'] = pars['b']*deg  # Galactic latitude
    aux['ra_g'] = 192.85948*deg  # North Galactic pole
    aux['dec_g'] = 27.12825*deg  # North Galactic pole
    aux['vN'] = pars['vN']  # Projected velocity of Earth in North direction
    aux['vE'] = pars['vE']  # Projected velocity of Earth in East direction
    aux['mu_min'] = pars['mu_min']  # Minimal mu_rel (mas/yr)
    aux['mu_max'] = pars['mu_max']  # Maximal mu_rel (mas/yr)
    aux['mu_E_source'] = pars['mu_E_source']  # Gaia PM of the source
    aux['mu_N_source'] = pars['mu_N_source']  # Gaia PM of the source
    aux['mu_E_err_source'] = pars['mu_E_err_source']  # Gaia PM of the source
    aux['mu_N_err_source'] = pars['mu_N_err_source']  # Gaia PM of the source
    aux['pm_corr_source'] = pars['pm_corr_source']  # Gaia PM of the source
    aux['V_Sun'] = 232.2  # Velocity of the Sun relative to the GC
    aux['W_Sun'] = 7.3  # Velocity of the Sun relative to the GC
    aux['R0'] = 8.0  # Galactic Center distance
    aux['Ds'] = 8.0  # Source distance
    aux['massfunction'] = IMF(0.3, 1.3, 2.3, 0.01, 0.08, 0.5, 150.0)  # IMF

    # These velocities are taken from Batista et al. (2011)
    aux['v_y_disk'] = 220.0
    aux['v_z_disk'] = 0.0
    aux['sigma_y_disk'] = 30.0
    aux['sigma_z_disk'] = 20.0
    aux['sigma_y_blg'] = 100.0
    aux['sigma_z_blg'] = 100.0

    # Angle between Galactic and Equatorial coordinates
    C1 = np.sin(aux['dec_g'])*np.cos(aux['dec'])
    C1 += -np.cos(aux['dec_g'])*np.sin(aux['dec'])*np.cos(aux['ra']-aux['ra_g'])
    C2 = np.cos(aux['dec_g'])*np.sin(aux['ra']-aux['ra_g'])
    cosb = np.sqrt(C1**2 + C2**2)
    aux['C1'] = C1 / cosb
    aux['C2'] = C2 / cosb
    # angle is atan2(C2,C1) = 59.99 deg for PAR-02

    aux['cosb'] = np.cos(aux['b'])
    aux['sinb'] = np.sin(aux['b'])
    aux['cosl'] = np.cos(aux['l'])
    aux['sinl'] = np.sin(aux['l'])

    nwalkers = pars['nwalkers']
    nburnin = pars['nburnin']
    nsamples = pars['nsamples']
    ndim = 8

    # Running MCMC

    fname = pars['output'] + '.h5'
    backend = emcee.backends.HDFBackend(fname)
    backend.reset(nwalkers, ndim)

    sampler = emcee.EnsembleSampler(
        nwalkers,
        ndim,
        ln_prob,
        backend=backend,
        args=(hjd, flux, ferr, qn, qe, aux),
    )

    p0 = np.zeros((nwalkers, ndim))

    p0[:, 0] = pars['t0'] + np.random.randn(nwalkers)*pars['sigma_t0']
    p0[:, 1] = pars['tE'] + np.random.randn(nwalkers)*pars['sigma_tE']
    p0[:, 2] = pars['u0'] + np.random.randn(nwalkers)*pars['sigma_u0']
    p0[:, 3] = pars['piEN'] + np.random.randn(nwalkers)*pars['sigma_piEN']
    p0[:, 4] = pars['piEE'] + np.random.randn(nwalkers)*pars['sigma_piEE']
    p0[:, 5] = pars['mu_min'] \
        + np.random.rand(nwalkers)*(pars['mu_max']-pars['mu_min'])
    p0[:, 6] = pars['mu_E_source'] \
        + np.random.randn(nwalkers)*pars['mu_E_err_source']
    p0[:, 7] = pars['mu_N_source'] \
        + np.random.randn(nwalkers)*pars['mu_N_err_source']

    sampler.run_mcmc(p0, nburnin+nsamples)
