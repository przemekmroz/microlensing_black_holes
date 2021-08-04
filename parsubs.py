import numpy as np
import astropy.units as u

from astropy.time import Time
from astropy.coordinates import get_body_barycentric
from astropy.coordinates.builtin_frames.utils import get_jd12
from astropy import _erfa as erfa

"""
This a Python version of the parsubs code by Andy Gould.

Based on Radek's implementation in MulensModel
"""


def geta(_times, _alpha, _delta, _t0_par):

    if _times[0] < 2450000.0:
        offset = 2450000.0
    else:
        offset = 0.0

    times = Time(_times+offset, format='jd', scale='tdb')

    if _t0_par < 2450000.0:
        offset = 2450000.0
    else:
        offset = 0.0

    t_ref = Time(_t0_par+offset, format='jd', scale='tdb')

    # Calculating North and East versors
    deg = np.pi/180.0
    delta = _delta*deg
    alpha = _alpha*deg
    direction = np.array([np.cos(delta)*np.cos(alpha),
                          np.cos(delta)*np.sin(alpha),
                          np.sin(delta)])
    north = np.array([0., 0., 1.])
    east_projected = np.cross(north, direction)
    east_projected = east_projected/np.linalg.norm(east_projected)
    north_projected = np.cross(direction, east_projected)

    # Calculating velocity of Earth @ t0,par
    (jd1, jd2) = get_jd12(t_ref, 'tdb')
    (earth_pv_helio, earth_pv_bary) = erfa.epv00(jd1, jd2)
    velocity = np.asarray(earth_pv_bary[1])  # AU/days

    # Calculating position of Earth
    position = get_body_barycentric(body='earth', time=times)
    position_ref = get_body_barycentric(body='earth', time=t_ref)

    delta_s = (position_ref.xyz.T-position.xyz.T).to(u.au).value
    delta_s += np.outer(_times-_t0_par, velocity)

    # Projecting on the plane of the sky
    out_n = np.dot(delta_s, north_projected)
    out_e = np.dot(delta_s, east_projected)

    return out_n, out_e


def velocity_of_Earth(full_BJD):

    time = Time(full_BJD, format='jd', scale='tdb')
    (jd1, jd2) = get_jd12(time, 'tdb')
    (earth_pv_helio, earth_pv_bary) = erfa.epv00(jd1, jd2)
    factor = 1731.45683  # This scales AU/day to km/s.
    velocity = np.asarray(earth_pv_bary[1]) * factor

    return velocity


def velocity_of_Earth_projected(full_BJD, _alpha, _delta):

    # Calculating North and East versors

    deg = np.pi/180.0
    delta = _delta*deg
    alpha = _alpha*deg
    direction = np.array([np.cos(delta)*np.cos(alpha),
                          np.cos(delta)*np.sin(alpha),
                          np.sin(delta)])

    north = np.array([0., 0., 1.])
    east_projected = np.cross(north, direction)
    east_projected = east_projected/np.linalg.norm(east_projected)
    north_projected = np.cross(direction, east_projected)

    # Calculating velocity of Earth @ full_BJD

    velocity = velocity_of_Earth(full_BJD)

    # Calculate velocity vector projections

    v_Earth_perp_N = np.dot(velocity, north_projected)
    v_Earth_perp_E = np.dot(velocity, east_projected)

    return (v_Earth_perp_N, v_Earth_perp_E)


def pm_ra_dec_to_l_b(_ra, _dec, pmra, pmdec):
    """ Transformation of the equatorial proper motion to the galactic system
        using equations of Poleski (2013)
    """

    deg = np.pi/180.0
    ra_g = 192.85948*deg
    dec_g = 27.12825*deg
    ra = _ra*deg
    dec = _dec*deg

    C1 = np.sin(dec_g)*np.cos(dec)-np.cos(dec_g)*np.sin(dec)*np.cos(ra-ra_g)
    C2 = np.cos(dec_g)*np.sin(ra-ra_g)
    cosb = np.sqrt(C1**2+C2**2)
    C1 /= cosb
    C2 /= cosb
    pm_l = C1*pmra+C2*pmdec
    pm_b = -C2*pmra+C1*pmdec
    return pm_l, pm_b


def pm_ra_dec_to_l_b_w_errorbars(
        _ra, _dec, pmra, pmdec, pmra_err, pmdec_err, pmra_pmdec_corr):

    """ Transformation of the equatorial proper motion to the galactic system
        using equations of Poleski (2013)
    """

    deg = np.pi/180.0
    ra_g = 192.85948*deg
    dec_g = 27.12825*deg
    ra = _ra*deg
    dec = _dec*deg

    C1 = np.sin(dec_g)*np.cos(dec)-np.cos(dec_g)*np.sin(dec)*np.cos(ra-ra_g)
    C2 = np.cos(dec_g)*np.sin(ra-ra_g)
    cosb = np.sqrt(C1**2+C2**2)
    C1 /= cosb
    C2 /= cosb
    pm_l = C1*pmra+C2*pmdec
    pm_b = -C2*pmra+C1*pmdec

    # Transforming the uncertainties, see Luri et al. 2018
    # https://ui.adsabs.harvard.edu/abs/2018A%26A...616A...9L/abstract

    Jp = [[C1, -C2], [C2, C1]]
    J = [[C1, C2], [-C2, C1]]
    C = [[pmra_err**2, pmra_pmdec_corr*pmra_err*pmdec_err],
         [pmra_pmdec_corr*pmra_err*pmdec_err, pmdec_err**2]]
    C_lb = np.matmul(C, Jp)
    C_lb = np.matmul(J, C_lb)

    pm_l_err = np.sqrt(C_lb[0][0])
    pm_b_err = np.sqrt(C_lb[1][1])
    pm_l_pm_b_corr = C_lb[0][1]/pm_l_err/pm_b_err

    return pm_l, pm_b, pm_l_err, pm_b_err, pm_l_pm_b_corr


def radec_to_gal(_ra, _dec):

    """ Equatorial to Galactic coordinates, all angles are in degrees"""

    deg = np.pi/180.0
    ra_g = 192.85948*deg
    dec_g = 27.12825*deg
    l_NGP = 122.93192*deg
    ra = _ra*deg
    dec = _dec*deg

    sinb = np.cos(dec)*np.cos(dec_g)*np.cos(ra-ra_g) + np.sin(dec)*np.sin(dec_g)
    sinlcosb = np.cos(dec)*np.sin(ra-ra_g)
    coslcosb = np.sin(dec)*np.cos(dec_g) \
        - np.cos(dec)*np.sin(dec_g)*np.cos(ra-ra_g)
    b = np.arcsin(sinb)
    l = np.arctan2(sinlcosb, coslcosb)
    l = l_NGP - l
    if l < 0:
        l += 2.0*np.pi

    return l/deg, b/deg
