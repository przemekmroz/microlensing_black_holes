import argparse

from parsubs import pm_ra_dec_to_l_b_w_errorbars, radec_to_gal

"""
Transform proper motion and its uncertainty to Galactic coordinates
(including non-zero correlation between pm_ra and pm_dec)

P. Mroz @ Caltech, 18 Feb 2021

Example: OGLE3-ULENS-PAR-02 (Wyrzykowski et al. 2016)
python3 transform_proper_motion.py 269.346417 -28.775556 -4.42 -6.50 0.11 0.08 0.40
l = 1.453657
b = -2.133106
pm_l = -7.84 +/- 0.10 mas/yr
pm_b = 0.58 +/- 0.09 mas/yr
correlation(pm_l,pm_b) = -0.46
"""

parser = argparse.ArgumentParser()
parser.add_argument('ra', type=float)
parser.add_argument('dec', type=float)
parser.add_argument('pmra', type=float)
parser.add_argument('pmdec', type=float)
parser.add_argument('pmra_err', type=float)
parser.add_argument('pmdec_err', type=float)
parser.add_argument('pmra_pmdec_corr', type=float)
args = parser.parse_args()

l, b = radec_to_gal(args.ra, args.dec)

pm_l, pm_b, pm_l_err, pm_b_err, corr = pm_ra_dec_to_l_b_w_errorbars(
    args.ra, args.dec, args.pmra, args.pmdec, args.pmra_err, args.pmdec_err,
    args.pmra_pmdec_corr)

print('l = {:.6f}'.format(l))
print('b = {:.6f}'.format(b))
print('pm_l = {:.2f} +/- {:.2f} mas/yr'.format(pm_l, pm_l_err))
print('pm_b = {:.2f} +/- {:.2f} mas/yr'.format(pm_b, pm_b_err))
print('correlation(pm_l,pm_b) = {:.2f}'.format(corr))
