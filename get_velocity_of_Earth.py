import argparse

from parsubs import velocity_of_Earth_projected

"""
Get the velocity of the Earth (as seen from the Solar System barycenter)
at a given time projected at a given position in the sky (ra,dec).

P. Mroz @ Caltech, 18 Feb 2021

This code is based on Radek Poleski's implementation in MulensModel.

Example 1: MOA-2009-BLG-387 (Batista et al. 2011)
python3 get_velocity_of_Earth.py 5042.34 268.4616250 -33.9902778
v_Earth_perp_N = -3.59 km/s
v_Earth_perp_E = 22.94 km/s

Example 2: OGLE3-ULENS-PAR-02 (Wyrzykowski et al. 2016)
python3 get_velocity_of_Earth.py 2454018.6 269.346417 -28.775556
v_Earth_perp_N = -2.60 km/s
v_Earth_perp_E = -9.27 km/s
"""

parser = argparse.ArgumentParser()
parser.add_argument('time', type=float)
parser.add_argument('ra', type=float)
parser.add_argument('dec', type=float)
args = parser.parse_args()

if args.time < 2450000.0:
    args.time += 2450000.0

v_Earth_perp_N, v_Earth_perp_E = velocity_of_Earth_projected(
    args.time, args.ra, args.dec)

print('v_Earth_perp_N = {:.2f} km/s'.format(v_Earth_perp_N))
print('v_Earth_perp_E = {:.2f} km/s'.format(v_Earth_perp_E))
