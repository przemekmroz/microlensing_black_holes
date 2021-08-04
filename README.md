# Measuring the mass function of isolated stellar remnants with gravitational microlensing

## Fit the microlensing model to the light curve using emcee

    python single_parallax_fit_geocentric.py FILENAME
    
where FILENAME is a YAML file containing all parameters (see example/BLG100.6.38608A.inp).

## Get the velocity of the Earth projected at a given position in the sky (ra, dec) at a given time

    python get_velocity_of_Earth.py time ra dec
    
## Transform proper motion and its uncertainty to Galactic coordinates

    python transform_proper_motion.py ra dec pm_ra pm_dec pm_ra_err pm_dec_err pmra_pmdec_corr
    python transform_proper_motion.py 269.346417 -28.775556 -4.42 -6.50 0.11 0.08 0.40

## Credit

If you are using these codes, please cite:
* Mróz, P. and Wyrzykowski, Ł. 2021, "Measuring the mass function of isolated stellar remnants with gravitational microlensing. I. Revisiting the OGLE-III dark lens candidates", arXiv:2107.13701
* Mróz, P. et al. 2021, "Measuring the mass function of isolated stellar remnants with gravitational microlensing. II. Analysis of the OGLE-III data", arXiv:2107.13697
