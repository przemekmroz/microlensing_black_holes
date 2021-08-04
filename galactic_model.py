import numpy as np


def rho_bulge(x, y, z):

    """
    Calculates density of bulge stars (Table 2, Batista et al. 2011)

    Parameters:
    -----------
    x : x-coordinate (dist*sin(l)*cos(b))
    y : y-coordinate (R_0 - dist*cos(l)*cos(b))
    z : z-coordinate (dist*sin(b))

    THETA = 20 deg is the bar inclination

    Returns:
    --------
    rho : density of bulge stars (M_Solar / pc^3)

    For the bulge cut-off see:
    https://www.aanda.org/articles/aa/pdf/2003/38/aa3188.pdf
    http://articles.adsabs.harvard.edu/pdf/1995ApJ...445..716D (model G2)
    """

    THETA = 20.0*np.pi/180.0

    xp = x*np.sin(THETA) + y*np.cos(THETA)
    yp = x*np.cos(THETA) - y*np.sin(THETA)

    r = np.sqrt(x**2 + y**2)

    rs_4 = ((xp/1.58)**2 + (yp/0.62)**2)**2 + (z/0.43)**4
    rs_2 = np.sqrt(rs_4)

    rho = 1.23 * np.exp(-0.5*rs_2)
    if r > 2.4:
        rho *= np.exp(-0.5*((r-2.4)/0.5)**2)

    return rho


def rho_disk(x, y, z):

    """
    Calculates density of disk stars (Table 2, Batista et al. 2011)

    Parameters:
    -----------
    x : x-coordinate (dist*sin(l)*cos(b))
    y : y-coordinate (R_0 - dist*cos(l)*cos(b))
    z : z-coordinate (dist*sin(b))

    Returns:
    --------
    rho : density of disk stars (M_Solar / pc^3)
    """

    beta = 0.381
    R = np.sqrt(x**2 + y**2)

    rho = (1.0-beta)*np.exp(-abs(z)/0.156) + beta*np.exp(-abs(z)/0.439)
    rho = rho * 1.07 * np.exp(-R/2.75)

    if R < 1.0:
        rho = 0.0

    return rho
