import numpy as np
from warnings import warn

import gncpy.wgs84 as wgs84


def ecc_anom_from_mean(mean_anom, ecc, **kwargs):
    """ Calculate the eccentric anomaly from the mean anomaly.

    Args:
        mean_anom (float): Mean anomaly in radians
        ecc (float): eccentricity
    Keyword Args:
        tol (float): Minimum tolerance for convergence
        max_iter (int): Maximum number of iterations to run
    Returns:
        (float): Eccentric anomaly in radians
    """
    tol = kwargs.get('tol', 1 * 10**-7)
    max_iter = kwargs.get('max_iter', 1 * 10**3)

    delta = 1 * 10**3
    ecc_anom = mean_anom
    ctr = 0
    while np.abs(delta) > tol:
        delta = (ecc_anom - ecc * np.sin(ecc_anom) - mean_anom) \
            / (1 - ecc * np.cos(ecc_anom))
        ecc_anom = ecc_anom - delta

        ctr = ctr + 1
        if ctr >= max_iter:
            warn('Maximum iterations reached when finding eccentric anomaly.',
                 RuntimeWarning)
            return ecc_anom
    return ecc_anom


def true_anom_from_ecc(ecc_anom, ecc):
    """ Calculate the true anomaly from the eccentric anomaly

    Args:
        ecc_anom (float): eccentric anomaly in radians
        ecc (float): eccentricity
    Returns:
        (float): true anomaly in radians
    """
    den = 1 / (1 - ecc * np.cos(ecc_anom))
    cos_true = (np.cos(ecc_anom) - ecc) * den
    sin_true = np.sqrt(1 - ecc**2) * np.sin(ecc_anom) * den
    return np.arctan2(sin_true, cos_true)


def correct_lon_ascend(lon_ascend, lon_rate, tk, toe):
    """ Correct the longitude of the ascending node

    Args:
        lon_ascend (float): Original longitude of the ascending node in radians
        lon_rate (float): Rate of change of the longitude of the ascending node
        tk (float): Current time of the week
        toe (float): Time of ephemeris
    Returns:
        (float): corrected longitude of the ascending node in radians
    """
    return lon_ascend + (lon_rate - wgs84.EARTH_ROT_RATE) * tk \
        - wgs84.EARTH_ROT_RATE * toe


def ecef_from_orbit(arg_lat, rad, lon_ascend, inc):
    """ Calculates the ECEF position from orbital parameters.

    Args:
        arg_lat (float): Argument of latitude in radians
        rad (float): Orbital radius in meters
        lon_ascend (float): Longitude of the ascending node in radians
        inc (float): Orbital inclination angle in radians
    Returns:
        (3 x 1 numpy array): ECEF position in meters
    """
    xp = rad * np.cos(arg_lat)
    yp = rad * np.sin(arg_lat)

    c_lon = np.cos(lon_ascend)
    s_lon = np.sin(lon_ascend)
    c_inc = np.cos(inc)

    x = xp * c_lon - yp * c_inc * s_lon
    y = xp * s_lon + yp * c_inc * c_lon
    z = yp * np.sin(inc)

    return np.array([x, y, z]).reshape((3, 1))
