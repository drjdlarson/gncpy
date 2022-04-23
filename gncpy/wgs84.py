"""Constants and utility functions relating to the WGS-84 model."""
import numpy as np
from warnings import warn


MU = 3.986005 * 10**14  # m^3/s^2
SPEED_OF_LIGHT = 2.99792458 * 10**8  # m/s
EARTH_ROT_RATE = 7.2921151467 * 10**-5  # rad/s
PI = 3.1415926535898
FLATTENING = 1 / 298.257223563
ECCENTRICITY = np.sqrt(FLATTENING * (2 - FLATTENING))
EQ_RAD = 6378137  # m
POL_RAD = EQ_RAD * (1 - FLATTENING)
GRAVITY = 9.7803253359


_egm_lut = np.array([])


def calc_earth_rate(lat):
    """ Calculate the earth rate

    Args:
        lat (float): Latitude in radians
    Returns:
        (3 x 1 numpy array): Earth rate in radians
    """
    return EARTH_ROT_RATE * np.array([[np.cos(lat)], [0], [-np.sin(lat)]])


def calc_transport_rate(v_N, alt, lat):
    """ Calculates the transport rate

    Args:
        v_N (3 numpy array): Velocity in the NED frame in m/s
        alt (float): Altitude in meters
        lat (float): Latitude in radians
    Returns:
        (3 x 1 numpy array): transport rate in rad/s
    """
    rn = calc_ns_rad(lat)
    re = calc_ew_rad(lat)
    return np.array([v_N[1] / (re + alt),
                     -v_N[0] / (rn + alt),
                     -v_N[1] * np.tan(lat) / (re + alt)])


def calc_ns_rad(lat):
    """ Calculates the North/South radius

    Args:
        lat (float) latitude in radians
    Returns:
        (float): North/South radius in meters
    """
    return EQ_RAD * (1 - ECCENTRICITY**2) / (1 - ECCENTRICITY**2
                                             * np.sin(lat)**2)**1.5


def calc_ew_rad(lat):
    """ Calculates the East/West radius

    Args:
        lat (float) latitude in radians
    Returns:
        (float): East/West radius in meters
    """
    return EQ_RAD / np.sqrt(1 - ECCENTRICITY**2 * np.sin(lat)**2)


def calc_gravity(lat, alt):
    """ Calculates gravity vector in NED coordinates

    Args:
        lat (float): Latitude in radians
        alt (float): Altitude in meters
    Returns:
        (3 x 1 numpy array): Gravity vector in NED frame
    """
    frac = alt / EQ_RAD
    g0 = GRAVITY / np.sqrt(1 - FLATTENING * (2 - FLATTENING)
                           * np.sin(lat)**2) * (1 + 0.0019311853
                                                * np.sin(lat)**2)
    ch = 1 - 2 * (1 + FLATTENING + (EQ_RAD**3 * (1 - FLATTENING)
                  * EARTH_ROT_RATE**2) / MU) * frac + 3 * frac**2
    return np.array([[0], [0], [ch * g0]])


def init_egm_lookup_table(bin_file):
    global _egm_lut
    warn('Lookup table has not been implemented yet')
    _egm_lut = np.array([])


def convert_wgs_to_msl(lat, lon, alt):
    global _egm_lut
    if _egm_lut.size == 0:
        warn('EGM table was not loaded. Can not convert to height above geoid')
        return alt
    else:
        raise NotImplemented
        # row, col = (None, None)
        # return alt - _egm_lut[row, col]
