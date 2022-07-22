"""Implements functions for useful coordinate transforms."""
import numpy as np

import gncpy.wgs84 as wgs84


def ecef_to_LLA(pos):
    """Convert an ECEF position to LLA coordinates.

    Parameters
    ----------
    pos : 3 x 1 numpy array
        Position in meters in the ECEF frame

    Returns
    -------
    3 x 1 numpy array
        Latitude (rad), Longitude (rad), Altitude (m)
    """
    xyz = pos.copy().squeeze()
    lon = np.arctan2(xyz[1], xyz[0])

    p = np.sqrt(xyz[0] ** 2 + xyz[1] ** 2)
    E = np.sqrt(wgs84.EQ_RAD ** 2 - wgs84.POL_RAD ** 2)
    F = 54 * (wgs84.POL_RAD * xyz[2]) ** 2
    G = (
        p ** 2
        + (1 - wgs84.ECCENTRICITY ** 2) * (xyz[2] ** 2)
        - (wgs84.ECCENTRICITY * E) ** 2
    )
    c = wgs84.ECCENTRICITY ** 4 * F * p ** 2 / G ** 3
    s = (1 + c + np.sqrt(c ** 2 + 2 * c)) ** (1 / 3)
    P = (F / (3 * G ** 2)) / (s + 1 / s + 1) ** 2
    Q = np.sqrt(1 + 2 * wgs84.ECCENTRICITY ** 4 * P)
    k1 = -P * wgs84.ECCENTRICITY ** 2 * p / (1 + Q)
    k2 = 0.5 * wgs84.EQ_RAD ** 2 * (1 + 1 / Q)
    k3 = -P * (1 - wgs84.ECCENTRICITY ** 2) * xyz[2] ** 2 / (Q * (1 + Q))
    k4 = -0.5 * P * p ** 2
    k5 = p - wgs84.ECCENTRICITY ** 2 * (k1 + np.sqrt(k2 + k3 + k4))
    U = np.sqrt(k5 ** 2 + xyz[2] ** 2)
    V = np.sqrt(k5 ** 2 + (1 - wgs84.ECCENTRICITY ** 2) * xyz[2] ** 2)
    alt = U * (1 - wgs84.POL_RAD ** 2 / (wgs84.EQ_RAD * V))

    z0 = wgs84.POL_RAD ** 2 * xyz[2] / (wgs84.EQ_RAD * V)
    ep = wgs84.EQ_RAD / wgs84.POL_RAD * wgs84.ECCENTRICITY
    lat = np.arctan((xyz[2] + z0 * ep ** 2) / p)

    return np.array([lat, lon, alt]).reshape((3, 1))


def lla_to_ECEF(lat, lon, alt):
    """Convert LLA coordinates to ECEF coordinates.

    Parameters
    ----------
    lat : float
        Latitude in radians
    lon : float
        Longitude in radians
    alt : float
        Altitude in meters

    Returns
    -------
    3 x 1 numpy array
        ECEF position in meters
    """
    re = wgs84.calc_ew_rad(lat)
    c_lat = np.cos(lat)
    s_lat = np.sin(lat)
    c_lon = np.cos(lon)
    s_lon = np.sin(lon)

    x = (re + alt) * c_lat * c_lon
    y = (re + alt) * c_lat * s_lon
    z = ((1 - wgs84.ECCENTRICITY ** 2) * re + alt) * s_lat
    return np.array([x, y, z]).reshape((3, 1))


def lla_to_NED(ref_lat, ref_lon, ref_alt, pos_lat, pos_lon, pos_alt):
    """Convert LLA to NED coordinates.

    Parameters
    ----------
    ref_lat : float
        Reference latitude (rad)
    ref_lon : float
        Reference longitude (rad)
    ref_alt : float
        Reference altitude (m)
    pos_lat : float
        Latitude (rad)
    pos_lon : float
        Longitude (rad)
    pos_alt : float
        Altitude (m)

    Returns
    -------
    3 x 1 numpy array
        Position in NED coordinates
    """
    c_lat = np.cos(ref_lat)
    s_lat = np.sin(ref_lat)
    c_lon = np.cos(ref_lon)
    s_lon = np.sin(ref_lon)
    R = np.array(
        [
            [-s_lat * c_lon, -s_lon, -c_lat * c_lon],
            [-s_lat * s_lon, c_lon, -c_lat * s_lon],
            [c_lat, 0, -s_lat],
        ]
    )
    ref_E = lla_to_ECEF(ref_lat, ref_lon, ref_alt)
    pos_E = lla_to_ECEF(pos_lat, pos_lon, pos_alt)
    return R.T @ (pos_E - ref_E)


def ecef_to_NED(ref_xyz, pos_xyz):
    """Convert an ECEF position to the NED frame.

    Parameters
    ----------
    ref_xyz : 3 x 1 numpy array
        Reference position (m) in the ECEF frame
    pos_xyz : 3 x 1 numpy array
        Position (m) in the ECEF frame

    Returns
    -------
    3 x 1 numpy array
        Position (m) in the NED frame
    """
    ref_LLA = ecef_to_LLA(ref_xyz).squeeze()
    c_lat = np.cos(ref_LLA[0])
    s_lat = np.sin(ref_LLA[0])
    c_lon = np.cos(ref_LLA[1])
    s_lon = np.sin(ref_LLA[1])
    R = np.array(
        [
            [-s_lat * c_lon, -s_lon, -c_lat * c_lon],
            [-s_lat * s_lon, c_lon, -c_lat * s_lon],
            [c_lat, 0, -s_lat],
        ]
    )
    return R.T @ (pos_xyz - ref_xyz)


def ned_to_LLA(ned, ref_lat, ref_lon, ref_alt):
    """Convert NED to LLA.

    Parameters
    ----------
    ned : numpy array
        NED positon.
    ref_lat : flaot
        Reference latitude (radians).
    ref_lon : flaot
        Reference longitude (radians).
    ref_alt : float
        Reference altitude (meters).

    Returns
    -------
    numpy array
        Lat/lon/alt in rad/rad/m.
    """
    alt = ref_alt + -ned[2]
    if isinstance(alt, np.ndarray):
        alt = alt.item()
    f_fact = 2 * wgs84.FLATTENING - wgs84.FLATTENING ** 2
    s_lat2 = np.sin(ref_lat) ** 2
    Rn = wgs84.EQ_RAD / np.sqrt(1 - f_fact * s_lat2)
    Rm = Rn * ((1 - f_fact) / (1 - f_fact * s_lat2))
    dlat = ned[0] * np.arctan2(1, Rm)
    dlon = ned[1] * np.arctan2(1, Rn * np.cos(ref_lat))
    lat = ref_lat + dlat
    lon = ref_lon + dlon
    if isinstance(lat, np.ndarray):
        lat = lat.item()
    if isinstance(lon, np.ndarray):
        lon = lon.item()
    return np.array([lat, lon, alt]).reshape((3, 1))
