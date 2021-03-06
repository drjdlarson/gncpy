import abc
import io
import numpy as np

import gncpy.wgs84 as wgs84
from gncpy.orbital_mechanics import ecc_anom_from_mean, true_anom_from_ecc, \
    correct_lon_ascend, ecef_from_orbit
from gncpy.coordinate_transforms import ecef_to_NED


"""
-------------------------------------------------------------------------------
------------------------------ Inertial Sensors -------------------------------
-------------------------------------------------------------------------------
"""


class InertialBase:
    """ Defines basic functionality for inertial sensors.

    This implements a basic measurement function and calculates the associated
    bias, scale factor, and misalignment terms needed by an inertial sensor.

    Attributes:
        misalign_sig (list): Sigma values for the misalignment terms, 1 per
            axis, assumes 3 axis
        scale_factor_sig (list): Sigma values for the scale factor terms,
            1 per axis, assumes 3 axis
        corr_times (list): Correlation times for the bias, 1 per axis,
            assumes 3 axis
        gm_vars (list): Variance of the Gauss-Markov processes for each bias,
            1 per axis, assumes 3 axis
        sample_time (float): Sampling time of the Gauss-Markov process
        init_bias_var (list): Variance of the initial bias value
        wn_var (list): Variance of the white noise, 1 per axis, assumes 3 axis
    """

    def __init__(self, **kwargs):
        self.misalign_sig = kwargs.get('misalign_sig', [0, 0, 0])
        self.scale_factor_sig = kwargs.get('scale_factor_sig', [0, 0, 0])

        self.corr_times = kwargs.get('corr_times', [1, 1, 1])
        self.gm_vars = kwargs.get('gm_vars', [0, 0, 0])
        self.sample_time = kwargs.get('sample_time', 1)
        self.init_bias_var = kwargs.get('init_bias_var', [0, 0, 0])

        self.wn_var = kwargs.get('wn_var', [0, 0, 0])

        self._sf = np.array([[]])
        self._last_bias = np.array([[]])
        self._gm_drive_sig = 0

    def measure(self, true):
        """Returns measurements based on the true values.

        Applies scale factor/misalignments, adds bias, and adds noise to true
        values.

        Args:
            true (N x 1 numpy array): array of true sensor values, N <= 3

        Returns:
            (N x 1 numpy array): array of measured values, N <= 3
        """

        noise = np.sqrt(np.array(self.wn_var)) * np.random.randn(3)
        sf = self.calculate_scale_factor()
        return sf @ true + self.calculate_bias() + noise.reshape((3, 1))

    def calculate_bias(self):
        """ Calculates the bias for each axis.

        This assumes a Gauss-Markov model for the bias terms, see
        :cite:`Quinchia2013_AComparisonbetweenDifferentErrorModelingofMEMSAppliedtoGPSINSIntegratedSystems`
        """
        if self._last_bias.size == 0:
            self._last_bias = (np.array(self.init_bias_var)
                               * np.random.randn(3))
            self._last_bias.reshape((3, 1))
            beta = 1 / np.array(self.corr_times)
            var = np.array(self.gm_vars)
            self._gm_drive_sig = np.sqrt(var * (1 - np.exp(-2
                                                           * self.sample_time
                                                           * beta)))
            self._gm_drive_sig = self._gm_drive_sig.reshape((3, 1))

        beta = (1 / np.array(self.corr_times)).reshape((3, 1))
        w = self._gm_drive_sig * np.random.randn(3, 1)
        bias = (1 - self.sample_time * beta) * self._last_bias + w
        self._last_bias = bias
        return bias

    def calculate_scale_factor(self):
        """ Calculates the scale factors and misalignment matrix.
        """
        if self._sf.size == 0:
            sf = np.array(self.scale_factor_sig) * np.random.randn(1, 3)
            ma = np.zeros((3, 3))
            for row in range(0, 3):
                for col in range(0, 3):
                    if row == col:
                        continue
                    ma[row, col] = self.misalign_sig[row] * np.random.randn()
            self._sf = np.eye(3) + np.diag(sf) + ma

        return self._sf


"""
-------------------------------------------------------------------------------
------------------------------- GNSS Sensors ----------------------------------
-------------------------------------------------------------------------------
"""


class BaseSV(metaclass=abc.ABCMeta):
    """ Base class for GNSS Space Vehicles.

    Constellations should define their own SV class that properly accounts for
    their propagation and error models. Inherited classes must define
    :py:meth:`gncpy.sensors.BaseSV.propagate`

    Attributes:
        true_pos_ECEF (3 x 1 numpy array): Position of SV in ECEF coordinates
            from orbital mechanics
        real_pos_ECEF (3 x 1 numpy array): True position corrupted by errors
    """
    def __init__(self, **kwargs):
        self.true_pos_ECEF = np.array([[]])
        self.real_pos_ECEF = np.array([[]])

    @abc.abstractmethod
    def propagate(self, time, **kwargs):
        """ Calculates SV position at a given time.

        This must be defined by child classes.

        Args:
            time (float): Time to find position
        """
        pass


class BaseConstellation:
    """ Base class for GNSS constellations.

    Collects common functionality between different constellations and provides
    a common interface.

    Attributes:
        sats (dict): Dictionary of :py:class:`gncpy.sensors.BaseSV` objects
    """
    def __init__(self, **kwargs):
        self.sats = {}

    def propagate(self, time, **kwargs):
        """ Propagate all SV's to the given time.

        Attributes:
            time (float): Time to find position
            **kwargs : passed through to :py:meth:`gncpy.sensors.BaseSV.propagate`
        """
        for k, v in self.sats.items():
            self.sats[k].propagate(time, **kwargs)


class BaseGNSSReceiver(metaclass=abc.ABCMeta):
    """ Defines the interface for common GNSS Receiver functions.

    Attributes:
        max_channels (int): Maximum number of signals to track
        mask_angle (float): Masking angle in degrees
    """
    def __init__(self, **kwargs):
        self.max_channels = kwargs.get('max_channels', 12)
        self.mask_angle = kwargs.get('mask_angle', 0)

    @abc.abstractmethod
    def measure_PR(self, true_pos, const, **kwargs):
        pass


class GPSSat(BaseSV):
    """ Custom SV for GPS Constellation.

    Attributes:
        health (str): satellite health
        ecc (float): eccentricity
        toe (float): time of applicability/ephemeris in seconds
        inc (float): Orbital inclination in radians
        ascen_rate (float): Rate of right ascension in rad/s
        sqrt_a (float): Square root of semi-major axis in m^1/2
        ascen (float): Right ascension at week in radians
        peri (float): Argument of perigee in radians
        mean_anom (float): Mean anomaly in radians
        af0 (float): Zeroth order clock correction in seconds
        af1 (float): First order cloc correction in sec/sec
        week (int): Week number
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.health = ''
        self.ecc = 0
        self.toe = 0
        self.inc = 0
        self.ascen_rate = 0
        self.sqrt_a = 0
        self.ascen = 0
        self.peri = 0
        self.mean_anom = 0
        self.af0 = 0
        self.af1 = 0
        self.week = 0

    def propagate(self, time, **kwargs):
        """ Calculates the GPS SV's position at the given time

        Args:
            time (float): Time of week to find position
        """
        semimajor = self.sqrt_a**2
        mean_motion = np.sqrt(wgs84.MU / semimajor**3)

        tk = time - self.toe
        if tk > 302400:
            tk = tk - 604800
        elif tk < -302400:
            tk = tk + 604800

        mean_anom = self.mean_anom + mean_motion * tk
        ecc_anom = ecc_anom_from_mean(mean_anom, self.ecc, **kwargs)
        true_anom = true_anom_from_ecc(ecc_anom, self.ecc)

        arg_lat = true_anom + self.peri
        cor_ascen = correct_lon_ascend(self.ascen, self.ascen_rate, tk,
                                       self.toe)
        rad = semimajor * (1 - self.ecc * np.cos(ecc_anom))

        self.true_pos_ECEF = ecef_from_orbit(arg_lat, rad, cor_ascen, self.inc)
        self.real_pos_ECEF = self.true_pos_ECEF.copy()


class GPSConstellation(BaseConstellation):
    """ Handles the GPS constellation.

    This maintains the satellite list, file parsing operations, and propagation
    functions for the GPS satellite constellation.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def parse_almanac(self, alm_f):
        """ Parses an almanac file to setup constellation parameters.

        Args:
            alm_f (str): file path and name of almanac file
        """
        cur_prn = ''
        with io.open(alm_f, 'r') as fin:
            for line in fin:
                line = line.lower()

                if "almanac" in line:
                    pass

                elif "id" in line:
                    string = line.split(":")[1]
                    cur_prn = str(int(string.strip()))
                    self.sats[cur_prn] = GPSSat()

                elif cur_prn and ":" in line:
                    val = line.split(":")[1].strip()
                    if "health" in line:
                        self.sats[cur_prn].health = val
                    elif "eccentricity" in line:
                        self.sats[cur_prn].ecc = float(val)
                    elif "applicability" in line:
                        self.sats[cur_prn].toe = float(val)
                    elif "inclination" in line:
                        self.sats[cur_prn].inc = float(val)
                    elif "rate of right" in line:
                        self.sats[cur_prn].ascen_rate = float(val)
                    elif "sqrt" in line:
                        self.sats[cur_prn].sqrt_a = float(val)
                    elif "right ascen" in line:
                        self.sats[cur_prn].ascen = float(val)
                    elif "argument" in line:
                        self.sats[cur_prn].peri = float(val)
                    elif "mean anom" in line:
                        self.sats[cur_prn].mean_anom = float(val)
                    elif "af0" in line:
                        self.sats[cur_prn].af0 = float(val)
                    elif "af1" in line:
                        self.sats[cur_prn].af1 = float(val)
                    elif "week" in line:
                        self.sats[cur_prn].week = int(val)


class GPSReceiver(BaseGNSSReceiver):
    """ Emulates a GPS receiver.

    Manages the measurement functions for a GPS reciever, and models receiver
    and signal noise.

    Attributes:
        tracked_SVs (list): The PRNs of all SVs currently tracked
    Todo:
        Add PR error models
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tracked_SVs = []

    def measure_PR(self, true_pos, const, **kwargs):
        """ Measures the pseudorange to each satellite in the constellation

        Args:
            true_pos (3 x 1 numpy array): True receiver position in ECEF
            const (:py:class:`gncpy.sensors.GPSConstellation`): Constellation
                to use, propagated to current time
        Returns:
            (dict): Keys are PRN's and values are pseudoranges in meters

        Todo:
            add error terms
        """
        new_SVs = {}
        for prn, sv in const.sats.items():
            diff = sv.real_pos_ECEF - true_pos
            los_NED = ecef_to_NED(true_pos, sv.real_pos_ECEF)
            los_NED = los_NED / np.sqrt(los_NED.T @ los_NED)
            elev = -np.arctan2(los_NED[2] / np.sqrt(los_NED[0]**2
                                                    + los_NED[1]**2))
            if elev >= self.mask_angle:
                pr = np.sqrt(diff.T @ diff)
                new_SVs[prn] = pr

        tracked = {}
        if len(new_SVs) >= self.max_channels:
            for prn in new_SVs.keys():
                if prn in self.tracked_SVs:
                    tracked[prn] = new_SVs[prn]
                    del new_SVs[prn]

        if len(tracked) < self.max_channels and len(new_SVs) > 0:
            for prn in new_SVs.keys():
                if prn not in tracked:
                    tracked[prn] = new_SVs[prn]
                    if len(tracked) == self.max_channels:
                        break

        self.tracked_SVs = tracked.keys()
        return tracked
