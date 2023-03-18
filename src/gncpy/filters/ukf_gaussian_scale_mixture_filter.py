from gncpy.filters.gsm_filter_base import GSMFilterBase
from gncpy.filters.unscented_kalman_filter import UnscentedKalmanFilter


class UKFGaussianScaleMixtureFilter(GSMFilterBase):
    """Implementation of a UKF Gaussian Scale Mixture filter.

    Notes
    -----
    This is based on the SKQF GSM derivation in
    :cite:`VilaValls2012_NonlinearBayesianFilteringintheGaussianScaleMixtureContext`
    but utilizes a UKF as the core filter instead.
    """

    def __init__(self, sigmaPoints=None, **kwargs):
        """Initialize the object."""
        super().__init__(**kwargs)

        self._coreFilter = UnscentedKalmanFilter(sigmaPoints=sigmaPoints)

    def set_state_model(self, **kwargs):
        """Wrapper for the core filter; see :meth:`.UnscentedKalmanFilter.set_state_model` for details."""
        super().set_state_model(**kwargs)

    def set_measurement_model(self, **kwargs):
        """Wrapper for the core filter; see :meth:`.UnscentedKalmanFilter.set_measurement_model` for details."""
        super().set_measurement_model(**kwargs)

    def init_sigma_points(self, *args, **kwargs):
        """Wrapper for the core filter; see :meth:`.UnscentedKalmanFilter.init_sigma_points` for details."""
        self._coreFilter.init_sigma_points(*args, **kwargs)

    @property
    def dt(self):
        """Wrapper for the core filter: see :attr:`.UnscentedKalmanFilter.dt` for details."""
        return self._coreFilter.dt

    @dt.setter
    def dt(self, val):
        self._coreFilter.dt = val

    @property
    def alpha(self):
        """Wrapper for the core filter: see :attr:`.UnscentedKalmanFilter.alpha` for details."""
        return self._coreFilter.alpha

    @alpha.setter
    def alpha(self, val):
        self._coreFilter.alpha = val

    @property
    def beta(self):
        """Wrapper for the core filter: see :attr:`.UnscentedKalmanFilter.beta` for details."""
        return self._coreFilter.beta

    @beta.setter
    def beta(self, val):
        self._coreFilter.beta = val

    @property
    def kappa(self):
        """Wrapper for the core filter: see :attr:`.UnscentedKalmanFilter.kappa` for details."""
        return self._coreFilter.kappa

    @kappa.setter
    def kappa(self, val):
        self._coreFilter.kappa = val
