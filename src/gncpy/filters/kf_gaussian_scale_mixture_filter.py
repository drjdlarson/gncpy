from gncpy.filters.gsm_filter_base import GSMFilterBase
from gncpy.filters.kalman_filter import KalmanFilter


class KFGaussianScaleMixtureFilter(GSMFilterBase):
    """Implementation of a KF Gaussian Scale Mixture filter.

    This is provided for documentation mostly for purposes. It exposes some of
    the core KF variables.
    """

    def __init__(self, **kwargs):
        """Initialize the object."""
        super().__init__(**kwargs)

        self._coreFilter = KalmanFilter()

    @property
    def dt(self):
        """Wrapper for the core filter; see :attr:`.KalmanFilter.dt`."""
        return self._coreFilter.dt

    @dt.setter
    def dt(self, val):
        self._coreFilter.dt = val

    def set_state_model(self, **kwargs):
        """Wrapper for the core filter; see :meth:`.KalmanFilter.set_state_model` for details."""
        super().set_state_model(**kwargs)

    def set_measurement_model(self, **kwargs):
        """Wrapper for the core filter; see :meth:`.KalmanFilter.set_measurement_model` for details."""
        super().set_measurement_model(**kwargs)
