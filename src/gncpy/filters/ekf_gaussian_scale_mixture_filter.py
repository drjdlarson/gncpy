from gncpy.filters.kf_gaussian_scale_mixture_filter import KFGaussianScaleMixtureFilter
from gncpy.filters.extended_kalman_filter import ExtendedKalmanFilter


class EKFGaussianScaleMixtureFilter(KFGaussianScaleMixtureFilter):
    """Implementation of a EKF Gaussian Scale Mixture filter.

    This is provided for documentation purposes. It has the same functionality
    as the :class:`.GSMFilterBase` class.
    """

    def __init__(self, **kwargs):
        """Initialize the object."""
        super().__init__(**kwargs)

        self._coreFilter = ExtendedKalmanFilter()

    @property
    def cont_cov(self):
        """Wrapper for the core filter; see :attr:`.ExtendedKalmanFilter.cont_cov`."""
        return self._coreFilter.cont_cov

    @cont_cov.setter
    def cont_cov(self, val):
        self._coreFilter.cont_cov = val

    @property
    def integrator_type(self):
        """Wrapper for the core filter; see :attr:`.ExtendedKalmanFilter.integrator_type`."""
        return self._coreFilter.integrator_type

    @integrator_type.setter
    def integrator_type(self, val):
        self._coreFilter.integrator_type = val

    @property
    def integrator_params(self):
        """Wrapper for the core filter; see :attr:`.ExtendedKalmanFilter.integrator_params`."""
        return self._coreFilter.integrator_params

    @integrator_params.setter
    def integrator_params(self, val):
        self._coreFilter.integrator_params = val

    def set_state_model(self, **kwargs):
        """Wrapper for the core filter; see :meth:`.ExtendedKalmanFilter.set_state_model` for details."""
        super().set_state_model(**kwargs)

    def set_measurement_model(self, **kwargs):
        """Wrapper for the core filter; see :meth:`.ExtendedKalmanFilter.set_measurement_model` for details."""
        super().set_measurement_model(**kwargs)
