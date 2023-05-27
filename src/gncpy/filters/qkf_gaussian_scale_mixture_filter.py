from gncpy.filters.gsm_filter_base import GSMFilterBase
from gncpy.filters.quadrature_kalman_filter import QuadratureKalmanFilter


class QKFGaussianScaleMixtureFilter(GSMFilterBase):
    """Implementation of a QKF Gaussian Scale Mixture filter.

    Notes
    -----
    This is based on the derivation in
    :cite:`VilaValls2012_NonlinearBayesianFilteringintheGaussianScaleMixtureContext`
    but uses a QKF as the core filter instead of an SQKF. It should functionally
    be the same and provide the same results, however this should have better
    runtime performance at the cost of potential numerical problems with the
    covariance matrix.
    """

    def __init__(self, **kwargs):
        """Initialize the object."""
        super().__init__(**kwargs)

        self._coreFilter = QuadratureKalmanFilter()

    @property
    def points_per_axis(self):
        """Wrapper for the  number of quadrature points per axis."""
        return self._coreFilter.quadPoints.points_per_axis

    @points_per_axis.setter
    def points_per_axis(self, val):
        self._coreFilter.quadPoints.points_per_axis = val

    def set_state_model(self, **kwargs):
        """Wrapper for the core filter; see :meth:`.QuadratureKalmanFilter.set_state_model` for details."""
        super().set_state_model(**kwargs)

    def set_measurement_model(self, **kwargs):
        """Wrapper for the core filter; see :meth:`.QuadratureKalmanFilter.set_measurement_model` for details."""
        super().set_measurement_model(**kwargs)

