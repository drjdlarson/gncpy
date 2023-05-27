from gncpy.filters.qkf_gaussian_scale_mixture_filter import QKFGaussianScaleMixtureFilter
from gncpy.filters.square_root_qkf import SquareRootQKF


class SQKFGaussianScaleMixtureFilter(QKFGaussianScaleMixtureFilter):
    """Implementation of a SQKF Gaussian Scale Mixture filter.

    This is provided for documentation purposes. It functions the same as
    the :class:`.QKFGaussianScaleMixtureFilter` class.

    Notes
    -----
    This is based on the derivation in
    :cite:`VilaValls2012_NonlinearBayesianFilteringintheGaussianScaleMixtureContext`.
    """

    def __init__(self, **kwargs):
        """Initialize the object."""
        super().__init__(**kwargs)

        self._coreFilter = SquareRootQKF()

    def set_state_model(self, **kwargs):
        """Wrapper for the core filter; see :meth:`.SquareRootQKF.set_state_model` for details."""
        super().set_state_model(**kwargs)

    def set_measurement_model(self, **kwargs):
        """Wrapper for the core filter; see :meth:`.SquareRootQKF.set_measurement_model` for details."""
        super().set_measurement_model(**kwargs)
