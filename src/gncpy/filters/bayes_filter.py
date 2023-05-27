import abc


class BayesFilter(metaclass=abc.ABCMeta):
    """Generic base class for Bayesian Filters such as a Kalman Filter.

    This defines the required functions and provides their recommended function
    signature for inherited classes.

    Attributes
    ----------
    use_cholesky_inverse : bool
        Flag indicating if a cholesky decomposition should be performed before
        taking the inverse. This can improve numerical stability but may also
        increase runtime. The default is True.
    """

    def __init__(self, use_cholesky_inverse=True, **kwargs):
        self.use_cholesky_inverse = use_cholesky_inverse

        super().__init__()

    @abc.abstractmethod
    def predict(self, timestep, *args, **kwargs):
        """Generic method for the filters prediction step.

        This must be overridden in the inherited class. It is recommended to
        keep the same structure/order for the arguments to allow for
        standardized implementation of wrapper code.
        """
        pass

    @abc.abstractmethod
    def correct(self, timestep, meas, *args, **kwargs):
        """Generic method for the filters correction step.

        This must be overridden in the inherited class. It is recommended to
        keep the same structure/order for the arguments to allow for
        standardized implementation of wrapper code.
        """
        pass

    @abc.abstractmethod
    def set_state_model(self, **kwargs):
        """Generic method for tsetting the state model.

        This must be overridden in the inherited class. The signature for this
        is arbitrary.
        """
        pass

    @abc.abstractmethod
    def set_measurement_model(self, **kwargs):
        """Generic method for setting the measurement model.

        This must be overridden in the inherited class. The signature for this
        is arbitrary.
        """
        pass

    @abc.abstractmethod
    def save_filter_state(self):
        """Generic method for saving key filter variables.

        This must be overridden in the inherited class. It is recommended to keep
        the signature the same to allow for standardized implemenation of
        wrapper classes. This should return a single variable that can be passed
        to the loading function to setup a filter to the same internal state
        as the current instance when this function was called.
        """
        filt_state = {}
        filt_state["use_cholesky_inverse"] = self.use_cholesky_inverse

        return filt_state

    @abc.abstractmethod
    def load_filter_state(self, filt_state):
        """Generic method for saving key filter variables.

        This must be overridden in the inherited class. It is recommended to keep
        the signature the same to allow for standardized implemenation of
        wrapper classes. This initialize all internal variables saved by the
        filter save function such that a new instance would generate the same
        output as the original instance that called the save function.
        """
        self.use_cholesky_inverse = filt_state["use_cholesky_inverse"]