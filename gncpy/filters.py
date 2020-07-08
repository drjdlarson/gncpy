import numpy as np
import numpy.linalg as la
import abc


class BayesFilter(metaclass=abc.ABCMeta):
    """ Generic base class for Bayesian Filters such as a Kalman Filter.

    Attributes:
        cov (N x N numpy array): Covariance matrix
        meas_mat (Nm x N numpy array): Mapping from states to measurements
        meas_noise (Nm x Nm numpy array): Measurement noise matrix
    """

    def __init__(self, **kwargs):
        self.cov = np.array([[]])
        self.meas_mat = np.array([[]])
        self.meas_noise = np.array([[]])

        self._state_mat = np.array([[]])
        self._input_mat = np.array([[]])
        self._proc_noise = np.array([[]])
        super().__init__(**kwargs)

    @abc.abstractmethod
    def predict(self, **kwargs):
        """ Generic method for the filters prediction step.

        This must be overridden in the inherited class.
        """
        pass

    @abc.abstractmethod
    def correct(self, **kwargs):
        """ Generic method for the filters correction step.

        This must be overridden in the inherited class.
        """
        pass

    def get_state_mat(self, **kwargs):
        """ Returns the state matrix.

        It returns the F matrix in the state space system :math:`x_{k+1} = F x_k + G u_k`
        """
        return self._state_mat.copy()

    def set_state_mat(self, **kwargs):
        """ Sets the state matrix.

        This can be overridden in inherited classes for LTV systems. It
        sets the F matrix in the state space system :math:`x_{k+1} = F x_k + G u_k`

        Keyword Args:
            mat (N x N numpy array): State matrix
        """
        self._state_mat = kwargs.get('mat', np.array([[]]))

    def get_input_mat(self, **kwargs):
        """ Returns the input matrix.

        It returns the G matrix in the state space system :math:`x_{k+1} = F x_k + G u_k`
        """
        return self._input_mat.copy()

    def set_input_mat(self, **kwargs):
        """ Sets the input matrix.

        This can be overridden in inherited classes for LTV systems. It
        sets the G matrix in the state space system :math:`x_{k+1} = F x_k + G u_k`

        Keyword Args:
            mat (N x Nu numpy array): State matrix
        """
        self._input_mat = kwargs.get('mat', np.array([[]]))

    def get_proc_noise(self, **kwargs):
        """ Returns the process noise matrix.

        It returns the :math:`\\Upsilon Q \\Upsilon^T` matrix in the state
        space system :math:`x_{k+1} = F x_k + G u_k + \\Upsilon w_k` where

        .. math::
            E\{w_kw_j^T\} = \\begin{cases}
                0 & k \\neq j \\\\
                Q_k & k = j
                \\end{cases}

        Returns:
            (N x N numpy array): Discrete process noise matrix
        """
        return self._proc_noise

    def set_proc_noise(self, **kwargs):
        """ Sets the process noise matrix.

        This can be overridden in inherited classes. It sets
        the :math:`\\Upsilon Q \\Upsilon^T` matrix in the state
        space system :math:`x_{k+1} = F x_k + G u_k + \\Upsilon w_k` where

        .. math::
            E\{w_kw_j^T\} = \\begin{cases}
                0 & k \\neq j \\\\
                Q_k & k = j
                \\end{cases}

        Keyword Args:
            mat (N x Nu numpy array): Discrete process noise matrix
        """
        self._proc_noise = kwargs.get('mat', np.array([[]]))


class KalmanFilter(BayesFilter):
    """ Implementation of a discrete time Kalman Filter.

    This is loosely based on :cite:`Crassidis2011_OptimalEstimationofDynamicSystems`
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def predict(self, **kwargs):
        """ Implements a discrete time prediction step for a Kalman Filter.

        Keyword Args:
            cur_state (N x 1 numpy array): Current state
            cur_input (N x Nu numpy array): Current input

        Returns:
            (N x 1 numpy array): Next state
        """
        cur_state = kwargs['cur_state']
        cur_input = kwargs['cur_input']

        state_mat = self.get_state_mat(**kwargs)
        input_mat = self.get_input_mat(**kwargs)
        next_state = state_mat @ cur_state + input_mat @ cur_input
        self.cov = state_mat @ self.cov @ state_mat.T \
            + self.get_proc_noise(**kwargs)
        return next_state

    def correct(self, **kwargs):
        """ Implementss a discrete time correction step for a Kalman Filter.

        Keyword Args:
            cur_state (N x 1 numpy array): Current predicted state
            meas (Nm x 1 numpy array): Current measurements

        Returns:
            (N x 1 numpy array): Corrected state
        """
        cur_state = kwargs['cur_state']
        meas = kwargs['meas']

        cov_meas_T = self.cov @ self.meas_mat.T
        kalman_gain = cov_meas_T @ la.inv(self.meas_mat @ cov_meas_T
                                          + self.meas_noise)
        inov = meas - self.meas_mat @ cur_state
        next_state = cur_state + kalman_gain @ inov

        n_states = cur_state.shape[1]
        self.cov = (np.eye(n_states) - kalman_gain @ self.meas_mat) @ self.cov

        return next_state
