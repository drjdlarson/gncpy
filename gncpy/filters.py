import numpy as np
import numpy.linalg as la
from scipy.linalg import expm
import abc
from warnings import warn

from gncpy.math import rk4, get_state_jacobian, disrw


class BayesFilter(metaclass=abc.ABCMeta):
    """ Generic base class for Bayesian Filters such as a Kalman Filter.

    Attributes:
        cov (N x N numpy array): Covariance matrix
        meas_noise (Nm x Nm numpy array): Measurement noise matrix
    """

    def __init__(self, **kwargs):
        self.cov = np.array([[]])
        self.meas_noise = np.array([[]])

        self._state_mat = np.array([[]])
        self._input_mat = np.array([[]])
        self._meas_mat = np.array([[]])
        self._meas_fnc = None
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

    def set_meas_mat(self, **kwargs):
        """ Sets the measurement matrix.

        This can specify a function of the current state and kwargs that 
        returns the matrix, or it can directly give the matrix.

        Keyword Args:
            mat (Nm x N numpy array): Measurement matrix
            fnc (function): Function that takes the state and kwargs and
                returns an Nm x N numpy array
        """
        mat = kwargs.get('mat', None)
        fnc = kwargs.get('fnc', None)
        if mat is not None:
            self._meas_mat = mat
        elif fnc is not None:
            self._meas_fnc = fnc
        else:
            warn('Must set a matrix or function for measurements')

    def get_meas_mat(self, state, **kwargs):
        """ Returns the measurement matrix.

        First checks if a matrix is specified, if not then it evaluates the
        a function at the current state to calculate the matrix.

        Args:
            state (N x 1 numpy array): current state

        Returns:
            (Nm x N numpy array): measurment matrix

        Raises:
            RuntimeError: if the set function hasn't been properly called
        """
        if self._meas_mat.size > 0:
            return self._meas_mat
        elif self._meas_fnc is not None:
            return self._meas_fnc(state, **kwargs)
        else:
            msg = 'Must set the measurement matrix before getting it'
            raise RuntimeError(msg)

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

        meas_mat = self.get_meas_mat(cur_state, **kwargs)
        cov_meas_T = self.cov @ meas_mat.T
        meas_pred_cov = meas_mat @ cov_meas_T + self.meas_noise
        inv_meas_cov = la.inv(meas_pred_cov)
        kalman_gain = cov_meas_T @ inv_meas_cov
        inov = meas - meas_mat @ cur_state
        next_state = cur_state + kalman_gain @ inov

        meas_fit_prob = np.exp(-0.5 * (len(meas) * np.log(2 * np.pi)
                                       + np.log(la.det(meas_pred_cov))
                                       + inov.T @ inv_meas_cov @ inov))
        meas_fit_prob = np.asscalar(meas_fit_prob)

        n_states = cur_state.shape[0]
        self.cov = (np.eye(n_states) - kalman_gain @ meas_mat) @ self.cov

        return (next_state, meas_fit_prob)


class ExtendedKalmanFilter(KalmanFilter):
    def __init__(self, **kwargs):
        self.dyn_fncs = []

        self.proc_map = np.array([[]])
        self.proc_cov = np.array([[]])
        super().__init__(**kwargs)

    def set_proc_mat(self, **kwargs):
        warn("Extended Kalman filter does not use set_proc_mat")

    def get_proc_noise(self, dt, cur_state, cur_input, **kwargs):
        # if the user is manually setting the process noise then return it
        if self._proc_noise.size > 0:
            return self._proc_noise

        # check that we have everything to calculate the process noise
        state_jac = kwargs.get('state_jac', None)
        if state_jac is None:
            state_jac = get_state_jacobian(cur_state, cur_input,
                                           self.dyn_fncs, **kwargs)
        if state_jac.size == 0:
            msg = "State jacobian must be set before getting process noise"
        if self.proc_map.size == 0:
            msg = "Process noise mapping must be set before getting " \
                + "process noise"
            warn(msg)
            return np.array([[]])
        if self.proc_cov.size == 0:
            msg = "Process noise covariance must be set before getting " \
                + "process noise"
            warn(msg)
            return np.array([[]])

        # discritize the process noise
        return disrw(state_jac, self.proc_map, dt, self.proc_cov)

    def set_input_mat(self, **kwargs):
        warn("Extended Kalman filter does not use set_input_mat")

    def get_input_mat(self, **kwargs):
        warn("Extended Kalman filter does not use get_input_mat")
        return np.array([[]])

    def set_state_mat(self, **kwargs):
        warn("Extended Kalman filter does not use set_state_mat")

    def get_state_mat(self, **kwargs):
        cur_state = kwargs['cur_state']
        cur_input = kwargs['cur_input']
        dt = kwargs['dt']

        state_jac = get_state_jacobian(cur_state, cur_input, self.dyn_fncs,
                                       **kwargs)
        return expm(state_jac * dt)

    def predict(self, **kwargs):
        cur_state = kwargs['cur_state']
        dt = kwargs['dt']

        # numerical integration for each state
        def comb_func(x, cur_input, **kwargs):
            out = np.zeros(x.shape)
            for ii, f in enumerate(self.dyn_fncs):
                out[ii] = f(x, cur_input, **kwargs)
            return out

        next_state = rk4(comb_func, cur_state, dt, **kwargs)

        state_mat = self.get_state_mat(**kwargs)
        self.cov = state_mat @ self.cov @ state_mat.T \
            + self.get_proc_noise(state_mat=state_mat, **kwargs)

        return next_state
