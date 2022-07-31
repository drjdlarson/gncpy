"""Definitions for common filters."""
import numpy as np
import numpy.random as rnd
import numpy.linalg as la
import scipy.linalg as sla
import scipy.integrate as s_integrate
import scipy.stats as stats
import abc
from collections import deque
from warnings import warn
from copy import deepcopy
import matplotlib.pyplot as plt

import gncpy.math as gmath
import gncpy.plotting as pltUtil
import gncpy.distributions as gdistrib
from serums.enums import GSMTypes
import gncpy.dynamics as gdyn
import gncpy.errors as gerr


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

        super().__init__(**kwargs)

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


class KalmanFilter(BayesFilter):
    """Implementation of a discrete time Kalman Filter.

    Notes
    -----
    This is loosely based on :cite:`Crassidis2011_OptimalEstimationofDynamicSystems`

    Attributes
    ----------
    cov : N x N numpy array
        Covariance matrix
    meas_noise : Nm x Nm numpy array
        Measurement noise matrix
    proc_noise : N x N numpy array
        Process noise matrix
    dt : float
        Time difference between simulation steps. Required if not using a
        dynamic object for the state model.

    """

    def __init__(
        self, cov=np.array([[]]), meas_noise=np.array([[]]), dt=None, **kwargs
    ):
        self.cov = cov
        self.meas_noise = meas_noise
        self.proc_noise = np.array([[]])
        self.dt = dt

        self._dyn_obj = None
        self._state_mat = np.array([[]])
        self._input_mat = np.array([[]])
        self._get_state_mat = None
        self._get_input_mat = None
        self._meas_mat = np.array([[]])
        self._meas_fnc = None

        self._est_meas_noise_fnc = None

        super().__init__(**kwargs)

    def save_filter_state(self):
        """Saves filter variables so they can be restored later."""
        filt_state = super().save_filter_state()

        filt_state["cov"] = self.cov.copy()
        if self.meas_noise is not None:
            filt_state["meas_noise"] = self.meas_noise.copy()
        else:
            filt_state["meas_noise"] = self.meas_noise

        if self.proc_noise is not None:
            filt_state["proc_noise"] = self.proc_noise.copy()
        else:
            filt_state["proc_noise"] = self.proc_noise

        filt_state["dt"] = self.dt
        filt_state["_dyn_obj"] = deepcopy(self._dyn_obj)

        if self._state_mat is not None:
            filt_state["_state_mat"] = self._state_mat.copy()
        else:
            filt_state["_state_mat"] = self._state_mat

        if self._input_mat is not None:
            filt_state["_input_mat"] = self._input_mat.copy()
        else:
            filt_state["_input_mat"] = self._input_mat

        filt_state["_get_state_mat"] = self._get_state_mat
        filt_state["_get_input_mat"] = self._get_input_mat

        if self._meas_mat is not None:
            filt_state["_meas_mat"] = self._meas_mat.copy()
        else:
            filt_state["_meas_mat"] = self._meas_mat

        filt_state["_meas_fnc"] = self._meas_fnc
        filt_state["_est_meas_noise_fnc"] = self._est_meas_noise_fnc

        return filt_state

    def load_filter_state(self, filt_state):
        """Initializes filter using saved filter state.

        Attributes
        ----------
        filt_state : dict
            Dictionary generated by :meth:`save_filter_state`.
        """
        super().load_filter_state(filt_state)

        self.cov = filt_state["cov"]
        self.meas_noise = filt_state["meas_noise"]
        self.proc_noise = filt_state["proc_noise"]
        self.dt = filt_state["dt"]

        self._dyn_obj = filt_state["_dyn_obj"]
        self._state_mat = filt_state["_state_mat"]
        self._input_mat = filt_state["_input_mat"]
        self._get_state_mat = filt_state["_get_state_mat"]
        self._get_input_mat = filt_state["_get_input_mat"]
        self._meas_mat = filt_state["_meas_mat"]
        self._meas_fnc = filt_state["_meas_fnc"]
        self._est_meas_noise_fnc = filt_state["_est_meas_noise_fnc"]

    def set_state_model(
        self,
        state_mat=None,
        input_mat=None,
        cont_time=False,
        state_mat_fun=None,
        input_mat_fun=None,
        dyn_obj=None,
    ):
        r"""Sets the state model equation for the filter.

        If the continuous time model is used then a `dt` must be provided, see
        the note for algorithm details. Alternatively, if the system is time
        varying then functions can be specified to return the matrices at each
        time step.

        Note
        -----
        This can use a continuous or discrete model. The continuous model will
        be automatically discretized so standard matrix equations can be used.

        If the discrete model is used it is assumed to have the form

        .. math::
            x_{k+1} = F x_k + G u_k

        If the continuous model is used it is assumed to have the form

        .. math::
            \dot{x} = A x + B u

        and is discretized according to

        .. math::
            expm\left[\begin{bmatrix}
                A & B\\
                0 & 0
                \end{bmatrix}dt\right]=\begin{bmatrix}
                F & G\\
                0 & I
                \end{bmatrix}

        Parameters
        ----------
        state_mat : N x N numpy array, optional
            State matrix, continuous or discrete case. The default is None.
        input_mat : N x Nu numpy array, optional
            Input matrixx, continuous or discrete case. The default is None.
        cont_time : bool, optional
            Flag inidicating if the continuous model is provided. The default
            is False.
        state_mat_fun : callable, optional
            Function that returns the `state_mat`, must take timestep and
            `*args`. The default is None.
        input_mat_fun : callable, optional
            Function that returns the `input_mat`, must take timestep, and
            `*args`. The default is None.
        dyn_obj : :class:`gncpy.dynamics.LinearDynamicsBase`, optional
            Sets the dynamics according to the class. The default is None.

        Raises
        ------
        RuntimeError
            If the improper combination of input arguments are specified.

        Returns
        -------
        None.

        """
        have_obj = dyn_obj is not None
        have_mats = state_mat is not None
        have_funs = state_mat_fun is not None
        if have_obj:
            self._dyn_obj = dyn_obj
        elif have_mats and not cont_time:
            self._state_mat = state_mat
            self._input_mat = input_mat
        elif have_mats:
            if self.dt is None:
                msg = "dt must be specified when using continuous time model"
                raise RuntimeError(msg)
            n_cols = state_mat.shape[1] + input_mat.shape[1]
            big_mat = np.vstack(
                (
                    np.hstack((state_mat, input_mat)),
                    np.zeros((input_mat.shape[1], n_cols)),
                )
            )
            res = sla.expm(big_mat * self.dt)
            r_s = 0
            r_e = state_mat.shape[0]
            c_s = 0
            c_e = state_mat.shape[1]
            self._state_mat = res[r_s:r_e, c_s:c_e]
            c_s = c_e
            c_e = res.shape[1]
            self._input_mat = res[r_s:r_e, c_s:c_e]
        elif have_funs:
            self._get_state_mat = state_mat_fun
            self._get_input_mat = input_mat_fun
        else:
            raise RuntimeError("Invalid combination of inputs")

    def set_measurement_model(self, meas_mat=None, meas_fun=None):
        r"""Sets the measurement model for the filter.

        This can either set the constant measurement matrix, or the matrix can
        be time varying.

        Notes
        -----
        This assumes a measurement model of the form

        .. math::
            \tilde{y}_{k+1} = H_{k+1} x_{k+1}^-

        where :math:`H_{k+1}` can be constant over time.

        Parameters
        ----------
        meas_mat : Nm x N numpy array, optional
            Measurement matrix that transforms the state to estimated
            measurements. The default is None.
        meas_fun : callable, optional
            Function that returns the matrix for transforming the state to
            estimated measurements. Must take timestep, and `*args` as
            arguments. The default is None.

        Raises
        ------
        RuntimeError
            Rasied if no arguments are specified.

        Returns
        -------
        None.

        """
        if meas_mat is not None:
            self._meas_mat = meas_mat
        elif meas_fun is not None:
            self._meas_fnc = meas_fun
        else:
            raise RuntimeError("Invalid combination of inputs")

    def set_measurement_noise_estimator(self, function):
        """Sets the model used for estimating the measurement noise parameters.

        This is an optional step and the filter will work properly if this is
        not called. If it is called, the measurement noise will be estimated
        during the filter's correction step and the measurement noise attribute
        will not be used.

        Parameters
        ----------
        function : callable
            A function that implements the prediction and correction steps for
            an appropriate filter to estimate the measurement noise covariance
            matrix. It must have the signature `f(est_meas)` where `est_meas`
            is an Nm x 1 numpy array and it must return an Nm x Nm numpy array
            representing the measurement noise covariance matrix.

        Returns
        -------
        None.
        """
        self._est_meas_noise_fnc = function

    def _predict_next_state(
        self, timestep, cur_state, cur_input, state_mat_args, input_mat_args
    ):
        if self._dyn_obj is not None:
            next_state = self._dyn_obj.propagate_state(
                timestep,
                cur_state,
                u=cur_input,
                state_args=state_mat_args,
                ctrl_args=input_mat_args,
            )
            state_mat = self._dyn_obj.get_state_mat(timestep, *state_mat_args)
        else:
            if self._get_state_mat is not None:
                state_mat = self._get_state_mat(timestep, *state_mat_args)
            elif self._state_mat is not None:
                state_mat = self._state_mat
            else:
                raise RuntimeError("State model not set")

            if self._get_input_mat is not None:
                input_mat = self._get_input_mat(timestep, *input_mat_args)
            elif self._input_mat is not None:
                input_mat = self._input_mat
            else:
                input_mat = None

            next_state = state_mat @ cur_state

            if input_mat is not None and cur_input is not None:
                next_state += input_mat @ cur_input

        return next_state, state_mat

    def predict(
        self, timestep, cur_state, cur_input=None, state_mat_args=(), input_mat_args=()
    ):
        """Implements a discrete time prediction step for a Kalman Filter.

        Parameters
        ----------
        timestep : float
            Current timestep.
        cur_state : N x 1 numpy array
            Current state.
        cur_input : N x Nu numpy array, optional
            Current input. The default is None.
        state_mat_args : tuple, optional
            keyword arguments for the get state matrix function if one has
            been specified or the propagate state function if using a dynamic
            object. The default is ().
        input_mat_args : tuple, optional
            keyword arguments for the get input matrix function if one has
            been specified or the propagate state function if using a dynamic
            object. The default is ().

        Raises
        ------
        RuntimeError
            If the state model has not been set

        Returns
        -------
        next_state : N x 1 numpy array
            Next state.

        """
        next_state, state_mat = self._predict_next_state(
            timestep, cur_state, cur_input, state_mat_args, input_mat_args
        )

        self.cov = state_mat @ self.cov @ state_mat.T + self.proc_noise
        self.cov = (self.cov + self.cov.T) * 0.5
        return next_state

    def _get_meas_mat(self, t, state, n_meas, meas_fun_args):
        # time varying matrix
        if self._meas_fnc is not None:
            meas_mat = self._meas_fnc(t, *meas_fun_args)
        else:
            # constant matrix
            meas_mat = self._meas_mat

        return meas_mat

    def _est_meas(self, timestep, cur_state, n_meas, meas_fun_args):
        meas_mat = self._get_meas_mat(timestep, cur_state, n_meas, meas_fun_args)

        est_meas = meas_mat @ cur_state

        return est_meas, meas_mat

    def _meas_fit_pdf(self, meas, est_meas, meas_cov):
        return stats.multivariate_normal.pdf(
            meas.ravel(), mean=est_meas.ravel(), cov=meas_cov
        )

    def _calc_meas_fit(self, meas, est_meas, meas_cov):
        try:
            meas_fit_prob = self._meas_fit_pdf(meas, est_meas, meas_cov)
        except la.LinAlgError:
            # if self._est_meas_noise_fnc is None:
            #     raise

            msg = (
                "Inovation matrix is singular, likely from bad "
                + "measurement-state pairing for measurement noise estimation."
            )
            raise gerr.ExtremeMeasurementNoiseError(msg) from None
        return meas_fit_prob

    def correct(self, timestep, meas, cur_state, meas_fun_args=()):
        """Implements a discrete time correction step for a Kalman Filter.

        Parameters
        ----------
        timestep : float
            Current timestep.
        meas : Nm x 1 numpy array
            Current measurement.
        cur_state : N x 1 numpy array
            Current state.
        meas_fun_args : tuple, optional
            Arguments for the measurement matrix function if one has
            been specified. The default is ().

        Raises
        ------
        gncpy.errors.ExtremeMeasurementNoiseError
            If the measurement fit probability calculation fails.

        Returns
        -------
        next_state : N x 1 numpy array
            The corrected state.
        meas_fit_prob : float
            Goodness of fit of the measurement based on the state and
            covariance assuming Gaussian noise.

        """
        est_meas, meas_mat = self._est_meas(
            timestep, cur_state, meas.size, meas_fun_args
        )

        # get the Kalman gain
        cov_meas_T = self.cov @ meas_mat.T
        inov_cov = meas_mat @ cov_meas_T

        # estimate the measurement noise online if applicable
        if self._est_meas_noise_fnc is not None:
            self.meas_noise = self._est_meas_noise_fnc(est_meas, inov_cov)

        inov_cov += self.meas_noise
        inov_cov = (inov_cov + inov_cov.T) * 0.5
        if self.use_cholesky_inverse:
            sqrt_inv_inov_cov = la.inv(la.cholesky(inov_cov))
            inv_inov_cov = sqrt_inv_inov_cov.T @ sqrt_inv_inov_cov
        else:
            inv_inov_cov = la.inv(inov_cov)
        kalman_gain = cov_meas_T @ inv_inov_cov

        # update the state with measurement
        inov = meas - est_meas
        next_state = cur_state + kalman_gain @ inov

        # update the covariance
        n_states = cur_state.shape[0]
        self.cov = (np.eye(n_states) - kalman_gain @ meas_mat) @ self.cov

        # calculate the measuremnt fit probability assuming Gaussian
        meas_fit_prob = self._calc_meas_fit(meas, est_meas, inov_cov)

        return (next_state, meas_fit_prob)


class ExtendedKalmanFilter(KalmanFilter):
    """Implementation of a continuous-discrete time Extended Kalman Filter.

    This is loosely based on :cite:`Crassidis2011_OptimalEstimationofDynamicSystems`

    Attributes
    ----------
    cont_cov : bool, optional
        Flag indicating if a continuous model of the covariance matrix should
        be used in the filter update step. The default is True.
    integrator_type : string, optional
        integrator type as defined by scipy's integrate.ode function. The
        default is `dopri5`. Only used if a dynamic object is not specified.
    integrator_params : dict, optional
        additional parameters for the integrator. The default is {}. Only used
        if a dynamic object is not specified.
    """

    def __init__(self, cont_cov=True, dyn_obj=None, ode_lst=None, **kwargs):
        super().__init__(**kwargs)

        self.cont_cov = cont_cov
        self.integrator_type = "dopri5"
        self.integrator_params = {}

        self._ode_lst = None

        # if dyn_obj is not None or ode_lst is not None:
        #     self.set_state_model(dyn_obj=dyn_obj, ode_lst=ode_lst)

        self._integrator = None

    def save_filter_state(self):
        """Saves filter variables so they can be restored later."""
        filt_state = super().save_filter_state()

        filt_state["cont_cov"] = self.cont_cov
        filt_state["integrator_type"] = self.integrator_type
        filt_state["integrator_params"] = deepcopy(self.integrator_params)
        filt_state["_ode_lst"] = self._ode_lst
        filt_state["_integrator"] = self._integrator

        return filt_state

    def load_filter_state(self, filt_state):
        """Initializes filter using saved filter state.

        Attributes
        ----------
        filt_state : dict
            Dictionary generated by :meth:`save_filter_state`.
        """
        super().load_filter_state(filt_state)

        self.cont_cov = filt_state["cont_cov"]
        self.integrator_type = filt_state["integrator_type"]
        self.integrator_params = filt_state["integrator_params"]
        self._ode_lst = filt_state["_ode_lst"]
        self._integrator = filt_state["_integrator"]

    def set_state_model(self, dyn_obj=None, ode_lst=None):
        r"""Sets the state model equations.

        This allows for setting the differential equations directly

        .. math::
            \dot{x} = f(t, x, u)

        or setting a :class:`gncpy.dynamics.NonlinearDynamicsBase` object. If
        the object is specified then a local copy is created. A
        :class:`gncpy.dynamics.LinearDynamicsBase` can also be used in which
        case the dynamics follow the same form as the KF. If a linear dynamics
        object is used then it is recommended to set the filters dt manually so
        a continuous covariance model can be used in the prediction step.

        Parameters
        ----------
        dyn_obj : :class:`gncpy.dynamics.NonlinearDynamicsBase` or :class:`gncpy.dynamics.LinearDynamicsBase`, optional
            Sets the dynamics according to the class. The default is None.
        ode_lst : list, optional
            callable functions, 1 per ode/state. The callabale must have the
            signature `f(t, x, *f_args)` just like scipy.integrate's ode
            function. The default is None.

        Raises
        ------
        RuntimeError
            If neither argument is specified.

        Returns
        -------
        None.

        """
        if dyn_obj is not None and (
            isinstance(dyn_obj, gdyn.NonlinearDynamicsBase)
            or isinstance(dyn_obj, gdyn.LinearDynamicsBase)
        ):
            self._dyn_obj = deepcopy(dyn_obj)
        elif ode_lst is not None and len(ode_lst) > 0:
            self._ode_lst = ode_lst
        else:
            msg = "Invalid state model specified. Check arguments"
            raise RuntimeError(msg)

    def _cont_dyn(self, t, x, *args):
        """Used in integrator if an ode list is specified."""
        out = np.zeros(x.shape)

        for ii, f in enumerate(self._ode_lst):
            out[ii] = f(t, x, *args)

        return out

    def _predict_next_state(self, timestep, cur_state, dyn_fun_params):
        if self._dyn_obj is not None:
            next_state = self._dyn_obj.propagate_state(
                timestep, cur_state, state_args=dyn_fun_params
            )
            if isinstance(self._dyn_obj, gdyn.LinearDynamicsBase):
                state_mat = self._dyn_obj.get_state_mat(timestep, *dyn_fun_params)
                dt = self.dt
            else:
                state_mat = self._dyn_obj.get_state_mat(
                    timestep, cur_state, dyn_fun_params
                )
                dt = self._dyn_obj.dt
        elif self._ode_lst is not None:
            self._integrator = s_integrate.ode(self._cont_dyn)
            self._integrator.set_integrator(
                self.integrator_type, **self.integrator_params
            )
            self._integrator.set_initial_value(cur_state, timestep)
            self._integrator.set_f_params(*dyn_fun_params)

            if self.dt is None:
                raise RuntimeError("dt must be set when using an ODE list")

            next_time = timestep + self.dt
            next_state = self._integrator.integrate(next_time).reshape(cur_state.shape)
            if not self._integrator.successful():
                msg = "Integration failed at time {}".format(timestep)
                raise RuntimeError(msg)

            state_mat = gmath.get_state_jacobian(
                timestep, cur_state, self._ode_lst, dyn_fun_params
            )

            dt = self.dt
        else:
            raise RuntimeError("State model not set")

        return next_state, state_mat, dt

    def predict(self, timestep, cur_state, dyn_fun_params=None):
        r"""Prediction step of the EKF.

        This assumes continuous time dynamics and integrates the ode's to get
        the next state.

        .. math::
            x_{k+1} = \int_t^{t+dt} f(t, x, \phi) dt

        for arbitrary parameters :math:`\phi`


        Parameters
        ----------
        timestep : float
            Current timestep.
        cur_state : N x 1 numpy array
            Current state.
        dyn_fun_params : tuple, optional
            Extra arguments to be passed to the dynamics function. The default
            is None.

        Raises
        ------
        RuntimeError
            Integration fails, or state model not set.

        Returns
        -------
        next_state : N x 1 numpy array
            The predicted state.

        """
        if dyn_fun_params is None:
            dyn_fun_params = ()
        next_state, state_mat, dt = self._predict_next_state(
            timestep, cur_state, dyn_fun_params
        )

        if self.cont_cov:

            def ode(t, x, n_states, F, proc_noise):
                P = x.reshape((n_states, n_states))
                P_dot = F @ P + P @ F.T + proc_noise
                return P_dot.ravel()

            integrator = s_integrate.ode(ode)
            integrator.set_integrator(self.integrator_type, **self.integrator_params)
            integrator.set_initial_value(self.cov.flatten(), timestep)
            integrator.set_f_params(cur_state.size, state_mat, self.proc_noise)
            tmp = integrator.integrate(timestep + dt)
            if not integrator.successful():
                msg = "Failed to integrate covariance at {}".format(timestep)
                raise RuntimeError(msg)

            self.cov = tmp.reshape(self.cov.shape)

        else:
            self.cov = state_mat @ self.cov @ state_mat.T + self.proc_noise

        return next_state

    def _get_meas_mat(self, t, state, n_meas, meas_fun_args):
        # non-linear mapping, potentially time varying
        if self._meas_fnc is not None:
            # calculate partial derivatives
            meas_mat = np.zeros((n_meas, state.size))
            for ii, h in enumerate(self._meas_fnc):
                res = gmath.get_jacobian(
                    state.copy(),
                    lambda _x, *_f_args: h(t, _x, *_f_args),
                    f_args=meas_fun_args,
                )
                meas_mat[[ii], :] = res.T
        else:
            # constant matrix
            meas_mat = self._meas_mat

        return meas_mat

    def _est_meas(self, timestep, cur_state, n_meas, meas_fun_args):
        meas_mat = self._get_meas_mat(timestep, cur_state, n_meas, meas_fun_args)

        if self._meas_fnc is not None:
            est_meas = np.nan * np.ones((n_meas, 1))
            for ii, h in enumerate(self._meas_fnc):
                est_meas[ii] = h(timestep, cur_state, *meas_fun_args)
        else:
            est_meas = meas_mat @ cur_state

        return est_meas, meas_mat

    def set_measurement_model(self, meas_mat=None, meas_fun_lst=None):
        r"""Sets the measurement model for the filter.

        This can either set the constant measurement matrix, or a set of
        non-linear functions (potentially time varying) to map states to
        measurements.

        Notes
        -----
        The constant matrix assumes a measurement model of the form

        .. math::
            \tilde{y}_{k+1} = H x_{k+1}^-

        and the non-linear case assumes

        .. math::
            \tilde{y}_{k+1} = h(t, x_{k+1}^-)

        Parameters
        ----------
        meas_mat : Nm x N numpy array, optional
            Measurement matrix that transforms the state to estimated
            measurements. The default is None.
        meas_fun_lst : list, optional
            Non-linear functions that return the expected measurement for the
            given state. Each function must have the signature `h(t, x, *args)`.
            The default is None.

        Raises
        ------
        RuntimeError
            Rasied if no arguments are specified.

        Returns
        -------
        None.
        """
        super().set_measurement_model(meas_mat=meas_mat, meas_fun=meas_fun_lst)


class StudentsTFilter(KalmanFilter):
    r"""Implementation of a Students T filter.

    This is based on :cite:`Liu2018_AStudentsTMixtureProbabilityHypothesisDensityFilterforMultiTargetTrackingwithOutliers`
    and :cite:`Roth2013_AStudentsTFilterforHeavyTailedProcessandMeasurementNoise`
    and uses moment matching to limit the degree of freedom growth.

    Notes
    -----
    This models the multi-variate Student's t-distribution as

    .. math::
        \begin{align}
            p(x) &= \frac{\Gamma(\frac{\nu + 2}{2})}{\Gamma(\frac{\nu}{2})}
                \frac{1}{(\nu \pi)^{d/2}}
                \frac{1}{\sqrt{\vert \Sigma \vert}}\left( 1 +
                \frac{\Delta^2}{\nu}\right)^{-\frac{\nu + 2}{\nu}} \\
            \Delta^2 &= (x - m)^T \Sigma^{-1} (x - m)
        \end{align}

    or compactly as :math:`St(x; m,\Sigma, \nu) = p(x)` for scale matrix
    :math:`\Sigma` and degree of freedom :math:`\nu`

    Attributes
    ----------
    scale : N x N numpy array, optional
        Scaling matrix of the Students T distribution.  The default is np.array([[]]).
    dof : int , optional
        Degree of freedom for the state distribution. The default is 3.
    proc_noise_dof : int, optional
        Degree of freedom for the process noise model. The default is 3.
    meas_noise_dof : int, optional
        Degree of freedom for the measurement noise model. The default is 3.
    use_moment_matching : bool, optional
        Flag indicating if moment matching is used to maintain the heavy tail
        property as the filter propagates over time. The default is True.
    """

    def __init__(
        self,
        scale=np.array([[]]),
        dof=3,
        proc_noise_dof=3,
        meas_noise_dof=3,
        use_moment_matching=True,
        **kwargs
    ):
        self.scale = scale
        self.dof = dof
        self.proc_noise_dof = proc_noise_dof
        self.meas_noise_dof = meas_noise_dof
        self.use_moment_matching = use_moment_matching

        super().__init__(**kwargs)

    def save_filter_state(self):
        """Saves filter variables so they can be restored later."""
        filt_state = super().save_filter_state()

        filt_state["scale"] = self.scale
        filt_state["dof"] = self.dof
        filt_state["proc_noise_dof"] = self.proc_noise_dof
        filt_state["meas_noise_dof"] = self.meas_noise_dof
        filt_state["use_moment_matching"] = self.use_moment_matching

        return filt_state

    def load_filter_state(self, filt_state):
        """Initializes filter using saved filter state.

        Attributes
        ----------
        filt_state : dict
            Dictionary generated by :meth:`save_filter_state`.
        """
        super().load_filter_state(filt_state)

        self.scale = filt_state["scale"]
        self.dof = filt_state["dof"]
        self.proc_noise_dof = filt_state["proc_noise_dof"]
        self.meas_noise_dof = filt_state["meas_noise_dof"]
        self.use_moment_matching = filt_state["use_moment_matching"]

    @property
    def cov(self):
        """Read only covariance matrix.

        This is calculated from the scale matrix and degree of freedom.

        Raises
        ------
        RuntimeError
            If the degree of freedom is less than or equal to 2

        Returns
        -------
        N x 1 numpy array
            Calcualted covariance matrix
        """
        if self.dof <= 2:
            msg = "Degrees of freedom ({}) must be > 2"
            raise RuntimeError(msg.format(self.dof))
        return self.dof / (self.dof - 2) * self.scale

    @cov.setter
    def cov(self, cov):
        pass
        # msg = 'Covariance is read only. Set the scale matrix instead'
        # raise RuntimeError(msg)

    def predict(
        self, timestep, cur_state, cur_input=None, state_mat_args=(), input_mat_args=()
    ):
        """Implements the prediction step of the Students T filter.

        Parameters
        ----------
        timestep : float
            Current timestep.
        cur_state : N x 1 numpy array
            Current state.
        cur_input : N x Nu numpy array, optional
            Current input. The default is None.
        state_mat_args : tuple, optional
            keyword arguments for the get state matrix function if one has
            been specified or the propagate state function if using a dynamic
            object. The default is ().
        input_mat_args : tuple, optional
            keyword arguments for the get input matrix function if one has
            been specified or the propagate state function if using a dynamic
            object. The default is ().

        Returns
        -------
        next_state : N x 1 numpy array
            Next state.
        """
        next_state, state_mat = self._predict_next_state(
            timestep, cur_state, cur_input, state_mat_args, input_mat_args
        )

        factor = (
            self.proc_noise_dof
            * (self.dof - 2)
            / (self.dof * (self.proc_noise_dof - 2))
        )
        self.scale = state_mat @ self.scale @ state_mat.T + factor * self.proc_noise
        self.scale = (self.scale + self.scale.T) * 0.5

        return next_state

    def _meas_fit_pdf(self, meas, est_meas, meas_cov):
        return stats.multivariate_t.pdf(
            meas.ravel(), loc=est_meas.ravel(), shape=meas_cov, df=self.meas_noise_dof
        )

    def correct(self, timestep, meas, cur_state, meas_fun_args=()):
        """Implements the correction step of the students T filter.

        This also performs the moment matching.

        Parameters
        ----------
        timestep : float
            Current timestep.
        meas : Nm x 1 numpy array
            Current measurement.
        cur_state : N x 1 numpy array
            Current state.
        meas_fun_args : tuple, optional
            Arguments for the measurement matrix function if one has
            been specified. The default is ().

        Returns
        -------
        next_state : N x 1 numpy array
            The corrected state.
        meas_fit_prob : float
            Goodness of fit of the measurement based on the state and
            scale assuming Student's t noise.
        """
        est_meas, meas_mat = self._est_meas(
            timestep, cur_state, meas.size, meas_fun_args
        )

        # get gain
        scale_meas_T = self.scale @ meas_mat.T
        factor = (
            self.meas_noise_dof
            * (self.dof - 2)
            / (self.dof * (self.meas_noise_dof - 2))
        )
        inov_cov = meas_mat @ scale_meas_T + factor * self.meas_noise
        inov_cov = (inov_cov + inov_cov.T) * 0.5
        if self.use_cholesky_inverse:
            sqrt_inv_inov_cov = la.inv(la.cholesky(inov_cov))
            inv_inov_cov = sqrt_inv_inov_cov.T @ sqrt_inv_inov_cov
        else:
            inv_inov_cov = la.inv(inov_cov)
        gain = scale_meas_T @ inv_inov_cov
        P_kk = (np.eye(cur_state.shape[0]) - gain @ meas_mat) @ self.scale

        # update state
        innov = meas - est_meas
        delta_2 = innov.T @ inv_inov_cov @ innov
        next_state = cur_state + gain @ innov

        # moment matching
        if self.use_moment_matching:
            dof_p = self.dof + meas.size
            factor = (self.dof + delta_2) / dof_p
            P_k = factor * P_kk

            factor = dof_p * (self.dof - 2) / (self.dof * (dof_p - 2))
            self.scale = factor * P_k

        else:
            self.scale = P_kk

        # get measurement fit
        meas_fit_prob = self._calc_meas_fit(meas, est_meas, inov_cov)

        return next_state, meas_fit_prob


class UnscentedKalmanFilter(ExtendedKalmanFilter):
    """Implements an unscented kalman filter.

    This allows for linear or non-linear dynamics by utilizing the either the
    underlying KF or EKF functions where appropriate. It utilizes the same
    constraints on the measurement model as the EKF.

    Notes
    -----
    For details on the filter see
    :cite:`Wan2000_TheUnscentedKalmanFilterforNonlinearEstimation`. This
    implementation assumes that the noise is purely addative and as such,
    appropriate simplifications have been made. These simplifications include
    not using sigma points to track the noise, and using the fixed process and
    measurement noise covariance matrices in the filter's covariance updates.

    Attributes
    ----------
    alpha : float
        Tunig parameter for sigma points, influences the spread of sigma points about the
        mean. In range (0, 1]. If specified then a value does not need to be
        given to the :meth:`.init_sigma_points` function.
    kappa : float
        Tunig parameter for sigma points, influences the spread of sigma points about the
        mean. In range [0, inf]. If specified then a value does not need to be
        given to the :meth:`.init_sigma_points` function.
    beta : float
        Tunig parameter for sigma points. In range [0, Inf]. If specified then
        a value does not need to be given to the :meth:`.init_sigma_points` function.
        Defaults to 2 (ideal for gaussians).
    """

    def __init__(self, sigmaPoints=None, **kwargs):
        """Initialize an instance.

        Parameters
        ----------
        sigmaPoints : :class:`.distributions.SigmaPoints`, optional
            Set of initialized sigma points to use. The default is None.
        **kwargs : dict, optional
            Additional arguments for parent constructors.
        """
        self.alpha = 1
        self.kappa = 0
        self.beta = 2

        self._stateSigmaPoints = None
        if isinstance(sigmaPoints, gdistrib.SigmaPoints):
            self._stateSigmaPoints = sigmaPoints
            self.alpha = sigmaPoints.alpha
            self.beta = sigmaPoints.beta
            self.kappa = sigmaPoints.kappa

        self._use_lin_dyn = False
        self._use_non_lin_dyn = False
        self._est_meas_noise_fnc = None

        super().__init__(**kwargs)

    def save_filter_state(self):
        """Saves filter variables so they can be restored later."""
        filt_state = super().save_filter_state()

        filt_state["alpha"] = self.alpha
        filt_state["kappa"] = self.kappa
        filt_state["beta"] = self.beta

        filt_state["_stateSigmaPoints"] = deepcopy(self._stateSigmaPoints)
        filt_state["_use_lin_dyn"] = self._use_lin_dyn
        filt_state["_use_non_lin_dyn"] = self._use_non_lin_dyn
        filt_state["_est_meas_noise_fnc"] = self._est_meas_noise_fnc

        return filt_state

    def load_filter_state(self, filt_state):
        """Initializes filter using saved filter state.

        Attributes
        ----------
        filt_state : dict
            Dictionary generated by :meth:`save_filter_state`.
        """
        super().load_filter_state(filt_state)

        self.alpha = filt_state["alpha"]
        self.kappa = filt_state["kappa"]
        self.beta = filt_state["beta"]

        self._stateSigmaPoints = deepcopy(filt_state["_stateSigmaPoints"])
        self._use_lin_dyn = filt_state["_use_lin_dyn"]
        self._use_non_lin_dyn = filt_state["_use_non_lin_dyn"]
        self._est_meas_noise_fnc = filt_state["_est_meas_noise_fnc"]

    def init_sigma_points(self, state0, alpha=None, kappa=None, beta=None):
        """Initializes the sigma points used by the filter.

        Parameters
        ----------
        state0 : N x 1 numpy array
            Initial state.
        alpha : float, optional
            Tunig parameter, influences the spread of sigma points about the
            mean. In range (0, 1]. If not supplied the class value will be used.
            If a value is given here then the class value will be updated.
        kappa : float, optional
            Tunig parameter, influences the spread of sigma points about the
            mean. In range [0, inf]. If not supplied the class value will be used.
            If a value is given here then the class value will be updated.
        beta : float, optional
            Tunig parameter for distribution type. In range [0, Inf]. If not
            supplied the class value will be used. If a value is given here
            then the class value will be updated.
            Defaults to 2 for gaussians.
        """
        num_axes = state0.size
        if alpha is None:
            alpha = self.alpha
        else:
            self.alpha = alpha
        if kappa is None:
            kappa = self.kappa
        else:
            self.kappa = kappa
        if beta is None:
            beta = self.beta
        else:
            self.beta = beta

        self._stateSigmaPoints = gdistrib.SigmaPoints(
            alpha=alpha, kappa=kappa, beta=beta, num_axes=num_axes
        )
        self._stateSigmaPoints.init_weights()
        self._stateSigmaPoints.update_points(state0, self.cov)

    def set_state_model(
        self,
        state_mat=None,
        input_mat=None,
        cont_time=False,
        state_mat_fun=None,
        input_mat_fun=None,
        dyn_obj=None,
        ode_lst=None,
    ):
        """Sets the state model for the filter.

        This can use either linear dynamics (by calling the kalman filters
        :meth:`gncpy.filters.KalmanFilter.set_state_model`) or non-linear dynamics
        (by calling :meth:`gncpy.filters.ExtendedKalmanFilter.set_state_model`).
        The linearness is automatically determined by the input arguments specified.

        Parameters
        ----------
        state_mat : N x N numpy array, optional
            State matrix, continuous or discrete case. The default is None.
        input_mat : N x Nu numpy array, optional
            Input matrixx, continuous or discrete case. The default is None.
        cont_time : bool, optional
            Flag inidicating if the continuous model is provided. The default
            is False.
        state_mat_fun : callable, optional
            Function that returns the `state_mat`, must take timestep and
            `*args`. The default is None.
        input_mat_fun : callable, optional
            Function that returns the `input_mat`, must take timestep, and
            `*args`. The default is None.
        dyn_obj : :class:`gncpy.dynamics.LinearDynamicsBase` or :class:`gncpy.dynamics.NonlinearDynamicsBase`, optional
            Sets the dynamics according to the class. The default is None.
        ode_lst : list, optional
            callable functions, 1 per ode/state. The callabale must have the
            signature `f(t, x, *f_args)` just like scipy.integrate's ode
            function. The default is None.

        Raises
        ------
        RuntimeError
            If an invalid state model or combination of inputs is specified.

        Returns
        -------
        None.
        """
        self._use_lin_dyn = (
            state_mat is not None
            or state_mat_fun is not None
            or isinstance(dyn_obj, gdyn.LinearDynamicsBase)
        )
        self._use_non_lin_dyn = (
            isinstance(dyn_obj, gdyn.NonlinearDynamicsBase) or ode_lst is not None
        ) and not self._use_lin_dyn

        # allow for linear or non linear dynamics by calling the appropriate parent
        if self._use_lin_dyn:
            KalmanFilter.set_state_model(
                self,
                state_mat=state_mat,
                input_mat=input_mat,
                cont_time=cont_time,
                state_mat_fun=state_mat_fun,
                input_mat_fun=input_mat_fun,
                dyn_obj=dyn_obj,
            )
        elif self._use_non_lin_dyn:
            ExtendedKalmanFilter.set_state_model(self, dyn_obj=dyn_obj, ode_lst=ode_lst)
        else:
            raise RuntimeError("Invalid state model.")

    def set_measurement_noise_estimator(self, function):
        """Sets the model used for estimating the measurement noise parameters.

        This is an optional step and the filter will work properly if this is
        not called. If it is called, the measurement noise will be estimated
        during the filter's correction step and the measurement noise attribute
        will not be used.

        Parameters
        ----------
        function : callable
            A function that implements the prediction and correction steps for
            an appropriate filter to estimate the measurement noise covariance
            matrix. It must have the signature `f(est_meas)` where `est_meas`
            is an Nm x 1 numpy array and it must return an Nm x Nm numpy array
            representing the measurement noise covariance matrix.

        Returns
        -------
        None.
        """
        self._est_meas_noise_fnc = function

    def predict(
        self,
        timestep,
        cur_state,
        cur_input=None,
        state_mat_args=(),
        input_mat_args=(),
        dyn_fun_params=(),
    ):
        """Prediction step of the UKF.

        Automatically calls the state propagation method from either
        :meth:`gncpy.KalmanFilter.predict` or :meth:`gncpy.ExtendedKalmanFilter.predict`
        depending on if a linear or non-linear state model was specified.
        If a linear model is used only the parameters that can be passed to
        :meth:`gncpy.KalmanFilter.predict` will be used by this function.
        Otherwise the parameters for :meth:`gncpy.ExtendedKalmanFilter.predict`
        will be used.

        Parameters
        ----------
        timestep : float
            Current timestep.
        cur_state : N x 1 numpy array
            Current state.
        cur_input : N x Nu numpy array, optional
            Current input for linear models. The default is None.
        state_mat_args : tuple, optional
            keyword arguments for the get state matrix function if one has
            been specified or the propagate state function if using a linear
            dynamic object. The default is ().
        input_mat_args : tuple, optional
            keyword arguments for the get input matrix function if one has
            been specified or the propagate state function if using a linear
            dynamic object. The default is ().
        dyn_fun_params : tuple, optional
            Extra arguments to be passed to the dynamics function for non-linear
            models. The default is ().

        Raises
        ------
        RuntimeError
            If a state model has not been set.

        Returns
        -------
        next_state : N x 1 numpy array
            The next state.

        """
        self._stateSigmaPoints.update_points(cur_state, self.cov)

        # propagate points
        if self._use_lin_dyn:
            new_points = np.array(
                [
                    KalmanFilter._predict_next_state(
                        self,
                        timestep,
                        x.reshape((x.size, 1)),
                        cur_input,
                        state_mat_args,
                        input_mat_args,
                    )[0].ravel()
                    for x in self._stateSigmaPoints.points
                ]
            )
        elif self._use_non_lin_dyn:
            new_points = np.array(
                [
                    ExtendedKalmanFilter._predict_next_state(
                        self, timestep, x.reshape((x.size, 1)), dyn_fun_params
                    )[0].ravel()
                    for x in self._stateSigmaPoints.points
                ]
            )
        else:
            raise RuntimeError("State model not specified")

        self._stateSigmaPoints.points = new_points

        # update covariance
        self.cov = self._stateSigmaPoints.cov + self.proc_noise
        self.cov = (self.cov + self.cov.T) * 0.5

        # estimate weighted state output
        next_state = self._stateSigmaPoints.mean

        return next_state

    def _calc_meas_cov(self, timestep, n_meas, meas_fun_args):
        est_points = np.array(
            [
                self._est_meas(timestep, x.reshape((x.size, 1)), n_meas, meas_fun_args)[
                    0
                ]
                for x in self._stateSigmaPoints.points
            ]
        )
        est_meas = gmath.weighted_sum_vec(
            self._stateSigmaPoints.weights_mean, est_points
        )
        diff = est_points - est_meas
        meas_cov_lst = diff @ diff.reshape((est_points.shape[0], 1, est_meas.size))

        partial_cov = gmath.weighted_sum_mat(
            self._stateSigmaPoints.weights_cov, meas_cov_lst
        )
        # estimate the measurement noise if applicable
        if self._est_meas_noise_fnc is not None:
            self.meas_noise = self._est_meas_noise_fnc(est_meas, partial_cov)

        meas_cov = self.meas_noise + partial_cov

        return meas_cov, est_points, est_meas

    def correct(self, timestep, meas, cur_state, meas_fun_args=()):
        """Correction step of the UKF.

        Parameters
        ----------
        timestep : float
            Current timestep.
        meas : Nm x 1 numpy array
            Current measurement.
        cur_state : N x 1 numpy array
            Current state.
        meas_fun_args : tuple, optional
            Arguments for the measurement matrix function if one has
            been specified. The default is ().

        Raises
        ------
        :class:`.errors.ExtremeMeasurementNoiseError`
            If estimating the measurement noise and the measurement fit calculation fails.
        LinAlgError
            Numpy exception raised if not estimating noise and measurement fit fails.

        Returns
        -------
        next_state : N x 1 numpy array
            corrected state.
        meas_fit_prob : float
            measurement fit probability assuming a Gaussian distribution.

        """
        meas_cov, est_points, est_meas = self._calc_meas_cov(
            timestep, meas.size, meas_fun_args
        )

        state_diff = self._stateSigmaPoints.points - cur_state.ravel()
        meas_diff = (est_points - est_meas).reshape(
            (est_points.shape[0], 1, est_meas.size)
        )
        cross_cov_lst = state_diff.reshape((*state_diff.shape, 1)) @ meas_diff
        cross_cov = gmath.weighted_sum_mat(
            self._stateSigmaPoints.weights_cov, cross_cov_lst
        )

        if self.use_cholesky_inverse:
            sqrt_inv_meas_cov = la.inv(la.cholesky(meas_cov))
            inv_meas_cov = sqrt_inv_meas_cov.T @ sqrt_inv_meas_cov
        else:
            inv_meas_cov = la.inv(meas_cov)
        gain = cross_cov @ inv_meas_cov
        inov = meas - est_meas

        self.cov = self.cov - gain @ meas_cov @ gain.T
        self.cov = (self.cov + self.cov.T) * 0.5
        next_state = cur_state + gain @ inov

        meas_fit_prob = self._calc_meas_fit(meas, est_meas, meas_cov)

        return next_state, meas_fit_prob


class BootstrapFilter(BayesFilter):
    """Stripped down version of the :class:`.ParticleFilter`.

    This is an alternative implementation of a basic Particle filter. This
    removes some of the quality of life features of the :class:`.ParticleFilter`
    class and can be more complicated to setup. But it may provide runtime improvements
    for simple cases. Most times it is advised to use the :class:`.ParticleFilter`
    instead of this class. Most other derived classes use the :class:`.ParticleFilter`
    class as a base.
    """

    def __init__(
        self,
        importance_dist_fnc=None,
        importance_weight_fnc=None,
        particleDistribution=None,
        rng=None,
        **kwargs
    ):
        """Initializes the object.

        Parameters
        ----------
        importance_dist_fnc : callable, optional
            Must have the signature `f(parts, rng)` where `parts` is an
            instance of :class:`.distributions.SimpleParticleDistribution`
            and `rng` is a numpy random generator. It must return a numpy array
            of new particles for a :class:`.distributions.SimpleParticleDistribution`.
            Any state transitions to a new timestep must happen within this
            function. The default is None.
        importance_weight_fnc : callable, optional
            Must have the signature `f(meas, parts)` where `meas` is an Nm x 1
            numpy array representing the measurement and `parts` is the
            numpy array of particles from a :class:`.distributions.SimpleParticleDistribution`
            object. It must return a numpy array of weights, one for each
            particle. The default is None.
        particleDistribution : :class:`.distributions.SimpleParticleDistribution`, optional
            Initial particle distribution to use. The default is None.
        rng : numpy random generator, optional
            Random number generator to use. If none supplied then the numpy default
            is used. The default is None.
        **kwargs : dict
            Additional arguments for the parent constructor.
        """
        super().__init__(**kwargs)
        self.importance_dist_fnc = importance_dist_fnc
        self.importance_weight_fnc = importance_weight_fnc
        if particleDistribution is None:
            self.particleDistribution = gdistrib.SimpleParticleDistribution()
        else:
            self.particleDistribution = particleDistribution
        if rng is None:
            self.rng = rnd.default_rng()
        else:
            self.rng = rng

    def save_filter_state(self):
        """Saves filter variables so they can be restored later."""
        filt_state = super().save_filter_state()
        filt_state["importance_dist_fnc"] = self.importance_dist_fnc
        filt_state["importance_weight_fnc"] = self.importance_weight_fnc
        filt_state["particleDistribution"] = deepcopy(self.particleDistribution)
        filt_state["rng"] = self.rng

        return filt_state

    def load_filter_state(self, filt_state):
        """Initializes filter using saved filter state.

        Attributes
        ----------
        filt_state : dict
            Dictionary generated by :meth:`save_filter_state`.
        """
        super().load_filter_state(filt_state)

        self.importance_dist_fnc = filt_state["importance_dist_fnc"]
        self.importance_weight_fnc = filt_state["importance_weight_fnc"]
        self.particleDistribution = filt_state["particleDistribution"]
        self.rng = filt_state["rng"]

    def predict(self, timestep):
        """Prediction step of the filter.

        Calls the importance distribution function to generate new samples of
        particles.

        Parameters
        ----------
        timestep : float
            Current timestep.

        Returns
        -------
        N x 1
            mean estimate of the particles.
        """
        self.particleDistribution.particles = self.importance_dist_fnc(
            self.particleDistribution, self.rng
        )

        shape = (self.particleDistribution.particles.shape[1], 1)
        return np.mean(self.particleDistribution.particles, axis=0).reshape(shape)

    def correct(self, timestep, meas):
        """Correction step of the filter.

        Parameters
        ----------
        timestep : float
            Current timestep.
        meas : Nm x 1 numpy array
            Current measurement.

        Raises
        ------
        :class:`gerr.ParticleDepletionError`
            If all particles weights sum to zero (all particles will be removed).

        Returns
        -------
        N x 1 numpy array
            mean estimate of the particles.
        """
        self.particleDistribution.weights *= self.importance_weight_fnc(
            meas, self.particleDistribution.particles
        )
        tot = np.sum(self.particleDistribution.weights)
        if tot <= 0:
            raise gerr.ParticleDepletionError("Importance weights sum to 0.")
        self.particleDistribution.weights /= tot

        # selection
        num_parts = self.particleDistribution.num_particles
        keep_inds = self.rng.choice(
            np.array(range(self.particleDistribution.weights.size)),
            p=self.particleDistribution.weights,
            size=num_parts,
        )
        unique_inds, counts = np.unique(keep_inds, return_counts=True)
        self.particleDistribution.num_parts_per_ind = counts
        self.particleDistribution.particles = self.particleDistribution.particles[
            unique_inds, :
        ]
        self.particleDistribution.weights = (
            1 / num_parts * np.ones(self.particleDistribution.particles.shape[0])
        )

        if unique_inds.size <= 1:
            msg = "Only {:d} particles selected".format(unique_inds.size)
            raise gerr.ParticleDepletionError(msg)

        # weights are all equal here so don't need weighted sum
        shape = (self.particleDistribution.particles.shape[1], 1)
        return np.mean(self.particleDistribution.particles, axis=0).reshape(shape)

    def set_state_model(self, **kwargs):
        """Not used by the Bootstrap filter."""
        warn(
            "Not used by BootstrapFilter, directly handled by importance_dist_fnc.",
            RuntimeWarning,
        )

    def set_measurement_model(self, **kwargs):
        """Not used by the Bootstrap filter."""
        warn(
            "Not used by BootstrapFilter, directly handled by importance_weight_fnc.",
            RuntimeWarning,
        )

    def plot_particles(self, inds, **kwargs):
        """Wrapper for :class:`.distributions.SimpleParticleDistribution.plot_particles`."""
        return self.particleDistribution.plot_particles(inds, **kwargs)


class ParticleFilter(BayesFilter):
    """Implements a basic Particle Filter.

    Notes
    -----
    The implementation is based on
    :cite:`Simon2006_OptimalStateEstimationKalmanHInfinityandNonlinearApproaches`
    and uses Sampling-Importance Resampling (SIR) sampling. Other resampling
    methods can be added in derived classes.

    Attributes
    ----------
    require_copy_prop_parts : bool
        Flag indicating if the propagated particles need to be copied if this
        filter is being manipulated externally. This is a constant value that
        should not be modified outside of the class, but can be overridden by
        inherited classes.
    require_copy_can_dist : bool
        Flag indicating if a candidate distribution needs to be copied if this
        filter is being manipulated externally. This is a constant value that
        should not be modified outside of the class, but can be overridden by
        inherited classes.
    """

    require_copy_prop_parts = True
    require_copy_can_dist = False

    def __init__(
        self,
        dyn_obj=None,
        dyn_fun=None,
        part_dist=None,
        transition_prob_fnc=None,
        rng=None,
        **kwargs
    ):

        self.__meas_likelihood_fnc = None
        self.__proposal_sampling_fnc = None
        self.__proposal_fnc = None
        self.__transition_prob_fnc = None

        if rng is None:
            rng = rnd.default_rng(1)
        self.rng = rng

        self._dyn_fnc = None
        self._dyn_obj = None

        self._meas_mat = None
        self._meas_fnc = None

        if dyn_obj is not None or dyn_fun is not None:
            self.set_state_model(dyn_obj=dyn_obj, dyn_fun=dyn_fun)

        self._particleDist = gdistrib.ParticleDistribution()
        if part_dist is not None:
            self.init_from_dist(part_dist)

        self.prop_parts = []

        super().__init__(**kwargs)

    def save_filter_state(self):
        """Saves filter variables so they can be restored later."""
        filt_state = super().save_filter_state()

        filt_state["__meas_likelihood_fnc"] = self.__meas_likelihood_fnc
        filt_state["__proposal_sampling_fnc"] = self.__proposal_sampling_fnc
        filt_state["__proposal_fnc"] = self.__proposal_fnc
        filt_state["__transition_prob_fnc"] = self.__transition_prob_fnc

        filt_state["rng"] = self.rng

        filt_state["_dyn_fnc"] = self._dyn_fnc
        filt_state["_dyn_obj"] = self._dyn_obj

        filt_state["_meas_mat"] = self._meas_mat
        filt_state["_meas_fnc"] = self._meas_fnc

        filt_state["_particleDist"] = deepcopy(self._particleDist)
        filt_state["prop_parts"] = deepcopy(self.prop_parts)

        return filt_state

    def load_filter_state(self, filt_state):
        """Initializes filter using saved filter state.

        Attributes
        ----------
        filt_state : dict
            Dictionary generated by :meth:`save_filter_state`.
        """
        super().load_filter_state(filt_state)

        self.__meas_likelihood_fnc = filt_state["__meas_likelihood_fnc"]
        self.__proposal_sampling_fnc = filt_state["__proposal_sampling_fnc"]
        self.__proposal_fnc = filt_state["__proposal_fnc"]
        self.__transition_prob_fnc = filt_state["__transition_prob_fnc"]

        self.rng = filt_state["rng"]

        self._dyn_fnc = filt_state["_dyn_fnc"]
        self._dyn_obj = filt_state["_dyn_obj"]

        self._meas_mat = filt_state["_meas_mat"]
        self._meas_fnc = filt_state["_meas_fnc"]

        self._particleDist = filt_state["_particleDist"]
        self.prop_parts = filt_state["prop_parts"]

    @property
    def meas_likelihood_fnc(self):
        r"""A function that returns the likelihood of the measurement.

        This must have the signature :code:`f(y, y_hat, *args)` where `y` is
        the measurement as an Nm x 1 numpy array, and `y_hat` is the estimated
        measurement.

        Notes
        -----
        This represents :math:`p(y_t \vert x_t)` in the importance
        weight

        .. math::

            w_t = w_{t-1} \frac{p(y_t \vert x_t) p(x_t \vert x_{t-1})}{q(x_t \vert x_{t-1}, y_t)}

        Returns
        -------
        callable
            function to return the measurement likelihood.
        """
        return self.__meas_likelihood_fnc

    @meas_likelihood_fnc.setter
    def meas_likelihood_fnc(self, val):
        self.__meas_likelihood_fnc = val

    @property
    def proposal_fnc(self):
        r"""A function that returns the probability for the proposal distribution.

        This must have the signature :code:`f(x_hat, x, y, *args)` where
        `x_hat` is a :class:`gncpy.distributions.Particle` of the estimated
        state, `x` is the particle it is conditioned on, and `y` is the
        measurement.

        Notes
        -----
        This represents :math:`q(x_t \vert x_{t-1}, y_t)` in the importance
        weight

        .. math::

            w_t = w_{t-1} \frac{p(y_t \vert x_t) p(x_t \vert x_{t-1})}{q(x_t \vert x_{t-1}, y_t)}

        Returns
        -------
        callable
            function to return the proposal probability.
        """
        return self.__proposal_fnc

    @proposal_fnc.setter
    def proposal_fnc(self, val):
        self.__proposal_fnc = val

    @property
    def proposal_sampling_fnc(self):
        """A function that returns a random sample from the proposal distribtion.

        This should be consistent with the PDF specified in the
        :meth:`gncpy.filters.ParticleFilter.proposal_fnc`.

        Returns
        -------
        callable
            function to return a random sample.
        """
        return self.__proposal_sampling_fnc

    @proposal_sampling_fnc.setter
    def proposal_sampling_fnc(self, val):
        self.__proposal_sampling_fnc = val

    @property
    def transition_prob_fnc(self):
        r"""A function that returns the transition probability for the state.

        This must have the signature :code:`f(x_hat, x, *args)` where
        `x_hat` is an N x 1 numpy array representing the propagated state, and
        `x` is the state it is conditioned on.

        Notes
        -----
        This represents :math:`p(x_t \vert x_{t-1})` in the importance
        weight

        .. math::

            w_t = w_{t-1} \frac{p(y_t \vert x_t) p(x_t \vert x_{t-1})}{q(x_t \vert x_{t-1}, y_t)}

        Returns
        -------
        callable
            function to return the transition probability.
        """
        return self.__transition_prob_fnc

    @transition_prob_fnc.setter
    def transition_prob_fnc(self, val):
        self.__transition_prob_fnc = val

    def set_state_model(self, dyn_obj=None, dyn_fun=None):
        """Sets the state model.

        Parameters
        ----------
        dyn_obj : :class:gncpy.dynamics.DynamicsBase`, optional
            Dynamic object to use. The default is None.
        dyn_fun : callable, optional
            function that returns the next state. It must have the signature
            `f(t, x, *args)` and return a N x 1 numpy array. The default is None.

        Raises
        ------
        RuntimeError
            If no model is specified.

        Returns
        -------
        None.
        """
        if dyn_obj is not None:
            self._dyn_obj = deepcopy(dyn_obj)
        elif dyn_fun is not None:
            self._dyn_fnc = dyn_fun
        else:
            msg = "Invalid state model specified. Check arguments"
            raise RuntimeError(msg)

    def set_measurement_model(self, meas_mat=None, meas_fun=None):
        r"""Sets the measurement model for the filter.

        This can either set the constant measurement matrix, or a set of
        non-linear functions (potentially time varying) to map states to
        measurements.

        Notes
        -----
        The constant matrix assumes a measurement model of the form

        .. math::
            \tilde{y}_{k+1} = H x_{k+1}^-

        and the non-linear case assumes

        .. math::
            \tilde{y}_{k+1} = h(t, x_{k+1}^-)

        Parameters
        ----------
        meas_mat : Nm x N numpy array, optional
            Measurement matrix that transforms the state to estimated
            measurements. The default is None.
        meas_fun_lst : list, optional
            Non-linear functions that return the expected measurement for the
            given state. Each function must have the signature `h(t, x, *args)`.
            The default is None.

        Raises
        ------
        RuntimeError
            Rasied if no arguments are specified.

        Returns
        -------
        None.
        """
        if meas_mat is not None:
            self._meas_mat = meas_mat
        elif meas_fun is not None:
            self._meas_fnc = meas_fun
        else:
            raise RuntimeError("Invalid combination of inputs")

    @property
    def cov(self):
        """Read only covariance of the particles.

        Returns
        -------
        N x N numpy array
            covariance matrix.

        """
        return self._particleDist.covariance

    @cov.setter
    def cov(self, x):
        raise RuntimeError("Covariance is read only")

    @property
    def num_particles(self):
        """Read only number of particles used by the filter.

        Returns
        -------
        int
            Number of particles.
        """
        return self._particleDist.num_particles

    def init_from_dist(self, dist, make_copy=True):
        """Initialize the distribution from a distribution object.

        Parameters
        ----------
        dist : :class:`gncpy.distributions.ParticleDistribution`
            Distribution object to use.
        make_copy : bool, optional
            Flag indicating if a deepcopy of the input distribution should be
            performed. The default is True.

        Returns
        -------
        None.
        """
        if make_copy:
            self._particleDist = deepcopy(dist)
        else:
            self._particleDist = dist

    def extract_dist(self, make_copy=True):
        """Extracts the particle distribution used by the filter.

        Parameters
        ----------
        make_copy : bool, optional
            Flag indicating if a deepcopy of the distribution should be
            performed. The default is True.

        Returns
        -------
        :class:`gncpy.distributions.ParticleDistribution`
            Particle distribution object used by the filter
        """
        if make_copy:
            return deepcopy(self._particleDist)
        else:
            return self._particleDist

    def init_particles(self, particle_lst):
        """Initializes the particle distribution with the given list of points.

        Parameters
        ----------
        particle_lst : list
            List of numpy arrays, one for each particle.
        """
        num_parts = len(particle_lst)
        if num_parts <= 0:
            warn("No particles to initialize. SKIPPING")
            return

        self._particleDist.clear_particles()
        self._particleDist.add_particle(particle_lst, [1.0 / num_parts] * num_parts)

    def _calc_state(self):
        return self._particleDist.mean

    def predict(
        self, timestep, dyn_fun_params=(), sampling_args=(), transition_args=()
    ):
        """Predicts the next state.

        Parameters
        ----------
        timestep : float
            Current timestep.
        dyn_fun_params : tuple, optional
            Extra arguments to be passed to the dynamics function. The default
            is ().
        sampling_args : tuple, optional
            Extra arguments to be passed to the proposal sampling function.
            The default is ().

        Raises
        ------
        RuntimeError
            If no state model is set.

        Returns
        -------
        N x 1 numpy array
            predicted state.

        """
        if self._dyn_obj is not None:
            self.prop_parts = [
                self._dyn_obj.propagate_state(timestep, x, state_args=dyn_fun_params)
                for x in self._particleDist.particles
            ]
            mean = self._dyn_obj.propagate_state(
                timestep, self._particleDist.mean, state_args=dyn_fun_params
            )

        elif self._dyn_fnc is not None:
            self.prop_parts = [
                self._dyn_fnc(timestep, x, *dyn_fun_params)
                for x in self._particleDist.particles
            ]
            mean = self._dyn_fnc(timestep, self._particleDist.mean, *dyn_fun_params)

        else:
            raise RuntimeError("No state model set")

        new_weights = [
            w * self.transition_prob_fnc(x, mean, *transition_args)
            if self.transition_prob_fnc is not None
            else w
            for x, w in zip(self.prop_parts, self._particleDist.weights)
        ]

        new_parts = [
            self.proposal_sampling_fnc(p, self.rng, *sampling_args)
            for p in self.prop_parts
        ]

        self._particleDist.clear_particles()
        for p, w in zip(new_parts, new_weights):
            part = gdistrib.Particle(point=p)
            self._particleDist.add_particle(part, w)

        return self._calc_state()

    def _est_meas(self, timestep, cur_state, n_meas, meas_fun_args):
        if self._meas_fnc is not None:
            est_meas = self._meas_fnc(timestep, cur_state, *meas_fun_args)
        elif self._meas_mat is not None:
            est_meas = self._meas_mat @ cur_state
        else:
            raise RuntimeError("No measurement model set")

        return est_meas

    def _selection(self, unnorm_weights, rel_likeli_in=None):
        new_parts = [None] * self.num_particles
        old_weights = [None] * self.num_particles
        rel_likeli_out = [None] * self.num_particles
        inds_kept = []
        probs = self.rng.random(self.num_particles)
        cumulative_weight = np.cumsum(self._particleDist.weights)
        failed = False
        for ii, r in enumerate(probs):
            inds = np.where(cumulative_weight >= r)[0]
            if inds.size > 0:
                new_parts[ii] = deepcopy(self._particleDist._particles[inds[0]])
                old_weights[ii] = unnorm_weights[inds[0]]
                if rel_likeli_in is not None:
                    rel_likeli_out[ii] = rel_likeli_in[inds[0]]
                if inds[0] not in inds_kept:
                    inds_kept.append(inds[0])
            else:
                failed = True

        if failed:
            tot = np.sum(self._particleDist.weights)
            self._particleDist.clear_particles()
            msg = (
                "Failed to select enough particles, "
                + "check weights (sum = {})".format(tot)
            )
            raise gerr.ParticleDepletionError(msg)

        inds_removed = [
            ii for ii in range(0, self.num_particles) if ii not in inds_kept
        ]

        self._particleDist.clear_particles()
        w = 1 / len(new_parts)
        self._particleDist.add_particle(new_parts, [w] * len(new_parts))

        return inds_removed, old_weights, rel_likeli_out

    def correct(
        self,
        timestep,
        meas,
        meas_fun_args=(),
        meas_likely_args=(),
        proposal_args=(),
        selection=True,
    ):
        """Corrects the state estimate.

        Parameters
        ----------
        timestep : float
            Current timestep.
        meas : Nm x 1 numpy array
            Current measurement.
        meas_fun_args : tuple, optional
            Arguments for the measurement matrix function if one has
            been specified. The default is ().
        meas_likely_args : tuple, optional
            additional agruments for the measurement likelihood function.
            The default is ().
        proposal_args : tuple, optional
            Additional arguments for the proposal distribution function. The
            default is ().
        selection : bool, optional
            Flag indicating if the selection step should be performed. The
            default is True.

        Raises
        ------
        RuntimeError
            If no measurement model is set

        Returns
        -------
        state : N x 1 numpy array
            corrected state.
        rel_likeli : numpy array
            The unnormalized measurement likelihood of each particle.
        inds_removed : list
            each element is an int representing the index of any particles
            that were removed during the selection process.

        """
        # calculate weights
        est_meas = [
            self._est_meas(timestep, p, meas.size, meas_fun_args)
            for p in self._particleDist.particles
        ]
        if self.meas_likelihood_fnc is None:
            rel_likeli = np.ones(len(est_meas))
        else:
            rel_likeli = np.array(
                [self.meas_likelihood_fnc(meas, y, *meas_likely_args) for y in est_meas]
            ).ravel()
        if self.proposal_fnc is None or len(self.prop_parts) == 0:
            prop_fit = np.ones(len(self._particleDist.particles))
        else:
            prop_fit = np.array(
                [
                    self.proposal_fnc(x_hat, cond, meas, *proposal_args)
                    for x_hat, cond in zip(
                        self._particleDist.particles, self.prop_parts
                    )
                ]
            ).ravel()

        inds = np.where(prop_fit < np.finfo(float).eps)[0]
        if inds.size > 0:
            prop_fit[inds] = np.finfo(float).eps
        unnorm_weights = rel_likeli / prop_fit * np.array(self._particleDist.weights)

        tot = np.sum(unnorm_weights)
        if tot > 0 and tot != np.inf:
            weights = unnorm_weights / tot
        else:
            weights = np.inf * np.ones(unnorm_weights.size)
        self._particleDist.update_weights(weights)

        # resample
        if selection:
            inds_removed, rel_likeli = self._selection(
                unnorm_weights, rel_likeli_in=rel_likeli.tolist()
            )[0:3:2]
        else:
            inds_removed = []

        return (self._calc_state(), rel_likeli, inds_removed)

    def plot_particles(
        self,
        inds,
        title="Particle Distribution",
        x_lbl="State",
        y_lbl="Probability",
        **kwargs
    ):
        """Plots the particle distribution.

        This will either plot a histogram for a single index, or plot a 2-d
        heatmap/histogram if a list of 2 indices are given. The 1-d case will
        have the counts normalized to represent the probability.

        Parameters
        ----------
        inds : int or list
            Index of the particle vector to plot.
        title : string, optional
            Title of the plot. The default is 'Particle Distribution'.
        x_lbl : string, optional
            X-axis label. The default is 'State'.
        y_lbl : string, optional
            Y-axis label. The default is 'Probability'.
        **kwargs : dict
            Additional plotting options for :meth:`gncpy.plotting.init_plotting_opts`
            function. Values implemented here are `f_hndl`, `lgnd_loc`, and
            any values relating to title/axis text formatting.

        Returns
        -------
        f_hndl : matplotlib figure
            Figure object the data was plotted on.
        """
        opts = pltUtil.init_plotting_opts(**kwargs)
        f_hndl = opts["f_hndl"]
        lgnd_loc = opts["lgnd_loc"]

        if f_hndl is None:
            f_hndl = plt.figure()
            f_hndl.add_subplot(1, 1, 1)

        h_opts = {"histtype": "stepfilled", "bins": "auto", "density": True}
        if (not isinstance(inds, list)) or len(inds) == 1:
            if isinstance(inds, list):
                ii = inds[0]
            else:
                ii = inds
            x = [p[ii, 0] for p in self._particleDist.particles]
            f_hndl.axes[0].hist(x, **h_opts)
        else:
            x = [p[inds[0], 0] for p in self._particleDist.particles]
            y = [p[inds[1], 0] for p in self._particleDist.particles]
            f_hndl.axes[0].hist2d(x, y)

        pltUtil.set_title_label(f_hndl, 0, opts, ttl=title, x_lbl=x_lbl, y_lbl=y_lbl)
        if lgnd_loc is not None:
            plt.legend(loc=lgnd_loc)
        plt.tight_layout()

        return f_hndl

    def plot_weighted_particles(
        self,
        inds,
        x_lbl="State",
        y_lbl="Weight",
        title="Weighted Particle Distribution",
        **kwargs
    ):
        """Plots the weight vs state distribution of the particles.

        This generates a bar chart and only works for single indices.

        Parameters
        ----------
        inds : int
            Index of the particle vector to plot.
        x_lbl : string, optional
            X-axis label. The default is 'State'.
        y_lbl : string, optional
            Y-axis label. The default is 'Weight'.
        title : string, optional
            Title of the plot. The default is 'Weighted Particle Distribution'.
        **kwargs : dict
            Additional plotting options for :meth:`gncpy.plotting.init_plotting_opts`
            function. Values implemented here are `f_hndl`, `lgnd_loc`, and
            any values relating to title/axis text formatting.

        Returns
        -------
        f_hndl : matplotlib figure
            Figure object the data was plotted on.
        """
        opts = pltUtil.init_plotting_opts(**kwargs)
        f_hndl = opts["f_hndl"]
        lgnd_loc = opts["lgnd_loc"]

        if f_hndl is None:
            f_hndl = plt.figure()
            f_hndl.add_subplot(1, 1, 1)

        if (not isinstance(inds, list)) or len(inds) == 1:
            if isinstance(inds, list):
                ii = inds[0]
            else:
                ii = inds
            x = [p[ii, 0] for p in self._particleDist.particles]
            y = [w for p, w in self._particleDist]
            f_hndl.axes[0].bar(x, y)
        else:
            warn("Only 1 element supported for weighted particle distribution")

        pltUtil.set_title_label(f_hndl, 0, opts, ttl=title, x_lbl=x_lbl, y_lbl=y_lbl)
        if lgnd_loc is not None:
            plt.legend(loc=lgnd_loc)
        plt.tight_layout()

        return f_hndl


class MCMCParticleFilterBase(ParticleFilter):
    """Generic base class for Particle filters with an optional Markov Chain Monte Carlo move step.

    Attributes
    ----------
    use_MCMC : bool, optional
        Flag indicating if the move step is run. The default is False.
    """

    require_copy_can_dist = True

    def __init__(self, use_MCMC=False, **kwargs):
        self.use_MCMC = use_MCMC

        super().__init__(**kwargs)

    @abc.abstractmethod
    def move_particles(self, timestep, meas, old_weights, **kwargs):
        """Generic interface for the movement function.

        This must be overridden in the child class. It is recommended to keep
        the same function signature to allow for standardized wrappers.
        """
        raise RuntimeError("Must implement thid function in derived class")


class MaxCorrEntUKF(UnscentedKalmanFilter):
    """Implements a Maximum Correntropy Unscented Kalman filter.

    Notes
    -----
    This is based on
    :cite:`Hou2018_MaximumCorrentropyUnscentedKalmanFilterforBallisticMissileNavigationSystemBasedonSINSCNSDeeplyIntegratedMode`

    Attributes
    ----------
    kernel_bandwidth : float, optional
        Bandwidth of the Gaussian Kernel. The default is 1.
    """

    def __init__(self, kernel_bandwidth=1, **kwargs):
        self.kernel_bandwidth = kernel_bandwidth

        # for correction/calc_meas_cov wrapper function
        self._past_state = np.array([[]])
        self._cur_state = np.array([[]])
        self._meas = np.array([[]])

        super().__init__(**kwargs)

    def save_filter_state(self):
        """Saves filter variables so they can be restored later."""
        filt_state = super().save_filter_state()

        filt_state["kernel_bandwidth"] = self.kernel_bandwidth
        filt_state["_past_state"] = self._past_state
        filt_state["_cur_state"] = self._cur_state
        filt_state["_meas"] = self._meas

        return filt_state

    def load_filter_state(self, filt_state):
        """Initializes filter using saved filter state.

        Attributes
        ----------
        filt_state : dict
            Dictionary generated by :meth:`save_filter_state`.
        """
        super().load_filter_state(filt_state)

        self.kernel_bandwidth = filt_state["kernel_bandwidth"]
        self._past_state = filt_state["_past_state"]
        self._cur_state = filt_state["_cur_state"]
        self._meas = filt_state["_meas"]

    def _calc_meas_cov(self, timestep, n_meas, meas_fun_args):
        meas_cov, est_points, est_meas = super()._calc_meas_cov(
            timestep, n_meas, meas_fun_args
        )

        # find square root of combined covariance matrix
        n_state = self.cov.shape[0]
        n_meas = est_meas.shape[0]
        z_12 = np.zeros((n_state, n_meas))
        z_21 = np.zeros((n_meas, n_state))
        comb_cov = np.vstack(
            (np.hstack((self.cov, z_12)), np.hstack((z_21, self.meas_noise)))
        )
        comb_cov = (comb_cov + comb_cov.T) * 0.5
        sqrt_comb = la.cholesky(comb_cov)
        inv_sqrt_comb = la.inv(sqrt_comb)

        # find error vector
        pred_meas = self._est_meas(timestep, self._past_state, n_meas, meas_fun_args)[0]
        g = inv_sqrt_comb @ np.vstack((self._past_state, pred_meas))
        d = inv_sqrt_comb @ np.vstack((self._cur_state, self._meas))
        e = (d - g).ravel()

        # kernel function on error
        kern_lst = [gmath.gaussian_kernel(e_ii, self.kernel_bandwidth) for e_ii in e]
        c = np.diag(kern_lst)
        c_inv = la.inv(c)

        # calculate the measurement covariance
        scaled_mat = sqrt_comb @ c_inv @ sqrt_comb.T
        scaled_meas_noise = scaled_mat[n_state:, n_state:]
        meas_cov = meas_cov + scaled_meas_noise

        return meas_cov, est_points, est_meas

    def correct(self, timestep, meas, cur_state, past_state, **kwargs):
        """Correction function for the Max Correntropy UKF.

        This is a wrapper for the parent method to allow for additional
        parameters.

        Parameters
        ----------
        timestep : float
            Current timestep.
        meas : Nm x 1 numpy array
            Current measurement.
        cur_state : N x 1 numpy array
            Current state.
        past_state : N x 1 numpy array
            State from before the prediction step.
        **kwargs : dict, optional
            See the parent function for additional parameters.

        Returns
        -------
        tuple
            See the parent method.
        """
        self._past_state = past_state.copy()
        self._cur_state = cur_state.copy()
        self._meas = meas.copy()

        return super().correct(timestep, meas, cur_state, **kwargs)


class UnscentedParticleFilter(MCMCParticleFilterBase):
    """Implements an unscented particle filter.

    Notes
    -----
    For details on the filter see
    :cite:`VanDerMerwe2000_TheUnscentedParticleFilter` and
    :cite:`VanDerMerwe2001_TheUnscentedParticleFilter`.
    """

    require_copy_prop_parts = False

    def __init__(self, **kwargs):
        self.candDist = None

        self._filt = UnscentedKalmanFilter()

        super().__init__(**kwargs)

    def save_filter_state(self):
        """Saves filter variables so they can be restored later."""
        filt_state = super().save_filter_state()

        filt_state["candDist"] = deepcopy(self.candDist)
        filt_state["_filt"] = self._filt.save_filter_state()

        return filt_state

    def load_filter_state(self, filt_state):
        """Initializes filter using saved filter state.

        Attributes
        ----------
        filt_state : dict
            Dictionary generated by :meth:`save_filter_state`.
        """
        super().load_filter_state(filt_state)

        self.candDist = filt_state["candDist"]
        self._filt.load_filter_state(filt_state["_filt"])

    @property
    def meas_likelihood_fnc(self):
        r"""A function that returns the likelihood of the measurement.

        This has the signature :code:`f(y, y_hat, *args)` where `y` is
        the measurement as an Nm x 1 numpy array, and `y_hat` is the estimated
        measurement.

        Notes
        -----
        This represents :math:`p(y_t \vert x_t)` in the importance
        weight

        .. math::

            w_t \propto \frac{p(y_t \vert x_t) p(x_t \vert x_{t-1})}{q(x_t \vert x_{t-1}, y_t)}

        and has the assumed form :math:`\mathcal{N}(y_t, R)` for measurement
        noise covariance :math:`R`.

        Returns
        -------
        callable
            function to return the measurement likelihood.
        """
        return lambda y, y_hat: stats.multivariate_normal.pdf(
            y.ravel(), y_hat.ravel(), self._filt.meas_noise
        )

    @meas_likelihood_fnc.setter
    def meas_likelihood_fnc(self, val):
        warn("Measuremnet likelihood has an assumed form.")

    @property
    def proposal_fnc(self):
        r"""A function that returns the probability for the proposal distribution.

        This has the signature :code:`f(x_hat, x, y, *args)` where
        `x_hat` is a :class:`gncpy.distributions.Particle` of the estimated
        state, `x` is the particle it is conditioned on, and `y` is the
        measurement.

        Notes
        -----
        This represents :math:`q(x_t \vert x_{t-1}, y_t)` in the importance
        weight

        .. math::

            w_t \propto \frac{p(y_t \vert x_t) p(x_t \vert x_{t-1})}{q(x_t \vert x_{t-1}, y_t)}

        and has the assumed form :math:`\mathcal{N}(\bar{x}_{t}, \hat{P}_t)`

        Returns
        -------
        callable
            function to return the proposal probability.
        """
        return lambda x_hat, part: stats.multivariate_normal.pdf(
            x_hat.ravel(), part.mean.ravel(), part.uncertainty
        )

    @proposal_fnc.setter
    def proposal_fnc(self, val):
        warn("Proposal distribution has an assumed form.")

    @property
    def proposal_sampling_fnc(self):
        r"""A function that returns a random sample from the proposal distribtion.

        This should be consistent with the PDF specified in the
        :meth:`gncpy.filters.ParticleFilter.proposal_fnc`.

        Notes
        -----
        This assumes :math:`x` is drawn from :math:`\mathcal{N}(\bar{x}_{t}, \hat{P}_t)`
        Returns
        -------
        callable
            function to return a random sample.
        """
        return lambda part: self.rng.multivariate_normal(
            part.mean.ravel(), part.uncertainty
        ).reshape(part.mean.shape)

    @proposal_sampling_fnc.setter
    def proposal_sampling_fnc(self, val):
        warn("Proposal sampling has an assumed form.")

    @property
    def transition_prob_fnc(self):
        r"""A function that returns the transition probability for the state.

        This has the signature :code:`f(x_hat, x, cov)` where
        `x_hat` is an N x 1 numpy array representing the propagated state, and
        `x` is the state it is conditioned on.

        Notes
        -----
        This represents :math:`p(x_t \vert x_{t-1})` in the importance
        weight

        .. math::

            w_t \propto \frac{p(y_t \vert x_t) p(x_t \vert x_{t-1})}{q(x_t \vert x_{t-1}, y_t)}

        and has the assumed form :math:`\mathcal{N}(f(x_{t-1}), P_t)` for the
        covariance :math:`P_t`.

        Returns
        -------
        callable
            function to return the transition probability.
        """
        return lambda x_hat, x, cov: stats.multivariate_normal.pdf(
            x_hat.ravel(), x.ravel(), cov
        )

    @transition_prob_fnc.setter
    def transition_prob_fnc(self, val):
        warn("Transistion distribution has an assumed form.")

    # @property
    # def cov(self):
    #     """Read only covariance of the particles.

    #     This is a weighted sum of each particles uncertainty.

    #     Returns
    #     -------
    #     N x N numpy array
    #         covariance matrix.

    #     """
    #     return gmath.weighted_sum_mat(self._particleDist.weights,
    #                                   self._particleDist.uncertainties)

    @property
    def meas_noise(self):
        """Measurement noise matrix.

        This is a wrapper to keep the UPF measurement noise and the internal UKF
        measurement noise synced.

        Returns
        -------
        Nm x Nm numpy array
            measurement noise.
        """
        return self._filt.meas_noise

    @meas_noise.setter
    def meas_noise(self, meas_noise):
        self._filt.meas_noise = meas_noise

    @property
    def proc_noise(self):
        """Process noise matrix.

        This is a wrapper to keep the UPF process noise and the internal UKF
        process noise synced.

        Returns
        -------
        N x N numpy array
            process noise.
        """
        return self._filt.proc_noise

    @proc_noise.setter
    def proc_noise(self, proc_noise):
        self._filt.proc_noise = proc_noise

    def set_state_model(self, **kwargs):
        """Sets the state model for the filter.

        This calls the UKF's set state function
        (:meth:`gncpy.filters.UnscentedKalmanFilter.set_state_model`).
        """
        self._filt.set_state_model(**kwargs)

    def set_measurement_model(self, **kwargs):
        r"""Sets the measurement model for the filter.

        This is a wrapper for the inner UKF's set_measurement model function.
        It is assumed that the measurement model is the same as that of the UKF.
        See :meth:`gncpy.filters.UnscentedKalmanFilter.set_measurement_model`
        for details.
        """
        self._filt.set_measurement_model(**kwargs)

    def _predict_loop(self, timestep, ukf_kwargs, dist):
        newDist = gdistrib.ParticleDistribution()
        num_parts = dist.num_particles
        new_parts = [None] * num_parts
        new_weights = [None] * num_parts
        for ii, (origPart, w) in enumerate(dist):
            part = gdistrib.Particle()
            self._filt.cov = origPart.uncertainty.copy()
            self._filt._stateSigmaPoints = deepcopy(origPart.sigmaPoints)
            part.point = self._filt.predict(timestep, origPart.point, **ukf_kwargs)
            part.uncertainty = self._filt.cov
            part.sigmaPoints = self._filt._stateSigmaPoints
            new_parts[ii] = part
            new_weights[ii] = w

        newDist.add_particle(new_parts, new_weights)
        return newDist

    def predict(self, timestep, ukf_kwargs={}):
        """Prediction step of the UPF.

        This calls the UKF prediction function on every particle.

        Parameters
        ----------
        timestep : float
            Current timestep.
        ukf_kwargs : dict, optional
            Additional arguments to pass to the UKF prediction function. The
            default is {}.

        Returns
        -------
        state : N x 1 numpy array
            The predicted state.
        """
        self._particleDist = self._predict_loop(
            timestep, ukf_kwargs, self._particleDist
        )
        if self.use_MCMC:
            if self.candDist is None:  # first timestep
                self.candDist = deepcopy(self._particleDist)
            else:
                self.candDist = self._predict_loop(timestep, ukf_kwargs, self.candDist)

        return self._calc_state()

    def _inner_correct(self, timestep, meas, state, filt_kwargs):
        """Wrapper so child class can override."""
        return self._filt.correct(timestep, meas, state, **filt_kwargs)

    def _est_meas(self, timestep, cur_state, n_meas, meas_fun_args):
        return self._filt._est_meas(timestep, cur_state, n_meas, meas_fun_args)[0]

    def _correct_loop(self, timestep, meas, ukf_kwargs, dist):
        unnorm_weights = np.nan * np.ones(dist.num_particles)
        new_parts = [None] * dist.num_particles
        rel_likeli = [None] * dist.num_particles
        for ii, (p, w) in enumerate(dist):
            self._filt.cov = p.uncertainty.copy()
            self._filt._stateSigmaPoints = deepcopy(p.sigmaPoints)

            # create a new particle and correct the predicted point with the measurement
            part = gdistrib.Particle()
            part.point, rel_likeli[ii] = self._inner_correct(
                timestep, meas, p.point, ukf_kwargs
            )
            part.uncertainty = self._filt.cov

            # draw a sample from the proposal distribution using the corrected point
            samp = self.proposal_sampling_fnc(part)

            # transition probability of the sample given the predicted point
            trans_prob = self.transition_prob_fnc(samp, p.point, p.uncertainty)

            # probability of the sampled value given the corrected value
            proposal_prob = self.proposal_fnc(samp, part)

            # get new weight
            if proposal_prob < np.finfo(float).eps:
                proposal_prob = np.finfo(float).eps
            unnorm_weights[ii] = rel_likeli[ii] * trans_prob / proposal_prob

            # update the new particle with the sampled point
            part.point = samp
            part.sigmaPoints = self._filt._stateSigmaPoints
            part.sigmaPoints.update_points(part.point, part.uncertainty)
            new_parts[ii] = part

        # update info for next UKF
        newDist = gdistrib.ParticleDistribution()
        tot = np.sum(unnorm_weights)
        # protect against divide by 0
        if tot < np.finfo(float).eps:
            tot = np.finfo(float).eps
        newDist.add_particle(new_parts, (unnorm_weights / tot).tolist())

        return newDist, rel_likeli, unnorm_weights

    def correct(self, timestep, meas, ukf_kwargs={}, move_kwargs={}):
        """Correction step of the UPF.

        This optionally can perform a MCMC move step as well.

        Parameters
        ----------
        timestep : float
            Current timestep.
        meas : Nm x 1 numpy array
            measurement.
        ukf_kwargs : dict, optional
            Additional arguments to pass to the UKF correction function. The
            default is {}.
        move_kwargs : dict, optional
            Additional arguments to pass to the movement function.
            The default is {}.

        Returns
        -------
        state : N x 1 numpy array
            corrected state.
        rel_likeli : list
            each element is a float representing the relative likelihood of the
            particles (unnormalized).
        inds_removed : list
            each element is an int representing the index of any particles
            that were removed during the selection process.
        """
        # if first timestep and have not called predict yet
        if self.use_MCMC and self.candDist is None:
            self.candDist = deepcopy(self._particleDist)

        # call UKF correction on each particle
        (self._particleDist, rel_likeli, unnorm_weights) = self._correct_loop(
            timestep, meas, ukf_kwargs, self._particleDist
        )
        if self.use_MCMC:
            (self.candDist, can_rel_likeli, can_unnorm_weights) = self._correct_loop(
                timestep, meas, ukf_kwargs, self.candDist
            )

        # perform selection/resampling
        (inds_removed, old_weights, rel_likeli) = self._selection(
            unnorm_weights, rel_likeli_in=rel_likeli
        )

        # optionally move particles
        if self.use_MCMC:
            rel_likeli = self.move_particles(
                timestep,
                meas,
                old_weights,
                rel_likeli,
                can_unnorm_weights,
                can_rel_likeli,
                **move_kwargs
            )

        return (self._calc_state(), rel_likeli, inds_removed)

    def move_particles(
        self, timestep, meas, old_weights, old_likeli, can_weight, can_likeli
    ):
        r"""Movement function for the MCMC move step.

        This modifies the internal particle distribution but does not return a
        value.

        Notes
        -----
        This implements a metropolis-hastings algorithm.

        Parameters
        ----------
        timestep : float
            Current timestep.
        meas : Nm x 1 numpy array
            measurement.
        old_weights : :class:`gncpy.distributions.ParticleDistribution`
            Distribution before the measurement correction.

        Returns
        -------
        None.
        """
        accept_prob = self.rng.random()
        num_parts = self._particleDist.num_particles
        new_parts = [None] * num_parts
        new_likeli = [None] * num_parts
        for ii, (can, exp, ex_weight, can_weight, ex_like, can_like) in enumerate(
            zip(
                self.candDist,
                self._particleDist,
                old_weights,
                can_weight,
                old_likeli,
                can_likeli,
            )
        ):

            if accept_prob <= np.min([1, can_weight / ex_weight]):
                # accept move
                new_parts[ii] = deepcopy(can[0])
                new_likeli[ii] = can_like
            else:
                # reject move
                new_parts[ii] = exp[0]
                new_likeli[ii] = ex_like

        self._particleDist.clear_particles()
        self._particleDist.add_particle(new_parts, [1 / num_parts] * num_parts)

        return new_likeli


class MaxCorrEntUPF(UnscentedParticleFilter):
    """Implements a Maximum Correntropy Unscented Particle Filter.

    Notes
    -----
    This is based on
    :cite:`Fan2018_MaximumCorrentropyBasedUnscentedParticleFilterforCooperativeNavigationwithHeavyTailedMeasurementNoises`

    """

    def __init__(self, **kwargs):
        self._past_state = np.array([[]])

        super().__init__(**kwargs)
        self._filt = MaxCorrEntUKF()

    def save_filter_state(self):
        """Saves filter variables so they can be restored later."""
        filt_state = super().save_filter_state()

        filt_state["_past_state"] = self._past_state

        return filt_state

    def load_filter_state(self, filt_state):
        """Initializes filter using saved filter state.

        Attributes
        ----------
        filt_state : dict
            Dictionary generated by :meth:`save_filter_state`.
        """
        super().load_filter_state(filt_state)

        self._past_state = filt_state["_past_state"]

    def _inner_correct(self, timestep, meas, state, filt_kwargs):
        """Wrapper so child class can override."""
        return self._filt.correct(
            timestep, meas, state, self._past_state, **filt_kwargs
        )

    def correct(self, timestep, meas, past_state, **kwargs):
        """Correction step of the MCUPF.

        This is a wrapper for the parent method to allow for an additional
        parameter.

        Parameters
        ----------
        timestep : float
            Current timestep.
        meas : Nm x 1 numpy array
            Current measurement.
        past_state : N x 1 numpy array
            State from before the prediction step.
        **kwargs : dict
            See the parent method.

        Returns
        -------
        tuple
            See the parent method.
        """
        self._past_state = past_state
        return super().correct(timestep, meas, **kwargs)

    @property
    def kernel_bandwidth(self):
        """Bandwidth for the Gaussian Kernel in the MCUKF.

        Returns
        -------
        float
            bandwidth
        """
        return self._filt.kernel_bandwidth

    @kernel_bandwidth.setter
    def kernel_bandwidth(self, kernel_bandwidth):
        self._filt.kernel_bandwidth = kernel_bandwidth


class QuadratureKalmanFilter(KalmanFilter):
    """Implementation of a Quadrature Kalman Filter.

    Notes
    -----
    This implementation is based on
    :cite:`Arasaratnam2007_DiscreteTimeNonlinearFilteringAlgorithmsUsingGaussHermiteQuadrature`
    and uses Gauss-Hermite quadrature points.

    Attributes
    ----------
    quadPoints : :class:`gncpy.distributions.QuadraturePoints`
        Quadrature points used by the filter.
    """

    def __init__(self, points_per_axis=None, **kwargs):
        super().__init__(**kwargs)

        self.quadPoints = gdistrib.QuadraturePoints(points_per_axis=points_per_axis)
        self._sqrt_cov = np.array([[]])
        self._dyn_fnc = None

    def save_filter_state(self):
        """Saves filter variables so they can be restored later."""
        filt_state = super().save_filter_state()

        filt_state["quadPoints"] = self.quadPoints

        filt_state["_sqrt_cov"] = self._sqrt_cov
        filt_state["_dyn_fnc"] = self._dyn_fnc

        return filt_state

    def load_filter_state(self, filt_state):
        """Initializes filter using saved filter state.

        Attributes
        ----------
        filt_state : dict
            Dictionary generated by :meth:`save_filter_state`.
        """
        super().load_filter_state(filt_state)

        self.quadPoints = filt_state["quadPoints"]
        self._sqrt_cov = filt_state["_sqrt_cov"]
        self._dyn_fnc = filt_state["_dyn_fnc"]

    @property
    def points_per_axis(self):
        """Wrapper for the  number of quadrature points per axis."""
        return self.quadPoints.points_per_axis

    @points_per_axis.setter
    def points_per_axis(self, val):
        self.quadPoints.points_per_axis = val

    def set_state_model(self, dyn_fun=None, **kwargs):
        """Sets the state model.

        This can either be a non-linear dyanmic function or any valid model
        for the :class:`.KalmanFilter`.

        Parameters
        ----------
        dyn_fun : callable
            The a function that propagates the state. This is assumed to be
            non-linear but can also be a linear function. It must have the
            signature :code:`f(t, x, *args)` where `t` is the timestep,
            `x` is a N x 1 numpy array, and it returns an N x 1 numpy array
            representing the next state.
        **kwargs : dict
            Additional agruments to pass to :class:`.KalmanFilter` if `dyn_fun`
            is not used.
        """
        if dyn_fun is not None:
            self._dyn_fnc = dyn_fun
        else:
            super().set_state_model(**kwargs)

    def set_measurement_model(self, meas_mat=None, meas_fun=None):
        r"""Sets the measurement model for the filter.

        This can either set the constant measurement matrix, or a potentially
        non-linear function.

        Notes
        -----
        This assumes a measurement model of the form

        .. math::
            \tilde{y}_{k+1} = H x_{k+1}^-

        for the measurement matrix case. Or of the form

        .. math::
            \tilde{y}_{k+1} = h(t, x_{k+1}^-)

        for the potentially non-linear case.

        Parameters
        ----------
        meas_mat : Nm x N numpy array, optional
            Measurement matrix that transforms the state to estimated
            measurements. The default is None.
        meas_fun : callable, optional
            Function that transforms the state to estimated measurements. Must
            have the signature :code:`h(t, x, *args)` where `t` is the timestep,
            `x` is an N x 1 numpy array of the current state, and return an
            Nm x 1 numpy array of the estimated measurement. The default is None.

        Raises
        ------
        RuntimeError
            Rasied if no arguments are specified.
        """
        super().set_measurement_model(meas_mat=meas_mat, meas_fun=meas_fun)

    def _factorize_cov(self, val=None):
        if val is None:
            val = self.cov
        # numpy linalg is lower triangular
        self._sqrt_cov = la.cholesky(val)

    def _pred_update_cov(self):
        self.cov = self.proc_noise + self.quadPoints.cov

    def _predict_next_state(
        self, timestep, cur_state, cur_input, state_mat_args, input_mat_args
    ):
        if self._dyn_fnc is not None:
            return self._dyn_fnc(timestep, cur_state, *state_mat_args)
        else:
            return super()._predict_next_state(
                timestep, cur_state, cur_input, state_mat_args, input_mat_args
            )[0]

    def predict(
        self, timestep, cur_state, cur_input=None, state_mat_args=(), input_mat_args=()
    ):
        """Prediction step of the filter.

        Parameters
        ----------
        timestep : float
            Current timestep.
        cur_state : N x 1 numpy array
            Current state.
        cur_input : N x Nu numpy array, optional
            Current input. The default is None.
        state_mat_args : tuple, optional
            Additional arguments for the get state matrix function if one has
            been specified, the propagate state function if using a dynamic
            object, or the dynamic function is a non-linear model is used.
            The default is ().
        input_mat_args : tuple, optional
            Additional arguments for the get input matrix function if one has
            been specified or the propagate state function if using a dynamic
            object. The default is ().

        Raises
        ------
        RuntimeError
            If the state model has not been set

        Returns
        -------
        N x 1 numpy array
            The predicted state.
        """
        # factorize covariance as P = sqrt(P) * sqrt(P)^T
        self._factorize_cov()

        # generate quadrature points as X_i = sqrt(P) * xi_i + x_hat for m points
        self.quadPoints.update_points(cur_state, self._sqrt_cov, have_sqrt=True)

        # predict each point using the dynamics
        for ii, (point, _) in enumerate(self.quadPoints):
            pred_point = self._predict_next_state(
                timestep, point, cur_input, state_mat_args, input_mat_args
            )
            self.quadPoints.points[ii, :] = pred_point.ravel()

        # update covariance as Q - m * x * x^T + sum(w_i * X_i * X_i^T)
        self._pred_update_cov()

        return self.quadPoints.mean

    def _corr_update_cov(self, gain, inov_cov):
        self.cov = self.cov - gain @ inov_cov @ gain.T
        self.cov = 0.5 * (self.cov + self.cov.T)

    def _est_meas(self, timestep, cur_state, n_meas, meas_fun_args):
        if self._meas_fnc is not None:
            return self._meas_fnc(timestep, cur_state, *meas_fun_args).ravel()
        else:
            return (
                super()._est_meas(timestep, cur_state, n_meas, meas_fun_args)[0].ravel()
            )

    def _corr_core(self, timestep, cur_state, meas, meas_fun_args):
        # factorize covariance as P = sqrt(P) * sqrt(P)^T
        self._factorize_cov()

        # generate quadrature points as X_i = sqrt(P) * xi_i + x_hat for m points
        self.quadPoints.update_points(cur_state, self._sqrt_cov, have_sqrt=True)

        # Estimate a measurement for each quad point, Z_i
        measQuads = gdistrib.QuadraturePoints(num_axes=meas.size)
        measQuads.points = np.nan * np.ones(
            (self.quadPoints.points.shape[0], meas.size)
        )
        measQuads.weights = self.quadPoints.weights
        for ii, (point, _) in enumerate(self.quadPoints):
            measQuads.points[ii, :] = self._est_meas(
                timestep, point, meas.size, meas_fun_args
            )

        # estimate predicted measurement as sum of est measurement quad points
        est_meas = measQuads.mean

        return measQuads, est_meas

    def correct(self, timestep, meas, cur_state, meas_fun_args=()):
        """Implements the correction step of the filter.

        Parameters
        ----------
        timestep : float
            Current timestep.
        meas : Nm x 1 numpy array
            Current measurement.
        cur_state : N x 1 numpy array
            Current state.
        meas_fun_args : tuple, optional
            Arguments for the measurement matrix function if one has
            been specified. The default is ().

        Raises
        ------
        :class:`.errors.ExtremeMeasurementNoiseError`
            If estimating the measurement noise and the measurement fit calculation fails.
        LinAlgError
            Numpy exception raised if not estimating noise and measurement fit fails.

        Returns
        -------
        next_state : N x 1 numpy array
            The corrected state.
        meas_fit_prob : float
            Goodness of fit of the measurement based on the state and
            covariance assuming Gaussian noise.

        """
        measQuads, est_meas = self._corr_core(timestep, cur_state, meas, meas_fun_args)

        # estimate the measurement noise online if applicable
        if self._est_meas_noise_fnc is not None:
            self.meas_noise = self._est_meas_noise_fnc(est_meas, measQuads.cov)

        # estimate innovation cov as P_zz = R - m * z_hat * z_hat^T + sum(w_i * Z_i * Z_i^T)
        inov_cov = self.meas_noise + measQuads.cov

        if self.use_cholesky_inverse:
            sqrt_inv_inov_cov = la.inv(la.cholesky(inov_cov))
            inv_inov_cov = sqrt_inv_inov_cov.T @ sqrt_inv_inov_cov
        else:
            inv_inov_cov = la.inv(inov_cov)

        # estimate cross cov as P_xz = sum(w_i * X_i * Z_i^T) - m * x * z_hat^T
        cov_lst = [None] * self.quadPoints.num_points
        for ii, (qp, mp) in enumerate(zip(self.quadPoints, measQuads)):
            cov_lst[ii] = (qp[0] - cur_state) @ (mp[0] - est_meas).T
        cross_cov = gmath.weighted_sum_mat(self.quadPoints.weights, cov_lst)

        # calc Kalman gain as K = P_xz * P_zz^-1
        gain = cross_cov @ inv_inov_cov

        # state is x_hat + K *(z - z_hat)
        innov = meas - est_meas
        cor_state = cur_state + gain @ innov

        # update covariance as P = P_k - K * P_zz * K^T
        self._corr_update_cov(gain, inov_cov)

        meas_fit_prob = self._calc_meas_fit(meas, est_meas, inov_cov)

        return (cor_state, meas_fit_prob)

    def plot_quadrature(self, inds, **kwargs):
        """Wrapper function for :meth:`gncpy.distributions.QuadraturePoints.plot_points`."""
        return self.quadPoints.plot_points(inds, **kwargs)


class SquareRootQKF(QuadratureKalmanFilter):
    """Implementation of a Square root Quadrature Kalman Filter (SQKF).

    Notes
    -----
    This is based on :cite:`Arasaratnam2008_SquareRootQuadratureKalmanFiltering`.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._meas_noise = np.array([[]])
        self._sqrt_p_noise = np.array([[]])
        self._sqrt_m_noise = np.array([[]])

    def save_filter_state(self):
        """Saves filter variables so they can be restored later."""
        filt_state = super().save_filter_state()

        filt_state["_meas_noise"] = self._meas_noise
        filt_state["_sqrt_p_noise"] = self._sqrt_p_noise
        filt_state["_sqrt_m_noise"] = self._sqrt_m_noise

        return filt_state

    def load_filter_state(self, filt_state):
        """Initializes filter using saved filter state.

        Attributes
        ----------
        filt_state : dict
            Dictionary generated by :meth:`save_filter_state`.
        """
        super().load_filter_state(filt_state)

        self._meas_noise = filt_state["_meas_noise"]
        self._sqrt_p_noise = filt_state["_sqrt_p_noise"]
        self._sqrt_m_noise = filt_state["_sqrt_m_noise"]

    def set_measurement_noise_estimator(self, function):
        """Sets the model used for estimating the measurement noise parameters.

        This is an optional step and the filter will work properly if this is
        not called. If it is called, the measurement noise will be estimated
        during the filter's correction step and the measurement noise attribute
        will not be used.

        Parameters
        ----------
        function : callable
            A function that implements the prediction and correction steps for
            an appropriate filter to estimate the measurement noise covariance
            matrix. It must have the signature `f(est_meas)` where `est_meas`
            is an Nm x 1 numpy array and it must return an Nm x Nm numpy array
            representing the measurement noise covariance matrix.

        Returns
        -------
        None.
        """
        self._est_meas_noise_fnc = function

    @property
    def cov(self):
        """Covariance of the filter."""
        # sqrt cov is lower triangular
        return self._sqrt_cov @ self._sqrt_cov.T

    @cov.setter
    def cov(self, val):
        if val.size == 0:
            self._sqrt_cov = val
        else:
            super()._factorize_cov(val=val)

    @property
    def proc_noise(self):
        """Process noise of the filter."""
        return self._sqrt_p_noise @ self._sqrt_p_noise.T

    @proc_noise.setter
    def proc_noise(self, val):
        if val.size == 0 or np.all(val == 0):
            self._sqrt_p_noise = val
        else:
            self._sqrt_p_noise = la.cholesky(val)

    @property
    def meas_noise(self):
        """Measurement noise of the filter."""
        return self._sqrt_m_noise @ self._sqrt_m_noise.T

    @meas_noise.setter
    def meas_noise(self, val):
        if val.size == 0 or np.all(val == 0):
            self._sqrt_m_noise = val
        else:
            self._sqrt_m_noise = la.cholesky(val)

    def _factorize_cov(self):
        pass

    def _pred_update_cov(self):
        weight_mat = np.diag(np.sqrt(self.quadPoints.weights))
        x_hat = self.quadPoints.mean
        state_mat = np.concatenate(
            [x.reshape((x.size, 1)) - x_hat for x in self.quadPoints.points], axis=1
        )

        self._sqrt_cov = la.qr(
            np.concatenate((state_mat @ weight_mat, self._sqrt_p_noise.T), axis=1).T,
            mode="r",
        ).T

    def _corr_update_cov(self, gain, state_mat, meas_mat):
        self._sqrt_cov = la.qr(
            np.concatenate(
                (state_mat - gain @ meas_mat, gain @ self._sqrt_m_noise), axis=1
            ).T,
            mode="r",
        ).T

    def correct(self, timestep, meas, cur_state, meas_fun_args=()):
        """Implements the correction step of the filter.

        Parameters
        ----------
        timestep : float
            Current timestep.
        meas : Nm x 1 numpy array
            Current measurement.
        cur_state : N x 1 numpy array
            Current state.
        meas_fun_args : tuple, optional
            Arguments for the measurement matrix function if one has
            been specified. The default is ().

        Raises
        ------
        :class:`.errors.ExtremeMeasurementNoiseError`
            If estimating the measurement noise and the measurement fit calculation fails.
        LinAlgError
            Numpy exception raised if not estimating noise and measurement fit fails.

        Returns
        -------
        next_state : N x 1 numpy array
            The corrected state.
        meas_fit_prob : float
            Goodness of fit of the measurement based on the state and
            covariance assuming Gaussian noise.

        """
        measQuads, est_meas = self._corr_core(timestep, cur_state, meas, meas_fun_args)

        weight_mat = np.diag(np.sqrt(self.quadPoints.weights))

        # calculate sqrt of the measurement covariance
        meas_mat = (
            np.concatenate(
                [z.reshape((z.size, 1)) - est_meas for z in measQuads.points], axis=1
            )
            @ weight_mat
        )
        if self._est_meas_noise_fnc is not None:
            self.meas_noise = self._est_meas_noise_fnc(est_meas, meas_mat @ meas_mat.T)

        sqrt_inov_cov = la.qr(
            np.concatenate((meas_mat, self._sqrt_m_noise), axis=1).T, mode="r"
        ).T

        # calculate cross covariance
        x_hat = self.quadPoints.mean
        state_mat = (
            np.concatenate(
                [x.reshape((x.size, 1)) - x_hat for x in self.quadPoints.points], axis=1
            )
            @ weight_mat
        )
        cross_cov = state_mat @ meas_mat.T

        # calculate gain
        inter = sla.solve_triangular(sqrt_inov_cov.T, cross_cov.T)
        gain = sla.solve_triangular(sqrt_inov_cov, inter, lower=True).T

        # the above gain is equavalent to
        # inv_sqrt_inov_cov = la.inv(sqrt_inov_cov)
        # gain = cross_cov @ (inv_sqrt_inov_cov.T @ inv_sqrt_inov_cov)

        # state is x_hat + K *(z - z_hat)
        innov = meas - est_meas
        cor_state = cur_state + gain @ innov

        # update covariance
        inov_cov = sqrt_inov_cov @ sqrt_inov_cov.T

        self._corr_update_cov(gain, state_mat, meas_mat)

        meas_fit_prob = self._calc_meas_fit(meas, est_meas, inov_cov)

        return (cor_state, meas_fit_prob)


class _GSMProcNoiseEstimator:
    """Helper class for estimating proc noise in the GSM filters."""

    def __init__(self):
        self.q_fifo = deque([], None)
        self.b_fifo = deque([], None)

        self.startup_delay = 1

        self._last_q_hat = None
        self._call_count = 0

    @property
    def maxlen(self):
        return self.q_fifo.maxlen

    @maxlen.setter
    def maxlen(self, val):
        self.q_fifo = deque([], val)
        self.b_fifo = deque([], val)

    @property
    def win_len(self):
        return len(self.q_fifo)

    def estimate_next(self, cur_est, pred_state, pred_cov, cor_state, cor_cov):
        self._call_count += 1

        # update bk
        bk = pred_cov - cur_est - cor_cov
        if self.b_fifo.maxlen is None or len(self.b_fifo) < self.b_fifo.maxlen:
            bk_last = np.zeros(bk.shape)
        else:
            bk_last = self.b_fifo.pop()
        self.b_fifo.appendleft(bk)

        # update qk and qk_hat
        qk = cor_state - pred_state
        if self._last_q_hat is None:
            self._last_q_hat = np.zeros(qk.shape)

        if self.q_fifo.maxlen is None or len(self.q_fifo) < self.q_fifo.maxlen:
            qk_last = np.zeros(qk.shape)
        else:
            qk_last = self.q_fifo.pop()
        self.q_fifo.appendleft(qk)

        qk_klast_diff = qk - qk_last
        win_len = self.win_len
        inv_win_len = 1 / win_len
        win_len_m1 = win_len - 1
        if self._call_count <= np.max([1, self.startup_delay]):
            return cur_est

        self._last_q_hat += inv_win_len * qk_klast_diff

        # estimate cov
        qk_khat_diff = qk - self._last_q_hat
        qklast_khat_diff = qk_last - self._last_q_hat
        bk_diff = bk_last - bk
        next_est = cur_est + 1 / win_len_m1 * (
            qk_khat_diff @ qk_khat_diff.T
            - qklast_khat_diff @ qklast_khat_diff.T
            + inv_win_len * qk_klast_diff @ qk_klast_diff.T
            + win_len_m1 * inv_win_len * bk_diff
        )

        # for numerical reasons
        for ii in range(next_est.shape[0]):
            next_est[ii, ii] = np.abs(next_est[ii, ii])

        return next_est


class GSMFilterBase(BayesFilter):
    """Base implementation of a Gaussian Scale Mixture (GSM) filter.

    This should be inherited from to include specific implementations of the
    core filter by exposing the necessary core filter attributes.

    Notes
    -----
    This is based on a generic version of
    :cite:`VilaValls2012_NonlinearBayesianFilteringintheGaussianScaleMixtureContext`
    which extends :cite:`VilaValls2011_BayesianFilteringforNonlinearStateSpaceModelsinSymmetricStableMeasurementNoise`.
    This class does not implement a specific core filter, that is up to the
    child classes.

    Attributes
    ----------
    enable_proc_noise_estimation : bool
        Flag indicating if the process noise should be estimated.
    """

    def __init__(self, enable_proc_noise_estimation=False, **kwargs):
        """Initializes the class.

        Parameters
        ----------
        enable_proc_noise_estimation : bool, optional
            Flag indicating if the process noise should be estimated. The
            default is False.
        **kwargs : dict
            Additional keyword arguements for the parent class.
        """
        super().__init__(**kwargs)

        self.enable_proc_noise_estimation = enable_proc_noise_estimation

        self._coreFilter = None
        self._meas_noise_filters = []
        self._import_w_factory_lst = None
        self._procNoiseEstimator = _GSMProcNoiseEstimator()

    def save_filter_state(self):
        """Saves filter variables so they can be restored later.

        Note that to pickle the resulting dictionary the :code:`dill` package
        may need to be used due to potential pickling of functions.
        """
        filt_state = super().save_filter_state()

        filt_state["enable_proc_noise_estimation"] = self.enable_proc_noise_estimation

        if self._coreFilter is not None:
            filt_state["_coreFilter"] = (
                type(self._coreFilter),
                self._coreFilter.save_filter_state(),
            )
        else:
            filt_state["_coreFilter"] = (None, self._coreFilter)

        filt_state["_meas_noise_filters"] = [
            (type(f), f.save_filter_state()) for f in self._meas_noise_filters
        ]

        filt_state["_import_w_factory_lst"] = self._import_w_factory_lst
        filt_state["_procNoiseEstimator"] = self._procNoiseEstimator

        return filt_state

    def load_filter_state(self, filt_state):
        """Initializes filter using saved filter state.

        Attributes
        ----------
        filt_state : dict
            Dictionary generated by :meth:`save_filter_state`.
        """
        super().load_filter_state(filt_state)

        self.enable_proc_noise_estimation = filt_state["enable_proc_noise_estimation"]

        cls_type = filt_state["_coreFilter"][0]
        if cls_type is not None:
            self._coreFilter = cls_type()
            self._coreFilter.load_filter_state(filt_state["_coreFilter"][1])
        else:
            self._coreFilter = None

        num_m_filts = len(filt_state["_meas_noise_filters"])
        self._meas_noise_filters = [None] * num_m_filts
        for ii, (cls_type, vals) in enumerate(filt_state["_meas_noise_filters"]):
            if cls_type is not None:
                self._meas_noise_filters[ii] = cls_type()
                self._meas_noise_filters[ii].load_filter_state(vals)

        self._import_w_factory_lst = filt_state["_import_w_factory_lst"]
        self._procNoiseEstimator = filt_state["_procNoiseEstimator"]

    def set_state_model(self, **kwargs):
        """Wrapper for the core filters set state model function."""
        if self._coreFilter is not None:
            self._coreFilter.set_state_model(**kwargs)
        else:
            warn("Core filter is not set, use an inherited class.", RuntimeWarning)

    def set_measurement_model(self, **kwargs):
        """Wrapper for the core filters set measurement model function."""
        if self._coreFilter is not None:
            self._coreFilter.set_measurement_model(**kwargs)
        else:
            warn("Core filter is not set, use an inherited class.", RuntimeWarning)

    def _define_student_t_pf(self, gsm, rng, num_parts):
        def import_w_factory(inov_cov):
            def import_w_fnc(meas, parts):
                stds = np.sqrt(parts[:, 2] * parts[:, 1] ** 2 + inov_cov)
                return np.array(
                    [stats.norm.pdf(meas.item(), scale=scale) for scale in stds]
                )

            return import_w_fnc

        def gsm_import_dist_factory():
            def import_dist_fnc(parts, _rng):
                new_parts = np.nan * np.ones(parts.particles.shape)

                disc = 0.99
                a = (3 * disc - 1) / (2 * disc)
                h = np.sqrt(1 - a ** 2)
                last_means = np.mean(parts.particles, axis=0)
                means = a * parts.particles[:, 0:2] + (1 - a) * last_means[0:2]

                # df, sig
                for ind in range(means.shape[1]):
                    std = np.sqrt(h ** 2 * np.cov(parts.particles[:, ind]))

                    for ii, m in enumerate(means):
                        samp = stats.norm.rvs(loc=m[ind], scale=std, random_state=_rng)
                        new_parts[ii, ind] = samp

                df = np.mean(new_parts[:, 0])
                if df < 0:
                    msg = "Degree of freedom must be > 0 {:.4f}".format(df)
                    raise gerr.ParticleEstimationDomainError(msg)
                new_parts[:, 2] = stats.invgamma.rvs(
                    df / 2,
                    scale=1 / (2 / df),
                    random_state=_rng,
                    size=new_parts.shape[0],
                )
                return new_parts

            return import_dist_fnc

        pf = BootstrapFilter()
        pf.importance_dist_fnc = gsm_import_dist_factory()
        pf.particleDistribution = gdistrib.SimpleParticleDistribution()

        df_scale = gsm.df_range[1] - gsm.df_range[0]
        df_loc = gsm.df_range[0]
        df_particles = stats.uniform.rvs(
            loc=df_loc, scale=df_scale, size=num_parts, random_state=rng
        )

        sig_scale = gsm.scale_range[1] - gsm.scale_range[0]
        sig_loc = gsm.scale_range[0]
        sig_particles = stats.uniform.rvs(
            loc=sig_loc, scale=sig_scale, size=num_parts, random_state=rng
        )

        z_particles = np.nan * np.ones(num_parts)
        for ii, v in enumerate(df_particles):
            z_particles[ii] = stats.invgamma.rvs(
                v / 2, scale=1 / (2 / v), random_state=rng
            )

        pf.particleDistribution.particles = np.stack(
            (df_particles, sig_particles, z_particles), axis=1
        )
        pf.particleDistribution.num_parts_per_ind = np.ones(num_parts)
        pf.particleDistribution.weights = 1 / num_parts * np.ones(num_parts)
        pf.rng = rng

        return pf, import_w_factory

    def _define_cauchy_pf(self, gsm, rng, num_parts):
        def import_w_factory(inov_cov):
            def import_w_fnc(meas, parts):
                stds = np.sqrt(parts[:, 1] * parts[:, 0] ** 2 + inov_cov)
                return np.array(
                    [stats.norm.pdf(meas.item(), scale=scale) for scale in stds]
                )

            return import_w_fnc

        def gsm_import_dist_factory():
            def import_dist_fnc(parts, _rng):
                new_parts = np.nan * np.ones(parts.particles.shape)

                disc = 0.99
                a = (3 * disc - 1) / (2 * disc)
                h = np.sqrt(1 - a ** 2)
                last_means = np.mean(parts.particles, axis=0)
                means = a * parts.particles[:, 0] + (1 - a) * last_means[0]

                # df, sig
                for ind in range(means.shape[1]):
                    std = np.sqrt(h ** 2 * np.cov(parts.particles[:, ind]))

                    for ii, m in enumerate(means):
                        samp = stats.norm.rvs(loc=m[ind], scale=std, random_state=_rng)
                        new_parts[ii, ind] = samp

                new_parts[:, 1] = stats.invgamma.rvs(
                    1 / 2, scale=1 / 2, random_state=_rng, size=new_parts.shape[0]
                )
                return new_parts

            return import_dist_fnc

        pf = BootstrapFilter()
        pf.importance_dist_fnc = gsm_import_dist_factory()
        pf.particleDistribution = gdistrib.SimpleParticleDistribution()

        sig_scale = gsm.scale_range[1] - gsm.scale_range[0]
        sig_loc = gsm.scale_range[0]
        sig_particles = stats.uniform.rvs(
            loc=sig_loc, scale=sig_scale, size=num_parts, random_state=rng
        )

        z_particles = stats.invgamma.rvs(
            1 / 2, scale=1 / 2, random_state=rng, size=num_parts
        )

        pf.particleDistribution.particles = np.stack(
            (sig_particles, z_particles), axis=1
        )
        pf.particleDistribution.num_parts_per_ind = np.ones(num_parts)
        pf.particleDistribution.weights = 1 / num_parts * np.ones(num_parts)
        pf.rng = rng

        return pf, import_w_factory

    def set_meas_noise_model(
        self,
        bootstrap_lst=None,
        importance_weight_factory_lst=None,
        gsm_lst=None,
        num_parts=None,
        rng=None,
    ):
        """Initializes the measurement noise estimators.

        The filters and importance
        weight factories can be provided or a list of
        :class:`serums.models.GaussianScaleMixture` objecst, list of particles,
        and a random number generator. If the latter set is given then
        bootstrap filters are constructed automatically. The recommended way
        of specifying the filters is to provide GSM objects.

        Notes
        -----
        This uses independent bootstrap particle filters for each measurement
        based on the provided model information.

        Parameters
        ----------
        bootstrap_lst : list, optional
            List of :class:`.BootstrapFilter` objects that have already been
            initialized. If given then importance weight factory list must also
            be given and of the same length. The default is None.
        importance_weight_factory_lst : list, optional
            List of callables, each takes a 1 x 1 numpy array or float as input
            and returns a callable with the signature `f(z, parts)` where
            `z` is a float that is the difference between the estimated measurement
            and actual measurement (so the distribution is 0 mean) and `parts` is
            a numpy array of all the particles from the bootrap filter `f` must
            return a numpy array of weights for each particle. See the
            :attr:`.BootstrapFilter.importance_weight_fnc` for more details. The
            default is None.
        gsm_lst : list, optional
            List of :class:`serums.models.GaussianScaleMixture` objects, one
            per measurement. Requires `num_parts` also be specified and optionally
            `rng`. The default is None.
        num_parts : list or int, optional
            Number of particles to use in each automatically constructed filter.
            If only one number is supplied all filters will use the same number
            of particles. The default is None.
        rng : numpy random generator, optional
            Random number generator to use in constructed filters. If supplied,
            each filter uses the same instance, otherwise a new generator is
            created for each filter using numpy's default initialization routine
            with no supplied seed. Only used if a `gsm_lst` is supplied.

        Raises
        ------
        RuntimeError
            If an incorrect combination of input arguments is provided.

        Todo
        ----
        Allow for GSM Object to use location parameter when specifying noise models
        """
        if bootstrap_lst is not None:
            self._meas_noise_filters = bootstrap_lst
            if importance_weight_factory_lst is not None:
                if not len(importance_weight_factory_lst) == len(bootstrap_lst):
                    msg = (
                        "Importance weight factory list "
                        + "length ({:d}) ".format(len(importance_weight_factory_lst))
                        + "does not match the number of bootstrap filters ({:d})".format(
                            len(bootstrap_lst)
                        )
                    )
                    raise RuntimeError(msg)

                self._import_w_factory_lst = importance_weight_factory_lst
            else:
                msg = (
                    "Must supply an importance weight factory when "
                    + "specifying the bootstrap filters."
                )
                raise RuntimeError(msg)
        elif gsm_lst is not None:
            num_filts = len(gsm_lst)
            if not (isinstance(num_parts, list) or isinstance(num_parts, tuple)):
                if num_parts is None:
                    msg = "Must specify number of particles when giving a list of GSM objects."
                    raise RuntimeError(msg)
                else:
                    num_parts = [num_parts] * num_filts

            self._meas_noise_filters = [None] * num_filts
            self._import_w_factory_lst = [None] * num_filts
            for ii, gsm in enumerate(gsm_lst):
                if rng is None:
                    rng = rnd.default_rng()

                if gsm.type is GSMTypes.STUDENTS_T:
                    (
                        self._meas_noise_filters[ii],
                        self._import_w_factory_lst[ii],
                    ) = self._define_student_t_pf(gsm, rng, num_parts[ii])

                elif GSMTypes.CAUCHY:
                    (
                        self._meas_noise_filters[ii],
                        self._import_w_factory_lst[ii],
                    ) = self._define_cauchy_pf(gsm, rng, num_parts[ii])

                else:
                    msg = (
                        "GSM filter can not automatically setup Bootstrap "
                        + "filter for GSM Type {:s}. ".format(gsm.type)
                        + "Update implementation."
                    )
                    raise RuntimeError(msg)
        else:
            msg = (
                "Incorrect input arguement combination. See documentation for details."
            )
            raise RuntimeError(msg)

    def set_process_noise_model(
        self, initial_est=None, filter_length=None, startup_delay=None
    ):
        """Sets the filter parameters for estimating process noise.

        This assumes the same process noise model as in
        :cite:`VilaValls2011_BayesianFilteringforNonlinearStateSpaceModelsinSymmetricStableMeasurementNoise`.

        Parameters
        ----------
        initial_est : N x N numpy array, optional
            Initial estimate of the covariance, can either be set here or by
            manually setting the process noise variable. The default is None,
            this assumes process noise was manually set.
        filter_length : int, optional
            Number of past samples to use in the filter, must be > 1. The
            default is None, this implies all past samples are used or the
            previously set value is maintained.
        startup_delay : int, optional
            Number of samples to delay before runing the filter, used to fill
            the FIFO buffers. Must be >= 1. The default is None, this means
            either the previous value is maintained or a value of 1 is used.

        Returns
        -------
        None.
        """
        self.enable_proc_noise_estimation = True

        if filter_length is not None:
            self._procNoiseEstimator.maxlen = filter_length

        if initial_est is None and (
            self.proc_noise is None or self.proc_noise.size <= 0
        ):
            msg = (
                "Please manually set the initial process noise "
                + "or specify a value here."
            )
            warn(msg)
        elif initial_est is not None:
            self.proc_noise = initial_est

        if startup_delay is not None:
            self._procNoiseEstimator.startup_delay = startup_delay

    @property
    def cov(self):
        """Covariance of the filter."""
        if self._coreFilter is None:
            return np.array([[]])
        else:
            return self._coreFilter.cov

    @cov.setter
    def cov(self, val):
        self._coreFilter.cov = val

    @property
    def proc_noise(self):
        """Wrapper for the process noise covariance of the core filter."""
        if self._coreFilter is None:
            return np.array([[]])
        else:
            return self._coreFilter.proc_noise

    @proc_noise.setter
    def proc_noise(self, val):
        self._coreFilter.proc_noise = val

    @property
    def meas_noise(self):
        """Measurement noise of the core filter, estimated online and does not need to be set."""
        if self._coreFilter is None:
            return np.array([[]])
        else:
            return self._coreFilter.meas_noise

    @meas_noise.setter
    def meas_noise(self, val):
        warn(
            "Measurement noise is estimated online. NOT SETTING VALUE HERE.",
            RuntimeWarning,
        )

    def predict(self, timestep, cur_state, **kwargs):
        """Prediction step of the GSM filter.

        This optionally estimates the process noise then calls the core filters
        prediction function.
        """
        return self._coreFilter.predict(timestep, cur_state, **kwargs)

    def correct(self, timestep, meas, cur_state, core_filt_kwargs={}):
        """Correction step of the GSM filter.

        This optionally estimates the measurement noise then calls the core
        filters correction function.
        """
        # setup core filter for estimating measurement noise during
        # correction function call
        def est_meas_noise(est_meas, inov_cov):
            m_diag = np.nan * np.ones(est_meas.size)
            f_meas = (meas - est_meas).ravel()
            inov_cov = np.diag(inov_cov)
            for ii, filt in enumerate(self._meas_noise_filters):
                filt.importance_weight_fnc = self._import_w_factory_lst[ii](
                    inov_cov[ii]
                )
                filt.predict(timestep)
                state = filt.correct(timestep, f_meas[ii].reshape((1, 1)))
                m_diag[ii] = state[2] * state[1] ** 2  # z * sig^2

            return np.diag(m_diag)

        self._coreFilter.set_measurement_noise_estimator(est_meas_noise)

        pred_cov = self.cov.copy()
        cor_state, meas_fit_prob = self._coreFilter.correct(
            timestep, meas, cur_state, **core_filt_kwargs
        )

        # update process noise estimate (if applicable)
        if self.enable_proc_noise_estimation:
            self.proc_noise = self._procNoiseEstimator.estimate_next(
                self.proc_noise, cur_state, pred_cov, cor_state, self.cov
            )

        return cor_state, meas_fit_prob

    def plot_particles(self, filt_inds, dist_inds, **kwargs):
        """Plots the particle distribution for every given measurement index.

        Parameters
        ----------
        filt_inds : int or list
            Index of the measurement index/indices to plot the particle distribution
            for.
        dist_inds : int or list
            Index of the particle(s) in the given filter(s) to plot. See the
            :meth:`.BootstrapFilter.plot_particles` for details.
        **kwargs : dict
            Additional keyword arguements. See :meth:`.BootstrapFilter.plot_particles`.

        Returns
        -------
        figs : dict
            Each value in the dictionary is a matplotlib figure handle.
        keys : list
            Each value is a string corresponding to a key in the resulting dictionary
        """
        figs = {}
        key_base = "meas_noise_particles_F{:02d}_D{:02d}"
        keys = []
        if not isinstance(filt_inds, list):
            key = key_base.format(filt_inds, dist_inds)
            figs[key] = self._meas_noise_filters[filt_inds].plot_particles(
                dist_inds, **kwargs
            )
            keys.append(key)
        else:
            for ii in filt_inds:
                key = key_base.format(ii, dist_inds)
                figs[key] = self._meas_noise_filters[ii].plot_particles(
                    dist_inds, **kwargs
                )
                keys.append(key)

        return figs, keys


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
