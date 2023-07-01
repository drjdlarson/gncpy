import numpy as np
import numpy.linalg as la
import scipy.linalg as sla
import scipy.integrate as s_integrate
from copy import deepcopy

import gncpy.dynamics.basic as gdyn
import gncpy.math as gmath
import gncpy.filters._filters as cpp_bindings
from gncpy.filters.kalman_filter import KalmanFilter


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

        if dyn_obj is not None or ode_lst is not None:
            self.set_state_model(dyn_obj=dyn_obj, ode_lst=ode_lst)

        self._integrator = None

        self.__model = None
        self.__predParams = None
        self.__corrParams = None

    def save_filter_state(self):
        """Saves filter variables so they can be restored later."""
        filt_state = super().save_filter_state()

        filt_state["cont_cov"] = self.cont_cov
        filt_state["integrator_type"] = self.integrator_type
        filt_state["integrator_params"] = deepcopy(self.integrator_params)
        filt_state["_ode_lst"] = self._ode_lst
        filt_state["_integrator"] = self._integrator

        filt_state["__model"] = self.__model
        filt_state["__predParams"] = self.__predParams
        filt_state["__corrParams"] = self.__corrParams

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

        self.__model = filt_state["__model"]
        self.__predParams = filt_state["__predParams"]
        self.__corrParams = filt_state["__corrParams"]

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
                    timestep, cur_state, *dyn_fun_params, use_continuous=self.cont_cov
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
            if self.cont_cov:
                state_mat = gmath.get_state_jacobian(
                    timestep, cur_state, self._ode_lst, dyn_fun_params
                )
            else:
                raise NotImplementedError(
                    "Non-continous covariance is not implemented yet for ode list"
                )
            dt = self.dt
        else:
            raise RuntimeError("State model not set")
        return next_state, state_mat, dt

    def _init_model(self):
        self._cpp_needs_init = (
            self.__model is None
            and (self._dyn_obj is not None and self._dyn_obj.allow_cpp)
            and self._measObj is not None
        )
        if self._cpp_needs_init:
            self.__model = cpp_bindings.ExtendedKalman()
            self.__predParams = cpp_bindings.BayesPredictParams()
            self.__corrParams = cpp_bindings.BayesCorrectParams()

            # make sure the cpp filter has its values set based on what python user gave (init only)
            self.__model.cov = self._cov.astype(np.float64)
            self.__model.set_state_model(self._dyn_obj.model, self.proc_noise)
            self.__model.set_measurement_model(self._measObj, self.meas_noise)

    def predict(
        self,
        timestep,
        cur_state,
        cur_input=None,
        dyn_fun_params=None,
        control_fun_params=None,
    ):
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
        control_fun_params : tuple, optional
            Extra arguments to be passed to the control input function. The default is None.

        Raises
        ------
        RuntimeError
            Integration fails, or state model not set.

        Returns
        -------
        next_state : N x 1 numpy array
            The predicted state.

        """
        self._init_model()
        if self.__model is not None:
            if control_fun_params is None:
                control_fun_params = ()
            (
                self.__predParams.stateTransParams,
                self.__predParams.controlParams,
            ) = self._dyn_obj.args_to_params(dyn_fun_params, control_fun_params)
            return self.__model.predict(
                timestep, cur_state, cur_input, self.__predParams
            ).reshape((-1, 1))

        else:
            if dyn_fun_params is None:
                dyn_fun_params = ()
            next_state, state_mat, dt = self._predict_next_state(
                timestep, cur_state, dyn_fun_params
            )

            if self.cont_cov:
                if dt is None:
                    raise RuntimeError(
                        "dt can not be None when using a continuous covariance model"
                    )

                def ode(t, x, n_states, F, proc_noise):
                    P = x.reshape((n_states, n_states))
                    P_dot = F @ P + P @ F.T + proc_noise
                    return P_dot.ravel()

                integrator = s_integrate.ode(ode)
                integrator.set_integrator(
                    self.integrator_type, **self.integrator_params
                )
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

    def set_measurement_model(self, meas_mat=None, meas_fun_lst=None, measObj=None):
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
        super().set_measurement_model(
            meas_mat=meas_mat, meas_fun=meas_fun_lst, measObj=measObj
        )

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
        self._init_model()

        if self.__model is not None:
            self.__corrParams.measParams = self._measObj.args_to_params(meas_fun_args)
            out = self.__model.correct(timestep, meas, cur_state, self.__corrParams)
            return out[0].reshape((-1, 1)), out[1]

        else:
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
