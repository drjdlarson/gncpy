import numpy as np
import numpy.linalg as la
import scipy.linalg as sla
import scipy.stats as stats
from copy import deepcopy

import gncpy.errors as gerr
import gncpy.filters._filters as cpp_bindings
from gncpy.filters.bayes_filter import BayesFilter


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
        self._cov = cov
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
        self._measObj = None

        self._est_meas_noise_fnc = None

        self.__model = None
        self.__predParams = None
        self.__corrParams = None

        super().__init__(**kwargs)

    def __repr__(self) -> str:
        if self.__model is not None:
            return self.__model.__repr__()
        else:
            return super().__repr__()

    def __str__(self) -> str:
        if self.__model is not None:
            return self.__model.__str__()
        else:
            return super().__str__()

    @property
    def cov(self):
        if self.__model is not None:
            return self.__model.cov
        else:
            return self._cov

    @cov.setter
    def cov(self, val):
        if self.__model is not None:
            self.__model.cov = val
        else:
            self._cov = val

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

        filt_state["_measObj"] = self._measObj
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

        self._measObj = filt_state["_measObj"]
        self.__model = filt_state["__model"]
        self.__predParams = filt_state["__predParams"]
        self.__corrParams = filt_state["__corrParams"]

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
            self.__model = None
            self._state_mat = state_mat
            self._input_mat = input_mat
        elif have_mats:
            self.__model = None
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
            self.__model = None
            self._get_state_mat = state_mat_fun
            self._get_input_mat = input_mat_fun
        else:
            raise RuntimeError("Invalid combination of inputs")

    def set_measurement_model(self, meas_mat=None, meas_fun=None, measObj=None):
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
        measObj : class instance
            Measurement class instance

        Raises
        ------
        RuntimeError
            Rasied if no arguments are specified.

        Returns
        -------
        None.

        """
        if measObj is not None:
            self._measObj = measObj
            self._meas_mat = None
            self._meas_fnc = None
        elif meas_mat is not None:
            self.__model = None
            self._measObj = None
            self._meas_mat = meas_mat
            self._meas_fnc = None
        elif meas_fun is not None:
            self.__model = None
            self._measObj = None
            self._meas_mat = None
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

    def _init_model(self):
        self._cpp_needs_init = (
            self.__model is None
            and (self._dyn_obj is not None and self._dyn_obj.allow_cpp)
            and self._measObj is not None
        )
        if self._cpp_needs_init:
            self.__model = cpp_bindings.Kalman()
            self.__predParams = cpp_bindings.BayesPredictParams()
            self.__corrParams = cpp_bindings.BayesCorrectParams()

            # make sure the cpp filter has its values set based on what python user gave (init only)
            self.__model.cov = self._cov.astype(np.float64)
            self.__model.set_state_model(self._dyn_obj.model, self.proc_noise)
            self.__model.set_measurement_model(self._measObj, self.meas_noise)

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
        self._init_model()

        if self.__model is not None:
            (
                self.__predParams.stateTransParams,
                self.__predParams.controlParams,
            ) = self._dyn_obj.args_to_params(state_mat_args, input_mat_args)[:2]
            return self.__model.predict(
                timestep, cur_state, cur_input, self.__predParams
            ).reshape((-1, 1))
        else:
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
        extra_args = {}
        if np.abs(np.linalg.det(meas_cov)) > np.finfo(float).eps:
            extra_args["allow_singular"] = True
        return stats.multivariate_normal.pdf(
            meas.ravel(), mean=est_meas.ravel(), cov=meas_cov, **extra_args
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
