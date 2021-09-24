"""Definitions for common filters."""
import numpy as np
import numpy.random as rnd
import numpy.linalg as la
from scipy.linalg import expm
import scipy.integrate as s_integrate
import scipy.stats as stats
import abc
from warnings import warn
from copy import deepcopy
import matplotlib.pyplot as plt

import gncpy.math as gmath
import gncpy.plotting as pltUtil
import gncpy.distributions as gdistrib
import gncpy.dynamics as gdyn


class BayesFilter(metaclass=abc.ABCMeta):
    """Generic base class for Bayesian Filters such as a Kalman Filter.

    This defines the required functions and provides their recommended function
    signature for inherited classes.

    """

    def __init__(self, **kwargs):
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
        """Generic method for tsetting the measurement model.

        This must be overridden in the inherited class. The signature for this
        is arbitrary.
        """
        pass


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
    dt : float, optional
        Time difference between simulation steps.

    """

    def __init__(self, cov=np.array([[]]), meas_noise=np.array([[]]), dt=None,
                 **kwargs):
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

        super().__init__(**kwargs)

    def set_state_model(self, state_mat=None, input_mat=None, cont_time=False,
                        state_mat_fun=None, input_mat_fun=None, dyn_obj=None):
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
                msg = 'dt must be specified when using continuous time model'
                raise RuntimeError(msg)
            n_cols = state_mat.shape[1] + input_mat.shape[1]
            big_mat = np.vstack((np.hstack((state_mat, input_mat)),
                                 np.zeros((input_mat.shape[1], n_cols))))
            res = expm(big_mat * self.dt)
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
            raise RuntimeError('Invalid combination of inputs')

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
            raise RuntimeError('Invalid combination of inputs')

    def _predict_next_state(self, timestep, cur_state, cur_input, state_mat_args,
                            input_mat_args):
        if self._dyn_obj is not None:
            next_state = self._dyn_obj.propagate_state(timestep, cur_state,
                                                       u=cur_input,
                                                       state_args=state_mat_args,
                                                       ctrl_args=input_mat_args)
            state_mat = self._dyn_obj.get_state_mat(timestep, *state_mat_args)
        else:
            if self._get_state_mat is not None:
                state_mat = self._get_state_mat(timestep, *state_mat_args)
            elif self._state_mat is not None:
                state_mat = self._state_mat
            else:
                raise RuntimeError('State model not set')

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

    def predict(self, timestep, cur_state, cur_input=None, state_mat_args=(),
                input_mat_args=()):
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
        next_state, state_mat = self._predict_next_state(timestep, cur_state,
                                                         cur_input, state_mat_args,
                                                         input_mat_args)

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
        meas_mat = self._get_meas_mat(timestep, cur_state, n_meas,
                                      meas_fun_args)

        est_meas = meas_mat @ cur_state

        return est_meas, meas_mat

    def correct(self, timestep, meas, cur_state, meas_fun_args=()):
        """Implementss a discrete time correction step for a Kalman Filter.

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
            covariance assuming Gaussian noise.

        """
        est_meas, meas_mat = self._est_meas(timestep, cur_state, meas.size,
                                            meas_fun_args)

        # get the Kalman gain
        cov_meas_T = self.cov @ meas_mat.T
        inov_cov = meas_mat @ cov_meas_T + self.meas_noise
        inov_cov = (inov_cov + inov_cov.T) * 0.5
        sqrt_inv_inov_cov = la.inv(la.cholesky(inov_cov))
        inv_inov_cov = sqrt_inv_inov_cov.T @ sqrt_inv_inov_cov
        kalman_gain = cov_meas_T @ inv_inov_cov

        # update the state with measurement
        inov = meas - est_meas
        next_state = cur_state + kalman_gain @ inov

        # update the covariance
        n_states = cur_state.shape[0]
        self.cov = (np.eye(n_states) - kalman_gain @ meas_mat) @ self.cov

        # calculate the measuremnt fit probability assuming Gaussian
        meas_fit_prob = np.exp(-0.5 * (meas.size * np.log(2 * np.pi)
                                       + np.log(la.det(inov_cov))
                                       + inov.T @ inv_inov_cov @ inov))
        meas_fit_prob = meas_fit_prob.item()

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
        self.cont_cov = cont_cov
        self.integrator_type = 'dopri5'
        self.integrator_params = {}

        self._dyn_obj = None
        self._ode_lst = None

        if dyn_obj is not None or ode_lst is not None:
            self.set_state_model(dyn_obj=dyn_obj, ode_lst=ode_lst)

        self._integrator = None

        super().__init__(**kwargs)

    def set_state_model(self, dyn_obj=None, ode_lst=None):
        r"""Sets the state model equations.

        This allows for setting the differential equations directly

        .. math::
            \dot{x} = f(t, x, u)

        or setting a :class:`gncpy.dynamics.NonlinearDynamicsBase` object. If
        the object is specified then a local copy is created.

        Parameters
        ----------
        dyn_obj : :class:`gncpy.dynamics.NonlinearDynamicsBase`, optional
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
        if dyn_obj is not None:
            self._dyn_obj = deepcopy(dyn_obj)
        elif ode_lst is not None and len(ode_lst) > 0:
            self._ode_lst = ode_lst
        else:
            msg = 'Invalid state model specified. Check arguments'
            raise RuntimeError(msg)

    def _predict_next_state(self, timestep, cur_state, dyn_fun_params):
        if self._dyn_obj is not None:
            next_state = self._dyn_obj.propagate_state(timestep, cur_state,
                                                       state_args=dyn_fun_params)
            state_mat = self._dyn_obj.get_state_mat(timestep, cur_state,
                                                    dyn_fun_params)
            dt = self._dyn_obj.dt
        elif self._ode_lst is not None:
            next_state = np.nan * np.ones(cur_state.shape)
            for ii, f in enumerate(self._ode_lst):
                self._integrator = s_integrate.ode(f)
                self._integrator.set_integrator(self.integrator_type,
                                                **self.integrator_params)
                self._integrator.set_initial_value(cur_state, timestep)
                self._integrator.set_f_params(*dyn_fun_params)

                next_time = timestep + self.dt
                next_state[ii, 0] = self._integrator.integrate(next_time)
                if not self._integrator.successful():
                    msg = 'Integration failed at time {}'.format(timestep)
                    raise RuntimeError(msg)

            state_mat = gmath.get_state_jacobian(timestep, cur_state,
                                                 self._ode_lst, dyn_fun_params)

            dt = self.dt
        else:
            raise RuntimeError('State model not set')

        return next_state, state_mat, dt

    def predict(self, timestep, cur_state, dyn_fun_params=()):
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
            is ().

        Raises
        ------
        RuntimeError
            Integration fails, or state model not set.

        Returns
        -------
        next_state : N x 1 numpy array
            The predicted state.

        """
        next_state, state_mat, dt = self._predict_next_state(timestep,
                                                             cur_state,
                                                             dyn_fun_params)

        if self.cont_cov:
            def ode(t, x, n_states, F, proc_noise):
                P = x.reshape((n_states, n_states))
                P_dot = F @ P + P @ F.T + proc_noise
                return P_dot.flatten()

            integrator = s_integrate.ode(ode)
            integrator.set_integrator(self.integrator_type,
                                      **self.integrator_params)
            integrator.set_initial_value(self.cov.flatten(), timestep)
            integrator.set_f_params(cur_state.size, state_mat, self.proc_noise)
            tmp = integrator.integrate(timestep + dt)
            if not integrator.successful():
                msg = 'Failed to integrate covariance at {}'.format(timestep)
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
                res = gmath.get_jacobian(state.copy(),
                                         lambda _x, *_f_args: h(t, _x, *_f_args),
                                         f_args=meas_fun_args)
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

    def __init__(self, scale=np.array([[]]), dof=3, proc_noise_dof=3,
                 meas_noise_dof=3, use_moment_matching=True, **kwargs):
        self.scale = scale
        self.dof = dof
        self.proc_noise_dof = proc_noise_dof
        self.meas_noise_dof = meas_noise_dof
        self.use_moment_matching = use_moment_matching

        self._dyn_obj = None
        self._state_mat = np.array([[]])
        self._input_mat = np.array([[]])
        self._get_state_mat = None
        self._get_input_mat = None
        self._meas_mat = np.array([[]])
        self._meas_fnc = None

        super().__init__(**kwargs)

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

    def predict(self, timestep, cur_state, cur_input=None, state_mat_args=(),
                input_mat_args=()):
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
        next_state, state_mat = self._predict_next_state(timestep, cur_state,
                                                         cur_input, state_mat_args,
                                                         input_mat_args)

        factor = self.proc_noise_dof * (self.dof - 2) \
            / (self.dof * (self.proc_noise_dof - 2))
        self.scale = state_mat @ self.scale @ state_mat.T \
            + factor * self.proc_noise
        self.scale = (self.scale + self.scale.T) * 0.5

        return next_state

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
        est_meas, meas_mat = self._est_meas(timestep, cur_state, meas.size,
                                            meas_fun_args)

        # get gain
        scale_meas_T = self.scale @ meas_mat.T
        factor = self.meas_noise_dof * (self.dof - 2) \
            / (self.dof * (self.meas_noise_dof - 2))
        inov_cov = meas_mat @ scale_meas_T + factor * self.meas_noise
        inov_cov = (inov_cov + inov_cov.T) * 0.5
        sqrt_inv_inov_cov = la.inv(la.cholesky(inov_cov))
        inv_inov_cov = sqrt_inv_inov_cov.T @ sqrt_inv_inov_cov
        gain = scale_meas_T @ inv_inov_cov
        P_kk = (np.eye(cur_state.shape[0]) - gain @ meas_mat) @ self.scale

        # update state
        innov = (meas - est_meas)
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
        meas_fit_prob = stats.multivariate_t.pdf(meas.flatten(),
                                                 loc=est_meas.flatten(),
                                                 shape=inov_cov,
                                                 df=self.meas_noise_dof)

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
    appropriate simplifications have been made.
    """

    def __init__(self, sigmaPoints=None, **kwargs):
        self._stateSigmaPoints = None
        if isinstance(sigmaPoints, gdistrib.SigmaPoints):
            self._stateSigmaPoints = sigmaPoints

        self._use_lin_dyn = False
        self._use_non_lin_dyn = False

        super().__init__(**kwargs)

    def init_sigma_points(self, state0, alpha, kappa, beta=2):
        """Initializes the sigma points used by the filter.

        Parameters
        ----------
        state0 : N x 1 numpy array
            Initial state.
        alpha : float
            Tunig parameter, influences the spread of sigma points about the
            mean. In range (0, 1].
        kappa : float
            Tunig parameter, influences the spread of sigma points about the
            mean. In range [0, inf].
        beta : float, optional
            Tunig parameter for distribution type. In range [0, Inf].
            Defaults to 2 for gaussians.
        """
        n = state0.size
        self._stateSigmaPoints = gdistrib.SigmaPoints(alpha=alpha,
                                                      kappa=kappa,
                                                      beta=beta, n=n)
        self._stateSigmaPoints.init_weights()
        self._stateSigmaPoints.update_points(state0, self.cov)

    def set_state_model(self, state_mat=None, input_mat=None, cont_time=False,
                        state_mat_fun=None, input_mat_fun=None,
                        dyn_obj=None, ode_lst=None):
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
        self._use_lin_dyn = (state_mat is not None or state_mat_fun is not None
                             or isinstance(dyn_obj, gdyn.LinearDynamicsBase))
        self._use_non_lin_dyn = ((isinstance(dyn_obj, gdyn.NonlinearDynamicsBase)
                                  or ode_lst is not None) and not self._use_lin_dyn)

        # allow for linear or non linear dynamics by calling the appropriate parent
        if self._use_lin_dyn:
            KalmanFilter.set_state_model(self, state_mat=state_mat,
                                         input_mat=input_mat, cont_time=cont_time,
                                         state_mat_fun=state_mat_fun,
                                         input_mat_fun=input_mat_fun,
                                         dyn_obj=dyn_obj)
        elif self._use_non_lin_dyn:
            ExtendedKalmanFilter.set_state_model(self, dyn_obj=dyn_obj,
                                                 ode_lst=ode_lst)
        else:
            raise RuntimeError('Invalid state model.')

    def predict(self, timestep, cur_state, cur_input=None, state_mat_args=(),
                input_mat_args=(), dyn_fun_params=()):
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
            new_points = [KalmanFilter._predict_next_state(self, timestep,
                                                           x, cur_input,
                                                           state_mat_args,
                                                           input_mat_args)[0]
                          for x in self._stateSigmaPoints.points]
        elif self._use_non_lin_dyn:
            new_points = [ExtendedKalmanFilter._predict_next_state(self,
                                                                   timestep,
                                                                   x,
                                                                   dyn_fun_params)[0]
                          for x in self._stateSigmaPoints.points]
        else:
            raise RuntimeError('State model not specified')

        self._stateSigmaPoints.points = new_points

        # estimate weighted state output
        next_state = self._stateSigmaPoints.mean

        # update covariance
        self.cov = self._stateSigmaPoints.cov + self.proc_noise
        self.cov = (self.cov + self.cov.T) * 0.5

        return next_state

    def _calc_meas_cov(self, timestep, n_meas, meas_fun_args):
        est_points = [self._est_meas(timestep, x, n_meas, meas_fun_args)[0]
                      for x in self._stateSigmaPoints.points]
        est_meas = gmath.weighted_sum_vec(self._stateSigmaPoints.weights_mean,
                                          est_points)
        meas_cov_lst = [(z - est_meas) @ (z - est_meas).T
                        for z in est_points]
        meas_cov = self.meas_noise \
            + gmath.weighted_sum_mat(self._stateSigmaPoints.weights_cov,
                                     meas_cov_lst)

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

        Returns
        -------
        next_state : N x 1 numpy array
            corrected state.
        meas_fit_prob : float
            measurement fit probability assuming a Gaussian distribution.

        """
        meas_cov, est_points, est_meas = self._calc_meas_cov(timestep,
                                                             meas.size,
                                                             meas_fun_args)

        cross_cov_lst = [(x - cur_state) @ (z - est_meas).T
                         for x, z in zip(self._stateSigmaPoints.points,
                                         est_points)]
        cross_cov = gmath.weighted_sum_mat(self._stateSigmaPoints.weights_cov,
                                           cross_cov_lst)

        gain = cross_cov @ la.inv(meas_cov)
        inov = (meas - est_meas)

        self.cov = self.cov - gain @ meas_cov @ gain.T
        self.cov = (self.cov + self.cov.T) * 0.5
        next_state = cur_state + gain @ inov

        meas_fit_prob = np.exp(-0.5 * (meas.size * np.log(2 * np.pi)
                                       + np.log(la.det(meas_cov))
                                       + inov.T @ meas_cov @ inov))

        return next_state, meas_fit_prob


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
    meas_likelihood_fnc : callable
        Function of the measurement, estimated measurement, and `*args`. It
        returns the relative likelihood of the estimated measurement
    proposal_sampling_fnc : callable
        Function of the state and `*args`. It returns a sampled state from
        some proposal distribution. The return value must be an N x 1 numpy array.
    proposal_fnc : callable
        Function of the state, expected state, and `*args`. It returns the
        relativel likelihood of the state.
    """

    def __init__(self, meas_likelihood_fnc=None, proposal_sampling_fnc=None,
                 proposal_fnc=None, dyn_obj=None, dyn_fun=None, part_dist=None,
                 **kwargs):

        self.meas_likelihood_fnc = meas_likelihood_fnc
        self.proposal_sampling_fnc = proposal_sampling_fnc
        self.proposal_fnc = proposal_fnc

        self._dyn_fnc = None
        self._dyn_obj = None

        self._meas_mat = None
        self._meas_fnc = None

        if dyn_obj is not None or dyn_fun is not None:
            self.set_state_model(dyn_obj=dyn_obj, dyn_fun=dyn_fun)

        self._particleDist = gdistrib.ParticleDistribution()
        if part_dist is not None:
            self.init_from_dist(part_dist)

        self._prop_parts = []

        super().__init__(**kwargs)

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
            msg = 'Invalid state model specified. Check arguments'
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
            raise RuntimeError('Invalid combination of inputs')

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
        raise RuntimeError('Covariance is read only')

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
        if num_parts > 0:
            w = 1.0 / num_parts
        else:
            warn('No particles to initialize. SKIPPING')
            return
        w_lst = [w for ii in range(0, num_parts)]
        self._particleDist.clear_particles()
        for (p, w) in zip(particle_lst, w_lst):
            part = gdistrib.Particle(point=p)
            self._particleDist.add_particle(part, w)

    def _update_particles(self, particle_lst):
        num_parts = len(particle_lst)
        if num_parts > 0:
            w = 1.0 / num_parts
        else:
            w = 1
        w_lst = [w for ii in range(0, num_parts)]
        self._particleDist.clear_particles()
        for (p, w) in zip(particle_lst, w_lst):
            part = gdistrib.Particle(point=p)
            self._particleDist.add_particle(part, w)

    def _calc_state(self):
        return self._particleDist.mean

    def predict(self, timestep, dyn_fun_params=(), sampling_args=()):
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
            self._prop_parts = [self._dyn_obj.propagate_state(timestep, p.point,
                                                              state_args=dyn_fun_params)
                                for p, w in self._particleDist]
        elif self._dyn_fnc is not None:
            self._prop_parts = [self._dyn_fnc(timestep, p.point, *dyn_fun_params)
                                for (p, w) in self._particleDist]
        else:
            raise RuntimeError('No state model set')

        new_parts = [self.proposal_sampling_fnc(x, *sampling_args)
                     for x in self._prop_parts]
        self._update_particles(new_parts)

        return self._calc_state()

    def _est_meas(self, timestep, cur_state, n_meas, meas_fun_args):
        if self._meas_fnc is not None:
            est_meas = self._meas_fnc(timestep, cur_state, *meas_fun_args)
        elif self._meas_mat is not None:
            est_meas = self._meas_mat @ cur_state
        else:
            raise RuntimeError('No measurement model set')

        return est_meas

    def _calc_relative_likelihoods(self, meas, est_meas, renorm=True,
                                   meas_likely_args=()):
        if len(est_meas) == 0:
            weights = np.array([0])
            return weights

        weights = np.array([self.meas_likelihood_fnc(meas, y, *meas_likely_args)
                            for y in est_meas])
        if renorm:
            tot = np.sum(weights)
            if tot > 0:
                weights /= tot
        return weights

    def _calc_weights(self, meas, est_meas, conditioned_lst, meas_likely_args,
                      proposal_args):
        rel_likeli = self._calc_relative_likelihoods(meas, est_meas,
                                                     renorm=False,
                                                     meas_likely_args=meas_likely_args)
        prop_fit = np.array([self.proposal_fnc(x_hat, cond, *proposal_args)
                             for x_hat, cond in zip(self._particleDist.particles,
                                                    conditioned_lst)])
        # weights = np.nan * np.ones(len(rel_likeli))
        weights = rel_likeli / prop_fit
        inds = np.where(prop_fit < np.finfo(float).tiny)[0]
        if inds.size > 0:
            weights[inds] = np.inf

        tot = np.sum(weights)

        if tot > 0 and tot != np.inf:
            weights /= tot
        else:
            weights[:] = np.inf
        self._particleDist.update_weights(weights)

        return rel_likeli

    def _selection(self, rng):
        new_parts = []
        inds_kept = []
        probs = rng.random(self.num_particles)
        cumulative_weight = np.cumsum(self._particleDist.weights)
        failed = False
        for r in probs:
            inds = np.where(cumulative_weight >= r)[0]
            if inds.size > 0:
                new_parts.append(deepcopy(self._particleDist._particles[inds[0]]))
                if inds[0] not in inds_kept:
                    inds_kept.append(inds[0])
            else:
                failed = True

        if failed:
            warn('Failed to select particle, check weights')

        inds_removed = [ii for ii in range(0, self.num_particles)
                        if ii not in inds_kept]

        self._particleDist.clear_particles()
        w = 1 / len(new_parts)
        for p in new_parts:
            self._particleDist.add_particle(p, w)

        return inds_removed

    def correct(self, timestep, meas, selection=True,
                meas_fun_args=(), meas_likely_args=(), proposal_args=(),
                rng=rnd.default_rng()):
        """Corrects the state estimate.

        Parameters
        ----------
        timestep : float
            Current timestep.
        meas : Nm x 1 numpy array
            Current measurement.
        selection : bool, optional
            flag indicating if the selection step should be run. The default is True.
        meas_fun_args : tuple, optional
            Arguments for the measurement matrix function if one has
            been specified. The default is ().
        meas_likely_args : tuple, optional
            additional agruments for the measurement likelihood function.
            The default is ().
        proposal_args : tuple, optional
            Additional arguments for the proposal distribution function. The
            default is ().
        rng : numpy random generator, optional
            Random number generator. The default is rnd.default_rng().

        Raises
        ------
        RuntimeError
            If no measurement model is set

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
        # calculate weights
        est_meas = [self._est_meas(timestep, p.point, meas.size, meas_fun_args)
                    for p, w in self._particleDist]
        rel_likeli = self._calc_weights(meas, est_meas, self._prop_parts,
                                        meas_likely_args, proposal_args)

        # resample
        if selection:
            inds_removed = self._selection(rng)
            est_meas = [self._est_meas(timestep, p.point, meas.size, meas_fun_args)
                        for p, w in self._particleDist]
            # update likelihoods
            rel_likeli = self._calc_relative_likelihoods(meas, est_meas,
                                                         renorm=False,
                                                         meas_likely_args=meas_likely_args)

        else:
            inds_removed = []

        return (self._calc_state(), rel_likeli.tolist(), inds_removed)

    def plot_particles(self, inds, title='Particle Distribution',
                       x_lbl='State', y_lbl='Count', **kwargs):
        opts = pltUtil.init_plotting_opts(**kwargs)
        f_hndl = opts['f_hndl']
        lgnd_loc = opts['lgnd_loc']

        if f_hndl is None:
            f_hndl = plt.figure()
            f_hndl.add_subplot(1, 1, 1)

        h_opts = {"histtype": "stepfilled"}
        if (not isinstance(inds, list)) or len(inds) == 1:
            if isinstance(inds, list):
                ii = inds[0]
            else:
                ii = inds
            x = [p.point[ii, 0] for (p, w) in self._particleDist]
            f_hndl.axes[0].hist(x, **h_opts)
        else:
            x = [p.point[inds[0], 0] for p, w in self._particleDist]
            y = [p.point[inds[1], 0] for p, w in self._particleDist]
            f_hndl.axes[0].hist2d(x, y)

        pltUtil.set_title_label(f_hndl, 0, opts, ttl=title,
                                x_lbl=x_lbl, y_lbl=y_lbl)
        if lgnd_loc is not None:
            plt.legend(loc=lgnd_loc)
        plt.tight_layout()

        return f_hndl

    def plot_weighted_particles(self, inds,  x_lbl='State', y_lbl='Weight',
                                title='Weighted Particle Distribution',
                                **kwargs):
        opts = pltUtil.init_plotting_opts(**kwargs)
        f_hndl = opts['f_hndl']
        lgnd_loc = opts['lgnd_loc']

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
            warn('Only 1 element supported for weighted particle distribution')

        pltUtil.set_title_label(f_hndl, 0, opts, ttl=title,
                                x_lbl=x_lbl, y_lbl=y_lbl)
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

    def __init__(self, use_MCMC=False, **kwargs):
        self.use_MCMC = use_MCMC

        super().__init__(**kwargs)

    def move_particles(self, **kwargs):
        """Generic interface for the movement function.

        This must be overridden in the child class. It is recommended to keep
        the same function signature to allow for standardized wrappers.
        """
        raise RuntimeError('Must implement thid function in derived class')


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

    def _calc_meas_cov(self, timestep, n_meas, meas_fun_args):
        est_points = [self._est_meas(timestep, x, n_meas, meas_fun_args)[0]
                      for x in self._stateSigmaPoints.points]
        est_meas = gmath.weighted_sum_vec(self._stateSigmaPoints.weights_mean,
                                          est_points)
        meas_cov_lst = [(z - est_meas) @ (z - est_meas).T
                        for z in est_points]

        # find square root of combined covariance matrix
        n_state = self.cov.shape[0]
        n_meas = est_meas.shape[0]
        z_12 = np.zeros((n_state, n_meas))
        z_21 = np.zeros((n_meas, n_state))
        comb_cov = np.vstack((np.hstack((self.cov, z_12)),
                              np.hstack((z_21, self.meas_noise))))
        comb_cov = (comb_cov + comb_cov.T) * 0.5
        sqrt_comb = la.cholesky(comb_cov)
        inv_sqrt_comb = la.inv(sqrt_comb)

        # find error vector
        pred_meas = self._est_meas(timestep, self._past_state, n_meas,
                                   meas_fun_args)[0]
        g = inv_sqrt_comb @ np.vstack((self._past_state, pred_meas))
        d = inv_sqrt_comb @ np.vstack((self._cur_state, self._meas))
        e = (d - g).flatten()

        # kernel function on error
        kern_lst = [gmath.gaussian_kernel(e_ii, self.kernel_bandwidth)
                    for e_ii in e]
        c = np.diag(kern_lst)
        c_inv = la.inv(c)

        # calculate the measurement covariance
        scaled_mat = sqrt_comb @ c_inv @ sqrt_comb.T
        scaled_meas_noise = scaled_mat[n_state:, n_state:]
        meas_cov = scaled_meas_noise \
            + gmath.weighted_sum_mat(self._stateSigmaPoints.weights_cov,
                                     meas_cov_lst)

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

    def __init__(self, **kwargs):
        self._filt = UnscentedKalmanFilter()

        super().__init__(**kwargs)

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

    def set_state_model(self, state_mat=None, input_mat=None, cont_time=False,
                        state_mat_fun=None, input_mat_fun=None,
                        dyn_obj=None, ode_lst=None):
        """Sets the state model for the filter.

        This calls the UKF's set state function
        (:meth:`gncpy.filters.UnscentedKalmanFilter.set_state_model`).
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
        self._filt.set_state_model(state_mat=state_mat, input_mat=input_mat,
                                   cont_time=cont_time,
                                   state_mat_fun=state_mat_fun,
                                   input_mat_fun=input_mat_fun, dyn_obj=dyn_obj,
                                   ode_lst=ode_lst)

    def set_measurement_model(self, meas_mat=None, meas_fun_lst=None):
        r"""Sets the measurement model for the filter.

        This is a wrapper for the inner UKF's set_measurement model function.
        It is assumed that the measurement model is the same as that of the UKF.
        See :meth:`gncpy.filters.UnscentedKalmanFilter.set_measurement_model`
        for details.
        """
        self._filt.set_measurement_model(meas_mat=meas_mat,
                                         meas_fun_lst=meas_fun_lst)

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
        newDist = gdistrib.ParticleDistribution()
        new_weight = 1 / self._particleDist.num_particles
        for origPart, w in self._particleDist:
            part = gdistrib.Particle()
            self._filt.cov = origPart.uncertainty
            self._filt._stateSigmaPoints = origPart.sigmaPoints
            part.point = self._filt.predict(timestep, origPart.point, **ukf_kwargs)
            part.uncertainty = self._filt.cov
            part.sigmaPoints = self._filt._stateSigmaPoints
            newDist.add_particle(part, new_weight)

        self._particleDist = newDist
        return self._calc_state()

    def _inner_correct(self, timestep, meas, state, filt_kwargs):
        """Wrapper so child class can override."""
        return self._filt.correct(timestep, meas, state, **filt_kwargs)

    def correct(self, timestep, meas, selection=True, ukf_kwargs={},
                rng=rnd.default_rng(), sampling_args=(), meas_fun_args=(),
                move_kwargs={}, meas_likely_args=(), proposal_args=()):
        """Correction step of the UPF.

        This optionally can perform a MCMC move step as well.

        Parameters
        ----------
        timestep : float
            Current timestep.
        meas : Nm x 1 numpy array
            measurement.
        selection : bool, optional
            Flag indicating if selection/resampling should be performed. The
            default is True.
        ukf_kwargs : dict, optional
            Additional arguments to pass to the UKF correction function. The
            default is {}.
        rng : numpy random generator, optional
            Random number generator. The default is rnd.default_rng().
        sampling_args : tuple, optional
            Extra arguments to be passed to the proposal sampling function.
            The default is ().
        meas_fun_args : tuple, optional
            Arguments for the measurement matrix function if one has
            been specified. The default is ().
        move_kwargs : dict, optional
            Additional arguments to pass to the movement function.
            The default is {}.
        meas_likely_args : tuple, optional
            additional agruments for the measurement likelihood function.
            The default is ().
        proposal_args : tuple, optional
            Additional arguments for the proposal distribution function. The
            default is ().

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
        if self.use_MCMC:
            oldDist = deepcopy(self._particleDist)

        # call UKF correction on each particle
        self._prop_parts = []
        newDist = gdistrib.ParticleDistribution()
        new_weight = 1 / self._particleDist.num_particles
        for p, w in self._particleDist:
            part = gdistrib.Particle()

            self._filt.cov = p.uncertainty
            self._filt._stateSigmaPoints = p.sigmaPoints
            ns = self._inner_correct(timestep, meas, p.point, ukf_kwargs)[0]
            cov = self._filt.cov

            self._prop_parts.append(ns)
            samp = self.proposal_sampling_fnc(ns, *sampling_args)

            part.point = samp
            part.uncertainty = cov
            part.sigmaPoints = self._filt._stateSigmaPoints

            newDist.add_particle(part, new_weight)

        # update info for next UKF
        self._particleDist = newDist

        # resample/selection
        est_meas = [self._filt._est_meas(timestep, p.point, meas.size, meas_fun_args)[0]
                    for p, w in self._particleDist]
        cov_lst = [p.uncertainty for p, w in self._particleDist]
        rel_likeli = self._calc_weights(meas, est_meas, self._prop_parts, cov_lst,
                                        meas_likely_args, proposal_args)

        if selection:
            inds_removed = self._selection(rng)
        else:
            inds_removed = True

        move_parts = (self.use_MCMC
                      and oldDist.num_particles == self._particleDist.num_particles)
        if move_parts:
            self.move_particles(timestep, oldDist, meas, **move_kwargs)

        if selection or move_parts:
            est_meas = [self._filt._est_meas(timestep, p.point, meas.size, meas_fun_args)[0]
                        for p, w in self._particleDist]
            rel_likeli = self._calc_relative_likelihoods(meas, est_meas,
                                                         renorm=False,
                                                         meas_likely_args=meas_likely_args)

        return (self._calc_state(), rel_likeli.tolist(), inds_removed)

    def _calc_weights(self, meas, est_meas, conditioned_lst, cov_lst,
                      meas_likely_args, proposal_args):
        rel_likeli = self._calc_relative_likelihoods(meas, est_meas,
                                                     renorm=False,
                                                     meas_likely_args=meas_likely_args)
        prop_fit = np.array([self.proposal_fnc(x_hat, cond, p_hat, *proposal_args)
                             for x_hat, cond, p_hat in zip(self._particleDist.particles,
                                                           conditioned_lst, cov_lst)])

        weights = rel_likeli / prop_fit
        inds = np.where(prop_fit < np.finfo(float).tiny)[0]
        if inds.size > 0:
            weights[inds] = np.inf
        tot = np.sum(weights)

        if tot > 0 and tot != np.inf:
            weights /= tot

        self._particleDist.update_weights(weights)

        return rel_likeli

    def move_particles(self, timestep, oldDist, meas, ukf_kwargs={},
                       rng=rnd.default_rng(), sampling_args=(), meas_fun_args=(),
                       meas_likely_args=(), proposal_args=()):
        """Movement function for the MCMC move step.

        This modifies the internal particle distribution but does not return a
        value.

        Notes
        -----
        This implements a metropolis-hastings algorithm.

        Parameters
        ----------
        timestep : float
            Current timestep.
        oldDist : :class:`gncpy.distributions.ParticleDistribution`
            Distribution before the measurement correction.
        meas : Nm x 1 numpy array
            measurement.
        ukf_kwargs : dict, optional
            Additional arguments to pass to the UKF correction function. The
            default is {}.
        rng : numpy random generator, optional
            Random number generator. The default is rnd.default_rng().
        sampling_args : tuple, optional
            Extra arguments to be passed to the proposal sampling function.
            The default is ().
        meas_fun_args : tuple, optional
            Arguments for the measurement matrix function if one has
            been specified. The default is ().
        meas_likely_args : tuple, optional
            additional agruments for the measurement likelihood function.
            The default is ().
        proposal_args : tuple, optional
            Additional arguments for the proposal distribution function. The
            default is ().

        Returns
        -------
        None.

        """
        accept_prob = rng.random()
        newDist = gdistrib.ParticleDistribution()
        new_weight = 1 / self._particleDist.num_particles
        for ii, (p, w) in enumerate(oldDist):
            self._filt.cov = p.uncertainty
            self._filt._stateSigmaPoints = p.sigmaPoints
            ns = self._inner_correct(timestep, meas, p.point, ukf_kwargs)[0]
            cov = self._filt.cov

            cand = self.proposal_sampling_fnc(ns, *sampling_args)

            est_cand_meas = self._filt._est_meas(timestep, cand, meas.size, meas_fun_args)[0]
            cand_likeli = self._calc_relative_likelihoods(meas, [est_cand_meas],
                                                          meas_likely_args=meas_likely_args,
                                                          renorm=False).item()
            cand_fit = self.proposal_fnc(cand, ns, cov, *proposal_args)

            part = self._particleDist._particles[ii]
            est_meas = self._filt._est_meas(timestep, part.point,
                                            meas.size, meas_fun_args)[0]
            part_likeli = self._calc_relative_likelihoods(meas,
                                                          [est_meas],
                                                          renorm=False)[0]
            part_fit = self.proposal_fnc(self._particleDist._particles[ii], ns,
                                         cov, *proposal_args)

            num = cand_likeli * part_fit
            den = part_likeli * cand_fit

            if den < np.finfo(np.float32).eps:
                ratio = np.inf
            else:
                ratio = num / den
            accept_val = np.min((ratio, 1))

            if accept_prob <= accept_val:
                newPart = gdistrib.Particle()
                newPart.point = cand
                newPart.uncertainty = cov
                newPart.sigmaPoints = self._filt._stateSigmaPoints
            else:
                newPart = part

            newDist.add_particle(newPart, new_weight)

        self._particleDist = newDist


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

    def _inner_correct(self, timestep, meas, state, filt_kwargs):
        """Wrapper so child class can override."""
        return self._filt.correct(timestep, meas, state, self._past_state,
                                  **filt_kwargs)

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
