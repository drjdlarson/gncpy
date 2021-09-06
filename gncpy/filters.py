"""Definitions for common filters."""
import numpy as np
import numpy.random as rnd
import numpy.linalg as la
from scipy.linalg import expm
import scipy.integrate as s_integrate
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
    def predict(self, timestep, cur_state, **kwargs):
        """Generic method for the filters prediction step.

        This must be overridden in the inherited class. It is recommended to
        keep the same structure/order for the arguments to allow for
        standardized implementation of wrapper code.
        """
        pass

    @abc.abstractmethod
    def correct(self, timestep, cur_state, meas, **kwargs):
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
    def set_measurment_model(self, **kwargs):
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
            *args. The default is None.
        input_mat_fun : callable, optional
            Function that returns the `input_mat`, must take timestep, and
            *args. The default is None.
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

    def set_measurment_model(self, meas_mat=None, meas_fun=None):
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
            Measurment matrix that transforms the state to estimated
            measurements. The default is None.
        meas_fun : callable, optional
            Function that returns the matrix for transforming the state to
            estimated measurements. Must take timestep, and *args as
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

    def correct(self, timestep, cur_state, meas, meas_fun_args=()):
        """Implementss a discrete time correction step for a Kalman Filter.

        Parameters
        ----------
        timestep : float
            Current timestep.
        cur_state : N x 1 numpy array
            Current state.
        meas : Nm x 1 numpy array
            Current measurement.
        meas_fun_args : tuple, optional
            Arguments for the measurement matrix function if one has
            been specified. The default is ().

        Returns
        -------
        next_state : N x 1 numpy array
            The corrected state.
        meas_fit_prob : float
            Goodness of fit of the measurment based on the state and
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

    def set_measurment_model(self, meas_mat=None, meas_fun_lst=None):
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
            Measurment matrix that transforms the state to estimated
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
        super().set_measurment_model(meas_mat=meas_mat, meas_fun=meas_fun_lst)


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
    """

    def __init__(self, scale=np.array([[]]), dof=3, proc_noise_dof=3,
                 meas_noise_dof=3, **kwargs):
        self.scale = scale
        self.dof = dof
        self.proc_noise_dof = proc_noise_dof
        self.meas_noise_dof = meas_noise_dof

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

    def correct(self, timestep, cur_state, meas, meas_fun_args=()):
        """Implements the correction step of the students T filter.

        This also performs the moment matching.

        Parameters
        ----------
        timestep : float
            Current timestep.
        cur_state : N x 1 numpy array
            Current state.
        meas : Nm x 1 numpy array
            Current measurement.
        meas_fun_args : tuple, optional
            Arguments for the measurement matrix function if one has
            been specified. The default is ().

        Returns
        -------
        next_state : N x 1 numpy array
            The corrected state.
        meas_fit_prob : float
            Goodness of fit of the measurment based on the state and
            scale assuming Student's t noise.
        """
        def _pdf(x, mu, sig, v):
            d = x.size
            del2 = (x - mu).T @ la.inv(sig) @ (x - mu)
            inv_det = 1 / np.sqrt(la.det(sig))
            gam_rat = gmath.gamma_fnc(np.floor((v + d) / 2)) \
                / gmath.gamma_fnc(np.floor(v / 2))
            return gam_rat / (v * np.pi)**(d / 2) * inv_det \
                * (1 + del2 / v)**(-(v + d) / 2)

        est_meas, meas_mat = self._est_meas(timestep, cur_state, meas.size,
                                            meas_fun_args)

        # update state
        factor = self.meas_noise_dof * (self.dof - 2) \
            / (self.dof * (self.meas_noise_dof - 2))
        P_zz = meas_mat @ self.scale @ meas_mat.T + factor * self.meas_noise
        inv_P_zz = la.inv(P_zz)
        gain = self.scale @ meas_mat.T @ inv_P_zz
        P_kk = self.scale - gain @ meas_mat @ self.scale

        innov = (meas - est_meas)
        delta_2 = innov.T @ inv_P_zz @ innov
        next_state = cur_state + gain @ innov

        # moment matching
        factor = (self.dof + delta_2) / (self.dof + meas.size)
        P_k = factor * P_kk
        dof_p = self.dof + meas.size

        factor = dof_p * (self.dof - 2) / (self.dof * (dof_p - 2))
        self.scale = factor * P_k

        # self.scale = P_kk
        # self.dof = dof_p

        # get measurement fit
        meas_fit_prob = _pdf(meas, est_meas, P_zz, self.meas_noise_dof)
        meas_fit_prob = meas_fit_prob.item()

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
            *args. The default is None.
        input_mat_fun : callable, optional
            Function that returns the `input_mat`, must take timestep, and
            *args. The default is None.
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

    def correct(self, timestep, cur_state, meas, meas_fun_args=()):
        """Correction step of the UKF.

        Parameters
        ----------
        timestep : float
            Current timestep.
        cur_state : N x 1 numpy array
            Current state.
        meas : Nm x 1 numpy array
            Current measurement.
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
    """ This implements a basic Particle Filter.

    The implementation is based on
    :cite:`Simon2006_OptimalStateEstimationKalmanHInfinityandNonlinearApproaches`
    and uses Sampling-Importance Resampling (SIR) sampling. Other resampling
    methods can be added in derived classes.

    Attributes:
        dyn_fnc (function): Function that takes the current state and kwargs
            and returns the next state. Leave as None to use the state
            transition matrix
        meas_likelihood_fnc (function): Function of the measurements,
            estimated measurements, and kwargs and returns the relative
            likelihood of the estimated measurement
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
            `f(t, x, *args) and return a N x 1 numpy array. The default is None.

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

    def set_measurment_model(self, meas_mat=None, meas_fun=None):
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
            Measurment matrix that transforms the state to estimated
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

    def init_from_dist(self, dist):
        """Initialize the distribution from a distribution object.

        Parameters
        ----------
        dist : :class:`gncpy.distributions.ParticleDistribution`
            Distribution object to use.

        Returns
        -------
        None.

        """
        self._particleDist = deepcopy(dist)

    def extract_dist(self):
        """Extracts the particle distribution used by the filter.

        Returns
        -------
        :class:`gncpy.distributions.ParticleDistribution`
            copy of the internal particle distribution object.

        """
        return deepcopy(self._particleDist)

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

    def predict(self, timestep, cur_state, dyn_fun_params=(), sampling_args=()):
        """Predicts the next state.

        Parameters
        ----------
        timestep : float
            Current timestep.
        cur_state : N x 1 numpy array
            Current state.
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
            self._prop_parts = [self._dyn_obj.propagate_state(timestep, p,
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
            weights = [0]
            return weights

        weights = [self.meas_likelihood_fnc(meas, y, *meas_likely_args)
                   for y in est_meas]
        if renorm:
            tot = np.sum(weights)
            if tot > 0:
                weights = [qi / tot for qi in weights]
        return weights

    def _calc_weights(self, meas, est_meas, conditioned_lst, meas_likely_args,
                      proposal_args):
        rel_likeli = self._calc_relative_likelihoods(meas, est_meas,
                                                     renorm=False,
                                                     meas_likely_args=meas_likely_args)
        prop_fit = [self.proposal_fnc(x_hat, cond, *proposal_args)
                    for x_hat, cond in zip(self._particleDist.particles,
                                           conditioned_lst)]
        weights = []
        for p_ii, q_ii in zip(rel_likeli, prop_fit):
            if q_ii < np.finfo(float).tiny:
                w = np.inf
            else:
                w = p_ii / q_ii
            weights.append(w)

        tot = np.sum(weights)

        if tot > 0 and tot != np.inf:
            up_weights = [w / tot for w in weights]
        else:
            up_weights = [np.inf] * len(weights)
        self._particleDist.update_weights(up_weights)

    def _selection(self, rng):
        new_parts = []
        inds_kept = []
        for m in range(0, self.num_particles):
            r = rng.random()
            cumulative_weight = 0
            n = -1
            failed = False
            while cumulative_weight < r:
                n += 1
                if n >= self.num_particles:
                    failed = True
                    break
                cumulative_weight += self._particleDist.weights[n]

            if failed:
                continue

            new_parts.append(deepcopy(self._particleDist._particles[n]))
            if n not in inds_kept:
                inds_kept.append(n)

        inds_removed = [ii for ii in range(0, self.num_particles)
                        if ii not in inds_kept]

        self._particleDist.clear_particles()
        for p in new_parts:
            self._particleDist.add_particle(p, 1 / len(new_parts))

        return inds_removed

    def correct(self, timestep, cur_state, meas, selection=True,
                meas_fun_args=(), meas_likely_args=(), proposal_args=(),
                rng=rnd.default_rng()):
        """Corrects the state estimate.

        Parameters
        ----------
        timestep : float
            Current timestep.
        cur_state : N x 1 numpy array
            Current state.
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
        self._calc_weights(meas, est_meas, self._prop_parts, meas_likely_args,
                           proposal_args)

        # resample
        if selection:
            inds_removed = self._selection(rng)
            est_meas = [self._est_meas(timestep, p.point, meas.size, meas_fun_args)
                        for p, w in self._particleDist]
        else:
            inds_removed = []

        # update likelihoods
        rel_likeli = self._calc_relative_likelihoods(meas, est_meas,
                                                     renorm=False,
                                                     meas_likely_args=meas_likely_args)

        return (self._calc_state(), rel_likeli, inds_removed)

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
    """Implements a Macimum Correntropy Unscented Kalman filter.

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

    def correct(self, timestep, cur_state, meas, past_state, meas_fun_args=()):
        self._past_state = past_state.copy()
        self._cur_state = cur_state.copy()
        self._meas = meas.copy()

        super().correct(timestep, cur_state, meas, meas_fun_args=meas_fun_args)
