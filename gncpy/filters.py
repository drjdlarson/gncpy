import numpy as np
import numpy.random as rnd
import numpy.linalg as la
from scipy.linalg import expm
import abc
from warnings import warn
from copy import deepcopy
import matplotlib.pyplot as plt

import gncpy.math as gmath
import gncpy.plotting as pltUtil
import gncpy.distributions as gdistrib


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
        self._meas_model = None
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

    def set_meas_model(self, fnc):
        self._meas_model = fnc

    def get_est_meas(self, state, **kwargs):
        if self._meas_model is None:
            m_mat = self.get_meas_mat(state, **kwargs)
            return m_mat @ state
        else:
            return self._meas_model(state, **kwargs)

    def get_proc_noise(self, **kwargs):
        r""" Returns the process noise matrix.

        It returns the :math:`\Upsilon Q \Upsilon^T` matrix in the state
        space system :math:`x_{k+1} = F x_k + G u_k + \Upsilon w_k` where

        .. math::
            E\{w_kw_j^T\} = \begin{cases}
                0 & k \neq j \\
                Q_k & k = j
                \end{cases}

        Returns:
            (N x N numpy array): Discrete process noise matrix
        """
        return self._proc_noise

    def set_proc_noise(self, **kwargs):
        r""" Sets the process noise matrix.

        This can be overridden in inherited classes. It sets
        the :math:`\Upsilon Q \Upsilon^T` matrix in the state
        space system :math:`x_{k+1} = F x_k + G u_k + \Upsilon w_k` where

        .. math::
            E\{w_kw_j^T\} = \begin{cases}
                0 & k \neq j \\
                Q_k & k = j
                \end{cases}

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
        self.cov = (self.cov + self.cov.T) * 0.5
        return next_state

    def correct(self, **kwargs):
        """ Implementss a discrete time correction step for a Kalman Filter.

        Keyword Args:
            cur_state (N x 1 numpy array): Current predicted state
            meas (Nm x 1 numpy array): Current measurements

        Returns:
            tuple containing

                - (N x 1 numpy array): Corrected state
                - (float): Measurement fit probability
        """
        cur_state = kwargs['cur_state']
        meas = kwargs['meas']

        meas_mat = self.get_meas_mat(cur_state, **kwargs)
        cov_meas_T = self.cov @ meas_mat.T
        meas_pred_cov = meas_mat @ cov_meas_T + self.meas_noise
        meas_pred_cov = (meas_pred_cov + meas_pred_cov.T) * 0.5
#        inv_meas_cov = la.inv(meas_pred_cov)
        sqrt_inv_meas_cov = la.inv(la.cholesky(meas_pred_cov))
        inv_meas_cov = sqrt_inv_meas_cov.T @ sqrt_inv_meas_cov
        kalman_gain = cov_meas_T @ inv_meas_cov
        inov = meas - self.get_est_meas(cur_state, **kwargs)
        next_state = cur_state + kalman_gain @ inov

        meas_fit_prob = np.exp(-0.5 * (meas.size * np.log(2 * np.pi)
                                       + np.log(la.det(meas_pred_cov))
                                       + inov.T @ inv_meas_cov @ inov))
        meas_fit_prob = meas_fit_prob.item()

        n_states = cur_state.shape[0]
        self.cov = (np.eye(n_states) - kalman_gain @ meas_mat) @ self.cov

        return (next_state, meas_fit_prob)


class ExtendedKalmanFilter(KalmanFilter):
    r""" Implementation of a continuous-discrete time Extended Kalman Filter.

    This is loosely based on :cite:`Crassidis2011_OptimalEstimationofDynamicSystems`

    Attributes:
        dyn_fncs (list): List of dynamics functions. One per state (in order)
            and of the form :math:`\dot{x} = f(x, u)`
        proc_map (N x N numpy array): Array mapping continuous process noise
            to states
        proc_cov (N x N numpy array): Continuous time covariance matrix for
            the process noise
    """

    def __init__(self, **kwargs):
        self.dyn_fncs = []

        self.proc_map = np.array([[]])
        self.proc_cov = np.array([[]])
        super().__init__(**kwargs)

    def get_proc_noise(self, dt, cur_state, cur_input, **kwargs):
        # if the user is manually setting the process noise then return it
        if self._proc_noise.size > 0:
            return self._proc_noise

        # check that we have everything to calculate the process noise
        state_jac = kwargs.get('state_jac', None)
        if state_jac is None:
            state_jac = gmath.get_state_jacobian(cur_state, cur_input,
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
        return gmath.disrw(state_jac, self.proc_map, dt, self.proc_cov)

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

        state_jac = gmath.get_state_jacobian(cur_state, cur_input, self.dyn_fncs,
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

        next_state = gmath.rk4(comb_func, cur_state, dt, **kwargs)

        state_mat = self.get_state_mat(**kwargs)
        self.cov = state_mat @ self.cov @ state_mat.T \
            + self.get_proc_noise(state_mat=state_mat, **kwargs)

        return next_state


class UnscentedKalmanFilter(BayesFilter):
    """ This implements an unscented kalman filter.

    For details on the filter see
    :cite:`Wan2000_TheUnscentedKalmanFilterforNonlinearEstimation`. This
    implementation assumes that the noise is purely addative and as such,
    appropriate simplifications have been made.

    Attributes:
        dyn_fnc (function): Function that takes the current state and kwargs
            and returns the next state.
    """

    def __init__(self, **kwargs):
        self.dyn_fnc = kwargs.get('dyn_fnc', None)

        self._stateSigmaPoints = gdistrib.SigmaPoints()
        super().__init__(**kwargs)

    def init_sigma_points(self, state0, alpha, kappa, beta=2):
        """ Initializes the sigma points used by the filter

        Args:
            state0 (n x 1 numpy array): Initial state.
            alpha (float): Tunig parameter, influences the spread of sigma
                points about the mean. In range (0, 1].
            kappa (float): Tunig parameter, influences the spread of sigma
                points about the mean. In range [0, inf].
            beta (float, optional): Tunig parameter for distribution type.
                In range [0, Inf]. Defaults to 2 for gaussians.
        """
        n = state0.size
        self._stateSigmaPoints = gdistrib.SigmaPoints(alpha=alpha,
                                             kappa=kappa,
                                             beta=beta, n=n)
        self._stateSigmaPoints.init_weights()
        self._stateSigmaPoints.update_points(state0, self.cov)

    def predict(self, **kwargs):
        cur_state = kwargs['cur_state']
        try:
            self._stateSigmaPoints.update_points(cur_state, self.cov)
        except Exception:
            brk = 1
        # propagate points
        new_points = [self.dyn_fnc(x, **kwargs)
                      for x in self._stateSigmaPoints.points]
        self._stateSigmaPoints.points = new_points

        # estimate weighted state output
        next_state = self._stateSigmaPoints.mean

        # update covariance
        proc_noise = self.get_proc_noise(**kwargs)
        self.cov = self._stateSigmaPoints.cov + proc_noise

        return next_state

    def _calc_meas_cov(self, **kwargs):
        est_points = [self.get_est_meas(x, **kwargs)
                      for x in self._stateSigmaPoints.points]
        est_meas = gmath.weighted_sum_vec(self._stateSigmaPoints.weights_mean,
                                          est_points)
        meas_cov_lst = [(z - est_meas) @ (z - est_meas).T
                        for z in est_points]
        meas_cov = self.meas_noise \
            + gmath.weighted_sum_mat(self._stateSigmaPoints.weights_cov,
                                     meas_cov_lst)

        return meas_cov, est_points, est_meas

    def correct(self, **kwargs):
        cur_state = kwargs['cur_state']
        meas = kwargs['meas']

        meas_cov, est_points, est_meas = self._calc_meas_cov(**kwargs)

        cross_cov_lst = [(x - cur_state) @ (z - est_meas).T
                         for x, z in zip(self._stateSigmaPoints.points,
                                         est_points)]
        cross_cov = gmath.weighted_sum_mat(self._stateSigmaPoints.weights_cov,
                                           cross_cov_lst)

        gain = cross_cov @ la.inv(meas_cov)
        inov = (meas - est_meas)

        self.cov = self.cov - gain @ meas_cov @ gain.T
        next_state = cur_state + gain @ inov

        meas_fit_prob = np.exp(-0.5 * (meas.size * np.log(2 * np.pi)
                                       + np.log(la.det(meas_cov))
                                       + inov.T @ meas_cov @ inov))

        return next_state, meas_fit_prob


class MaxCorrEntUKF(UnscentedKalmanFilter):
    """ This implements a Macimum Correntropy Unscented Kalman filter.

    This is based on
    :cite:`Hou2018_MaximumCorrentropyUnscentedKalmanFilterforBallisticMissileNavigationSystemBasedonSINSCNSDeeplyIntegratedMode`

    Attributes:
        kernel_bandwidth (float): Bandwidth of the Gaussian Kernel
    """

    def __init__(self, **kwargs):
        self.kernel_bandwidth = kwargs.get('kernel_bandwidth', 1)
        super().__init__(**kwargs)

    def _calc_meas_cov(self, **kwargs):
        past_state = kwargs['past_state']  # before prediction step
        cur_state = kwargs['cur_state']  # output from prediction step
        meas = kwargs['meas']

        est_points = [self.get_est_meas(x, **kwargs)
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
        sqrt_comb = la.cholesky(comb_cov)
        inv_sqrt_comb = la.inv(sqrt_comb)

        # find error vector
        pred_meas = self.get_est_meas(past_state, **kwargs)
        g = inv_sqrt_comb @ np.vstack((past_state, pred_meas))
        d = inv_sqrt_comb @ np.vstack((cur_state, meas))
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


class StudentsTFilter(BayesFilter):
    """ Impplementation of a Students T filter.

    This is based on :cite:`Liu2018_AStudentsTMixtureProbabilityHypothesisDensityFilterforMultiTargetTrackingwithOutliers`
    and :cite:`Roth2013_AStudentsTFilterforHeavyTailedProcessandMeasurementNoise`
    and uses moment matching to limit the degree of freedom growth.

    Attributes:
        scale (N x N numpy array): Scaling matrix of the Students T
            distribution
        dof (int): Degree of freedom for the state distribution
        proc_noise_dof (int): Degree of freedom for the process noise model
        meas_noise_dof (int): Degree of freedom for the measurement noise model
    """

    def __init__(self, **kwargs):
        self.scale = np.array([[]])
        self.dof = 3
        self.proc_noise_dof = 3
        self.meas_noise_dof = 3

        super().__init__(**kwargs)

    @property
    def cov(self):
        """ Covariance matrix calculated from the scale matrix and degree of
        freedom. This is read only

        Raises:
            RuntimeError: If the degree of freedom is less than or equal to 2

        Returns:
            N x 1 numpy array: Calcualted covariance matrix
        """
        if self.dof <= 2:
            msg = "Degrees of freedom ({}) must be > 2"
            raise RuntimeError(msg.format(self.dof))
        return self.dof / (self.dof - 2) * self.scale

    @cov.setter
    def cov(self, cov):
        pass

    def predict(self, **kwargs):
        """ This implements the prediction step of the Students T filter.

        Keyword Args:
            cur_state (N x 1 numpy array): current state

        Returns:
            next_state (N x 1 numpy array): Next state

        """
        cur_state = kwargs['cur_state']

        state_mat = self.get_state_mat(**kwargs)
        next_state = state_mat @ cur_state

        factor = self.proc_noise_dof * (self.dof - 2) \
            / (self.dof * (self.proc_noise_dof - 2))
        self.scale = state_mat.T @ self.scale @ state_mat \
            + factor * self.get_proc_noise(**kwargs)

        return next_state

    def correct(self, **kwargs):
        """ Implements the correction step of the students T filter, and the
        moment matching.

        Keyword Args:
            cur_state (N x 1 numpy array): current state
            meas (M x 1 numpy array): current measurement

        Returns:
            tuple containing

                - (N x 1 numpy array): Corrected state
                - (float): Measurement fit probability
        """
        def pdf(x, mu, sig, v):
            d = x.size
            del2 = (x - mu).T @ la.inv(sig) @ (x - mu)
            inv_det = 1 / np.sqrt(la.det(sig))
            gam_rat = gmath.gamma_fnc(np.floor((v + d) / 2)) \
                / gmath.gamma_fnc(np.floor(v / 2))
            return gam_rat / (v * np.pi)**(d/2) * inv_det \
                * (1 + del2 / v)**(-(v + d) / 2)

        cur_state = kwargs['cur_state']
        meas = kwargs['meas']

        meas_mat = self.get_meas_mat(cur_state, **kwargs)

        factor = self.meas_noise_dof * (self.dof - 2) \
            / (self.dof * (self.meas_noise_dof - 2))
        P_zz = meas_mat @ self.scale @ meas_mat.T + factor * self.meas_noise
        inv_P_zz = la.inv(P_zz)
        gain = self.scale @ meas_mat.T @ inv_P_zz
        P_kk = self.scale - gain @ meas_mat @ self.scale
        est_meas = self.get_est_meas(cur_state, **kwargs)
        innov = (meas - est_meas)
        delta_2 = innov.T @ inv_P_zz @ innov
        next_state = cur_state + gain @ innov

        factor = (self.dof + delta_2) / (self.dof + meas.size)
        P_k = factor * P_kk
        dof_p = self.dof + meas.size

        # moment matching
        factor = dof_p * (self.dof - 2) / (self.dof * (dof_p - 2))
        self.scale = factor * P_k

        meas_fit_prob = pdf(meas, est_meas, P_zz, self.meas_noise_dof)
        meas_fit_prob = meas_fit_prob.item()

        return (next_state, meas_fit_prob)


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

    def __init__(self, **kwargs):
        self.dyn_fnc = None
        self.meas_likelihood_fnc = None
        self.proposal_sampling_fnc = None
        self.proposal_fnc = None

        self._particleDist = gdistrib.ParticleDistribution()
        self._prop_parts = []
        super().__init__(**kwargs)

    @property
    def cov(self):
        """ The covariance of the particles
        """
        return self._particleDist.covariance

    @cov.setter
    def cov(self, x):
        pass

    def init_from_dist(self, dist):
        self._particleDist = deepcopy(dist)

    def extract_dist(self):
        return deepcopy(self._particleDist)

    @property
    def num_particles(self):
        """ The number of particles used by the filter
        """
        return self._particleDist.num_particles

    def set_meas_mat(self, **kwargs):
        warn("Particle filter does not use set_meas_mat")

    def init_particles(self, particle_lst):
        """ Initializes the particles

        Args:
            particle_lst (list): List of numpy arrays, one for each particle.
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

    def predict(self, **kwargs):
        """ Predicts the next state

        Args:
            **kwargs (kwargs): Passed through to the dynamics function.

        Raises:
            RuntimeError: If the dynamics function is not set.

        Returns:
            numpy array: The predicteed state.

        """
        if self.dyn_fnc is not None:
            self._prop_parts = [self.dyn_fnc(p.point, **kwargs)
                                for (p, w) in self._particleDist]
        else:
            msg = 'Predict function not implemented when dyn_fnc is None'
            raise RuntimeError(msg)

        new_parts = [self.proposal_sampling_fnc(x, **kwargs)
                     for x in self._prop_parts]
        self._update_particles(new_parts)
        return self._calc_state()

    def correct(self, meas, **kwargs):
        """ Corrects the state estimate

        Args:
            meas (N x 1 numpy array): The measurement.
            **kwargs (kwargs): Passed through to the measurement, relative
                likelihood, proposal, and selection functions.

        Todo:
            Add the meas_fit_prob calculation

        Returns:
            Tuple containing

                - (N x 1 numpy array): the corrected state
                - (list): Unnormalized relative likelihood of the particles
                after resampling
        """

        # calculate weights
        est_meas = [self.get_est_meas(p.point, **kwargs)
                    for (p, w) in self._particleDist]
        self._calc_weights(meas, est_meas, self._prop_parts, **kwargs)

        inds_removed = self._selection(**kwargs)

        est_meas = [self.get_est_meas(p.point, **kwargs)
                    for p, w in self._particleDist]
        rel_likeli = self._calc_relative_likelihoods(meas, est_meas,
                                                     renorm=False, **kwargs)

        return (self._calc_state(), rel_likeli, inds_removed)

    def _calc_weights(self, meas, est_meas, conditioned_lst, **kwargs):
        rel_likeli = self._calc_relative_likelihoods(meas, est_meas,
                                                     renorm=False, **kwargs)
        prop_fit = [self.proposal_fnc(x_hat, cond, **kwargs)
                    for x_hat, cond in zip(self._particleDist.particles,
                                           conditioned_lst)]
        weights = []
        for p_ii, q_ii in zip(rel_likeli, prop_fit):
            if q_ii < 1e-16:
                w = 0
            else:
                w = p_ii / q_ii
            weights.append(w)

        tot = np.sum(weights)

        if tot > 0:
            up_weights = [w / tot for w in weights]
        else:
            up_weights = weights
        self._particleDist.update_weights(up_weights)

    def _calc_relative_likelihoods(self, meas, est_meas, renorm=True,
                                   **kwargs):
        weights = [self.meas_likelihood_fnc(meas, y, **kwargs)
                   for y in est_meas]
        if renorm:
            tot = np.sum(weights)
            if tot > 0:
                weights = [qi / tot for qi in weights]
        return weights

    def _selection(self, **kwargs):
        rng = kwargs.get('rng', rnd.default_rng())

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
            f_hndl.axes[0].hist2d(x, y, **h_opts)

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
    """ This provides a generic base class for Particle filters that have an
    optional Markov Chain Monte Carlo move step.

    Attributes:
        use_MCMC (bool): Flag indicating if move step is run
        sampler (:mod:`gncpy.sampling`): Class instance from the sampling
            module. Must already be setup and have a sample method.
    """

    def __init__(self, **kwargs):
        self.use_MCMC = False
        self.sampler = None

        super().__init__(**kwargs)

    def move_particles(self, **kwargs):
        raise RuntimeError('Must implement thid function in derived class')


class UnscentedParticleFilter(MCMCParticleFilterBase):
    """ This implements an unscented kalman filter.
    For details on the filter see
    :cite:`VanDerMerwe2000_TheUnscentedParticleFilter` and
    :cite:`VanDerMerwe2001_TheUnscentedParticleFilter`.
    """

    def __init__(self, **kwargs):
        self._filt = UnscentedKalmanFilter()
        self._dyn_fnc = None
        self._sig_points = []

        super().__init__(**kwargs)

    def set_meas_model(self, fnc):
        self._filt.set_meas_model(fnc)

    def set_proc_noise(self, **kwargs):
        self._filt.set_proc_noise(**kwargs)

    @property
    def meas_noise(self):
        return self._filt.meas_noise

    @meas_noise.setter
    def meas_noise(self, meas_noise):
        self._filt.meas_noise = meas_noise

    @property
    def dyn_fnc(self):
        return self._dyn_fnc

    @dyn_fnc.setter
    def dyn_fnc(self, f):
        self._filt.dyn_fnc = f
        self._dyn_fnc = f

    def predict(self, **kwargs):
        newDist = gdistrib.ParticleDistribution()
        for p, w in self._particleDist:
            part = gdistrib.Particle()
            self._filt.cov = p.uncertainty
            self._filt._stateSigmaPoints = deepcopy(p.sigmaPoints)
            part.point = self._filt.predict(cur_state=p.point, **kwargs)
            part.uncertainty = self._filt.cov
            part.sigmaPoints = deepcopy(self._filt._stateSigmaPoints)
            newDist.add_particle(part, 1 / self._particleDist.num_particles)

        self._particleDist = newDist
        return self._calc_state()

    def correct(self, meas, **kwargs):
        rng = kwargs.get('rng', rnd.default_rng())
        prop_samp_kw = deepcopy(kwargs)
        if 'rng' not in prop_samp_kw:
            prop_samp_kw['rng'] = rng

        if self.use_MCMC:
            move_kw = deepcopy(kwargs)
            if 'rng' not in move_kw:
                move_kw['rng'] = rng
            old_parts = deepcopy(self._particleDist.particles)
            old_covs = deepcopy(self._covs)
            old_sig_points = deepcopy(self._sig_points)

        # call UKF correction on each particle
        self._prop_parts = []
        newDist = gdistrib.ParticleDistribution()
        for p, w in self._particleDist:
            part = gdistrib.Particle()

            self._filt.cov = p.uncertainty
            self._filt._stateSigmaPoints = deepcopy(p.sigmaPoints)
            ns = self._filt.correct(cur_state=p.point, meas=meas, **kwargs)[0]
            cov = self._filt.cov.copy()

            self._prop_parts.append(ns)
            samp = self.proposal_sampling_fnc(ns, cov=cov, **prop_samp_kw)

            part.point = samp
            part.uncertainty = cov.copy()
            part.sigmaPoints = deepcopy(self._filt._stateSigmaPoints)

            newDist.add_particle(part, 1 / self._particleDist.num_particles)

        # update info for next UKF
        self._particleDist = newDist

        # resample/selection
        est_meas = [self._filt.get_est_meas(p.point, **kwargs)
                    for p, w in self._particleDist]
        cov_lst = [p.uncertainty.copy() for p, w in self._particleDist]
        self._calc_weights(meas, est_meas, self._prop_parts, cov_lst)

        inds_removed = self._selection(**kwargs)

        if self.use_MCMC:
            self.move_particles(old_parts, old_covs, old_sig_points,  meas,
                                **move_kw)

        est_meas = [self._filt.get_est_meas(p.point, **kwargs)
                    for p, w in self._particleDist]
        rel_likeli = self._calc_relative_likelihoods(meas, est_meas,
                                                     renorm=False)

        return (self._calc_state(), rel_likeli, inds_removed)

    def _calc_weights(self, meas, est_meas, conditioned_lst, cov_lst):
        rel_likeli = self._calc_relative_likelihoods(meas, est_meas,
                                                      renorm=False)
        prop_fit = [self.proposal_fnc(x_hat, cond, cov=p_hat)
                    for x_hat, cond, p_hat in zip(self._particleDist.particles,
                                                  conditioned_lst, cov_lst)]
        weights = []
        for p_ii, q_ii in zip(rel_likeli, prop_fit):
            if q_ii < 1e-16:
                w = np.inf
            else:
                w = p_ii / q_ii
            weights.append(w)

        tot = np.sum(weights)

        if tot > 0 and tot != np.inf:
            up_weights = [w / tot for w in weights]
        else:
            up_weights = weights
        self._particleDist.update_weights(up_weights)

    def move_particles(self, old_parts, old_covs, old_sig_points, meas,
                       **kwargs):
        rng = kwargs['rng']
        new_parts = []
        new_covs = []
        new_sig_points = []

        accept_prob = rng.random()
        for ii, (x, P) in enumerate(zip(old_parts, old_covs)):
            self._filt.cov = P
            self._filt._stateSigmaPoints.points = old_sig_points[ii]
            ns = self._filt.correct(cur_state=x, meas=meas, **kwargs)[0]
            cov = self._filt.cov

            cand = self.proposal_sampling_fnc(ns, cov=cov, **kwargs)

            est_cand_meas = self._filt.get_est_meas(cand, **kwargs)
            cand_likeli = self._calc_relative_likelihoods(meas,
                                                          [est_cand_meas],
                                                          renorm=False)[0]
            cand_fit = self.proposal_fnc(cand, ns, cov=cov)

            part = self._particleDist.particles[ii].copy()
            part_likeli = self._calc_relative_likelihoods(meas,
                                                          [part],
                                                          renorm=False)[0]
            part_fit = self.proposal_fnc(part, ns, cov=cov)

            num = cand_likeli * part_fit
            den = part_likeli * cand_fit

            if den < np.finfo(np.float32).eps:
                ratio = 0.0
            else:
                ratio = num / den
            accept_val = np.min((ratio, 1))

            if accept_prob <= accept_val:
                new_parts.append(cand)
                new_covs.append(cov)
                new_sig_points.append(self._filt._stateSigmaPoints.points)
            else:
                new_parts.append(part)
                new_covs.append(self._covs[ii])
                new_sig_points.append(self._sig_points)

        self._update_particles(new_parts)
        self._covs = new_covs
        self._sig_points = new_sig_points


class MaxCorrEntUPF(UnscentedParticleFilter):
    """ This implements a Maximum Correntropy Unscented Particle Filter.

    This is based on
    :cite:`Fan2018_MaximumCorrentropyBasedUnscentedParticleFilterforCooperativeNavigationwithHeavyTailedMeasurementNoises`

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._filt = MaxCorrEntUKF()

    def correct(self, meas, past_state, **kwargs):
        return super().correct(meas, past_state=past_state, **kwargs)

    @property
    def kernel_bandwidth(self):
        """ Bandwidth for the Gaussian Kernel in the MCUKF

        Returns:
            float: bandwidth

        """
        return self._filt.kernel_bandwidth

    @kernel_bandwidth.setter
    def kernel_bandwidth(self, kernel_bandwidth):
        self._filt.kernel_bandwidth = kernel_bandwidth
