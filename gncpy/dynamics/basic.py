"""Basic models and classes that can be extended.

These provide an easy to use interface for some common dynamic objects and
and their associated models. They have been designed to integrate well with the
filters in :mod:`gncpy.filters`.
"""
from abc import abstractmethod, ABC
import numpy as np
import scipy.integrate as s_integrate
from warnings import warn

import gncpy.math as gmath


class DynamicsBase(ABC):
    r"""Defines common attributes for all dynamics models.

    Attributes
    ----------
    control_model : callable or list of callables, optional
        For objects of :class:`gncpy.dynamics.LinearDynamicsBase` it is a
        callable with the signature `t, x, *ctrl_args` and returns the
        input matrix :math:`G_k` from :math:`x_{k+1} = F_k x_k + G_k u_k`.
        For objects of :class:`gncpy.dynamics.NonlinearDynamicsBase` it is a
        list of callables where each callable returns the modification to the
        corresponding state, :math:`g(t, x_i, u_i)`, in the differential equation
        :math:`\dot{x}_i = f(t, x_i) + g(t, x_i, u_i)` and has the signature
        `t, u, *ctrl_args`.
    state_constraint : callable
        Has the signature `t, x` where `t` is the current timestep and `x`
        is the propagated state. It returns the propagated state with
        any constraints applied to it.
    """

    __slots__ = 'control_model', 'state_constraint'

    state_names = ()
    """Tuple of strings for the name of each state. The order should match
    that of the state vector.
    """

    def __init__(self, control_model=None, state_constraint=None):
        super().__init__()
        self.control_model = control_model
        self.state_constraint = state_constraint

    @abstractmethod
    def propagate_state(self, *args, **kwargs):
        """Abstract method for propagating the state forward in time.

        Must be overridden in child classes.

        Parameters
        ----------
        *args : tuple
            Specifics defined by child class.
        **kwargs : dict
            Specifics defined by child class.

        Raises
        ------
        NotImplementedError
            If child class does not implement this function.

        Returns
        -------
        next_state : N x 1 numpy array
            The propagated state.
        """
        raise NotImplementedError


class LinearDynamicsBase(DynamicsBase):
    """Base class for all linear dynamics models.

    Child classes should define their own get_state_mat function and set the
    state names class variable. The remainder of the functions autogenerate
    based on these values.

    """

    __slots__ = ()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @abstractmethod
    def get_state_mat(self, timestep, *args):
        """Abstract method for getting the discrete time state matrix.

        Must be overridden in child classes.

        Parameters
        ----------
        timestep : float
            timestep.
        args : tuple, optional
            any additional arguments needed.

        Returns
        -------
        N x N numpy array
            state matrix.
        """
        raise NotImplementedError

    def get_dis_process_noise_mat(self, dt, *f_args):
        """Class method for getting the process noise.

        Should be overridden in child classes. Should maintain the same
        signature to allow for standardized wrappers.

        Parameters
        ----------
        dt : float
            delta time.
        *kwargs : tuple, optional
            any additional arguments needed.

        Returns
        -------
        N x N numpy array
            discrete time process noise matrix.

        """
        msg = 'get_dis_process_noise_mat function is undefined'
        warn(msg, RuntimeWarning)
        return np.array([[]])

    def propagate_state(self, timestep, state, u=None, state_args=None, ctrl_args=None):
        r"""Propagates the state to the next time step.

        Notes
        -----
        This implements the equation

        .. math::
            x_{k+1} = F_k x_k + G_k u_k

        Parameters
        ----------
        timestep : float
            timestep.
        state : N x 1 numpy array
            state vector.
        u : N x Nu numpy array, optional
            Control effort vector. The default is None.
        state_args : tuple, optional
            Additional arguments needed by the `get_state_mat` function. The
            default is ().
        ctrl_args : tuple, optional
            Additional arguments needed by the get input mat function. The
            default is (). Only used if a control effort is supplied.

        Returns
        -------
        next_state : N x 1 numpy array
            The propagated state.

        """
        if state_args is None:
            state_args = ()
        if ctrl_args is None:
            ctrl_args = ()

        state_trans_mat = self.get_state_mat(timestep, *state_args)
        next_state = state_trans_mat @ state

        if self.control_model is not None:
            input_mat = self.control_model(timestep, state, *ctrl_args)
            ctrl = input_mat @ u
            next_state += ctrl

        if self.state_constraint is not None:
            next_state = self.state_constraint(timestep, next_state)

        return next_state


class NonlinearDynamicsBase(DynamicsBase):
    """Base class for non-linear dynamics models.

    Child classes should define their own cont_fnc_lst property and set the
    state names class variable. The remainder of the functions autogenerate
    based on these values.

    Attributes
    ----------
    dt : float
        time difference for the integration period
    integrator_type : string, optional
        integrator type as defined by scipy's integrate.ode function. The
        default is `dopri5`.
    integrator_params : dict, optional
        additional parameters for the integrator. The default is {}.
    """

    __slots__ = ('dt', 'integrator_type', 'integrator_params', '_integrator')

    def __init__(self, integrator_type='dopri5', integrator_params={},
                 dt=np.nan, **kwargs):
        super().__init__(**kwargs)

        self.dt = dt
        self.integrator_type = integrator_type
        self.integrator_params = integrator_params

        self._integrator = None

    @property
    @abstractmethod
    def cont_fnc_lst(self):
        r"""Class property for the continuous time dynamics functions.

        Must be overridden in the child classes.

        Notes
        -----
        This is a list of functions that implement part of the differential
        equations :math:`\dot{x} = f(t, x) + g(t, u)` for each state, in order.
        These functions only represent the dynamics :math:`f(t, x)`.

        Returns
        -------
        list
            Each element is a function that take the timestep, the state, and
            *f_args. They must return the new state for the given index in the
            list/state vector.
        """
        raise NotImplementedError
        # msg = 'cont_fnc_lst not implemented'
        # warn(msg, RuntimeWarning)
        # return []

    def _cont_dyn(self, t, x, u, state_args, ctrl_args):
        r"""Implements the continuous time dynamics.

        This automatically sets up the combined differential equation based on
        the supplied continuous function list.

        This implements the equation :math:`\dot{x} = f(t, x) + g(t, u)` and
        returns the state derivatives as a vector. Note the control input is
        only used if an appropriate control model is set by the user.

        Parameters
        ----------
        t : float
            timestep.
        x : N x 1 numpy array
            current state.
        u : Nu x 1 numpy array
            Control effort, not used if no control model.
        state_args : tuple
            Additional arguements for the state functions.
        ctrl_args : tuple
            Additional arguments for the control functions. Not used if no
            control model

        Returns
        -------
        x_dot : N x 1 numpy array
            state derivative
        """
        out = np.zeros((len(self.state_names), 1))
        for ii, f in enumerate(self.cont_fnc_lst):
            out[ii] = f(t, x, *state_args)

        if self.control_model is not None:
            for ii, g in enumerate(self.control_model):
                out[ii] += g(t, x, u, *ctrl_args)

        return out

    def get_state_mat(self, timestep, state, *f_args):
        """Calculates the state matrix from the ode list.

        Parameters
        ----------
        timestep : float
            timestep.
        state : N x 1 numpy array
            current state.
        *f_args : tuple
            Additional arguments to pass to the ode functions.

        Returns
        -------
        N x N numpy array
            state transition matrix.
        """
        return gmath.get_state_jacobian(timestep, state, self.cont_fnc_lst,
                                        f_args)

    def propagate_state(self, timestep, state, u=None, state_args=None,
                        ctrl_args=None):
        """Propagates the continuous time dynamics.

        Automatically integrates the defined ode list and adds control inputs
        (held constant over the integration period).

        Parameters
        ----------
        timestep : float
            current timestep.
        state : N x 1 numpy array
            Current state vector.
        u : Nu x 1 numpy array, optional
            Current control effort. The default is None. Only used if a control
            model is set.
        state_args : tuple, optional
            Additional arguments to pass to the state odes. The default is ().
        ctrl_args : tuple, optional
            Additional arguments to pass to the control functions. The default
            is ().

        Raises
        ------
        RuntimeError
            If the integration fails.

        Returns
        -------
        next_state : N x 1 numpy array
            the state at time :math:`t + dt`.
        """
        if state_args is None:
            state_args = ()
        if ctrl_args is None:
            ctrl_args = ()

        self._integrator = s_integrate.ode(lambda t, y, *f_args:
                                           self._cont_dyn(t, y, u, f_args,
                                                          ctrl_args).flatten())
        self._integrator.set_integrator(self.integrator_type,
                                        **self.integrator_params)
        self._integrator.set_initial_value(state, timestep)
        self._integrator.set_f_params(*state_args)

        if np.isnan(self.dt) or np.isinf(self.dt):
            raise RuntimeError('Invalid value for dt ({}).'.format(self.dt))

        next_time = timestep + self.dt
        next_state = self._integrator.integrate(next_time)
        next_state = next_state.reshape((next_state.size, 1))
        if not self._integrator.successful():
            msg = 'Integration failed at time {}'.format(timestep)
            raise RuntimeError(msg)

        if self.state_constraint is not None:
            next_state = self.state_constraint(timestep, next_state)

        return next_state


class DoubleIntegrator(LinearDynamicsBase):
    """Implements a double integrator model."""

    __slots__ = ()

    state_names = ('x pos', 'y pos', 'x vel', 'y vel')

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_dis_process_noise_mat(self, dt, proc_cov):
        """Discrete process noise matrix.

        Parameters
        ----------
        dt : float
            time difference, unused.
        proc_cov : N x N numpy array
            Covariance matrix of the process noise.

        Returns
        -------
        4 x 4 numpy array
            process noise matrix.

        """
        gamma = np.array([0, 0, 1, 1]).reshape((4, 1))
        return gamma @ proc_cov @ gamma.T

    def get_state_mat(self, timestep, dt):
        """Class method for getting the discrete time state matrix.

        Parameters
        ----------
        timestep : float
            timestep.
        dt : float
            time difference

        Returns
        -------
        4 x 4 numpy array
            state matrix.

        """
        return np.array([[1., 0, dt, 0],
                         [0., 1., 0, dt],
                         [0, 0, 1., 0],
                         [0, 0, 0, 1]])


class CoordinatedTurn(NonlinearDynamicsBase):
    """Implements the non-linear coordinated turn dynamics model."""

    __slots__ = ()

    state_names = ('x pos', 'x vel', 'y pos', 'y vel', 'turn angle')

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def cont_fnc_lst(self):
        # returns x_dot
        def f0(t, x, *args):
            return x[1]

        # returns x_dot_dot
        def f1(t, x, *args):
            return -x[4] * x[3]

        # returns y_dot
        def f2(t, x, *args):
            return x[3]

        # returns y_dot_dot
        def f3(t, x, *args):
            return x[4] * x[1]

        # returns omega_dot
        def f4(t, x, *args):
            return 0

        return [f0, f1, f2, f3, f4]

    def get_dis_process_noise_mat(self, dt, posx_std, posy_std, turn_std):
        """Discrete process noise matrix.

        Parameters
        ----------
        dt : float
            time difference
        posx_std : float
            Standard deviation of the x position position process noise.
        posy_std : float
            Standard deviation of the y position position process noise.
        turn_std : float
            Standard deviation of the turn angle process noise.

        Returns
        -------
        5 x 5 numpy array
            Process noise matrix.

        """
        gamma = np.array([[dt**2 / 2, 0, 0],
                          [dt, 0, 0],
                          [0, dt**2 / 2, 0],
                          [0, dt, 0],
                          [0, 0, 1]])
        Q = np.diag([posx_std**2, posy_std**2, turn_std**2])
        return gamma @ Q @ gamma.T


class ClohessyWiltshireOrbit(LinearDynamicsBase):
    """Implements the Clohessy Wiltshire orbit model.

    This must be instantiated so individual attributes can be assigned. This
    based on :cite:`Clohessy1960_TerminalGuidanceSystemforSatelliteRendezvous`
    and :cite:`Desai2013_AComparativeStudyofEstimationModelsforSatelliteRelativeMotion`

    Attributes
    ----------
    mean_motion :float
        mean motion of reference spacecraft
    """

    state_names = ('x pos', 'y pos', 'z pos',
                   'x vel', 'y vel', 'z vel')

    def __init__(self, mean_motion=None, **kwargs):
        self.mean_motion = mean_motion

        super().__init__(**kwargs)

    def get_dis_process_noise_mat(self, dt):
        """Calculates the process noise.

        Parameters
        ----------
        dt : float
            time difference, not used but needed for standardized interface.

        Returns
        -------
        6 x 6 numpy array
            discrete time process noise matrix.
        """
        return np.zeros((len(self.state_names), len(self.state_names)))

    def get_state_mat(self, timestep, dt):
        """Calculates the state transition matrix.

        Parameters
        ----------
        timestep : float
            current timestep.
        dt : float
            time difference.

        Returns
        -------
        F : 6 x 6 numpy array
            state transition matrix.

        """
        n = self.mean_motion
        c_dtn = np.cos(dt * n)
        s_dtn = np.sin(dt * n)
        F = np.array([[4 - 3 * c_dtn, 0, 0, s_dtn / n, -(2 * c_dtn - 2) / n, 0],
                      [6 * s_dtn - 6 * dt * n, 1, 0, (2 * c_dtn - 2) / n,
                       (4 * s_dtn - 3 * dt * n) / n, 0],
                      [0, 0, c_dtn, 0, 0, s_dtn / n],
                      [3 * n * s_dtn, 0, 0, c_dtn, 2 * s_dtn, 0],
                      [6 * n * (c_dtn - 1), 0, 0, -2 * s_dtn, 4 * c_dtn - 3, 0],
                      [0, 0, -n * s_dtn, 0, 0, c_dtn]])
        return F


class TschaunerHempelOrbit(NonlinearDynamicsBase):
    """Implements the non-linear Tschauner-Hempel elliptical orbit model.

    Notes
    -----
    It is the general elliptical orbit of an object around another target
    object as defined in
    :cite:`Tschauner1965_RendezvousZuEineminElliptischerBahnUmlaufendenZiel`.
    The states are defined as positions in a
    Local-Vertical-Local-Horizontal (LVLH) frame. Note, the true anomaly is
    that of the target object. For more details see
    :cite:`Okasha2013_GuidanceNavigationandControlforSatelliteProximityOperationsUsingTschaunerHempelEquations`
    and :cite:`Lange1965_FloquetTheoryOrbitalPerturbations`

    Attributes
    ----------
    mu : float, optional
        gravitational parameter in :math:`m^3 s^{-2}`. The default is 3.986004418 * 10**14.
    semi_major : float
        semi-major axis in meters. The default is None.
    eccentricity : float, optional
        eccentricity. The default is 1.
    """

    state_names = ('x pos', 'y pos', 'z pos',
                   'x vel', 'y vel', 'z vel',
                   'targets true anomaly')

    def __init__(self, mu=3.986004418 * 10**14, semi_major=None, eccentricity=1,
                 **kwargs):
        self.mu = mu
        self.semi_major = semi_major
        self.eccentricity = eccentricity

        super().__init__(**kwargs)

    @property
    def cont_fnc_lst(self):
        # returns x velocity
        def f0(t, x, *args):
            return x[3]

        # returns y velocity
        def f1(t, x, *args):
            return x[4]

        # returns z velocity
        def f2(t, x, *args):
            return x[5]

        # returns x acceleration
        def f3(t, x, *args):
            e = self.eccentricity
            a = self.semi_major
            mu = self.mu

            e2 = e**2
            R3 = ((a * (1 - e2)) / (1 + e * np.cos(x[6])))**3
            n = np.sqrt(mu / a**3)

            C1 = mu / R3
            wz = n * (1 + e * np.cos(x[6]))**2 / (1 - e2)**(3. / 2.)
            wz_dot = -2 * mu * e * np.sin(x[6]) / R3

            return (wz**2 + 2 * C1) * x[0] + wz_dot * x[1] + 2 * wz * x[4]

        # returns y acceleration
        def f4(t, x, *args):
            e = self.eccentricity
            a = self.semi_major
            mu = self.mu

            e2 = e**2
            R3 = ((a * (1 - e2)) / (1 + e * np.cos(x[6])))**3
            n = np.sqrt(mu / a**3)

            C1 = mu / R3
            wz = n * (1 + e * np.cos(x[6]))**2 / (1 - e2)**(3. / 2.)
            wz_dot = -2 * mu * e * np.sin(x[6]) / R3

            return (wz**2 - C1) * x[1] - wz_dot * x[0] - 2 * wz * x[3]

        # returns z acceleration
        def f5(t, x, *args):
            e = self.eccentricity
            a = self.semi_major
            mu = self.mu

            e2 = e**2
            R3 = ((a * (1 - e2)) / (1 + e * np.cos(x[6])))**3

            C1 = mu / R3

            return -C1 * x[2]

        # returns true anomaly ROC
        def f6(t, x, *args):
            e = self.eccentricity
            a = self.semi_major
            p = a * (1 - e**2)

            H = np.sqrt(self.mu * p)
            R = p / (1 + e * np.cos(x[6]))
            return H / R**2

        return [f0, f1, f2, f3, f4, f5, f6]


class KarlgaardOrbit(NonlinearDynamicsBase):
    """Implements the non-linear Karlgaar elliptical orbit model.

    Notes
    -----
    It uses the numerical integration of the second order approximation in
    dimensionless sphereical coordinates. See
    :cite:`Karlgaard2003_SecondOrderRelativeMotionEquations` for details.
    """

    state_names = ('non-dim radius', 'non-dim az angle', 'non-dim elv angle',
                   'non-dim radius ROC', 'non-dim az angle ROC',
                   'non-dim elv angle ROC')

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def cont_fnc_lst(self):
        # returns non-dim radius ROC
        def f0(t, x, *args):
            return x[3]

        # returns non-dim az angle ROC
        def f1(t, x, *args):
            return x[4]

        # returns non-dim elv angle ROC
        def f2(t, x, *args):
            return x[5]

        # returns non-dim radius ROC ROC
        def f3(t, x, *args):
            r = x[0]
            phi = x[2]
            theta_d = x[4]
            phi_d = x[5]
            return ((-3 * r**2 + 2 * r * theta_d - phi**2 + theta_d**2
                    + phi_d**2) + 3 * r + 2 * theta_d)

        # returns non-dim az angle ROC ROC
        def f4(t, x, *args):
            r = x[0]
            theta = x[1]
            r_d = x[3]
            theta_d = x[4]
            return ((2 * r * r_d + 2 * theta * theta_d - 2 * theta_d * r_d)
                    - 2 * r_d)

        # returns non-dim elv angle ROC ROC
        def f5(t, x, *args):
            phi = x[2]
            r_d = x[3]
            theta_d = x[4]
            phi_d = x[5]
            return ((-2 * theta_d * phi - 2 * phi_d * r_d) - phi)

        return [f0, f1, f2, f3, f4, f5]