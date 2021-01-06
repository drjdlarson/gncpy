import numpy as np
import numpy.random as rnd
import scipy.linalg as la
# import abc
from warnings import warn

import gncpy.math as gmath
# import gncpy.utilities as util


class DynamicsBase:
    """ This defines common attributes for all dynamics models.

    Attributes:
        state_names (tuple): Tuple of strings for the name of each state. The
            order should match that of the state vector.
    """
    # __metaclass__ = util.ClassPropertyMetaClass
    state_names = ()


class LinearDynamicsBase(DynamicsBase):
    """ Base class for all linear dynamics models.
    """

    def get_dis_process_noise_mat(self, dt, **kwargs):
        """ Class method for getting the process noise. Must be overridden in
        child classes.

        Args:
            dt (float): delta time.
            **kwargs (dict): any additional arguments needed.

        Returns:
            2d numpy array: discrete time process noise matrix.

        """
        msg = 'get_dis_process_noise_mat function is undefined'
        warn(msg, RuntimeWarning)
        return np.array([[]])

    def get_state_mat(self, **kwargs):
        """ Class method for getting the discrete time state matrix. Must be
        overridden in child classes.

        Args:
            **kwargs (TYPE): any additional arguments needed.

        Returns:
            2d numpy array: state matrix.

        """
        msg = 'get_state_mat function is undefined'
        warn(msg, RuntimeWarning)
        return np.array([[]])


class NonlinearDynamicsBase(LinearDynamicsBase):
    """ Base class for all non-linear dynamics models.
    """
    @property
    def cont_fnc_lst(self):
        r""" Class property for the continuous time dynamics functions. Must be
        overridden in the child classes.

        This is a list of functions that implement differential equations
        :math:`\dot{x} = f(x, u)` for each state, in order.

        Returns:
            list: functions that take an N x 1 numpy array for the state, an
            N x Nu numpy array for control input, and other arguments as kwargs
        """
        msg = 'cont_fnc_lst not implemented'
        warn(msg, RuntimeWarning)
        return []

    @property
    def disc_fnc_lst(self):
        """ Class property for the discrete time dynamics functions.
        Automatically generates the list by integrating the continuous time
        dynamics functions.

        Returns:
            list: functions for each state variable that take an N x 1 numpy
            array for the state, an N x Nu numpy array for control input,
            delta time, and other arguments as kwargs.

        """
        lst = []

        def g(f):
            return lambda x, u, dt, **kwargs: gmath.rk4(f, x, dt, u=u,
                                                        **kwargs)

        for f in self.cont_fnc_lst:
            lst.append(g(f))
        return lst

    def cont_dyn(self, x, **kwargs):
        r""" This implements the continuous time dynamics based on the supplied
        continuous function list.

        This implements the equation :math:`\dot{x} = f(x, u)` and returns the
        state derivatives as a vector

        Args:
            x (N x 1 numpy array): current state.
            **kwargs (dict): Passed through to the dynamics function.

        Returns:
            out (N x 1 numpy array): derivative of next state.

        """
        u = kwargs['cur_input']
        out = np.zeros((len(self.state_names), 1))
        for ii, f in enumerate(self.cont_fnc_lst):
            out[ii] = f(x, u, **kwargs)

        return out

    def propagate_state(self, x, u, dt, add_noise=False, **kwargs):
        r"""This propagates the continuous time dynamics based on the supplied
        continuous function list.

        This implements the equation :math:`x_{k+1} = \int \dot{x}_k dt` and
        returns the next state as a vector

        Args:
            x (N x 1 numpy array): current state.
            u (N x Nu numpy array): DESCRIPTION.
            dt (float): delta time.
            add_noise (bool, optional): flag indicating if noise should be
                added to the output. Defaults to False.
            **kwargs (dict): Passed through to the dynamics function.

        Keyword Args:
            rng (random.default_rng generator): Random number generator.
                Only used if adding noise

        Returns:
            ns (TYPE): DESCRIPTION.

        """
        ns = gmath.rk4(self.cont_dyn, x.copy(), dt, cur_input=u, **kwargs)
        if add_noise:
            rng = kwargs.get('rng', rnd.default_rng(1))
            proc_mat = self.get_dis_process_noise_mat(dt, **kwargs)
            ns += proc_mat @ rng.standard_normal(ns.shape)
        return ns

    def get_state_mat(self, x, u, dt, **kwargs):
        """ Calculates the jacobian of the differential equations.

        Args:
            x (N x 1 numpy array): current state.
            u (Nu x 1 numpy array): DESCRIPTION.
            dt (float): delta time.
            **kwargs (dict): Passed through to the dynamics function.

        Returns:
            TYPE: DESCRIPTION.

        """
        return gmath.get_state_jacobian(x, u, self.cont_fnc_lst, dt=dt,
                                        **kwargs)


class CoordinatedTurn(NonlinearDynamicsBase):
    """ This implements the non-linear coordinated turn dynamics model.
    """
    state_names = ('x pos', 'x vel', 'y pos', 'y vel', 'turn angle')

    def cont_fnc_lst(self):
        # returns x_dot
        def f0(x, u, **kwargs):
            return x[1]

        # returns x_dot_dot
        def f1(x, u, **kwargs):
            return -x[4] * x[3]

        # returns y_dot
        def f2(x, u, **kwargs):
            return x[3]

        # returns y_dot_dot
        def f3(x, u, **kwargs):
            return x[4] * x[1]

        # returns omega_dot
        def f4(x, u, **kwargs):
            return 0

        return [f0, f1, f2, f3, f4]

    def get_dis_process_noise_mat(self, dt, **kwargs):
        pos_std = kwargs['pos_std']
        turn_std = kwargs['turn_std']

        G = np.array([[dt**2 / 2, 0, 0],
                      [dt, 0, 0],
                      [0, dt**2 / 2, 0],
                      [0, dt, 0],
                      [0, 0, 1]])
        Q = la.block_diag(pos_std**2 * np.eye(2), np.array([[turn_std**2]]))
        return G @ Q @ G.T


class ClohessyWiltshireOrbit(LinearDynamicsBase):
    """Implements the Clohessy Wiltshire orbit model.

    This must be instantiated so individual attributes can be assigned. This
    based on :cite:`Clohessy1960_TerminalGuidanceSystemforSatelliteRendezvous`
    and :cite:`Desai2013_AComparativeStudyofEstimationModelsforSatelliteRelativeMotion`

    Attritbutes:
        dt (float): delta time.
        mean_motion (float): mean motion
    """

    state_names = ('x position', 'y position', 'z position',
                   'x velocity', 'y velocity', 'z velocity')

    def __init__(self):
        self.dt = 0
        self.mean_motion = 0

    def get_dis_process_noise_mat(self, **kwargs):
        """ Class method for returning the process noise.

        Returns:
            2d numpy array: discrete time process noise matrix.

        """
        return np.zeros((len(self.state_names), len(self.state_names)))

    def get_state_mat(self, **kwargs):
        """ Class method for getting the discrete time state matrix. Must be
        overridden in child classes.

        Returns:
            2d numpy array: state matrix.

        """
        dt = self.dt
        n = self.mean_motion
        F = np.array([[4 - 3*np.cos(dt * n), 0, 0, np.sin(dt * n) / n,
                       -(2 * np.cos(dt * n) - 2) / n, 0],
                      [6 * np.sin(dt*n) - 6 * dt * n, 1, 0,
                       (2 * np.cos(dt * n) - 2) / n,
                       (4 * np.sin(dt * n) - 3 * dt * n) / n, 0],
                      [0, 0, np.cos(dt * n), 0, 0, np.sin(dt * n) / n],
                      [3 * n * np.sin(dt * n), 0, 0, np.cos(dt * n),
                       2 * np.sin(dt * n), 0],
                      [6 * n * (np.cos(dt * n) - 1), 0, 0, -2 * np.sin(dt * n),
                       4 * np.cos(dt * n) - 3, 0],
                      [0, 0, -n * np.sin(dt * n), 0, 0, np.cos(dt * n)]])
        return F


class TschaunerHempelOrbit(NonlinearDynamicsBase):
    """This implements the non-linear Tschauner-Hempel elliptical orbit model.

    It is the general elliptical orbit of an object around another target
    object as defined in
    :cite:`Tschauner1965_RendezvousZuEineminElliptischerBahnUmlaufendenZiel`.
    The states are defined as positions in a
    Local-Vertical-Local-Horizontal (LVLH) frame. Note, the true anomaly is
    that of the target object. For more details see
    :cite:`Okasha2013_GuidanceNavigationandControlforSatelliteProximityOperationsUsingTschaunerHempelEquations`
    and :cite:`Lange1965_FloquetTheoryOrbitalPerturbations`

    Attributes:
        mu (float): gravitational parameter in m^3 s^-2
        semi_major (float): semi-major axis in meters
        eccentricity (float): eccentricity
    """
    state_names = ('x position', 'y position', 'z position',
                   'x velocity', 'y velocity', 'z velocity',
                   'targets true anomaly')

    def __init__(self, **kwargs):
        self.mu = kwargs.get('mu', 3.986004418 * 10**14)
        self.semi_major = kwargs.get('semi_major', 0)
        self.eccentricity = kwargs.get('eccentricity', 1)

    @property
    def cont_fnc_lst(self):
        # returns x velocity
        def f0(x, u, **kwargs):
            return x[3]

        # returns y velocity
        def f1(x, u, **kwargs):
            return x[4]

        # returns z velocity
        def f2(x, u, **kwargs):
            return x[5]

        # returns x acceleration
        def f3(x, u, **kwargs):
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
        def f4(x, u, **kwargs):
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
        def f5(x, u, **kwargs):
            e = self.eccentricity
            a = self.semi_major
            mu = self.mu

            e2 = e**2
            R3 = ((a * (1 - e2)) / (1 + e * np.cos(x[6])))**3

            C1 = mu / R3

            return -C1 * x[2]

        # returns true anomaly ROC
        def f6(x, u, **kwargs):
            e = self.eccentricity
            a = self.semi_major
            p = a * (1 - e**2)

            H = np.sqrt(self.mu * p)
            R = p / (1 + e * np.cos(x[6]))
            return H / R**2

        return [f0, f1, f2, f3, f4, f5]

    def get_dis_process_noise_mat(self, **kwargs):
        """ Returns the process noise.

        Returns:
            2d numpy array: discrete time process noise matrix.

        """
        return np.zeros((len(self.state_names), len(self.state_names)))


class KarlgaardOrbit(NonlinearDynamicsBase):
    """ This implements the non-linear Karlgaar elliptical orbit model.

    It uses the numerical integration of the second order approximation in
    dimensionless sphereical coordinates. See
    :cite:`Karlgaard2003_SecondOrderRelativeMotionEquations` for details.
    """
    state_names = ('non-dim radius', 'non-dim az angle', 'non-dim elv angle',
                   'non-dim radius ROC', 'non-dim az angle ROC',
                   'non-dim elv angle ROC')

    def cont_fnc_lst(self):
        # returns non-dim radius ROC
        def f0(x, u, **kwargs):
            return x[3]

        # returns non-dim az angle ROC
        def f1(x, u, **kwargs):
            return x[4]

        # returns non-dim elv angle ROC
        def f2(x, u, **kwargs):
            return x[5]

        # returns non-dim radius ROC ROC
        def f3(x, u, **kwargs):
            r = x[0]
            phi = x[2]
            theta_d = x[4]
            phi_d = x[5]
            return ((-3 * r**2 + 2 * r * theta_d - phi**2 + theta_d**2
                    + phi_d**2) + 3 * r + 2 * theta_d)

        # returns non-dim az angle ROC ROC
        def f4(x, u, **kwargs):
            r = x[0]
            theta = x[1]
            r_d = x[3]
            theta_d = x[4]
            return ((2 * r * r_d + 2 * theta * theta_d - 2 * theta_d * r_d)
                    - 2 * r_d)

        # returns non-dim elv angle ROC ROC
        def f5(x, u, **kwargs):
            phi = x[2]
            r_d = x[3]
            theta_d = x[4]
            phi_d = x[5]
            return ((-2 * theta_d * phi - 2 * phi_d * r_d) - phi)

        return [f0, f1, f2, f3, f4, f5]

    def get_dis_process_noise_mat(self, **kwargs):
        """ Returns the process noise.

        Returns:
            2d numpy array: discrete time process noise matrix.

        """
        return np.zeros((len(self.state_names), len(self.state_names)))



# class DynamicObject(metaclass=abc.ABCMeta):
#     """ Base class for dynamic objects.

#     This defaults to assuming nonlinear dynamics, and automates the
#     linearization, and discritization of a list of continuous time dynamics
#     functions.

#     Attributes:
#         nom_ctrl (Nu x 1 numpy array): Nominal control input
#     """
#     def __init___(self, **kwargs):
#         self.nom_ctrl = kwargs.get('nom_ctrl', np.array([[]]))

#     @property
#     @abc.abstractmethod
#     def cont_dyn_funcs(self):
#         """ List of continuous time dynamics functions.

#         Must be defined in child class, one element per state, in order, must
#         take at least the following arguments (in order): state, control input
#         """
#         pass

#     @property
#     def disc_dyn_funcs(self):
#         """ List of discrete time dynamics functions

#         one element per state, in order, integration of continuous time, each
#         function takes the following arguments (in order) state, control input,
#         time step
#         """
#         lst = []

#         def g(f):
#             return lambda x, u, dt: gmath.rk4(f, x, dt, u=u)

#         for f in self.cont_dyn_funcs:
#             lst.append(g(f))
#         return lst

#     @property
#     def disc_inv_dyn_funcs(self):
#         """ List of discrete time inverse dynamics functions
#         """
#         lst = []

#         def g_bar(f):
#             return lambda x, u, dt: gmath.rk4_backward(f, x, dt, u=u)

#         for f in self.cont_dyn_funcs:
#             lst.append(g_bar(f))
#         return lst

#     def get_disc_state_mat(self, state, u, dt, **kwargs):
#         """ Returns the discrete time state matrix.

#         This assumes the dynamics functions are nonlinear and calculates the
#         jacobian. If this is not the case, the child class should override
#         the implementation.

#         Args:
#             state (N x 1 numpy array): Current state
#             u (Nu x 1 numpy array): Current control input
#             dt (float): time step
#             kwargs : any additional arguments needed by the dynamics functions
#         Returns:
#             (N x N numpy array): State matrix
#         """
#         return gmath.get_state_jacobian(state, u, self.disc_dyn_funcs, dt=dt,
#                                         **kwargs)

#     def get_disc_input_mat(self, state, u, dt, **kwargs):
#         """ Returns the discrete time input matrix.

#         This assumes the dynamics functions are nonlinear and calculates the
#         jacobian. If this is not the case, the child class should override
#         the implementation.

#         Args:
#             state (N x 1 numpy array): Current state
#             u (Nu x 1 numpy array): Current control input
#             dt (float): time step
#             kwargs : any additional arguments needed by the dynamics functions
#         Returns:
#             (N x Nu numpy array): Input matrix
#         """
#         return gmath.get_input_jacobian(state, u, self.disc_dyn_funcs, dt=dt,
#                                         **kwargs)
