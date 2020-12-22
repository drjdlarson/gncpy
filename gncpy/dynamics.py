import numpy as np
import numpy.random as rnd
import scipy.linalg as la
# import abc
from warnings import warn

import gncpy.math as gmath
import gncpy.utilities as util


class DynamicsBase:
    """ This defines common attributes for all dynamics models.

    Attributes:
        state_names (tuple): Tuple of strings for the name of each state. The
            order should match that of the state vector.
    """
    __metaclass__ = util.ClassPropertyMetaClass
    state_names = ()


class LinearDynamicsBase(DynamicsBase):
    """ Base class for all linear dynamics models.
    """
    @classmethod
    def get_dis_process_noise_mat(cls, dt, **kwargs):
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

    @classmethod
    def get_state_mat(cls, **kwargs):
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
    @util.classproperty
    def cont_fnc_lst(cls):
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

    @util.classproperty
    def disc_fnc_lst(cls):
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

        for f in cls.cont_fnc_lst:
            lst.append(g(f))
        return lst

    @classmethod
    def cont_dyn(cls, x, **kwargs):
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
        out = np.zeros((len(cls.state_names), 1))
        for ii, f in enumerate(cls.cont_fnc_lst):
            out[ii] = f(x, u, **kwargs)

        return out

    @classmethod
    def propagate_state(cls, x, u, dt, add_noise=False, **kwargs):
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
        ns = gmath.rk4(cls.cont_dyn, x.copy(), dt, cur_input=u, **kwargs)
        if add_noise:
            rng = kwargs.get('rng', rnd.default_rng(1))
            proc_mat = cls.get_dis_process_noise_mat(dt, **kwargs)
            ns += proc_mat @ rng.standard_normal(ns.shape)
        return ns

    @classmethod
    def get_state_mat(cls, x, u, dt, **kwargs):
        """ Calculates the jacobian of the differential equations.

        Args:
            x (N x 1 numpy array): current state.
            u (N x Nu numpy array): DESCRIPTION.
            dt (float): delta time.
            **kwargs (dict): Passed through to the dynamics function.

        Returns:
            TYPE: DESCRIPTION.

        """
        return gmath.get_state_jacobian(x, u, cls.cont_fnc_lst, dt=dt,
                                        **kwargs)


class CoordinatedTurn(NonlinearDynamicsBase):
    """ This implements the non-linear coordinated turn dynamics model.
    """
    state_names = ('x pos', 'x vel', 'y pos', 'y vel', 'turn angle')

    @util.classproperty
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

    @classmethod
    def get_dis_process_noise_mat(cls, dt, **kwargs):
        pos_std = kwargs['pos_std']
        turn_std = kwargs['turn_std']

        G = np.array([[dt**2 / 2, 0, 0],
                      [dt, 0, 0],
                      [0, dt**2 / 2, 0],
                      [0, dt, 0],
                      [0, 0, 1]])
        Q = la.block_diag(pos_std**2 * np.eye(2), np.array([[turn_std**2]]))
        return G @ Q @ G.T


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
