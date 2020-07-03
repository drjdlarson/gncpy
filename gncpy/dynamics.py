import numpy as np
import abc

from gncpy.math import rk4, rk4_backward, get_state_jacobian, \
    get_input_jacobian


class DynamicObject(metaclass=abc.ABCMeta):
    """ Base class for dynamic objects.

    This defaults to assuming nonlinear dynamics, and automates the
    linearization, and discritization of a list of continuous time dynamics
    functions.

    Attributes:
        nom_ctrl (Nu x 1 numpy array): Nominal control input
    """
    def __init___(self, **kwargs):
        self.nom_ctrl = kwargs.get('nom_ctrl', np.array([[]]))

    @property
    @abc.abstractmethod
    def cont_dyn_funcs(self):
        """ List of continuous time dynamics functions.

        Must be defined in child class, one element per state, in order, must
        take at least the following arguments (in order): state, control input
        """
        pass

    @property
    def disc_dyn_funcs(self):
        """ List of discrete time dynamics functions

        one element per state, in order, integration of continuous time, each
        function takes the following arguments (in order) state, control input,
        time step
        """
        lst = []

        def g(f):
            return lambda x, u, dt: rk4(f, x, dt, u=u)

        for f in self.cont_dyn_funcs:
            lst.append(g(f))
        return lst

    @property
    def disc_inv_dyn_funcs(self):
        """ List of discrete time inverse dynamics functions
        """
        lst = []

        def g_bar(f):
            return lambda x, u, dt: rk4_backward(f, x, dt, u=u)

        for f in self.cont_dyn_funcs:
            lst.append(g_bar(f))
        return lst

    def get_disc_state_mat(self, state, u, dt, **kwargs):
        """ Returns the discrete time state matrix.

        This assumes the dynamics functions are nonlinear and calculates the
        jacobian. If this is not the case, the child class should override
        the implementation.

        Args:
            state (N x 1 numpy array): Current state
            u (Nu x 1 numpy array): Current control input
            dt (float): time step
            kwargs : any additional arguments needed by the dynamics functions
        Returns:
            (N x N numpy array): State matrix
        """
        return get_state_jacobian(state, u, self.disc_dyn_funcs, dt=dt,
                                  **kwargs)

    def get_disc_input_mat(self, state, u, dt, **kwargs):
        """ Returns the discrete time input matrix.

        This assumes the dynamics functions are nonlinear and calculates the
        jacobian. If this is not the case, the child class should override
        the implementation.

        Args:
            state (N x 1 numpy array): Current state
            u (Nu x 1 numpy array): Current control input
            dt (float): time step
            kwargs : any additional arguments needed by the dynamics functions
        Returns:
            (N x Nu numpy array): Input matrix
        """
        return get_input_jacobian(state, u, self.disc_dyn_funcs, dt=dt,
                                  **kwargs)
