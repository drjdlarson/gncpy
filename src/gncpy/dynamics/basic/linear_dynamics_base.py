import numpy as np

from warnings import warn
from .dynamics_base import DynamicsBase

class LinearDynamicsBase(DynamicsBase):
    """Base class for all linear dynamics models.

    Child classes should define their own get_state_mat function and set the
    state names class variable. The remainder of the functions autogenerate
    based on these values.

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_input_mat(self, timestep, *ctrl_args):
        """Calculates the input matrix from the control model.

        This calculates the jacobian of the control model. If no control model
        is specified than it returns a zero matrix.

        Parameters
        ----------
        timestep : float
            current timestep.
        state : N x 1 numpy array
            current state.
        *ctrl_args : tuple
            Additional arguments to pass to the control model.

        Returns
        -------
        N x Nu numpy array
            Control input matrix.
        """
        if self._control_model is None:
            raise RuntimeWarning("Control model is not set.")
        return self._control_model(timestep, *ctrl_args)

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
        msg = "get_dis_process_noise_mat function is undefined"
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
        u : Nu x 1 numpy array, optional
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

        if self._control_model is not None:
            
            input_mat = self._control_model(timestep, state, *ctrl_args)
            ctrl = input_mat @ u
            next_state += ctrl
        if self.state_constraint is not None:
            next_state = self.state_constraint(timestep, next_state)
        return next_state
