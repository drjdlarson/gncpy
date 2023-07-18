import numpy as np

from warnings import warn
from .nonlinear_dynamics_base import NonlinearDynamicsBase



class CurvilinearMotion(NonlinearDynamicsBase):
    r"""Implements general curvilinear motion model in 2d.

    This is a slight variation from normal :class:`.NonlinearDynamicsBase` classes
    because it does not use a list of continuous functions but instead has
    the state and input matrices coded directly. As a result, it also does not
    use the control model attribute because it is hardcoded in the
    :meth:`.get_input_mat` function. Also, the angle state should be kept between 0-360
    degrees.

    Notes
    -----
    This implements the following system of ODEs.

    .. math::

        \begin{align}
            \dot{x} &= v cos(\psi) \\
            \dot{y} &= v sin(\psi) \\
            \dot{v} &= u_0 \\
            \dot{\psi} &= u_1
        \end{align}

    See :cite:`Li2000_SurveyofManeuveringTargetTrackingDynamicModels` for
    details.
    """

    state_names = (
        "x pos",
        "y pos",
        "speed",
        "turn angle",
    )

    def __init__(self, **kwargs):
        """Initialize an object.

        Parameters
        ----------
        **kwargs : dict
            Additional key word arguments for the parent.
        """
        super().__init__(**kwargs)
        self.control_model = [None] * len(self.state_names)
        self.control_constraint = None

    @property
    def cont_fnc_lst(self):
        """Continuous time ODEs, not used."""
        warn("Not used by this class")
        return []

    def get_state_mat(
        self, timestep, state, *args, u=None, ctrl_args=None, use_continuous=False
    ):
        """Returns the linearized state matrix that has been hardcoded.

        Parameters
        ----------
        timestep : float
            Current timestep.
        state : N x 1 numpy array
            current state.
        *args : tuple
            Additional arguments placeholde, not used.
        u : Nu x 1 numpy array, optional
            Control input. The default is None.
        ctrl_args : tuple, optional
            Additional agruments needed to get the input matrix. The default is
            None.
        use_continuous : bool, optional
            Flag indicating if the continuous time A matrix should be returned.
            The default is False.

        Returns
        -------
        N x N numpy array
            state transition matrix.
        """
        x = state.ravel()

        A = np.array(
            [
                [0, 0, np.cos(x[3]), -x[2] * np.sin(x[3]) * self.dt],
                [0, 0, np.sin(x[3]), x[2] * np.cos(x[3]) * self.dt],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ]
        )
        if use_continuous:
            return A
        return np.eye(A.shape[0]) + self.dt * A

    def get_input_mat(self, timestep, state, u, state_args=None, ctrl_args=None):
        """Returns the linearized input matrix.

        Parameters
        ----------
        timestep : float
            Current timestep.
        state : N x 1 numpy array
            Current state.
        u : 2 x 1 numpy array
            Current control input.
        state_args : tuple, optional
            Additional arguements needed to get the state matrix. The default
            is None.
        ctrl_args : tuple, optional
            Additional arguments needed to get the input matrix. The default is
            None.

        Returns
        -------
        N x 2 numpy array
            Input matrix.
        """
        return np.array([[0, 0], [0, 0], [self.dt, 0], [0, self.dt]])

    def propagate_state(self, timestep, state, u=None, state_args=None, ctrl_args=None):
        """Propagates the state forward one timestep.

        This uses the hardcoded form for the linearized state and input matrices
        instead of numerical integration.

        Parameters
        ----------
        timestep : float
            Current timestep.
        state : N x 1 numpy array
            current state.
        u : 2 x 1 numpy array, optional
            Control input. The default is None.
        state_args : tuple, optional
            Additional arguements needed to get the state matrix. The default
            is None. These are not needed.
        ctrl_args : tuple, optional
            Additional arguments needed to get the input matrix. The default is
            None. These are not needed.

        Returns
        -------
        N x 1 numpy array
            Next state.
        """
        if state_args is None:
            state_args = ()
        if ctrl_args is None:
            ctrl_args = ()
        F = self.get_state_mat(
            timestep, state, *state_args, u=u, ctrl_args=ctrl_args, use_continuous=False
        )
        _state = state.copy()
        _state[3] = np.mod(_state[3], 2 * np.pi)
        next_state = F @ _state
        if u is None:
            if self.state_constraint is not None:
                next_state = self.state_constraint(timestep, next_state)
            next_state[3] = np.mod(next_state[3], 2 * np.pi)
            return next_state
        G = self.get_input_mat(
            timestep, state, u, state_args=state_args, ctrl_args=ctrl_args
        )
        if self.control_constraint is not None:
            next_state += G @ self.control_constraint(timestep, u.copy())
        else:
            next_state += G @ u
        if self.state_constraint is not None:
            next_state = self.state_constraint(timestep, next_state)
        next_state[3] = np.mod(next_state[3], 2 * np.pi)
        return next_state
