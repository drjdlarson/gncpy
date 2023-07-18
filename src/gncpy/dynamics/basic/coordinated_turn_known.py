import numpy as np
from .linear_dynamics_base import LinearDynamicsBase


class CoordinatedTurnKnown(LinearDynamicsBase):
    """Implements the linear coordinated turn with known turn rate model.

    This is a slight variation from normal :class:`.LinearDynamicsBase` classes
    because it does not allow for control models, instead it has the input
    matrix coded directly. It also has the turn angle included in the state
    to help with debugging and coding but is not strictly required by the
    dynamics.

    Notes
    -----
    See :cite:`Li2000_SurveyofManeuveringTargetTrackingDynamicModels` and
    :cite:`Blackman1999_DesignandAnalysisofModernTrackingSystems` for details.

    Attributes
    ----------
    turn_rate : float
        Turn rate in rad/s
    """

    state_names = ("x pos", "y pos", "x vel", "y vel", "turn angle")

    def __init__(self, turn_rate=5 * np.pi / 180, **kwargs):
        super().__init__(**kwargs)
        self.turn_rate = turn_rate
        self.control_model = None

    def get_state_mat(self, timestep, dt):
        """Returns the discrete time state matrix.

        Parameters
        ----------
        timestep : float
            timestep.
        dt : float
            time difference

        Returns
        -------
        N x N numpy array
            state matrix.
        """
        # avoid division by 0
        if abs(self.turn_rate) < 1e-8:
            return np.array(
                [
                    [1, 0, dt, 0, 0],
                    [0, 1, 0, dt, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 1],
                ]
            )
        ta = self.turn_rate * dt
        s_ta = np.sin(ta)
        c_ta = np.cos(ta)
        return np.array(
            [
                [1, 0, s_ta / self.turn_rate, -(1 - c_ta) / self.turn_rate, 0],
                [0, 1, (1 - c_ta) / self.turn_rate, s_ta / self.turn_rate, 0],
                [0, 0, c_ta, -s_ta, 0],
                [0, 0, s_ta, c_ta, 0],
                [0, 0, 0, 0, 1],
            ]
        )

    def get_input_mat(self, timestep, *args):
        """Gets the input matrix.

        This enforces the no control model requirement of these dynamics by
        forcing the input matrix to be zeros.

        Parameters
        ----------
        timestep : float
            Current timestep.
        *args : tuple
            additional arguments, not used.

        Returns
        -------
        N x 1 numpy array
            input matrix.
        """
        return np.zeros((5, 1))

    def propagate_state(self, timestep, state, u=None, state_args=None, ctrl_args=None):
        """Propagates the state forward in time.

        Parameters
        ----------
        timestep : float
            Current timestep.
        state : N x 1 numpy array
            current state.
        u : 1 x 1 numpy array, optional
            Control input, should not be needed with this model. The default is
            None.
        state_args : tuple, optional
            Additional arguments needed to get the state matrix. The default is
            None.
        ctrl_args : tuple, optional
            Additional arguments needed to get the input matrix, not needed.
            The default is None.

        Raises
        ------
        RuntimeError
            If state_args is None.

        Returns
        -------
        N x 1 numpy array
            Next state.
        """
        if state_args is None:
            raise RuntimeError("state_args must be (dt, )")
        if ctrl_args is None:
            ctrl_args = ()
        F = self.get_state_mat(
            timestep,
            *state_args,
        )
        if u is None:
            return F @ state + np.array(
                [0, 0, 0, 0, state_args[0] * self.turn_rate]
            ).reshape((5, 1))
        G = self.get_input_mat(timestep, *ctrl_args)
        return (
            F @ state
            + G @ u
            + np.array([0, 0, 0, 0, state_args[0] * self.turn_rate]).reshape((5, 1))
        )
