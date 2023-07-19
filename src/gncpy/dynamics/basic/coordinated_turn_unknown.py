import numpy as np
from warnings import warn
from .nonlinear_dynamics_base import NonlinearDynamicsBase


class CoordinatedTurnUnknown(NonlinearDynamicsBase):
    r"""Implements the non-linear coordinated turn with unknown turn rate model.

    Notes
    -----
    This can use either a Wiener process (:math:`\alpha=1`) or a first order
    Markov process model (:math:`\alpha \neq 1`) for the turn rate. This is
    controlled by setting :attr:`.turn_rate_cor_time`. See
    :cite:`Li2000_SurveyofManeuveringTargetTrackingDynamicModels` and
    :cite:`Blackman1999_DesignandAnalysisofModernTrackingSystems` for details.

    .. math::
        \dot{\omega} = -\alpha w + w_{\omega}

    Attributes
    ----------
    turn_rate_cor_time : float
        Correlation time for the turn rate. If None then a Wiener process is used.
    """

    state_names = ("x pos", "y pos", "x vel", "y vel", "turn rate")

    def __init__(self, turn_rate_cor_time=None, **kwargs):
        """Initialize an object.

        Parameters
        ----------
        turn_rate_cor_time : float, optional
            Correlation time of the turn rate. The default is None.
        **kwargs : dict
            Additional arguments for the parent.
        """
        super().__init__(**kwargs)
        self.turn_rate_cor_time = turn_rate_cor_time

        self._control_model = [None] * len(self.state_names)

    @property
    def alpha(self):
        """Read only inverse of the turn rate correlation time."""
        if self.turn_rate_cor_time is None:
            return 0
        else:
            return 1 / self.turn_rate_cor_time

    @property
    def beta(self):
        """Read only value for correlation time in state matrix."""
        if self.turn_rate_cor_time is None:
            return 1
        else:
            return np.exp(-self.dt / self.turn_rate_cor_time)

    @property
    def cont_fnc_lst(self):
        """Continuous time ODEs, not used."""
        warn("Not used by this class")
        return []

    @property
    def control_model(self):
        return self._control_model

    @control_model.setter
    def control_model(self, model):
        self._control_model = model

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
        w = x[4]

        if use_continuous:
            return np.array(
                [
                    [0, 0, 1, 0, 0],
                    [0, 0, 0, 1, 0],
                    [0, 0, 0, -w, 0],
                    [0, 0, w, 0, 0],
                    [0, 0, 0, 0, -self.alpha],
                ]
            )
        # avoid division by 0
        if abs(w) < 1e-8:
            return np.array(
                [
                    [1, 0, self.dt, 0, 0],
                    [0, 1, 0, self.dt, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 0, 1, 0],
                    [0, 0, 0, 0, self.beta],
                ]
            )
        ta = w * self.dt
        s_ta = np.sin(ta)
        c_ta = np.cos(ta)

        w2 = w**2
        F04 = (w * self.dt * c_ta - s_ta) * x[2] / w2 - (
            w * self.dt * s_ta - 1 + c_ta
        ) * x[3] / w2
        F14 = (w * self.dt * s_ta - 1 + c_ta) * x[2] / w2 + (
            w * self.dt * c_ta - s_ta
        ) * x[3] / w2
        F24 = -self.dt * s_ta * x[2] - self.dt * c_ta * x[3]
        F34 = self.dt * c_ta * x[2] - self.dt * s_ta * x[3]
        return np.array(
            [
                [1, 0, s_ta / w, -(1 - c_ta) / w, F04],
                [0, 1, (1 - c_ta) / w, s_ta / w, F14],
                [0, 0, c_ta, -s_ta, F24],
                [0, 0, s_ta, c_ta, F34],
                [0, 0, 0, 0, self.beta],
            ]
        )

    def get_input_mat(self, timestep, state, u, state_args=None, ctrl_args=None):
        """Returns the linearized input matrix.

        This assumes the control input is an AWGN signal.

        Parameters
        ----------
        timestep : float
            Current timestep.
        state : N x 1 numpy array
            Current state.
        u : 3 x 1 numpy array
            Current control input.
        state_args : tuple, optional
            Additional arguements needed to get the state matrix. The default
            is None.
        ctrl_args : tuple, optional
            Additional arguments needed to get the input matrix. The default is
            None.

        Returns
        -------
        N x 3 numpy array
            Input matrix.
        """
        return np.array(
            [
                [0.5 * self.dt**2, 0, 0],
                [0, 0.5 * self.dt**2, 0],
                [self.dt, 0, 0],
                [0, self.dt, 0],
                [0, 0, 1],
            ]
        )

    def propagate_state(self, timestep, state, u=None, state_args=None, ctrl_args=None):
        """Propagates the state forward one timestep.

        This uses the hardcoded form for the linearized state and input matrices
        instead of numerical integration. It assumes the control input is an
        AWGN signal.

        Parameters
        ----------
        timestep : float
            Current timestep.
        state : N x 1 numpy array
            current state.
        u : 3 x 1 numpy array, optional
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
        if u is None:
            return F @ state
        G = self.get_input_mat(
            timestep, state, u, state_args=state_args, ctrl_args=ctrl_args
        )
        return F @ state + G @ u
