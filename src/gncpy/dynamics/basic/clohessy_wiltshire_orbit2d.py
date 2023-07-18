import numpy as np
import gncpy.dynamics._dynamics as cpp_bindings
import gncpy.control._control as cpp_control

from warnings import warn
from .linear_dynamics_base import LinearDynamicsBase


class ClohessyWiltshireOrbit2d(LinearDynamicsBase):
    """Implements the Clohessy Wiltshire orbit model.

    Notes
    -----
    This is based on
    :cite:`Clohessy1960_TerminalGuidanceSystemforSatelliteRendezvous`
    and :cite:`Desai2013_AComparativeStudyofEstimationModelsforSatelliteRelativeMotion`
    but only implements the 2D case

    Attributes
    ----------
    mean_motion :float
        mean motion of reference spacecraft
    """

    state_names = ("x pos", "y pos", "x vel", "y vel")

    def __init__(self, mean_motion=None, **kwargs):
        self.mean_motion = mean_motion

        super().__init__(**kwargs)
        self.__controlParams = cpp_control.ControlParams()
        self.__stateTransParams = cpp_bindings.StateTransParams()
        self.__model = cpp_bindings.ClohessyWiltshire2D(0.01, mean_motion)
        if self.mean_motion is not None:
            self.__model.mean_motion = self.mean_motion
        if kwargs["control_model"] is not None:
            self.__model.setControlModel(kwargs("control_model"))

    @property
    def allow_cpp(self):
        return True

    @property
    def control_model(self):
        warn("viewing the control model is not supported for this class")
        return None

    @control_model.setter
    def control_model(self, model):
        if isinstance(model, cpp_control.ILinearControlModel):
            self.__model.setControlModel(model)
        else:
            raise TypeError("must be ILinearControlModel type")

    # must be provided if allow_cpp is true
    def args_to_params(self, state_args, control_args):
        if len(state_args) != 1:
            raise RuntimeError(
                "state args must be only (dt,) not {}".format(repr(state_args))
            )

        if len(control_args) != 0 and self._control_model is None:
            warn("Control agruments supplied but no control model specified")
        elif self._control_model is not None:
            try:
                self.__controlParams = self._control_model.args_to_params(control_args)
            except Exception:
                warn(
                    "Supplied control model does not support c++ backend but model is supposed to allow c++ backend. Not generating parameters"
                )
                self.__controlParams = cpp_control.ControlParams()

        # hack since state params is empty but things are set in the model
        self.__model.dt = state_args[0]
        return self.__stateTransParams, self.__controlParams

    @property
    def model(self):
        return self.__model

    @property
    def state_names(self):
        return self.__model.state_names()

    def propagate_state(self, timestep, state, u=None, state_args=None, ctrl_args=None):
        if state_args is None:
            raise RuntimeError("state_args must be (dt,) not None")
        if self._control_model is None:
            self.__model.dt = state_args[0]
            return self.__model.propagate_state(timestep, state).reshape((-1, 1))
        else:
            return super().propagate_state(
                timestep, state, u=u, state_args=state_args, ctrl_args=ctrl_args
            )

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
        """Calculates the state transition matrix.

        Parameters
        ----------
        timestep : float
            current timestep.
        dt : float
            time difference.

        Returns
        -------
        F : 4 x 4 numpy array
            state transition matrix.

        """
        self.__model.dt = dt
        return self.__model.get_state_mat(timestep, self.__stateTransParams)
        # n = self.mean_motion
        # c_dtn = np.cos(dt * n)
        # s_dtn = np.sin(dt * n)
        # F = np.array(
        #     [
        #         [4 - 3 * c_dtn, 0, s_dtn / n, -(2 * c_dtn - 2) / n],
        #         [
        #             6 * s_dtn - 6 * dt * n,
        #             1,
        #             (2 * c_dtn - 2) / n,
        #             (4 * s_dtn - 3 * dt * n) / n,
        #         ],
        #         [3 * n * s_dtn, 0, c_dtn, 2 * s_dtn],
        #         [6 * n * (c_dtn - 1), 0, -2 * s_dtn, 4 * c_dtn - 3],
        #     ]
        # )
        # return F
