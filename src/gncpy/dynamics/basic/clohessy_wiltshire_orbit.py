import numpy as np
from warnings import warn

import gncpy.dynamics._dynamics as cpp_bindings
import gncpy.control._control as cpp_control
from .clohessy_wiltshire_orbit2d import ClohessyWiltshireOrbit2d


class ClohessyWiltshireOrbit(ClohessyWiltshireOrbit2d):
    """Implements the Clohessy Wiltshire orbit model.

    This adds on the z component to make the model 3d.

    Attributes
    ----------
    mean_motion :float
        mean motion of reference spacecraft
    """

    state_names = ("x pos", "y pos", "z pos", "x vel", "y vel", "z vel")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__controlParams = cpp_control.ControlParams()
        self.__stateTransParams = cpp_bindings.StateTransParams()
        self.__model = cpp_bindings.ClohessyWiltshire(0.1, 0.01)
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
        self.__model.dt = dt
        return self.__model.get_state_mat(timestep, self.__stateTransParams)
        # n = self.mean_motion
        # c_dtn = np.cos(dt * n)
        # s_dtn = np.sin(dt * n)
        # F = np.zeros((6, 6))
        # F2d = super().get_state_mat(timestep, dt)
        # F[:2, :2] = F2d[:2, :2]
        # F[:2, 3:5] = F2d[:2, 2:]
        # F[3:5, :2] = F2d[2:, :2]
        # F[3:5, 3:5] = F2d[2:, 2:]
        # F[2, 2] = c_dtn
        # F[2, 5] = s_dtn / n
        # F[5, 2] = -n * s_dtn
        # F[5, 5] = c_dtn
        # return F
