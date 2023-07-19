import numpy as np
import gncpy.control._control as cpp_control
import gncpy.dynamics._dynamics as cpp_bindings
from warnings import warn

from .linear_dynamics_base import LinearDynamicsBase


class DoubleIntegrator(LinearDynamicsBase):
    """Implements a double integrator model.

    Todo
    ----
    Implement the control model in c++ for this class
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__controlParams = cpp_control.ControlParams()
        self.__stateTransParams = cpp_bindings.StateTransParams()
        self.__model = cpp_bindings.DoubleIntegrator(0.1)
        if "control_model" in kwargs and kwargs["control_model"] is not None:
            self.__model.set_control_model(kwargs("control_model"))

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
            self._control_model = model
            self.__model.set_control_model(self._control_model)
        else:
            raise TypeError("must be ILinearControlModel type")

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
        self.args_to_params((0.1,), ctrl_args)
        return self._control_model.get_input_mat(timestep, self.__controlParams)

    # must be provided if allow_cpp is true
    def args_to_params(self, state_args, control_args):
        if len(state_args) != 1:
            raise RuntimeError(
                "state args must be only (dt,) not {}".format(repr(state_args))
            )

        if len(control_args) != 0 and self._control_model is None:
            warn("Control agruments supplied but no control model specified")
        elif self._control_model is not None:
            self.__controlParams = self._control_model.args_to_params(
                tuple(control_args)
            )

        self.__constraintParams = cpp_bindings.ConstraintParams()

        # hack since state params is empty but things are set in the model
        self.__model.dt = state_args[0]
        return self.__stateTransParams, self.__controlParams, self.__constraintParams

    # must be provided if allow_cpp is true
    @property
    def model(self):
        return self.__model

    @property
    def state_names(self):
        return self.__model.state_names()

    def propagate_state(self, timestep, state, u=None, state_args=None, ctrl_args=None):
        if state_args is None:
            raise RuntimeError("state_args must be (dt,) not None")
        self.__model.dt = state_args[0]
        if self._control_model is None:
            next_state = self.__model.propagate_state(timestep, state).reshape((-1, 1))
        else:
            next_state = self.__model.propagate_state(
                timestep, state, u, *self.args_to_params(state_args, ctrl_args)
            )
        if self.state_constraint is not None:
            next_state = self.state_constraint(timestep, next_state)
        return next_state.reshape((-1, 1))

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
        self.__model.dt = dt
        return self.__model.get_state_mat(timestep, self.__stateTransParams)
