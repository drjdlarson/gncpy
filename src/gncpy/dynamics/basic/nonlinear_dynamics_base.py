import numpy as np
import scipy.integrate as s_integrate

import gncpy.math as gmath
from abc import abstractmethod
from warnings import warn
from .dynamics_base import DynamicsBase

class NonlinearDynamicsBase(DynamicsBase):
    """Base class for non-linear dynamics models.

    Child classes should define their own cont_fnc_lst property and set the
    state names class variable. The remainder of the functions autogenerate
    based on these values.

    Attributes
    ----------
    dt : float
        time difference for the integration period
    integrator_type : string, optional
        integrator type as defined by scipy's integrate.ode function. The
        default is `dopri5`.
    integrator_params : dict, optional
        additional parameters for the integrator. The default is {}.
    """

    def __init__(
        self, integrator_type="dopri5", integrator_params={}, dt=np.nan, **kwargs
    ):
        super().__init__(**kwargs)

        self.dt = dt
        self.integrator_type = integrator_type
        self.integrator_params = integrator_params

        self._integrator = None

    @property
    @abstractmethod
    def cont_fnc_lst(self):
        r"""Class property for the continuous time dynamics functions.

        Must be overridden in the child classes.

        Notes
        -----
        This is a list of functions that implement part of the differential
        equations :math:`\dot{x} = f(t, x) + g(t, u)` for each state, in order.
        These functions only represent the dynamics :math:`f(t, x)`.

        Returns
        -------
        list
            Each element is a function that take the timestep, the state, and
            f_args. They must return the new state for the given index in the
            list/state vector.
        """
        raise NotImplementedError()

    def _cont_dyn(self, t, x, u, state_args, ctrl_args):
        r"""Implements the continuous time dynamics.

        This automatically sets up the combined differential equation based on
        the supplied continuous function list.

        This implements the equation :math:`\dot{x} = f(t, x) + g(t, x, u)` and
        returns the state derivatives as a vector. Note the control input is
        only used if an appropriate control model is set by the user.

        Parameters
        ----------
        t : float
            timestep.
        x : N x 1 numpy array
            current state.
        u : Nu x 1 numpy array
            Control effort, not used if no control model.
        state_args : tuple
            Additional arguements for the state functions.
        ctrl_args : tuple
            Additional arguments for the control functions. Not used if no
            control model

        Returns
        -------
        x_dot : N x 1 numpy array
            state derivative
        """
        out = np.zeros((len(self.state_names), 1))
        for ii, f in enumerate(self.cont_fnc_lst):
            out[ii] = f(t, x, *state_args)
        if self.control_model is not None:
            for ii, g in enumerate(self.control_model):
                out[ii] += g(t, x, u, *ctrl_args)
        return out

    def get_state_mat(
        self, timestep, state, *f_args, u=None, ctrl_args=None, use_continuous=False
    ):
        """Calculates the state matrix from the ode list.

        Parameters
        ----------
        timestep : float
            timestep.
        state : N x 1 numpy array
            current state.
        *f_args : tuple
            Additional arguments to pass to the ode functions.

        Returns
        -------
        N x N numpy array
            state transition matrix.
        """
        if ctrl_args is None:
            ctrl_args = ()
        if use_continuous:
            if self.control_model is not None and u is not None:

                def factory(ii):
                    return lambda _t, _x, _u, *_args: self._cont_dyn(
                        _t, _x, _u, _args, ctrl_args
                    )[ii]

                return gmath.get_state_jacobian(
                    timestep,
                    state,
                    [factory(ii) for ii in range(state.size)],
                    f_args,
                    u=u,
                )
            return gmath.get_state_jacobian(timestep, state, self.cont_fnc_lst, f_args)
        else:

            def factory(ii):
                return lambda _t, _x, *_args: self.propagate_state(
                    _t, _x, u=u, state_args=_args, ctrl_args=ctrl_args
                )[ii]

            return gmath.get_state_jacobian(
                timestep,
                state,
                [factory(ii) for ii in range(state.size)],
                f_args,
            )

    def get_input_mat(self, timestep, state, u, state_args=None, ctrl_args=None):
        """Calculates the input matrix from the control model.

        This calculates the jacobian of the control model. If no control model
        is specified than it returns a zero matrix.

        Parameters
        ----------
        timestep : float
            current timestep.
        state : N x 1 numpy array
            current state.
        u : Nu x 1
            current control input.
        *f_args : tuple
            Additional arguments to pass to the control model.

        Returns
        -------
        N x Nu numpy array
            Control input matrix.
        """
        if self.control_model is None:
            warn("Control model is None")
            return np.zeros((state.size, u.size))
        if state_args is None:
            state_args = ()
        if ctrl_args is None:
            ctrl_args = ()

        def factory(ii):
            return lambda _t, _x, _u, *_args: self.propagate_state(
                _t, _x, u=_u, state_args=state_args, ctrl_args=ctrl_args
            )[ii]

        return gmath.get_input_jacobian(
            timestep,
            state,
            u,
            [factory(ii) for ii in range(state.size)],
            (),
        )

    def propagate_state(self, timestep, state, u=None, state_args=None, ctrl_args=None):
        """Propagates the continuous time dynamics.

        Automatically integrates the defined ode list and adds control inputs
        (held constant over the integration period).

        Parameters
        ----------
        timestep : float
            current timestep.
        state : N x 1 numpy array
            Current state vector.
        u : Nu x 1 numpy array, optional
            Current control effort. The default is None. Only used if a control
            model is set.
        state_args : tuple, optional
            Additional arguments to pass to the state odes. The default is ().
        ctrl_args : tuple, optional
            Additional arguments to pass to the control functions. The default
            is ().

        Raises
        ------
        RuntimeError
            If the integration fails.

        Returns
        -------
        next_state : N x 1 numpy array
            the state at time :math:`t + dt`.
        """
        if state_args is None:
            state_args = ()
        if ctrl_args is None:
            ctrl_args = ()
        self._integrator = s_integrate.ode(
            lambda t, y: self._cont_dyn(t, y, u, state_args, ctrl_args).flatten()
        )
        self._integrator.set_integrator(self.integrator_type, **self.integrator_params)
        self._integrator.set_initial_value(state, timestep)

        if np.isnan(self.dt) or np.isinf(self.dt):
            raise RuntimeError("Invalid value for dt ({}).".format(self.dt))
        next_time = timestep + self.dt
        next_state = self._integrator.integrate(next_time)
        next_state = next_state.reshape((next_state.size, 1))
        if not self._integrator.successful():
            msg = "Integration failed at time {}".format(timestep)
            raise RuntimeError(msg)
        if self.state_constraint is not None:
            next_state = self.state_constraint(timestep, next_state)
        return next_state
