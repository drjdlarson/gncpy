"""Basic models and classes that can be extended.

These provide an easy to use interface for some common dynamic objects and
and their associated models. They have been designed to integrate well with the
filters in :mod:`gncpy.filters`.
"""
from abc import abstractmethod, ABC
import numpy as np
import scipy.integrate as s_integrate
from warnings import warn

import gncpy.math as gmath

import gncpy.dynamics._dynamics as cpp_bindings


class DynamicsBase(ABC):
    r"""Defines common attributes for all dynamics models.

    Attributes
    ----------
    control_model : callable or list of callables, optional
        For objects of :class:`gncpy.dynamics.LinearDynamicsBase` it is a
        callable with the signature `t, *ctrl_args` and returns the
        input matrix :math:`G_k` from :math:`x_{k+1} = F_k x_k + G_k u_k`.
        For objects of :class:`gncpy.dynamics.NonlinearDynamicsBase` it is a
        list of callables where each callable returns the modification to the
        corresponding state, :math:`g(t, x_i, u_i)`, in the differential equation
        :math:`\dot{x}_i = f(t, x_i) + g(t, x_i, u_i)` and has the signature
        `t, x, u, *ctrl_args`.
    state_constraint : callable
        Has the signature `t, x` where `t` is the current timestep and `x`
        is the propagated state. It returns the propagated state with
        any constraints applied to it.
    """

    __slots__ = "control_model", "state_constraint"

    state_names = ()
    """Tuple of strings for the name of each state. The order should match
    that of the state vector.
    """

    def __init__(self, control_model=None, state_constraint=None):
        super().__init__()
        self.control_model = control_model
        self.state_constraint = state_constraint

    @abstractmethod
    def propagate_state(self, timestep, state, u=None, state_args=None, ctrl_args=None):
        """Abstract method for propagating the state forward in time.

        Must be overridden in child classes.

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

        Raises
        ------
        NotImplementedError
            If child class does not implement this function.

        Returns
        -------
        next_state : N x 1 numpy array
            The propagated state.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_state_mat(self, timestep, *args, **kwargs):
        """Abstract method for getting the discrete time state matrix.

        Must be overridden in child classes.

        Parameters
        ----------
        timestep : float
            timestep.
        args : tuple, optional
            any additional arguments needed.

        Returns
        -------
        N x N numpy array
            state matrix.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_input_mat(self, timestep, *args, **kwargs):
        """Should return the input matrix.

        Must be overridden by the child class.

        Parameters
        ----------
        timestep : float
            Current timestep.
        *args : tuple
            Placeholder for additional arguments.
        **kwargs : dict
            Placeholder for additional arguments.

        Raises
        ------
        NotImplementedError
            Child class must implement this.

        Returns
        -------
        N x Nu numpy array
            input matrix for the system
        """
        raise NotImplementedError()


class LinearDynamicsBase(DynamicsBase):
    """Base class for all linear dynamics models.

    Child classes should define their own get_state_mat function and set the
    state names class variable. The remainder of the functions autogenerate
    based on these values.

    """

    __slots__ = ()

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
        if self.control_model is None:
            raise RuntimeWarning("Control model is not set.")
        return self.control_model(timestep, *ctrl_args)

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

        if self.control_model is not None:
            input_mat = self.control_model(timestep, state, *ctrl_args)
            ctrl = input_mat @ u
            next_state += ctrl
        if self.state_constraint is not None:
            next_state = self.state_constraint(timestep, next_state)
        return next_state


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

    __slots__ = ("dt", "integrator_type", "integrator_params", "_integrator")

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
                timestep, state, [factory(ii) for ii in range(state.size)], f_args,
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
            timestep, state, u, [factory(ii) for ii in range(state.size)], (),
        )
        # return gmath.get_input_jacobian(timestep, state, u, self.control_model, (),)

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


class DoubleIntegrator(LinearDynamicsBase):
    """Implements a double integrator model.
    
    Todo
    ----
    Implement the control model in c++ for this class
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__model = cpp_bindings.DoubleIntegrator(0.1)

    @property
    def state_names(self):
        return self.__model.state_names()

    def propagate_state(self, timestep, state, u=None, state_args=None, ctrl_args=None):
        if state_args is None:
            raise RuntimeError("state_args must be (dt,) not None")
        self.__model.dt = state_args[0]
        return self.__model.propagate_state(timestep, state)

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
        return self.__model.get_state_mat(timestep, self.__state_trans_params)


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

    __slots__ = "control_constraint"

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

    __slots__ = "turn_rate"

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
        F = self.get_state_mat(timestep, *state_args,)
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

    __slots__ = ("turn_rate_cor_time",)

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

        self.control_model = [None] * len(self.state_names)

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

        w2 = w ** 2
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
                [0, 1, (1 - c_ta) / x[4], s_ta / w, F14],  # shouldn't the x[4] be w?
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
                [0.5 * self.dt ** 2, 0, 0],
                [0, 0.5 * self.dt ** 2, 0],
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
        n = self.mean_motion
        c_dtn = np.cos(dt * n)
        s_dtn = np.sin(dt * n)
        F = np.array(
            [
                [4 - 3 * c_dtn, 0, s_dtn / n, -(2 * c_dtn - 2) / n],
                [
                    6 * s_dtn - 6 * dt * n,
                    1,
                    (2 * c_dtn - 2) / n,
                    (4 * s_dtn - 3 * dt * n) / n,
                ],
                [3 * n * s_dtn, 0, c_dtn, 2 * s_dtn],
                [6 * n * (c_dtn - 1), 0, -2 * s_dtn, 4 * c_dtn - 3],
            ]
        )
        return F


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
        n = self.mean_motion
        c_dtn = np.cos(dt * n)
        s_dtn = np.sin(dt * n)
        F = np.zeros((6, 6))
        F2d = super().get_state_mat(timestep, dt)
        F[:2, :2] = F2d[:2, :2]
        F[:2, 3:5] = F2d[:2, 2:]
        F[3:5, :2] = F2d[2:, :2]
        F[3:5, 3:5] = F2d[2:, 2:]
        F[2, 2] = c_dtn
        F[2, 5] = s_dtn / n
        F[5, 2] = -n * s_dtn
        F[5, 5] = c_dtn
        return F


class TschaunerHempelOrbit(NonlinearDynamicsBase):
    """Implements the non-linear Tschauner-Hempel elliptical orbit model.

    Notes
    -----
    It is the general elliptical orbit of an object around another target
    object as defined in
    :cite:`Tschauner1965_RendezvousZuEineminElliptischerBahnUmlaufendenZiel`.
    The states are defined as positions in a
    Local-Vertical-Local-Horizontal (LVLH) frame. Note, the true anomaly is
    that of the target object. For more details see
    :cite:`Okasha2013_GuidanceNavigationandControlforSatelliteProximityOperationsUsingTschaunerHempelEquations`
    and :cite:`Lange1965_FloquetTheoryOrbitalPerturbations`

    Attributes
    ----------
    mu : float, optional
        gravitational parameter in :math:`m^3 s^{-2}`. The default is 3.986004418 * 10**14.
    semi_major : float
        semi-major axis in meters. The default is None.
    eccentricity : float, optional
        eccentricity. The default is 1.
    """

    state_names = (
        "x pos",
        "y pos",
        "z pos",
        "x vel",
        "y vel",
        "z vel",
        "targets true anomaly",
    )

    def __init__(
        self, mu=3.986004418 * 10 ** 14, semi_major=None, eccentricity=1, **kwargs
    ):
        self.mu = mu
        self.semi_major = semi_major
        self.eccentricity = eccentricity

        super().__init__(**kwargs)

    @property
    def cont_fnc_lst(self):
        """Continuous time dynamics.

        Returns
        -------
        list
            functions of the form :code:`(t, x, *args)`.
        """
        # returns x velocity
        def f0(t, x, *args):
            return x[3]

        # returns y velocity
        def f1(t, x, *args):
            return x[4]

        # returns z velocity
        def f2(t, x, *args):
            return x[5]

        # returns x acceleration
        def f3(t, x, *args):
            e = self.eccentricity
            a = self.semi_major
            mu = self.mu

            e2 = e ** 2
            R3 = ((a * (1 - e2)) / (1 + e * np.cos(x[6]))) ** 3
            n = np.sqrt(mu / a ** 3)

            C1 = mu / R3
            wz = n * (1 + e * np.cos(x[6])) ** 2 / (1 - e2) ** (3.0 / 2.0)
            wz_dot = -2 * mu * e * np.sin(x[6]) / R3

            return (wz ** 2 + 2 * C1) * x[0] + wz_dot * x[1] + 2 * wz * x[4]

        # returns y acceleration
        def f4(t, x, *args):
            e = self.eccentricity
            a = self.semi_major
            mu = self.mu

            e2 = e ** 2
            R3 = ((a * (1 - e2)) / (1 + e * np.cos(x[6]))) ** 3
            n = np.sqrt(mu / a ** 3)

            C1 = mu / R3
            wz = n * (1 + e * np.cos(x[6])) ** 2 / (1 - e2) ** (3.0 / 2.0)
            wz_dot = -2 * mu * e * np.sin(x[6]) / R3

            return (wz ** 2 - C1) * x[1] - wz_dot * x[0] - 2 * wz * x[3]

        # returns z acceleration
        def f5(t, x, *args):
            e = self.eccentricity
            a = self.semi_major
            mu = self.mu

            e2 = e ** 2
            R3 = ((a * (1 - e2)) / (1 + e * np.cos(x[6]))) ** 3

            C1 = mu / R3

            return -C1 * x[2]

        # returns true anomaly ROC
        def f6(t, x, *args):
            e = self.eccentricity
            a = self.semi_major
            p = a * (1 - e ** 2)

            H = np.sqrt(self.mu * p)
            R = p / (1 + e * np.cos(x[6]))
            return H / R ** 2

        return [f0, f1, f2, f3, f4, f5, f6]


class KarlgaardOrbit(NonlinearDynamicsBase):
    """Implements the non-linear Karlgaar elliptical orbit model.

    Notes
    -----
    It uses the numerical integration of the second order approximation in
    dimensionless sphereical coordinates. See
    :cite:`Karlgaard2003_SecondOrderRelativeMotionEquations` for details.
    """

    state_names = (
        "non-dim radius",
        "non-dim az angle",
        "non-dim elv angle",
        "non-dim radius ROC",
        "non-dim az angle ROC",
        "non-dim elv angle ROC",
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def cont_fnc_lst(self):
        """Continuous time dynamics.

        Returns
        -------
        list
            functions of the form :code:`(t, x, *args)`.
        """
        # returns non-dim radius ROC
        def f0(t, x, *args):
            return x[3]

        # returns non-dim az angle ROC
        def f1(t, x, *args):
            return x[4]

        # returns non-dim elv angle ROC
        def f2(t, x, *args):
            return x[5]

        # returns non-dim radius ROC ROC
        def f3(t, x, *args):
            r = x[0]
            phi = x[2]
            theta_d = x[4]
            phi_d = x[5]
            return (
                (-3 * r ** 2 + 2 * r * theta_d - phi ** 2 + theta_d ** 2 + phi_d ** 2)
                + 3 * r
                + 2 * theta_d
            )

        # returns non-dim az angle ROC ROC
        def f4(t, x, *args):
            r = x[0]
            theta = x[1]
            r_d = x[3]
            theta_d = x[4]
            return (2 * r * r_d + 2 * theta * theta_d - 2 * theta_d * r_d) - 2 * r_d

        # returns non-dim elv angle ROC ROC
        def f5(t, x, *args):
            phi = x[2]
            r_d = x[3]
            theta_d = x[4]
            phi_d = x[5]
            return (-2 * theta_d * phi - 2 * phi_d * r_d) - phi

        return [f0, f1, f2, f3, f4, f5]


class IRobotCreate(NonlinearDynamicsBase):
    """A differential drive robot based on the iRobot Create.

    This has a control model predefined because the dynamics themselves do not
    change the state.

    Notes
    -----
    This is taken from :cite:`Berg2016_ExtendedLQRLocallyOptimalFeedbackControlforSystemswithNonLinearDynamicsandNonQuadraticCost`
    It represents a 2 wheel robot with some distance between its wheels.
    """

    state_names = ("pos_x", "pos_v", "turn_angle")

    def __init__(self, wheel_separation=0.258, radius=0.335 / 2, **kwargs):
        """Initialize an object.

        Parameters
        ----------
        wheel_separation : float, optional
            Distance between the two wheels in meters.
        radius : float
            Radius of the bounding box for the robot in meters.
        **kwargs : dict
            Additional arguments for the parent class.
        """
        super().__init__(**kwargs)
        self._wheel_separation = wheel_separation
        self.radius = radius

        def g0(t, x, u, *args):
            return 0.5 * ((u[0] + u[1]) * np.cos(x[2])).item()

        def g1(t, x, u, *args):
            return 0.5 * ((u[0] + u[1]) * np.sin(x[2])).item()

        def g2(t, x, u, *args):
            return (u[1] - u[0]) / self.wheel_separation

        self.control_model = [g0, g1, g2]

    @property
    def wheel_separation(self):
        """Read only wheel separation distance."""
        return self._wheel_separation

    @property
    def cont_fnc_lst(self):
        """Implements the contiuous time dynamics."""

        def f0(t, x, *args):
            return 0

        def f1(t, x, *args):
            return 0

        def f2(t, x, *args):
            return 0

        return [f0, f1, f2]
