from abc import abstractmethod, ABC


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

    state_names = ()
    """Tuple of strings for the name of each state. The order should match
    that of the state vector.
    """

    def __init__(self, control_model=None, state_constraint=None):
        super().__init__()
        self._control_model = control_model
        self.state_constraint = state_constraint

    @property
    def allow_cpp(self):
        return False

    @property
    def control_model(self):
        return self._control_model

    @control_model.setter
    def control_model(self, model):
        self._control_model = model

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

