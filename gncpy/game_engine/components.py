"""Defines components for use by game entities."""
import numpy as np
import pygame

import serums.models as smodels


class CShape:
    """Contains properties related to the drawn shape.

    Attributes
    ----------
    shape : pygame object
        type of shape to create.
    color : tuple
        RGB triplet for the color in range [0, 255].
    zorder : int
        Determines draw order, lower numbers are drawn first.
    """

    __slots__ = "shape", "color", "zorder"

    def __init__(self, shape=None, w=None, h=None, color=None, zorder=None):
        """Initialize an object.

        Parameters
        ----------
        shape : string
            type of shape to create.
        w : int
            width in pixels.
        h : int
            height in pixels.
        color : tuple
            RGB triplet for the color in range [0, 255].
        zorder : int
            Determines draw order, lower numbers are drawn first.

        Todo
        ----
        Allow for other shape types.
        """
        if shape.lower() == "rect":
            self.shape = pygame.Rect((0, 0), (w, h))
        else:
            self.shape = shape

        self.color = color
        self.zorder = zorder


class CTransform:
    """Contains properties relating to the spatial components.

    Attributes
    ----------
    pos : 2 x 1 numpy array
        position of the center in pixels
    last_pos : 2 x 1 numpy array
        last position of the center in pixels.
    vel 2 x 1 numpy array
        velocity of the center in pixels per timestep
    """

    __slots__ = "pos", "last_pos", "vel"

    def __init__(self):
        self.pos = np.nan * np.ones((2, 1))
        self.last_pos = np.nan * np.ones((2, 1))
        self.vel = np.nan * np.ones((2, 1))


class CCollision:
    """Handles the bounding box used for collisions.

    Attrbutes
    ---------
    aabb : pygame Rect
        Rectangle representing the axis aligned bounding box.
    """

    __slots__ = "aabb"

    def __init__(self, w=None, h=None):
        if w is not None and h is not None:
            self.aabb = pygame.Rect((0, 0), (w, h))
        else:
            self.aabb = None


class CBirth:
    """Handles the birth model properties.

    Also handles the generation of the birth location through the use of
    a distribution object from SERUMS.
    """

    __slots__ = "_model", "loc", "_rng", "randomize"

    def __init__(
        self, b_type=None, loc=None, scale=None, params=None, rng=None, randomize=True
    ):
        """Initializes an object.

        Parameters
        ----------
        b_type : string
            Model type.
        loc : numpy array
            location parameter of the model.
        scale : N x N numpy array
            scale parameter of the model.
        params : dict
            additional parameters for the model.
        """
        if b_type.lower() == "gaussian":
            if len(scale.shape) == 1:
                std = np.diag(scale)
            else:
                std = scale
            self._model = smodels.Gaussian(mean=loc, covariance=std ** 2)
        else:
            self._model = None
        self.loc = loc
        self._rng = rng
        self.randomize = randomize

    def sample(self):
        """Draw a sample from the distribution."""
        if self.randomize:
            return self._model.sample(rng=self._rng).reshape((-1, 1))
        else:
            return self._model.location.reshape((-1, 1)).copy()


class CEvents:
    """Holds the events properties.

    Attributes
    ----------
    events : list
        Each element is a tuple with the first being an event identifier and
        the second a dict with extra info.
    """

    __slots__ = "events"

    def __init__(self):
        self.events = []


class CHazard:
    """Hold the properties for hazard info.

    Attributes
    ----------
    prob_of_death : float
        Probability of death at each timestep in the hazard. Must be in the
        range (0, 1].
    entrance_times : dict
        mapping of player id to entrance time.
    """

    __slots__ = "prob_of_death", "entrance_times"

    def __init__(self, prob_of_death=None):
        """Initialize an object.

        Parameters
        ----------
        prob_of_death : float
            Probability of death at each timestep.
        """
        self.prob_of_death = prob_of_death
        self.entrance_times = {}


class CCapabilities:
    """Hold properties about player/target capabilities.

    Attributes
    ----------
    capabilities : list
        Each element defines a capability. The :code:`in` keyword must work
        for chcking if an item is in the list.
    """

    __slots__ = "capabilities"

    def __init__(self, capabilities=None):
        """Initialize an object.

        Parameters
        ----------
        capabilities : list
            List of capabilities.
        """
        if capabilities is None:
            capabilities = []
        self.capabilities = capabilities


class CPriority:
    """Hold properties about the priority.

    Attributes
    ----------
    priority : float
        Priority value.
    """

    __slots__ = "priority"

    def __init__(self, priority=None):
        """Initialize an object.

        Parameters
        ----------
        priority : float
            priority of the object.
        """
        self.priority = priority


class CDynamics:
    """Handles all the properties relaing to the dynamics.

    Also implements the logic for propagating the state via a dynamics
    object.

    Attributes
    ----------
    dynObj : :class:`gncpy.dynamics.DynamicsBase`
        Implements the dynamics equations, control, and state constraints.
    pos_inds : list
        Indices of the state vector containing the position info.
    vel_inds : list
        Indices of the state vector containing the velocity info.
    state_args : tuple
        Additional arguments for propagating the state.
    ctrl_args : tuple
        Additional arguments for propagating the state.
    state_low : numpy array
        Lower limit of each state.
    state_high : numpy array
        Upper limit of each state.
    """

    __slots__ = (
        "dynObj",
        "last_state",
        "state",
        "pos_inds",
        "vel_inds",
        "state_args",
        "ctrl_args",
        "state_low",
        "state_high",
    )

    def __init__(
        self,
        dynObj=None,
        pos_inds=None,
        vel_inds=None,
        state_args=None,
        ctrl_args=None,
        state_low=None,
        state_high=None,
    ):
        """Initialize an object.

        Parameters
        ----------
        dynObj : :class:`gncpy.dynamics.DynamicsBase`
            Implements the dynamics equations, control, and state constraints.
        pos_inds : list
            Indices of the state vector containing the position info.
        vel_inds : list
            Indices of the state vector containing the velocity info.
        state_args : tuple
            Additional arguments for propagating the state.
        ctrl_args : tuple
            Additional arguments for propagating the state.
        state_low : numpy array
            Lower limit of each state.
        state_high : numpy array
            Upper limit of each state.
        """
        self.dynObj = dynObj
        if self.dynObj is not None:
            n_states = len(self.dynObj.state_names)
            self.last_state = np.nan * np.ones((n_states, 1))
            self.state = np.zeros((n_states, 1))

        self.pos_inds = pos_inds
        self.vel_inds = vel_inds
        if state_args is None:
            state_args = ()
        self.state_args = state_args
        if ctrl_args is None:
            ctrl_args = ()
        self.ctrl_args = ctrl_args
        self.state_low = state_low
        self.state_high = state_high
