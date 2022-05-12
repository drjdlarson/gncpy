"""Implements basic games for RL environments."""
from abc import ABC, abstractmethod
import numpy as np
import pathlib

import os
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"
import pygame

from ruamel.yaml import YAML

import gncpy.dynamics as gdyn
import gncpy.planning.reinforcement_learning.rewards as grewards
from gncpy.planning.reinforcement_learning.enums import EventType
import serums.models as smodels


yaml = YAML()


# %% Entities
class EntityManager:
    """Handles creation and deletion of entities."""

    def __init__(self):
        self._entities = []
        self._entities_to_add = []
        self._entity_map = {}
        self._total_entities = 0

    def _remove_dead_entities(self, vec):
        e_to_rm = []
        for ii, e in enumerate(vec):
            if not e.active:
                e_to_rm.append(ii)

        for ii in e_to_rm[::-1]:
            del vec[ii]

    def update(self):
        """Updates the list of entities.

        Should be called once per timestep. Adds new entities to the list and
        and removes dead ones.

        Returns
        -------
        None.
        """
        for e in self._entities_to_add:
            self._entities.append(e)
            if e.tag not in self._entity_map:
                self._entity_map[e.tag] = []
            self._entity_map[e.tag].append(e)
        self._entities_to_add = []

        self._remove_dead_entities(self._entities)

        for tag, ev in self._entity_map.items():
            self._remove_dead_entities(ev)

    def add_entity(self, tag):
        """Creates a new entity.

        The entity is queued to be added. It is not part of the entity list
        until after the update function has been called.

        Parameters
        ----------
        tag : string
            Tag to identify the type of entity.

        Returns
        -------
        e : :class:`.Entity`
            Reference to the created entity.
        """
        self._total_entities += 1
        e = Entity(self._total_entities, tag)
        self._entities_to_add.append(e)

        return e

    def get_entities(self, tag=None):
        """Return a list of references to entities.

        Can also get all entities with a given tag. Note that changing entities
        returned by this function modifies the entities managed by this class.

        Parameters
        ----------
        tag : string, optional
            If provided only return entities with this tag. The default is None.

        Returns
        -------
        list
            Each element is an :class:`.Entity`.
        """
        if tag is None:
            return self._entities
        else:
            if tag in self._entity_map:
                return self._entity_map[tag]
            else:
                return []


class Entity:
    """Elements in a game.

    Attributes
    ----------
    c_transform : :class:`.CTransform`
        Transform component.
    c_dynamics : :class:`.CDynamics`
        Dynamics component.
    c_shape : :class:`.CShape`
        Shape component.
    c_collision : :class:`.CCollision`
        Collision component.
    c_birth: :class:`.CBirth`
        Birth component.
    c_events: :class:`.CEvents`
        Events component.
    c_capabilities : :class:`.CCapabilities`
        Capabilities component.
    c_priority: :class:`.CPriority`
        Priority component.
    """

    def __init__(self, e_id, tag):
        """Initialize an object.

        This should not be called outside of the :class:`.EntityManager` class.

        Parameters
        ----------
        e_id : int
            Unique ID number for the entity.
        tag : string
            Type of entity.
        """
        self._active = True
        self._id = e_id
        self._tag = tag

        self.c_transform = None
        self.c_dynamics = None
        self.c_shape = None
        self.c_collision = None
        self.c_birth = None
        self.c_events = None
        self.c_capabilities = None
        self.c_priority = None

    @property
    def active(self):
        """Flag indicating if the entity is alive."""
        return self._active

    @property
    def tag(self):
        """Read only tag of for the entity."""
        return self._tag

    @property
    def id(self):
        """Read only unique id of the entity."""
        return self._id

    def destroy(self):
        """Handles destruction of the entity."""
        self._active = False


# %% Components
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

    __slots__ = ('dynObj', 'last_state', 'state', 'pos_inds', 'vel_inds',
                 'state_args', 'ctrl_args', 'state_low', 'state_high')

    def __init__(self, dynObj, pos_inds, vel_inds, state_args, ctrl_args,
                 state_low, state_high):
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
        n_states = len(self.dynObj.state_names)
        self.last_state = np.nan * np.ones((n_states, 1))
        self.state = np.zeros((n_states, 1))

        self.pos_inds = pos_inds
        self.vel_inds = vel_inds
        self.state_args = state_args
        self.ctrl_args = ctrl_args
        self.state_low = state_low
        self.state_high = state_high


class CShape:
    """Contains properties related to the drawn shape.

    Attributes
    ----------
    shape : string
        type of shape to create.
    color : tuple
        RGB triplet for the color in range [0, 255].
    zorder : int
        Determines draw order, lower numbers are drawn first.
    """

    __slots__ = 'shape', 'color', 'zorder'

    def __init__(self, shape, w, h, color, zorder):
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
        if shape.lower() == 'rect':
            self.shape = pygame.Rect((0, 0), (w, h))

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

    __slots__ = 'pos', 'last_pos', 'vel'

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

    __slots__ = 'aabb'

    def __init__(self, w, h):
        self.aabb = pygame.Rect((0, 0), (w, h))


class CBirth:
    """Handles the birth model properties.

    Also handles the generation of the birth location through the use of
    a distribution object from SERUMS.
    """

    __slots__ = '_model'

    def __init__(self, b_type, loc, scale, params):
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
        self._model = smodels.Gaussian(mean=loc,
                                       covariance=scale**2)

    @property
    def loc(self):
        """Read only location sample from the distribution."""
        # TODO fix this hack by updating serums
        return self._model.sample().reshape((2, 1))


class CEvents:
    """Holds the events properties.

    Attributes
    ----------
    events : list
        Each element is a tuple with the first being a
        :class:`gncpy.planning.reinforcement_learning.enums.EventType` and
        the second a dict with extra info.
    """

    __slots__ = 'events'

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

    __slots__ = 'prob_of_death', 'entrance_times'

    def __init__(self, prob_of_death):
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

    __slots__ = 'capabilities'

    def __init__(self, capabilities):
        """Initialize an object.

        Parameters
        ----------
        capabilities : list
            List of capabilities.
        """
        self.capabilities = capabilities


class CPriority:
    """Hold properties about the priority.

    Attributes
    ----------
    priority : float
        Priority value.
    """

    __slots__ = 'priority'

    def __init__(self, priority):
        """Initialize an object.

        Parameters
        ----------
        priority : float
            priority of the object.
        """
        self.priority = priority


# %% Systems
# %%% Base Classes
class Game(ABC):
    """Base class for defining games.

    This should implement all the necessary systems (i.e. game logic) by operating
    on entities. It must be subclassed and defines the expected interface of games.

    Attributes
    ----------
    _entity_manager : :class:`.EntityManager`
        Handles creating and deleting entiies.
    render_fps : int
        fps to render the visualization at if in :code:`'human'` mode.
    game_over : bool
        Flag indicating if the game has ended.
    score : float
        The score from the last call to :meth:`.step`
    """

    __slots__ = ('_entity_manager', '_current_frame', '_render_mode',
                 'render_fps', 'game_over', 'score')

    def __init__(self, render_mode, render_fps=60):
        """Initalize an object.

        The child class should call the :meth:`.parse_config_file` function
        after declaring all its attributes.

        Parameters
        ----------
        render_mode : string
            Mode to render in.
        render_fps : int
            FPS to render at.
        """
        super().__init__()

        self._entity_manager = EntityManager()
        self._current_frame = -1
        self._render_mode = render_mode
        self.render_fps = render_fps

        self.game_over = False
        self.score = 0

    @abstractmethod
    def parse_config_file(self, config_file, **kwargs):
        """Abstract method for parsing the config file and setting up the game.

        Must be implemented by the child class and should be called at the end
        of the constructor.

        Parameters
        ----------
        config_file : string
            Full path of the configuration file.
        kwargs : dict, optional
            Additional arguments needed by subclasses.

        Returns
        -------
        conf : dict
            Dictionary containnig values read from config file.
        """
        raise NotImplementedError

    @abstractmethod
    def step(self, action):
        """Abstract method defining what to do each frame.

        This must be implemented by the child class.

        Parameters
        ----------
        action : numpy array, int, bool, etc.
            action to take in the game.

        Returns
        -------
        info : dict
            Extra infomation for debugging.
        """
        raise NotImplementedError

    @abstractmethod
    def render(self):
        """Implements the rendering system."""
        raise NotImplementedError

    def reset(self):
        """Resets to the base state."""
        self._entity_manager = EntityManager()
        self._current_frame = -1
        self.game_over = False
        self.score = 0

    @abstractmethod
    def s_movement(self, action):
        """Move entities according to their dynamics.

        Parameters
        ----------
        action : numpy array, int, bool, etc.
            action to take in the game.
        """
        raise NotImplementedError

    @abstractmethod
    def s_collision(self):
        """Check for collisions between entities."""
        raise NotImplementedError

    @abstractmethod
    def s_game_over(self):
        """Checks for game over conditions."""
        raise NotImplementedError

    @abstractmethod
    def s_score(self):
        """Calculates the score.

        Returns
        -------
        info : dict
            Extra info for debugging.
        """
        raise NotImplementedError


class Game2d(Game):
    """Base class for defining 2d games.

    This should implement all the necessary systems (i.e. game logic) by operating
    on entities. It assumes the rendering will be done by pygame.

    Attributes
    ----------
    _window : pygame surface
        Main surface (window) to draw on.
    _clock : pygame clock
        main pygame clock for rendering at a given fps.
    max_n_targets : int
        maximum number of targets
    dt : float
        Time increment
    max_time : float
        Maximum game time.
    min_pos : numpy array
        minimum position in the x/y directions in real units
    dist_height : float
        Maximum distance along the vertical axis in real units.
    dist_width : float
        Maximum distance along the horizontal axis in real units.
    dist_per_pix : float
        Distance units per pixel on the scsreen.
    """

    __slots__ = ('_window', '_clock', '_img', 'max_n_targets', 'dt', 'max_time',
                 'min_pos', 'dist_height', 'dist_width', 'dist_per_pix',
                 '_config_file')

    def __init__(self, config_file, render_mode, render_fps=60):
        """Initalize an object.

        The child class should call the :meth:`.parse_config_file` function
        after declaring all its attributes.

        Parameters
        ----------
        config_file : string
            Ful path to the configuration YAML file.
        render_mode : string
            Mode to render in.
        render_fps : int
            FPS to render at.
        """
        super().__init__(render_mode, render_fps=render_fps)

        self._config_file = self.validate_config_file(config_file)

        pygame.init()

        self._window = None
        self._clock = pygame.time.Clock()
        self._img = None

        self.max_n_targets = None
        self.dt = None
        self.max_time = None
        self.min_pos = None
        self.dist_width = None
        self.dist_height = None
        self.dist_per_pix = None

    def reset(self):
        """Resets to the base state."""
        super().reset()
        self._img = 255 * np.ones((*self.get_image_size(), 3), dtype=np.uint8)

    def validate_config_file(self, config_file):
        if os.pathsep in config_file:
            cf = config_file
        else:
            cf = os.path.join(os.getcwd(), config_file)
            if not os.path.isfile(cf):
                cf = os.path.join(pathlib.Path(__file__).parent.resolve(),
                                  config_file)
                if not os.path.isfile(cf):
                    raise RuntimeError('Failed to find config file {}'.format(config_file))
        return cf

    def parse_config_file(self, config_file, extra_keys=None):
        """Parses the config file and sets up the game.

        Should be called at the end of the constructor.

        Parameters
        ----------
        config_file : string
            Full path of the configuration file.
        extra_keys : list
            Each element is a string of extra keys in the config file not
            processed by this function. Errors are printed if any keys are
            found that are not used and are not in extra_keys.

        Returns
        -------
        conf : dict
            Dictionary containnig values read from config file.
        """
        if extra_keys is None:
            extra_keys = ()

        with open(config_file, 'r') as fin:
            conf = yaml.load(fin)

        if 'window' in conf.keys():
            self.setup_window(conf['window'])
        else:
            raise RuntimeError('Must specify window parameters in config')

        if 'physics' in conf.keys():
            self.setup_physics(conf['physics'])
        else:
            raise RuntimeError('Must specify physics parameters in config')

        for key, params in conf.items():
            if key in ('window', 'physics'):
                continue

            elif key == 'player':
                self.setup_player(params)

            elif key == 'obstacles':
                self.setup_obstacles(params)

            elif key == 'targets':
                self.setup_targets(params)

            elif key == 'hazards':
                self.setup_hazards(params)

            elif key not in extra_keys:
                print('Unrecognized key ({}) in config file'.format(key))

        # ensure all things are properly added to the manager so wrappers can use values before calling step
        self._entity_manager.update()

        return conf

    def _pixels_to_dist(self, pt, ind, translate):
        """Convert pixel units to real units.

        Parameters
        ----------
        pt : numpy array, float
            point to convert
        ind : int, list
            index to use from min_pos and dist_per_pix
        translate : bool
            Flag indicating if translation needs to be done (i.e. for position
            conversion).

        Returns
        -------
        numpy array, float
            converted point
        """
        if translate:
            res = pt * self.dist_per_pix[ind] + self.min_pos[ind]
        else:
            res = pt * self.dist_per_pix[ind]

        if isinstance(res, np.ndarray):
            if res.size > 1:
                return res
            else:
                return res.item()
        else:
            return res

    def _dist_to_pixels(self, pt, ind, translate):
        """Convert real units to pixel units.

        Parameters
        ----------
        pt : numpy array, float
            point to convert
        ind : int, list
            index to use from min_pos and dist_per_pix
        translate : bool
            Flag indicating if translation needs to be done (i.e. for position
            conversion).

        Returns
        -------
        numpy array, float
            converted point
        """
        if translate:
            res = (pt - self.min_pos[ind]) / self.dist_per_pix[ind]
        else:
            res = pt / self.dist_per_pix[ind]

        if isinstance(res, np.ndarray):
            if res.size > 1:
                return res.astype(int)
            else:
                return res.astype(int).item()
        else:
            return int(res)

    def setup_window(self, params):
        """Setup the window based on the config file.

        Must have the parameters:

            - width
            - height

        Parameters
        ----------
        params : dict
            window config.
        """
        extra = {}
        if self._render_mode !='human':
            extra['flags'] = pygame.HIDDEN

        self._window = pygame.display.set_mode((int(params['width']),
                                                int(params['height'])),
                                               **extra)

    @abstractmethod
    def get_num_targets(self):
        """Get the number of targets.

        Returns
        -------
        int
            Number of targets.
        """
        raise NotImplementedError

    @abstractmethod
    def get_player_state_bounds(self):
        """Calculate the bounds on the player state

        Returns
        -------
        2 x N numpy array
            minimum and maximum values of the player state.
        """
        raise NotImplementedError

    @abstractmethod
    def get_player_state(self):
        """Return the player state.

        Returns
        -------
        numpy array
            player state.
        """
        raise NotImplementedError

    @abstractmethod
    def setup_player(self, params):
        """Setup the player based on the config file.

        Parameters
        ----------
        params : dict
            Config values.
        """
        raise NotImplementedError

    @abstractmethod
    def setup_obstacles(self, params):
        """Setup the obstacles based on the config file.

        Parameters
        ----------
        params : dict
            Config values.
        """
        raise NotImplementedError

    @abstractmethod
    def setup_targets(self, params):
        """Setup the targets based on the config file.

        Parameters
        ----------
        params : dict
            Config values.
        """
        raise NotImplementedError

    @abstractmethod
    def setup_hazards(self, params):
        """Setup the hazards based on the config file.

        Parameters
        ----------
        params : dict
            Config values.
        """
        raise NotImplementedError

    def setup_physics(self, params):
        """Setup the physics based on the config file.

        Must have parameters:

            - dt
            - max_time
            - min_x
            - min_y
            - dist_width
            - dist_height

        Parameters
        ----------
        params : dict
            Config values.
        """
        self.dt = float(params['dt'])
        self.max_time = float(params['max_time'])
        self.min_pos = np.array([[float(params['min_x'])],
                                 [float(params['min_y'])]])

        self.dist_width = float(params['dist_width'])
        self.dist_height = float(params['dist_height'])
        self.dist_per_pix = np.array([[float(params['dist_width'] / self._window.get_width())],
                                      [float(params['dist_height'] / self._window.get_height())]])

    def _append_name_to_keys(self, in_dict, prefix):
        out = {}
        for key, val in in_dict.items():
            n_key = '{:s}.{:s}'.format(prefix, key)
            out[n_key] = val
        return out

    def step(self, action):
        """Perform one iteration of the game loop.

        Parameters
        ----------
        action : numpy array, int, bool, etc.
            action to take in the game.

        Returns
        -------
        info : dict
            Extra infomation for debugging.
        """
        info = {}
        self._current_frame += 1
        self._entity_manager.update()

        # clear events for entities
        for e in self._entity_manager.get_entities():
            if e.c_events is not None:
                e.c_events.events = []

        self.s_movement(action)
        self.s_collision()

        self.render()

        self.s_game_over()

        s_info = self.s_score()
        info.update(self._append_name_to_keys(s_info, 'Reward'))

        return info

    def render(self):
        """Render a frame of the game."""
        surf = pygame.Surface(self._window.get_size())
        surf.fill((255, 255, 255))

        drawable = list(filter(lambda _e: _e.c_shape and _e.c_transform,
                               self._entity_manager.get_entities()))
        drawable.sort(key=lambda _e: _e.c_shape.zorder)
        for e in drawable:
            if e.c_shape is not None and e.c_transform is not None:
                e.c_shape.shape.centerx = e.c_transform.pos[0].item()
                e.c_shape.shape.centery = e.c_transform.pos[1].item()

                pygame.draw.rect(surf, e.c_shape.color, e.c_shape.shape)

        flip_surf = pygame.transform.flip(surf, False, True)

        if self._render_mode == 'human':
            self._window.blit(flip_surf, (0, 0))

            pygame.event.pump()
            self._clock.tick(self.render_fps)
            pygame.display.flip()

        self._img = np.transpose(np.array(pygame.surfarray.pixels3d(flip_surf),
                                          dtype=np.uint8),
                                 axes=(1, 0, 2))

    def get_screen_rgb(self):
        """Gets a maxtrix representing the screen.

        The order is H, W, C.

        Returns
        -------
        H x W x 3 numpy array
            screen pixel values
        """
        return self._img

    def get_image_size(self):
        """Gets the size of the window.

        Returns
        -------
        tuple
            first is the height next is the width, in pixels.
        """
        sz = self._window.get_size()
        return sz[1], sz[0]

    def close(self):
        """Closes the game."""
        if self._window is not None:
            pygame.quit()


# %%% Custom Games
class SimpleUAV2d(Game2d):
    """Implements a simple 2d UAV scenario.

    Attributes
    ----------
    suppored_reward_types : tuple
        "Each element is a string of a supported reward type.
    """

    __slots__ = ('_scoreCls', '_all_capabilities', '_reward_type', '_start_time')

    supported_reward_types = ('BasicReward', )

    def __init__(self, render_mode, render_fps=None, config_file=None):
        """Initialize an object.

        Parameters
        ----------
        render_mode : string
            Render mode.
        render_fps : int, optional
            Render fps, if set to None then the game dt is used. The default is None.
        config_file : string, optional
            Ful path to the configuration YAML file. The default is simple_uav_2d_config.yaml
        """
        if config_file is None:
            config_file = 'simple_uav_2d_config.yaml'
        super().__init__(config_file, render_mode, render_fps=render_fps)

        # self._config_file = config_file
        self._scoreCls = None
        self._all_capabilities = []
        self._reward_type = None
        self._start_time = 0

        self.parse_config_file(self._config_file)

        if self.render_fps is None:
            self.render_fps = 1 / self.dt

    def setup_score(self, params):
        """Setup the score based on the config file.

        Must have the parameters:

            - type

        Can also have:

            - params

        Parameters
        ----------
        params : dict
            score config.
        """
        rtype = params['type']
        if rtype not in self.supported_reward_types:
            raise RuntimeError('Unsupported score type given ({})'.format(rtype))

        assert hasattr(grewards, rtype), "Failed to find reward class {}"

        self._reward_type = rtype

        cls_type = getattr(grewards, rtype)
        kwargs = {}
        if 'params' in params:
            kwargs = params['params']
        self._scoreCls = cls_type(**kwargs)

    def parse_config_file(self, config_file, extra_keys=None):
        """Parses the config file and sets up the game.

        Should be called at the end of the constructor.

        Parameters
        ----------
        config_file : string
            Full path of the configuration file.
        extra_keys : list
            Each element is a string of extra keys in the config file not
            processed by this function. Errors are printed if any keys are
            found that are not used and are not in extra_keys.

        Returns
        -------
        conf : dict
            Dictionary containnig values read from config file.
        """
        conf = super().parse_config_file(config_file, extra_keys=('score',
                                                                  'start_time'))

        if 'start_time' in conf.keys():
            self._start_time = conf['start_time']

        if 'score' in conf.keys():
            self.setup_score(conf['score'])
        else:
            raise RuntimeError('Must specify score parameters in config')

        return conf

    def _add_capabilities(self, lst):
        for c in lst:
            if c not in self._all_capabilities:
                self._all_capabilities.append(c)

    def get_num_targets(self):
        """Return the number of active targets.

        Returns
        -------
        int
            Number of active targets.
        """
        return sum([1 for t in self._entity_manager.get_entities('target')
                    if t.active])

    def get_player_state_bounds(self):
        """Calculate the bounds on the player state.

        Returns
        -------
        2 x N numpy array
            minimum and maximum values of the player state.
        """
        p_lst = self._entity_manager.get_entities('player')
        p = p_lst[0]
        return np.vstack((p.c_dynamics.state_low.copy(),
                          p.c_dynamics.state_high.copy()))

    def get_player_state(self):
        """Return the player state.

        Returns
        -------
        numpy array
            player state.
        """
        p_lst = self._entity_manager.get_entities('player')
        p = p_lst[0]
        return p.c_dynamics.state.copy().ravel()

    @property
    def current_time(self):
        """Current time in real units."""
        return self.dt * self._current_frame + self._start_time

    def reset(self):
        """Resets to the base state."""
        super().reset()
        self._all_capabilities = []

        self.parse_config_file(self._config_file)

    def _create_dynamics(self, params, c_birth):
        if not hasattr(gdyn, params['type']):
            msg = 'Failed to find dynamics model {}'.format(params['type'])
            raise RuntimeError(msg)
        cls_type = getattr(gdyn, params['type'])

        kwargs = {}
        if params['type'] == 'DoubleIntegrator':
            pos_inds = [0, 1]
            vel_inds = [2, 3]
            state_args = (self.dt, )

            c_params = params['control_model']
            if c_params['type'] == 'velocity':
                ctrl_args = ()

                def _ctrl_mod(t, x, *args):
                    if 'max_vel_x' in c_params and 'max_vel_y' in c_params:
                        mat = np.diag((float(c_params['max_vel_x']),
                                       float(c_params['max_vel_y'])))
                    else:
                        mat = c_params['max_vel'] * np.eye(2)
                    return np.vstack((np.zeros((2, 2)), mat))

            else:
                msg = 'Control model type {} not implemented for dynamics {}'.format(c_params['type'],
                                                                                     params['type'])
                raise NotImplementedError(msg)
            kwargs['control_model'] = _ctrl_mod

            state_low = np.hstack((self.min_pos.ravel(),
                                   np.array([-np.inf, -np.inf])))
            state_high = np.hstack((self.min_pos.ravel() + np.array([self.dist_width, self.dist_height]),
                                    np.array([np.inf, np.inf])))
            if 'state_constraint' in params:
                s_params = params['state_constraint']
                if s_params['type'] == 'velocity':
                    state_low = np.hstack((self.min_pos.ravel(),
                                           np.array(s_params['min_vels'])))
                    state_high = np.hstack((self.min_pos.ravel() + np.array([self.dist_width, self.dist_height]),
                                            np.array(s_params['max_vels'])))

                    def _state_constraint(t, x):
                        x[vel_inds] = np.min(np.vstack((x[vel_inds].ravel(),
                                                        np.array(s_params['max_vels']))),
                                             axis=0).reshape((len(vel_inds), 1))
                        x[vel_inds] = np.max(np.vstack((x[vel_inds].ravel(),
                                                        np.array(s_params['min_vels']))),
                                             axis=0).reshape((len(vel_inds), 1))
                        return x
                else:
                    msg = 'State constraint type {} not implemented for dynamics {}'.format(s_params['type'],
                                                                                            params['type'])
                    raise NotImplementedError(msg)
                kwargs['state_constraint'] = _state_constraint

        elif params['type'] == 'CoordinatedTurn':
            pos_inds = [0, 2]
            vel_inds = [1, 3]
            state_args = ()

            c_params = params['control_model']
            if c_params['type'] == 'velocity_turn':
                ctrl_args = ()

                def _g1(t, x, u, *args):
                    return c_params['max_vel'] * np.cos(x[4].item()) * u[0].item()

                def _g0(t, x, u, *args):
                    return 0

                def _g3(t, x, u, *args):
                    return c_params['max_vel'] * np.sin(x[4].item()) * u[0].item()

                def _g2(t, x, u, *args):
                    return 0

                def _g4(t, x, u, *args):
                    return c_params['max_turn_rate'] * np.pi / 180 * u[1].item()

            else:
                msg = 'Control model type {} not implemented for dynamics {}'.format(c_params['type'],
                                                                                     params['type'])
                raise NotImplementedError(msg)
            kwargs['control_model'] = [_g0, _g1, _g2, _g3, _g4]

            state_low = np.hstack((self.min_pos[0],
                                   np.array([-np.inf]),
                                   self.min_pos[1],
                                   np.array([-np.inf, -2 * np.pi])))
            state_high = np.hstack((self.min_pos[0] + self.dist_width,
                                    np.array([np.inf]),
                                    self.min_pos[1] + self.dist_height,
                                    np.array([np.inf, 2 * np.pi])))
            if 'state_constraint' in params:
                s_params = params['state_constraint']
                if s_params['type'] == 'velocity':
                    state_low = np.hstack((self.min_pos[0],
                                           np.array([s_params['min_vels'][0]]),
                                           self.min_pos[1],
                                           np.array([s_params['min_vels'][1],
                                                     -2 * np.pi])))
                    state_high = np.hstack((self.min_pos[0] + self.dist_width,
                                            np.array([s_params['max_vels'][0]]),
                                            self.min_pos[1] + self.dist_height,
                                            np.array([s_params['max_vels'][1],
                                                      2 * np.pi])))

                    def _state_constraint(t, x):
                        x[vel_inds] = np.min(np.vstack((x[vel_inds].ravel(),
                                                        np.array(s_params['max_vels']))),
                                             axis=0).reshape((len(vel_inds), 1))
                        x[vel_inds] = np.max(np.vstack((x[vel_inds].ravel(),
                                                        np.array(s_params['min_vels']))),
                                             axis=0).reshape((len(vel_inds), 1))
                        if x[4] < 0:
                            x[4] = np.mod(x[4], -2 * np.pi)
                        else:
                            x[4] = np.mod(x[4], 2 * np.pi)

                        return x
                else:
                    msg = 'State constraint type {} not implemented for dynamics {}'.format(s_params['type'],
                                                                                            params['type'])
                    raise NotImplementedError(msg)
                kwargs['state_constraint'] = _state_constraint

        if 'params' in params:
            kwargs.update(params['params'])

        c_dynamics = CDynamics(cls_type(**kwargs), pos_inds, vel_inds, state_args,
                               ctrl_args, state_low, state_high)
        c_dynamics.state[pos_inds] = c_birth.loc

        rng = np.random.default_rng()
        if params['type'] == 'CoordinatedTurn':
            c_dynamics.state[4] = rng.random() * 2 * np.pi

        return c_dynamics

    def setup_player(self, params):
        """Setup the player based on the config file.

        Must have the parameters:

            - birth_model
                - location
                - scale
                - type
                - params
            - dynamics_model
                - type
                - params
                - control_model
                    - type
                    - max_vel or (max_vel_x and max_vel_y)
                    - max_turn_rate
                - state_constraint
                    - type
                    - min_vels
                    - max_vels
            -shape_model
                - type
                - height
                - width
                - color
            - collision_model
                - width
                - height

        Can also have:

            - capabilities

        Parameters
        ----------
        params : dict
            player config.
        """
        e = self._entity_manager.add_entity('player')

        # TODO: allow for other birth types
        b_params = params['birth_model']
        b_loc = np.array(b_params['location']).reshape((len(b_params['location']), 1))
        b_scale = np.diag(b_params['scale'])
        e.c_birth = CBirth(b_params['type'], b_loc, b_scale, b_params['params'])

        # TODO: allow for other dyanmics models
        e.c_dynamics = self._create_dynamics(params['dynamics_model'], e.c_birth)

        e.c_transform = CTransform()

        # TODO: allow for other shape types
        s_params = params['shape_model']
        e.c_shape = CShape(s_params['type'], self._dist_to_pixels(s_params['width'], 0, False),
                           self._dist_to_pixels(s_params['height'], 1, False),
                           tuple(s_params['color']), 100)

        c_params = params['collision_model']
        e.c_collision = CCollision(self._dist_to_pixels(c_params['width'], 0, False),
                                   self._dist_to_pixels(c_params['height'], 1, False))

        e.c_events = CEvents()

        key = 'capabilities'
        if key in params:
            capabilities = params[key]
        else:
            capabilities = []
        e.c_capabilities = CCapabilities(capabilities)
        self._add_capabilities(capabilities)

    def setup_obstacles(self, params):
        """Setup the obstacles based on the config file.

        Must be a list where each element has the parameters:

            - loc_x
            - loc_y
            - shape_type
            - width
            - height
            - shape_color
            - collision_width
            - collision_height

        Parameters
        ----------
        params : dict
            obstacle config.
        """
        for o_params in params:
            e = self._entity_manager.add_entity('obstacle')

            e.c_transform = CTransform()
            e.c_transform.pos[0] = self._dist_to_pixels(o_params['loc_x'], 0, True)
            e.c_transform.pos[1] = self._dist_to_pixels(o_params['loc_y'], 1, True)
            e.c_transform.last_pos[0] = e.c_transform.pos[0]
            e.c_transform.last_pos[1] = e.c_transform.pos[1]

            e.c_shape = CShape(o_params['shape_type'],
                               self._dist_to_pixels(o_params['width'], 0, False),
                               self._dist_to_pixels(o_params['height'], 1, False),
                               tuple(o_params['shape_color']), 1000)
            e.c_collision = CCollision(self._dist_to_pixels(o_params['collision_width'],
                                                            0, False),
                                       self._dist_to_pixels(o_params['collision_height'],
                                                            1, False))

    def setup_targets(self, params):
        """Setup the targets  based on the config file.

        Must be a list where each element has the parameters:

            - loc_x
            - loc_y
            - shape_type
            - width
            - height
            - shape_color
            - collision_width
            - collision_height

        Each can also have:

            - capabilities

        Parameters
        ----------
        params : dict
            target config.

        Todo
        ----
        Allow for a sequence of targets.
        """
        self.max_n_targets = len(params)
        for t_params in params:
            e = self._entity_manager.add_entity('target')

            e.c_transform = CTransform()
            e.c_transform.pos[0] = self._dist_to_pixels(t_params['loc_x'], 0, True)
            e.c_transform.pos[1] = self._dist_to_pixels(t_params['loc_y'], 1, True)
            e.c_transform.last_pos[0] = e.c_transform.pos[0]
            e.c_transform.last_pos[1] = e.c_transform.pos[1]

            e.c_shape = CShape(t_params['shape_type'],
                               self._dist_to_pixels(t_params['width'], 0, False),
                               self._dist_to_pixels(t_params['height'], 1, False),
                               tuple(t_params['shape_color']), 1)
            e.c_collision = CCollision(self._dist_to_pixels(t_params['collision_width'],
                                                            0, False),
                                       self._dist_to_pixels(t_params['collision_height'],
                                                            1, False))

            key = 'capabilities'
            if key in t_params:
                capabilities = t_params[key]
            else:
                capabilities = []
            e.c_capabilities = CCapabilities(capabilities)
            self._add_capabilities(capabilities)

            e.c_priority = CPriority(t_params['priority'])

    def setup_hazards(self, params):
        """Setup the hazards  based on the config file.

        Must be a list where each element has the parameters:

            - loc_x
            - loc_y
            - shape_type
            - width
            - height
            - shape_color
            - collision_width
            - collision_height
            - prob_of_death

        Parameters
        ----------
        params : dict
            hazard config.
        """
        for h_params in params:
            e = self._entity_manager.add_entity('hazard')

            e.c_transform = CTransform()
            e.c_transform.pos[0] = self._dist_to_pixels(h_params['loc_x'], 0, True)
            e.c_transform.pos[1] = self._dist_to_pixels(h_params['loc_y'], 1, True)
            e.c_transform.last_pos[0] = e.c_transform.pos[0]
            e.c_transform.last_pos[1] = e.c_transform.pos[1]

            e.c_shape = CShape(h_params['shape_type'],
                               self._dist_to_pixels(h_params['width'], 0, False),
                               self._dist_to_pixels(h_params['height'], 1, False),
                               tuple(h_params['shape_color']), -100)
            e.c_collision = CCollision(self._dist_to_pixels(h_params['collision_width'],
                                                            0, False),
                                       self._dist_to_pixels(h_params['collision_height'],
                                                            1, False))
            pd = float(h_params['prob_of_death'])
            if pd > 1:
                pd = pd / 100.
            e.c_hazard = CHazard(pd)

    def s_movement(self, action):
        """Move entities according to their dynamics.

        Parameters
        ----------
        action : 2 x 1 numpy array
            Control inputs for the given dynamics model.

        Returns
        -------
        None.
        """
        for e in self._entity_manager.get_entities():
            if e.c_transform is not None:
                e.c_transform.last_pos[0] = e.c_transform.pos[0]
                e.c_transform.last_pos[1] = e.c_transform.pos[1]

            if e.c_dynamics is not None and e.c_transform is not None:
                e.c_dynamics.last_state = e.c_dynamics.state.copy()
                e.c_dynamics.state = e.c_dynamics.dynObj.propagate_state(self.current_time,
                                                                         e.c_dynamics.last_state,
                                                                         u=action.reshape((action.size, 1)),
                                                                         state_args=e.c_dynamics.state_args,
                                                                         ctrl_args=e.c_dynamics.ctrl_args)

                p_ii = e.c_dynamics.pos_inds
                v_ii = e.c_dynamics.vel_inds
                e.c_transform.pos = self._dist_to_pixels(e.c_dynamics.state[p_ii],
                                                         [0, 1], True)
                e.c_transform.vel = self._dist_to_pixels(e.c_dynamics.state[v_ii],
                                                         [0, 1], False)

    def _get_overlap(self, bb1, bb2):
        delta = (abs(bb1.centerx - bb2.centerx), abs(bb1.centery - bb2.centery))
        ox = bb1.width / 2 + bb2.width / 2 - delta[0]
        oy = bb1.height / 2 + bb2.height / 2 - delta[1]
        return ox, oy

    def _get_last_overlap(self, pt1, pt2, bb1, bb2):
        delta = np.abs(pt1 - pt2)
        ox = bb1.width / 2 + bb2.width / 2 - delta[0].item()
        oy = bb1.height / 2 + bb2.height / 2 - delta[1].item()
        return ox, oy

    def s_collision(self):
        """Check for collisions between entities.

        This also handles player death if a hazard destroys a player, and
        updates the events.

        Returns
        -------
        None.
        """
        rng = np.random.default_rng()

        # update all bounding boxes
        for e in self._entity_manager.get_entities():
            if e.c_transform is not None and e.c_collision is not None:
                e.c_collision.aabb.centerx = e.c_transform.pos[0].item()
                e.c_collision.aabb.centery = e.c_transform.pos[1].item()

        # check for collision of player
        for e in self._entity_manager.get_entities('player'):
            if e.c_transform is not None and e.c_collision is not None:
                p_aabb = e.c_collision.aabb
                p_trans = e.c_transform

                # check for out of bounds, stop at out of bounds
                went_oob = False
                if p_aabb.left < 0:
                    p_aabb.left = 0
                    p_trans.vel[0] = 0
                    went_oob = True
                elif p_aabb.right > self._window.get_width():
                    p_aabb.right = self._window.get_width()
                    p_trans.vel[0] = 0
                    went_oob = True

                if went_oob and e.c_events is not None:
                    e.c_events.events.append((EventType.WALL, None))
                went_oob = False

                if p_aabb.top < 0:
                    p_aabb.top = 0
                    p_trans.vel[1] = 0
                    went_oob = True
                elif p_aabb.bottom > self._window.get_height():
                    p_aabb.bottom = self._window.get_height()
                    p_trans.vel[1] = 0
                    went_oob = True

                if went_oob and e.c_events is not None:
                    e.c_events.events.append((EventType.WALL, None))

                # check for collision with wall
                for w in self._entity_manager.get_entities('obstacle'):
                    w_aabb = w.c_collision.aabb
                    ox, oy = self._get_overlap(p_aabb, w_aabb)

                    if ox > 0 and oy > 0:
                        opx, opy = self._get_last_overlap(p_trans.last_pos,
                                                          w.c_transform.last_pos,
                                                          p_aabb, w_aabb)
                        if opy > 0:
                            p_trans.vel[0] = 0
                            if p_trans.last_pos[0] < p_trans.pos[0]:
                                p_aabb.centerx -= ox
                            else:
                                p_aabb.centerx += ox
                        elif opx > 0:
                            p_trans.vel[1] = 0
                            if p_trans.last_pos[1] < p_trans.pos[1]:
                                p_aabb.centery -= oy
                            else:
                                p_aabb.centery += oy

                        if e.c_events is not None:
                            e.c_events.events.append((EventType.WALL, None))

                # check for collision with hazard
                for h in self._entity_manager.get_entities('hazard'):
                    h_aabb = h.c_collision.aabb
                    if not pygame.Rect.colliderect(p_aabb, h_aabb):
                        if e.id in h.c_hazard.entrance_times:
                            del h.c_hazard.entrance_times[e.id]
                    else:
                        if rng.uniform(0., 1.) < h.c_hazard.prob_of_death:
                            e.destroy()
                            e.c_events.events.append((EventType.DEATH, None))
                            if e.id in h.c_hazard.entrance_times:
                                del h.c_hazard.entrance_times[e.id]
                            break

                        else:
                            if e.id not in h.c_hazard.entrance_times:
                                h.c_hazard.entrance_times[e.id] = self.current_time
                            e.c_events.events.append((EventType.HAZARD,
                                                      {'prob': h.c_hazard.prob_of_death,
                                                       't_ent': h.c_hazard.entrance_times[e.id]}))

                if not e.active:
                    continue

                # check for collision with target
                for ii, t in enumerate(self._entity_manager.get_entities('target')):
                    if not t.active:
                        continue

                    if pygame.Rect.colliderect(p_aabb, t.c_collision.aabb):
                        e.c_events.events.append((EventType.TARGET, {'target': t}))
                        t.destroy()
                        break

                # update position
                p_trans.pos[0] = p_aabb.centerx
                p_trans.pos[1] = p_aabb.centery

                # update state
                if e.c_dynamics is not None:
                    p_ii = e.c_dynamics.pos_inds
                    v_ii = e.c_dynamics.vel_inds

                    e.c_dynamics.state[p_ii] = self._pixels_to_dist(p_trans.pos,
                                                                    [0, 1], True)
                    e.c_dynamics.state[v_ii] = self._pixels_to_dist(p_trans.vel,
                                                                    [0, 1], False)

    def s_game_over(self):
        """Determines if the game ends.

        Game ends if there are no active players, the entity manager has no
        targets remaining, or the time is greater than or equal to the max time.

        Returns
        -------
        None.
        """
        alive_players = list(filter(lambda _x: _x.active,
                                    self._entity_manager.get_entities('player')))
        self.game_over = (self.current_time >= self.max_time
                          or len(self._entity_manager.get_entities('target')) == 0
                          or len(alive_players) == 0)

    def s_score(self):
        """Calculate the score using the reward class instance.

        Raises
        ------
        NotImplementedError
            If the reward type has not been implemented.

        Returns
        -------
        info : dict
            extra info for debugging.
        """
        if self._reward_type == 'BasicReward':
            self.score, info = self._scoreCls.calc_reward(self.current_time,
                                                          self._entity_manager.get_entities('player'),
                                                          self._entity_manager.get_entities('target'),
                                                          self._all_capabilities, self.game_over)
        else:
            msg = 'Score system has no implementation for reward type {}'.format(self._reward_type)
            raise NotImplementedError(msg)

        return info
