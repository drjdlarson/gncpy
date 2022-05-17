from abc import ABC, abstractmethod
import pathlib
import os
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"
import pygame
import numpy as np
from ruamel.yaml import YAML
from warnings import warn

from gncpy.planning.reinforcement_learning.entities import EntityManager
import gncpy.planning.reinforcement_learning.components as gcomp
import gncpy.planning.reinforcement_learning.rewards as grewards
import gncpy.planning.reinforcement_learning.physics as gphysics
import gncpy.dynamics as gdyn
from gncpy.planning.reinforcement_learning.enums import EventType


yaml = YAML()


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
    def step(self, *args):
        """Abstract method defining what to do each frame.

        This must be implemented by the child class.

        Returns
        -------
        info : dict
            Extra infomation for debugging.
        """
        raise NotImplementedError

    @abstractmethod
    def s_render(self):
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

    __slots__ = ('_window', '_clock', '_img', 'max_n_targets', 'dt', '_start_time',
                 '_max_time', 'min_pos', 'dist_height', 'dist_width', 'dist_per_pix',
                 '_config_file', 'step_factor', '_current_update_count')

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
        self._start_time = 0
        self._max_time = None
        self.min_pos = None
        self.dist_width = None
        self.dist_height = None
        self.dist_per_pix = None
        self.step_factor = 1
        self._current_update_count = -1

    @property
    def max_time(self):
        # TODO: figure out if this works
        return self._max_time  # + self._start_time

    def reset(self):
        """Resets to the base state."""
        super().reset()
        self._img = 255 * np.ones((*self.get_image_size(), 3), dtype=np.uint8)
        self._current_update_count = -1

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

        if 'start_time' in conf.keys():
            self._start_time = conf['start_time']

        if 'step_factor' in conf.keys():
            self.step_factor = conf['step_factor']

        self.dt /= self.step_factor

        for key, params in conf.items():
            if key in ('window', 'physics', 'start_time', 'step_factor'):
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
        if self._render_mode != 'human':
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
        self._max_time = float(params['max_time'])
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

    @abstractmethod
    def s_input(self, user_input):
        raise NotImplementedError('Must be implemented by child class')

    def step(self, user_input):
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

        self.score = 0
        for ii in range(self.step_factor):
            self._current_update_count += 1
            self._entity_manager.update()

            # clear events for entities
            for e in self._entity_manager.get_entities():
                if e.has_component(gcomp.CEvents):
                    e.get_component(gcomp.CEvents).events = []

            action = self.s_input(user_input)
            self.s_movement(action)
            self.s_collision()
            score, s_info = self.s_score()
            self.score += score
        self.score /= self.step_factor

        self.s_render()

        self.s_game_over()

        info.update(self._append_name_to_keys(s_info, 'Reward'))

        return info

    def s_render(self):
        """Render a frame of the game."""
        surf = pygame.Surface(self._window.get_size())
        surf.fill((255, 255, 255))

        drawable = list(filter(lambda _e: _e.has_component(gcomp.CShape) and _e.has_component(gcomp.CTransform),
                               self._entity_manager.get_entities()))
        drawable.sort(key=lambda _e: _e.get_component(gcomp.CShape).zorder)
        for e in drawable:
            e_shape = e.get_component(gcomp.CShape)
            e_trans = e.get_component(gcomp.CTransform)

            e_shape.shape.centerx = e_trans.pos[0].item()
            e_shape.shape.centery = e_trans.pos[1].item()

            if isinstance(e_shape.shape, pygame.Rect):
                pygame.draw.rect(surf, e_shape.color, e_shape.shape)
            else:
                warn('No rendering method for this shape')

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


class SimpleUAV2d(Game2d):
    """Implements a simple 2d UAV scenario.

    Attributes
    ----------
    suppored_reward_types : tuple
        "Each element is a string of a supported reward type.
    """

    __slots__ = ('_scoreCls', '_all_capabilities', '_reward_type', '_rng', '_seed')

    supported_reward_types = ('BasicReward', )

    def __init__(self, render_mode, render_fps=None, config_file=None, seed=None):
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

        self._scoreCls = None
        self._all_capabilities = []
        self._reward_type = None
        if seed is None:
            self._rng = np.random.default_rng()
        else:
            self._rng = np.random.default_rng(seed)
        self._seed = seed

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
        conf = super().parse_config_file(config_file, extra_keys=('score',))

        if 'score' in conf.keys():
            self.setup_score(conf['score'])
        else:
            raise RuntimeError('Must specify score parameters in config')

        return conf

    def _add_capabilities(self, lst):
        if lst is None:
            return

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
        return np.vstack((p.get_component(gcomp.CDynamics).state_low.copy(),
                          p.get_component(gcomp.CDynamics).state_high.copy()))

    def get_player_state(self):
        """Return the player state.

        Returns
        -------
        numpy array
            player state.
        """
        p_lst = self._entity_manager.get_entities('player')
        p = p_lst[0]
        return p.get_component(gcomp.CDynamics).state.copy().ravel()

    @property
    def current_time(self):
        """Current time in real units."""
        return self.dt * self._current_update_count + self._start_time

    def reset(self, seed=None):
        """Resets to the base state."""
        super().reset()
        self._all_capabilities = []

        if seed is None:
            if self._seed is None:
                self._rng = np.random.default_rng()
            else:
                self._rng = np.random.default_rng(self._seed)
        else:
            self._rng = np.random.default_rng(seed)
            self._seed = seed

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

        dynObj = cls_type(**kwargs)
        state0 = np.zeros((state_low.size, 1))
        if c_birth.randomize:
            state0[pos_inds] = c_birth.sample().reshape(state0[pos_inds].shape)
        else:
            state0[pos_inds] = c_birth.loc.copy().reshape(state0[pos_inds].shape)

        if params['type'] == 'CoordinatedTurn':
            state0[4] = self._rng.random() * 2 * np.pi

        return (dynObj, pos_inds, vel_inds, state_args, ctrl_args, state_low,
                state_high, state0)

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

        b_params = params['birth_model']
        b_loc = np.array(b_params['location']).reshape((len(b_params['location']), 1))
        b_scale = np.diag(b_params['scale'])
        randomize = 'randomize' not in b_params or b_params['randomize']
        e.add_component(gcomp.CBirth, b_type=b_params['type'], loc=b_loc,
                        scale=b_scale, params=b_params['params'], rng=self._rng,
                        randomize=randomize)

        e.add_component(gcomp.CDynamics)
        c_dynamics = e.get_component(gcomp.CDynamics)
        (c_dynamics.dynObj, c_dynamics.pos_inds, c_dynamics.vel_inds,
         c_dynamics.state_args, c_dynamics.ctrl_args, c_dynamics.state_low,
         c_dynamics.state_high, c_dynamics.state) = self._create_dynamics(params['dynamics_model'],
                                                                          e.get_component(gcomp.CBirth))

        e.add_component(gcomp.CTransform)

        s_params = params['shape_model']
        e.add_component(gcomp.CShape, shape=s_params['type'],
                        w=self._dist_to_pixels(s_params['width'], 0, False),
                        h=self._dist_to_pixels(s_params['height'], 1, False),
                        color=tuple(s_params['color']), zorder=100)

        c_params = params['collision_model']
        e.add_component(gcomp.CCollision,
                        w=self._dist_to_pixels(c_params['width'], 0, False),
                        h=self._dist_to_pixels(c_params['height'], 1, False))

        e.add_component(gcomp.CEvents)

        key = 'capabilities'
        if key in params:
            capabilities = params[key]
        else:
            capabilities = None
        e.add_component(gcomp.CCapabilities, capabilities=capabilities)
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

            e.add_component(gcomp.CTransform)
            c_transform = e.get_component(gcomp.CTransform)
            c_transform.pos[0] = self._dist_to_pixels(o_params['loc_x'], 0, True)
            c_transform.pos[1] = self._dist_to_pixels(o_params['loc_y'], 1, True)
            c_transform.last_pos[0] = c_transform.pos[0]
            c_transform.last_pos[1] = c_transform.pos[1]

            e.add_component(gcomp.CShape, shape=o_params['shape_type'],
                            w=self._dist_to_pixels(o_params['width'], 0, False),
                            h=self._dist_to_pixels(o_params['height'], 1, False),
                            color=tuple(o_params['shape_color']), zorder=1000)
            e.add_component(gcomp.CCollision,
                            w=self._dist_to_pixels(o_params['collision_width'], 0, False),
                            h=self._dist_to_pixels(o_params['collision_height'], 1, False))

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

            e.add_component(gcomp.CTransform)
            c_transform = e.get_component(gcomp.CTransform)
            c_transform.pos[0] = self._dist_to_pixels(t_params['loc_x'], 0, True)
            c_transform.pos[1] = self._dist_to_pixels(t_params['loc_y'], 1, True)
            c_transform.last_pos[0] = c_transform.pos[0]
            c_transform.last_pos[1] = c_transform.pos[1]

            e.add_component(gcomp.CShape, shape=t_params['shape_type'],
                            w=self._dist_to_pixels(t_params['width'], 0, False),
                            h=self._dist_to_pixels(t_params['height'], 1, False),
                            color=tuple(t_params['shape_color']), zorder=1)
            e.add_component(gcomp.CCollision,
                            w=self._dist_to_pixels(t_params['collision_width'], 0, False),
                            h=self._dist_to_pixels(t_params['collision_height'], 1, False))

            key = 'capabilities'
            if key in t_params:
                capabilities = t_params[key]
            else:
                capabilities = None
            e.add_component(gcomp.CCapabilities, capabilities=capabilities)
            self._add_capabilities(capabilities)

            e.add_component(gcomp.CPriority, priority=t_params['priority'])

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

            e.add_component(gcomp.CTransform)
            c_transform = e.get_component(gcomp.CTransform)
            c_transform.pos[0] = self._dist_to_pixels(h_params['loc_x'], 0, True)
            c_transform.pos[1] = self._dist_to_pixels(h_params['loc_y'], 1, True)
            c_transform.last_pos[0] = c_transform.pos[0]
            c_transform.last_pos[1] = c_transform.pos[1]

            e.add_component(gcomp.CShape, shape=h_params['shape_type'],
                            w=self._dist_to_pixels(h_params['width'], 0, False),
                            h=self._dist_to_pixels(h_params['height'], 1, False),
                            color=tuple(h_params['shape_color']), zorder=-100)

            e.add_component(gcomp.CCollision,
                            w=self._dist_to_pixels(h_params['collision_width'], 0, False),
                            h=self._dist_to_pixels(h_params['collision_height'], 1, False))

            pd = float(h_params['prob_of_death'])
            if pd > 1:
                pd = pd / 100.
            e.add_component(gcomp.CHazard, prob_of_death=pd)

    def _propagate_dynamics(self, e_dyn, action):
        e_dyn.state = e_dyn.dynObj.propagate_state(self.current_time,
                                                   e_dyn.last_state,
                                                   u=action,
                                                   state_args=e_dyn.state_args,
                                                   ctrl_args=e_dyn.ctrl_args)

    def s_input(self, user_input):
        return user_input.reshape((-1, 1))

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
            if e.has_component(gcomp.CTransform):
                e_transform = e.get_component(gcomp.CTransform)
                e_transform.last_pos[0] = e_transform.pos[0]
                e_transform.last_pos[1] = e_transform.pos[1]

                if e.has_component(gcomp.CDynamics):
                    e_dyn = e.get_component(gcomp.CDynamics)
                    e_dyn.last_state = e_dyn.state.copy()
                    self._propagate_dynamics(e_dyn, action)

                    p_ii = e_dyn.pos_inds
                    v_ii = e_dyn.vel_inds
                    e_transform.pos = self._dist_to_pixels(e_dyn.state[p_ii],
                                                           [0, 1], True)
                    if v_ii is not None:
                        e_transform.vel = self._dist_to_pixels(e_dyn.state[v_ii],
                                                               [0, 1], False)

    def s_collision(self):
        """Check for collisions between entities.

        This also handles player death if a hazard destroys a player, and
        updates the events.

        Returns
        -------
        None.
        """
        # update all bounding boxes
        for e in self._entity_manager.get_entities():
            if e.has_component(gcomp.CTransform) and e.has_component(gcomp.CCollision):
                c_collision = e.get_component(gcomp.CCollision)
                c_transform = e.get_component(gcomp.CTransform)
                c_collision.aabb.centerx = c_transform.pos[0].item()
                c_collision.aabb.centery = c_transform.pos[1].item()

        # check for collision of player
        for e in self._entity_manager.get_entities('player'):
            p_aabb = e.get_component(gcomp.CCollision).aabb
            p_trans = e.get_component(gcomp.CTransform)
            p_events = e.get_component(gcomp.CEvents)

            # check for out of bounds, stop at out of bounds
            out_side, out_top = gphysics.clamp_window_bounds2d(p_aabb, p_trans,
                                                               self._window.get_width(),
                                                               self._window.get_height())
            if out_side:
                p_events.events.append((EventType.WALL, None))
            if out_top:
                p_events.events.append((EventType.WALL, None))

            # check for collision with wall
            for w in self._entity_manager.get_entities('obstacle'):
                w_aabb = w.get_component(gcomp.CCollision).aabb
                if gphysics.check_collision2d(p_aabb, w_aabb):
                    gphysics.resolve_collision2d(p_aabb, w_aabb, p_trans,
                                                 w.get_component(gcomp.CTransform))
                    p_events.events.append((EventType.WALL, None))

            # check for collision with hazard
            for h in self._entity_manager.get_entities('hazard'):
                h_aabb = h.get_component(gcomp.CCollision).aabb
                c_hazard = h.get_component(gcomp.CHazard)
                if gphysics.check_collision2d(p_aabb, h_aabb):
                    if self._rng.uniform(0., 1.) < c_hazard.prob_of_death:
                        e.destroy()
                        p_events.events.append((EventType.DEATH, None))
                        if e.id in c_hazard.entrance_times:
                            del c_hazard.entrance_times[e.id]

                    else:
                        if e.id not in c_hazard.entrance_times:
                            c_hazard.entrance_times[e.id] = self.current_time
                        e.c_events.events.append((EventType.HAZARD,
                                                  {'prob': c_hazard.prob_of_death,
                                                   't_ent': c_hazard.entrance_times[e.id]}))
                else:
                    if e.id in c_hazard.entrance_times:
                        del c_hazard.entrance_times[e.id]

            if not e.active:
                continue

            # check for collision with target
            for t in self._entity_manager.get_entities('target'):
                if not t.active:
                    continue

                if gphysics.check_collision2d(p_aabb,
                                              t.get_component(gcomp.CCollision).aabb):
                    p_events.events.append((EventType.TARGET, {'target': t}))
                    t.destroy()
                    break

            # update state
            p_dynamics = e.get_component(gcomp.CDynamics)
            p_ii = p_dynamics.pos_inds
            v_ii = p_dynamics.vel_inds

            p_dynamics.state[p_ii] = self._pixels_to_dist(p_trans.pos,
                                                          [0, 1], True)
            if v_ii is not None:
                p_dynamics.state[v_ii] = self._pixels_to_dist(p_trans.vel,
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
            score, info = self._scoreCls.calc_reward(self.current_time,
                                                     self._entity_manager.get_entities('player'),
                                                     self._entity_manager.get_entities('target'),
                                                     self._all_capabilities, self.game_over)
        else:
            msg = 'Score system has no implementation for reward type {}'.format(self._reward_type)
            raise NotImplementedError(msg)

        return score, info

    def valid_start(self):
        valid = True
        for e in self._entity_manager.get_entities('player'):
            p_aabb = e.get_component(gcomp.CCollision).aabb

            for w in self._entity_manager.get_entities('obstacle'):
                if gphysics.check_collision2d(p_aabb,
                                              w.get_component(gcomp.CCollision).aabb):
                    valid = False
                    break
            if not valid:
                break
        return valid


class SimpleLagerSUPER(SimpleUAV2d):
    def _create_dynamics(self, params, c_birth):
        pass

    def _propagate_dynamics(self, e_dyn, action):
        pass

    def s_input(self, user_input):
        pass
