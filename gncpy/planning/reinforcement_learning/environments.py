import os
import pathlib
import gym
from gym import spaces
import numpy as np
from abc import abstractmethod

import gncpy.planning.reinforcement_learning.game as rl_games


class BaseEnv(gym.Env):
    """Abstract base class for all environments.

    This defines the interrface common to all environment.
    """

    metadata = {"render.modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, game):
        """Initialize an object.

        Parameters
        ----------
        game : :class:`gncpy.planning.reinforcement_learning.game.Game`
            game to learn to play.
        """
        super().__init__()

        self.action_space = None
        self.observation_space = None

        self._game = game
        self.metadata['render_fps'] = self._game.render_fps

    def step(self, action):
        """Perform one step of the environment.

        Parameters
        ----------
        action : numpy array, float, int, dict, etc.
            action to take.

        Returns
        -------
        obs : numpy array, float, int, dict, etc.
            observation after taking the action.
        reward : float
            reward from the current step.
        done : bool
            Flag indicating if the game is over.
        info : dict
            extra info for debugging.
        """
        info = {}

        info.update(self._game.step(action))

        info.update(self.generate_step_info())
        obs = self.generate_observation()

        return obs, self._game.score, self._game.game_over, info

    @abstractmethod
    def generate_observation(self):
        """Abstract method for generating observations."""
        raise NotImplementedError

    def generate_step_info(self):
        """Create the dictionary containing extra info for debugging the step function.

        Returns
        -------
        dict
            extra debugging info.
        """
        return {}

    @abstractmethod
    def render(self):
        """Abstract method for implementing the rendering system."""
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        """Abstract method for implementing the reset function.

        Returns
        -------
        obs : numpy arra, float, int, dict, etc
            observation
        """
        raise NotImplementedError


class SimpleUAV2d(BaseEnv):
    """Implements a simple 2d UAV scenario.

    This is based on the simple uav 2d game. The state is configurable through
    constructor parameters, and the scenario can be customized by supplying
    a custom YAML configuration file.
    """

    def __init__(self, config_file='simple_uav_2d_config.yaml',
                 render_mode='rgb_array', render_fps=None, obs_type='player_state',
                 aux_use_n_targets=False, aux_use_time=False):
        """Initialize an object.

        Parameters
        ----------
        config_file : string, optional
            Full path to the config YAML file. The default is 'simple_uav_2d_config.yaml'.
        render_mode : string, optional
            render mode. If human is specified here then render does not need
            to be called to visualize the results. The default is 'rgb_array'.
        render_fps : int, optional
            Render FPS if none then the games dt is used. The default is None.
        obs_type : string, optional
            Observation type to use. Can be `image` or `player_state`. The default
            is 'player_state'.
        aux_use_n_targets : bool, optional
            Flag indicating if the auxilary state should use the number
            of targets remaining. The default is False.
        aux_use_time : bool, optional
            Flag indicating if the axuilary state should use the timestep. The
            default is False.

        Raises
        ------
        RuntimeError
            If the config file is not found.
        """
        if os.pathsep in config_file:
            cf = config_file
        else:
            cf = os.path.join(os.getcwd(), config_file)
            if not os.path.isfile(cf):
                cf = os.path.join(pathlib.Path(__file__).parent.resolve(),
                                  config_file)
                if not os.path.isfile(cf):
                    raise RuntimeError('Failed to find config file {}'.format(config_file))

        game = rl_games.SimpleUAV2d(render_mode, config_file=cf, render_fps=render_fps)

        super().__init__(game)

        self._render_mode = render_mode
        self._obs_type = obs_type
        self._aux_use_n_targets = aux_use_n_targets
        self._aux_use_time = aux_use_time
        self._fig = None

        self.action_space = spaces.Box(low=-1., high=1., shape=(2,))
        # self.action_space = spaces.Box(low=np.array([0, -1]), high=np.array([1, 1]))
        self.observation_space = self._calc_obs_space()

    def _calc_obs_space(self):
        # determine main state
        main_state = None
        if self._obs_type == 'image':
            shape = (*self._game.get_image_size(), 3)
            main_state = spaces.Box(low=0, high=255, shape=shape, dtype=np.uint8)

        elif self._obs_type =='player_state':
            state_bnds = self._game.get_player_state_bounds()
            main_state = spaces.Box(low=state_bnds[0],
                                    high=state_bnds[1])

        else:
           raise RuntimeError('Invalid observation type ({})'.format(self._obs_type))

        # create aux state if needed
        aux_state_low = np.array([])
        aux_state_high = np.array([])

        if self._aux_use_n_targets:
            aux_state_low = np.append(aux_state_low, 0)
            aux_state_high = np.append(aux_state_high, self._game.max_n_targets)

        if self._aux_use_time:
            aux_state_low = np.append(aux_state_low, 0)
            aux_state_high = np.append(aux_state_high, self._game.max_time)

        # combine into final space
        if self._obs_type == 'image':
            if aux_state_low.size > 0:
                aux_state = spaces.Box(aux_state_low, aux_state_high)
                out = spaces.Dict({'img': main_state, 'aux': aux_state})

            else:
                out = main_state

        else:
            if aux_state_low.size > 0:
                low = np.concatenate((main_state.low, aux_state_low))
                high = np.concatenate((main_state.high, aux_state_high))
                out = spaces.Box(low, high)

            else:
                out = main_state

        return out

    def close(self):
        """Closes the environment."""
        self._game.close()

    def render(self, mode='human'):
        """Renders a frame of the environment according to the mode.

        The mode should be set in the constructor, and if using human then this
        function does not need to be called. If using rgb_array then this returns
        a H x W x 3 numpy array of the screen pixels.
        """
        img = self._game.get_screen_rgb()

        if mode == "rgb_array":
            return img

        # make sure if this is called when initialized with human mode that this function does nothing
        elif mode == "human" and self._render_mode != 'human':
            import matplotlib.pyplot as plt

            if self._fig is None:
                px2in = 1 / plt.rcParams['figure.dpi']  # pixel in inches
                orig = plt.rcParams['toolbar']
                plt.rcParams['toolbar'] = 'None'

                self._fig = plt.figure(figsize=(px2in * img.shape[0],
                                                px2in * img.shape[1]))
                self._fig.add_axes([0, 0, 1, 1], frame_on=False, rasterized=True)

                plt.rcParams['toolbar'] = orig

            self._fig.axes[0].clear()
            self._fig.axes[0].imshow(img)
            plt.pause(1 / self._game.render_fps)

    def generate_observation(self):
        """Generates an observation."""
        # get main state
        if self._obs_type == 'image':
            main_state = self._game.get_screen_rgb()
        elif self._obs_type == 'player_state':
            main_state = self._game.get_player_state()
        else:  # catch all in case a new case is forgotten
            msg = 'Failed to generate observation for type {}'.format(self._obs_type)
            raise NotImplementedError(msg)

        # get aux state, if any
        aux_state = np.array([])
        if self._aux_use_n_targets:
            aux_state = np.append(aux_state, self._game.get_num_targets())

        if self._aux_use_time:
            aux_state = np.append(aux_state, self._game.current_time)

        # combine into output
        if self._obs_type == 'image':
            if aux_state.size > 0:
                return dict(img=main_state, aux=aux_state)
            else:
                return main_state
        else:
            if aux_state.size > 0:
                return np.concatenate((main_state, aux_state))
            else:
                return main_state

    def reset(self):
        """Resets the environment.

        Returns
        -------
        numpy array, dict
            observation
        """
        self._game.reset()
        self._game.step(np.zeros_like(self.action_space.low))

        # make sure the agent didn't die on first step
        while self._game.game_over:
            self._game.reset()
            self._game.step(np.zeros_like(self.action_space.low))
        return self.generate_observation()


class SimpleUAVHazards2d(SimpleUAV2d):
    """Simple 2d UAV environment with hazards.

    This follows the same underlying game logic as the :class:`.SimpleUAV2d`
    environment but has some hazards added to its default configuration.
    """

    def __init__(self, config_file='simple_uav_hazards_2d_config.yaml', **kwargs):
        """Initialize an object."""
        super().__init__(config_file=config_file, **kwargs)
