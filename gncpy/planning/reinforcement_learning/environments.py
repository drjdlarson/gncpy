import os
import pathlib
import gym
from gym import spaces
import numpy as np
from abc import abstractmethod

import gncpy.planning.reinforcement_learning.game as rl_games


class BaseEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, game):
        super().__init__()

        self.action_space = None
        self.observation_space = None

        self._game = game

    def step(self, action):
        info = {}

        info.update(self._game.step(action))

        info.update(self.generate_step_info())
        obs = self.generate_observation()

        return obs, self._game.score, self._game.game_over, info

    @abstractmethod
    def generate_observation(self):
        raise NotImplementedError

    def generate_step_info(self):
        return {}

    @abstractmethod
    def render(self):
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        raise NotImplementedError


class SimpleUAV2d(BaseEnv):
    def __init__(self, config_file='simple_uav_2d_config.yaml',
                 render_mode='rgb_array', render_fps=None, obs_type='player_state',
                 aux_use_n_targets=False, aux_use_time=False):
        if os.pathsep in config_file:
            cf = config_file
        else:
            cf = os.path.join(os.getcwd(), config_file)
            if not os.path.isfile(cf):
                cf = os.path.join(pathlib.Path(__file__).parent.resolve(),
                                  config_file)
                if not os.path.isfile(cf):
                    raise RuntimeError('Failed to find config file {}'.format(config_file))

        game = rl_games.SimpleUAV2d(cf, render_mode, render_fps=render_fps)

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
        self._game.close()

    def render(self, mode='human'):
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
        # get main state
        if self._obs_type == 'image':
            main_state = self._game.get_screen_rgb()
        elif self._obs_type == 'player_state':
            main_state = self._game.get_player_state()
        else:  # catch all in case a new case is forgotten
            msg = 'Failed to generate observation for type {}'.format(self._obs_type)
            raise NotImplementedError(msg)

        #get aux state, if any
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
        self._game.reset()
        self._game.step(np.zeros_like(self.action_space.low))
        return self.generate_observation()
