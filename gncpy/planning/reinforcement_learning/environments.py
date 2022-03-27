import os
import pathlib
import gym
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
        self._game.step(action)

        info = self.generate_step_info()
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
                 render_mode='rgb_array', render_fps=None):
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

        self.action_space = gym.spaces.Box(low=-1., high=1., shape=(2,))
        self.observation_space = gym.spaces.Box(low=0, high=255,
                                                shape=(*self._game.get_image_size(), 3),
                                                dtype=np.uint8)

        self._render_mode = render_mode

        self._fig = None

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
                plt.rcParams['toolbar'] = 'None'

                self._fig = plt.figure(figsize=(px2in * img.shape[0],
                                                px2in * img.shape[1]))
                self._fig.add_axes([0, 0, 1, 1], frame_on=False, rasterized=True)

            self._fig.axes[0].clear()
            self._fig.axes[0].imshow(img)
            plt.pause(1 / self._game.render_fps)

    def generate_observation(self):
        return self._game.get_screen_rgb()

    def reset(self):
        self._game.reset()
        return self.generate_observation()
