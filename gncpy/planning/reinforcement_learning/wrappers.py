import gym
import numpy as np
import cv2


# class Preprocess2dGame(gym.Wrapper):
#     r"""Atari 2600 preprocessings.
#     This class follows the guidelines in
#     Machado et al. (2018), "Revisiting the Arcade Learning Environment:
#     Evaluation Protocols and Open Problems for General Agents".
#     Specifically:
#     * NoopReset: obtain initial state by taking random number of no-ops on reset.
#     * Frame skipping: 4 by default
#     * Max-pooling: most recent two observations
#     * Termination signal when a life is lost: turned off by default. Not recommended by Machado et al. (2018).
#     * Resize to a square image: 84x84 by default
#     * Grayscale observation: optional
#     * Scale observation: optional
#     Args:
#         env (Env): environment
#         noop_max (int): max number of no-ops
#         frame_skip (int): the frequency at which the agent experiences the game.
#         screen_size (int): resize Atari frame
#         terminal_on_life_loss (bool): if True, then step() returns done=True whenever a
#             life is lost.
#         grayscale_obs (bool): if True, then gray scale observation is returned, otherwise, RGB observation
#             is returned.
#         grayscale_newaxis (bool): if True and grayscale_obs=True, then a channel axis is added to
#             grayscale observations to make them 3-dimensional.
#         scale_obs (bool): if True, then observation normalized in range [0,1] is returned. It also limits memory
#             optimization benefits of FrameStack Wrapper.
#     """

#     def __init__(
#         self,
#         env: gym.Env,
#         frame_skip: int = 4,
#         screen_size: int = 84,
#         grayscale_obs: bool = True,
#         grayscale_newaxis: bool = False,
#         scale_obs: bool = False,
#     ):
#         super().__init__(env)
#         assert frame_skip > 0
#         assert screen_size > 0


#         self.frame_skip = frame_skip
#         self.screen_size = screen_size
#         self.grayscale_obs = grayscale_obs
#         self.grayscale_newaxis = grayscale_newaxis
#         self.scale_obs = scale_obs

#         # buffer of most recent two observations for max pooling
#         if grayscale_obs:
#             self.obs_buffer = [
#                 np.empty(env.observation_space.shape[:2], dtype=np.uint8),
#                 np.empty(env.observation_space.shape[:2], dtype=np.uint8),
#             ]
#         else:
#             self.obs_buffer = [
#                 np.empty(env.observation_space.shape, dtype=np.uint8),
#                 np.empty(env.observation_space.shape, dtype=np.uint8),
#             ]

#         self.game_over = False

#         _low, _high, _obs_dtype = (
#             (0, 255, np.uint8) if not scale_obs else (0, 1, np.float32)
#         )
#         _shape = (screen_size, screen_size, 1 if grayscale_obs else 3)
#         if grayscale_obs and not grayscale_newaxis:
#             _shape = _shape[:-1]  # Remove channel axis
#         self.observation_space = gym.spaces.Box(low=_low, high=_high,
#                                                 shape=_shape, dtype=_obs_dtype)

#     def step(self, action):
#         R = 0.0

#         for t in range(self.frame_skip):
#             _, reward, done, info = self.env.step(action)
#             R += reward
#             self.game_over = done

#             if done:
#                 break
#             if t == self.frame_skip - 2:
#                 if self.grayscale_obs:
#                     self.ale.getScreenGrayscale(self.obs_buffer[1])
#                 else:
#                     self.ale.getScreenRGB(self.obs_buffer[1])
#             elif t == self.frame_skip - 1:
#                 if self.grayscale_obs:
#                     self.ale.getScreenGrayscale(self.obs_buffer[0])
#                 else:
#                     self.ale.getScreenRGB(self.obs_buffer[0])

#         return self._get_obs(), R, done, info

#     def reset(self):
#         if self.grayscale_obs:
#             self.ale.getScreenGrayscale(self.obs_buffer[0])
#         else:
#             self.ale.getScreenRGB(self.obs_buffer[0])
#         self.obs_buffer[1].fill(0)

#         return self._get_obs()

#     def _get_obs(self):
#         if self.frame_skip > 1:  # more efficient in-place pooling
#             np.maximum(self.obs_buffer[0], self.obs_buffer[1], out=self.obs_buffer[0])
#         obs = cv2.resize(
#             self.obs_buffer[0],
#             (self.screen_size, self.screen_size),
#             interpolation=cv2.INTER_AREA,
#         )

#         if self.scale_obs:
#             obs = np.asarray(obs, dtype=np.float32) / 255.0
#         else:
#             obs = np.asarray(obs, dtype=np.uint8)

#         if self.grayscale_obs and self.grayscale_newaxis:
#             obs = np.expand_dims(obs, axis=-1)  # Add a channel axis
#         return obs


class ResizeImage(gym.ObservationWrapper):
    def __init__(self, env=None, n_rows=84, n_cols=84):
        super().__init__(env)
        print('Wrapping the env in a', repr(type(self).__name__), 'wrapper.')

        self._nrows = n_rows
        self._ncols = n_cols
        self.observation_space = gym.spaces.Box(low=0., high = 255.,
                                                shape=(n_rows, n_cols, 3),
                                                dtype=np.uint8)

    def observation(self, obs):
        resized_screen = cv2.resize(obs, (self._ncols, self._nrows),
                                    interpolation=cv2.INTER_AREA)
        return resized_screen.astype(np.uint8)


# class ResizeImageToGrayscale(gym.ObservationWrapper):
#     def __init__(self, env=None, n_rows=84, n_cols=84):
#         super().__init__(env)
#         print('Wrapping the env in a', repr(type(self).__name__), 'wrapper.')

#         self._nrows = n_rows
#         self._ncols = n_cols
#         self.observation_space = gym.spaces.Box(low=0., high = 255.,
#                                                 shape=(n_rows, n_cols, 1),
#                                                 dtype=np.uint8)

#     def observation(self, obs):
#         img = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
#         resized_screen = cv2.resize(img, (self._nrows, self._ncols),
#                                     interpolation=cv2.INTER_AREA)
#         return resized_screen.reshape((self._nrows, self._ncols, 1)).astype(np.uint8)


# class BufferObservationWrapper(gym.ObservationWrapper):
#     def __init__(self, env, n_steps):
#         super().__init__(env)
#         print('Wrapping the env in a', repr(type(self).__name__), 'wrapper.')

#         self.dtype = env.observation_space.dtype
#         old_space = env.observation_space
#         self.observation_space = gym.spaces.Box(old_space.low.repeat(n_steps,
#                                                                      axis=0),
#                                                 old_space.high.repeat(n_steps,
#                                                                       axis=0),
#                                                 dtype=self.dtype)
#         self.buffer = np.zeros_like(self.observation_space.low, dtype=self.dtype)

#     def reset(self):
#         self.buffer = np.zeros_like(self.observation_space.low, dtype=self.dtype)
#         return self.observation(self.env.reset())

#     def observation(self, obs):
#         self.buffer[:-1] = self.buffer[1:]
#         self.buffer[-1] = obs
#         return self.buffer


# class ImageToPyTorch(gym.ObservationWrapper):
#     def __init__(self, env):
#         super().__init__(env)
#         print('Wrapping the env in a', repr(type(self).__name__), 'wrapper.')

#         old_shape = env.observation_space.shape
#         self.observation_space = gym.spaces.Box(low=0, high=255,
#                                                 shape=(old_shape[-1], old_shape[0],
#                                                        old_shape[1]),
#                                                 dtype=np.uint8)

#     def observation(self, obs):
#         return np.moveaxis(obs, 2, 0)
