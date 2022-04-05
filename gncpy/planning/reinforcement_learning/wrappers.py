import gym
import numpy as np
import cv2

from collections import deque


class ResizeImage(gym.ObservationWrapper):
    def __init__(self, env=None, n_rows=84, n_cols=84, key='img'):
        super().__init__(env)
        print('Wrapping the env in a', repr(type(self).__name__), 'wrapper.')

        self._nrows = n_rows
        self._ncols = n_cols
        self._key = key
        if isinstance(self.observation_space, gym.spaces.Dict):
            spaces = {}
            for k, v in self.observation_space.spaces.items():
                if k == key:
                    new_space = gym.spaces.Box(low=0., high = 255.,
                                               shape=(n_rows, n_cols, 3),
                                               dtype=np.uint8)
                else:
                    new_space = v

                spaces[k] = new_space
            self.observation_space = gym.spaces.Dict(spaces)

        else:
            self.observation_space = gym.spaces.Box(low=0., high = 255.,
                                                    shape=(n_rows, n_cols, 3),
                                                    dtype=np.uint8)

    def observation(self, obs):
        if isinstance(obs, dict) or isinstance(obs, gym.spaces.Dict):
            resized_screen = cv2.resize(obs[self._key], (self._ncols, self._nrows),
                                        interpolation=cv2.INTER_AREA)
            obs[self._key] = resized_screen.astype(np.uint8)
            return obs
        else:
            resized_screen = cv2.resize(obs, (self._ncols, self._nrows),
                                        interpolation=cv2.INTER_AREA)
            return resized_screen.astype(np.uint8)


class GrayScaleObservation(gym.ObservationWrapper):
    r"""Convert the image observation from RGB to gray scale.

    Mostly the same as the open ai gym implementation except this allows
    for the wrapper to be applied to a single element inside a Dict observation.
    """

    def __init__(self, env, keep_dim=True, key='img'):
        super().__init__(env)
        print('Wrapping the env in a', repr(type(self).__name__), 'wrapper.')
        self.keep_dim = keep_dim
        self._key = key

        if isinstance(self.observation_space, gym.spaces.Dict):
            spaces = {}
            for k, v in self.observation_space.spaces.items():
                if k == key:
                    assert (len(v.shape) == 3 and v.shape[-1] == 3)

                    obs_shape = v.shape[:2]
                    if self.keep_dim:
                        new_space = gym.spaces.Box(low=0, high=255,
                                                   shape=(obs_shape[0],
                                                          obs_shape[1], 1),
                                                   dtype=np.uint8)
                    else:
                        new_space = gym.spaces.Box(low=0, high=255,
                                                   shape=obs_shape, dtype=np.uint8)
                else:
                    new_space = v

                spaces[k] = new_space
            self.observation_space = gym.spaces.Dict(spaces)

        else:
            assert (
                len(env.observation_space.shape) == 3
                and env.observation_space.shape[-1] == 3
            )

            obs_shape = self.observation_space.shape[:2]
            if self.keep_dim:
                self.observation_space = gym.spaces.Box(low=0, high=255,
                                                        shape=(obs_shape[0],
                                                               obs_shape[1], 1),
                                                        dtype=np.uint8)
            else:
                self.observation_space = gym.spaces.Box(low=0, high=255,
                                                        shape=obs_shape,
                                                        dtype=np.uint8)

    def observation(self, obs):
        if isinstance(obs, dict) or isinstance(obs, gym.spaces.Dict):
            obs[self._key] = cv2.cvtColor(obs[self._key], cv2.COLOR_RGB2GRAY)
            if self.keep_dim:
                obs[self._key] = np.expand_dims(obs[self._key], -1)
            return obs
        else:
            obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
            if self.keep_dim:
                obs = np.expand_dims(obs, -1)
            return obs


class BufferFames(gym.ObservationWrapper):
    """Buffer frames.

    Similar to the open ai gym implementation except this allows
    for the wrapper to be applied to a single element inside a Dict observation,
    and does not use a LazyFrame wrapper.
    """
    def __init__(self, env, num_stack, key='img'):
        super().__init__(env)
        print('Wrapping the env in a', repr(type(self).__name__), 'wrapper.')

        self._key = key
        self._num_stack = num_stack

        if isinstance(self.observation_space, gym.spaces.Dict):
            spaces = {}
            for k, v in self.observation_space.spaces.items():
                if k == key:
                    low = np.repeat(v.low[np.newaxis, ...], num_stack, axis=0)
                    high = np.repeat(v.high[np.newaxis, ...],
                                      num_stack, axis=0)
                    new_space = gym.spaces.Box(low=low, high=high,
                                                dtype=v.dtype)
                    self.buffer = 255 * np.ones_like(new_space.low)


                    self.buffer = np.repeat(255 * np.ones_like(new_space.low[np.newaxis, ...]),
                                            num_stack, axis=0).astype(np.uint8)

                else:
                    new_space = v

                spaces[k] = new_space
            self.observation_space = gym.spaces.Dict(spaces)
        else:

            low = np.repeat(self.observation_space.low[np.newaxis, ...], num_stack, axis=0)
            high = np.repeat(self.observation_space.high[np.newaxis, ...],
                             num_stack, axis=0)
            self.observation_space = gym.spaces.Box(low=low, high=high,
                                                    dtype=self.observation_space.dtype)
            self.buffer = 255 * np.ones_like(self.observation_space.low)

    def reset(self):
        if isinstance(self.observation_space, gym.spaces.Dict):
            self.buffer = 255 * np.ones_like(self.observation_space[self._key].low)
        else:
            self.buffer = np.zeros_like(self.observation_space.low)
        return self.observation(self.env.reset())

    def observation(self, obs):
        if isinstance(obs, dict) or isinstance(obs, gym.spaces.Dict):
            self.buffer[:-1] = self.buffer[1:]
            self.buffer[-1] = obs[self._key]
            obs[self._key] = self.buffer
            return obs
        else:
            self.buffer[:-1] = self.buffer[1:]
            self.buffer[-1] = obs
            return self.buffer


class StackFrames(gym.ObservationWrapper):
    """Buffer frames.

    Similar to the open ai gym implementation except this allows
    for the wrapper to be applied to a single element inside a Dict observation,
    and does not use a LazyFrame wrapper.
    """
    def __init__(self, env, num_stack, key='img'):
        super().__init__(env)
        print('Wrapping the env in a', repr(type(self).__name__), 'wrapper.')

        self._key = key
        if isinstance(self.observation_space, gym.spaces.Dict):
            assert self._key in self.observation_space.spaces.keys(), f"Key {self._key} not in observation space"

        self._num_stack = num_stack
        self.buffer = deque([], maxlen=num_stack)

    def reset(self):
        self.buffer.clear()
        return self.observation(self.env.reset())

    def observation(self, obs):
        if isinstance(obs, dict) or isinstance(obs, gym.spaces.Dict):
            self.buffer.appendleft(obs[self._key])
            obs[self._key] = (np.sum(self.buffer, axis=0) / len(self.buffer)).astype(np.uint8)
            return obs
        else:
            self.buffer.appendleft(obs)
            obs = (np.sum(self.buffer, axis=0) / len(self.buffer)).astype(np.uint8)
            return obs


class MaxFrames(gym.ObservationWrapper):
    """Buffer frames.

    Similar to the open ai gym implementation except this allows
    for the wrapper to be applied to a single element inside a Dict observation,
    and does not use a LazyFrame wrapper.
    """
    def __init__(self, env, num_stack, key='img'):
        super().__init__(env)
        print('Wrapping the env in a', repr(type(self).__name__), 'wrapper.')

        self._key = key
        if isinstance(self.observation_space, gym.spaces.Dict):
            assert self._key in self.observation_space.spaces.keys(), f"Key {self._key} not in observation space"

        self._num_stack = num_stack
        self.buffer = deque([], maxlen=num_stack)

    def reset(self):
        self.buffer.clear()
        return self.observation(self.env.reset())

    def observation(self, obs):
        if isinstance(obs, dict) or isinstance(obs, gym.spaces.Dict):
            self.buffer.appendleft(obs[self._key])
            obs[self._key] = np.max(self.buffer, axis=0).astype(np.uint8)
            return obs
        else:
            self.buffer.appendleft(obs)
            obs = np.max(self.buffer, axis=0).astype(np.uint8)
            return obs


class SkipFrames(gym.Wrapper):
    def __init__(self, env, frame_skip):
        super().__init__(env)
        print('Wrapping the env in a', repr(type(self).__name__), 'wrapper.')
        assert frame_skip > 0, "frame_skip must be > 0"

        self.frame_skip = frame_skip

    def step(self, action):
        R = 0.0

        for _ in range(self.frame_skip):
            obs, reward, done, info = self.env.step(action)
            R += reward

            if done:
                break

        return obs, R, done, info
