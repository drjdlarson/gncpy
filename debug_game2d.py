import numpy as np
from time import sleep
import gym

from gncpy.planning.reinforcement_learning.game import SimpleUAV2d as Game


if __name__ == '__main__':
    rng = np.random.default_rng(1)

    # while True:
    #     g = Game('config.yaml', 'human', int(1. / 0.1))

    #     while not g.game_over:
    #         action = rng.uniform(low=-1, high=1, size=2).reshape((2, 1))
    #         g.step(action)

    #     sleep(0.5)

    env = gym.make('SimpleUAV2d-v0', render_mode='human')

    while True:
        done = False
        obs = env.reset()
        tot_reward = 0
        while not done:
            action = rng.uniform(low=-1, high=1, size=2).reshape((2, 1))
            # action = np.array([[-1], [0]])
            obs, rewards, done, info = env.step(action)
            env.render()
            # tot_reward += rewards
