import numpy as np
from time import sleep

from gncpy.planning.reinforcement_learning.game import Game2d


if __name__ == '__main__':
    rng = np.random.default_rng(1)

    while True:
        g = Game2d('config.yaml', 'human')

        while not g.game_over:
            action = rng.uniform(low=-1, high=1, size=2).reshape((2, 1))
            g.step(action)

        sleep(0.5)
