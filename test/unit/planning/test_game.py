import numpy as np

import gncpy.planning.reinforcement_learning.game as rl_games


def test_SimpleUAV2d():
    game = rl_games.SimpleUAV2d("rgb_array")
    game.step(np.array([0.25, 0]))


if __name__ == "__main__":
    test_SimpleUAV2d()
