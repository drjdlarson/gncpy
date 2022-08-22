import numpy as np
import scipy.linalg

import gncpy.dynamics.basic as gdyn
from gncpy.planning.rrt_star import RRTStar


def main():
    dt = 0.01

    # define dynamics
    dynObj = gdyn.DoubleIntegrator()
    uSize = 2

    # define starting and ending state for control calculation
    xdes = np.array([0, 2.5, 0, 0]).reshape((4, 1))
    x0 = np.array([0, -2.5, 0, 0]).reshape((4, 1))

    # define some circular obstacles with center pos and radius (x, y, radius)
    obstacles = np.array(
        [
            [0, -1.35, 0.2],
            [1.0, -0.5, 0.2],
            [-0.95, -0.5, 0.2],
            [-0.2, 0.3, 0.2],
            [0.8, 0.7, 0.2],
            [1.1, 2.0, 0.2],
            [-1.2, 0.8, 0.2],
            [-1.1, 2.1, 0.2],
            [-0.1, 1.6, 0.2],
            [-1.1, -1.9, 0.2],
            [1.0 + np.sqrt(2), -1.5 - np.sqrt(2), 0.2],
        ]
    )

    # define Q and R weights for using standard cost function
    Q = 50 * np.eye(len(dynObj.state_names))
    R = 0.6 * np.eye(uSize)

    # define enviornment bounds for the robot
    minxy = np.array([-2.0, -3])
    maxxy = np.array([2, 3])
    randArea = np.concatenate((minxy, maxxy))

    # Initialize LQR-RRT* Planner
    param = RRTStar(x0, xdes, obstacles, randArea, Q, R, dynObj, dt)

    a = 3


if __name__ == "__main__":
    main()
