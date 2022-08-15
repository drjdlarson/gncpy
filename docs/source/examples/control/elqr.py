def main():
    import numpy as np

    import gncpy.control as gcontrol
    from gncpy.dynamics.basic import IRobotCreate

    tt = 0  # starting time when calculating control
    dt = 1 / 6
    time_horizon = 150 * dt  # run for 150 timesteps

    # define starting and ending state for control calculation
    end_state = np.array([0, 25, np.pi]).reshape((3, 1))
    start_state = np.array([0, -25, np.pi]).reshape((3, 1))

    # define nominal control input
    u_nom = np.array([2.5, 2.5]).reshape((2, 1))

    # define dynamics
    # IRobot Create has a dt that can be set here or it can be set by the control
    # algorithm
    dynObj = IRobotCreate(wheel_separation=0.258, radius=3.35/2.0)

    # define some circular obstacles with center pos and radius (x, y, radius)
    obstacles = np.array([
        [0, -13.5, 2],
        [10, -5, 2],
        [-9.5, -5, 2],
        [-2, 3, 2],
        [8, 7, 2],
        [11, 20, 2],
        [-12, 8, 2],
        [-11, 21, 2],
        [-1, 16, 2],
        [-11, -19, 2],
        [10+np.sqrt(2), -15-np.sqrt(2), 2],
        ])

    # define enviornment bounds for the robot
    bottom_left = np.array([-20, -30])
    top_right = np.array([20, 30])

    # define Q and R weights for using standard cost function
    Q = 50 * np.eye(len(dynObj.state_names))
    R = 0.6 * np.eye(u_nom.size)

    # define non-quadratic term for cost function
    # has form: (tt, state, ctrl_input, end_state, is_initial, is_final, *args)
    obs_factor = 1
    scale_factor = 1
    cost_args = (obstacles, obs_factor, scale_factor, bottom_left, top_right)
    def non_quadratic_cost(
        tt,
        state,
        ctrl_input,
        end_state,
        is_initial,
        is_final,
        _obstacles,
        _obs_factor,
        _scale_factor,
        _bottom_left,
        _top_right,
    ):
        cost = 0
        # cost for obstacles
        for obs in _obstacles:
            diff = state[0:2] - obs[0:2]
            dist = np.sqrt(np.sum(diff * diff))
            # signed distance is negative if the robot is within the obstacle
            signed_dist = (dist - dynObj.radius) - obs[2]
            cost += _obs_factor * np.exp(-_scale_factor * signed_dist)

        # add cost for going out of bounds
        for ii, b in enumerate(_bottom_left):
            dist = (state[ii] - b) - dynObj.radius
            cost += _obs_factor * np.exp(-_scale_factor*dist)

        for ii, b in enumerate(_top_right):
            dist = (b - state[ii]) - dynObj.radius
            cost += _obs_factor * np.exp(-_scale_factor*dist)

        return cost

    # create control obect
    elqr = gcontrol.ELQR()
    elqr.set_state_model(u_nom, dynObj=dynObj)
    elqr.dt = dt  # set here or within the dynamic object
    elqr.set_cost_model(Q=Q, R=R, non_quadratic_fun=non_quadratic_cost)

    # calculate control
    u, cost, state_trajectory, control_signal = elqr.calculate_control(tt, start_state, end_state, cost_args=cost_args, provide_details=True)

    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(state_trajectory[:, 0], state_trajectory[:, 1])


if __name__ == "__main__":
    main()
