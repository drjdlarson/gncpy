import matplotlib.pyplot as plt
from matplotlib.patches import Circle


def basic():
    import numpy as np

    import gncpy.plotting as gplot
    import gncpy.control as gctrl
    from gncpy.dynamics.basic import IRobotCreate

    tt = 0  # starting time when calculating control
    dt = 1 / 6
    time_horizon = 150 * dt  # run for 150 timesteps

    # define starting and ending state for control calculation
    end_state = np.array([0, 2.5, np.pi]).reshape((3, 1))
    start_state = np.array([0, -2.5, np.pi]).reshape((3, 1))

    # define nominal control input
    u_nom = np.zeros((2, 1))

    # define dynamics
    # IRobot Create has a dt that can be set here or it can be set by the control
    # algorithm
    dynObj = IRobotCreate(wheel_separation=0.258, radius=0.335 / 2.0)

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
            [(10 + np.sqrt(2)) / 10, (-15 - np.sqrt(2)) / 10, 0.2],
        ]
    )

    # define enviornment bounds for the robot
    bottom_left = np.array([-2, -3])
    top_right = np.array([2, 3])

    # define Q and R weights for using standard cost function
    # Q = np.diag([50, 50, 0.4 * np.pi / 2])
    Q = 50 * np.eye(3)
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
            diff = state.ravel()[0:2] - obs[0:2]
            dist = np.sqrt(np.sum(diff * diff))
            # signed distance is negative if the robot is within the obstacle
            signed_dist = (dist - dynObj.radius) - obs[2]
            if signed_dist > 0:
                continue
            cost += _obs_factor * np.exp(-_scale_factor * signed_dist).item()

        # add cost for going out of bounds
        for ii, b in enumerate(_bottom_left):
            dist = (state[ii] - b) - dynObj.radius
            cost += _obs_factor * np.exp(-_scale_factor * dist).item()

        for ii, b in enumerate(_top_right):
            dist = (b - state[ii]) - dynObj.radius
            cost += _obs_factor * np.exp(-_scale_factor * dist).item()

        return cost

    # create control obect
    elqr = gctrl.ELQR(time_horizon=time_horizon)
    elqr.set_state_model(u_nom, dynObj=dynObj)
    elqr.dt = dt  # set here or within the dynamic object
    elqr.set_cost_model(Q=Q, R=R, non_quadratic_fun=non_quadratic_cost)

    # create figure with obstacles to plot animation on
    fig = plt.figure()
    fig.add_subplot(1, 1, 1)
    fig.axes[0].set_aspect("equal", adjustable="box")
    fig.axes[0].set_xlim((bottom_left[0], top_right[0]))
    fig.axes[0].set_ylim((bottom_left[1], top_right[1]))
    fig.axes[0].scatter(
        start_state[0], start_state[1], marker="o", color="g", zorder=1000
    )
    for obs in obstacles:
        c = Circle(obs[:2], radius=obs[2], color="k", zorder=1000)
        fig.axes[0].add_patch(c)
    plt_opts = gplot.init_plotting_opts(f_hndl=fig)
    gplot.set_title_label(fig, 0, plt_opts, ttl="ELQR")

    # calculate control
    u, cost, state_trajectory, control_signal, fig, frame_list = elqr.calculate_control(
        tt,
        start_state,
        end_state,
        cost_args=cost_args,
        provide_details=True,
        show_animation=True,
        save_animation=True,
        plt_inds=[0, 1],
        fig=fig,
    )

    return frame_list


def modify_quadratize():
    import numpy as np

    import gncpy.plotting as gplot
    import gncpy.control as gctrl
    from gncpy.dynamics.basic import IRobotCreate

    tt = 0  # starting time when calculating control
    dt = 1 / 6
    time_horizon = 150 * dt  # run for 150 timesteps

    # define starting and ending state for control calculation
    end_state = np.array([0, 2.5, np.pi]).reshape((3, 1))
    start_state = np.array([0, -2.5, np.pi]).reshape((3, 1))

    # define nominal control input
    u_nom = np.zeros((2, 1))

    # define dynamics
    # IRobot Create has a dt that can be set here or it can be set by the control
    # algorithm
    dynObj = IRobotCreate(wheel_separation=0.258, radius=0.335 / 2.0)

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
            [(10 + np.sqrt(2)) / 10, (-15 - np.sqrt(2)) / 10, 0.2],
        ]
    )

    # define enviornment bounds for the robot
    bottom_left = np.array([-2, -3])
    top_right = np.array([2, 3])

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
            diff = state.ravel()[0:2] - obs[0:2]
            dist = np.sqrt(np.sum(diff * diff))
            # signed distance is negative if the robot is within the obstacle
            signed_dist = (dist - dynObj.radius) - obs[2]
            if signed_dist > 0:
                continue
            cost += _obs_factor * np.exp(-_scale_factor * signed_dist).item()

        # add cost for going out of bounds
        for ii, b in enumerate(_bottom_left):
            dist = (state[ii] - b) - dynObj.radius
            cost += _obs_factor * np.exp(-_scale_factor * dist).item()

        for ii, b in enumerate(_top_right):
            dist = (b - state[ii]) - dynObj.radius
            cost += _obs_factor * np.exp(-_scale_factor * dist).item()

        return cost

    # define modifications for quadratizing the cost function
    def quad_modifier(itr, tt, P, Q, R, q, r):
        rot_cost = 0.4
        # only modify if within the first 2 iterations
        if itr < 2:
            Q[-1, -1] = rot_cost
            q[-1] = -rot_cost * np.pi / 2

        return P, Q, R, q, r

    # create control obect
    elqr = gctrl.ELQR(time_horizon=time_horizon)
    elqr.set_state_model(u_nom, dynObj=dynObj)
    elqr.dt = dt  # set here or within the dynamic object
    elqr.set_cost_model(
        Q=Q, R=R, non_quadratic_fun=non_quadratic_cost, quad_modifier=quad_modifier
    )

    fig = plt.figure()
    fig.add_subplot(1, 1, 1)
    fig.axes[0].set_aspect("equal", adjustable="box")
    fig.axes[0].set_xlim((bottom_left[0], top_right[0]))
    fig.axes[0].set_ylim((bottom_left[1], top_right[1]))
    fig.axes[0].scatter(
        start_state[0], start_state[1], marker="o", color="g", zorder=1000
    )
    for obs in obstacles:
        c = Circle(obs[:2], radius=obs[2], color="k", zorder=1000)
        fig.axes[0].add_patch(c)
    plt_opts = gplot.init_plotting_opts(f_hndl=fig)
    gplot.set_title_label(fig, 0, plt_opts, ttl="ELQR with Modified Quadratize Cost")

    # calculate control
    u, cost, state_trajectory, control_signal, fig, frame_list = elqr.calculate_control(
        tt,
        start_state,
        end_state,
        cost_args=cost_args,
        provide_details=True,
        show_animation=True,
        save_animation=True,
        plt_inds=[0, 1],
        fig=fig,
    )

    return frame_list


def linear():
    import numpy as np

    import gncpy.plotting as gplot
    import gncpy.control as gctrl
    from gncpy.dynamics.basic import DoubleIntegrator

    tt = 0  # starting time when calculating control
    dt = 1 / 6
    time_horizon = 20 * dt  # run for 150 timesteps

    # define starting and ending state for control calculation
    end_state = np.array([0, 2.5, 0, 0]).reshape((-1, 1))
    start_state = np.array([0, -2.5, 0, 0]).reshape((-1, 1))

    # define nominal control input
    u_nom = np.zeros((2, 1))

    # define dynamics
    dynObj = DoubleIntegrator()

    # set control model
    dynObj.control_model = lambda t, *args: np.array(
        [[0, 0], [0, 0], [1, 0], [0, 1]], dtype=float
    )

    # set a state constraint on the velocities
    dynObj.state_constraint = lambda t, x: np.vstack(
        (
            x[:2],
            np.max(
                (np.min((x[2:], 2 * np.ones((2, 1))), axis=0), -2 * np.ones((2, 1))),
                axis=0,
            ),
        )
    )

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
            [(10 + np.sqrt(2)) / 10, (-15 - np.sqrt(2)) / 10, 0.2],
        ]
    )

    # define enviornment bounds for the robot
    bottom_left = np.array([-2, -3])
    top_right = np.array([2, 3])

    # define Q and R weights for using standard cost function
    Q = 10 * np.eye(len(dynObj.state_names))
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
        radius = 0.335 / 2.0
        # cost for obstacles
        for obs in _obstacles:
            diff = state.ravel()[0:2] - obs[0:2]
            dist = np.sqrt(np.sum(diff * diff))
            # signed distance is negative if the robot is within the obstacle
            signed_dist = (dist - radius) - obs[2]
            if signed_dist > 0:
                continue
            cost += _obs_factor * np.exp(-_scale_factor * signed_dist).item()

        # add cost for going out of bounds
        for ii, b in enumerate(_bottom_left):
            dist = (state[ii] - b) - radius
            cost += _obs_factor * np.exp(-_scale_factor * dist).item()

        for ii, b in enumerate(_top_right):
            dist = (b - state[ii]) - radius
            cost += _obs_factor * np.exp(-_scale_factor * dist).item()

        return cost

    # create control obect
    elqr = gctrl.ELQR(time_horizon=time_horizon)
    elqr.set_state_model(u_nom, dynObj=dynObj)
    elqr.dt = dt  # set here or within the dynamic object
    elqr.set_cost_model(Q=Q, R=R, non_quadratic_fun=non_quadratic_cost)

    fig = plt.figure()
    fig.add_subplot(1, 1, 1)
    fig.axes[0].set_aspect("equal", adjustable="box")
    fig.axes[0].set_xlim((bottom_left[0], top_right[0]))
    fig.axes[0].set_ylim((bottom_left[1], top_right[1]))
    fig.axes[0].scatter(
        start_state[0], start_state[1], marker="o", color="g", zorder=1000
    )
    for obs in obstacles:
        c = Circle(obs[:2], radius=obs[2], color="k", zorder=1000)
        fig.axes[0].add_patch(c)
    plt_opts = gplot.init_plotting_opts(f_hndl=fig)
    gplot.set_title_label(fig, 0, plt_opts, ttl="ELQR with Linear Dynamics Cost")

    # calculate control
    u, cost, state_trajectory, control_signal, fig, frame_list = elqr.calculate_control(
        tt,
        start_state,
        end_state,
        state_args=(dt,),
        cost_args=cost_args,
        inv_state_args=(-dt,),
        provide_details=True,
        show_animation=True,
        save_animation=True,
        plt_inds=[0, 1],
        fig=fig,
    )

    return frame_list


def run():
    import os

    print("Generating ELQR examples")
    fps = 10
    duration = int(1 / fps * 1e3)

    # %% Basic case
    fout = os.path.join(os.path.dirname(__file__), "elqr_basic.gif")
    if not os.path.isfile(fout):
        frame_list = basic()

        frame_list[0].save(
            fout,
            save_all=True,
            append_images=frame_list[1:],
            duration=duration,  # convert s to ms
            loop=0,
        )

    # %% Modify quadratize case
    fout = os.path.join(os.path.dirname(__file__), "elqr_modify_quadratize.gif")
    if not os.path.isfile(fout):
        frame_list = modify_quadratize()

        frame_list[0].save(
            fout,
            save_all=True,
            append_images=frame_list[1:],
            duration=duration,  # convert s to ms
            loop=0,
        )

    # %% Linear case
    fout = os.path.join(os.path.dirname(__file__), "elqr_linear.gif")
    if not os.path.isfile(fout):
        frame_list = linear()

        frame_list[0].save(
            fout,
            save_all=True,
            append_images=frame_list[1:],
            duration=duration,  # convert s to ms
            loop=0,
        )


# %% Testing entry point
if __name__ == "__main__":
    plt.close("all")

    run()

    plt.show()
