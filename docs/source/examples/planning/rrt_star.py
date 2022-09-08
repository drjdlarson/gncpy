def lin_lqrrrtstar():
    import numpy as np

    import gncpy.dynamics.basic as gdyn
    import gncpy.control as gcontrol
    from gncpy.planning.rrt_star import LQRRRTStar

    SEED = 29

    start_time = 0
    dt = 0.01

    rng = np.random.default_rng(SEED)

    # define dynamics
    dynObj = gdyn.DoubleIntegrator()
    pos_inds = [0, 1]
    dynObj.control_model = lambda _t, *_args: np.array([[0, 0], [0, 0], [1, 0], [0, 1]])
    u_nom = np.zeros((2, 1))
    state_args = (dt,)

    # define starting and ending state
    end_state = np.array([0, 2.5, 0, 0]).reshape((4, 1))
    start_state = np.array([0, -2.5, 0, 0]).reshape((4, 1))

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
            [1.0 + np.sqrt(2) / 10, -1.5 - np.sqrt(2) / 10, 0.2],
        ]
    )

    # define enviornment bounds
    minxy = np.array([-2, -3])
    maxxy = np.array([2, 3])
    search_area = np.vstack((minxy, maxxy))

    # define the LQR planner
    lqr = gcontrol.LQR()
    Q = 50 * np.eye(len(dynObj.state_names))
    R = 0.6 * np.eye(u_nom.size)
    lqr.set_cost_model(Q, R)
    lqr.set_state_model(u_nom, dynObj=dynObj, dt=dt)

    # Initialize LQR-RRT* Planner
    lqrRRTStar = LQRRRTStar(rng=rng, max_iter=75)
    lqrRRTStar.set_environment(search_area=search_area, obstacles=obstacles)
    lqrRRTStar.set_control_model(lqr, pos_inds)

    # Run Planner
    trajectory, u_traj, fig, frame_list = lqrRRTStar.plan(
        start_time,
        start_state,
        end_state,
        state_args=state_args,
        disp=True,
        plt_inds=[0, 1],
        show_animation=True,
        save_animation=True,
        provide_details=True,
    )

    return frame_list


def nonlin_lqrrrtstar():
    import numpy as np

    import gncpy.dynamics.basic as gdyn
    import gncpy.control as gcontrol
    from gncpy.planning.rrt_star import LQRRRTStar

    d2r = np.pi / 180
    SEED = 29

    start_time = 0
    dt = 0.01
    time_horizon = 100 * dt

    rng = np.random.default_rng(SEED)

    # define dynamics
    dynObj = gdyn.CurvilinearMotion()
    pos_inds = [0, 1]

    u_nom = np.zeros((2, 1))

    # define starting and ending state
    ang = 90 * d2r
    vel = 0
    start_state = np.array(
        [0, -2.5, vel, ang]
    ).reshape((-1, 1))
    end_state = np.array(
        [0, 2.5, vel, ang]
    ).reshape((-1, 1))

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
            [1.0 + np.sqrt(2) / 10, -1.5 - np.sqrt(2) / 10, 0.2],
        ]
    )

    # define enviornment bounds
    minxy = np.array([-2, -3])
    maxxy = np.array([2, 3])
    search_area = np.vstack((minxy, maxxy))

    # define the LQR planner
    lqr = gcontrol.LQR(time_horizon=time_horizon)
    Q = np.diag([50, 50, 0.01, 0.1])
    R = np.diag([0.0001, 0.001])
    lqr.set_cost_model(Q, R)
    lqr.set_state_model(u_nom, dynObj=dynObj, dt=dt)

    # define the state sampler for the LQR-RRT* object
    def sampling_fun(rng, pos_inds, min_pos, max_pos):
        out = np.zeros((len(dynObj.state_names), 1))
        out[pos_inds] = rng.uniform(min_pos.ravel(), max_pos.ravel()).reshape((-1, 1))

        # sample velocity to make sure its not 0
        bnd = 1
        ii = 2
        out[ii] = rng.uniform(-bnd, bnd)
        while out[ii] < 1e-8:
            out[ii] = rng.uniform(-bnd, bnd)

        # sample turn angle to make sure its not 0
        bnd = 5 * d2r
        ii = 3
        out[ii] = rng.uniform(ang -bnd, ang + bnd)
        while out[ii] < 1e-8:
            out[ii] = rng.uniform(ang -bnd, ang + bnd)

        return out

    # Initialize LQR-RRT* Planner
    lqrRRTStar = LQRRRTStar(rng=rng, sampling_fun=sampling_fun)
    lqrRRTStar.set_environment(search_area=search_area, obstacles=obstacles)
    lqrRRTStar.set_control_model(lqr, pos_inds)

    # Run Planner
    trajectory, u_traj, fig, frame_list = lqrRRTStar.plan(
        start_time,
        start_state,
        end_state,
        disp=True,
        plt_inds=[0, 1],
        show_animation=True,
        save_animation=True,
        provide_details=True,
        ttl="Non-Linear LQR-RRT*",
    )

    return frame_list


def run():
    import os

    print("Generating RRT* examples")

    fout = os.path.join(os.path.dirname(__file__), "lqrrrtstar_linear.gif")
    if not os.path.isfile(fout):
        frame_list = lin_lqrrrtstar()
        frame_list[0].save(
            fout,
            save_all=True,
            append_images=frame_list[1:],
            duration=int(1 / 30 * 1e3),  # convert s to ms
            loop=0,
        )

    fout = os.path.join(os.path.dirname(__file__), "lqrrrtstar_nonlinear.gif")
    if not os.path.isfile(fout):
        frame_list = nonlin_lqrrrtstar()
        frame_list[0].save(
            fout,
            save_all=True,
            append_images=frame_list[1:],
            duration=int(1 / 30 * 1e3),  # convert s to ms
            loop=0,
        )


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    plt.close("all")
    run()
    plt.show()
