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
    minxy = np.array([-2.0, -3])
    maxxy = np.array([2, 3])
    search_area = np.concatenate((minxy, maxxy))

    # define the LQR planner
    lqr = gcontrol.LQR()
    Q = 50 * np.eye(len(dynObj.state_names))
    R = 0.6 * np.eye(u_nom.size)
    lqr.set_cost_model(Q, R)
    lqr.set_state_model(u_nom, dynObj=dynObj, dt=dt)

    # Initialize LQR-RRT* Planner
    lqrRRTStar = LQRRRTStar(rng=rng, max_iter=75, update_plot=5)
    lqrRRTStar.set_environment(search_area=search_area, obstacles=obstacles)
    lqrRRTStar.set_control_model(lqr)

    # Run Planner
    trajectory, u_traj, fig, frame_list = lqrRRTStar.plan(
        start_time,
        start_state,
        end_state,
        state_args=state_args,
        search_until_max_iter=True,
        disp=True,
        plt_inds=[0, 1],
        show_animation=True,
        save_animation=True,
        provide_details=True,
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


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    plt.close("all")
    run()
    plt.show()