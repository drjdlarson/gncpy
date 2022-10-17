import numpy as np

import gncpy.dynamics.basic as gdyn
import gncpy.control as gcontrol
import gncpy.planning.rrt_star as gplan


SEED = 29
DEBUG_PLOTS = False


def test_lin_lqrrrtstar():
    global SEED, DEBUG_PLOTS

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

    # define the state sampler for the LQR-RRT* object
    def sampling_fun(rng, pos_inds, min_pos, max_pos):
        out = np.zeros((len(dynObj.state_names), 1))
        out[pos_inds] = rng.uniform(min_pos.ravel(), max_pos.ravel()).reshape((-1, 1))

        return out

    # Initialize LQR-RRT* Planner
    lqrRRTStar = gplan.LQRRRTStar(rng=rng, sampling_fun=sampling_fun)
    lqrRRTStar.set_environment(search_area=search_area, obstacles=obstacles)
    lqrRRTStar.set_control_model(lqr, pos_inds)

    # Run Planner
    trajectory, cost, u_traj, fig, frame_list = lqrRRTStar.plan(
        start_time,
        start_state,
        end_state,
        use_convergence=True,
        state_args=state_args,
        disp=True,
        plt_inds=[0, 1],
        show_animation=DEBUG_PLOTS,
        save_animation=False,
        provide_details=True,
    )

    np.testing.assert_allclose(trajectory[-1, :], end_state.ravel())


def test_elqrrrtstar():
    global SEED

    d2r = np.pi / 180

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
    start_state = np.array([0, -2.5, vel, ang]).reshape((-1, 1))
    end_state = np.array([0, 2.5, vel, ang]).reshape((-1, 1))

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

    # define the LQR controller
    controller = gcontrol.ELQR(time_horizon=time_horizon, max_iters=300, tol=1e-2)
    controller_args = dict()
    Q = np.diag([50, 50, 4, 2])
    R = np.diag([0.05, 0.001])

    obs_factor = 1
    scale_factor = 1
    cost_args = (obstacles, obs_factor, scale_factor, minxy, maxxy)

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
        radius = 0
        cost = 0
        # cost for obstacles
        for obs in _obstacles:
            diff = state.ravel()[pos_inds] - obs[0:2]
            dist = np.sqrt(np.sum(diff * diff))
            # signed distance is negative if the robot is within the obstacle
            signed_dist = (dist - radius) - obs[2]
            if signed_dist > 0:
                continue
            cost += _obs_factor * np.exp(-_scale_factor * signed_dist).item()
        # add cost for going out of bounds
        for ii, b in enumerate(_bottom_left):
            dist = (state[pos_inds[ii]] - b) - radius
            cost += _obs_factor * np.exp(-_scale_factor * dist).item()
        for ii, b in enumerate(_top_right):
            dist = (b - state[pos_inds[ii]]) - radius
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

    controller.set_cost_model(
        Q=Q, R=R, non_quadratic_fun=non_quadratic_cost, quad_modifier=quad_modifier
    )
    controller.set_state_model(u_nom, dynObj=dynObj, dt=dt)

    # define the state sampler for the LQR-RRT* object
    def sampling_fun(rng, pos_inds, min_pos, max_pos):
        out = np.zeros((len(dynObj.state_names), 1))
        out[pos_inds] = rng.uniform(min_pos.ravel(), max_pos.ravel()).reshape((-1, 1))

        # sample velocity to make sure its not 0
        bnd = 1
        ii = 2
        out[ii] = rng.uniform(0.1, bnd)
        while out[ii] < 1e-8:
            out[ii] = rng.uniform(-bnd, bnd)
        # # sample turn angle to make sure its not 0
        out[3] = ang

        return out

    # Initialize ELQR-RRT* Planner
    elqrRRTStar = gplan.ExtendedLQRRRTStar(rng=rng, sampling_fun=sampling_fun)
    elqrRRTStar.set_environment(search_area=search_area, obstacles=obstacles)
    elqrRRTStar.set_control_model(controller, pos_inds, controller_args=controller_args)

    # Run Planner
    trajectory, cost, u_traj, fig, frame_list = elqrRRTStar.plan(
        start_time,
        start_state,
        end_state,
        cost_args=cost_args,
        disp=True,
        plt_inds=pos_inds,
        show_animation=DEBUG_PLOTS,
        show_planner=DEBUG_PLOTS,
        save_animation=False,
        provide_details=True,
        ttl="ELQR-RRT*",
    )

    # Check output
    np.testing.assert_allclose(trajectory[-1, :], end_state.ravel())


# %% Debugging entry point
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    DEBUG_PLOTS = True

    plt.close("all")

    # test_lin_lqrrrtstar()
    test_elqrrrtstar()

    plt.show()
