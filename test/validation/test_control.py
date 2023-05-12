import numpy as np
import numpy.testing as test

import gncpy.dynamics.basic as gdyn
import gncpy.control as gctrl


def test_lqr_lin_inf_hor():
    cur_time = 0
    dt = 0.1
    cur_state = np.array([0.5, 1, 0, 0]).reshape((-1, 1))
    end_state = np.array([2, 4, 0, 0]).reshape((-1, 1))
    time_horizon = float("inf")

    # Create dynamics object
    dynObj = gdyn.DoubleIntegrator()
    dynObj.control_model = lambda _t, *_args: np.array([[0, 0], [0, 0], [1, 0], [0, 1]])
    state_args = (dt,)

    # Setup LQR Object
    u_nom = np.zeros((2, 1))
    Q = 10 * np.eye(len(dynObj.state_names))
    R = 0.5 * np.eye(u_nom.size)
    lqr = gctrl.LQR(time_horizon=time_horizon)
    # need to set dt here so the controller can generate a state trajectory
    lqr.set_state_model(u_nom, dynObj=dynObj, dt=dt)
    lqr.set_cost_model(Q, R)

    u, cost, state_traj, ctrl_signal = lqr.calculate_control(
        cur_time,
        cur_state,
        end_state=end_state,
        end_state_tol=0.1,
        check_inds=[0, 1],
        state_args=state_args,
        provide_details=True,
    )

    test.assert_allclose(state_traj[-1, :], end_state.ravel(), rtol=0.025, atol=0.085)


def test_lqr_lin_finite_hor():
    cur_time = 0
    dt = 0.1
    cur_state = np.array([0.5, 1, 0, 0]).reshape((-1, 1))
    end_state = np.array([2, 4, 0, 0]).reshape((-1, 1))
    time_horizon = 3 * dt

    # Create dynamics object
    dynObj = gdyn.DoubleIntegrator()
    dynObj.control_model = lambda _t, *_args: np.array([[0, 0], [0, 0], [1, 0], [0, 1]])
    state_args = (dt,)
    inv_state_args = (-dt,)

    # Setup LQR Object
    u_nom = np.zeros((2, 1))
    Q = 200 * np.eye(len(dynObj.state_names))
    R = 0.01 * np.eye(u_nom.size)
    lqr = gctrl.LQR(time_horizon=time_horizon)
    # need to set dt here so the controller can generate a state trajectory
    lqr.set_state_model(u_nom, dynObj=dynObj, dt=dt)
    lqr.set_cost_model(Q, R)

    u, cost, state_traj, ctrl_signal = lqr.calculate_control(
        cur_time,
        cur_state,
        end_state=end_state,
        state_args=state_args,
        inv_state_args=inv_state_args,
        provide_details=True,
    )

    test.assert_allclose(state_traj[-1, :], end_state.ravel(), rtol=0.002, atol=0.008)


def test_lqr_nonlin_finite_hor():
    d2r = np.pi / 180

    cur_time = 0
    dt = 0.1
    cur_state = np.array([0.5, 0, 1, 0, 5 * d2r]).reshape((-1, 1))
    end_state = np.array([2, 4, 0, 0, 0]).reshape((-1, 1))
    time_horizon = 50 * dt

    # Create dynamics object
    dynObj = gdyn.CoordinatedTurnUnknown()

    # Setup LQR Object
    u_nom = np.zeros((3, 1))
    Q = np.diag([200, 2000, 1, 1, 2000])
    R = 0.01 * np.eye(u_nom.size)
    lqr = gctrl.LQR(time_horizon=time_horizon)
    # need to set dt here so the controller can generate a state trajectory
    lqr.set_state_model(u_nom, dynObj=dynObj, dt=dt, 
                        control_constraints=lambda t, u: np.array([
                                                                    [u.ravel()[0]],
                                                                    [u.ravel()[1]],
                                                                    [5 * d2r * np.min([np.max([u.ravel()[2], -1]), 1])]
                                                                  ])
    )
    lqr.set_cost_model(Q, R)

    u, cost, state_traj, ctrl_signal = lqr.calculate_control(
        cur_time, cur_state, end_state=end_state, provide_details=True, state_args=(dt,),
    )

    test.assert_allclose(state_traj[-1, :], end_state.ravel(), rtol=1e-3, atol=0.07)



def test_elqr_non_lin():
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
    dynObj = gdyn.IRobotCreate(wheel_separation=0.258, radius=0.335 / 2.0)

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

    # calculate control
    u, cost, state_trajectory, control_signal, fig, frame_list = elqr.calculate_control(
        tt,
        start_state,
        end_state,
        cost_args=cost_args,
        provide_details=True,
        show_animation=False,
    )

    np.testing.assert_allclose(state_trajectory[-1, :], end_state.ravel(), atol=0.15)


# %% Debugging
if __name__ == "__main__":
    # test_lqr_lin_inf_hor()
    # test_lqr_lin_finite_hor()
    # test_lqr_nonlin_finite_hor()
    test_elqr()
