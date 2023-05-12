import numpy as np
import numpy.testing as test

import gncpy.dynamics.basic as gdyn


def test_double_integrator_mat():
    dynObj = gdyn.DoubleIntegrator()
    make_exp = lambda _dt: np.array(
        [[1, 0, _dt, 0], [0, 1, 0, _dt], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float
    )

    dt = 0.25
    F = dynObj.get_state_mat(0, dt)
    F_exp = make_exp(dt)
    test.assert_allclose(F, F_exp)

    dt = 0.5
    F = dynObj.get_state_mat(0, dt)
    F_exp = make_exp(dt)
    test.assert_allclose(F, F_exp)


def test_double_integrator_prop():
    dt = 0.01
    t1 = 10
    time = np.arange(0, t1, dt)
    cur_state = np.array([0.5, 1, 0, 0]).reshape((-1, 1))
    end_state = np.array([2, 4, 0, 0]).reshape((-1, 1))
    time_horizon = float("inf")

    # Create dynamics object
    dynObj = gdyn.DoubleIntegrator()
    # dynObj.control_model = lambda _t, *_args: np.array([[0, 0], [0, 0], [1, 0], [0, 0.5]])
    state = np.zeros((time.size, len(dynObj.state_names)))
    state[0] = np.array([0, 0, 1, 0])

    for kk, tt in enumerate(time[:-1]):
        state[kk + 1] = dynObj.propagate_state(
            tt, state[kk].reshape((-1, 1)), state_args=(dt,)
        ).flatten()

    x_end = state[0, 2] * t1
    test.assert_allclose(state[-1], np.array([x_end, 0, 1, 0], dtype=float))


def test_double_integrator_control():
    dt = 0.01
    t1 = 10
    time = np.arange(0, t1, dt)
    cur_state = np.array([0.5, 1, 0, 0]).reshape((-1, 1))
    end_state = np.array([2, 4, 0, 0]).reshape((-1, 1))
    time_horizon = float("inf")

    # Create dynamics object
    dynObj = gdyn.DoubleIntegrator()
    dynObj.control_model = lambda _t, *_args: np.array(
        [[0, 0], [0, 0], [1, 0], [0, 0.5]]
    )
    state = np.zeros((time.size, len(dynObj.state_names)))
    state[0] = np.array([0, 0, 1, 0])

    for kk, tt in enumerate(time[:-1]):
        state[kk + 1] = dynObj.propagate_state(
            tt, state[kk].reshape((-1, 1)), state_args=(dt,), u=np.ones((2, 1))
        ).flatten()

    x_end = 0.5 * dynObj.control_model(0)[2, 0] * t1
    y_end = 0.5 * dynObj.control_model(0)[3, 1] * t1
    xvel_end = dynObj.control_model(0)[2, 0] * t1
    yvel_end = dynObj.control_model(0)[3, 1] * t1
    test.assert_allclose(state[-1], np.array([x_end, y_end, xvel_end, yvel_end]))


if __name__ == "__main__":
    test_double_integrator_mat()
    test_double_integrator_prop()
    test_double_integrator_control()
