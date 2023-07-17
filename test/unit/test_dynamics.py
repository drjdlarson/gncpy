import numpy as np
import numpy.testing as test

import gncpy.dynamics.basic as gdyn

import gncpy.control._control as gcont

DEBUG = False


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
    time = np.arange(0, t1 + dt, dt)
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


# TODO: change this to accommodate new control
def test_double_integrator_control():
    dt = 0.01
    t1 = 10
    time = np.arange(0, t1 + dt, dt)

    # Create dynamics object
    # dynObj = gdyn.DoubleIntegrator(control_model=gcont.StateControl())
    dynObj = gdyn.DoubleIntegrator()

    # Setup control model: 1 m/s^2 accel control in x, 0.5 m/s^2 control in y
    # dynObj.control_model = lambda _t, *_args: np.array(
    #     [[0, 0], [0, 0], [1 * dt, 0], [0, 0.5 * dt]]
    # )
    dynObj.control_model = gcont.StateControl()
    # dynObj.control_model(gcont.StateControl())
    cont_args = (2, 3)

    # simulate for some time
    state = np.zeros((time.size, len(dynObj.state_names)))
    state[0] = np.array([0, 0, 1, 0])

    for kk, tt in enumerate(time[:-1]):
        state[kk + 1] = dynObj.propagate_state(
            tt,
            state[kk].reshape((-1, 1)),
            state_args=(dt,),
            u=np.ones((2, 1)),
            ctrl_args=cont_args,
        ).flatten()

    # debug plots
    if DEBUG:
        fig = plt.figure()
        fig.add_subplot(2, 1, 1)
        fig.add_subplot(2, 1, 2)

        fig.axes[0].plot(time, state[:, 0])
        fig.axes[0].set_ylabel("x-pos (m)")
        fig.axes[0].grid(True)

        fig.axes[1].plot(time, state[:, 1])
        fig.axes[1].set_ylabel("y-pos (m)")
        fig.axes[1].set_xlabel("time (s)")
        fig.axes[1].grid(True)

        fig.suptitle("Double Integrator Pos w/ Control")

        fig = plt.figure()
        fig.add_subplot(2, 1, 1)
        fig.add_subplot(2, 1, 2)

        fig.axes[0].plot(time, state[:, 2])
        fig.axes[0].set_ylabel("x-vel (m/s)")
        fig.axes[0].grid(True)

        fig.axes[1].plot(time, state[:, 3])
        fig.axes[1].set_ylabel("y-vel (m/s)")
        fig.axes[1].set_xlabel("time (s)")
        fig.axes[1].grid(True)

        fig.suptitle("Double Integrator Vel w/ Control")

    # calculate expected state
    x_end = (
        0.5 * dynObj.control_model(0)[2, 0] / dt * t1**2
        + state[0, 2] * t1
        + state[0, 0]
    )
    y_end = (
        0.5 * dynObj.control_model(0)[3, 1] / dt * t1**2
        + state[0, 3] * t1
        + state[0, 1]
    )
    xvel_end = dynObj.control_model(0)[2, 0] / dt * t1 + state[0, 2]
    yvel_end = dynObj.control_model(0)[3, 1] / dt * t1 + state[0, 3]
    exp_state = np.array([x_end, y_end, xvel_end, yvel_end])

    # test expected against code
    test.assert_allclose(state[-1], exp_state, atol=0.05, rtol=0.001)


def test_clohessy_wiltshire2d_mat():
    mu = 3.986e14
    earth_rad = 6371e3
    alt = 300e3

    # mean_motion = np.sqrt(mu / (earth_rad + alt) ** 3)
    mean_motion = np.pi / 2

    dynObj = gdyn.ClohessyWiltshireOrbit2d(mean_motion=mean_motion)
    make_exp = lambda _dt, _n: np.array(
        [
            [
                4 - 3 * np.cos(_dt * _n),
                0,
                np.sin(_dt * _n) / _n,
                -(2 * np.cos(_dt * _n) - 2) / _n,
            ],
            [
                6 * np.sin(_dt * _n) - 6 * _dt * _n,
                1,
                (2 * np.cos(_dt * _n) - 2) / _n,
                (4 * np.sin(_dt * _n) - 3 * _dt * _n) / _n,
            ],
            [3 * _n * np.sin(_dt * _n), 0, np.cos(_dt * _n), 2 * np.sin(_dt * _n)],
            [
                6 * _n * (np.cos(_dt * _n) - 1),
                0,
                -2 * np.sin(_dt * _n),
                4 * np.cos(_dt * _n) - 3,
            ],
        ],
        dtype=float,
    )

    dt = 0.25
    F = dynObj.get_state_mat(0, dt)
    F_exp = make_exp(dt, mean_motion)
    test.assert_allclose(F, F_exp)

    dt = 0.5
    F = dynObj.get_state_mat(0, dt)
    F_exp = make_exp(dt, mean_motion)
    test.assert_allclose(F, F_exp)


def test_clohessy_wiltshire2d_prop():
    dt = 0.1
    t1 = 10
    time = np.arange(0, t1, dt)
    time_horizon = float("inf")

    mu = 3.986e14
    earth_rad = 6371e3
    alt = 300e3

    mean_motion = np.sqrt(mu / (earth_rad + alt) ** 3)
    # mean_motion = np.pi / 2

    # Create dynamics object
    dynObj = gdyn.ClohessyWiltshireOrbit2d(mean_motion=mean_motion)
    # dynObj.control_model = lambda _t, *_args: np.array([[0, 0], [0, 0], [1, 0], [0, 0.5]])
    state = np.zeros((time.size, len(dynObj.state_names)))
    state[0] = np.array([0, 0, 1, 0])

    for kk, tt in enumerate(time[:-1]):
        state[kk + 1] = dynObj.propagate_state(
            tt, state[kk].reshape((-1, 1)), state_args=(dt,)
        ).flatten()

    test.assert_allclose(
        state[-1],
        np.array([9.89978287, -0.11356588, 0.9999342, -0.02294235], dtype=float),
    )


# TODO: change this to accommodate new control
def test_clohessy_wiltshire2d_control():
    dt = 0.01
    t1 = 10
    time = np.arange(0, t1 + dt, dt)

    mu = 3.986e14
    earth_rad = 6371e3
    alt = 300e3

    mean_motion = np.sqrt(mu / (earth_rad + alt) ** 3)

    # Create dynamics object
    dynObj = gdyn.ClohessyWiltshireOrbit2d(mean_motion=mean_motion)

    # Setup control model: 1 m/s^2 accel control in x, 0.5 m/s^2 control in y
    dynObj.control_model = lambda _t, *_args: np.array(
        [[0, 0], [0, 0], [1 * dt, 0], [0, 0.5 * dt]]
    )

    # simulate for some time
    state = np.zeros((time.size, len(dynObj.state_names)))
    state[0] = np.array([0, 0, 1, 0])

    for kk, tt in enumerate(time[:-1]):
        state[kk + 1] = dynObj.propagate_state(
            tt, state[kk].reshape((-1, 1)), state_args=(dt,), u=np.ones((2, 1))
        ).flatten()

    # debug plots
    if DEBUG:
        fig = plt.figure()
        fig.add_subplot(2, 1, 1)
        fig.add_subplot(2, 1, 2)

        fig.axes[0].plot(time, state[:, 0])
        fig.axes[0].set_ylabel("x-pos (m)")
        fig.axes[0].grid(True)

        fig.axes[1].plot(time, state[:, 1])
        fig.axes[1].set_ylabel("y-pos (m)")
        fig.axes[1].set_xlabel("time (s)")
        fig.axes[1].grid(True)

        fig.suptitle("Clohessy Wiltshire Pos w/ Control")

        fig = plt.figure()
        fig.add_subplot(2, 1, 1)
        fig.add_subplot(2, 1, 2)

        fig.axes[0].plot(time, state[:, 2])
        fig.axes[0].set_ylabel("x-vel (m/s)")
        fig.axes[0].grid(True)

        fig.axes[1].plot(time, state[:, 3])
        fig.axes[1].set_ylabel("y-vel (m/s)")
        fig.axes[1].set_xlabel("time (s)")
        fig.axes[1].grid(True)

        fig.suptitle("Clohessy Wiltshire Vel w/ Control")

    # calculate expected state

    exp_state = np.array([60.142049, 24.47235, 11.057587, 4.860623])

    # test expected against code
    test.assert_allclose(state[-1], exp_state, atol=0.05, rtol=0.001)


def test_clohessy_wiltshire_mat():
    mu = 3.986e14
    earth_rad = 6371e3
    alt = 300e3

    mean_motion = np.sqrt(mu / (earth_rad + alt) ** 3)

    dynObj = gdyn.ClohessyWiltshireOrbit(mean_motion=mean_motion)
    make_exp = lambda _dt, _n: np.array(
        [
            [
                4 - 3 * np.cos(_dt * _n),
                0,
                0,
                np.sin(_dt * _n) / _n,
                -(2 * np.cos(_dt * _n) - 2) / _n,
                0,
            ],
            [
                6 * np.sin(_dt * _n) - 6 * _dt * _n,
                1,
                0,
                (2 * np.cos(_dt * _n) - 2) / _n,
                (4 * np.sin(_dt * _n) - 3 * _dt * _n) / _n,
                0,
            ],
            [0, 0, np.cos(_n * _dt), 0, 0, 1 / _n * np.sin(_n * _dt)],
            [
                3 * _n * np.sin(_dt * _n),
                0,
                0,
                np.cos(_dt * _n),
                2 * np.sin(_dt * _n),
                0,
            ],
            [
                6 * _n * (np.cos(_dt * _n) - 1),
                0,
                0,
                -2 * np.sin(_dt * _n),
                4 * np.cos(_dt * _n) - 3,
                0,
            ],
            [0, 0, -_n * np.sin(_n * _dt), 0, 0, np.cos(_n * _dt)],
        ],
        dtype=float,
    )

    dt = 0.25
    F = dynObj.get_state_mat(0, dt)
    F_exp = make_exp(dt, mean_motion)
    test.assert_allclose(F, F_exp)

    dt = 0.5
    F = dynObj.get_state_mat(0, dt)
    F_exp = make_exp(dt, mean_motion)
    test.assert_allclose(F, F_exp)


def test_clohessy_wiltshire_prop():
    dt = 0.1
    t1 = 10
    time = np.arange(0, t1 + dt, dt)
    time_horizon = float("inf")

    mu = 3.986e14
    earth_rad = 6371e3
    alt = 300e3

    mean_motion = np.sqrt(mu / (earth_rad + alt) ** 3)
    # mean_motion = np.pi / 2

    # Create dynamics object
    dynObj = gdyn.ClohessyWiltshireOrbit(mean_motion=mean_motion)
    # dynObj.control_model = lambda _t, *_args: np.array([[0, 0], [0, 0], [1, 0], [0, 0.5]])
    state = np.zeros((time.size, len(dynObj.state_names)))
    state[0] = np.array([0, 0, 0, 1, 0, 1])

    for kk, tt in enumerate(time[:-1]):
        state[kk + 1] = dynObj.propagate_state(
            tt, state[kk].reshape((-1, 1)), state_args=(dt,)
        ).flatten()

    x_end = state[0, 2] * t1

    test.assert_allclose(
        state[-1],
        np.array(
            [09.99977623, -0.1158717, 9.99977623, 0.99993287, -0.02317408, 0.99993287],
            dtype=float,
        ),
    )


# TODO: change this to accommodate new control
def test_clohessy_wiltshire_control():
    dt = 0.01
    t1 = 10
    time = np.arange(0, t1 + dt, dt)

    mu = 3.986e14
    earth_rad = 6371e3
    alt = 300e3

    mean_motion = np.sqrt(mu / (earth_rad + alt) ** 3)

    # Create dynamics object
    dynObj = gdyn.ClohessyWiltshireOrbit(mean_motion=mean_motion)

    # Setup control model: 1 m/s^2 accel control in x, 0.5 m/s^2 control in y
    dynObj.control_model = lambda _t, *_args: np.array(
        [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [1 * dt, 0, 0],
            [0, 0.5 * dt, 0],
            [0, 0, 1 * dt],
        ]
    )

    # simulate for some time
    state = np.zeros((time.size, len(dynObj.state_names)))
    state[0] = np.array([0, 0, 0, 1, 0, 1])

    for kk, tt in enumerate(time[:-1]):
        state[kk + 1] = dynObj.propagate_state(
            tt, state[kk].reshape((-1, 1)), state_args=(dt,), u=np.ones((3, 1))
        ).flatten()

    # debug plots
    # if DEBUG:
    # fig = plt.figure()
    # fig.add_subplot(3, 1, 1)
    # fig.add_subplot(3, 1, 2)
    # fig.add_subplot(3, 1, 3)

    # fig.axes[0].plot(time, state[:, 0])
    # fig.axes[0].set_ylabel("x-pos (m)")
    # fig.axes[0].grid(True)

    # fig.axes[1].plot(time, state[:, 1])
    # fig.axes[1].set_ylabel("y-pos (m)")
    # fig.axes[1].set_xlabel("time (s)")
    # fig.axes[1].grid(True)

    # fig.suptitle("Clohessy Wiltshire Pos w/ Control")

    # fig = plt.figure()
    # fig.add_subplot(2, 1, 1)
    # fig.add_subplot(2, 1, 2)

    # fig.axes[0].plot(time, state[:, 2])
    # fig.axes[0].set_ylabel("x-vel (m/s)")
    # fig.axes[0].grid(True)

    # fig.axes[1].plot(time, state[:, 3])
    # fig.axes[1].set_ylabel("y-vel (m/s)")
    # fig.axes[1].set_xlabel("time (s)")
    # fig.axes[1].grid(True)

    # fig.suptitle("Clohessy Wiltshire Vel w/ Control")

    # calculate expected state
    exp_state = np.array(
        [60.142049, 24.47235, 59.949218, 11.057587, 4.860623, 10.999709]
    )

    # test expected against code
    test.assert_allclose(state[-1], exp_state, atol=0.05, rtol=0.001)


if __name__ == "__main__":
    DEBUG = True
    if DEBUG:
        import matplotlib.pyplot as plt

        plt.close("all")

    test_double_integrator_mat()
    test_double_integrator_prop()
    test_double_integrator_control()

    test_clohessy_wiltshire2d_mat()
    test_clohessy_wiltshire2d_prop()
    # test_clohessy_wiltshire2d_control()

    test_clohessy_wiltshire_mat()
    test_clohessy_wiltshire_prop()
    # test_clohessy_wiltshire_control()

    if DEBUG:
        plt.show()
