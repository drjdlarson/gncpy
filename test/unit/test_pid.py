import numpy as np
import numpy.testing as test

import gncpy.control as ctrl


def test_pid_proportional_integral_response():
    pid = ctrl.PID(kp=2.0, ki=1.0)

    u0 = pid.calculate_control(0.0, measurement=np.array([0.0]), reference=np.array([1.0]))
    u1 = pid.calculate_control(0.1, measurement=np.array([0.0]), reference=np.array([1.0]))

    test.assert_allclose(u0, np.array([2.0]))
    test.assert_allclose(u1, np.array([2.1]))


def test_pid_derivative_filter_reduces_spike():
    pid = ctrl.PID(kp=0.0, ki=0.0, kd=1.0, tau=0.1)

    pid.calculate_control(0.0, measurement=np.array([0.0]), reference=np.array([0.0]))
    u = pid.calculate_control(0.1, measurement=np.array([1.0]), reference=np.array([0.0]))

    raw_derivative = -10.0
    assert u.shape == (1,)
    assert u[0] < 0.0
    assert abs(u[0]) < abs(raw_derivative)


def test_pid_saturation_prevents_integral_windup():
    pid = ctrl.PID(kp=0.0, ki=1.0, kd=0.0, u_min=-0.5, u_max=0.5)

    u0 = pid.calculate_control(0.0, measurement=np.array([0.0]), reference=np.array([1.0]))
    u1 = pid.calculate_control(1.0, measurement=np.array([0.0]), reference=np.array([1.0]))
    u2 = pid.calculate_control(2.0, measurement=np.array([0.0]), reference=np.array([1.0]))

    test.assert_allclose(u0, np.array([0.0]))
    test.assert_allclose(u1, np.array([0.5]))
    test.assert_allclose(u2, np.array([0.5]))
    test.assert_allclose(pid.integral_state, np.array([0.5]))


def test_pid_vector_gains_and_reset():
    pid = ctrl.PID(kp=np.array([1.0, 2.0]), ki=0.0, kd=0.0)

    u = pid.calculate_control(
        0.0,
        measurement=np.array([1.0, -1.0]),
        reference=np.array([2.0, 1.0]),
    )
    test.assert_allclose(u, np.array([1.0, 4.0]))

    pid.reset(integral_state=np.array([3.0, -2.0]), deriv_state=np.array([0.5, -0.5]))
    test.assert_allclose(pid.integral_state, np.array([3.0, -2.0]))
    test.assert_allclose(pid.deriv_state, np.array([0.5, -0.5]))
