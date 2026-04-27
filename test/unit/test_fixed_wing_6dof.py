import numpy as np
import numpy.testing as test

from gncpy.dynamics.aircraft import FixedWing6DOF


def _make_state():
    state = np.zeros((13, 1))
    state[3] = 15.0
    state[6] = 1.0
    return state


def test_fixed_wing_defaults_are_finite():
    dyn = FixedWing6DOF(dt=0.01)
    state = _make_state()
    u = np.zeros((4, 1))

    x_dot = dyn.state_derivative(0.0, state, u)
    next_state = dyn.propagate_state(0.0, state, u=u)

    assert x_dot.shape == (13, 1)
    assert next_state.shape == (13, 1)
    assert np.all(np.isfinite(x_dot))
    assert np.all(np.isfinite(next_state))
    test.assert_allclose(np.linalg.norm(next_state[6:10]), 1.0)


def test_fixed_wing_partial_coefficients_linearize():
    params = {
        "mass": {
            "mass_kg": 1.959,
            "inertia_kgm2": [
                [0.07151, 0.0, 0.014],
                [0.0, 0.08636, 0.0],
                [-0.014, 0.0, 0.15364],
            ],
        },
        "geometry": {
            "wing_area_m2": 0.3097,
            "wing_span_m": 1.27,
            "mean_chord_m": 0.25,
        },
        "aero": {
            "CL0": 0.1086,
            "CL_alpha": 4.58,
            "CD_min": 0.0434,
            "CL_minD": 0.23,
            "oswald_eff": 0.75,
            "Cm0": -0.0278,
            "Cm_alpha": -0.723,
            "Cm_delta_e": -0.8488,
        },
    }
    dyn = FixedWing6DOF(params=params, dt=0.01)
    state = _make_state()
    u = np.array([[0.01], [0.0], [0.0], [0.5]])

    A = dyn.get_state_mat(0.0, state, u=u, use_continuous=True)
    F = dyn.get_state_mat(0.0, state, u=u)
    B = dyn.get_input_mat(0.0, state, u)
    Bc = dyn.get_input_mat(0.0, state, u, use_continuous=True)

    assert A.shape == (13, 13)
    assert F.shape == (13, 13)
    assert B.shape == (13, 4)
    assert Bc.shape == (13, 4)
    assert np.all(np.isfinite(A))
    assert np.all(np.isfinite(F))
    assert np.all(np.isfinite(B))
    assert np.all(np.isfinite(Bc))


def test_fixed_wing_aero_center_moment_is_opt_in():
    params = {
        "geometry": {
            "wing_area_m2": 0.5,
            "wing_span_m": 2.0,
            "mean_chord_m": 0.3,
            "aero_center_m": [0.25, 0.0, 0.05],
        },
        "aero": {
            "CL0": 0.6,
            "CD0": 0.05,
        },
    }
    state = _make_state()
    u = np.zeros((4, 1))

    dyn = FixedWing6DOF(params=params, dt=0.01)
    force, moment = dyn.calc_force_mom(state, u)
    test.assert_allclose(moment, np.zeros(3), atol=1e-12)

    params["geometry"]["apply_aero_center_moment"] = True
    dyn_shifted = FixedWing6DOF(params=params, dt=0.01)
    force_shifted, moment_shifted = dyn_shifted.calc_force_mom(state, u)
    aero_force_shifted, _ = dyn_shifted._calc_aero_force_mom(state.ravel(), u)

    test.assert_allclose(force_shifted, force)
    test.assert_allclose(
        moment_shifted,
        np.cross(np.asarray(params["geometry"]["aero_center_m"]), aero_force_shifted),
        atol=1e-12,
    )


def test_fixed_wing_air_data_uses_true_relative_wind():
    dyn = FixedWing6DOF(dt=0.01)
    state = _make_state()
    wind_ned = np.array([0.0, 3.0, 0.0])

    air_data = dyn.calc_air_data(state, state_args=(wind_ned,))

    test.assert_allclose(air_data["wind_ned"], wind_ned)
    test.assert_allclose(air_data["wind_body"], wind_ned)
    test.assert_allclose(air_data["air_vel_body"], np.array([15.0, -3.0, 0.0]))
    test.assert_allclose(air_data["airspeed"], np.sqrt(15.0**2 + 3.0**2))
    test.assert_allclose(air_data["alpha"], 0.0)
    test.assert_allclose(
        air_data["beta"], np.arcsin(-3.0 / np.sqrt(15.0**2 + 3.0**2))
    )


def test_fixed_wing_linearization_accepts_constant_wind():
    params = {
        "geometry": {
            "wing_area_m2": 0.3097,
            "wing_span_m": 1.27,
            "mean_chord_m": 0.25,
        },
        "aero": {
            "CL0": 0.1086,
            "CL_alpha": 4.58,
            "CD_min": 0.0434,
            "CL_minD": 0.23,
            "oswald_eff": 0.75,
            "Cm0": -0.0278,
            "Cm_alpha": -0.723,
            "Cm_delta_e": -0.8488,
        },
    }
    dyn = FixedWing6DOF(params=params, dt=0.01)
    state = _make_state()
    u = np.array([[0.01], [0.0], [0.0], [0.5]])
    wind_ned = np.array([2.0, -1.0, 0.5])

    A = dyn.get_state_mat(0.0, state, wind_ned, u=u, use_continuous=True)
    B = dyn.get_input_mat(0.0, state, u, state_args=(wind_ned,), use_continuous=True)
    xdot = dyn.state_derivative(0.0, state, u, state_args=(wind_ned,))

    assert A.shape == (13, 13)
    assert B.shape == (13, 4)
    assert xdot.shape == (13, 1)
    assert np.all(np.isfinite(A))
    assert np.all(np.isfinite(B))
    assert np.all(np.isfinite(xdot))
