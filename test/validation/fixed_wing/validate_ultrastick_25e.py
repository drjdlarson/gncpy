"""Open-loop Ultra Stick 25e fixed-wing validation run.

This script loads the Ultra Stick YAML, computes a steady-flight trim,
simulates the trimmed aircraft open loop, and saves plots under
``ValidationResults``.
"""

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import gncpy.math as gmath
from gncpy.dynamics.aircraft import FixedWing6DOF


THIS_DIR = Path(__file__).resolve().parent
CONFIG_FILE = THIS_DIR / "ultrastick_25e_config.yaml"
RESULTS_DIR = THIS_DIR / "ValidationResults"


def make_trim_state(airspeed, alpha, beta, roll, pitch, yaw=0.0):
    """Create a compact state for steady-flight trim variables."""
    state = np.zeros((13, 1))
    state[3] = airspeed * np.cos(alpha) * np.cos(beta)
    state[4] = airspeed * np.sin(beta)
    state[5] = airspeed * np.sin(alpha) * np.cos(beta)
    state[6:10, 0] = gmath.euler_to_quat(roll, pitch, yaw)
    return state


def trim_ultrastick(dyn, airspeed=17.0):
    """Solve a steady level-flight trim for the asymmetric propulsion case."""

    def residual(z):
        alpha, beta, roll, pitch, delta_e, delta_a, delta_r, throttle = z
        state = make_trim_state(airspeed, alpha, beta, roll, pitch)
        u = np.array([[delta_e], [delta_a], [delta_r], [throttle]])
        xdot = dyn.state_derivative(0.0, state, u).ravel()
        return np.array(
            [
                xdot[1],
                xdot[2],
                xdot[3],
                xdot[4],
                xdot[5],
                xdot[10],
                xdot[11],
                xdot[12],
            ]
        )

    sol = optimize.least_squares(
        residual,
        x0=np.array(
            [
                np.deg2rad(3.0),
                0.0,
                0.0,
                np.deg2rad(3.0),
                np.deg2rad(10.0),
                0.0,
                0.0,
                0.65,
            ]
        ),
        bounds=(
            np.array(
                [
                    np.deg2rad(-8.0),
                    np.deg2rad(-15.0),
                    np.deg2rad(-20.0),
                    np.deg2rad(-5.0),
                    np.deg2rad(-25.0),
                    np.deg2rad(-20.0),
                    np.deg2rad(-25.0),
                    0.0,
                ]
            ),
            np.array(
                [
                    np.deg2rad(14.0),
                    np.deg2rad(15.0),
                    np.deg2rad(20.0),
                    np.deg2rad(20.0),
                    np.deg2rad(25.0),
                    np.deg2rad(20.0),
                    np.deg2rad(25.0),
                    1.0,
                ]
            ),
        ),
        xtol=1e-10,
        ftol=1e-10,
        gtol=1e-10,
    )
    alpha, beta, roll, pitch, delta_e, delta_a, delta_r, throttle = sol.x
    state = make_trim_state(airspeed, alpha, beta, roll, pitch)
    control = np.array([[delta_e], [delta_a], [delta_r], [throttle]])
    return state, control, sol


def simulate(dyn, state0, control, duration=12.0):
    """Simulate the aircraft open loop at a fixed control."""
    time = np.arange(0.0, duration + dyn.dt, dyn.dt)
    states = np.zeros((time.size, state0.size))
    states[0] = state0.ravel()
    state = state0.copy()
    for ii, tt in enumerate(time[:-1]):
        state = dyn.propagate_state(tt, state, u=control)
        states[ii + 1] = state.ravel()
    return time, states


def plot_results(time, states, control, output_file):
    """Save validation plots."""
    euler = np.array([gmath.quat_to_euler(q) for q in states[:, 6:10]])
    body_vel = states[:, 3:6]
    speed = np.linalg.norm(body_vel, axis=1)
    alpha = np.rad2deg(np.arctan2(body_vel[:, 2], body_vel[:, 0]))

    fig, axs = plt.subplots(4, 1, figsize=(9, 10), sharex=True)
    axs[0].plot(time, speed)
    axs[0].set_ylabel("Airspeed (m/s)")
    axs[0].grid(True)

    axs[1].plot(time, -states[:, 2])
    axs[1].set_ylabel("Altitude (m)")
    axs[1].grid(True)

    axs[2].plot(time, np.rad2deg(euler[:, 0]), label="roll")
    axs[2].plot(time, np.rad2deg(euler[:, 1]), label="pitch")
    axs[2].plot(time, np.rad2deg(euler[:, 2]), label="yaw")
    axs[2].set_ylabel("Euler (deg)")
    axs[2].legend(loc="best")
    axs[2].grid(True)

    axs[3].plot(time, alpha, label="alpha")
    axs[3].plot(time, np.rad2deg(states[:, 10]), label="p")
    axs[3].plot(time, np.rad2deg(states[:, 11]), label="q")
    axs[3].plot(time, np.rad2deg(states[:, 12]), label="r")
    axs[3].set_ylabel("Angles/rates")
    axs[3].set_xlabel("Time (s)")
    axs[3].legend(loc="best")
    axs[3].grid(True)

    fig.suptitle(
        "Ultra Stick 25e open-loop trim "
        "(de={:.2f} deg, da={:.3f} deg, dr={:.3f} deg, throttle={:.3f})".format(
            np.rad2deg(control[0, 0]),
            np.rad2deg(control[1, 0]),
            np.rad2deg(control[2, 0]),
            control[3, 0],
        )
    )
    fig.tight_layout()
    fig.savefig(output_file, dpi=150)
    plt.close(fig)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    dyn = FixedWing6DOF(params_file=str(CONFIG_FILE), dt=0.01)
    trim_state, trim_control, sol = trim_ultrastick(dyn)
    time, states = simulate(dyn, trim_state, trim_control)

    plot_file = RESULTS_DIR / "ultrastick_25e_open_loop_trim.png"
    summary_file = RESULTS_DIR / "ultrastick_25e_trim_summary.txt"
    plot_results(time, states, trim_control, plot_file)

    trim_resid = dyn.state_derivative(0.0, trim_state, trim_control).ravel()
    summary = [
        "Ultra Stick 25e fixed-wing validation",
        f"Config: {CONFIG_FILE}",
        f"Trim success: {sol.success}",
        f"Trim cost: {sol.cost:.8e}",
        f"Trim alpha deg: {np.rad2deg(sol.x[0]):.6f}",
        f"Trim beta deg: {np.rad2deg(sol.x[1]):.6f}",
        f"Trim roll deg: {np.rad2deg(sol.x[2]):.6f}",
        f"Trim pitch deg: {np.rad2deg(sol.x[3]):.6f}",
        f"Trim elevator deg: {np.rad2deg(sol.x[4]):.6f}",
        f"Trim aileron deg: {np.rad2deg(sol.x[5]):.6f}",
        f"Trim rudder deg: {np.rad2deg(sol.x[6]):.6f}",
        f"Trim throttle: {sol.x[7]:.6f}",
        (
            "Trim residual [ve, vd, u_dot, v_dot, w_dot, p_dot, q_dot, r_dot]: "
            f"{trim_resid[[1, 2, 3, 4, 5, 10, 11, 12]]}"
        ),
        f"Final position NED m: {states[-1, 0:3]}",
        f"Final body velocity m/s: {states[-1, 3:6]}",
        f"Plot: {plot_file}",
    ]
    summary_file.write_text("\n".join(summary) + "\n")
    print("\n".join(summary))


if __name__ == "__main__":
    main()
