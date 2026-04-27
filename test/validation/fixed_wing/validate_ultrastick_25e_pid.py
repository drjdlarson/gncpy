"""Closed-loop Ultra Stick 25e PID validation.

This script trims the aircraft, then applies simple cascaded PID control to
command a climbing turn while maintaining airspeed.
"""

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import gncpy.math as gmath
from gncpy.control import PID
from gncpy.dynamics.aircraft import FixedWing6DOF

from validate_ultrastick_25e import CONFIG_FILE, RESULTS_DIR, trim_ultrastick


THIS_DIR = Path(__file__).resolve().parent

DT = 0.01
DURATION = 20.0
MANEUVER_START_S = 2.0

WIND_NED_MPS = np.array([0.0, 0.0, 0.0])

CTRL_LIMITS = {
    "elevator": np.deg2rad(20.0),
    "aileron": np.deg2rad(20.0),
    "rudder": np.deg2rad(20.0),
    "flaps": np.deg2rad(20.0),
    "throttle_min": 0.0,
    "throttle_max": 1.0,
}

ATTITUDE_REFS = {
    "pitch_max": np.deg2rad(10.0),
    "roll_max": np.deg2rad(30.0),
    "roll_cmd": np.deg2rad(20.0),
}

TARGET_AIRSPEED_MPS = 17.0
TARGET_ALTITUDE_M = 30.0

PID_GAINS = {
    "pitch": {"p": 0.2, "i": 0.1, "d": 0.2},
    "roll": {"p": 0.15, "i": 0.1, "d": 0.2},
    "airspeed": {"p": 0.09, "i": 0.02, "d": 0.0},
    "altitude": {"p": 0.02, "i": 0.002, "d": 0.0},
}

DERIV_FILTER_TAU_S = 0.1


def wrap_to_pi(angle_rad):
    """Wrap angle(s) to [-pi, pi]."""
    return (np.asarray(angle_rad) + np.pi) % (2 * np.pi) - np.pi


def _make_pid_stack():
    return {
        "pitch": PID(
            kp=PID_GAINS["pitch"]["p"],
            ki=PID_GAINS["pitch"]["i"],
            kd=PID_GAINS["pitch"]["d"],
            tau=DERIV_FILTER_TAU_S,
        ),
        "roll": PID(
            kp=PID_GAINS["roll"]["p"],
            ki=PID_GAINS["roll"]["i"],
            kd=PID_GAINS["roll"]["d"],
            tau=DERIV_FILTER_TAU_S,
        ),
        "airspeed": PID(
            kp=PID_GAINS["airspeed"]["p"],
            ki=PID_GAINS["airspeed"]["i"],
            kd=PID_GAINS["airspeed"]["d"],
            tau=DERIV_FILTER_TAU_S,
            u_min=CTRL_LIMITS["throttle_min"] - 1.0,
            u_max=CTRL_LIMITS["throttle_max"] - 0.0,
        ),
        "altitude": PID(
            kp=PID_GAINS["altitude"]["p"],
            ki=PID_GAINS["altitude"]["i"],
            kd=PID_GAINS["altitude"]["d"],
            tau=DERIV_FILTER_TAU_S,
            u_min=-ATTITUDE_REFS["pitch_max"],
            u_max=ATTITUDE_REFS["pitch_max"],
        ),
    }


def _clip_control(u):
    out = u.copy()
    out[0, 0] = np.clip(out[0, 0], -CTRL_LIMITS["elevator"], CTRL_LIMITS["elevator"])
    out[1, 0] = np.clip(out[1, 0], -CTRL_LIMITS["aileron"], CTRL_LIMITS["aileron"])
    out[2, 0] = np.clip(out[2, 0], -CTRL_LIMITS["rudder"], CTRL_LIMITS["rudder"])
    out[3, 0] = np.clip(
        out[3, 0], CTRL_LIMITS["throttle_min"], CTRL_LIMITS["throttle_max"]
    )
    return out


def _command_schedule(time_s):
    if time_s < MANEUVER_START_S:
        return 0.0, 0.0
    return TARGET_ALTITUDE_M, ATTITUDE_REFS["roll_cmd"]


def simulate_closed_loop(dyn, trim_state, trim_control):
    pid = _make_pid_stack()
    time = np.arange(0.0, DURATION + dyn.dt, dyn.dt)
    states = np.zeros((time.size, trim_state.size))
    ctrls = np.zeros((time.size, 4))
    refs = np.zeros((time.size, 4))

    states[0] = trim_state.ravel()
    ctrls[0] = trim_control[:4, 0]
    state = trim_state.copy()

    trim_euler = np.array(gmath.quat_to_euler(trim_state[6:10, 0]))
    trim_altitude = -trim_state[2, 0]

    for ii, tt in enumerate(time):
        euler = np.array(gmath.quat_to_euler(state[6:10, 0]))
        air_data = dyn.calc_air_data(state, state_args=(WIND_NED_MPS,))
        airspeed = air_data["airspeed"]
        altitude = -state[2, 0]

        alt_ref_delta, roll_ref = _command_schedule(tt)
        altitude_ref = trim_altitude + alt_ref_delta
        pitch_ref = pid["altitude"].calculate_control(
            tt, measurement=np.array([altitude]), reference=np.array([altitude_ref])
        )[0]
        pitch_ref = np.clip(
            pitch_ref,
            -ATTITUDE_REFS["pitch_max"],
            ATTITUDE_REFS["pitch_max"],
        )
        roll_ref = np.clip(
            roll_ref, -ATTITUDE_REFS["roll_max"], ATTITUDE_REFS["roll_max"]
        )

        pitch_cmd = pid["pitch"].calculate_control(
            tt, measurement=np.array([euler[1]]), reference=np.array([pitch_ref])
        )[0]
        roll_cmd = pid["roll"].calculate_control(
            tt, measurement=np.array([euler[0]]), reference=np.array([roll_ref])
        )[0]
        throttle_cmd = pid["airspeed"].calculate_control(
            tt,
            measurement=np.array([airspeed]),
            reference=np.array([TARGET_AIRSPEED_MPS]),
        )[0]

        control = trim_control[:4].copy()
        control[0, 0] = trim_control[0, 0] - pitch_cmd
        control[1, 0] = trim_control[1, 0] - roll_cmd
        control[2, 0] = trim_control[2, 0]
        control[3, 0] = trim_control[3, 0] + throttle_cmd
        control = _clip_control(control)

        states[ii] = state.ravel()
        ctrls[ii] = control[:, 0]
        refs[ii] = [altitude_ref, roll_ref, pitch_ref, TARGET_AIRSPEED_MPS]

        if ii < time.size - 1:
            state = dyn.propagate_state(
                tt, state, u=control, state_args=(WIND_NED_MPS,)
            )

    return time, states, ctrls, refs, trim_euler


def plot_results(time, states, ctrls, refs, output_file):
    euler = wrap_to_pi(np.array([gmath.quat_to_euler(q) for q in states[:, 6:10]]))
    altitude = -states[:, 2]
    airspeed = np.linalg.norm(states[:, 3:6], axis=1)
    heading = np.rad2deg(euler[:, 2])

    fig, axs = plt.subplots(5, 1, figsize=(10, 13), sharex=True)

    axs[0].plot(time, airspeed, label="airspeed")
    axs[0].plot(time, refs[:, 3], "--", label="airspeed ref")
    axs[0].set_ylabel("Airspeed (m/s)")
    axs[0].legend(loc="best")
    axs[0].grid(True)

    axs[1].plot(time, altitude, label="altitude")
    axs[1].plot(time, refs[:, 0], "--", label="altitude ref")
    axs[1].set_ylabel("Altitude (m)")
    axs[1].legend(loc="best")
    axs[1].grid(True)

    axs[2].plot(time, np.rad2deg(euler[:, 0]), label="roll")
    axs[2].plot(time, np.rad2deg(refs[:, 1]), "--", label="roll ref")
    axs[2].plot(time, np.rad2deg(euler[:, 1]), label="pitch")
    axs[2].plot(time, np.rad2deg(refs[:, 2]), "--", label="pitch ref")
    axs[2].set_ylabel("Angles (deg)")
    axs[2].legend(loc="best")
    axs[2].grid(True)

    axs[3].plot(time, heading, label="yaw")
    axs[3].set_ylabel("Yaw (deg)")
    axs[3].legend(loc="best")
    axs[3].grid(True)

    axs[4].plot(time, np.rad2deg(ctrls[:, 0]), label="elevator")
    axs[4].plot(time, np.rad2deg(ctrls[:, 1]), label="aileron")
    axs[4].plot(time, np.rad2deg(ctrls[:, 2]), label="rudder")
    axs[4].plot(time, ctrls[:, 3], label="throttle")
    axs[4].set_ylabel("Controls")
    axs[4].set_xlabel("Time (s)")
    axs[4].legend(loc="best")
    axs[4].grid(True)

    fig.suptitle("Ultra Stick 25e PID climbing turn validation")
    fig.tight_layout()
    fig.savefig(output_file, dpi=150)
    plt.close(fig)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    dyn = FixedWing6DOF(params_file=str(CONFIG_FILE), dt=DT)
    trim_state, trim_control, trim_sol = trim_ultrastick(dyn)
    time, states, ctrls, refs, trim_euler = simulate_closed_loop(
        dyn, trim_state, trim_control
    )

    plot_file = RESULTS_DIR / "ultrastick_25e_pid_turn_climb.png"
    summary_file = RESULTS_DIR / "ultrastick_25e_pid_turn_climb_summary.txt"
    plot_results(time, states, ctrls, refs, plot_file)

    final_euler = wrap_to_pi(np.array(gmath.quat_to_euler(states[-1, 6:10])))
    final_air_data = dyn.calc_air_data(states[-1], state_args=(WIND_NED_MPS,))
    summary = [
        "Ultra Stick 25e PID climbing turn validation",
        f"Config: {CONFIG_FILE}",
        f"Trim success: {trim_sol.success}",
        f"Wind NED m/s: {WIND_NED_MPS}",
        f"Altitude target m: {refs[-1, 0]:.6f}",
        f"Final altitude m: {-states[-1, 2]:.6f}",
        f"Altitude gain m: {-states[-1, 2] + states[0, 2]:.6f}",
        f"Roll target deg: {np.rad2deg(refs[-1, 1]):.6f}",
        f"Final roll deg: {np.rad2deg(final_euler[0]):.6f}",
        f"Final pitch deg: {np.rad2deg(final_euler[1]):.6f}",
        f"Final yaw deg: {np.rad2deg(final_euler[2]):.6f}",
        f"Final airspeed m/s: {final_air_data['airspeed']:.6f}",
        f"Final alpha deg: {np.rad2deg(final_air_data['alpha']):.6f}",
        f"Final beta deg: {np.rad2deg(final_air_data['beta']):.6f}",
        f"Final controls [de, da, dr] deg: {np.rad2deg(ctrls[-1, 0:3])}",
        f"Final throttle: {ctrls[-1, 3]:.6f}",
        f"Plot: {plot_file}",
    ]
    summary_file.write_text("\n".join(summary) + "\n")
    print("\n".join(summary))


if __name__ == "__main__":
    main()
