"""Shared utilities for omnicopter INDI validation scripts.

These helpers keep the validation scripts focused on scenario definition rather
than repeating the same setup, sensor, simulation, and plotting code.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import gncpy.math as gmath
from gncpy.control.INDI import INDI
from gncpy.dynamics.aircraft.complex_multirotor import ComplexMultirotor, v_smap_quat
from gncpy.dynamics.aircraft.simple_multirotor import Effector, e_smap, yaml


DT = 0.001
TAU_MOT = 0.032

INITIAL_POSITION = np.array([0.0, 0.0, -10.0])
INITIAL_VELOCITY = np.array([0.0, 0.0, 0.0])
INITIAL_ATTITUDE = np.array([0.0, 0.0, 0.0])
INITIAL_ANGULAR_VELOCITY = np.array([0.0, 0.0, 0.0])

REF_LAT = 34.0
REF_LON = -86.0
TERRAIN_ALT = 0.0
NED_MAG_FIELD = np.array([20.0, 5.0, 45.0])

K_VEL = 5.0
K_OMEGA = 10.0


@dataclass(frozen=True)
class SensorConfig:
    sigma_vel: float = 0.05
    sigma_accel: float = 0.1
    sigma_omega: float = 0.01
    bias_vel: tuple[float, float, float] = (0.0, 0.0, 0.0)
    bias_accel: tuple[float, float, float] = (0.0, 0.0, 0.0)
    bias_omega: tuple[float, float, float] = (0.0, 0.0, 0.0)
    fc_vel: float = 5.0
    fc_accel: float = 20.0
    fc_omega: float = 20.0
    fc_alpha: float = 5.0


DEFAULT_SENSOR_CONFIG = SensorConfig()


class MotorDynamicsEffector(Effector):
    """First-order motor dynamics with optional per-motor effectiveness mismatch."""

    def __init__(
        self,
        num_motors,
        tau_mot,
        initial_state=None,
        efficiency=None,
        efficiency_mode="none",
    ):
        self.num_motors = num_motors
        self.tau_mot = tau_mot
        self.state = (
            np.array(initial_state, dtype=float).flatten().copy()
            if initial_state is not None
            else np.zeros(num_motors)
        )
        if efficiency is None:
            self.efficiency = np.ones(num_motors)
        else:
            self.efficiency = np.array(efficiency, dtype=float).flatten().copy()
        self.efficiency_mode = efficiency_mode

    def set_initial_state(self, initial_state):
        self.state = np.array(initial_state, dtype=float).flatten().copy()

    def step(self, input_cmds, dt):
        input_cmds = np.array(input_cmds, dtype=float).flatten()

        if self.efficiency_mode == "command":
            effector_cmds = input_cmds * self.efficiency[: self.num_motors]
        else:
            effector_cmds = input_cmds

        alpha = np.exp(-dt / self.tau_mot)
        self.state = effector_cmds + (self.state - effector_cmds) * alpha

        if self.efficiency_mode == "output":
            return self.state.copy() * self.efficiency[: self.num_motors]
        return self.state.copy()


class FilteredRateSensors:
    """Simple filtered sensor model for INDI validation."""

    def __init__(self, dt, config=DEFAULT_SENSOR_CONFIG, seed=42):
        self.dt = dt
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.alpha_vel = self._calc_alpha(config.fc_vel)
        self.alpha_accel = self._calc_alpha(config.fc_accel)
        self.alpha_omega = self._calc_alpha(config.fc_omega)
        self.alpha_alpha = self._calc_alpha(config.fc_alpha)
        self.reset()

    def _calc_alpha(self, cutoff_hz):
        return self.dt / (self.dt + 1.0 / (2.0 * np.pi * cutoff_hz))

    def reset(self):
        self.vel_filt = np.zeros(3)
        self.accel_filt = np.zeros(3)
        self.omega_filt = np.zeros(3)
        self.alpha_filt = np.zeros(3)
        self.omega_prev = np.zeros(3)

    def measure(self, true_state, first_step=False):
        cfg = self.config
        body_vel_true = true_state[v_smap_quat.body_vel].flatten()
        body_accel_true = true_state[v_smap_quat.body_accel].flatten()
        body_omega_true = true_state[v_smap_quat.body_rot_rate].flatten()

        vel_noise = self.rng.normal(0, cfg.sigma_vel, 3)
        accel_noise = self.rng.normal(0, cfg.sigma_accel, 3)
        omega_noise = self.rng.normal(0, cfg.sigma_omega, 3)

        body_vel_meas = body_vel_true + vel_noise + np.array(cfg.bias_vel)
        body_accel_meas = body_accel_true + accel_noise + np.array(cfg.bias_accel)
        body_omega_meas = body_omega_true + omega_noise + np.array(cfg.bias_omega)

        self.vel_filt = self.alpha_vel * body_vel_meas + (
            1.0 - self.alpha_vel
        ) * self.vel_filt
        self.accel_filt = self.alpha_accel * body_accel_meas + (
            1.0 - self.alpha_accel
        ) * self.accel_filt
        self.omega_filt = self.alpha_omega * body_omega_meas + (
            1.0 - self.alpha_omega
        ) * self.omega_filt

        if first_step:
            alpha_meas = np.zeros(3)
        else:
            alpha_meas = (self.omega_filt - self.omega_prev) / self.dt

        self.alpha_filt = self.alpha_alpha * alpha_meas + (
            1.0 - self.alpha_alpha
        ) * self.alpha_filt
        self.omega_prev = self.omega_filt.copy()

        x = np.concatenate([self.vel_filt, self.omega_filt])
        x_dot = np.concatenate([self.accel_filt, self.alpha_filt])
        return x, x_dot


class PerfectRateSensors:
    """Noise-free, delay-free state feedback for idealized baseline tests."""

    def measure(self, true_state, first_step=False):
        del first_step
        body_vel_true = true_state[v_smap_quat.body_vel].flatten()
        body_accel_true = true_state[v_smap_quat.body_accel].flatten()
        body_omega_true = true_state[v_smap_quat.body_rot_rate].flatten()
        body_rot_accel_true = true_state[v_smap_quat.body_rot_accel].flatten()
        x = np.concatenate([body_vel_true, body_omega_true])
        x_dot = np.concatenate([body_accel_true, body_rot_accel_true])
        return x, x_dot


def build_effectiveness_matrix(dyn):
    """Construct the fixed INDI control effectiveness matrix from the LoFi model."""
    mass = dyn.vehicle.params.mass.mass_kg
    inertia = np.array(dyn.vehicle.params.mass.inertia_kgm2)
    num_motors = dyn.vehicle.params.motor.num_motors
    t_max = dyn.vehicle.params.prop.poly_thrust[0]

    thrust_dirs = np.zeros((3, num_motors))
    motor_pos = np.zeros((3, num_motors))
    for idx in range(num_motors):
        thrust_dirs[:, idx] = dyn.vehicle.params.motor.thrust_dir[idx]
        motor_pos[:, idx] = dyn.vehicle.params.motor.pos_m[idx]

    p_cross_n = np.zeros((3, num_motors))
    for idx in range(num_motors):
        p_cross_n[:, idx] = np.cross(motor_pos[:, idx], thrust_dirs[:, idx])

    b0_force = (1.0 / mass) * t_max * thrust_dirs
    b0_moment = np.linalg.inv(inertia) @ (t_max * p_cross_n)
    return np.vstack([b0_force, b0_moment])


def compute_hover_commands(lofi_dyn, b0, hifi_config_file):
    """Compute hover commands for the HiFi model from the LoFi trim solution."""
    gravity = lofi_dyn.env.state[e_smap.gravity]
    desired_accel = np.array([0.0, 0.0, -gravity[2], 0.0, 0.0, 0.0])
    hover_cmds_lofi = np.linalg.pinv(b0) @ desired_accel

    with open(hifi_config_file, "r") as handle:
        hifi_params = yaml.load(handle)

    c2 = hifi_params.prop.poly_thrust[0]
    lofi_thrust = lofi_dyn.vehicle.params.prop.poly_thrust[0] * hover_cmds_lofi
    return np.sign(lofi_thrust) * np.sqrt(np.abs(lofi_thrust) / c2)


def create_omnicopter_setup(
    lofi_config_file,
    hifi_config_file,
    dt=DT,
    tau_mot=TAU_MOT,
    k_vel=K_VEL,
    k_omega=K_OMEGA,
    motor_efficiency=None,
    efficiency_mode="none",
    use_motor_dynamics=True,
):
    """Create LoFi/HiFi dynamics objects and the baseline INDI controller."""
    lofi_dyn = ComplexMultirotor(str(lofi_config_file))
    lofi_dyn.set_initial_conditions(
        INITIAL_POSITION,
        INITIAL_VELOCITY,
        INITIAL_ATTITUDE,
        INITIAL_ANGULAR_VELOCITY,
        REF_LAT,
        REF_LON,
        TERRAIN_ALT,
        NED_MAG_FIELD,
    )

    b0 = build_effectiveness_matrix(lofi_dyn)
    hover_cmds_hifi = compute_hover_commands(lofi_dyn, b0, hifi_config_file)
    num_motors = lofi_dyn.vehicle.params.motor.num_motors

    if use_motor_dynamics:
        motor_effector = MotorDynamicsEffector(
            num_motors=num_motors,
            tau_mot=tau_mot,
            initial_state=hover_cmds_hifi,
            efficiency=motor_efficiency,
            efficiency_mode=efficiency_mode,
        )
    else:
        motor_effector = Effector()

    hifi_dyn = ComplexMultirotor(str(hifi_config_file), effector=motor_effector)
    indi_ctrl = INDI(omit_A=True)
    gains = np.diag([k_vel, k_vel, k_vel, k_omega, k_omega, k_omega])
    indi_ctrl.set_state_model(dt=dt, K=gains, B0=b0)

    return {
        "dt": dt,
        "tau_mot": tau_mot,
        "lofi_dyn": lofi_dyn,
        "hifi_dyn": hifi_dyn,
        "indi_ctrl": indi_ctrl,
        "motor_effector": motor_effector,
        "b0": b0,
        "gravity": lofi_dyn.env.state[e_smap.gravity].copy(),
        "hover_cmds_hifi": hover_cmds_hifi,
        "num_motors": num_motors,
        "use_motor_dynamics": use_motor_dynamics,
        "lofi_config_file": Path(lofi_config_file),
        "hifi_config_file": Path(hifi_config_file),
    }


def reset_hifi_vehicle(
    setup,
    initial_position=INITIAL_POSITION,
    initial_velocity=INITIAL_VELOCITY,
    initial_attitude=INITIAL_ATTITUDE,
    initial_angular_velocity=INITIAL_ANGULAR_VELOCITY,
):
    """Reset the HiFi model to a clean initial condition."""
    if hasattr(setup["motor_effector"], "set_initial_state"):
        setup["motor_effector"].set_initial_state(setup["hover_cmds_hifi"])
    setup["hifi_dyn"].set_initial_conditions(
        np.array(initial_position, dtype=float),
        np.array(initial_velocity, dtype=float),
        np.array(initial_attitude, dtype=float),
        np.array(initial_angular_velocity, dtype=float),
        REF_LAT,
        REF_LON,
        TERRAIN_ALT,
        NED_MAG_FIELD,
    )
    setup["hifi_dyn"].vehicle.takenoff = True


def run_indi_rate_case(
    setup,
    test_name,
    sim_time,
    reference_func,
    sensor_config=DEFAULT_SENSOR_CONFIG,
    sensor_seed=42,
    initial_position=INITIAL_POSITION,
    initial_velocity=INITIAL_VELOCITY,
    initial_attitude=INITIAL_ATTITUDE,
    initial_angular_velocity=INITIAL_ANGULAR_VELOCITY,
    disturbance_callback=None,
    perfect_sensors=False,
):
    """Run a single INDI rate-tracking validation case."""
    reset_hifi_vehicle(
        setup,
        initial_position=initial_position,
        initial_velocity=initial_velocity,
        initial_attitude=initial_attitude,
        initial_angular_velocity=initial_angular_velocity,
    )

    dt = setup["dt"]
    num_steps = int(sim_time / dt)
    if perfect_sensors:
        sensors = PerfectRateSensors()
    else:
        sensors = FilteredRateSensors(dt, config=sensor_config, seed=sensor_seed)

    results = {
        "test_name": test_name,
        "time": np.zeros(num_steps),
        "body_vel": np.zeros((num_steps, 3)),
        "body_vel_ref": np.zeros((num_steps, 3)),
        "body_omega": np.zeros((num_steps, 3)),
        "body_omega_ref": np.zeros((num_steps, 3)),
        "ned_pos": np.zeros((num_steps, 3)),
        "euler": np.zeros((num_steps, 3)),
        "cmd": np.zeros((num_steps, setup["num_motors"])),
    }

    cur_state = setup["hifi_dyn"].vehicle.state.copy()
    cur_input = setup["hover_cmds_hifi"].copy()

    for idx in range(num_steps):
        cur_time = idx * dt
        results["time"][idx] = cur_time

        body_vel_true = cur_state[v_smap_quat.body_vel].flatten()
        body_omega_true = cur_state[v_smap_quat.body_rot_rate].flatten()
        quat = cur_state[v_smap_quat.quat].flatten()

        x, x_dot = sensors.measure(cur_state, first_step=idx == 0)
        vb_ref, omega_ref, vb_ref_dot, omega_ref_dot = reference_func(cur_time)
        ref = np.concatenate([vb_ref, omega_ref])
        ref_dot = np.concatenate([vb_ref_dot, omega_ref_dot])

        u_cmd = setup["indi_ctrl"].calculate_control(
            cur_time=cur_time,
            cur_state=x,
            cur_state_dot=x_dot,
            cur_input=cur_input,
            ref=ref,
            ref_dot=ref_dot,
        )
        u_cmd = np.clip(u_cmd, -1.0, 1.0)

        results["body_vel"][idx, :] = body_vel_true
        results["body_vel_ref"][idx, :] = vb_ref
        results["body_omega"][idx, :] = body_omega_true
        results["body_omega_ref"][idx, :] = omega_ref
        results["ned_pos"][idx, :] = cur_state[v_smap_quat.ned_pos].flatten()
        results["cmd"][idx, :] = u_cmd
        roll, pitch, yaw = gmath.quat_to_euler(quat)
        results["euler"][idx, :] = np.rad2deg([roll, pitch, yaw])

        next_state = setup["hifi_dyn"].propagate_state(dt, cur_state, u_cmd).flatten()
        if disturbance_callback is not None:
            next_state = disturbance_callback(cur_time, cur_state, next_state, setup)

        cur_state = next_state
        if hasattr(setup["motor_effector"], "state"):
            cur_input = setup["motor_effector"].state.copy()
        else:
            cur_input = u_cmd.copy()

    vel_err = np.linalg.norm(results["body_vel"] - results["body_vel_ref"], axis=1)
    omega_err = np.rad2deg(
        np.linalg.norm(results["body_omega"] - results["body_omega_ref"], axis=1)
    )

    results["vel_error_mean"] = float(vel_err.mean())
    results["vel_error_max"] = float(vel_err.max())
    results["omega_error_mean_deg"] = float(omega_err.mean())
    results["omega_error_max_deg"] = float(omega_err.max())
    results["sat_fraction"] = float(np.mean(np.abs(results["cmd"]) >= 0.999))

    return results


def print_setup_summary(setup):
    """Print the shared omnicopter setup summary."""
    print("=" * 70)
    print("Setting up omnicopter dynamics and INDI controller")
    print("=" * 70)
    print(f"Gravity: {setup['gravity'][2]:.4f} m/s^2")
    print(f"B0 matrix: {setup['b0'].shape}")
    print(f"Number of motors: {setup['num_motors']}")
    print(
        "Hover commands (HiFi): "
        f"[{setup['hover_cmds_hifi'].min():.3f}, {setup['hover_cmds_hifi'].max():.3f}]"
    )
    print(f"Motor dynamics enabled: {setup['use_motor_dynamics']}")
    print(f"INDI gains: K_vel={K_VEL}, K_omega={K_OMEGA}")
    print("Setup complete!\n")


def print_case_summary(results):
    """Print compact tracking metrics for a case."""
    print(
        f"Velocity error: mean={results['vel_error_mean']:.4f}, "
        f"max={results['vel_error_max']:.4f} m/s"
    )
    print(
        f"Rate error: mean={results['omega_error_mean_deg']:.4f}, "
        f"max={results['omega_error_max_deg']:.4f} deg/s"
    )
    print(f"Motor saturation fraction: {results['sat_fraction']:.4f}")


def plot_rate_tracking_results(results, title, output_file):
    """Plot a 2x2 summary for a tracking case."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    fig.suptitle(title, fontsize=14, fontweight="bold")

    ax = axes[0, 0]
    ax.plot(results["time"], results["body_vel"][:, 0], "r-", label="vb_x")
    ax.plot(
        results["time"],
        results["body_vel_ref"][:, 0],
        "r--",
        alpha=0.7,
        label="vb_x ref",
    )
    ax.plot(results["time"], results["body_vel"][:, 1], "g-", label="vb_y")
    ax.plot(
        results["time"],
        results["body_vel_ref"][:, 1],
        "g--",
        alpha=0.7,
        label="vb_y ref",
    )
    ax.plot(results["time"], results["body_vel"][:, 2], "b-", label="vb_z")
    ax.plot(
        results["time"],
        results["body_vel_ref"][:, 2],
        "b--",
        alpha=0.7,
        label="vb_z ref",
    )
    ax.axhline(0, color="k", linestyle="--", alpha=0.3)
    ax.set_ylabel("Body Velocity (m/s)")
    ax.set_ylim([-1.3, 1.3])
    ax.legend(loc="upper right", ncol=3, fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(results["time"], np.rad2deg(results["body_omega"][:, 0]), "r-", label="p")
    ax.plot(
        results["time"],
        np.rad2deg(results["body_omega_ref"][:, 0]),
        "r--",
        alpha=0.7,
        label="p ref",
    )
    ax.plot(results["time"], np.rad2deg(results["body_omega"][:, 1]), "g-", label="q")
    ax.plot(
        results["time"],
        np.rad2deg(results["body_omega_ref"][:, 1]),
        "g--",
        alpha=0.7,
        label="q ref",
    )
    ax.plot(results["time"], np.rad2deg(results["body_omega"][:, 2]), "b-", label="r")
    ax.plot(
        results["time"],
        np.rad2deg(results["body_omega_ref"][:, 2]),
        "b--",
        alpha=0.7,
        label="r ref",
    )
    ax.axhline(0, color="k", linestyle="--", alpha=0.3)
    ax.set_ylabel("Body Angular Rate (deg/s)")
    ax.set_ylim([-65, 65])
    ax.legend(loc="upper right", ncol=3, fontsize=7)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(results["time"], results["euler"][:, 0], "r-", label="Roll")
    ax.plot(results["time"], results["euler"][:, 1], "g-", label="Pitch")
    ax.plot(results["time"], results["euler"][:, 2], "b-", label="Yaw")
    ax.axhline(0, color="k", linestyle="--", alpha=0.3)
    ax.set_ylabel("Euler Angles (deg)")
    ax.set_ylim([-180, 180])
    ax.set_xlabel("Time (s)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    for idx in range(results["cmd"].shape[1]):
        ax.plot(results["time"], results["cmd"][:, idx], alpha=0.7, label=f"u{idx}")
    ax.axhline(0, color="k", linestyle="--", alpha=0.3)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Motor Commands")
    ax.set_ylim([-1.0, 1.0])
    ax.legend(loc="upper right", ncol=4, fontsize=7)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = Path(output_file)
    output_file.parent.mkdir(exist_ok=True)
    plt.savefig(output_file, dpi=150)
    print(f"Figure saved to: {output_file}")


def plot_hover_results(results, title, output_file, depth_ref):
    """Plot a 4-panel hover summary."""
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    ax = axes[0]
    ax.plot(results["time"], results["body_vel"][:, 0], "r-", label="vb_x")
    ax.plot(results["time"], results["body_vel"][:, 1], "g-", label="vb_y")
    ax.plot(results["time"], results["body_vel"][:, 2], "b-", label="vb_z")
    ax.axhline(0, color="k", linestyle="--", alpha=0.3)
    ax.set_ylabel("Body Velocity (m/s)")
    ax.legend(loc="upper right")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(results["time"], np.rad2deg(results["body_omega"][:, 0]), "r-", label="p")
    ax.plot(results["time"], np.rad2deg(results["body_omega"][:, 1]), "g-", label="q")
    ax.plot(results["time"], np.rad2deg(results["body_omega"][:, 2]), "b-", label="r")
    ax.axhline(0, color="k", linestyle="--", alpha=0.3)
    ax.set_ylabel("Body Angular Rate (deg/s)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(results["time"], results["ned_pos"][:, 0], "r-", label="N")
    ax.plot(results["time"], results["ned_pos"][:, 1], "g-", label="E")
    ax.plot(results["time"], results["ned_pos"][:, 2], "b-", label="D")
    ax.axhline(depth_ref, color="b", linestyle="--", alpha=0.3, label="D_ref")
    ax.set_ylabel("NED Position (m)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    ax = axes[3]
    for idx in range(results["cmd"].shape[1]):
        ax.plot(results["time"], results["cmd"][:, idx], alpha=0.7, label=f"u{idx}")
    ax.axhline(0, color="k", linestyle="--", alpha=0.3)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Motor Commands")
    ax.legend(loc="upper right", ncol=4, fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = Path(output_file)
    output_file.parent.mkdir(exist_ok=True)
    plt.savefig(output_file, dpi=150)
    print(f"Figure saved to: {output_file}")


def circular_velocity_reference(speed, period):
    omega = 2.0 * np.pi / period

    def _reference(tt):
        vb_ref = np.array(
            [speed * np.cos(omega * tt), speed * np.sin(omega * tt), 0.0]
        )
        omega_ref = np.zeros(3)
        vb_ref_dot = np.array(
            [-speed * omega * np.sin(omega * tt), speed * omega * np.cos(omega * tt), 0.0]
        )
        omega_ref_dot = np.zeros(3)
        return vb_ref, omega_ref, vb_ref_dot, omega_ref_dot

    return _reference


def roll_rate_reference(rate, duration):
    def _reference(tt):
        vb_ref = np.zeros(3)
        omega_ref = np.array([rate, 0.0, 0.0]) if tt < duration else np.zeros(3)
        return vb_ref, omega_ref, np.zeros(3), np.zeros(3)

    return _reference


def combined_velocity_roll_reference(speed, period, roll_rate, roll_duration):
    circle_reference = circular_velocity_reference(speed, period)
    roll_reference = roll_rate_reference(roll_rate, roll_duration)

    def _reference(tt):
        vb_ref, _, vb_ref_dot, _ = circle_reference(tt)
        _, omega_ref, _, omega_ref_dot = roll_reference(tt)
        return vb_ref, omega_ref, vb_ref_dot, omega_ref_dot

    return _reference


def helix_tumble_reference(
    circle_speed,
    circle_period,
    vertical_speed_amp,
    vertical_period,
    roll_rate,
    pitch_rate,
    yaw_rate,
    tumble_duration,
):
    circle_omega = 2.0 * np.pi / circle_period
    vertical_omega = 2.0 * np.pi / vertical_period

    def _reference(tt):
        vb_ref = np.array(
            [
                circle_speed * np.cos(circle_omega * tt),
                circle_speed * np.sin(circle_omega * tt),
                vertical_speed_amp * np.sin(vertical_omega * tt),
            ]
        )
        vb_ref_dot = np.array(
            [
                -circle_speed * circle_omega * np.sin(circle_omega * tt),
                circle_speed * circle_omega * np.cos(circle_omega * tt),
                vertical_speed_amp * vertical_omega * np.cos(vertical_omega * tt),
            ]
        )
        if tt < tumble_duration:
            omega_ref = np.array([roll_rate, pitch_rate, yaw_rate])
        else:
            omega_ref = np.zeros(3)
        return vb_ref, omega_ref, vb_ref_dot, np.zeros(3)

    return _reference


def compute_wind_force(wind_ned, vel_ned, quat, cd, frontal_area, air_density=1.225):
    """Compute aerodynamic drag from the relative wind in body coordinates."""
    v_rel_ned = wind_ned - vel_ned
    dcm_ned_to_body = gmath.quat_to_dcm(quat).T
    v_rel_body = dcm_ned_to_body @ v_rel_ned

    force_body = np.zeros(3)
    for idx in range(3):
        force_body[idx] = (
            0.5
            * air_density
            * cd
            * frontal_area[idx]
            * v_rel_body[idx]
            * np.abs(v_rel_body[idx])
        )
    return force_body


def make_wind_disturbance(
    wind_velocity,
    wind_gust_amp,
    wind_gust_freq,
    update_derived_states=True,
):
    """Create a disturbance callback that injects wind drag after propagation."""

    wind_velocity = np.array(wind_velocity, dtype=float)
    wind_gust_amp = np.array(wind_gust_amp, dtype=float)
    wind_gust_freq = np.array(wind_gust_freq, dtype=float)

    def _disturbance(cur_time, cur_state, next_state, setup):
        quat = cur_state[v_smap_quat.quat].flatten()
        ned_vel = cur_state[v_smap_quat.ned_vel].flatten()
        gust = wind_gust_amp * np.sin(2.0 * np.pi * wind_gust_freq * cur_time)
        wind_total = wind_velocity + gust

        cd = setup["hifi_dyn"].vehicle.params.aero.cd
        frontal_area = np.array(setup["hifi_dyn"].vehicle.params.geo.front_area_m2)
        wind_force_body = compute_wind_force(wind_total, ned_vel, quat, cd, frontal_area)
        wind_accel_body = wind_force_body / setup["hifi_dyn"].vehicle.params.mass.mass_kg

        next_state = next_state.copy()
        next_state[v_smap_quat.body_vel] += wind_accel_body * setup["dt"]

        if update_derived_states:
            next_quat = next_state[v_smap_quat.quat].flatten()
            next_body_vel = next_state[v_smap_quat.body_vel].flatten()
            next_state[v_smap_quat.ned_vel] = gmath.quat_rotate_vector(
                next_quat, next_body_vel
            )

        return next_state

    return _disturbance
