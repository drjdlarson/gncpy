"""Generate training data for NN-augmented INDI controller.

This script generates flight data for training a neural network to predict
the residual dynamics not captured by the baseline INDI model. The data
generation follows the literature approach using realistic sensor models.

Data Collection Overview:
    The script runs ~10 minutes of simulated flight with diverse maneuvers:
    - Phase 1 (0-2 min): Translational velocity tracking, zero angular rates
    - Phase 2 (2-4 min): Angular rate tracking, zero translational velocity
    - Phase 3 (4-6 min): Combined velocity + angular rate tracking
    - Phase 4 (6-8 min): Aggressive mixed maneuvers
    - Phase 5 (8-10 min): Random reference changes (step inputs)

Training Target (from equation 8 in literature):
    Δẋ_residual = ẋ^[k+1]_measured - (ẋ^[k]_measured + B0(u^[k] - u^[k-1]))

NN Observation Space (from equation 28):
    z^[k] = [v_b^[k], ω_b^[k], v̇_b^[k], ω̇_b^[k], u^[k-1]] ∈ R^20

Output:
    CSV files with:
    - nn_inputs.csv: Observation vectors for NN (filtered sensor data)
    - nn_targets.csv: Residual targets for training
    - metadata.csv: Simulation parameters and statistics
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

from gncpy.dynamics.aircraft.complex_multirotor import (
    ComplexMultirotor,
    v_smap_quat,
)
from gncpy.dynamics.aircraft.simple_multirotor import (
    Effector,
    e_smap,
    yaml,
)
import gncpy.math as gmath

from gncpy.control.INDI import INDI


# ============================================================================
# Motor dynamics effector (same as in test script)
# ============================================================================
class MotorDynamicsEffector(Effector):
    """First-order motor dynamics with per-motor efficiency variation."""

    MOTOR_EFFICIENCY = np.array([0.99, 1.01, 0.98, 1.02, 1.01, 0.97, 1.00, 0.98])

    def __init__(self, num_motors, tau_mot, initial_state=None):
        self.num_motors = num_motors
        self.tau_mot = tau_mot
        if initial_state is not None:
            self.state = np.array(initial_state).flatten().copy()
        else:
            self.state = np.zeros(num_motors)

    def set_initial_state(self, initial_state):
        self.state = np.array(initial_state).flatten().copy()

    def step(self, input_cmds, dt):
        input_cmds = np.array(input_cmds).flatten()
        scaled_cmds = input_cmds
        alpha = np.exp(-dt / self.tau_mot)
        self.state = scaled_cmds + (self.state - scaled_cmds) * alpha
        return self.state.copy() * self.MOTOR_EFFICIENCY[: self.num_motors]


# ============================================================================
# Wind force computation
# ============================================================================
def compute_wind_force(wind_ned, vel_ned, quat, cd, frontal_area, air_density=1.225):
    """Compute aerodynamic drag force from wind."""
    v_rel_ned = wind_ned - vel_ned
    dcm_ned_to_body = gmath.quat_to_dcm(quat).T
    v_rel_body = dcm_ned_to_body @ v_rel_ned
    effective_area = frontal_area[:3]
    force_body = np.zeros(3)
    for i in range(3):
        force_body[i] = (
            0.5
            * air_density
            * cd
            * effective_area[i]
            * v_rel_body[i]
            * np.abs(v_rel_body[i])
        )
    return force_body


# ============================================================================
# Reference trajectory generators
# ============================================================================
def generate_phase1_reference(t, phase_start):
    """Phase 1: Translational velocity tracking, zero angular rates.

    Includes figure-8 patterns, sinusoidal motions, and ramps.
    """
    t_rel = t - phase_start

    # Mix of circular and figure-8 patterns
    period = 10.0  # seconds per pattern cycle
    omega = 2.0 * np.pi / period

    # Figure-8 in horizontal plane with varying amplitude
    amp_x = 1.0 + 0.3 * np.sin(0.1 * t_rel)  # Slowly varying amplitude
    amp_y = 0.8 + 0.2 * np.cos(0.15 * t_rel)

    vb_ref = np.array(
        [
            amp_x * np.sin(omega * t_rel),
            amp_y * np.sin(2 * omega * t_rel),  # Double frequency for figure-8
            0.3 * np.sin(0.5 * omega * t_rel),  # Slow vertical
        ]
    )

    vb_ref_dot = np.array(
        [
            amp_x * omega * np.cos(omega * t_rel),
            amp_y * 2 * omega * np.cos(2 * omega * t_rel),
            0.3 * 0.5 * omega * np.cos(0.5 * omega * t_rel),
        ]
    )

    omega_ref = np.zeros(3)
    omega_ref_dot = np.zeros(3)

    return vb_ref, vb_ref_dot, omega_ref, omega_ref_dot


def generate_phase2_reference(t, phase_start):
    """Phase 2: Angular rate tracking, zero translational velocity.

    Roll, pitch, yaw rate commands with varying profiles.
    """
    t_rel = t - phase_start

    # Varying angular rate references
    period = 8.0
    omega = 2.0 * np.pi / period

    # Sequential and combined angular rates
    omega_ref = np.array(
        [
            0.5 * np.sin(omega * t_rel),  # Roll rate
            0.4 * np.sin(omega * t_rel + np.pi / 3),  # Pitch rate (phase shifted)
            0.3 * np.sin(0.5 * omega * t_rel),  # Yaw rate (slower)
        ]
    )

    omega_ref_dot = np.array(
        [
            0.5 * omega * np.cos(omega * t_rel),
            0.4 * omega * np.cos(omega * t_rel + np.pi / 3),
            0.3 * 0.5 * omega * np.cos(0.5 * omega * t_rel),
        ]
    )

    vb_ref = np.zeros(3)
    vb_ref_dot = np.zeros(3)

    return vb_ref, vb_ref_dot, omega_ref, omega_ref_dot


def generate_phase3_reference(t, phase_start):
    """Phase 3: Combined velocity + angular rate tracking.

    Moderate velocity with coordinated attitude changes.
    """
    t_rel = t - phase_start

    omega1 = 2.0 * np.pi / 12.0  # 12 second period
    omega2 = 2.0 * np.pi / 6.0  # 6 second period

    vb_ref = np.array(
        [
            0.8 * np.sin(omega1 * t_rel),
            0.6 * np.cos(omega1 * t_rel),
            0.2 * np.sin(omega2 * t_rel),
        ]
    )

    vb_ref_dot = np.array(
        [
            0.8 * omega1 * np.cos(omega1 * t_rel),
            -0.6 * omega1 * np.sin(omega1 * t_rel),
            0.2 * omega2 * np.cos(omega2 * t_rel),
        ]
    )

    omega_ref = np.array(
        [
            0.3 * np.sin(omega2 * t_rel),
            0.25 * np.cos(omega2 * t_rel + np.pi / 4),
            0.2 * np.sin(omega1 * t_rel),
        ]
    )

    omega_ref_dot = np.array(
        [
            0.3 * omega2 * np.cos(omega2 * t_rel),
            -0.25 * omega2 * np.sin(omega2 * t_rel + np.pi / 4),
            0.2 * omega1 * np.cos(omega1 * t_rel),
        ]
    )

    return vb_ref, vb_ref_dot, omega_ref, omega_ref_dot


def generate_phase4_reference(t, phase_start):
    """Phase 4: Aggressive mixed maneuvers.

    Higher amplitude, faster changes.
    """
    t_rel = t - phase_start

    omega1 = 2.0 * np.pi / 5.0  # 5 second period (faster)
    omega2 = 2.0 * np.pi / 3.0  # 3 second period

    vb_ref = np.array(
        [
            1.2 * np.sin(omega1 * t_rel) + 0.4 * np.sin(3 * omega1 * t_rel),
            1.0 * np.cos(omega1 * t_rel),
            0.5 * np.sin(omega2 * t_rel),
        ]
    )

    vb_ref_dot = np.array(
        [
            1.2 * omega1 * np.cos(omega1 * t_rel)
            + 0.4 * 3 * omega1 * np.cos(3 * omega1 * t_rel),
            -1.0 * omega1 * np.sin(omega1 * t_rel),
            0.5 * omega2 * np.cos(omega2 * t_rel),
        ]
    )

    omega_ref = np.array(
        [
            0.6 * np.sin(omega2 * t_rel),
            0.5 * np.sin(omega2 * t_rel + np.pi / 2),
            0.4 * np.sin(omega1 * t_rel),
        ]
    )

    omega_ref_dot = np.array(
        [
            0.6 * omega2 * np.cos(omega2 * t_rel),
            0.5 * omega2 * np.cos(omega2 * t_rel + np.pi / 2),
            0.4 * omega1 * np.cos(omega1 * t_rel),
        ]
    )

    return vb_ref, vb_ref_dot, omega_ref, omega_ref_dot


class Phase5ReferenceGenerator:
    """Phase 5: Random step changes (bandlimited noise).

    Generates piece-wise constant references that change at random intervals.
    """

    def __init__(self, seed=123):
        self.rng = np.random.default_rng(seed)
        self.current_vb_ref = np.zeros(3)
        self.current_omega_ref = np.zeros(3)
        self.next_change_time = 0.0
        self.hold_time_range = (1.0, 3.0)  # seconds

    def generate(self, t, phase_start):
        t_rel = t - phase_start

        if t_rel >= self.next_change_time:
            # Generate new random references
            self.current_vb_ref = self.rng.uniform(-1.0, 1.0, 3)
            self.current_vb_ref[2] *= 0.5  # Reduce vertical velocity
            self.current_omega_ref = self.rng.uniform(-0.5, 0.5, 3)

            # Schedule next change
            hold_time = self.rng.uniform(*self.hold_time_range)
            self.next_change_time = t_rel + hold_time

        # Zero derivatives for step inputs
        vb_ref_dot = np.zeros(3)
        omega_ref_dot = np.zeros(3)

        return (
            self.current_vb_ref.copy(),
            vb_ref_dot,
            self.current_omega_ref.copy(),
            omega_ref_dot,
        )


# ============================================================================
# Main data generation
# ============================================================================
def main():
    print("=" * 70)
    print("INDI Training Data Generation")
    print("=" * 70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # ========================================================================
    # Configuration
    # ========================================================================
    DT = 0.001  # Timestep (1 kHz)
    TAU_MOT = 0.032  # Motor time constant

    # Total simulation time (~10 minutes)
    TOTAL_TIME = 600.0  # 10 minutes

    # Phase durations (seconds)
    PHASE_DURATIONS = [120.0, 120.0, 120.0, 120.0, 120.0]  # 2 min each

    # Initial conditions
    INITIAL_POSITION = np.array([0.0, 0.0, -10.0])
    INITIAL_VELOCITY = np.array([0.0, 0.0, 0.0])
    INITIAL_ATTITUDE = np.array([0.0, 0.0, 0.0])
    INITIAL_ANGULAR_VELOCITY = np.array([0.0, 0.0, 0.0])
    REF_LAT, REF_LON, TERRAIN_ALT = 34.0, -86.0, 0.0

    # INDI gains
    K_VEL = 5.0
    K_OMEGA = 10.0

    # Sensor noise parameters
    SIGMA_VEL = 0.05
    SIGMA_ACCEL = 0.1
    SIGMA_OMEGA = 0.01

    # Filter cutoff frequencies
    FC_VEL = 5.0
    FC_ACCEL = 20.0
    FC_OMEGA = 20.0
    FC_ALPHA = 5.0

    # Wind parameters
    WIND_VELOCITY = np.array([8.0, 12.0, 1.0])
    WIND_GUST_AMP = np.array([4.0, 5.0, 1.0])
    WIND_GUST_FREQ = np.array([0.3, 0.4, 0.5])

    # Random seed for reproducibility
    np.random.seed(42)

    # Subsample rate for saving (save every N steps to reduce file size)
    # At 1kHz, 10 min = 600,000 samples. Save at 100 Hz = 60,000 samples
    SAVE_SUBSAMPLE = 10

    print(f"\nConfiguration:")
    print(f"  Timestep: {DT*1000:.1f} ms")
    print(f"  Total time: {TOTAL_TIME/60:.1f} minutes")
    print(f"  Save rate: {1.0/(DT*SAVE_SUBSAMPLE):.0f} Hz")
    print(f"  Expected samples: {int(TOTAL_TIME/(DT*SAVE_SUBSAMPLE))}")

    # ========================================================================
    # Setup dynamics and controller
    # ========================================================================
    print("\nSetting up dynamics...")

    config_dir = Path(__file__).parent
    lofi_config_file = config_dir / "omnicopter_config.yaml"
    hifi_config_file = config_dir / "omnicopter_config_hifi.yaml"

    # Load LoFi to get B0 matrix
    lofi_dyn = ComplexMultirotor(str(lofi_config_file))
    ned_mag = np.array([20.0, 5.0, 45.0])
    lofi_dyn.set_initial_conditions(
        INITIAL_POSITION,
        INITIAL_VELOCITY,
        INITIAL_ATTITUDE,
        INITIAL_ANGULAR_VELOCITY,
        REF_LAT,
        REF_LON,
        TERRAIN_ALT,
        ned_mag,
    )

    gravity = lofi_dyn.env.state[e_smap.gravity]
    mass = lofi_dyn.vehicle.params.mass.mass_kg
    inertia = np.array(lofi_dyn.vehicle.params.mass.inertia_kgm2)
    num_motors = lofi_dyn.vehicle.params.motor.num_motors
    T_max = lofi_dyn.vehicle.params.prop.poly_thrust[0]

    # Build B0 matrix
    N = np.zeros((3, num_motors))
    P = np.zeros((3, num_motors))
    for i in range(num_motors):
        N[:, i] = lofi_dyn.vehicle.params.motor.thrust_dir[i]
        P[:, i] = lofi_dyn.vehicle.params.motor.pos_m[i]

    B0_force = (1.0 / mass) * T_max * N
    P_cross_N = np.zeros((3, num_motors))
    for i in range(num_motors):
        P_cross_N[:, i] = np.cross(P[:, i], N[:, i])

    J_inv = np.linalg.inv(inertia)
    B0_moment = J_inv @ (T_max * P_cross_N)
    B0 = np.vstack([B0_force, B0_moment])
    B0_inv = np.linalg.pinv(B0)

    print(f"  B0 matrix shape: {B0.shape}")
    print(f"  Number of motors: {num_motors}")

    # Compute hover commands
    g_mag = gravity[2]
    desired_accel = np.array([0.0, 0.0, -g_mag, 0.0, 0.0, 0.0])
    hover_cmds_lofi = B0_inv @ desired_accel

    with open(hifi_config_file, "r") as f:
        hifi_params = yaml.load(f)
    c2 = hifi_params.prop.poly_thrust[0]
    lofi_thrust = T_max * hover_cmds_lofi
    hover_cmds_hifi = np.sign(lofi_thrust) * np.sqrt(np.abs(lofi_thrust) / c2)

    # Create motor effector
    motor_effector = MotorDynamicsEffector(
        num_motors=num_motors,
        tau_mot=TAU_MOT,
        initial_state=hover_cmds_hifi,
    )

    # Create HiFi dynamics
    hifi_dyn = ComplexMultirotor(str(hifi_config_file), effector=motor_effector)
    hifi_dyn.set_initial_conditions(
        INITIAL_POSITION,
        INITIAL_VELOCITY,
        INITIAL_ATTITUDE,
        INITIAL_ANGULAR_VELOCITY,
        REF_LAT,
        REF_LON,
        TERRAIN_ALT,
        ned_mag,
    )
    hifi_dyn.vehicle.takenoff = True

    # Create INDI controller
    K = np.diag([K_VEL, K_VEL, K_VEL, K_OMEGA, K_OMEGA, K_OMEGA])
    indi_ctrl = INDI(omit_A=True)
    indi_ctrl.set_state_model(dt=DT, K=K, B0=B0)

    # Get aerodynamic parameters for wind
    cd = hifi_dyn.vehicle.params.aero.cd
    frontal_area = np.array(hifi_dyn.vehicle.params.geo.front_area_m2)

    print("  Setup complete!")

    # ========================================================================
    # Initialize storage for training data
    # ========================================================================
    num_steps = int(TOTAL_TIME / DT)
    num_saved = num_steps // SAVE_SUBSAMPLE

    # NN inputs: [v_b(3), omega_b(3), v_dot_b(3), omega_dot_b(3), quat(4), u_prev(8)] = 24 dims
    # NOTE: quaternion added for attitude-dependent effects (wind in body frame, CG offset moments)
    nn_inputs = np.zeros((num_saved, 3 + 3 + 3 + 3 + 4 + num_motors))

    # NN targets: residual [delta_v_dot(3), delta_omega_dot(3)] = 6 dims
    nn_targets = np.zeros((num_saved, 6))

    # Additional info for analysis
    time_saved = np.zeros(num_saved)
    phase_saved = np.zeros(num_saved, dtype=int)
    ref_saved = np.zeros((num_saved, 6))  # velocity + angular rate references

    # For plotting/validation
    vel_error_hist = []
    omega_error_hist = []

    # ========================================================================
    # Filter coefficients
    # ========================================================================
    alpha_vel = DT / (DT + 1.0 / (2.0 * np.pi * FC_VEL))
    alpha_accel = DT / (DT + 1.0 / (2.0 * np.pi * FC_ACCEL))
    alpha_omega = DT / (DT + 1.0 / (2.0 * np.pi * FC_OMEGA))
    alpha_alpha = DT / (DT + 1.0 / (2.0 * np.pi * FC_ALPHA))

    # ========================================================================
    # Initialize state variables
    # ========================================================================
    cur_state = hifi_dyn.vehicle.state.copy()
    cur_input = hover_cmds_hifi.copy()
    u_prev = hover_cmds_hifi.copy()  # u^[k-1]

    # Filter states
    vel_filt = np.zeros(3)
    accel_filt = np.zeros(3)
    omega_filt = np.zeros(3)
    alpha_filt = np.zeros(3)
    omega_prev_for_diff = np.zeros(3)

    # Previous filtered x_dot for residual computation
    x_dot_prev = np.zeros(6)

    # Phase 5 reference generator
    phase5_gen = Phase5ReferenceGenerator(seed=123)

    # ========================================================================
    # Main simulation loop
    # ========================================================================
    print(f"\nRunning {TOTAL_TIME/60:.1f} minute simulation...")
    print("  Phase 1 (0-2 min): Translational velocity tracking")
    print("  Phase 2 (2-4 min): Angular rate tracking")
    print("  Phase 3 (4-6 min): Combined velocity + angular rate")
    print("  Phase 4 (6-8 min): Aggressive mixed maneuvers")
    print("  Phase 5 (8-10 min): Random step references")
    print()

    phase_starts = np.cumsum([0] + PHASE_DURATIONS[:-1])
    save_idx = 0

    for ii in range(num_steps):
        tt = ii * DT

        # Progress reporting
        if ii % (num_steps // 20) == 0:
            pct = 100 * ii / num_steps
            print(f"  Progress: {pct:5.1f}% (t = {tt:.1f}s)")

        # Determine current phase
        if tt < phase_starts[1]:
            phase = 1
        elif tt < phase_starts[2]:
            phase = 2
        elif tt < phase_starts[3]:
            phase = 3
        elif tt < phase_starts[4]:
            phase = 4
        else:
            phase = 5

        # ====================================================================
        # Get current true state
        # ====================================================================
        body_vel_true = cur_state[v_smap_quat.body_vel].flatten()
        body_accel_true = cur_state[v_smap_quat.body_accel].flatten()
        body_omega_true = cur_state[v_smap_quat.body_rot_rate].flatten()
        quat = cur_state[v_smap_quat.quat].flatten()
        ned_vel = cur_state[v_smap_quat.ned_vel].flatten()

        # ====================================================================
        # Compute wind
        # ====================================================================
        wind_gust = WIND_GUST_AMP * np.sin(2.0 * np.pi * WIND_GUST_FREQ * tt)
        wind_total = WIND_VELOCITY + wind_gust
        wind_force_body = compute_wind_force(
            wind_total, ned_vel, quat, cd, frontal_area
        )

        # ====================================================================
        # Sensor model (noisy measurements)
        # ====================================================================
        vel_noise = np.random.normal(0, SIGMA_VEL, 3)
        body_vel_meas = body_vel_true + vel_noise
        vel_filt = alpha_vel * body_vel_meas + (1.0 - alpha_vel) * vel_filt

        accel_noise = np.random.normal(0, SIGMA_ACCEL, 3)
        body_accel_meas = body_accel_true + accel_noise
        accel_filt = alpha_accel * body_accel_meas + (1.0 - alpha_accel) * accel_filt

        omega_noise = np.random.normal(0, SIGMA_OMEGA, 3)
        body_omega_meas = body_omega_true + omega_noise
        omega_filt = alpha_omega * body_omega_meas + (1.0 - alpha_omega) * omega_filt

        # Angular acceleration via numerical differentiation
        if ii == 0:
            alpha_meas = np.zeros(3)
        else:
            alpha_meas = (omega_filt - omega_prev_for_diff) / DT
        alpha_filt = alpha_alpha * alpha_meas + (1.0 - alpha_alpha) * alpha_filt
        omega_prev_for_diff = omega_filt.copy()

        # ====================================================================
        # Assemble filtered state and derivatives
        # ====================================================================
        x_filt = np.concatenate([vel_filt, omega_filt])
        x_dot_filt = np.concatenate([accel_filt, alpha_filt])

        # ====================================================================
        # Generate reference based on phase
        # ====================================================================
        if phase == 1:
            vb_ref, vb_ref_dot, omega_ref, omega_ref_dot = generate_phase1_reference(
                tt, phase_starts[0]
            )
        elif phase == 2:
            vb_ref, vb_ref_dot, omega_ref, omega_ref_dot = generate_phase2_reference(
                tt, phase_starts[1]
            )
        elif phase == 3:
            vb_ref, vb_ref_dot, omega_ref, omega_ref_dot = generate_phase3_reference(
                tt, phase_starts[2]
            )
        elif phase == 4:
            vb_ref, vb_ref_dot, omega_ref, omega_ref_dot = generate_phase4_reference(
                tt, phase_starts[3]
            )
        else:
            vb_ref, vb_ref_dot, omega_ref, omega_ref_dot = phase5_gen.generate(
                tt, phase_starts[4]
            )

        ref = np.concatenate([vb_ref, omega_ref])
        ref_dot = np.concatenate([vb_ref_dot, omega_ref_dot])

        # ====================================================================
        # INDI control
        # ====================================================================
        u_cmd = indi_ctrl.calculate_control(
            cur_time=tt,
            cur_state=x_filt,
            cur_state_dot=x_dot_filt,
            cur_input=cur_input,
            ref=ref,
            ref_dot=ref_dot,
        )
        u_cmd = np.clip(u_cmd, -1.0, 1.0)

        # ====================================================================
        # Compute and store training data (at save rate)
        # ====================================================================
        if ii > 0 and ii % SAVE_SUBSAMPLE == 0:
            # Compute residual: x_dot^[k] - (x_dot^[k-1] + B0 * delta_u)
            # where delta_u = u^[k-1] - u^[k-2]
            # Note: we use filtered measurements as per literature approach

            delta_u = cur_input - u_prev  # u^[k-1] - u^[k-2]
            x_dot_predicted = x_dot_prev + B0 @ delta_u
            residual = x_dot_filt - x_dot_predicted

            # NN input: [v_b, omega_b, v_dot_b, omega_dot_b, quat, u_prev]
            nn_input = np.concatenate(
                [
                    vel_filt,  # v_b (3)
                    omega_filt,  # omega_b (3)
                    accel_filt,  # v_dot_b (3)
                    alpha_filt,  # omega_dot_b (3)
                    quat,  # quaternion (4) - for attitude-dependent effects
                    cur_input,  # u^[k-1] (8)
                ]
            )

            nn_inputs[save_idx] = nn_input
            nn_targets[save_idx] = residual
            time_saved[save_idx] = tt
            phase_saved[save_idx] = phase
            ref_saved[save_idx] = ref

            save_idx += 1

        # Track errors for validation plots
        if ii % 100 == 0:  # Every 0.1 seconds
            vel_error_hist.append(np.linalg.norm(vel_filt - vb_ref))
            omega_error_hist.append(np.linalg.norm(omega_filt - omega_ref))

        # ====================================================================
        # Store previous values for next iteration
        # ====================================================================
        x_dot_prev = x_dot_filt.copy()
        u_prev = cur_input.copy()

        # ====================================================================
        # Propagate dynamics
        # ====================================================================
        next_state = hifi_dyn.propagate_state(DT, cur_state, u_cmd).flatten()

        # Add wind disturbance
        wind_accel_body = wind_force_body / mass
        next_state[v_smap_quat.body_vel] += wind_accel_body * DT

        cur_state = next_state
        cur_input = motor_effector.state.copy()

    print(f"  Progress: 100.0% (t = {TOTAL_TIME:.1f}s)")
    print("Simulation complete!")

    # ========================================================================
    # Trim data to actual saved samples (skip first few due to filter warmup)
    # ========================================================================
    warmup_samples = 100  # Skip first 100 samples (1 second at 100 Hz save rate)
    nn_inputs = nn_inputs[warmup_samples:save_idx]
    nn_targets = nn_targets[warmup_samples:save_idx]
    time_saved = time_saved[warmup_samples:save_idx]
    phase_saved = phase_saved[warmup_samples:save_idx]
    ref_saved = ref_saved[warmup_samples:save_idx]

    print(f"\nData saved: {len(nn_inputs)} samples")
    print(f"  NN input dimension: {nn_inputs.shape[1]}")
    print(f"  NN target dimension: {nn_targets.shape[1]}")

    # ========================================================================
    # Save to CSV
    # ========================================================================
    output_dir = Path(__file__).parent / "training_data"
    output_dir.mkdir(exist_ok=True)

    # Column names for NN inputs
    input_cols = (
        ["vb_x", "vb_y", "vb_z"]
        + ["omega_x", "omega_y", "omega_z"]
        + ["vdot_x", "vdot_y", "vdot_z"]
        + ["omegadot_x", "omegadot_y", "omegadot_z"]
        + ["quat_w", "quat_x", "quat_y", "quat_z"]
        + [f"u_prev_{i}" for i in range(num_motors)]
    )

    # Column names for NN targets
    target_cols = ["residual_vdot_x", "residual_vdot_y", "residual_vdot_z"] + [
        "residual_omegadot_x",
        "residual_omegadot_y",
        "residual_omegadot_z",
    ]

    # Save inputs
    df_inputs = pd.DataFrame(nn_inputs, columns=input_cols)
    df_inputs.insert(0, "time", time_saved)
    df_inputs.insert(1, "phase", phase_saved)
    inputs_file = output_dir / "nn_inputs.csv"
    df_inputs.to_csv(inputs_file, index=False)
    print(f"\nSaved: {inputs_file}")

    # Save targets
    df_targets = pd.DataFrame(nn_targets, columns=target_cols)
    df_targets.insert(0, "time", time_saved)
    df_targets.insert(1, "phase", phase_saved)
    targets_file = output_dir / "nn_targets.csv"
    df_targets.to_csv(targets_file, index=False)
    print(f"Saved: {targets_file}")

    # Save combined (for convenience)
    df_combined = pd.concat(
        [
            df_inputs.drop(columns=["time", "phase"]),
            df_targets.drop(columns=["time", "phase"]),
        ],
        axis=1,
    )
    df_combined.insert(0, "time", time_saved)
    df_combined.insert(1, "phase", phase_saved)
    combined_file = output_dir / "nn_training_data.csv"
    df_combined.to_csv(combined_file, index=False)
    print(f"Saved: {combined_file}")

    # Save metadata
    metadata = {
        "Parameter": [
            "Total simulation time (s)",
            "Timestep (s)",
            "Save rate (Hz)",
            "Number of samples",
            "NN input dimension",
            "NN target dimension",
            "Number of motors",
            "K_VEL",
            "K_OMEGA",
            "SIGMA_VEL (m/s)",
            "SIGMA_ACCEL (m/s^2)",
            "SIGMA_OMEGA (rad/s)",
            "FC_VEL (Hz)",
            "FC_ACCEL (Hz)",
            "FC_OMEGA (Hz)",
            "FC_ALPHA (Hz)",
            "Wind velocity N (m/s)",
            "Wind velocity E (m/s)",
            "Wind velocity D (m/s)",
            "Wind gust amplitude N (m/s)",
            "Wind gust amplitude E (m/s)",
            "Wind gust amplitude D (m/s)",
            "Random seed",
            "Generation timestamp",
        ],
        "Value": [
            TOTAL_TIME,
            DT,
            1.0 / (DT * SAVE_SUBSAMPLE),
            len(nn_inputs),
            nn_inputs.shape[1],
            nn_targets.shape[1],
            num_motors,
            K_VEL,
            K_OMEGA,
            SIGMA_VEL,
            SIGMA_ACCEL,
            SIGMA_OMEGA,
            FC_VEL,
            FC_ACCEL,
            FC_OMEGA,
            FC_ALPHA,
            WIND_VELOCITY[0],
            WIND_VELOCITY[1],
            WIND_VELOCITY[2],
            WIND_GUST_AMP[0],
            WIND_GUST_AMP[1],
            WIND_GUST_AMP[2],
            42,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        ],
    }
    df_metadata = pd.DataFrame(metadata)
    metadata_file = output_dir / "metadata.csv"
    df_metadata.to_csv(metadata_file, index=False)
    print(f"Saved: {metadata_file}")

    # ========================================================================
    # Compute and print statistics
    # ========================================================================
    print("\n" + "=" * 70)
    print("Data Statistics")
    print("=" * 70)

    print("\nNN Inputs (filtered sensor measurements):")
    for i, col in enumerate(input_cols):
        print(
            f"  {col:20s}: mean={nn_inputs[:, i].mean():8.4f}, std={nn_inputs[:, i].std():8.4f}, "
            f"min={nn_inputs[:, i].min():8.4f}, max={nn_inputs[:, i].max():8.4f}"
        )

    print("\nNN Targets (residuals):")
    for i, col in enumerate(target_cols):
        print(
            f"  {col:25s}: mean={nn_targets[:, i].mean():8.4f}, std={nn_targets[:, i].std():8.4f}, "
            f"min={nn_targets[:, i].min():8.4f}, max={nn_targets[:, i].max():8.4f}"
        )

    print("\nSamples per phase:")
    for p in range(1, 6):
        count = np.sum(phase_saved == p)
        print(f"  Phase {p}: {count} samples ({100*count/len(phase_saved):.1f}%)")

    # ========================================================================
    # Validation plots
    # ========================================================================
    print("\nGenerating validation plots...")

    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle(
        "INDI Training Data Generation - Validation", fontsize=14, fontweight="bold"
    )

    # Plot 1: Residual magnitude over time
    ax = axes[0, 0]
    residual_mag = np.linalg.norm(nn_targets, axis=1)
    ax.plot(time_saved, residual_mag, "b-", alpha=0.5, linewidth=0.5)
    ax.axhline(
        residual_mag.mean(),
        color="r",
        linestyle="--",
        label=f"Mean: {residual_mag.mean():.4f}",
    )
    for i, ps in enumerate(phase_starts):
        ax.axvline(ps, color="k", linestyle=":", alpha=0.5)
        ax.text(ps + 5, ax.get_ylim()[1] * 0.9, f"P{i+1}", fontsize=8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Residual Magnitude")
    ax.set_title("Residual ||Δẋ|| Over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Residual histogram
    ax = axes[0, 1]
    ax.hist(residual_mag, bins=50, density=True, alpha=0.7, color="blue")
    ax.axvline(
        residual_mag.mean(),
        color="r",
        linestyle="--",
        label=f"Mean: {residual_mag.mean():.4f}",
    )
    ax.axvline(
        residual_mag.mean() + residual_mag.std(),
        color="orange",
        linestyle="--",
        label=f"Std: {residual_mag.std():.4f}",
    )
    ax.set_xlabel("Residual Magnitude")
    ax.set_ylabel("Density")
    ax.set_title("Residual Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Velocity components over time (subsampled for clarity)
    ax = axes[1, 0]
    subsample = 10
    ax.plot(
        time_saved[::subsample],
        nn_inputs[::subsample, 0],
        "r-",
        alpha=0.7,
        label="vb_x",
    )
    ax.plot(
        time_saved[::subsample],
        nn_inputs[::subsample, 1],
        "g-",
        alpha=0.7,
        label="vb_y",
    )
    ax.plot(
        time_saved[::subsample],
        nn_inputs[::subsample, 2],
        "b-",
        alpha=0.7,
        label="vb_z",
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Body Velocity (m/s)")
    ax.set_title("Body Velocity Components")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Angular rate components over time
    ax = axes[1, 1]
    ax.plot(
        time_saved[::subsample],
        np.rad2deg(nn_inputs[::subsample, 3]),
        "r-",
        alpha=0.7,
        label="ω_x",
    )
    ax.plot(
        time_saved[::subsample],
        np.rad2deg(nn_inputs[::subsample, 4]),
        "g-",
        alpha=0.7,
        label="ω_y",
    )
    ax.plot(
        time_saved[::subsample],
        np.rad2deg(nn_inputs[::subsample, 5]),
        "b-",
        alpha=0.7,
        label="ω_z",
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Body Angular Rate (deg/s)")
    ax.set_title("Body Angular Rate Components")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 5: Tracking errors over time
    ax = axes[2, 0]
    time_errors = np.arange(len(vel_error_hist)) * 0.1
    ax.plot(time_errors, vel_error_hist, "b-", alpha=0.7, label="Velocity Error")
    ax.plot(time_errors, omega_error_hist, "r-", alpha=0.7, label="Angular Rate Error")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Tracking Error")
    ax.set_title("Reference Tracking Errors")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 6: Motor commands distribution
    ax = axes[2, 1]
    motor_data = nn_inputs[:, 12:20]  # u_prev columns
    bp = ax.boxplot(
        [motor_data[:, i] for i in range(num_motors)],
        labels=[f"M{i}" for i in range(num_motors)],
    )
    ax.axhline(0, color="k", linestyle="--", alpha=0.3)
    ax.set_xlabel("Motor")
    ax.set_ylabel("Command")
    ax.set_title("Motor Command Distribution")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    plot_file = output_dir / "data_validation.png"
    plt.savefig(plot_file, dpi=150)
    print(f"Saved: {plot_file}")

    # ========================================================================
    # Final summary
    # ========================================================================
    print("\n" + "=" * 70)
    print("Data Generation Complete!")
    print("=" * 70)
    print(f"\nOutput directory: {output_dir}")
    print(f"Files generated:")
    print(f"  - nn_inputs.csv ({nn_inputs.shape[0]} x {nn_inputs.shape[1]})")
    print(f"  - nn_targets.csv ({nn_targets.shape[0]} x {nn_targets.shape[1]})")
    print(f"  - nn_training_data.csv (combined)")
    print(f"  - metadata.csv (parameters)")
    print(f"  - data_validation.png (plots)")

    print(f"\nFor training:")
    print(f"  - Input features: {nn_inputs.shape[1]} dimensions")
    print(f"  - Output targets: {nn_targets.shape[1]} dimensions")
    print(f"  - Total samples: {nn_inputs.shape[0]}")

    # Sanity check: residual should be small on average if INDI is working
    avg_residual = residual_mag.mean()
    if avg_residual < 5.0:
        print(f"\n✓ Residual magnitude looks reasonable (avg: {avg_residual:.4f})")
    else:
        print(
            f"\n⚠ Warning: Large average residual ({avg_residual:.4f}) - check simulation!"
        )

    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
