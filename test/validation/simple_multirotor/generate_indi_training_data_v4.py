"""Generate training data for structured B-learning (V4).

This implements the structured η-learning approach:
- NN outputs η_F ∈ R^8 (force column scalings) and η_M ∈ R^8 (moment column scalings)
- B_hat = [B0_F @ diag(η_F); B0_M @ diag(η_M)]

Training Target Generation (Option A - finite differences):
1. For each (x, u) sample, compute B_true via central differences on ẏ
2. Project B_true columns onto nominal B0 columns to get η labels:
   - η_F,i = (B_true[0:3,i]^T @ B0_F[:,i]) / (||B0_F[:,i]||² + λ)
   - η_M,i = (B_true[3:6,i]^T @ B0_M[:,i]) / (||B0_M[:,i]||² + λ)

Key features:
- Motor dither phases for independent motor excitation
- Quaternion canonicalization (q_w >= 0)
- No filtering by Δu - we compute B_true explicitly

NN Input z ∈ R^17:
- v_B (3): body velocity
- ω_B (3): body angular rate
- g_B (3): gravity vector in body frame = R(q)^T @ g (captures attitude effects)
- u (8): applied motor inputs (effector state)

NN Output:
- η_F (8): force effectiveness scalings per motor
- η_M (8): moment effectiveness scalings per motor
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
# Motor dynamics effector
# ============================================================================
class MotorDynamicsEffector(Effector):
    """First-order motor dynamics with per-motor efficiency variation."""

    MOTOR_EFFICIENCY = np.array([0.99, 1.01, 0.98, 1.02, 1.01, 0.99, 1.00, 0.98])

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
        scaled_cmds = input_cmds * self.MOTOR_EFFICIENCY[: self.num_motors]
        alpha = np.exp(-dt / self.tau_mot)
        self.state = scaled_cmds + (self.state - scaled_cmds) * alpha
        return self.state.copy()


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


def canonicalize_quat(q):
    """Ensure q_w >= 0 for continuity."""
    if q[0] < 0:
        return -q
    return q.copy()


def compute_gravity_body(quat, g_ned=np.array([0, 0, 9.81])):
    """Compute gravity vector in body frame: g_B = R(q)^T @ g_NED.

    This is a better physics feature than raw quaternion because:
    - Directly captures attitude-dependent effects on thrust requirements
    - Smooth and continuous (no quaternion flip issues)
    - 3 dims instead of 4
    """
    dcm_ned_to_body = gmath.quat_to_dcm(quat).T
    return dcm_ned_to_body @ g_ned


# ============================================================================
# Reference generators (reused from V1)
# ============================================================================
def generate_phase1_reference(t, phase_start):
    """Translational velocity tracking."""
    t_rel = t - phase_start
    period = 10.0
    omega = 2.0 * np.pi / period
    amp_x = 1.0 + 0.3 * np.sin(0.1 * t_rel)
    amp_y = 0.8 + 0.2 * np.cos(0.15 * t_rel)
    vb_ref = np.array(
        [
            amp_x * np.sin(omega * t_rel),
            amp_y * np.sin(2 * omega * t_rel),
            0.3 * np.sin(0.5 * omega * t_rel),
        ]
    )
    vb_ref_dot = np.array(
        [
            amp_x * omega * np.cos(omega * t_rel),
            amp_y * 2 * omega * np.cos(2 * omega * t_rel),
            0.3 * 0.5 * omega * np.cos(0.5 * omega * t_rel),
        ]
    )
    return vb_ref, vb_ref_dot, np.zeros(3), np.zeros(3)


def generate_phase2_reference(t, phase_start):
    """Angular rate tracking."""
    t_rel = t - phase_start
    period = 8.0
    omega = 2.0 * np.pi / period
    omega_ref = np.array(
        [
            0.5 * np.sin(omega * t_rel),
            0.4 * np.sin(omega * t_rel + np.pi / 3),
            0.3 * np.sin(0.5 * omega * t_rel),
        ]
    )
    omega_ref_dot = np.array(
        [
            0.5 * omega * np.cos(omega * t_rel),
            0.4 * omega * np.cos(omega * t_rel + np.pi / 3),
            0.3 * 0.5 * omega * np.cos(0.5 * omega * t_rel),
        ]
    )
    return np.zeros(3), np.zeros(3), omega_ref, omega_ref_dot


def generate_phase3_reference(t, phase_start):
    """Combined velocity + angular rate."""
    t_rel = t - phase_start
    omega1 = 2.0 * np.pi / 12.0
    omega2 = 2.0 * np.pi / 6.0
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


def generate_dither_reference(t, phase_start, rng, num_motors):
    """Phase with motor dither for independent excitation.

    Adds random perturbations to motor commands while maintaining hover.
    """
    t_rel = t - phase_start

    # Base hover with slow sinusoidal velocity
    omega = 2.0 * np.pi / 15.0
    vb_ref = np.array(
        [
            0.3 * np.sin(omega * t_rel),
            0.3 * np.cos(omega * t_rel),
            0.1 * np.sin(2 * omega * t_rel),
        ]
    )
    vb_ref_dot = np.array(
        [
            0.3 * omega * np.cos(omega * t_rel),
            -0.3 * omega * np.sin(omega * t_rel),
            0.1 * 2 * omega * np.cos(2 * omega * t_rel),
        ]
    )
    omega_ref = np.array(
        [
            0.2 * np.sin(0.5 * omega * t_rel),
            0.15 * np.cos(0.5 * omega * t_rel),
            0.1 * np.sin(omega * t_rel),
        ]
    )
    omega_ref_dot = np.zeros(3)

    return vb_ref, vb_ref_dot, omega_ref, omega_ref_dot


class Phase5ReferenceGenerator:
    """Random step changes."""

    def __init__(self, seed=123):
        self.rng = np.random.default_rng(seed)
        self.current_vb_ref = np.zeros(3)
        self.current_omega_ref = np.zeros(3)
        self.next_change_time = 0.0
        self.hold_time_range = (1.0, 3.0)

    def generate(self, t, phase_start):
        t_rel = t - phase_start
        if t_rel >= self.next_change_time:
            self.current_vb_ref = self.rng.uniform(-1.0, 1.0, 3)
            self.current_vb_ref[2] *= 0.5
            self.current_omega_ref = self.rng.uniform(-0.5, 0.5, 3)
            hold_time = self.rng.uniform(*self.hold_time_range)
            self.next_change_time = t_rel + hold_time
        return (
            self.current_vb_ref.copy(),
            np.zeros(3),
            self.current_omega_ref.copy(),
            np.zeros(3),
        )


# ============================================================================
# Configuration
# ============================================================================
DT = 0.001
SAVE_SUBSAMPLE = 10  # Save at 100 Hz
TAU_MOT = 0.032

INITIAL_POSITION = np.array([0.0, 0.0, -10.0])
INITIAL_VELOCITY = np.array([0.0, 0.0, 0.0])
INITIAL_ATTITUDE = np.array([0.0, 0.0, 0.0])
INITIAL_ANGULAR_VELOCITY = np.array([0.0, 0.0, 0.0])
REF_LAT, REF_LON, TERRAIN_ALT = 34.0, -86.0, 0.0

K_VEL = 5.0
K_OMEGA = 10.0

SIGMA_VEL = 0.05
SIGMA_ACCEL = 0.1
SIGMA_OMEGA = 0.01

FC_VEL = 5.0
FC_ACCEL = 20.0
FC_OMEGA = 20.0
FC_ALPHA = 5.0

WIND_VELOCITY = np.array([8.0, 5.0, 1.0])
WIND_GUST_AMP = np.array([4.0, 3.0, 1.0])
WIND_GUST_FREQ = np.array([0.3, 0.4, 0.5])

# Phase durations - include dither phases for motor excitation
PHASE_DURATIONS = [90, 90, 90, 90, 90, 150]  # Last phase is dither-heavy
TOTAL_TIME = sum(PHASE_DURATIONS)

# Finite difference epsilon for B_true computation
FD_EPSILON = 0.01

# Regularization for projection
LAMBDA_REG = 1e-6

# Dither amplitude for motor excitation
DITHER_AMP = 0.05

np.random.seed(42)


def compute_y_dot(
    hifi_dyn, state, motor_cmds, motor_effector, wind_force_body, mass, dt_prop=0.001
):
    """Compute ẏ = [a_B; α_B] for given state and motor commands.

    Does a single propagation step to get accelerations.
    """
    # Save current effector state
    saved_effector = motor_effector.state.copy()

    # Set effector to desired motor state (bypass dynamics for FD)
    motor_effector.state = motor_cmds.copy()

    # Propagate one step
    next_state = hifi_dyn.propagate_state(dt_prop, state, motor_cmds).flatten()

    # Add wind
    next_state[v_smap_quat.body_vel] += (wind_force_body / mass) * dt_prop

    # Extract accelerations from state
    body_accel = next_state[v_smap_quat.body_accel].flatten()

    # Angular acceleration via velocity difference
    omega_cur = state[v_smap_quat.body_rot_rate].flatten()
    omega_next = next_state[v_smap_quat.body_rot_rate].flatten()
    alpha = (omega_next - omega_cur) / dt_prop

    # Restore effector
    motor_effector.state = saved_effector

    return np.concatenate([body_accel, alpha])


def compute_B_true(
    hifi_dyn,
    state,
    u_nominal,
    motor_effector,
    wind_force_body,
    mass,
    num_motors,
    epsilon=0.01,
):
    """Compute true B matrix via central finite differences.

    B_true[:,i] = (ẏ(u + εe_i) - ẏ(u - εe_i)) / (2ε)
    """
    B_true = np.zeros((6, num_motors))

    for i in range(num_motors):
        # Perturb motor i
        u_plus = u_nominal.copy()
        u_plus[i] += epsilon
        u_plus = np.clip(u_plus, -1.0, 1.0)

        u_minus = u_nominal.copy()
        u_minus[i] -= epsilon
        u_minus = np.clip(u_minus, -1.0, 1.0)

        # Compute ẏ at perturbed points
        y_dot_plus = compute_y_dot(
            hifi_dyn, state, u_plus, motor_effector, wind_force_body, mass
        )
        y_dot_minus = compute_y_dot(
            hifi_dyn, state, u_minus, motor_effector, wind_force_body, mass
        )

        # Central difference
        B_true[:, i] = (y_dot_plus - y_dot_minus) / (2 * epsilon)

    return B_true


def project_to_eta(B_true, B0_F, B0_M, lambda_reg=1e-6):
    """Project B_true columns onto nominal B0 columns to get η scalings.

    η_F,i = (B_true[0:3,i]^T @ B0_F[:,i]) / (||B0_F[:,i]||² + λ)
    η_M,i = (B_true[3:6,i]^T @ B0_M[:,i]) / (||B0_M[:,i]||² + λ)
    """
    num_motors = B_true.shape[1]
    eta_F = np.zeros(num_motors)
    eta_M = np.zeros(num_motors)

    for i in range(num_motors):
        # Force block (rows 0:3)
        b_true_F = B_true[0:3, i]
        b0_F = B0_F[:, i]
        eta_F[i] = np.dot(b_true_F, b0_F) / (np.dot(b0_F, b0_F) + lambda_reg)

        # Moment block (rows 3:6)
        b_true_M = B_true[3:6, i]
        b0_M = B0_M[:, i]
        eta_M[i] = np.dot(b_true_M, b0_M) / (np.dot(b0_M, b0_M) + lambda_reg)

    return eta_F, eta_M


def main():
    print("=" * 70)
    print("V4 Training Data Generation: Structured η-Learning")
    print("=" * 70)
    print(f"Target: η_F, η_M per-motor effectiveness scalings")
    print(f"Total simulation time: {TOTAL_TIME} seconds ({TOTAL_TIME/60:.1f} minutes)")
    print()

    # ========================================================================
    # Setup
    # ========================================================================
    print("Setting up simulation...")
    config_dir = Path(__file__).parent
    lofi_config_file = config_dir / "omnicopter_config.yaml"
    hifi_config_file = config_dir / "omnicopter_config_hifi.yaml"

    # Load LoFi for B0
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

    # Build B0 matrix blocks
    N = np.zeros((3, num_motors))
    P = np.zeros((3, num_motors))
    for i in range(num_motors):
        N[:, i] = lofi_dyn.vehicle.params.motor.thrust_dir[i]
        P[:, i] = lofi_dyn.vehicle.params.motor.pos_m[i]

    B0_F = (1.0 / mass) * T_max * N  # Force block (3 x 8)
    P_cross_N = np.zeros((3, num_motors))
    for i in range(num_motors):
        P_cross_N[:, i] = np.cross(P[:, i], N[:, i])
    J_inv = np.linalg.inv(inertia)
    B0_M = J_inv @ (T_max * P_cross_N)  # Moment block (3 x 8)
    B0 = np.vstack([B0_F, B0_M])
    B0_inv = np.linalg.pinv(B0)

    print(f"  B0_F shape: {B0_F.shape}")
    print(f"  B0_M shape: {B0_M.shape}")
    print(f"  Number of motors: {num_motors}")

    # Hover commands
    g_mag = gravity[2]
    hover_cmds_lofi = B0_inv @ np.array([0.0, 0.0, -g_mag, 0.0, 0.0, 0.0])

    with open(hifi_config_file, "r") as f:
        hifi_params = yaml.load(f)
    c2 = hifi_params.prop.poly_thrust[0]
    lofi_thrust = T_max * hover_cmds_lofi
    hover_cmds_hifi = np.sign(lofi_thrust) * np.sqrt(np.abs(lofi_thrust) / c2)

    # Create motor effector
    motor_effector = MotorDynamicsEffector(
        num_motors=num_motors, tau_mot=TAU_MOT, initial_state=hover_cmds_hifi
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

    # Aerodynamic parameters
    cd = hifi_dyn.vehicle.params.aero.cd
    frontal_area = np.array(hifi_dyn.vehicle.params.geo.front_area_m2)

    print("  Setup complete!")

    # ========================================================================
    # Initialize storage
    # ========================================================================
    num_steps = int(TOTAL_TIME / DT)
    num_saved = num_steps // SAVE_SUBSAMPLE

    # NN inputs: z = [v_B(3), ω_B(3), g_B(3), u(8)] = 17 dims
    nn_inputs = np.zeros((num_saved, 17))

    # NN targets: η_F(8) + η_M(8) = 16 dims
    nn_targets = np.zeros((num_saved, 16))

    time_saved = np.zeros(num_saved)
    phase_saved = np.zeros(num_saved, dtype=int)

    # Filter coefficients
    alpha_vel = DT / (DT + 1.0 / (2.0 * np.pi * FC_VEL))
    alpha_accel = DT / (DT + 1.0 / (2.0 * np.pi * FC_ACCEL))
    alpha_omega = DT / (DT + 1.0 / (2.0 * np.pi * FC_OMEGA))
    alpha_alpha = DT / (DT + 1.0 / (2.0 * np.pi * FC_ALPHA))

    # State variables
    cur_state = hifi_dyn.vehicle.state.copy()
    cur_input = hover_cmds_hifi.copy()

    vel_filt = np.zeros(3)
    accel_filt = np.zeros(3)
    omega_filt = np.zeros(3)
    alpha_filt = np.zeros(3)
    omega_prev_for_diff = np.zeros(3)

    phase5_gen = Phase5ReferenceGenerator(seed=123)
    dither_rng = np.random.default_rng(456)

    # ========================================================================
    # Main simulation loop
    # ========================================================================
    print(f"\nRunning {TOTAL_TIME/60:.1f} minute simulation...")
    print("  Computing B_true via finite differences at each save step...")
    print()

    phase_starts = np.cumsum([0] + PHASE_DURATIONS[:-1])
    save_idx = 0

    for ii in range(num_steps):
        tt = ii * DT

        if ii % (num_steps // 20) == 0:
            pct = 100 * ii / num_steps
            print(f"  Progress: {pct:5.1f}% (t = {tt:.1f}s)")

        # Determine phase
        if tt < phase_starts[1]:
            phase = 1
        elif tt < phase_starts[2]:
            phase = 2
        elif tt < phase_starts[3]:
            phase = 3
        elif tt < phase_starts[4]:
            phase = 4
        elif tt < phase_starts[5]:
            phase = 5
        else:
            phase = 6  # Dither phase

        # Get current state
        body_vel_true = cur_state[v_smap_quat.body_vel].flatten()
        body_accel_true = cur_state[v_smap_quat.body_accel].flatten()
        body_omega_true = cur_state[v_smap_quat.body_rot_rate].flatten()
        quat = cur_state[v_smap_quat.quat].flatten()
        ned_vel = cur_state[v_smap_quat.ned_vel].flatten()

        # Wind
        wind_gust = WIND_GUST_AMP * np.sin(2.0 * np.pi * WIND_GUST_FREQ * tt)
        wind_total = WIND_VELOCITY + wind_gust
        wind_force_body = compute_wind_force(
            wind_total, ned_vel, quat, cd, frontal_area
        )

        # Sensor model with filtering
        vel_noise = np.random.normal(0, SIGMA_VEL, 3)
        body_vel_meas = body_vel_true + vel_noise
        vel_filt = alpha_vel * body_vel_meas + (1.0 - alpha_vel) * vel_filt

        accel_noise = np.random.normal(0, SIGMA_ACCEL, 3)
        body_accel_meas = body_accel_true + accel_noise
        accel_filt = alpha_accel * body_accel_meas + (1.0 - alpha_accel) * accel_filt

        omega_noise = np.random.normal(0, SIGMA_OMEGA, 3)
        body_omega_meas = body_omega_true + omega_noise
        omega_filt = alpha_omega * body_omega_meas + (1.0 - alpha_omega) * omega_filt

        if ii == 0:
            alpha_meas = np.zeros(3)
        else:
            alpha_meas = (omega_filt - omega_prev_for_diff) / DT
        alpha_filt = alpha_alpha * alpha_meas + (1.0 - alpha_alpha) * alpha_filt
        omega_prev_for_diff = omega_filt.copy()

        x_filt = np.concatenate([vel_filt, omega_filt])
        x_dot_filt = np.concatenate([accel_filt, alpha_filt])

        # Reference
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
            vb_ref, vb_ref_dot, omega_ref, omega_ref_dot = generate_phase3_reference(
                tt, phase_starts[3]
            )
        elif phase == 5:
            vb_ref, vb_ref_dot, omega_ref, omega_ref_dot = phase5_gen.generate(
                tt, phase_starts[4]
            )
        else:
            vb_ref, vb_ref_dot, omega_ref, omega_ref_dot = generate_dither_reference(
                tt, phase_starts[5], dither_rng, num_motors
            )

        ref = np.concatenate([vb_ref, omega_ref])
        ref_dot = np.concatenate([vb_ref_dot, omega_ref_dot])

        # INDI control
        u_cmd = indi_ctrl.calculate_control(
            cur_time=tt,
            cur_state=x_filt,
            cur_state_dot=x_dot_filt,
            cur_input=cur_input,
            ref=ref,
            ref_dot=ref_dot,
        )

        # Add dither in phase 6 for motor excitation
        if phase == 6:
            dither = dither_rng.uniform(-DITHER_AMP, DITHER_AMP, num_motors)
            u_cmd = u_cmd + dither

        u_cmd = np.clip(u_cmd, -1.0, 1.0)

        # ====================================================================
        # Compute and store training data (at save rate)
        # ====================================================================
        if ii > 0 and ii % SAVE_SUBSAMPLE == 0:
            # Compute B_true via finite differences
            B_true = compute_B_true(
                hifi_dyn,
                cur_state,
                cur_input,
                motor_effector,
                wind_force_body,
                mass,
                num_motors,
                epsilon=FD_EPSILON,
            )

            # Project to get η labels
            eta_F, eta_M = project_to_eta(B_true, B0_F, B0_M, lambda_reg=LAMBDA_REG)

            # Compute gravity in body frame (better physics feature than raw quat)
            g_body = compute_gravity_body(quat)

            # NN input: z = [v_B, ω_B, g_B, u]
            nn_input = np.concatenate(
                [
                    vel_filt,  # v_B (3)
                    omega_filt,  # ω_B (3)
                    g_body,  # g_B (3) - gravity in body frame
                    cur_input,  # u (8) - applied motor state
                ]
            )

            nn_inputs[save_idx] = nn_input
            nn_targets[save_idx] = np.concatenate([eta_F, eta_M])
            time_saved[save_idx] = tt
            phase_saved[save_idx] = phase
            save_idx += 1

        # Propagate dynamics
        next_state = hifi_dyn.propagate_state(DT, cur_state, u_cmd).flatten()
        wind_accel_body = wind_force_body / mass
        next_state[v_smap_quat.body_vel] += wind_accel_body * DT
        cur_state = next_state
        cur_input = motor_effector.state.copy()

    print(f"  Progress: 100.0% (t = {TOTAL_TIME:.1f}s)")
    print("Simulation complete!")

    # ========================================================================
    # Trim and save
    # ========================================================================
    warmup_samples = 100
    nn_inputs = nn_inputs[warmup_samples:save_idx]
    nn_targets = nn_targets[warmup_samples:save_idx]
    time_saved = time_saved[warmup_samples:save_idx]
    phase_saved = phase_saved[warmup_samples:save_idx]

    print(f"\nData saved: {len(nn_inputs)} samples")
    print(f"  NN input dimension: {nn_inputs.shape[1]}")
    print(f"  NN target dimension: {nn_targets.shape[1]}")

    # ========================================================================
    # Save to CSV
    # ========================================================================
    output_dir = Path(__file__).parent / "training_data_v4"
    output_dir.mkdir(exist_ok=True)

    input_cols = (
        ["vb_x", "vb_y", "vb_z"]
        + ["omega_x", "omega_y", "omega_z"]
        + ["g_body_x", "g_body_y", "g_body_z"]
        + [f"u_{i}" for i in range(num_motors)]
    )
    target_cols = [f"eta_F_{i}" for i in range(num_motors)] + [
        f"eta_M_{i}" for i in range(num_motors)
    ]

    df_inputs = pd.DataFrame(nn_inputs, columns=input_cols)
    df_inputs.insert(0, "time", time_saved)
    df_inputs.insert(1, "phase", phase_saved)
    df_inputs.to_csv(output_dir / "nn_inputs.csv", index=False)
    print(f"\nSaved: {output_dir / 'nn_inputs.csv'}")

    df_targets = pd.DataFrame(nn_targets, columns=target_cols)
    df_targets.insert(0, "time", time_saved)
    df_targets.insert(1, "phase", phase_saved)
    df_targets.to_csv(output_dir / "nn_targets.csv", index=False)
    print(f"Saved: {output_dir / 'nn_targets.csv'}")

    # Save B0 blocks
    np.save(output_dir / "B0_F.npy", B0_F)
    np.save(output_dir / "B0_M.npy", B0_M)
    print(f"Saved: {output_dir / 'B0_F.npy'}, {output_dir / 'B0_M.npy'}")

    # ========================================================================
    # Statistics
    # ========================================================================
    print("\n" + "=" * 70)
    print("DATA STATISTICS")
    print("=" * 70)

    eta_F_data = nn_targets[:, :num_motors]
    eta_M_data = nn_targets[:, num_motors:]

    print("\nη_F (force scalings) statistics:")
    for i in range(num_motors):
        print(
            f"  η_F_{i}: mean={eta_F_data[:, i].mean():.4f}, "
            f"std={eta_F_data[:, i].std():.4f}, "
            f"min={eta_F_data[:, i].min():.4f}, max={eta_F_data[:, i].max():.4f}"
        )

    print("\nη_M (moment scalings) statistics:")
    for i in range(num_motors):
        print(
            f"  η_M_{i}: mean={eta_M_data[:, i].mean():.4f}, "
            f"std={eta_M_data[:, i].std():.4f}, "
            f"min={eta_M_data[:, i].min():.4f}, max={eta_M_data[:, i].max():.4f}"
        )

    print(
        f"\nExpected η near motor efficiency: {MotorDynamicsEffector.MOTOR_EFFICIENCY}"
    )

    # ========================================================================
    # Plot
    # ========================================================================
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle("V4 Training Data: Per-Motor η Scalings", fontweight="bold")

    for i in range(num_motors):
        ax = axes[0, i] if i < 4 else axes[1, i - 4]
        ax.hist(eta_F_data[:, i], bins=50, alpha=0.6, label=f"η_F_{i}", density=True)
        ax.hist(eta_M_data[:, i], bins=50, alpha=0.6, label=f"η_M_{i}", density=True)
        ax.axvline(1.0, color="k", ls="--", label="Nominal")
        ax.axvline(
            MotorDynamicsEffector.MOTOR_EFFICIENCY[i],
            color="r",
            ls=":",
            label="True eff",
        )
        ax.set_xlabel(f"Motor {i} η")
        ax.set_ylabel("Density")
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "eta_distributions.png", dpi=150)
    print(f"\nPlot saved: {output_dir / 'eta_distributions.png'}")
    plt.show()


if __name__ == "__main__":
    main()
