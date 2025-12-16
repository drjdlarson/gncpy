"""Test V4 Structured η-Learning INDI vs Baseline.

This script compares:
    - Baseline INDI: Uses fixed B0 matrix
    - η-Learning INDI: Uses B_hat = [B0_F @ diag(η_F); B0_M @ diag(η_M)]

Key difference from V3 (dense ΔB):
    - V3: B_θ(z) = B0 + ΔB_θ(z) where ΔB has 48 free parameters
    - V4: B_θ(z) = [B0_F @ diag(η_F); B0_M @ diag(η_M)] with 16 bounded scalings

This preserves physical structure - each motor's contribution is scaled,
not arbitrarily modified.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import torch.nn as nn

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
# Neural Network for η Scalings
# ============================================================================
class EtaScalingMLP(nn.Module):
    """MLP that predicts bounded per-motor effectiveness scalings.

    Output: η_F (8) + η_M (8) = 16 values, bounded to [0.5, 1.5]
    via η = 1 + bound * tanh(raw_output)
    """

    def __init__(self, input_dim, hidden_dims, num_motors=8, eta_bound=0.5):
        super().__init__()
        self.num_motors = num_motors
        self.eta_bound = eta_bound

        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, 2 * num_motors))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        raw = self.net(x)
        eta = 1.0 + self.eta_bound * torch.tanh(raw)
        return eta


# ============================================================================
# Motor Dynamics Effector
# ============================================================================
# class MotorDynamicsEffector(Effector):
#     """First-order motor dynamics with per-motor efficiency variation."""

#     MOTOR_EFFICIENCY = np.array([0.99, 1.01, 0.98, 1.02, 1.01, 0.99, 1.00, 0.98])

#     def __init__(self, num_motors, tau_mot, initial_state=None):
#         self.num_motors = num_motors
#         self.tau_mot = tau_mot
#         self.state = (
#             np.array(initial_state).flatten().copy()
#             if initial_state is not None
#             else np.zeros(num_motors)
#         )

#     def set_initial_state(self, initial_state):
#         self.state = np.array(initial_state).flatten().copy()

#     def step(self, input_cmds, dt):
#         input_cmds = np.array(input_cmds).flatten()
#         scaled_cmds = input_cmds * self.MOTOR_EFFICIENCY[: self.num_motors]
#         alpha = np.exp(-dt / self.tau_mot)
#         self.state = scaled_cmds + (self.state - scaled_cmds) * alpha
#         return self.state.copy()

from generate_indi_training_data_v4 import (
    MotorDynamicsEffector,
)


# ============================================================================
# Constants
# ============================================================================
DT = 0.001
CONTROL_SUBSAMPLE = 10
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
BIAS_VEL = np.zeros(3)
BIAS_ACCEL = np.zeros(3)
BIAS_OMEGA = np.zeros(3)

FC_VEL = 5.0
FC_ACCEL = 20.0
FC_OMEGA = 20.0
FC_ALPHA = 5.0

WIND_VELOCITY = np.array([8.0, 5.0, 1.0])
WIND_GUST_AMP = np.array([4.0, 3.0, 1.0])
WIND_GUST_FREQ = np.array([0.3, 0.4, 0.5])

np.random.seed(42)


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


def compute_gravity_body(quat):
    """Compute gravity vector in body frame: g_B = R(q)^T @ g_I"""
    # quat format: [w, x, y, z]
    w, x, y, z = quat
    # Rotation matrix from body to inertial
    R = np.array(
        [
            [1 - 2 * (y**2 + z**2), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x**2 + z**2), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x**2 + y**2)],
        ]
    )
    # Gravity in inertial NED frame (+z down)
    g_I = np.array([0, 0, 9.81])
    # Transform to body frame: g_B = R^T @ g_I
    return R.T @ g_I


# ============================================================================
# Setup
# ============================================================================
print("=" * 70)
print("V4 η-Learning INDI vs Baseline INDI Comparison Test")
print("=" * 70)

lofi_config = Path(__file__).parent / "omnicopter_config.yaml"
hifi_config = Path(__file__).parent / "omnicopter_config_hifi.yaml"
model_path = (
    Path(__file__).parent / "training_data_v4" / "models" / "eta_scalings_mlp.pt"
)

# Load NN model
if model_path.exists():
    print("Loading trained η-scalings NN model...")
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    nn_model = EtaScalingMLP(
        input_dim=17,  # v_B(3) + ω_B(3) + g_B(3) + u(8)
        hidden_dims=checkpoint["hidden_dims"],
        num_motors=checkpoint["num_motors"],
        eta_bound=checkpoint["eta_bound"],
    )
    nn_model.load_state_dict(checkpoint["state_dict"])
    nn_model.eval()
    X_mean = torch.tensor(checkpoint["X_mean"], dtype=torch.float32)
    X_std = torch.tensor(checkpoint["X_std"], dtype=torch.float32)
    B0_F_nn = checkpoint["B0_F"]
    B0_M_nn = checkpoint["B0_M"]
    print(f"  Model: {model_path.name}")
    print(f"  Hidden dims: {checkpoint['hidden_dims']}")
    print(f"  η bound: {checkpoint['eta_bound']}")
    print(f"  Input: [v_B, ω_B, g_B, u] = 17 dims")
    print(f"  Output: η_F(8) + η_M(8) = 16 bounded scalings")
    HAS_MODEL = True
else:
    print(f"WARNING: No trained model found at {model_path}")
    print("Running baseline only")
    HAS_MODEL = False

# Setup dynamics
lofi_dyn = ComplexMultirotor(str(lofi_config))
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
B0_moment = np.linalg.inv(inertia) @ (T_max * P_cross_N)
B0 = np.vstack([B0_force, B0_moment])
B0_F = B0[:3, :]  # Force part (3 x 8)
B0_M = B0[3:, :]  # Moment part (3 x 8)
B0_inv = np.linalg.pinv(B0)

# Hover commands
g_mag = gravity[2]
hover_cmds_lofi = B0_inv @ np.array([0.0, 0.0, -g_mag, 0.0, 0.0, 0.0])
with open(hifi_config, "r") as f:
    hifi_params = yaml.load(f)
c2 = hifi_params.prop.poly_thrust[0]
lofi_thrust = T_max * hover_cmds_lofi
hover_cmds_hifi = np.sign(lofi_thrust) * np.sqrt(np.abs(lofi_thrust) / c2)

# Create motor effector and HiFi dynamics
motor_effector = MotorDynamicsEffector(num_motors, TAU_MOT, hover_cmds_hifi)
hifi_dyn = ComplexMultirotor(str(hifi_config), effector=motor_effector)

# INDI controller (baseline uses fixed B0)
K = np.diag([K_VEL, K_VEL, K_VEL, K_OMEGA, K_OMEGA, K_OMEGA])
indi_ctrl = INDI(omit_A=True)
indi_ctrl.set_state_model(dt=DT, K=K, B0=B0)

# Aerodynamic params
cd = hifi_dyn.vehicle.params.aero.cd
frontal_area = np.array(hifi_dyn.vehicle.params.geo.front_area_m2)

# Filter coefficients
alpha_vel = DT / (DT + 1.0 / (2.0 * np.pi * FC_VEL))
alpha_accel = DT / (DT + 1.0 / (2.0 * np.pi * FC_ACCEL))
alpha_omega = DT / (DT + 1.0 / (2.0 * np.pi * FC_OMEGA))
alpha_alpha = DT / (DT + 1.0 / (2.0 * np.pi * FC_ALPHA))

print(f"  B0: {B0.shape}, {num_motors} motors")
print(f"  B0_F: {B0_F.shape}, B0_M: {B0_M.shape}")
print(f"  Control rate: {1/(DT*CONTROL_SUBSAMPLE):.0f} Hz")

# CRITICAL: Use saved B0 from training for η-scaling construction!
# The NN was trained to predict η relative to B0_F_nn/B0_M_nn, not the locally rebuilt B0.
if HAS_MODEL:
    # Check consistency between training B0 and local B0
    B0_F_diff = np.max(np.abs(B0_F - B0_F_nn))
    B0_M_diff = np.max(np.abs(B0_M - B0_M_nn))
    print(f"  B0_F max diff from training: {B0_F_diff:.6e}")
    print(f"  B0_M max diff from training: {B0_M_diff:.6e}")
    if B0_F_diff > 1e-6 or B0_M_diff > 1e-6:
        print("  WARNING: B0 mismatch detected! Using saved B0 from training.")
    # Use saved B0 for η scaling (critical for correct η application)
    B0_F_eta = B0_F_nn.copy()
    B0_M_eta = B0_M_nn.copy()
else:
    B0_F_eta = B0_F
    B0_M_eta = B0_M

print("Setup complete!\n")


def predict_eta_scalings(vel_filt, omega_filt, quat, u_current):
    """Predict per-motor effectiveness scalings using trained NN.

    Returns:
        eta_F: force scalings (8,)
        eta_M: moment scalings (8,)
    """
    # Compute gravity in body frame
    g_B = compute_gravity_body(quat)

    # Build input: z = [v_B, ω_B, g_B, u] = 17 dims
    z = np.concatenate([vel_filt, omega_filt, g_B, u_current])
    z_t = torch.tensor(z, dtype=torch.float32).unsqueeze(0)
    z_norm = (z_t - X_mean) / X_std

    with torch.no_grad():
        eta = nn_model(z_norm).squeeze(0).numpy()

    eta_F = eta[:num_motors]
    eta_M = eta[num_motors:]

    return eta_F, eta_M


def construct_B_hat(eta_F, eta_M):
    """Construct structured B matrix from η scalings.

    B_hat = [B0_F_eta @ diag(η_F); B0_M_eta @ diag(η_M)]

    CRITICAL: Uses B0_F_eta/B0_M_eta which are the SAME matrices used during
    training. If we use a different B0, the η scalings become meaningless.
    """
    B_F_hat = B0_F_eta @ np.diag(eta_F)
    B_M_hat = B0_M_eta @ np.diag(eta_M)
    return np.vstack([B_F_hat, B_M_hat])


# def run_test(test_name, sim_time, ref_func, use_eta_learning=False):
#     """Run a single test with either baseline or η-learning INDI."""
#     np.random.seed(42)

#     num_steps = int(sim_time / DT)

#     # Reset dynamics
#     motor_effector.set_initial_state(hover_cmds_hifi)
#     hifi_dyn.set_initial_conditions(
#         INITIAL_POSITION,
#         INITIAL_VELOCITY,
#         INITIAL_ATTITUDE,
#         INITIAL_ANGULAR_VELOCITY,
#         REF_LAT,
#         REF_LON,
#         TERRAIN_ALT,
#         ned_mag,
#     )
#     hifi_dyn.vehicle.takenoff = True

#     # Storage
#     time_hist = np.zeros(num_steps)
#     vel_hist = np.zeros((num_steps, 3))
#     vel_ref_hist = np.zeros((num_steps, 3))
#     omega_hist = np.zeros((num_steps, 3))
#     omega_ref_hist = np.zeros((num_steps, 3))
#     cmd_hist = np.zeros((num_steps, num_motors))
#     euler_hist = np.zeros((num_steps, 3))
#     eta_F_hist = np.zeros((num_steps, num_motors))
#     eta_M_hist = np.zeros((num_steps, num_motors))

#     # State
#     cur_state = hifi_dyn.vehicle.state.copy()
#     cur_input = hover_cmds_hifi.copy()
#     u_cmd = hover_cmds_hifi.copy()

#     # Filters
#     vel_filt = np.zeros(3)
#     accel_filt = np.zeros(3)
#     omega_filt = np.zeros(3)
#     alpha_filt = np.zeros(3)
#     omega_prev = np.zeros(3)

#     for ii in range(num_steps):
#         tt = ii * DT
#         time_hist[ii] = tt

#         # True state
#         body_vel_true = cur_state[v_smap_quat.body_vel].flatten()
#         body_accel_true = cur_state[v_smap_quat.body_accel].flatten()
#         body_omega_true = cur_state[v_smap_quat.body_rot_rate].flatten()
#         quat = cur_state[v_smap_quat.quat].flatten()
#         ned_vel = cur_state[v_smap_quat.ned_vel].flatten()

#         # Wind
#         wind_gust = WIND_GUST_AMP * np.sin(2.0 * np.pi * WIND_GUST_FREQ * tt)
#         wind_total = WIND_VELOCITY + wind_gust
#         wind_force = compute_wind_force(wind_total, ned_vel, quat, cd, frontal_area)

#         # Sensor model
#         vel_filt = (
#             alpha_vel * (body_vel_true + np.random.normal(0, SIGMA_VEL, 3) + BIAS_VEL)
#             + (1 - alpha_vel) * vel_filt
#         )
#         accel_filt = (
#             alpha_accel
#             * (body_accel_true + np.random.normal(0, SIGMA_ACCEL, 3) + BIAS_ACCEL)
#             + (1 - alpha_accel) * accel_filt
#         )
#         omega_filt = (
#             alpha_omega
#             * (body_omega_true + np.random.normal(0, SIGMA_OMEGA, 3) + BIAS_OMEGA)
#             + (1 - alpha_omega) * omega_filt
#         )

#         if ii == 0:
#             alpha_meas = np.zeros(3)
#         else:
#             alpha_meas = (omega_filt - omega_prev) / DT
#         alpha_filt = alpha_alpha * alpha_meas + (1 - alpha_alpha) * alpha_filt
#         omega_prev = omega_filt.copy()

#         # Control update (100 Hz)
#         if ii % CONTROL_SUBSAMPLE == 0:
#             x = np.concatenate([vel_filt, omega_filt])
#             x_dot = np.concatenate([accel_filt, alpha_filt])

#             vb_ref, vb_ref_dot, omega_ref, omega_ref_dot = ref_func(tt)
#             ref = np.concatenate([vb_ref, omega_ref])
#             ref_dot = np.concatenate([vb_ref_dot, omega_ref_dot])

#             if use_eta_learning and HAS_MODEL:
#                 # η-Learning INDI: Construct B_hat from predicted scalings
#                 eta_F, eta_M = predict_eta_scalings(
#                     vel_filt, omega_filt, quat, cur_input
#                 )
#                 B_hat = construct_B_hat(eta_F, eta_M)
#                 # Update INDI controller with structured B
#                 indi_ctrl.set_state_model(dt=DT, K=K, B0=B_hat)
#                 eta_F_hist[ii] = eta_F
#                 eta_M_hist[ii] = eta_M
#             else:
#                 eta_F_hist[ii] = 1.0
#                 eta_M_hist[ii] = 1.0

#             u_cmd = indi_ctrl.calculate_control(
#                 cur_time=tt,
#                 cur_state=x,
#                 cur_state_dot=x_dot,
#                 cur_input=cur_input,
#                 ref=ref,
#                 ref_dot=ref_dot,
#             )
#             u_cmd = np.clip(u_cmd, -1.0, 1.0)

#             if use_eta_learning and HAS_MODEL:
#                 # Reset to B0 for next iteration
#                 indi_ctrl.set_state_model(dt=DT, K=K, B0=B0)

#         # Get reference for storage
#         vb_ref, _, omega_ref, _ = ref_func(tt)

#         # Store
#         vel_hist[ii] = body_vel_true
#         vel_ref_hist[ii] = vb_ref
#         omega_hist[ii] = body_omega_true
#         omega_ref_hist[ii] = omega_ref
#         cmd_hist[ii] = u_cmd
#         roll, pitch, yaw = gmath.quat_to_euler(quat)
#         euler_hist[ii] = np.rad2deg([roll, pitch, yaw])

#         # Propagate
#         next_state = hifi_dyn.propagate_state(DT, cur_state, u_cmd).flatten()
#         next_state[v_smap_quat.body_vel] += (wind_force / mass) * DT
#         cur_state = next_state
#         cur_input = motor_effector.state.copy()

#     # Metrics
#     vel_error = np.linalg.norm(vel_hist - vel_ref_hist, axis=1)
#     omega_error = np.linalg.norm(omega_hist - omega_ref_hist, axis=1)

#     return {
#         "time": time_hist,
#         "vel": vel_hist,
#         "vel_ref": vel_ref_hist,
#         "omega": omega_hist,
#         "omega_ref": omega_ref_hist,
#         "cmd": cmd_hist,
#         "euler": euler_hist,
#         "eta_F": eta_F_hist,
#         "eta_M": eta_M_hist,
#         "vel_error_mean": vel_error.mean(),
#         "vel_error_max": vel_error.max(),
#         "omega_error_mean": omega_error.mean(),
#         "omega_error_max": omega_error.max(),
#     }


def run_test(test_name, sim_time, ref_func, use_eta_learning=False):
    np.random.seed(42)

    num_steps = int(sim_time / DT)

    motor_effector.set_initial_state(hover_cmds_hifi)
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

    time_hist = np.zeros(num_steps)
    pos_hist = np.zeros((num_steps, 3))
    vel_hist = np.zeros((num_steps, 3))
    vel_ref_hist = np.zeros((num_steps, 3))
    omega_hist = np.zeros((num_steps, 3))
    omega_ref_hist = np.zeros((num_steps, 3))
    cmd_hist = np.zeros((num_steps, num_motors))
    euler_hist = np.zeros((num_steps, 3))
    eta_F_hist = np.zeros((num_steps, num_motors))
    eta_M_hist = np.zeros((num_steps, num_motors))

    cur_state = hifi_dyn.vehicle.state.copy()
    cur_input = hover_cmds_hifi.copy()
    u_cmd = hover_cmds_hifi.copy()

    vel_filt = np.zeros(3)
    accel_filt = np.zeros(3)
    omega_filt = np.zeros(3)
    alpha_filt = np.zeros(3)
    omega_prev = np.zeros(3)

    for ii in range(num_steps):
        tt = ii * DT
        time_hist[ii] = tt

        ned_pos = cur_state[v_smap_quat.ned_pos].flatten()
        body_vel_true = cur_state[v_smap_quat.body_vel].flatten()
        body_accel_true = cur_state[v_smap_quat.body_accel].flatten()
        body_omega_true = cur_state[v_smap_quat.body_rot_rate].flatten()
        quat = cur_state[v_smap_quat.quat].flatten()
        ned_vel = cur_state[v_smap_quat.ned_vel].flatten()

        wind_gust = WIND_GUST_AMP * np.sin(2.0 * np.pi * WIND_GUST_FREQ * tt)
        wind_total = WIND_VELOCITY + wind_gust
        wind_force = compute_wind_force(wind_total, ned_vel, quat, cd, frontal_area)

        vel_filt = (
            alpha_vel * (body_vel_true + np.random.normal(0, SIGMA_VEL, 3) + BIAS_VEL)
            + (1 - alpha_vel) * vel_filt
        )
        accel_filt = (
            alpha_accel
            * (body_accel_true + np.random.normal(0, SIGMA_ACCEL, 3) + BIAS_ACCEL)
            + (1 - alpha_accel) * accel_filt
        )
        omega_filt = (
            alpha_omega
            * (body_omega_true + np.random.normal(0, SIGMA_OMEGA, 3) + BIAS_OMEGA)
            + (1 - alpha_omega) * omega_filt
        )

        if ii == 0:
            alpha_meas = np.zeros(3)
        else:
            alpha_meas = (omega_filt - omega_prev) / DT
        alpha_filt = alpha_alpha * alpha_meas + (1 - alpha_alpha) * alpha_filt
        omega_prev = omega_filt.copy()

        if ii % CONTROL_SUBSAMPLE == 0:
            x = np.concatenate([vel_filt, omega_filt])
            x_dot = np.concatenate([accel_filt, alpha_filt])

            vb_ref, vb_ref_dot, omega_ref, omega_ref_dot = ref_func(tt)
            ref = np.concatenate([vb_ref, omega_ref])
            ref_dot = np.concatenate([vb_ref_dot, omega_ref_dot])

            if use_eta_learning and HAS_MODEL:
                eta_F, eta_M = predict_eta_scalings(
                    vel_filt, omega_filt, quat, cur_input
                )
                B_hat = construct_B_hat(eta_F, eta_M)
                indi_ctrl.set_state_model(dt=DT, K=K, B0=B_hat)
                eta_F_hist[ii] = eta_F
                eta_M_hist[ii] = eta_M
            else:
                eta_F_hist[ii] = 1.0
                eta_M_hist[ii] = 1.0

            u_cmd = indi_ctrl.calculate_control(
                cur_time=tt,
                cur_state=x,
                cur_state_dot=x_dot,
                cur_input=cur_input,
                ref=ref,
                ref_dot=ref_dot,
            )
            u_cmd = np.clip(u_cmd, -1.0, 1.0)

            if use_eta_learning and HAS_MODEL:
                indi_ctrl.set_state_model(dt=DT, K=K, B0=B0)

        vb_ref, _, omega_ref, _ = ref_func(tt)

        pos_hist[ii] = ned_pos
        vel_hist[ii] = body_vel_true
        vel_ref_hist[ii] = vb_ref
        omega_hist[ii] = body_omega_true
        omega_ref_hist[ii] = omega_ref
        cmd_hist[ii] = u_cmd
        roll, pitch, yaw = gmath.quat_to_euler(quat)
        euler_hist[ii] = np.rad2deg([roll, pitch, yaw])

        next_state = hifi_dyn.propagate_state(DT, cur_state, u_cmd).flatten()
        next_state[v_smap_quat.body_vel] += (wind_force / mass) * DT
        cur_state = next_state
        cur_input = motor_effector.state.copy()

    vel_error = np.linalg.norm(vel_hist - vel_ref_hist, axis=1)
    omega_error = np.linalg.norm(omega_hist - omega_ref_hist, axis=1)

    return {
        "time": time_hist,
        "pos": pos_hist,
        "vel": vel_hist,
        "vel_ref": vel_ref_hist,
        "omega": omega_hist,
        "omega_ref": omega_ref_hist,
        "cmd": cmd_hist,
        "euler": euler_hist,
        "eta_F": eta_F_hist,
        "eta_M": eta_M_hist,
        "vel_error_mean": vel_error.mean(),
        "vel_error_max": vel_error.max(),
        "omega_error_mean": omega_error.mean(),
        "omega_error_max": omega_error.max(),
    }


# ============================================================================
# Reference Trajectories (same as test_omnicopter_na_indi_wind.py)
# ============================================================================

# Test 1: Circular Velocity Tracking
V_CIRCLE = 1.0
OMEGA_CIRCLE = 2.0 * np.pi / 8.0


def ref_test1(t):
    """Circular velocity tracking."""
    vb_ref = np.array(
        [V_CIRCLE * np.cos(OMEGA_CIRCLE * t), V_CIRCLE * np.sin(OMEGA_CIRCLE * t), 0.0]
    )
    vb_ref_dot = np.array(
        [
            -V_CIRCLE * OMEGA_CIRCLE * np.sin(OMEGA_CIRCLE * t),
            V_CIRCLE * OMEGA_CIRCLE * np.cos(OMEGA_CIRCLE * t),
            0.0,
        ]
    )
    return vb_ref, vb_ref_dot, np.zeros(3), np.zeros(3)


# Test 2: Moderate Roll Maneuver
ROLL_TIME = 10.0
ROLL_RATE = 0.5  # Matches training data Phase 2


def ref_test2(t):
    """Moderate roll maneuver."""
    omega_ref = np.array([ROLL_RATE, 0.0, 0.0]) if t < ROLL_TIME else np.zeros(3)
    return np.zeros(3), np.zeros(3), omega_ref, np.zeros(3)


# Test 3: Combined Velocity + Attitude
V_CIRCLE_3 = 0.8
OMEGA_CIRCLE_3 = 2.0 * np.pi / 12.0  # 12s period
ROLL_TIME_3 = 10.0
ROLL_RATE_3 = 0.3


def ref_test3(t):
    """Combined velocity + attitude tracking."""
    vb_ref = np.array(
        [
            V_CIRCLE_3 * np.cos(OMEGA_CIRCLE_3 * t),
            V_CIRCLE_3 * np.sin(OMEGA_CIRCLE_3 * t),
            0.0,
        ]
    )
    vb_ref_dot = np.array(
        [
            -V_CIRCLE_3 * OMEGA_CIRCLE_3 * np.sin(OMEGA_CIRCLE_3 * t),
            V_CIRCLE_3 * OMEGA_CIRCLE_3 * np.cos(OMEGA_CIRCLE_3 * t),
            0.0,
        ]
    )
    omega_ref = np.array([ROLL_RATE_3, 0.0, 0.0]) if t < ROLL_TIME_3 else np.zeros(3)
    return vb_ref, vb_ref_dot, omega_ref, np.zeros(3)


# Test 4: Moderate Helix + Multi-Axis
V_CIRCLE_4 = 0.8
OMEGA_CIRCLE_4 = 2.0 * np.pi / 10.0
V_VERT_AMP = 0.3
OMEGA_VERT = 2.0 * np.pi / 8.0
ROLL_RATE_4 = 0.5
PITCH_RATE_4 = 0.4
YAW_RATE_4 = 0.3
TUMBLE_TIME = 10.0


def ref_test4(t):
    """Moderate helix with multi-axis rotation."""
    vb_ref = np.array(
        [
            V_CIRCLE_4 * np.cos(OMEGA_CIRCLE_4 * t),
            V_CIRCLE_4 * np.sin(OMEGA_CIRCLE_4 * t),
            V_VERT_AMP * np.sin(OMEGA_VERT * t),
        ]
    )
    vb_ref_dot = np.array(
        [
            -V_CIRCLE_4 * OMEGA_CIRCLE_4 * np.sin(OMEGA_CIRCLE_4 * t),
            V_CIRCLE_4 * OMEGA_CIRCLE_4 * np.cos(OMEGA_CIRCLE_4 * t),
            V_VERT_AMP * OMEGA_VERT * np.cos(OMEGA_VERT * t),
        ]
    )
    omega_ref = (
        np.array([ROLL_RATE_4, PITCH_RATE_4, YAW_RATE_4])
        if t < TUMBLE_TIME
        else np.zeros(3)
    )
    return vb_ref, vb_ref_dot, omega_ref, np.zeros(3)


# ============================================================================
# Helper function for detailed per-test plots
# ============================================================================
def plot_test_results(test_name, plot_name, baseline, eta_indi, output_dir):
    """Create detailed comparison plot for a single test.

    Shows:
    - Row 1: Body velocity (all 3 components) for baseline and η-INDI
    - Row 2: Angular rates (all 3 components) for baseline and η-INDI
    - Row 3: Euler angles for baseline and η-INDI
    - Row 4: Motor commands for baseline and η-INDI
    """
    fig, axes = plt.subplots(4, 2, figsize=(16, 14), sharex=True)
    fig.suptitle(f"η-INDI vs Baseline: {test_name}", fontsize=14, fontweight="bold")

    bl = baseline
    eta = eta_indi

    # Row 0: Body velocity tracking
    # Baseline
    ax = axes[0, 0]
    ax.plot(bl["time"], bl["vel"][:, 0], "r-", label="vb_x")
    ax.plot(bl["time"], bl["vel_ref"][:, 0], "r--", alpha=0.7, label="vb_x ref")
    ax.plot(bl["time"], bl["vel"][:, 1], "g-", label="vb_y")
    ax.plot(bl["time"], bl["vel_ref"][:, 1], "g--", alpha=0.7, label="vb_y ref")
    ax.plot(bl["time"], bl["vel"][:, 2], "b-", label="vb_z")
    ax.plot(bl["time"], bl["vel_ref"][:, 2], "b--", alpha=0.7, label="vb_z ref")
    ax.axhline(0, color="k", linestyle="--", alpha=0.3)
    ax.set_ylabel("Body Velocity (m/s)")
    ax.legend(loc="upper right", ncol=3, fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_title("Baseline INDI - Velocity", fontsize=10)

    # η-INDI
    ax = axes[0, 1]
    if eta is not None:
        ax.plot(eta["time"], eta["vel"][:, 0], "r-", label="vb_x")
        ax.plot(eta["time"], eta["vel_ref"][:, 0], "r--", alpha=0.7, label="vb_x ref")
        ax.plot(eta["time"], eta["vel"][:, 1], "g-", label="vb_y")
        ax.plot(eta["time"], eta["vel_ref"][:, 1], "g--", alpha=0.7, label="vb_y ref")
        ax.plot(eta["time"], eta["vel"][:, 2], "b-", label="vb_z")
        ax.plot(eta["time"], eta["vel_ref"][:, 2], "b--", alpha=0.7, label="vb_z ref")
    ax.axhline(0, color="k", linestyle="--", alpha=0.3)
    ax.set_ylabel("Body Velocity (m/s)")
    ax.legend(loc="upper right", ncol=3, fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_title("η-INDI - Velocity", fontsize=10)

    # Row 1: Angular rates
    # Baseline
    ax = axes[1, 0]
    ax.plot(bl["time"], np.rad2deg(bl["omega"][:, 0]), "r-", label="p")
    ax.plot(
        bl["time"], np.rad2deg(bl["omega_ref"][:, 0]), "r--", alpha=0.7, label="p ref"
    )
    ax.plot(bl["time"], np.rad2deg(bl["omega"][:, 1]), "g-", label="q")
    ax.plot(
        bl["time"], np.rad2deg(bl["omega_ref"][:, 1]), "g--", alpha=0.7, label="q ref"
    )
    ax.plot(bl["time"], np.rad2deg(bl["omega"][:, 2]), "b-", label="r")
    ax.plot(
        bl["time"], np.rad2deg(bl["omega_ref"][:, 2]), "b--", alpha=0.7, label="r ref"
    )
    ax.axhline(0, color="k", linestyle="--", alpha=0.3)
    ax.set_ylabel("Angular Rate (deg/s)")
    ax.legend(loc="upper right", ncol=3, fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_title("Baseline INDI - Angular Rates", fontsize=10)

    # η-INDI
    ax = axes[1, 1]
    if eta is not None:
        ax.plot(eta["time"], np.rad2deg(eta["omega"][:, 0]), "r-", label="p")
        ax.plot(
            eta["time"],
            np.rad2deg(eta["omega_ref"][:, 0]),
            "r--",
            alpha=0.7,
            label="p ref",
        )
        ax.plot(eta["time"], np.rad2deg(eta["omega"][:, 1]), "g-", label="q")
        ax.plot(
            eta["time"],
            np.rad2deg(eta["omega_ref"][:, 1]),
            "g--",
            alpha=0.7,
            label="q ref",
        )
        ax.plot(eta["time"], np.rad2deg(eta["omega"][:, 2]), "b-", label="r")
        ax.plot(
            eta["time"],
            np.rad2deg(eta["omega_ref"][:, 2]),
            "b--",
            alpha=0.7,
            label="r ref",
        )
    ax.axhline(0, color="k", linestyle="--", alpha=0.3)
    ax.set_ylabel("Angular Rate (deg/s)")
    ax.legend(loc="upper right", ncol=3, fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_title("η-INDI - Angular Rates", fontsize=10)

    # Row 2: Euler angles
    # Baseline
    ax = axes[2, 0]
    ax.plot(bl["time"], bl["euler"][:, 0], "r-", label="Roll")
    ax.plot(bl["time"], bl["euler"][:, 1], "g-", label="Pitch")
    ax.plot(bl["time"], bl["euler"][:, 2], "b-", label="Yaw")
    ax.axhline(0, color="k", linestyle="--", alpha=0.3)
    ax.set_ylabel("Euler Angles (deg)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_title("Baseline INDI - Attitude", fontsize=10)

    # η-INDI
    ax = axes[2, 1]
    if eta is not None:
        ax.plot(eta["time"], eta["euler"][:, 0], "r-", label="Roll")
        ax.plot(eta["time"], eta["euler"][:, 1], "g-", label="Pitch")
        ax.plot(eta["time"], eta["euler"][:, 2], "b-", label="Yaw")
    ax.axhline(0, color="k", linestyle="--", alpha=0.3)
    ax.set_ylabel("Euler Angles (deg)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_title("η-INDI - Attitude", fontsize=10)

    # Row 3: Motor commands
    # Baseline
    ax = axes[3, 0]
    for ii in range(bl["cmd"].shape[1]):
        ax.plot(bl["time"], bl["cmd"][:, ii], alpha=0.7, label=f"u{ii}")
    ax.axhline(0, color="k", linestyle="--", alpha=0.3)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Motor Commands")
    ax.set_ylim([-1.1, 1.1])
    ax.legend(loc="upper right", ncol=4, fontsize=6)
    ax.grid(True, alpha=0.3)
    ax.set_title("Baseline INDI - Commands", fontsize=10)

    # η-INDI
    ax = axes[3, 1]
    if eta is not None:
        for ii in range(eta["cmd"].shape[1]):
            ax.plot(eta["time"], eta["cmd"][:, ii], alpha=0.7, label=f"u{ii}")
    ax.axhline(0, color="k", linestyle="--", alpha=0.3)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Motor Commands")
    ax.set_ylim([-1.1, 1.1])
    ax.legend(loc="upper right", ncol=4, fontsize=6)
    ax.grid(True, alpha=0.3)
    ax.set_title("η-INDI - Commands", fontsize=10)

    plt.tight_layout()
    plot_file = output_dir / f"eta_indi_{plot_name}.png"
    # plt.savefig(plot_file, dpi=150)
    plt.close(fig)
    # print(f"  Detailed plot saved: {plot_file}")

    # Also create error comparison plot
    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    fig2.suptitle(f"Error Comparison: {test_name}", fontsize=14, fontweight="bold")

    # Velocity error comparison
    ax = axes2[0, 0]
    vel_err_bl = np.linalg.norm(bl["vel"] - bl["vel_ref"], axis=1)
    ax.plot(bl["time"], vel_err_bl, "b-", label="Baseline", linewidth=1.5)
    if eta is not None:
        vel_err_eta = np.linalg.norm(eta["vel"] - eta["vel_ref"], axis=1)
        ax.plot(eta["time"], vel_err_eta, "r-", label="η-INDI", linewidth=1.5)
    ax.set_ylabel("Velocity Error (m/s)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_title("Velocity Tracking Error", fontsize=10)

    # Angular rate error comparison
    ax = axes2[0, 1]
    omega_err_bl = np.linalg.norm(bl["omega"] - bl["omega_ref"], axis=1)
    ax.plot(bl["time"], np.rad2deg(omega_err_bl), "b-", label="Baseline", linewidth=1.5)
    if eta is not None:
        omega_err_eta = np.linalg.norm(eta["omega"] - eta["omega_ref"], axis=1)
        ax.plot(
            eta["time"], np.rad2deg(omega_err_eta), "r-", label="η-INDI", linewidth=1.5
        )
    ax.set_ylabel("Angular Rate Error (deg/s)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_title("Angular Rate Tracking Error", fontsize=10)

    # Per-axis velocity error
    ax = axes2[1, 0]
    ax.plot(
        bl["time"],
        bl["vel"][:, 0] - bl["vel_ref"][:, 0],
        "b-",
        label="Baseline vx",
        alpha=0.7,
    )
    ax.plot(
        bl["time"],
        bl["vel"][:, 1] - bl["vel_ref"][:, 1],
        "b--",
        label="Baseline vy",
        alpha=0.7,
    )
    ax.plot(
        bl["time"],
        bl["vel"][:, 2] - bl["vel_ref"][:, 2],
        "b:",
        label="Baseline vz",
        alpha=0.7,
    )
    if eta is not None:
        ax.plot(
            eta["time"],
            eta["vel"][:, 0] - eta["vel_ref"][:, 0],
            "r-",
            label="η-INDI vx",
            alpha=0.7,
        )
        ax.plot(
            eta["time"],
            eta["vel"][:, 1] - eta["vel_ref"][:, 1],
            "r--",
            label="η-INDI vy",
            alpha=0.7,
        )
        ax.plot(
            eta["time"],
            eta["vel"][:, 2] - eta["vel_ref"][:, 2],
            "r:",
            label="η-INDI vz",
            alpha=0.7,
        )
    ax.axhline(0, color="k", linestyle="--", alpha=0.3)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Velocity Error (m/s)")
    ax.legend(loc="upper right", ncol=2, fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_title("Per-Axis Velocity Error", fontsize=10)

    # Per-axis angular rate error
    ax = axes2[1, 1]
    ax.plot(
        bl["time"],
        np.rad2deg(bl["omega"][:, 0] - bl["omega_ref"][:, 0]),
        "b-",
        label="Baseline p",
        alpha=0.7,
    )
    ax.plot(
        bl["time"],
        np.rad2deg(bl["omega"][:, 1] - bl["omega_ref"][:, 1]),
        "b--",
        label="Baseline q",
        alpha=0.7,
    )
    ax.plot(
        bl["time"],
        np.rad2deg(bl["omega"][:, 2] - bl["omega_ref"][:, 2]),
        "b:",
        label="Baseline r",
        alpha=0.7,
    )
    if eta is not None:
        ax.plot(
            eta["time"],
            np.rad2deg(eta["omega"][:, 0] - eta["omega_ref"][:, 0]),
            "r-",
            label="η-INDI p",
            alpha=0.7,
        )
        ax.plot(
            eta["time"],
            np.rad2deg(eta["omega"][:, 1] - eta["omega_ref"][:, 1]),
            "r--",
            label="η-INDI q",
            alpha=0.7,
        )
        ax.plot(
            eta["time"],
            np.rad2deg(eta["omega"][:, 2] - eta["omega_ref"][:, 2]),
            "r:",
            label="η-INDI r",
            alpha=0.7,
        )
    ax.axhline(0, color="k", linestyle="--", alpha=0.3)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Angular Rate Error (deg/s)")
    ax.legend(loc="upper right", ncol=2, fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_title("Per-Axis Angular Rate Error", fontsize=10)

    plt.tight_layout()
    plot_file2 = output_dir / f"eta_indi_{plot_name}_error.png"
    # plt.savefig(plot_file2, dpi=150)
    plt.close(fig2)
    # print(f"  Error plot saved: {plot_file2}")


# ============================================================================
# Run Tests
# ============================================================================
print("=" * 70)
print("Running Comparison Tests")
print("=" * 70)

# Create output directory for CSV + plots
output_dir = Path(__file__).parent / "Final_Results"
output_dir.mkdir(exist_ok=True)


# def save_csv(res_dict, fname):
#     """Save run data (baseline or eta) to CSV for later plotting."""
#     data = np.column_stack(
#         [
#             res_dict["time"],
#             res_dict["vel"],
#             res_dict["vel_ref"],
#             res_dict["omega"],
#             res_dict["omega_ref"],
#             res_dict["euler"],
#             res_dict["cmd"],
#             res_dict["eta_F"],
#             res_dict["eta_M"],
#         ]
#     )
#     num_motors = res_dict["cmd"].shape[1]
#     header_parts = [
#         "t",
#         "vb_x",
#         "vb_y",
#         "vb_z",
#         "vb_ref_x",
#         "vb_ref_y",
#         "vb_ref_z",
#         "omega_x",
#         "omega_y",
#         "omega_z",
#         "omega_ref_x",
#         "omega_ref_y",
#         "omega_ref_z",
#         "roll_deg",
#         "pitch_deg",
#         "yaw_deg",
#     ]
#     header_parts.extend([f"u{i}" for i in range(num_motors)])
#     header_parts.extend([f"eta_F_{i}" for i in range(num_motors)])
#     header_parts.extend([f"eta_M_{i}" for i in range(num_motors)])
#     np.savetxt(
#         fname,
#         data,
#         delimiter=",",
#         header=",".join(header_parts),
#         comments="",
#     )


def save_csv(res_dict, fname):
    data = np.column_stack(
        [
            res_dict["time"],
            res_dict["pos"],
            res_dict["vel"],
            res_dict["vel_ref"],
            res_dict["omega"],
            res_dict["omega_ref"],
            res_dict["euler"],
            res_dict["cmd"],
            res_dict["eta_F"],
            res_dict["eta_M"],
        ]
    )
    num_motors = res_dict["cmd"].shape[1]
    header_parts = [
        "t",
        "pn_x",
        "pn_y",
        "pn_z",
        "vb_x",
        "vb_y",
        "vb_z",
        "vb_ref_x",
        "vb_ref_y",
        "vb_ref_z",
        "omega_x",
        "omega_y",
        "omega_z",
        "omega_ref_x",
        "omega_ref_y",
        "omega_ref_z",
        "roll_deg",
        "pitch_deg",
        "yaw_deg",
    ]
    header_parts.extend([f"u{i}" for i in range(num_motors)])
    header_parts.extend([f"eta_F_{i}" for i in range(num_motors)])
    header_parts.extend([f"eta_M_{i}" for i in range(num_motors)])
    np.savetxt(
        fname,
        data,
        delimiter=",",
        header=",".join(header_parts),
        comments="",
    )


test_scenarios = [
    ("Test 1: Circular Velocity", 10.0, ref_test1, "test1"),
    ("Test 2: Moderate Roll", 10.0, ref_test2, "test2"),
    ("Test 3: Combined Vel+Att", 12.0, ref_test3, "test3"),
    ("Test 4: Helix+MultiAxis", 12.0, ref_test4, "test4"),
]

results = {}

for test_name, sim_time, ref_func, plot_name in test_scenarios:
    print(f"\n--- {test_name} ---")

    # Baseline INDI
    print("  Running Baseline INDI...")
    baseline = run_test(test_name, sim_time, ref_func, use_eta_learning=False)
    results[f"{test_name}_baseline"] = baseline
    print(
        f"    Vel error: mean={baseline['vel_error_mean']:.4f}, max={baseline['vel_error_max']:.4f}"
    )
    print(
        f"    Omega error: mean={baseline['omega_error_mean']:.4f}, max={baseline['omega_error_max']:.4f}"
    )

    eta_indi = None
    if HAS_MODEL:
        # η-Learning INDI
        print("  Running η-Learning INDI (V4)...")
        eta_indi = run_test(test_name, sim_time, ref_func, use_eta_learning=True)
        results[f"{test_name}_eta"] = eta_indi
        print(
            f"    Vel error: mean={eta_indi['vel_error_mean']:.4f}, max={eta_indi['vel_error_max']:.4f}"
        )
        print(
            f"    Omega error: mean={eta_indi['omega_error_mean']:.4f}, max={eta_indi['omega_error_max']:.4f}"
        )

        # Improvement
        vel_improve = (
            (baseline["vel_error_mean"] - eta_indi["vel_error_mean"])
            / baseline["vel_error_mean"]
            * 100
        )
        omega_improve = (
            (baseline["omega_error_mean"] - eta_indi["omega_error_mean"])
            / baseline["omega_error_mean"]
            * 100
        )
        print(f"    Improvement: vel={vel_improve:+.1f}%, omega={omega_improve:+.1f}%")

    # Create detailed plots for this test
    # plot_test_results(test_name, plot_name, baseline, eta_indi, output_dir)

    # Save CSVs
    save_csv(baseline, output_dir / f"{plot_name}_baseline.csv")
    if eta_indi is not None:
        save_csv(eta_indi, output_dir / f"{plot_name}_eta.csv")


# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"\nMotor efficiency variations: {MotorDynamicsEffector.MOTOR_EFFICIENCY}")
print(f"Wind: constant={WIND_VELOCITY}, gust_amp={WIND_GUST_AMP}")

if HAS_MODEL:
    print(
        "\n{:<30} {:>12} {:>12} {:>12}".format(
            "Test", "Baseline", "η-INDI", "Improvement"
        )
    )
    print("-" * 70)

    for test_name, sim_time, ref_func, plot_name in test_scenarios:
        baseline = results[f"{test_name}_baseline"]
        eta_indi = results[f"{test_name}_eta"]
        improve = (
            (baseline["vel_error_mean"] - eta_indi["vel_error_mean"])
            / baseline["vel_error_mean"]
            * 100
        )
        print(
            "{:<30} {:>12.4f} {:>12.4f} {:>+11.1f}%".format(
                test_name,
                baseline["vel_error_mean"],
                eta_indi["vel_error_mean"],
                improve,
            )
        )

# ============================================================================
# Combined Summary Plot
# ============================================================================
fig, axes = plt.subplots(4, 2, figsize=(14, 12), sharex="row")
fig.suptitle(
    "η-Learning INDI vs Baseline INDI Comparison", fontsize=14, fontweight="bold"
)

test_data = [
    (
        "Test 1: Circular Velocity",
        results["Test 1: Circular Velocity_baseline"],
        results.get("Test 1: Circular Velocity_eta"),
    ),
    (
        "Test 2: Moderate Roll",
        results["Test 2: Moderate Roll_baseline"],
        results.get("Test 2: Moderate Roll_eta"),
    ),
    (
        "Test 3: Combined Vel+Att",
        results["Test 3: Combined Vel+Att_baseline"],
        results.get("Test 3: Combined Vel+Att_eta"),
    ),
    (
        "Test 4: Helix+MultiAxis",
        results["Test 4: Helix+MultiAxis_baseline"],
        results.get("Test 4: Helix+MultiAxis_eta"),
    ),
]

for row, (title, bl, eta) in enumerate(test_data):
    # Velocity error
    ax = axes[row, 0]
    vel_err_bl = np.linalg.norm(bl["vel"] - bl["vel_ref"], axis=1)
    ax.plot(bl["time"], vel_err_bl, "b-", label="Baseline", alpha=0.7)
    if HAS_MODEL and eta is not None:
        vel_err_eta = np.linalg.norm(eta["vel"] - eta["vel_ref"], axis=1)
        ax.plot(eta["time"], vel_err_eta, "r-", label="η-INDI", alpha=0.7)
    ax.set_ylabel("Vel Error (m/s)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_title(f"{title} - Velocity", fontsize=10)

    # Angular rate error
    ax = axes[row, 1]
    omega_err_bl = np.linalg.norm(bl["omega"] - bl["omega_ref"], axis=1)
    ax.plot(bl["time"], np.rad2deg(omega_err_bl), "b-", label="Baseline", alpha=0.7)
    if HAS_MODEL and eta is not None:
        omega_err_eta = np.linalg.norm(eta["omega"] - eta["omega_ref"], axis=1)
        ax.plot(eta["time"], np.rad2deg(omega_err_eta), "r-", label="η-INDI", alpha=0.7)
    ax.set_ylabel("Omega Error (deg/s)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_title(f"{title} - Angular Rate", fontsize=10)

axes[-1, 0].set_xlabel("Time (s)")
axes[-1, 1].set_xlabel("Time (s)")

plt.tight_layout()
output_file = output_dir / "eta_indi_comparison.png"
# plt.savefig(output_file, dpi=150)
# print(f"\nCombined figure saved to: {output_file}")

plt.close("all")
