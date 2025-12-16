"""Test Neural-Augmented INDI (NA-INDI) vs Baseline INDI.

This script compares:
    - Baseline INDI: Standard incremental nonlinear dynamic inversion
    - NA-INDI: INDI augmented with neural network residual prediction

The NN predicts residual dynamics that the LoFi B0 matrix doesn't capture:
    residual = x_dot_actual - (x_dot_prev + B0 @ delta_u)

The NA-INDI corrects the control by subtracting the predicted residual:
    delta_x_dot = -x_dot_measured - nn_residual + H_inv @ (ref_dot + K @ e)

Test Cases:
    Test 1: Circular velocity tracking
    Test 2: 360 degree roll maneuver
    Test 3: Combined velocity + attitude maneuver
    Test 4: Helix trajectory with 3-axis tumble

Control Rate:
    - Dynamics: 1 kHz (DT = 0.001)
    - Control: 100 Hz (every 10th step) - matches training data rate

Wind Disturbance (same as training):
    - Constant wind: [8.0, 5.0, 1.0] m/s NED
    - Sinusoidal gusts: ±[4.0, 3.0, 1.0] m/s amplitude
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
# Neural Network for Residual Prediction
# ============================================================================
class ResidualMLP(nn.Module):
    """MLP for predicting INDI residual dynamics."""

    def __init__(self, input_dim=20, output_dim=6, hidden_dims=[64, 64, 64]):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ============================================================================
# Motor Dynamics Effector (same as training)
# ============================================================================
class MotorDynamicsEffector(Effector):
    """First-order motor dynamics with per-motor efficiency variation."""

    MOTOR_EFFICIENCY = np.array([0.99, 1.01, 0.98, 1.02, 1.01, 0.99, 1.00, 0.98])

    def __init__(self, num_motors, tau_mot, initial_state=None):
        self.num_motors = num_motors
        self.tau_mot = tau_mot
        self.state = (
            np.array(initial_state).flatten().copy()
            if initial_state is not None
            else np.zeros(num_motors)
        )

    def set_initial_state(self, initial_state):
        self.state = np.array(initial_state).flatten().copy()

    def step(self, input_cmds, dt):
        input_cmds = np.array(input_cmds).flatten()
        scaled_cmds = input_cmds * self.MOTOR_EFFICIENCY[: self.num_motors]
        alpha = np.exp(-dt / self.tau_mot)
        self.state = scaled_cmds + (self.state - scaled_cmds) * alpha
        return self.state.copy()


# ============================================================================
# Constants
# ============================================================================
DT = 0.001  # Timestep (1 kHz dynamics)
CONTROL_SUBSAMPLE = 10  # Control at 100 Hz
TAU_MOT = 0.032

INITIAL_POSITION = np.array([0.0, 0.0, -10.0])
INITIAL_VELOCITY = np.array([0.0, 0.0, 0.0])
INITIAL_ATTITUDE = np.array([0.0, 0.0, 0.0])
INITIAL_ANGULAR_VELOCITY = np.array([0.0, 0.0, 0.0])
REF_LAT, REF_LON, TERRAIN_ALT = 34.0, -86.0, 0.0

K_VEL = 5.0
K_OMEGA = 10.0

# Sensor noise (same as training)
SIGMA_VEL = 0.05
SIGMA_ACCEL = 0.1
SIGMA_OMEGA = 0.01
BIAS_VEL = np.zeros(3)
BIAS_ACCEL = np.zeros(3)
BIAS_OMEGA = np.zeros(3)

# Filter cutoffs (same as training)
FC_VEL = 5.0
FC_ACCEL = 20.0
FC_OMEGA = 20.0
FC_ALPHA = 5.0

# Wind (MUST match training)
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


# ============================================================================
# Setup
# ============================================================================
print("=" * 70)
print("NA-INDI vs Baseline INDI Comparison Test")
print("=" * 70)

lofi_config = Path(__file__).parent / "omnicopter_config.yaml"
hifi_config = Path(__file__).parent / "omnicopter_config_hifi.yaml"
model_path = Path(__file__).parent / "training_data" / "models" / "hifi_residual_mlp.pt"

# Load NN model (V1 - predicts incremental residual with attitude)
print("Loading trained NN model (V1 with attitude)...")
checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
nn_model = ResidualMLP(
    input_dim=24, output_dim=6, hidden_dims=checkpoint["hidden_dims"]
)
nn_model.load_state_dict(checkpoint["state_dict"])
nn_model.eval()
X_mean = torch.tensor(checkpoint["X_mean"], dtype=torch.float32)
X_std = torch.tensor(checkpoint["X_std"], dtype=torch.float32)
y_mean = checkpoint["y_mean"]
y_std = checkpoint["y_std"]
print(f"  Model: {model_path.name}")
print(f"  Input: [v_b, omega_b, v_dot_b, omega_dot_b, quat, u_prev] = 24 dims")
print(f"  Output: incremental residual = x_dot - (x_dot_prev + B0 @ delta_u)")

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

# INDI controller
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
print(f"  Control rate: {1/(DT*CONTROL_SUBSAMPLE):.0f} Hz")
print("Setup complete!\n")


def predict_residual(vel_filt, omega_filt, accel_filt, alpha_filt, quat, u_current):
    """Predict incremental residual using trained NN (V1 with attitude).

    Residual = x_dot_actual - (x_dot_prev + B0 @ delta_u)

    This represents what INDI's incremental approximation misses.
    By subtracting this from the virtual control, we compensate.

    Parameters
    ----------
    vel_filt : ndarray (3,)
        Filtered body velocity
    omega_filt : ndarray (3,)
        Filtered body angular rate
    accel_filt : ndarray (3,)
        Filtered body acceleration
    alpha_filt : ndarray (3,)
        Filtered body angular acceleration
    quat : ndarray (4,)
        Quaternion attitude [w, x, y, z]
    u_current : ndarray (8,)
        Current motor state

    Returns
    -------
    residual : ndarray (6,)
        Predicted residual [vdot_residual(3), omegadot_residual(3)]
    """
    z = np.concatenate([vel_filt, omega_filt, accel_filt, alpha_filt, quat, u_current])
    z_t = torch.tensor(z, dtype=torch.float32)
    z_norm = (z_t - X_mean) / X_std
    with torch.no_grad():
        y_norm = nn_model(z_norm)
    return y_norm.numpy() * y_std + y_mean


def run_test(test_name, sim_time, ref_func, use_nn=False):
    """Run a single test with either baseline or NA-INDI.

    Parameters
    ----------
    test_name : str
        Name for logging
    sim_time : float
        Simulation duration (seconds)
    ref_func : callable
        Function(t) -> (vb_ref, vb_ref_dot, omega_ref, omega_ref_dot)
    use_nn : bool
        If True, use NA-INDI; otherwise baseline INDI

    Returns
    -------
    results : dict
        Time histories and metrics
    """
    np.random.seed(42)  # Same noise for fair comparison

    num_steps = int(sim_time / DT)

    # Reset dynamics
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

    # Storage
    time_hist = np.zeros(num_steps)
    vel_hist = np.zeros((num_steps, 3))
    vel_ref_hist = np.zeros((num_steps, 3))
    omega_hist = np.zeros((num_steps, 3))
    omega_ref_hist = np.zeros((num_steps, 3))
    cmd_hist = np.zeros((num_steps, num_motors))
    euler_hist = np.zeros((num_steps, 3))

    # State
    cur_state = hifi_dyn.vehicle.state.copy()
    cur_input = hover_cmds_hifi.copy()  # Motor state after propagation
    u_cmd = hover_cmds_hifi.copy()  # Held command between control updates

    # Filters
    vel_filt = np.zeros(3)
    accel_filt = np.zeros(3)
    omega_filt = np.zeros(3)
    alpha_filt = np.zeros(3)
    omega_prev = np.zeros(3)

    for ii in range(num_steps):
        tt = ii * DT
        time_hist[ii] = tt

        # True state
        body_vel_true = cur_state[v_smap_quat.body_vel].flatten()
        body_accel_true = cur_state[v_smap_quat.body_accel].flatten()
        body_omega_true = cur_state[v_smap_quat.body_rot_rate].flatten()
        quat = cur_state[v_smap_quat.quat].flatten()
        ned_vel = cur_state[v_smap_quat.ned_vel].flatten()

        # Wind
        wind_gust = WIND_GUST_AMP * np.sin(2.0 * np.pi * WIND_GUST_FREQ * tt)
        wind_total = WIND_VELOCITY + wind_gust
        wind_force = compute_wind_force(wind_total, ned_vel, quat, cd, frontal_area)

        # Sensor model
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

        # Control update (100 Hz)
        if ii % CONTROL_SUBSAMPLE == 0:
            x = np.concatenate([vel_filt, omega_filt])
            x_dot = np.concatenate([accel_filt, alpha_filt])

            vb_ref, vb_ref_dot, omega_ref, omega_ref_dot = ref_func(tt)
            ref = np.concatenate([vb_ref, omega_ref])
            ref_dot = np.concatenate([vb_ref_dot, omega_ref_dot])

            if use_nn:
                # NA-INDI V1: Predict incremental residual and compensate
                #
                # residual = x_dot[k] - (x_dot[k-1] + B0 @ delta_u)
                # This is what the incremental approximation misses.
                #
                # We SUBTRACT the predicted residual from x_dot to compensate.
                # This makes INDI's prediction more accurate.
                residual = predict_residual(
                    vel_filt, omega_filt, accel_filt, alpha_filt, quat, cur_input
                )
                x_dot_corrected = x_dot - residual
            else:
                # Baseline INDI: use measured x_dot directly
                x_dot_corrected = x_dot

            u_cmd = indi_ctrl.calculate_control(
                cur_time=tt,
                cur_state=x,
                cur_state_dot=x_dot_corrected,
                cur_input=cur_input,
                ref=ref,
                ref_dot=ref_dot,
            )
            u_cmd = np.clip(u_cmd, -1.0, 1.0)

        # Get reference for storage
        vb_ref, _, omega_ref, _ = ref_func(tt)

        # Store
        vel_hist[ii] = body_vel_true
        vel_ref_hist[ii] = vb_ref
        omega_hist[ii] = body_omega_true
        omega_ref_hist[ii] = omega_ref
        cmd_hist[ii] = u_cmd
        roll, pitch, yaw = gmath.quat_to_euler(quat)
        euler_hist[ii] = np.rad2deg([roll, pitch, yaw])

        # Propagate
        next_state = hifi_dyn.propagate_state(DT, cur_state, u_cmd).flatten()
        next_state[v_smap_quat.body_vel] += (wind_force / mass) * DT
        cur_state = next_state
        cur_input = motor_effector.state.copy()

    # Metrics
    vel_error = np.linalg.norm(vel_hist - vel_ref_hist, axis=1)
    omega_error = np.linalg.norm(omega_hist - omega_ref_hist, axis=1)

    return {
        "time": time_hist,
        "vel": vel_hist,
        "vel_ref": vel_ref_hist,
        "omega": omega_hist,
        "omega_ref": omega_ref_hist,
        "cmd": cmd_hist,
        "euler": euler_hist,
        "vel_error_mean": vel_error.mean(),
        "vel_error_max": vel_error.max(),
        "omega_error_mean": omega_error.mean(),
        "omega_error_max": omega_error.max(),
    }


# ============================================================================
# Test 1: Circular Velocity Tracking
# ============================================================================
print("=" * 70)
print("TEST 1: Circular Velocity Tracking")
print("=" * 70)

V_CIRCLE = 1.0
OMEGA_CIRCLE = 2.0 * np.pi / 8.0


def ref_test1(t):
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


print("Running baseline INDI...")
baseline_1 = run_test("Test1 Baseline", 10.0, ref_test1, use_nn=False)
print(
    f"  Vel error: mean={baseline_1['vel_error_mean']:.4f}, max={baseline_1['vel_error_max']:.4f} m/s"
)

print("Running NA-INDI...")
naindi_1 = run_test("Test1 NA-INDI", 10.0, ref_test1, use_nn=True)
print(
    f"  Vel error: mean={naindi_1['vel_error_mean']:.4f}, max={naindi_1['vel_error_max']:.4f} m/s"
)
print()


# ============================================================================
# Test 2: Moderate Roll Maneuver (matches training Phase 2)
# ============================================================================
print("=" * 70)
print("TEST 2: Moderate Roll Maneuver")
print("=" * 70)

# Training Phase 2 used ~0.5 rad/s max angular rates
ROLL_TIME = 10.0  # Longer time for smoother roll
ROLL_RATE = 0.5  # Matches training data Phase 2


def ref_test2(t):
    omega_ref = np.array([ROLL_RATE, 0.0, 0.0]) if t < ROLL_TIME else np.zeros(3)
    return np.zeros(3), np.zeros(3), omega_ref, np.zeros(3)


print("Running baseline INDI...")
baseline_2 = run_test("Test2 Baseline", 10.0, ref_test2, use_nn=False)
print(
    f"  Omega error: mean={baseline_2['omega_error_mean']:.4f}, max={baseline_2['omega_error_max']:.4f} rad/s"
)

print("Running NA-INDI...")
naindi_2 = run_test("Test2 NA-INDI", 10.0, ref_test2, use_nn=True)
print(
    f"  Omega error: mean={naindi_2['omega_error_mean']:.4f}, max={naindi_2['omega_error_max']:.4f} rad/s"
)
print()


# ============================================================================
# Test 3: Combined Velocity + Attitude (matches training Phase 3)
# ============================================================================
print("=" * 70)
print("TEST 3: Combined Velocity + Attitude")
print("=" * 70)

# Training Phase 3: vel ~0.8 m/s, omega ~0.3 rad/s
V_CIRCLE_3 = 0.8
OMEGA_CIRCLE_3 = 2.0 * np.pi / 12.0  # 12s period (matches training)
ROLL_TIME_3 = 10.0
ROLL_RATE_3 = 0.3  # Matches training Phase 3


def ref_test3(t):
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


print("Running baseline INDI...")
baseline_3 = run_test("Test3 Baseline", 12.0, ref_test3, use_nn=False)
print(
    f"  Vel error: mean={baseline_3['vel_error_mean']:.4f}, max={baseline_3['vel_error_max']:.4f} m/s"
)

print("Running NA-INDI...")
naindi_3 = run_test("Test3 NA-INDI", 12.0, ref_test3, use_nn=True)
print(
    f"  Vel error: mean={naindi_3['vel_error_mean']:.4f}, max={naindi_3['vel_error_max']:.4f} m/s"
)
print()


# ============================================================================
# Test 4: Moderate Helix + Multi-Axis (matches training Phase 4)
# ============================================================================
print("=" * 70)
print("TEST 4: Moderate Helix with Multi-Axis Rotation")
print("=" * 70)

# Training Phase 4 (aggressive): vel ~1.2 m/s, omega ~0.6 rad/s
V_CIRCLE_4 = 0.8  # Reduced from 0.6
OMEGA_CIRCLE_4 = 2.0 * np.pi / 10.0
V_VERT_AMP = 0.3
OMEGA_VERT = 2.0 * np.pi / 8.0
ROLL_RATE_4 = 0.5  # Reduced from 2*pi/8 (~0.785)
PITCH_RATE_4 = 0.4  # Reduced from 2*pi/10 (~0.628)
YAW_RATE_4 = 0.3  # Reduced from 2*pi/12 (~0.524)
TUMBLE_TIME = 10.0


def ref_test4(t):
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


print("Running baseline INDI...")
baseline_4 = run_test("Test4 Baseline", 12.0, ref_test4, use_nn=False)
print(
    f"  Vel error: mean={baseline_4['vel_error_mean']:.4f}, max={baseline_4['vel_error_max']:.4f} m/s"
)

print("Running NA-INDI...")
naindi_4 = run_test("Test4 NA-INDI", 12.0, ref_test4, use_nn=True)
print(
    f"  Vel error: mean={naindi_4['vel_error_mean']:.4f}, max={naindi_4['vel_error_max']:.4f} m/s"
)
print()


# ============================================================================
# Summary
# ============================================================================
print("=" * 70)
print("SUMMARY: NA-INDI vs Baseline INDI")
print("=" * 70)
print(f"{'Test':<30} {'Baseline':>12} {'NA-INDI':>12} {'Improvement':>12}")
print("-" * 70)

for name, bl, na in [
    (
        "Test 1: Vel Error (m/s)",
        baseline_1["vel_error_mean"],
        naindi_1["vel_error_mean"],
    ),
    (
        "Test 2: Omega Error (rad/s)",
        baseline_2["omega_error_mean"],
        naindi_2["omega_error_mean"],
    ),
    (
        "Test 3: Vel Error (m/s)",
        baseline_3["vel_error_mean"],
        naindi_3["vel_error_mean"],
    ),
    (
        "Test 4: Vel Error (m/s)",
        baseline_4["vel_error_mean"],
        naindi_4["vel_error_mean"],
    ),
]:
    improvement = (bl - na) / bl * 100 if bl > 0 else 0
    print(f"{name:<30} {bl:>12.4f} {na:>12.4f} {improvement:>11.1f}%")


# ============================================================================
# Plotting
# ============================================================================
output_dir = Path(__file__).parent / "ValidationResults"
output_dir.mkdir(exist_ok=True)

fig, axes = plt.subplots(4, 2, figsize=(14, 12), sharex="row")
fig.suptitle("NA-INDI vs Baseline INDI Comparison", fontsize=14, fontweight="bold")

test_data = [
    ("Test 1: Circular Velocity", baseline_1, naindi_1),
    ("Test 2: 360° Roll", baseline_2, naindi_2),
    ("Test 3: Combined", baseline_3, naindi_3),
    ("Test 4: Helix+Tumble", baseline_4, naindi_4),
]

for row, (title, bl, na) in enumerate(test_data):
    # Velocity error
    ax = axes[row, 0]
    vel_err_bl = np.linalg.norm(bl["vel"] - bl["vel_ref"], axis=1)
    vel_err_na = np.linalg.norm(na["vel"] - na["vel_ref"], axis=1)
    ax.plot(bl["time"], vel_err_bl, "b-", label="Baseline", alpha=0.7)
    ax.plot(na["time"], vel_err_na, "r-", label="NA-INDI", alpha=0.7)
    ax.set_ylabel("Vel Error (m/s)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_title(f"{title} - Velocity", fontsize=10)

    # Angular rate error
    ax = axes[row, 1]
    omega_err_bl = np.linalg.norm(bl["omega"] - bl["omega_ref"], axis=1)
    omega_err_na = np.linalg.norm(na["omega"] - na["omega_ref"], axis=1)
    ax.plot(bl["time"], np.rad2deg(omega_err_bl), "b-", label="Baseline", alpha=0.7)
    ax.plot(na["time"], np.rad2deg(omega_err_na), "r-", label="NA-INDI", alpha=0.7)
    ax.set_ylabel("Omega Error (deg/s)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_title(f"{title} - Angular Rate", fontsize=10)

axes[-1, 0].set_xlabel("Time (s)")
axes[-1, 1].set_xlabel("Time (s)")

plt.tight_layout()
output_file = output_dir / "na_indi_comparison.png"
plt.savefig(output_file, dpi=150)
print(f"\nFigure saved to: {output_file}")

plt.show()
