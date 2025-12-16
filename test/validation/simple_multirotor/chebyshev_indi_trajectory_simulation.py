"""Closed-loop omnicopter INDI simulation driven by a Chebyshev trajectory.

This script builds a single position + attitude trajectory using the
`TrajectoryGeneration` class (Chebyshev polynomials) and then uses the
resulting velocity and angular-rate references as commands for the
`test_omnicopter_indi_maneuvers`-style INDI simulation.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import gncpy.math as gmath
from gncpy.control.INDI import INDI
from gncpy.dynamics.aircraft.complex_multirotor import ComplexMultirotor, v_smap_quat
from gncpy.dynamics.aircraft.simple_multirotor import Effector, e_smap, yaml

from TrajectoryGeneration import CommandGeneration, TrajectoryGeneration


class MotorDynamicsEffector(Effector):
    """First-order motor dynamics model.

    Models motor response as a first-order lag:
        omega_i = (1/tau_mot) * (omega_cmd,i - omega_i)
    """

    def __init__(self, num_motors, tau_mot, initial_state=None):
        self.num_motors = num_motors
        self.tau_mot = tau_mot

        if initial_state is not None:
            self.state = np.array(initial_state).flatten().copy()
        else:
            self.state = np.zeros(num_motors)

    def set_initial_state(self, initial_state):
        """Set the initial motor state."""
        self.state = np.array(initial_state).flatten().copy()

    def step(self, input_cmds, dt):
        """Propagate motor dynamics one timestep using exact solution."""
        input_cmds = np.array(input_cmds).flatten()
        alpha = np.exp(-dt / self.tau_mot)
        self.state = input_cmds + (self.state - input_cmds) * alpha
        return self.state.copy()


class OmnicopterINDITrajectorySim:
    """Run INDI omnicopter simulation for a single Chebyshev trajectory.

    The trajectory is specified via Chebyshev coefficients for position
    and quaternion attitude. The resulting reference body-frame velocity
    and body angular-rate profiles are used as the INDI reference
    (i.e., vb_ref and omega_ref) instead of the analytical sinusoidal
    references used in `test_omnicopter_indi_maneuvers.py`.
    """

    # Default simulation / sensor parameters (mirrors test_omnicopter_indi_maneuvers)
    DT = 0.001  # Time step
    TAU_MOT = 0.032  # Motor time constant

    INITIAL_POSITION = np.array([0.0, 0.0, -10.0])  # NED (m)
    INITIAL_VELOCITY = np.array([0.0, 0.0, 0.0])
    INITIAL_ATTITUDE = np.array([0.0, 0.0, 0.0])  # Roll, pitch, yaw (deg)
    INITIAL_ANGULAR_VELOCITY = np.array([0.0, 0.0, 0.0])

    REF_LAT, REF_LON, TERRAIN_ALT = 34.0, -86.0, 0.0

    # INDI gains
    K_VEL = 5.0
    K_OMEGA = 10.0

    # Measurement noise
    SIGMA_VEL = 0.05  # m/s
    SIGMA_ACCEL = 0.1  # m/s^2
    SIGMA_OMEGA = 0.01  # rad/s

    BIAS_VEL = np.array([0.0, 0.0, 0.0])  # m/s
    BIAS_ACCEL = np.array([0.0, 0.0, 0.0])  # m/s^2
    BIAS_OMEGA = np.array([0.0, 0.0, 0.0])  # rad/s

    # Low-pass filter cutoffs (Hz)
    FC_VEL = 5.0
    FC_ACCEL = 20.0
    FC_OMEGA = 20.0
    FC_ALPHA = 5.0

    def __init__(
        self,
        duration=8.0,
        sim_time=None,
        pos_coeffs=None,
        quat_coeffs=None,
        dt=None,
        seed=42,
    ):
        """Create the simulator and build trajectory.

        Parameters
        ----------
        duration : float
            Duration of the Chebyshev trajectory (s).
        sim_time : float or None
            Total simulation time (s). If None, uses ``duration``.
        pos_coeffs : (3, Np) array-like or None
            Chebyshev coefficients for position. If None, a default example
            set (matching the demo in ``TrajectoryGeneration.py``) is used.
        quat_coeffs : (4, Nq) array-like or None
            Chebyshev coefficients for quaternion. If None, a default example
            set (matching the demo in ``TrajectoryGeneration.py``) is used.
        dt : float or None
            Simulation time step. If None, uses the class DT.
        seed : int
            Random seed for measurement noise.
        """
        self.duration = float(duration)
        self.sim_time = float(sim_time) if sim_time is not None else self.duration
        self.dt = float(dt) if dt is not None else self.DT

        np.random.seed(seed)

        # Use same example coefficients as the TrajectoryGeneration demo if not provided
        if pos_coeffs is None or quat_coeffs is None:
            pos_coeffs_demo, quat_coeffs_demo = self._default_coeffs()
            if pos_coeffs is None:
                pos_coeffs = pos_coeffs_demo
            if quat_coeffs is None:
                quat_coeffs = quat_coeffs_demo

        self.pos_coeffs = np.asarray(pos_coeffs, dtype=float)
        self.quat_coeffs = np.asarray(quat_coeffs, dtype=float)

        # Paths to config files
        base_dir = Path(__file__).parent
        self.lofi_config_file = base_dir / "omnicopter_config.yaml"
        self.hifi_config_file = base_dir / "omnicopter_config_hifi.yaml"

        # Will be filled in during setup
        self.num_motors = None
        self.motor_effector = None
        self.lofi_dyn = None
        self.hifi_dyn = None
        self.indi_ctrl = None
        self.hover_cmds_hifi = None

        # Trajectory samples (Chebyshev-based reference states)
        self.traj_time = None
        self.traj_p_ref = None
        self.traj_v_ref_ned = None
        self.traj_q_ref = None
        self.traj_omega_ref_body = None

        # Command generator that converts trajectory into v, omega commands
        self.cmd_gen = None

        self._setup_dynamics_and_controller()
        self._build_trajectory_and_command_generator()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _default_coeffs(self):
        """Return example Chebyshev coefficients (from TrajectoryGeneration demo)."""
        duration = self.duration
        # Match Np, Nq used in the TrajectoryGeneration demo
        Np = 5
        Nq = 6

        pos_coeffs = np.zeros((3, Np))
        pos_coeffs[0, 0] = 0.0
        pos_coeffs[0, 1] = 2.0
        pos_coeffs[0, 2] = -0.5
        pos_coeffs[1, 0] = 0.0
        pos_coeffs[1, 1] = -1.5
        pos_coeffs[1, 3] = 0.4
        pos_coeffs[2, 0] = -2.0
        pos_coeffs[2, 2] = 0.6
        pos_coeffs[2, 4] = -0.2

        quat_coeffs = np.zeros((4, Nq))
        quat_coeffs[0, 0] = 1.0
        quat_coeffs[3, 1] = 0.25
        quat_coeffs[2, 2] = -0.15

        # Duration is already part of TrajectoryGeneration, just return coeffs
        return pos_coeffs, quat_coeffs

    def _setup_dynamics_and_controller(self):
        """Set up LoFi/HiFi dynamics, motor effector, and INDI controller."""
        # LoFi dynamics for B0
        lofi_dyn = ComplexMultirotor(str(self.lofi_config_file))
        ned_mag = np.array([20.0, 5.0, 45.0])
        lofi_dyn.set_initial_conditions(
            self.INITIAL_POSITION,
            self.INITIAL_VELOCITY,
            self.INITIAL_ATTITUDE,
            self.INITIAL_ANGULAR_VELOCITY,
            self.REF_LAT,
            self.REF_LON,
            self.TERRAIN_ALT,
            ned_mag,
        )

        gravity = lofi_dyn.env.state[e_smap.gravity]

        mass = lofi_dyn.vehicle.params.mass.mass_kg
        inertia = np.array(lofi_dyn.vehicle.params.mass.inertia_kgm2)
        num_motors = lofi_dyn.vehicle.params.motor.num_motors
        T_max = lofi_dyn.vehicle.params.prop.poly_thrust[0]

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

        # Hover motor commands (LoFi -> HiFi conversion)
        g_mag = gravity[2]
        desired_accel = np.array([0.0, 0.0, -g_mag, 0.0, 0.0, 0.0])
        hover_cmds_lofi = B0_inv @ desired_accel

        with open(self.hifi_config_file, "r") as f:
            hifi_params = yaml.load(f)
        hifi_thrust_poly = hifi_params.prop.poly_thrust
        c2 = hifi_thrust_poly[0]
        lofi_thrust = T_max * hover_cmds_lofi
        hover_cmds_hifi = np.sign(lofi_thrust) * np.sqrt(np.abs(lofi_thrust) / c2)

        # Motor effector and HiFi dynamics
        motor_effector = MotorDynamicsEffector(
            num_motors=num_motors,
            tau_mot=self.TAU_MOT,
            initial_state=hover_cmds_hifi,
        )

        hifi_dyn = ComplexMultirotor(
            str(self.hifi_config_file), effector=motor_effector
        )

        # INDI controller
        K = np.diag(
            [
                self.K_VEL,
                self.K_VEL,
                self.K_VEL,
                self.K_OMEGA,
                self.K_OMEGA,
                self.K_OMEGA,
            ]
        )
        indi_ctrl = INDI(omit_A=True)
        indi_ctrl.set_state_model(dt=self.dt, K=K, B0=B0)

        self.num_motors = num_motors
        self.motor_effector = motor_effector
        self.lofi_dyn = lofi_dyn
        self.hifi_dyn = hifi_dyn
        self.indi_ctrl = indi_ctrl
        self.hover_cmds_hifi = hover_cmds_hifi

    def _build_trajectory_and_command_generator(self):
        """Generate trajectory samples and set up command generator."""
        traj = TrajectoryGeneration(
            duration=self.duration,
            t0=0.0,
            pos_coeffs=self.pos_coeffs,
            quat_coeffs=self.quat_coeffs,
        )
        tr = traj.sample(dt=self.dt, t_final=self.sim_time)

        self.traj_time = tr["t"]
        self.traj_p_ref = tr["p_ref"]
        self.traj_v_ref_ned = tr["v_ref_ned"]
        self.traj_q_ref = tr["q_ref"]
        self.traj_omega_ref_body = tr["omega_ref_body"]

        # Use the provided command generator to map trajectory -> v, omega commands.
        # Gains and limits mirror the demo in TrajectoryGeneration.__main__.
        self.cmd_gen = CommandGeneration(
            Kp_pos=(0.8, 0.8, 0.8),
            Ki_pos=(0.05, 0.05, 0.05),
            Kp_att=(3.0, 3.0, 3.0),
            v_out_frame="body",
            i_pos_max=(2.0, 2.0, 2.0),
            i_att_max=(1.0, 1.0, 1.0),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self, make_plots=True):
        """Run the INDI simulation following the Chebyshev trajectory.

        Parameters
        ----------
        make_plots : bool
            If True, generate simple plots at the end of the run.

        Returns
        -------
        results : dict
            Dictionary containing time histories of key variables.
        """
        dt = self.dt
        num_steps = int(self.sim_time / dt)

        # Reset dynamics to hover
        self.motor_effector.set_initial_state(self.hover_cmds_hifi)
        ned_mag = np.array([20.0, 5.0, 45.0])
        self.hifi_dyn.set_initial_conditions(
            self.INITIAL_POSITION,
            self.INITIAL_VELOCITY,
            self.INITIAL_ATTITUDE,
            self.INITIAL_ANGULAR_VELOCITY,
            self.REF_LAT,
            self.REF_LON,
            self.TERRAIN_ALT,
            ned_mag,
        )
        self.hifi_dyn.vehicle.takenoff = True

        # Storage
        time_hist = np.zeros(num_steps)
        body_vel_hist = np.zeros((num_steps, 3))
        body_vel_ref_hist = np.zeros((num_steps, 3))
        body_omega_hist = np.zeros((num_steps, 3))
        body_omega_ref_hist = np.zeros((num_steps, 3))
        ned_pos_hist = np.zeros((num_steps, 3))
        euler_hist = np.zeros((num_steps, 3))
        euler_ref_hist = np.zeros((num_steps, 3))
        cmd_hist = np.zeros((num_steps, self.num_motors))

        # Initial conditions
        cur_state = self.hifi_dyn.vehicle.state.copy()
        cur_input = self.hover_cmds_hifi.copy()

        # Sensor filter states
        vel_filt = np.zeros(3)
        accel_filt = np.zeros(3)
        omega_filt = np.zeros(3)
        alpha_filt = np.zeros(3)

        # Filter coefficients
        alpha_vel = dt / (dt + 1.0 / (2.0 * np.pi * self.FC_VEL))
        alpha_accel = dt / (dt + 1.0 / (2.0 * np.pi * self.FC_ACCEL))
        alpha_omega = dt / (dt + 1.0 / (2.0 * np.pi * self.FC_OMEGA))
        alpha_alpha = dt / (dt + 1.0 / (2.0 * np.pi * self.FC_ALPHA))

        omega_prev = np.zeros(3)

        prev_vb_ref = np.zeros(3)
        prev_omega_ref = np.zeros(3)

        for ii in range(num_steps):
            tt = ii * dt
            time_hist[ii] = tt

            # True state
            body_vel_true = cur_state[v_smap_quat.body_vel].flatten()
            body_accel_true = cur_state[v_smap_quat.body_accel].flatten()
            body_omega_true = cur_state[v_smap_quat.body_rot_rate].flatten()
            quat = cur_state[v_smap_quat.quat].flatten()

            # Sensor model
            vel_noise = np.random.normal(0, self.SIGMA_VEL, 3)
            body_vel_meas = body_vel_true + vel_noise + self.BIAS_VEL
            vel_filt = alpha_vel * body_vel_meas + (1.0 - alpha_vel) * vel_filt

            accel_noise = np.random.normal(0, self.SIGMA_ACCEL, 3)
            body_accel_meas = body_accel_true + accel_noise + self.BIAS_ACCEL
            accel_filt = (
                alpha_accel * body_accel_meas + (1.0 - alpha_accel) * accel_filt
            )

            omega_noise = np.random.normal(0, self.SIGMA_OMEGA, 3)
            body_omega_meas = body_omega_true + omega_noise + self.BIAS_OMEGA
            omega_filt = (
                alpha_omega * body_omega_meas + (1.0 - alpha_omega) * omega_filt
            )

            if ii == 0:
                alpha_meas = np.zeros(3)
            else:
                alpha_meas = (omega_filt - omega_prev) / dt
            alpha_filt = alpha_alpha * alpha_meas + (1.0 - alpha_alpha) * alpha_filt
            omega_prev = omega_filt.copy()

            x = np.concatenate([vel_filt, omega_filt])
            x_dot = np.concatenate([accel_filt, alpha_filt])

            # ------------------------------------------------------------------
            # Reference from Chebyshev trajectory via CommandGeneration
            # ------------------------------------------------------------------
            if ii < self.traj_time.size:
                k = ii
            else:
                k = self.traj_time.size - 1

            # Current position (NED) and attitude (quat) for command generator
            p_ned = cur_state[v_smap_quat.ned_pos].flatten()
            q = cur_state[v_smap_quat.quat].flatten()

            traj_eval_k = {
                "t": np.array([self.traj_time[k]]),
                "p_ref": self.traj_p_ref[[k]],
                "v_ref_ned": self.traj_v_ref_ned[[k]],
                "q_ref": self.traj_q_ref[[k]],
                "omega_ref_body": self.traj_omega_ref_body[[k]],
            }

            cmd_out = self.cmd_gen.compute(
                p=p_ned,
                q=q,
                traj_eval=traj_eval_k,
                dt=dt,
            )

            # v_cmd is in body frame (by construction v_out_frame="body"),
            # omega_cmd is body angular-rate command.
            vb_ref = cmd_out.get("v_cmd", np.zeros(3))
            omega_ref = cmd_out.get("omega_cmd", np.zeros(3))

            if ii == 0:
                vb_ref_dot = np.zeros(3)
                omega_ref_dot = np.zeros(3)
            else:
                vb_ref_dot = (vb_ref - prev_vb_ref) * (1.0 / dt)
                omega_ref_dot = (omega_ref - prev_omega_ref) * (1.0 / dt)

            prev_vb_ref = vb_ref.copy()
            prev_omega_ref = omega_ref.copy()

            ref = np.concatenate([vb_ref, omega_ref])
            ref_dot = np.concatenate([vb_ref_dot, omega_ref_dot])

            # INDI control
            u_cmd = self.indi_ctrl.calculate_control(
                cur_time=tt,
                cur_state=x,
                cur_state_dot=x_dot,
                cur_input=cur_input,
                ref=ref,
                ref_dot=ref_dot,
            )
            u_cmd = np.clip(u_cmd, -1.0, 1.0)

            # Store histories
            body_vel_hist[ii, :] = body_vel_true
            body_vel_ref_hist[ii, :] = vb_ref
            body_omega_hist[ii, :] = body_omega_true
            body_omega_ref_hist[ii, :] = omega_ref
            ned_pos_hist[ii, :] = cur_state[v_smap_quat.ned_pos].flatten()
            cmd_hist[ii, :] = u_cmd

            roll, pitch, yaw = gmath.quat_to_euler(quat)
            euler_hist[ii, :] = np.rad2deg([roll, pitch, yaw])
            # Reference Euler angles from Chebyshev quaternion reference
            roll_ref, pitch_ref, yaw_ref = gmath.quat_to_euler(self.traj_q_ref[k])
            euler_ref_hist[ii, :] = np.rad2deg([roll_ref, pitch_ref, yaw_ref])

            # Propagate dynamics
            cur_state = self.hifi_dyn.propagate_state(dt, cur_state, u_cmd).flatten()
            cur_input = self.motor_effector.state.copy()

        results = {
            "t": time_hist,
            "body_vel": body_vel_hist,
            "body_vel_ref": body_vel_ref_hist,
            "body_omega": body_omega_hist,
            "body_omega_ref": body_omega_ref_hist,
            "ned_pos": ned_pos_hist,
            "euler_deg": euler_hist,
            "euler_ref_deg": euler_ref_hist,
            "u_cmd": cmd_hist,
        }

        if make_plots:
            self._plot_results(results)

        return results

    def _plot_results(self, res):
        """Simple plots of tracking performance."""
        t = res["t"]
        body_vel = res["body_vel"]
        body_vel_ref = res["body_vel_ref"]
        body_omega = res["body_omega"]
        body_omega_ref = res["body_omega_ref"]
        euler_deg = res["euler_deg"]
        euler_ref_deg = res["euler_ref_deg"]
        cmd = res["u_cmd"]

        fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
        fig.suptitle(
            "Chebyshev Trajectory Tracking with Omnicopter INDI",
            fontsize=14,
            fontweight="bold",
        )

        # Body velocity tracking
        ax = axes[0, 0]
        ax.plot(t, body_vel[:, 0], "r-", label="vb_x")
        ax.plot(t, body_vel_ref[:, 0], "r--", alpha=0.7, label="vb_x ref")
        ax.plot(t, body_vel[:, 1], "g-", label="vb_y")
        ax.plot(t, body_vel_ref[:, 1], "g--", alpha=0.7, label="vb_y ref")
        ax.plot(t, body_vel[:, 2], "b-", label="vb_z")
        ax.plot(t, body_vel_ref[:, 2], "b--", alpha=0.7, label="vb_z ref")
        ax.set_ylabel("Body Velocity (m/s)")
        ax.legend(loc="upper right", ncol=3, fontsize=8)
        ax.grid(True, alpha=0.3)

        # Body angular rate
        ax = axes[0, 1]
        ax.plot(t, np.rad2deg(body_omega[:, 0]), "r-", label="p")
        ax.plot(
            t,
            np.rad2deg(body_omega_ref[:, 0]),
            "r--",
            alpha=0.7,
            label="p ref",
        )
        ax.plot(t, np.rad2deg(body_omega[:, 1]), "g-", label="q")
        ax.plot(
            t,
            np.rad2deg(body_omega_ref[:, 1]),
            "g--",
            alpha=0.7,
            label="q ref",
        )
        ax.plot(t, np.rad2deg(body_omega[:, 2]), "b-", label="r")
        ax.plot(
            t,
            np.rad2deg(body_omega_ref[:, 2]),
            "b--",
            alpha=0.7,
            label="r ref",
        )
        ax.set_ylabel("Body Angular Rate (deg/s)")
        ax.legend(loc="upper right", ncol=3, fontsize=7)
        ax.grid(True, alpha=0.3)

        # Euler angles (true vs reference from Chebyshev trajectory)
        ax = axes[1, 0]
        ax.plot(t, euler_deg[:, 0], "r-", label="Roll")
        ax.plot(t, euler_ref_deg[:, 0], "r--", alpha=0.7, label="Roll ref")
        ax.plot(t, euler_deg[:, 1], "g-", label="Pitch")
        ax.plot(t, euler_ref_deg[:, 1], "g--", alpha=0.7, label="Pitch ref")
        ax.plot(t, euler_deg[:, 2], "b-", label="Yaw")
        ax.plot(t, euler_ref_deg[:, 2], "b--", alpha=0.7, label="Yaw ref")
        ax.set_ylabel("Euler Angles (deg)")
        ax.set_xlabel("Time (s)")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

        # Motor commands
        ax = axes[1, 1]
        for ii in range(self.num_motors):
            ax.plot(t, cmd[:, ii], alpha=0.7, label=f"u{ii}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Motor Commands")
        ax.legend(loc="upper right", ncol=4, fontsize=7)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()


if __name__ == "__main__":
    # Example: run the simulation with default Chebyshev coefficients
    sim = OmnicopterINDITrajectorySim(duration=8.0, sim_time=8.0)
    results = sim.run(make_plots=True)
    plt.show()
