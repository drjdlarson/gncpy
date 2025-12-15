"""This script holds classes to generate trajectories and further the rate commands for the omnicopter's INDI controller."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

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


class TrajectoryGeneration:
    def __init__(self, duration, t0=0.0, pos_coeffs=None, quat_coeffs=None):
        self.duration = float(duration)
        if self.duration <= 0:
            raise ValueError("duration must be > 0")
        self.t0 = float(t0)
        self.pos_coeffs = None
        self.quat_coeffs = None
        if pos_coeffs is not None:
            self.set_pos_coeffs(pos_coeffs)
        if quat_coeffs is not None:
            self.set_quat_coeffs(quat_coeffs)

    def set_pos_coeffs(self, pos_coeffs):
        c = np.asarray(pos_coeffs, dtype=float)
        if c.ndim != 2 or c.shape[0] != 3:
            raise ValueError("pos_coeffs must have shape (3, N)")
        self.pos_coeffs = c

    def set_quat_coeffs(self, quat_coeffs):
        c = np.asarray(quat_coeffs, dtype=float)
        if c.ndim != 2 or c.shape[0] != 4:
            raise ValueError("quat_coeffs must have shape (4, N)")
        self.quat_coeffs = c

    def evaluate_chebyshev(self, x, N):
        x = np.asarray(x).reshape(-1, 1)
        m = x.shape[0]
        t = [np.ones((m, 1)), x]

        for k in range(2, N):
            t_k = 2 * x * t[-1] - t[-2]
            t.append(t_k)

        return np.hstack(t[:N])

    def _time_to_x(self, t):
        t = np.asarray(t, dtype=float).reshape(-1)
        tau = (t - self.t0) / self.duration
        x = 2.0 * tau - 1.0
        return np.clip(x, -1.0, 1.0)

    def _eval_chebyshev_U(self, x, K):
        x = np.asarray(x, dtype=float).reshape(-1, 1)
        m = x.shape[0]
        if K <= 0:
            return np.zeros((m, 0))
        if K == 1:
            return np.ones((m, 1))
        u = [np.ones((m, 1)), 2.0 * x]
        for _ in range(2, K):
            u.append(2.0 * x * u[-1] - u[-2])
        return np.hstack(u[:K])

    def _dTdx(self, x, N):
        x = np.asarray(x, dtype=float).reshape(-1, 1)
        m = x.shape[0]
        if N <= 0:
            return np.zeros((m, 0))
        if N == 1:
            return np.zeros((m, 1))
        if N == 2:
            return np.hstack([np.zeros((m, 1)), np.ones((m, 1))])
        U = self._eval_chebyshev_U(x, N - 1)
        d = np.zeros((m, N))
        d[:, 1] = 1.0
        n = np.arange(2, N, dtype=float)
        d[:, 2:] = U[:, 1:] * n[None, :]
        return d

    def evaluate(self, t):
        t = np.asarray(t, dtype=float).reshape(-1)
        x = self._time_to_x(t)
        dxdt = 2.0 / self.duration
        out = {"t": t}

        if self.pos_coeffs is not None:
            Np = self.pos_coeffs.shape[1]
            T = self.evaluate_chebyshev(x, Np)
            dT = self._dTdx(x, Np)
            out["p_ref"] = T @ self.pos_coeffs.T
            out["v_ref_ned"] = (dT @ self.pos_coeffs.T) * dxdt

        if self.quat_coeffs is not None:
            Nq = self.quat_coeffs.shape[1]
            Tq = self.evaluate_chebyshev(x, Nq)
            dTq = self._dTdx(x, Nq)
            q_raw = Tq @ self.quat_coeffs.T
            q = np.stack(
                [gmath.quat_normalize(q_raw[i]) for i in range(q_raw.shape[0])], axis=0
            )

            qd_raw = (dTq @ self.quat_coeffs.T) * dxdt
            dot = np.sum(q * qd_raw, axis=1, keepdims=True)
            qd = qd_raw - q * dot

            omega = np.zeros((q.shape[0], 3))
            for i in range(q.shape[0]):
                omega[i] = (
                    2.0 * gmath.quat_multiply(gmath.quat_conjugate(q[i]), qd[i])[1:4]
                )

            out["q_ref"] = q
            out["omega_ref_body"] = omega

        return out

    def sample(self, dt, t_final=None):
        dt = float(dt)
        if dt <= 0:
            raise ValueError("dt must be > 0")
        tf = self.duration if t_final is None else float(t_final)
        if tf <= 0:
            raise ValueError("t_final must be > 0")
        t = np.arange(self.t0, self.t0 + tf + 0.5 * dt, dt)
        return self.evaluate(t)


class CommandGeneration:
    def __init__(
        self,
        Kp_pos=(1.0, 1.0, 1.0),
        Ki_pos=(0.0, 0.0, 0.0),
        Kp_att=(4.0, 4.0, 4.0),
        Ki_att=(0.0, 0.0, 0.0),
        v_max=None,
        omega_max=None,
        v_out_frame="body",
        i_pos_max=None,
        i_att_max=None,
    ):
        self.Kp_pos = np.asarray(Kp_pos, dtype=float).reshape(3)
        self.Ki_pos = np.asarray(Ki_pos, dtype=float).reshape(3)
        self.Kp_att = np.asarray(Kp_att, dtype=float).reshape(3)
        self.Ki_att = np.asarray(Ki_att, dtype=float).reshape(3)
        self.v_max = None if v_max is None else float(v_max)
        self.omega_max = None if omega_max is None else float(omega_max)
        self.v_out_frame = str(v_out_frame)
        self.i_pos_max = (
            None if i_pos_max is None else np.asarray(i_pos_max, dtype=float).reshape(3)
        )
        self.i_att_max = (
            None if i_att_max is None else np.asarray(i_att_max, dtype=float).reshape(3)
        )
        self._i_pos = np.zeros(3)
        self._i_att = np.zeros(3)

    def reset_integrators(self):
        self._i_pos[:] = 0.0
        self._i_att[:] = 0.0

    def _clamp_norm(self, v, vmax):
        if vmax is None:
            return v
        v = np.asarray(v, dtype=float)
        n = np.linalg.norm(v)
        if n <= vmax:
            return v
        if n <= np.finfo(float).eps:
            return v
        return v * (vmax / n)

    def _clamp_vec(self, v, vmin, vmax):
        v = np.asarray(v, dtype=float)
        if vmin is None and vmax is None:
            return v
        if vmin is None:
            return np.minimum(v, vmax)
        if vmax is None:
            return np.maximum(v, vmin)
        return np.minimum(np.maximum(v, vmin), vmax)

    def compute(self, p, q, traj_eval, dt):
        p = np.asarray(p, dtype=float).reshape(3)
        q = gmath.quat_normalize(np.asarray(q, dtype=float).reshape(4))
        dt = float(dt)
        if dt <= 0:
            raise ValueError("dt must be > 0")

        out = {}

        if "p_ref" in traj_eval:
            p_ref = np.asarray(traj_eval["p_ref"], dtype=float).reshape(-1, 3)[-1]
            v_ref_ned = np.zeros(3)
            if "v_ref_ned" in traj_eval:
                v_ref_ned = np.asarray(traj_eval["v_ref_ned"], dtype=float).reshape(
                    -1, 3
                )[-1]

            e_p = p_ref - p
            self._i_pos = self._i_pos + e_p * dt
            if self.i_pos_max is not None:
                self._i_pos = self._clamp_vec(
                    self._i_pos, -self.i_pos_max, self.i_pos_max
                )

            v_cmd_ned = v_ref_ned + self.Kp_pos * e_p + self.Ki_pos * self._i_pos
            v_cmd_ned = self._clamp_norm(v_cmd_ned, self.v_max)
            out["v_cmd_ned"] = v_cmd_ned
            out["i_pos"] = self._i_pos.copy()

            if self.v_out_frame == "ned":
                out["v_cmd"] = v_cmd_ned
            elif self.v_out_frame == "body":
                dcm_ned_to_body = gmath.quat_to_dcm(q)
                out["v_cmd"] = dcm_ned_to_body @ v_cmd_ned
            else:
                raise ValueError("v_out_frame must be 'body' or 'ned'")

        if "q_ref" in traj_eval:
            q_ref = np.asarray(traj_eval["q_ref"], dtype=float).reshape(-1, 4)[-1]
            q_ref = gmath.quat_normalize(q_ref)

            omega_ref_body = np.zeros(3)
            if "omega_ref_body" in traj_eval:
                omega_ref_body = np.asarray(
                    traj_eval["omega_ref_body"], dtype=float
                ).reshape(-1, 3)[-1]

            q_err = gmath.quat_multiply(q_ref, gmath.quat_conjugate(q))
            if q_err[0] < 0:
                q_err = -q_err
            e_rot = 2.0 * q_err[1:4]

            self._i_att = self._i_att + e_rot * dt
            if self.i_att_max is not None:
                self._i_att = self._clamp_vec(
                    self._i_att, -self.i_att_max, self.i_att_max
                )

            omega_cmd = omega_ref_body + self.Kp_att * e_rot + self.Ki_att * self._i_att
            omega_cmd = self._clamp_norm(omega_cmd, self.omega_max)

            out["omega_cmd"] = omega_cmd
            out["q_err"] = q_err
            out["i_att"] = self._i_att.copy()

        return out


if __name__ == "__main__":

    duration = 8.0
    t0 = 0.0
    dt = 0.01

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

    traj = TrajectoryGeneration(
        duration=duration, t0=t0, pos_coeffs=pos_coeffs, quat_coeffs=quat_coeffs
    )
    cmd = CommandGeneration(
        Kp_pos=(0.8, 0.8, 0.8),
        Ki_pos=(0.05, 0.05, 0.05),
        Kp_att=(3.0, 3.0, 3.0),
        v_out_frame="body",
        i_pos_max=(2.0, 2.0, 2.0),
        i_att_max=(1.0, 1.0, 1.0),
    )

    tr = traj.sample(dt=dt)
    t = tr["t"]
    p_ref = tr["p_ref"]
    v_ref_ned = tr["v_ref_ned"]
    q_ref = tr["q_ref"]
    omega_ref_body = tr["omega_ref_body"]

    p = p_ref[0] + np.array([0.6, -0.4, 0.3])
    q = gmath.quat_normalize(
        gmath.quat_multiply(q_ref[0], gmath.euler_to_quat(0.15, -0.10, 0.25))
    )

    v_cmd_ned_hist = np.zeros((t.size, 3))
    omega_cmd_hist = np.zeros((t.size, 3))
    q_err_hist = np.zeros((t.size, 4))

    for k in range(t.size):
        traj_eval_k = {
            "t": np.array([t[k]]),
            "p_ref": p_ref[[k]],
            "v_ref_ned": v_ref_ned[[k]],
            "q_ref": q_ref[[k]],
            "omega_ref_body": omega_ref_body[[k]],
        }

        out = cmd.compute(p=p, q=q, traj_eval=traj_eval_k, dt=dt)

        v_cmd_ned = out.get("v_cmd_ned", np.zeros(3))
        omega_cmd = out.get("omega_cmd", np.zeros(3))

        v_cmd_ned_hist[k] = v_cmd_ned
        omega_cmd_hist[k] = omega_cmd
        if "q_err" in out:
            q_err_hist[k] = out["q_err"]

        p = p + v_cmd_ned * dt

        qdot = 0.5 * gmath.quat_multiply(
            q, np.array([0.0, omega_cmd[0], omega_cmd[1], omega_cmd[2]])
        )
        q = gmath.quat_normalize(q + qdot * dt)

    fig_pos = plt.figure()
    plt.plot(t, v_cmd_ned_hist[:, 0], label="v_cmd_ned_x")
    plt.plot(t, v_cmd_ned_hist[:, 1], label="v_cmd_ned_y")
    plt.plot(t, v_cmd_ned_hist[:, 2], label="v_cmd_ned_z")
    plt.title("Position Command (Velocity in NED)")
    plt.xlabel("t [s]")
    plt.ylabel("m/s")
    plt.legend()
    plt.grid(True)

    fig_3d = plt.figure()
    ax = fig_3d.add_subplot(111, projection="3d")
    ax.plot(p_ref[:, 0], p_ref[:, 1], p_ref[:, 2], label="p_ref")
    ax.set_title("3D Position: Reference")
    ax.set_xlabel("x (NED)")
    ax.set_ylabel("y (NED)")
    ax.set_zlabel("z (NED)")
    ax.legend()

    fig_att = plt.figure()
    plt.plot(t, omega_cmd_hist[:, 0], label="omega_cmd_x")
    plt.plot(t, omega_cmd_hist[:, 1], label="omega_cmd_y")
    plt.plot(t, omega_cmd_hist[:, 2], label="omega_cmd_z")
    plt.title("Orientation Command (Body Rates)")
    plt.xlabel("t [s]")
    plt.ylabel("rad/s")
    plt.legend()
    plt.grid(True)

    plt.show()
