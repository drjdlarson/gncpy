"""Proportional-integral-derivative controller utilities."""

import numpy as np


class PID:
    """Elementwise PID controller with optional derivative filtering.

    The derivative term is filtered with a first-order low-pass model and, by
    default, taken on the measurement to avoid derivative kick.
    """

    def __init__(
        self,
        kp=0.0,
        ki=0.0,
        kd=0.0,
        tau=0.0,
        u_min=None,
        u_max=None,
        derivative_on_measurement=True,
    ):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.tau = tau
        self.u_min = u_min
        self.u_max = u_max
        self.derivative_on_measurement = derivative_on_measurement

        self.reset()

    def reset(
        self,
        integral_state=None,
        deriv_state=None,
        prev_error=None,
        prev_measurement=None,
        last_time=None,
    ):
        """Reset the controller memory."""
        self.integral_state = self._as_array(integral_state)
        self.deriv_state = self._as_array(deriv_state)
        self.prev_error = self._as_array(prev_error)
        self.prev_measurement = self._as_array(prev_measurement)
        self.last_time = last_time

    def _as_array(self, value):
        if value is None:
            return None
        return np.asarray(value, dtype=float).reshape((-1,))

    def _broadcast_param(self, value, size):
        arr = np.asarray(value, dtype=float)
        if arr.ndim == 0:
            return np.full(size, float(arr))
        arr = arr.reshape((-1,))
        if arr.size == 1:
            return np.full(size, float(arr[0]))
        if arr.size != size:
            raise ValueError("Controller parameter size does not match signal size")
        return arr

    def _clip(self, value):
        out = value.copy()
        if self.u_min is not None:
            out = np.maximum(out, self._broadcast_param(self.u_min, out.size))
        if self.u_max is not None:
            out = np.minimum(out, self._broadcast_param(self.u_max, out.size))
        return out

    def calculate_control(self, cur_time, measurement, reference=0.0, dt=None):
        """Calculate the PID control output."""
        meas = np.asarray(measurement, dtype=float).reshape((-1,))
        ref = np.asarray(reference, dtype=float).reshape((-1,))
        if ref.size == 1 and meas.size > 1:
            ref = np.full(meas.size, float(ref[0]))
        elif ref.size != meas.size:
            raise ValueError("Reference size must match measurement size")

        kp = self._broadcast_param(self.kp, meas.size)
        ki = self._broadcast_param(self.ki, meas.size)
        kd = self._broadcast_param(self.kd, meas.size)
        tau = self._broadcast_param(self.tau, meas.size)

        err = ref - meas
        if dt is None:
            if self.last_time is None:
                dt = 0.0
            else:
                dt = cur_time - self.last_time
        dt = float(dt)
        if dt < 0:
            raise ValueError("dt must be non-negative")

        if self.integral_state is None or self.integral_state.size != meas.size:
            self.integral_state = np.zeros(meas.size)
        if self.deriv_state is None or self.deriv_state.size != meas.size:
            self.deriv_state = np.zeros(meas.size)

        if dt > 0:
            self.integral_state = self.integral_state + err * dt

            if self.derivative_on_measurement:
                if self.prev_measurement is None or self.prev_measurement.size != meas.size:
                    deriv_raw = np.zeros(meas.size)
                else:
                    deriv_raw = -(meas - self.prev_measurement) / dt
            else:
                if self.prev_error is None or self.prev_error.size != meas.size:
                    deriv_raw = np.zeros(meas.size)
                else:
                    deriv_raw = (err - self.prev_error) / dt

            filt_inds = tau > np.finfo(float).eps
            if np.any(filt_inds):
                alpha = np.exp(-dt / tau[filt_inds])
                self.deriv_state[filt_inds] = (
                    alpha * self.deriv_state[filt_inds]
                    + (1.0 - alpha) * deriv_raw[filt_inds]
                )
            self.deriv_state[~filt_inds] = deriv_raw[~filt_inds]

        unsat = kp * err + ki * self.integral_state + kd * self.deriv_state
        output = self._clip(unsat)

        if dt > 0 and (self.u_min is not None or self.u_max is not None):
            pushed_high = output < unsat
            pushed_low = output > unsat
            saturated = (pushed_high & (err > 0)) | (pushed_low & (err < 0))
            ki_active = saturated & (np.abs(ki) > np.finfo(float).eps)
            if np.any(ki_active):
                self.integral_state[ki_active] = (
                    output[ki_active]
                    - kp[ki_active] * err[ki_active]
                    - kd[ki_active] * self.deriv_state[ki_active]
                ) / ki[ki_active]
            no_int = saturated & ~ki_active
            if np.any(no_int):
                self.integral_state[no_int] = (
                    self.integral_state[no_int] - err[no_int] * dt
                )
            if np.any(saturated):
                unsat = kp * err + ki * self.integral_state + kd * self.deriv_state
                output = self._clip(unsat)

        self.prev_error = err.copy()
        self.prev_measurement = meas.copy()
        self.last_time = cur_time
        return output
