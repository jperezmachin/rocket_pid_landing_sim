from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple

def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x

@dataclass
class PIDConfig:
    kp: float
    ki: float
    kd: float
    out_min: Optional[float] = None
    out_max: Optional[float] = None
    integrator_limit: Optional[float] = None
    # Derivative low-pass filter (0=no filter, closer to 1 = heavier smoothing)
    d_filter_alpha: float = 0.85
    dt_min: float = 1e-4
    dt_max: float = 0.1


class PID:
    
    def __init__(self, cfg: PIDConfig):
        self.cfg = cfg
        self.integral = 0.0
        self.prev_meas: Optional[float] = None
        self.prev_u: float = 0.0
        self.d_filt: float = 0.0

    def reset(self) -> None:
        self.integral = 0.0
        self.prev_meas = None
        self.prev_u = 0.0
        self.d_filt = 0.0

    def update(self, setpoint: float, measurement: float, dt: float) -> float:
        # dt guards
        dt = clamp(dt, self.cfg.dt_min, self.cfg.dt_max)

        error = setpoint - measurement

        # Derivative on measurement: d = -d(meas)/dt
        if self.prev_meas is None:
            d_raw = 0.0
        else:
            d_raw = -(measurement - self.prev_meas) / dt
        self.prev_meas = measurement

        # Low-pass derivative
        a = clamp(self.cfg.d_filter_alpha, 0.0, 0.999)
        self.d_filt = a * self.d_filt + (1.0 - a) * d_raw

        # Provisional integral step (may be undone if saturated)
        self.integral += error * dt
        if self.cfg.integrator_limit is not None:
            lim = abs(self.cfg.integrator_limit)
            self.integral = clamp(self.integral, -lim, lim)

        u_unsat = (
            self.cfg.kp * error
            + self.cfg.ki * self.integral
            + self.cfg.kd * self.d_filt
        )

        u = u_unsat
        saturated = False
        if self.cfg.out_min is not None:
            if u < self.cfg.out_min:
                u = self.cfg.out_min
                saturated = True
        if self.cfg.out_max is not None:
            if u > self.cfg.out_max:
                u = self.cfg.out_max
                saturated = True

        # Anti-windup: conditional integration
        # If we are saturated AND the error would push further into saturation, undo this step.
        if saturated and self.cfg.ki != 0.0:
            if (u == self.cfg.out_max and error > 0) or (u == self.cfg.out_min and error < 0):
                self.integral -= error * dt  # revert

        self.prev_u = u
        return u