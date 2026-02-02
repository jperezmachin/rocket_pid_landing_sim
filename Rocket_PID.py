from __future__ import annotations
import math
import random
from typing import List, Dict, Tuple

import matplotlib.pyplot as plt

from PID import PID, PIDConfig
from Rocket import Rocket2D, RocketParams, RocketState


def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def wrap_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class RocketController:
    def __init__(self, p: RocketParams):
        self.p = p

        self.pid_vz = PID(
            PIDConfig(
                kp=3.2,
                ki=0.45,
                kd=0.8,
                out_min=-8.0,
                out_max=8.0,
                integrator_limit=25.0,
                d_filter_alpha=0.92,
                dt_min=1e-4,
                dt_max=0.1,
            )
        )

        self.kp_theta = 7.0
        self.kd_omega = 2.5

        self.gimbal_prev = 0.0
        self.gimbal_rate = math.radians(60.0)

        self.kx_to_vx = 0.12
        self.vx_max_hi = 10.0
        self.vx_max_lo = 3.5

        self.kvx_to_theta = 0.22
        self.ki_x = 0.0015
        self.ix = 0.0
        self.ix_limit = 80.0

        self.theta_max_hi = math.radians(18.0)
        self.theta_max_mid = math.radians(10.0)
        self.theta_max_lo = math.radians(5.0)
        self.theta_max_touch = math.radians(2.0)

        self.vz_max_down = 18.0
        self.vz_min_down = 0.4
        self.k_sqrt = 1.7

    def reset(self) -> None:
        self.pid_vz.reset()
        self.gimbal_prev = 0.0
        self.ix = 0.0

    def scheduled_theta_max(self, z: float) -> float:
        if z < 3.0:
            return self.theta_max_touch
        if z < 12.0:
            return self.theta_max_lo
        if z < 40.0:
            return self.theta_max_mid
        return self.theta_max_hi

    def scheduled_vx_max(self, z: float) -> float:
        if z < 15.0:
            return self.vx_max_lo
        return self.vx_max_hi

    def desired_vz(self, z: float, x: float) -> float:
        z = max(z, 0.0)
        v = -clamp(self.k_sqrt * math.sqrt(z), self.vz_min_down, self.vz_max_down)
        if abs(x) > 25.0:
            v = max(v, -6.0)
        elif abs(x) > 12.0:
            v = max(v, -8.0)
        if z < 8.0:
            v = max(v, -2.2)
        if z < 2.5:
            v = max(v, -1.0)
        return v

    def attitude_pd(self, theta_des: float, theta: float, omega: float) -> float:
        e = wrap_pi(theta_des - theta)
        g = self.kp_theta * e - self.kd_omega * omega
        return clamp(g, -self.p.gimbal_max, self.p.gimbal_max)

    def update(self, s: RocketState, dt: float) -> Tuple[float, float, float]:
        p = self.p
        dt = clamp(dt, 1e-4, 0.1)

        theta_max = self.scheduled_theta_max(s.z)
        vx_max = self.scheduled_vx_max(s.z)

        if s.z > 2.0:
            self.ix += s.x * dt
            self.ix = clamp(self.ix, -self.ix_limit, self.ix_limit)

        vx_des = clamp(-self.kx_to_vx * s.x, -vx_max, vx_max)
        vx_err = s.vx - vx_des

        theta_des_raw = -(self.kvx_to_theta * vx_err + self.ki_x * self.ix)
        theta_des = clamp(theta_des_raw, -theta_max, theta_max)

        if abs(wrap_pi(s.theta)) > math.radians(45.0):
            theta_des = 0.0

        gimbal_cmd = self.attitude_pd(theta_des, s.theta, s.omega)

        dg = self.gimbal_rate * dt
        gimbal = clamp(gimbal_cmd, self.gimbal_prev - dg, self.gimbal_prev + dg)
        self.gimbal_prev = gimbal

        vz_des = self.desired_vz(s.z, s.x)
        az_cmd = self.pid_vz.update(vz_des, s.vz, dt)

        cos_t = math.cos(s.theta)
        if cos_t <= 0.05:
            throttle = 0.15
        else:
            thrust_cmd = s.mass * (p.g + az_cmd) / cos_t
            throttle = clamp(thrust_cmd / p.max_thrust, 0.0, 1.0)

        if s.z <= 0.0 and abs(s.vz) < 0.25:
            throttle = 0.0

        return throttle, gimbal, math.degrees(theta_des)


def simulate(seed: int = 3) -> Dict[str, List[float]]:
    random.seed(seed)

    p = RocketParams()
    p.I = 40000.0
    p.rot_damping = 200.0

    s = RocketState(
        x=55.0,
        z=260.0,
        vx=-6.0,
        vz=-18.0,
        theta=math.radians(10.0),
        omega=math.radians(-6.0),
        mass=p.mass0,
    )

    rocket = Rocket2D(p, s)
    ctrl = RocketController(p)

    dt = 0.02
    t_end = 40.0
    steps = int(t_end / dt)

    wind_base = 4.0
    wind_amp = 6.0

    noise_x = 0.15
    noise_z = 0.15
    noise_vx = 0.08
    noise_vz = 0.08
    noise_theta = math.radians(0.2)

    log: Dict[str, List[float]] = {k: [] for k in [
        "t", "x", "z", "vx", "vz", "theta", "omega", "mass",
        "throttle", "gimbal", "wind", "theta_des", "crash"
    ]}

    crashed = False

    for i in range(steps):
        t = i * dt
        wind_x = wind_base + wind_amp * math.sin(0.55 * t) + 1.2 * math.sin(1.7 * t)

        meas = RocketState(
            x=s.x + random.gauss(0.0, noise_x),
            z=s.z + random.gauss(0.0, noise_z),
            vx=s.vx + random.gauss(0.0, noise_vx),
            vz=s.vz + random.gauss(0.0, noise_vz),
            theta=s.theta + random.gauss(0.0, noise_theta),
            omega=s.omega,
            mass=s.mass,
        )

        throttle, gimbal, theta_des = ctrl.update(meas, dt)

        if s.z < 2.0 and s.vz < -8.0:
            crashed = True

        rocket.step(throttle, gimbal, dt, wind_x=wind_x)

        log["t"].append(t)
        log["x"].append(s.x)
        log["z"].append(s.z)
        log["vx"].append(s.vx)
        log["vz"].append(s.vz)
        log["theta"].append(s.theta)
        log["omega"].append(s.omega)
        log["mass"].append(s.mass)
        log["throttle"].append(throttle)
        log["gimbal"].append(gimbal)
        log["wind"].append(wind_x)
        log["theta_des"].append(theta_des)
        log["crash"].append(1.0 if crashed else 0.0)

        if crashed and s.z <= 0.0:
            break

        if s.z <= 0.0 and abs(s.vz) < 0.3 and abs(s.vx) < 0.5 and abs(wrap_pi(s.theta)) < math.radians(4.0):
            break

    return log


def plot_log(log: Dict[str, List[float]]) -> None:
    t = log["t"]

    def new_fig(title: str):
        plt.figure()
        plt.title(title)
        plt.grid(True)

    new_fig("Position")
    plt.plot(t, log["x"], label="x (m)")
    plt.plot(t, log["z"], label="z (m)")
    plt.legend()

    new_fig("Velocity")
    plt.plot(t, log["vx"], label="vx (m/s)")
    plt.plot(t, log["vz"], label="vz (m/s)")
    plt.legend()

    new_fig("Attitude")
    plt.plot(t, [math.degrees(wrap_pi(a)) for a in log["theta"]], label="theta (deg)")
    plt.plot(t, [math.degrees(w) for w in log["omega"]], label="omega (deg/s)")
    plt.legend()

    new_fig("Control — Throttle")
    plt.plot(t, log["throttle"], label="throttle (0..1)")
    plt.legend()

    new_fig("Control — Gimbal / Wind")
    plt.plot(t, [math.degrees(g) for g in log["gimbal"]], label="gimbal (deg)")
    plt.plot(t, log["wind"], label="wind_x (m/s)")
    plt.legend()

    new_fig("Theta_des")
    plt.plot(t, log["theta_des"], label="theta_des (deg)")
    plt.legend()

    new_fig("Mass")
    plt.plot(t, log["mass"], label="mass (kg)")
    plt.legend()

    plt.show()


def main() -> None:
    log = simulate(seed=3)

    xf, zf = log["x"][-1], log["z"][-1]
    vxf, vzf = log["vx"][-1], log["vz"][-1]
    thf = math.degrees(wrap_pi(log["theta"][-1]))
    crash_flag = bool(log["crash"][-1] > 0.5)

    print(f"Final: x={xf:.2f} z={zf:.2f} vx={vxf:.2f} vz={vzf:.2f} theta={thf:.2f}deg  crash={crash_flag}")

    p = RocketParams()
    gmax = p.gimbal_max
    g_sat = sum(1 for g in log["gimbal"] if abs(g) >= (gmax - 1e-6)) / max(1, len(log["gimbal"]))
    th_sat = sum(1 for td in log["theta_des"] if abs(td) >= 17.999) / max(1, len(log["theta_des"]))
    thr_sat = sum(1 for u in log["throttle"] if u >= 0.999) / max(1, len(log["throttle"]))

    print(f"gimbal saturation fraction: {g_sat:.3f}")
    print(f"theta_des saturation fraction: {th_sat:.3f}")
    print(f"throttle at 1.0 fraction: {thr_sat:.3f}")

    plot_log(log)


if __name__ == "__main__":
    main()
