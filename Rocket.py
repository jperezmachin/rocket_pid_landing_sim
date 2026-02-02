import math
import random
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple

def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x

@dataclass
class RocketParams:
    g: float = 9.81

    # Mass properties
    mass0: float = 1200.0       # kg
    dry_mass: float = 800.0     # kg
    I: float = 40000.0            # kg*m^2 (moment of inertia about COM)
    lever_arm: float = 3.0      # m (engine torque arm)

    # Engine
    max_thrust: float = 22000.0     # N
    isp: float = 240.0             # s
    gimbal_max: float = math.radians(12)  # rad

    # Aerodynamics (simple quadratic drag)
    rho: float = 1.225
    cd: float = 0.6
    area: float = 1.2             # m^2
    rot_damping: float = 200.0     # N*m*s (angular damping)

@dataclass
class RocketState:
    x: float
    z: float
    vx: float
    vz: float
    theta: float    # rad; 0 means upright
    omega: float    # rad/s
    mass: float


class Rocket2D:
    def __init__(self, p: RocketParams, s: RocketState):
        self.p = p
        self.s = s

    def step(self, throttle_cmd: float, gimbal_cmd: float, dt: float, wind_x: float = 0.0) -> None:
        p, s = self.p, self.s

        throttle = clamp(throttle_cmd, 0.0, 1.0)
        gimbal = clamp(gimbal_cmd, -p.gimbal_max, p.gimbal_max)

        # Fuel check
        fuel_left = max(0.0, s.mass - p.dry_mass)
        if fuel_left <= 0.0:
            thrust = 0.0
            mdot = 0.0
            throttle = 0.0
        else:
            thrust = throttle * p.max_thrust
            mdot = thrust / (p.isp * p.g)  # kg/s

        # Drag (wind)
        vrel_x = s.vx - wind_x
        vrel_z = s.vz
        vrel = math.hypot(vrel_x, vrel_z)
        # Quadratic drag vector ~ v*|v|
        drag_fx = 0.5 * p.rho * p.cd * p.area * vrel * vrel_x
        drag_fz = 0.5 * p.rho * p.cd * p.area * vrel * vrel_z

        # Thrust direction: body angle + gimbal
        # theta=0 means thrust points "up" in +z when gimbal=0
        ang = s.theta + gimbal
        thrust_fx = thrust * math.sin(ang)
        thrust_fz = thrust * math.cos(ang)

        # Translational acceleration
        ax = (thrust_fx - drag_fx) / max(s.mass, 1e-6)
        az = (thrust_fz - drag_fz) / max(s.mass, 1e-6) - p.g

        # Rotational dynamics (torque from gimbal)
        torque = thrust * p.lever_arm * math.sin(gimbal) - p.rot_damping * s.omega
        alpha = torque / p.I

        # Integrate
        s.vx += ax * dt
        s.vz += az * dt
        s.x += s.vx * dt
        s.z += s.vz * dt

        s.omega += alpha * dt
        s.theta += s.omega * dt

        s.mass = max(p.dry_mass, s.mass - mdot * dt)

        # Ground contact
        if s.z < 0.0:
            s.z = 0.0
            if s.vz < 0.0:
                s.vz = 0.0
            # friction on touchdown
            s.vx *= 0.9
            s.omega *= 0.7