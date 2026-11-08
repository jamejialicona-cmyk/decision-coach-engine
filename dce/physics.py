"""Vectorized flight physics for a spinning sphere: gravity, quadratic drag,
spin-induced lift (Magnus), and one ground contact.

Many trajectories are integrated at once with numpy (RK4, fixed step), which is
what makes counterfactual simulation and RL affordable in pure Python.

Frame: x lateral, y forward (toward the other side), z up. Spin is given as
signed rpm about the axis perpendicular to the horizontal direction of travel:
positive = forward roll ("topspin"), negative = backspin ("slice").

The default constants describe a generic light sphere. They are textbook-level
values chosen for plausibility, NOT a calibrated model of any real ball.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Z = np.array([0.0, 0.0, 1.0])


@dataclass(frozen=True)
class BodyParams:
    mass: float = 0.060       # kg
    radius: float = 0.035     # m
    cd: float = 0.55          # drag coefficient
    rho: float = 1.2          # air density, kg/m^3
    g: float = 9.81

    @property
    def area(self) -> float:
        return np.pi * self.radius ** 2


@dataclass(frozen=True)
class SurfaceParams:
    restitution: float = 0.75   # vertical bounce
    grip: float = 0.35          # share of contact-point slip removed per bounce (0..1)


@dataclass
class Flight:
    """Result of simulating N launches up to their first ground contact."""
    landed: np.ndarray        # (N,) bool: touched the ground within t_max
    t_land: np.ndarray        # (N,) s
    pos_land: np.ndarray      # (N,3) m
    vel_out: np.ndarray       # (N,3) m/s, just after the bounce
    crossed: np.ndarray       # (N,) bool: reached the barrier plane y = y_barrier
    z_cross: np.ndarray       # (N,) m, height when crossing the barrier plane (nan if not)


def spin_vector(rpm: np.ndarray, azimuth: np.ndarray) -> np.ndarray:
    """Spin axis for a ball travelling along azimuth (rad, 0 = +y, positive toward +x)."""
    w = rpm * 2.0 * np.pi / 60.0
    d = np.stack([np.sin(azimuth), np.cos(azimuth), np.zeros_like(azimuth)], axis=1)
    return w[:, None] * np.cross(Z, d)


def launch_velocity(speed: np.ndarray, azimuth: np.ndarray, elevation: np.ndarray) -> np.ndarray:
    ch = np.cos(elevation)
    return np.stack([speed * ch * np.sin(azimuth), speed * ch * np.cos(azimuth),
                     speed * np.sin(elevation)], axis=1)


def _accel(v: np.ndarray, omega: np.ndarray, body: BodyParams) -> np.ndarray:
    speed = np.linalg.norm(v, axis=1, keepdims=True)
    a = np.zeros_like(v)
    a[:, 2] = -body.g
    a -= (0.5 * body.rho * body.cd * body.area / body.mass) * speed * v      # drag
    w = np.linalg.norm(omega, axis=1, keepdims=True)
    cross = np.cross(omega, v)
    cn = np.linalg.norm(cross, axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        cl = np.where(w > 1e-9, 1.0 / (2.0 + speed / (body.radius * w)), 0.0)   # lift coefficient
        direction = np.where(cn > 1e-12, cross / cn, 0.0)
    a += (0.5 * body.rho * body.area / body.mass) * cl * speed ** 2 * direction  # Magnus
    return a


def _rk4(p, v, omega, dt, body):
    a1 = _accel(v, omega, body)
    p2, v2 = p + 0.5 * dt * v, v + 0.5 * dt * a1
    a2 = _accel(v2, omega, body)
    p3, v3 = p + 0.5 * dt * v2, v + 0.5 * dt * a2
    a3 = _accel(v3, omega, body)
    p4, v4 = p + dt * v3, v + dt * a3
    a4 = _accel(v4, omega, body)
    return (p + dt / 6.0 * (v + 2 * v2 + 2 * v3 + v4),
            v + dt / 6.0 * (a1 + 2 * a2 + 2 * a3 + a4))


def bounce(v_in: np.ndarray, omega: np.ndarray, surface: SurfaceParams, body: BodyParams) -> np.ndarray:
    """Simplified contact: vertical restitution plus partial removal of the slip at the contact point."""
    v = v_in.copy()
    contact = v[:, :2] - body.radius * np.cross(omega, Z)[:, :2]      # velocity of the contact point
    v[:, :2] = v[:, :2] - surface.grip * contact
    v[:, 2] = -surface.restitution * v_in[:, 2]
    return v


def simulate(p0: np.ndarray, v0: np.ndarray, omega: np.ndarray, *, y_barrier: float | None = None,
             body: BodyParams = BodyParams(), surface: SurfaceParams = SurfaceParams(),
             dt: float = 1.0 / 120.0, t_max: float = 4.0) -> Flight:
    """Integrate N launches until each first touches z = 0."""
    n = p0.shape[0]
    p, v = p0.astype(float).copy(), v0.astype(float).copy()
    alive = np.ones(n, bool)
    landed = np.zeros(n, bool)
    t_land = np.full(n, np.nan)
    pos_land = np.full((n, 3), np.nan)
    vel_out = np.full((n, 3), np.nan)
    crossed = np.zeros(n, bool)
    z_cross = np.full(n, np.nan)
    t = 0.0
    while alive.any() and t < t_max:
        idx = np.nonzero(alive)[0]
        pn, vn = _rk4(p[idx], v[idx], omega[idx], dt, body)
        if y_barrier is not None:
            hit = (p[idx, 1] < y_barrier) & (pn[:, 1] >= y_barrier) & ~crossed[idx]
            if hit.any():
                k = idx[hit]
                f = (y_barrier - p[k, 1]) / (pn[hit, 1] - p[k, 1])
                z_cross[k] = p[k, 2] + f * (pn[hit, 2] - p[k, 2])
                crossed[k] = True
        down = pn[:, 2] <= 0.0
        if down.any():
            k = idx[down]
            f = p[k, 2] / (p[k, 2] - pn[down, 2])
            pos_hit = p[k] + f[:, None] * (pn[down] - p[k])
            v_hit = v[k] + f[:, None] * (vn[down] - v[k])
            pos_hit[:, 2] = 0.0
            landed[k] = True
            t_land[k] = t + f * dt
            pos_land[k] = pos_hit
            vel_out[k] = bounce(v_hit, omega[k], surface, body)
            alive[k] = False
        keep = ~down
        p[idx[keep]], v[idx[keep]] = pn[keep], vn[keep]
        t += dt
    return Flight(landed, t_land, pos_land, vel_out, crossed, z_cross)
