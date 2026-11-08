"""A small, abstract two-sided "exchange" used to study decisions under physics.

One agent (the *actor*) must send a projectile across a barrier into the far
half of a bounded arena, where an *opponent* tries to reach it. Everything is
generic: no sport rules, no scoring system, no calibrated constants.

A decision is one of N discrete actions (lateral aim x launch angle x speed x
spin). Outcomes are physical: the projectile either fails (blocked by the
barrier, lands out of bounds), is returned, or is not reached.

    reward = -1                 the actor fails
              0 .. +1           1 - P(opponent reaches it); 0 = easy to return, 1 = unreachable

The reward is an *expected* value over the opponent's return chance, which keeps
learning signals low-variance; `kind` additionally samples the discrete outcome.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import product

import numpy as np

from .physics import BodyParams, SurfaceParams, launch_velocity, simulate, spin_vector

ERROR, CONTINUE, WIN = -1, 0, 1

AZIMUTH_DEG = (-6.0, 0.0, 6.0)
ELEVATION_DEG = (4.0, 7.0, 10.0)
SPEED = (18.0, 24.0)
SPIN_RPM = (2000.0, -1000.0)
ACTIONS = tuple(product(AZIMUTH_DEG, ELEVATION_DEG, SPEED, SPIN_RPM))   # 36
N_ACTIONS = len(ACTIONS)

LEVELS = {  # opponent level -> (top speed m/s, reaction time s)
    0: (3.0, 0.38),
    1: (4.2, 0.28),
    2: (5.4, 0.20),
}
N_FEATURES = 6


def describe(idx: int) -> str:
    az, el, sp, rpm = ACTIONS[idx]
    side = "left" if az < 0 else "right" if az > 0 else "centre"
    arc = {4.0: "flat", 7.0: "medium", 10.0: "high"}[el]
    pace = "fast" if sp >= 24 else "moderate"
    spin = "forward spin" if rpm > 0 else "backspin"
    return f"{pace}, {arc} arc, {side}, {spin}"


@dataclass(frozen=True)
class ArenaParams:
    half_width: float = 3.0     # |x| <= half_width is in bounds
    barrier_y: float = 10.0     # barrier plane
    barrier_h: float = 0.8      # must clear this height
    far_y: float = 19.0         # far edge of the arena
    launch_z: float = 1.0
    body: BodyParams = BodyParams()
    surface: SurfaceParams = SurfaceParams()


@dataclass
class States:
    """A batch of decision contexts (struct of arrays)."""
    xa: np.ndarray        # actor lateral position
    xb: np.ndarray        # opponent lateral position
    yb: np.ndarray        # opponent depth
    t_avail: np.ndarray   # seconds the actor had to prepare (time pressure)
    level: np.ndarray     # opponent level index

    def __len__(self):
        return len(self.xa)

    def features(self) -> np.ndarray:
        speed = np.array([LEVELS[int(l)][0] for l in self.level])
        react = np.array([LEVELS[int(l)][1] for l in self.level])
        return np.stack([self.xa / 2.0, self.xb / 2.5, (self.yb - 17.0) / 1.5,
                         (self.t_avail - 1.0) / 0.6, (speed - 4.2) / 1.2, (react - 0.28) / 0.1], axis=1)

    def take(self, idx) -> "States":
        return States(self.xa[idx], self.xb[idx], self.yb[idx], self.t_avail[idx], self.level[idx])

    def repeat(self, k: int) -> "States":
        return States(*(np.repeat(a, k) for a in (self.xa, self.xb, self.yb, self.t_avail, self.level)))


@dataclass
class Outcome:
    reward: np.ndarray     # expected reward (see module docstring)
    kind: np.ndarray       # sampled: ERROR / CONTINUE / WIN
    p_return: np.ndarray   # opponent's chance to reach it (nan when the actor fails)
    land_x: np.ndarray
    land_y: np.ndarray


def sample_states(n: int, rng: np.random.Generator, level: int | None = None) -> States:
    lv = rng.integers(0, 3, n) if level is None else np.full(n, level)
    return States(xa=rng.uniform(-2.0, 2.0, n), xb=rng.uniform(-2.5, 2.5, n),
                  yb=rng.uniform(15.5, 18.5, n), t_avail=rng.uniform(0.4, 1.6, n), level=lv)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def opponent_reach_prob(land: np.ndarray, vel_out: np.ndarray, t_land: np.ndarray,
                        xb: np.ndarray, yb: np.ndarray, level: np.ndarray) -> np.ndarray:
    """Chance the opponent gets to the projectile after its bounce.

    The opponent runs at a level-dependent speed after a reaction delay and may
    intercept anywhere along the first 3 m of the post-bounce path. Slack is
    turned into a probability with a logistic.
    """
    top = np.array([LEVELS[int(l)][0] for l in level])
    react = np.array([LEVELS[int(l)][1] for l in level])
    vh = vel_out[:, :2]
    sh = np.maximum(np.linalg.norm(vh, axis=1), 1e-6)
    u = vh / sh[:, None]
    best = np.full(len(land), -np.inf)
    for d in np.linspace(0.0, 3.0, 7):
        q = land[:, :2] + d * u
        dist = np.hypot(q[:, 0] - xb, q[:, 1] - yb)
        reach = top * np.maximum(t_land + d / sh - react, 0.0)
        best = np.maximum(best, reach - dist)
    return _sigmoid(best / 0.5)


def play(states: States, actions: np.ndarray, rng: np.random.Generator,
         arena: ArenaParams = ArenaParams()) -> Outcome:
    """Execute one action per state, with execution noise that grows under time pressure."""
    m = len(states)
    a = np.array(ACTIONS)[actions]
    az, el, sp, rpm = np.radians(a[:, 0]), np.radians(a[:, 1]), a[:, 2], a[:, 3]
    pressure = 1.0 + 1.5 * np.maximum(0.0, 1.0 - states.t_avail)
    pace = np.where(sp >= 24.0, 1.3, 1.0)                 # harder actions are less precise
    az = az + rng.normal(0, np.radians(1.2), m) * pressure * pace
    el = el + rng.normal(0, np.radians(0.9), m) * pressure * pace
    sp = sp + rng.normal(0, 0.5, m) * pressure * pace
    rpm = rpm * (1.0 + rng.normal(0, 0.05, m))
    p0 = np.stack([states.xa, np.zeros(m), np.full(m, arena.launch_z)], axis=1)
    f = simulate(p0, launch_velocity(sp, az, el), spin_vector(rpm, az), y_barrier=arena.barrier_y,
                 body=arena.body, surface=arena.surface)
    x, y = f.pos_land[:, 0], f.pos_land[:, 1]
    ok = (f.landed & f.crossed & (f.z_cross > arena.barrier_h)
          & (np.abs(x) <= arena.half_width) & (y > arena.barrier_y) & (y <= arena.far_y))
    reward = np.full(m, -1.0)
    p_ret = np.full(m, np.nan)
    kind = np.full(m, ERROR, dtype=np.int8)
    if ok.any():
        p = opponent_reach_prob(f.pos_land[ok], f.vel_out[ok], f.t_land[ok],
                                states.xb[ok], states.yb[ok], states.level[ok])
        p_ret[ok] = p
        reward[ok] = 1.0 - p
        kind[ok] = np.where(rng.random(p.shape) < p, CONTINUE, WIN)
    return Outcome(reward, kind, p_ret, np.where(ok, x, np.nan), np.where(ok, y, np.nan))
