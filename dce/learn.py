"""Learning a policy for the arena: the first two steps of an RL ladder.

1. Imitation: learn from the counterfactual teacher (the simulator's own best
   action per context). Cheap, stable, and a good warm start.
2. Policy gradient (REINFORCE with a learned baseline) on one-step decisions,
   where the reward is the physics-based outcome. This is a contextual bandit:
   the decision's consequences are one exchange long. Multi-step play, self-play
   and offline data are the next rungs (docs/RL-ROADMAP.md).

Everything is evaluated against fixed baselines, including the oracle that
simulates every alternative (an upper bound no deployed policy can use).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from . import counterfactual
from .arena import ERROR, N_ACTIONS, ArenaParams, States, play, sample_states
from .nn import PolicyValueNet

Policy = Callable[[States, np.random.Generator], np.ndarray]


def imitation(net: PolicyValueNet, rng: np.random.Generator, *, n_states: int = 600, n_samples: int = 16,
              epochs: int = 150, batch: int = 128, lr: float = 3e-3,
              arena: ArenaParams = ArenaParams()) -> list[float]:
    states = sample_states(n_states, rng)
    teacher = counterfactual.evaluate(states, rng, n_samples, arena).best
    x = states.features()
    history = []
    for _ in range(epochs):
        idx = rng.choice(n_states, size=min(batch, n_states), replace=False)
        loss, g = net.loss_and_grads(x[idx], actions=None, teacher=teacher[idx])
        net.step(g, lr)
        history.append(float(loss))
    return history


@dataclass
class RLLog:
    mean_reward: list
    error_rate: list


def reinforce(net: PolicyValueNet, rng: np.random.Generator, *, iters: int = 120, batch: int = 256,
              lr: float = 2e-3, entropy_start: float = 0.03, entropy_end: float = 0.003,
              arena: ArenaParams = ArenaParams()) -> RLLog:
    log = RLLog([], [])
    for it in range(iters):
        states = sample_states(batch, rng)
        x = states.features()
        actions = net.act(x, rng)
        out = play(states, actions, rng, arena)
        _, value, _ = net.forward(x)
        adv = out.reward - value
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)          # normalised advantage
        beta = entropy_start + (entropy_end - entropy_start) * it / max(1, iters - 1)
        _, g = net.loss_and_grads(x, actions=actions, advantages=adv, returns=out.reward, entropy_coef=beta)
        net.step(g, lr)
        log.mean_reward.append(float(out.reward.mean()))
        log.error_rate.append(float((out.kind == ERROR).mean()))
    return log


# ------------------------------------------------------------- evaluation
def evaluate_policy(policy: Policy, rng: np.random.Generator, *, n: int = 800,
                    arena: ArenaParams = ArenaParams()) -> dict:
    states = sample_states(n, rng)
    out = play(states, policy(states, rng), rng, arena)
    return {"reward": float(out.reward.mean()), "error_rate": float((out.kind == ERROR).mean())}


def random_policy(states: States, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, N_ACTIONS, len(states))


def fixed_policy(action: int) -> Policy:
    return lambda states, rng: np.full(len(states), action)


def net_policy(net: PolicyValueNet, greedy: bool = True) -> Policy:
    return lambda states, rng: net.act(states.features(), rng, greedy=greedy)


def oracle_policy(arena: ArenaParams = ArenaParams(), n_samples: int = 24) -> Policy:
    """Upper bound: simulates all alternatives per context. Not deployable (too slow, needs the model)."""
    return lambda states, rng: counterfactual.evaluate(states, rng, n_samples, arena).best
