"""Counterfactuals: "what if you had chosen differently?"

For each observed context, every available action is simulated many times with
the actor's own execution noise against the modelled opponent. The mean reward
per action gives a *relative* ranking, which is the honest way to use it: the
numbers are only as good as the opponent model and the noise model, so they are
meant for comparing actions in the same context, not for absolute probabilities.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .arena import ERROR, N_ACTIONS, ArenaParams, States, play


@dataclass
class ActionValues:
    value: np.ndarray       # (S, A) mean reward
    p_error: np.ndarray     # (S, A) share of attempts that fail outright
    best: np.ndarray        # (S,) argmax action

    @property
    def best_value(self) -> np.ndarray:
        return self.value.max(axis=1)


def evaluate(states: States, rng: np.random.Generator, n_samples: int = 24,
             arena: ArenaParams = ArenaParams(), chunk: int = 60) -> ActionValues:
    """Mean reward of every action in every state (Monte Carlo over execution noise)."""
    s = len(states)
    value = np.zeros((s, N_ACTIONS))
    p_err = np.zeros((s, N_ACTIONS))
    for lo in range(0, s, chunk):
        sub = states.take(np.arange(lo, min(lo + chunk, s)))
        k = len(sub)
        big = sub.repeat(N_ACTIONS * n_samples)
        acts = np.tile(np.repeat(np.arange(N_ACTIONS), n_samples), k)
        out = play(big, acts, rng, arena)
        value[lo:lo + k] = out.reward.reshape(k, N_ACTIONS, n_samples).mean(axis=2)
        p_err[lo:lo + k] = (out.kind == ERROR).reshape(k, N_ACTIONS, n_samples).mean(axis=2)
    return ActionValues(value, p_err, value.argmax(axis=1))
