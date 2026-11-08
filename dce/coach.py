"""The coach: judge decisions, not just outcomes.

Given what a person (or policy) actually chose in a set of contexts, compare each
choice to the best simulated alternative, separate **decision** from
**execution**, look for patterns that repeat, and say at most one thing.

Design rules (docs/COACH.md):
  * Decision vs execution. A miss after a near-best choice is execution; a gap to
    the best alternative is a decision, whether or not it happened to work.
  * Relative, not absolute. Values compare actions in the same context; they are
    not probabilities of winning.
  * Speak rarely. One insight, only if it repeats (min_repeat) and is worth
    enough (min_gap); otherwise stay silent.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from . import counterfactual
from .arena import ACTIONS, ERROR, ArenaParams, States, describe

AIM_REF_Y = 17.0      # depth at which the lateral aim point is compared with the opponent


def aim_x(states: States, action: np.ndarray) -> np.ndarray:
    az = np.radians(np.array(ACTIONS)[action][:, 0])
    return states.xa + AIM_REF_Y * np.tan(az)


@dataclass
class Review:
    states: States
    chosen: np.ndarray
    kinds: np.ndarray
    values: counterfactual.ActionValues
    value_chosen: np.ndarray = field(init=False)
    gap: np.ndarray = field(init=False)
    verdict: np.ndarray = field(init=False)

    def __post_init__(self):
        n = np.arange(len(self.chosen))
        self.value_chosen = self.values.value[n, self.chosen]
        self.gap = self.values.best_value - self.value_chosen


@dataclass(frozen=True)
class Pattern:
    key: str
    text: str                                  # what the person did, past tense, plain words
    test: Callable[["Review"], np.ndarray]     # (N,) bool: does this decision show the pattern?


def _p_err(r: Review, which: np.ndarray) -> np.ndarray:
    return r.values.p_error[np.arange(len(which)), which]


def _is_fast(a: np.ndarray) -> np.ndarray:
    return np.array(ACTIONS)[a][:, 2] >= 24.0


PATTERNS = (
    Pattern("into_opponent", "aimed where the opponent already was",
            lambda r: (np.abs(aim_x(r.states, r.chosen) - r.states.xb) < 1.2)
            & (np.abs(aim_x(r.states, r.values.best) - r.states.xb) > 1.2)),
    Pattern("force_under_pressure", "forced a fast action when short on time",
            lambda r: (r.states.t_avail < 0.8) & _is_fast(r.chosen)
            & (_p_err(r, r.chosen) - _p_err(r, r.values.best) >= 0.2)),
    Pattern("low_percentage", "chose actions that failed outright far too often",
            lambda r: (_p_err(r, r.chosen) >= 0.4) & (_p_err(r, r.values.best) <= 0.15)),
    Pattern("too_safe", "played it safe when a firmer action was clearly better",
            lambda r: (_p_err(r, r.chosen) <= 0.05) & ~_is_fast(r.chosen) & _is_fast(r.values.best)),
)


@dataclass(frozen=True)
class Insight:
    key: str
    n_hits: int
    n_total: int
    mean_gap: float
    sentence: str


def review(states: States, chosen: np.ndarray, kinds: np.ndarray, rng: np.random.Generator, *,
           n_samples: int = 24, arena: ArenaParams = ArenaParams(), min_gap: float = 0.10) -> Review:
    r = Review(states, chosen, kinds, counterfactual.evaluate(states, rng, n_samples, arena))
    decision = r.gap >= min_gap
    miss = (kinds == ERROR)
    r.verdict = np.where(decision, "decision", np.where(miss, "execution", "fine"))
    return r


def insights(r: Review, *, min_repeat: int = 3, min_gap: float = 0.10) -> list[Insight]:
    flagged = r.gap >= min_gap
    out = []
    for p in PATTERNS:
        hit = p.test(r) & flagged
        n = int(hit.sum())
        if n >= min_repeat:
            g = float(r.gap[hit].mean())
            out.append(Insight(p.key, n, len(r.chosen), g,
                               f"In {n} of {len(r.chosen)} decisions you {p.text}; the best simulated "
                               f"alternative was worth {g:+.2f} more on average (relative scale, -1 to 1)."))
    return sorted(out, key=lambda i: i.n_hits * i.mean_gap, reverse=True)


def one_thing(r: Review, **kw) -> str | None:
    """What to say between points: the single most valuable repeated insight, or nothing."""
    found = insights(r, **kw)
    return found[0].sentence if found else None


def summary(r: Review) -> dict:
    v = r.verdict
    n = len(v)
    return {"decisions": n, "decision_gaps": int((v == "decision").sum()),
            "execution_misses": int((v == "execution").sum()),
            "mean_gap": float(r.gap.mean()),
            "mean_value_chosen": float(r.value_chosen.mean())}


def best_alternative(r: Review, i: int) -> str:
    return describe(int(r.values.best[i]))
