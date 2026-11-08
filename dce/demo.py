"""End-to-end demo:  python -m dce.demo [--quick]

1. Compare baselines with a policy trained by imitation (counterfactual teacher) + policy gradient.
2. Let a "habitual" player choose in 120 contexts and have the coach review them.
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from . import coach, learn
from .arena import ACTIONS, N_ACTIONS, N_FEATURES, States, describe, play, sample_states
from .nn import PolicyValueNet


def habitual_player(states: States, rng: np.random.Generator) -> np.ndarray:
    """A made-up amateur: hits hard and aims at the opponent's side, whatever the context."""
    acts = np.array(ACTIONS)
    out = np.empty(len(states), int)
    for i in range(len(states)):
        side = -6.0 if states.xb[i] < 0 else 6.0              # same side as the opponent
        want = (side, 10.0, 24.0, 2000.0)                    # high arc, fast, forward spin
        out[i] = int(np.argmin(np.abs(acts - np.array(want)).sum(axis=1)))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smaller budgets (~15 s)")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args(argv)
    q = args.quick
    rng = np.random.default_rng(args.seed)
    t0 = time.time()

    net = PolicyValueNet(N_FEATURES, N_ACTIONS, seed=3)
    learn.imitation(net, rng, n_states=150 if q else 400, n_samples=12, epochs=120)
    after_il = learn.evaluate_policy(learn.net_policy(net), np.random.default_rng(99), n=500)
    learn.reinforce(net, rng, iters=60 if q else 120)
    after_rl = learn.evaluate_policy(learn.net_policy(net), np.random.default_rng(99), n=500)

    rows = [("random", learn.evaluate_policy(learn.random_policy, np.random.default_rng(99), n=500)),
            ("best single fixed action", learn.evaluate_policy(learn.fixed_policy(22), np.random.default_rng(99), n=500)),
            ("imitation", after_il), ("imitation + policy gradient", after_rl)]
    if not q:
        rows.append(("oracle (simulates all alternatives)", learn.evaluate_policy(
            learn.oracle_policy(n_samples=12), np.random.default_rng(99), n=300)))
    print(f"{'policy':<40}{'mean reward':>12}{'failure rate':>14}")
    for name, m in rows:
        print(f"{name:<40}{m['reward']:>12.3f}{m['error_rate']:>14.1%}")

    print("\n--- Coach review of a habitual player (120 decisions) ---")
    states = sample_states(120, np.random.default_rng(7))
    chosen = habitual_player(states, rng)
    out = play(states, chosen, rng)
    r = coach.review(states, chosen, out.kind, rng, n_samples=12 if q else 24)
    print(coach.summary(r))
    for ins in coach.insights(r):
        print(" -", ins.sentence)
    print("\nBetween points, the coach says:", coach.one_thing(r) or "(nothing: no repeated, valuable pattern)")
    worst = int(np.argmax(r.gap))
    print(f"\nWorst decision #{worst}: chose [{describe(int(chosen[worst]))}] "
          f"(value {r.value_chosen[worst]:+.2f}); best alternative [{coach.best_alternative(r, worst)}] "
          f"(value {r.values.best_value[worst]:+.2f}).")
    print(f"\n({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
