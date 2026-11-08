# Reinforcement learning: what exists and what comes next

## Built

1. **Imitation from a counterfactual teacher** (`learn.imitation`). For each
   sampled context the simulator finds the best action; the network learns to
   predict it. Stable and cheap. It gets close to the oracle but inherits the
   teacher's noise (the "best" action is the argmax of a Monte Carlo estimate).
2. **Policy gradient with a learned baseline** (`learn.reinforce`). The network
   samples an action, the simulator returns a reward, and a REINFORCE update with
   a value baseline, normalised advantages and a decaying entropy bonus follows.
   Warm-started by imitation it improves further; from scratch it also learns
   (see the tests), just more slowly.

Both treat a decision as **one step**: the consequence is a single exchange, so
this is a contextual bandit. That is a deliberate, honest first rung.

## Evaluation protocol (already in place)

Every policy is scored on the same held-out contexts and compared with:
random, the best single fixed action, and the **oracle** that simulates all
alternatives. A learned policy is only interesting relative to those.

## Next rungs

| Rung | What changes | What it needs |
|---|---|---|
| **Multi-step exchanges** | The opponent's reply produces the next context; the return is the discounted result of the whole exchange | A transition model and a value network trained with bootstrapping (actor-critic / PPO) |
| **Self-play league** | The opponent is also a learned policy, drawn from a pool of past versions | A population, an exploitability check (can a fresh best-response beat the champion?) |
| **Offline learning from logs** | Real decisions and outcomes replace or correct simulated ones | Conservative offline RL, behaviour regularisation, off-policy evaluation |
| **Sim-to-real** | Calibrate noise and opponent models; randomise what cannot be calibrated | Logged data, a calibration-error metric, domain randomisation |
| **Larger function approximators** | Sequences instead of snapshots (a recurrent or attention model over recent play) | A framework backend; the interface in `nn.py` is kept narrow for this |

## Risks to watch

- **Reward hacking the simulator.** A policy can exploit the opponent model's
  blind spots. Evaluate against opponents it was not trained on.
- **Overconfidence from a single seed.** Report spreads over seeds, not one run.
- **Teacher noise.** Argmax of a noisy estimate is a noisy label; use more
  samples, or soften the labels (distillation with temperature).
