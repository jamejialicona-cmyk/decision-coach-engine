# Architecture

Five layers, each usable alone and each tested alone.

| Layer | Module | Responsibility | Does not do |
|---|---|---|---|
| Physics | `physics.py` | Integrate many flights at once; one ground contact; barrier crossing | Know about actions, players or rewards |
| World | `arena.py` | Contexts, the action set, execution noise, the opponent reach model, outcomes | Learn anything |
| Counterfactuals | `counterfactual.py` | Value every action in a context by simulation | Choose for anyone |
| Learning | `nn.py`, `learn.py` | Distil the simulator into a fast policy; evaluate against baselines and an oracle | Touch physics directly |
| Coach | `coach.py` | Judge what a person chose, find repeated patterns, say one thing | Run the simulator itself (it asks `counterfactual`) |

## Why this shape

- **The simulator is the teacher.** Real labelled data of good decisions does
  not exist at scale. A simulator with a defensible physics and a modelled
  opponent can label millions of contexts. Everything downstream is only as
  good as that model, which is why the limits are stated up front.
- **Vectorize once, benefit everywhere.** `physics.simulate` takes arrays. The
  counterfactual table, the oracle, the RL batch and the coach review are all
  the same call with different shapes.
- **Two ways to use the learned policy.** As a recommender ("what would a good
  player do here?", one forward pass) and as a ruler for the coach (the
  counterfactual table is slower but is the ground truth for the review).

## Data flow of a coaching session

1. Contexts are observed (positions, time available, opponent level).
2. The person's choices and outcomes are logged.
3. `coach.review` asks `counterfactual.evaluate` for the value of every
   alternative in each context.
4. Each decision gets a gap (best − chosen) and a verdict: *decision*,
   *execution* or *fine*.
5. `coach.insights` counts repeated patterns among decisions with a real gap.
6. `coach.one_thing` returns at most one sentence, or nothing.

## Extending it

- **A new setting:** replace `arena.py` (contexts, actions, `play`). Keep the
  `play(states, actions, rng) -> Outcome` contract and everything else works.
- **A better opponent:** replace `opponent_reach_prob`; nothing else changes.
- **A bigger network:** implement `forward`, `loss_and_grads`, `step` and `act`
  with the same signatures (see `nn.py`).
- **New patterns for the coach:** add a `Pattern(key, text, test)`. A pattern is
  a boolean test over a `Review`; the repeat and gap gates are applied for you.
