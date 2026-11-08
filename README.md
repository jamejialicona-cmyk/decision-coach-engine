# decision-coach-engine

A small, readable engine for **coaching decisions made under physics**: simulate
what would have happened, learn a policy from the simulator, and tell a person
the one thing worth changing.

[![CI](https://github.com/jamejialicona-cmyk/decision-coach-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/jamejialicona-cmyk/decision-coach-engine/actions/workflows/ci.yml)

> **Status: reference implementation.** Physics, counterfactual simulation, a
> numpy policy/value network, imitation + policy-gradient learning and the coach
> are built and tested (26 tests). The arena, constants and "players" are
> **synthetic and uncalibrated**: this repository shows the architecture and the
> reasoning, not a validated model of any real activity. See [Limits](#limits-stated-plainly).

---

## The idea in one paragraph

Statistics tell you what happened. A coach has to say whether the **decision**
was right, and an outcome can hide that: a good choice can miss, a bad one can
work. So the engine does three things. It *simulates* every alternative in the
same context (with the actor's own execution noise), which gives a **relative**
value per action. It *learns* a policy from that simulator, so a good answer is
available instantly instead of after thousands of rollouts. And a *coach* layer
compares what the person chose with the best alternative, separates **decision**
from **execution**, looks for patterns that repeat, and speaks **at most once**.

```
context ──► physics ──► outcome            (what really happens, vectorized, RK4)
   │            ▲
   │            └── execution noise grows with time pressure
   ├──► counterfactual: all actions × many noisy samples → relative value per action
   │         │
   │         ├──► teacher labels ──► imitation ─┐
   │         └──► oracle (upper bound)          ├──► policy/value net ──► fast recommendation
   └──► policy gradient on simulated reward ────┘
                                   │
 what the person chose ────────────┴──► coach: gap to best · decision vs execution · patterns · one sentence
```

## The setting

Deliberately abstract: an *actor* sends a spinning sphere over a barrier into
the far half of a bounded arena; an *opponent* of one of three levels tries to
reach it. Each decision is one of 36 actions (lateral aim × launch angle × speed
× spin). Nothing about it is tied to a sport; it stands for any problem where
the consequences of a choice are physical, noisy and affected by an opponent
(ball games are the obvious family).

The physics is real enough to produce real tradeoffs: gravity, quadratic drag,
spin-induced lift (forward spin dips, backspin floats), a simplified bounce, and
execution noise that grows with time pressure and with how hard the action is
played. Most of the 36 actions *fail*; only a handful are good, and which ones
depend on where the opponent stands.

## Run it

```bash
pip install -r requirements.txt            # numpy only
python -m dce.demo --quick                 # ~15 s
python -m dce.demo                         # ~40 s
python -m unittest discover -s tests -v    # 26 tests, ~15 s
```

Output of `python -m dce.demo` (seed 1, one run; the synthetic arena is the benchmark):

```
policy                                   mean reward  failure rate
random                                        -0.603         69.2%
best single fixed action                       0.212          5.0%
imitation                                      0.322          8.6%
imitation + policy gradient                    0.346          5.6%
oracle (simulates all alternatives)            0.388          7.3%
```

Reading it: guessing fails two times out of three. One good habit is worth
0.21. A network that has learned from the simulator reaches 0.35, close to the
oracle that simulates every alternative for every decision (0.39), but answers
in one forward pass. The oracle is an upper bound, not a deployable policy.

Then the coach reviews 120 decisions of a made-up "habitual" player:

```
In 65 of 120 decisions you aimed where the opponent already was; the best simulated
alternative was worth +0.52 more on average (relative scale, -1 to 1).
In 10 of 120 decisions you forced a fast action when short on time; ...
Between points, the coach says: In 65 of 120 decisions you aimed where the opponent already was; ...
```

The habitual player is a caricature built to have a flaw. The point is the
mechanism: a pattern is reported only if it repeats (`min_repeat`) and is worth
enough (`min_gap`), and misses after a near-best choice are labelled
**execution**, not decision.

## What is in here

```
dce/physics.py         vectorized RK4: gravity, drag, spin lift, simplified bounce, barrier crossing
dce/arena.py           contexts, 36 actions, execution noise, opponent reach model, outcomes
dce/counterfactual.py  every action × many noisy samples → relative value and failure rate
dce/nn.py              policy/value network in numpy, manual backprop, Adam (gradients checked)
dce/learn.py           imitation from the counterfactual teacher; REINFORCE with a baseline; baselines + oracle
dce/coach.py           decision vs execution, repeated patterns, one sentence
dce/demo.py            the end-to-end demo above
docs/                  ARCHITECTURE · COACH · RL-ROADMAP
tests/                 26 tests
```

## Design choices worth knowing

- **Vectorized physics.** Thousands of trajectories are integrated at once; a
  full counterfactual table (60 contexts × 36 actions × 12 samples) runs in
  seconds in pure numpy. That is what makes simulation-in-the-loop affordable.
- **Relative values, not probabilities.** Rewards compare actions *in the same
  context*. They are only as good as the opponent and noise models, so the coach
  never reports them as chances of winning.
- **Gradients are checked, not trusted.** The network's backprop is verified
  against finite differences, and the physics against the analytic parabola. A
  deliberate bug (switching off the spin force) makes a test fail.
- **Decision before execution.** A miss does not make a decision wrong.
- **Silence is a feature.** One insight, only when it repeats and matters.

## Limits, stated plainly

- **Synthetic.** The arena, the constants and the opponent are made up and
  uncalibrated. Nothing here has been validated against real play, and the
  numbers above are properties of this toy world.
- **One-step decisions.** The RL here is a contextual bandit: each decision's
  consequences are a single exchange. Multi-step play, self-play and learning
  from real logs are the next rungs ([docs/RL-ROADMAP.md](docs/RL-ROADMAP.md)).
- **No perception.** There is no video, pose or tracking. The engine starts from
  clean contexts; turning a camera feed into them is a separate problem.
- **A simple opponent.** The opponent is a reach model with a level, not a
  learned adversary, so "best action" means best against that model.
- **Single-seed demo numbers.** Differences of a few hundredths between rows are
  within evaluation noise at these sample sizes.

## Roadmap

- [x] Vectorized physics with spin, drag and bounce, verified against closed forms
- [x] Counterfactual simulation, imitation, policy gradient, oracle upper bound
- [x] Coach: decision vs execution, repeated patterns, one sentence
- [ ] Multi-step exchanges: the opponent's reply becomes the next context
- [ ] Self-play against a pool of opponents (league), with exploitability checks
- [ ] Calibrate noise and opponent models from logged data; report calibration error
- [ ] A framework backend (same interface) for larger networks

## License

[MIT](LICENSE)
