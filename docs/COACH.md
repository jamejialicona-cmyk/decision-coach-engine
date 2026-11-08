# The coach layer

The product question is not "how did it go?" but **"was that the right choice,
and what is the one thing to change?"**

## 1. Decision vs execution

The same bad outcome can come from two different mistakes.

| Verdict | Meaning | What to say |
|---|---|---|
| **decision** | The chosen action was worth at least `min_gap` less than the best simulated alternative | Change what you choose |
| **execution** | The choice was near-best but the attempt failed | Practise the action; do not change the plan |
| **fine** | Near-best and it did not fail outright | Nothing |

A decision can be flagged *even when it worked*: luck is not evidence. In the
real world intent is not observable, so the engine judges the **implicit
decision** (what was done in that context) and leaves room to ask the person
what they were aiming at, which also produces labels.

## 2. Relative values only

A value is the mean reward of an action in one context, from simulation with
the actor's own noise against a modelled opponent. It ranks alternatives in the
same context. It is **not** a probability of winning, and the report says so
("relative scale, -1 to 1"). Its accuracy is the accuracy of the opponent and
noise models, which must be calibrated before anyone is told a number.

## 3. Patterns, not single events

One bad choice is noise. The coach reports a pattern only when:

- the decision has a real gap (`min_gap`, default 0.10), and
- the pattern repeats (`min_repeat`, default 3).

Built-in patterns: aiming where the opponent already is; forcing a fast action
when short on time; choosing actions that fail outright too often; playing safe
when a firmer action was clearly better. Add your own as boolean tests.

## 4. A time budget for talking

Speaking during action is disruptive; the useful moments are the gaps between
exchanges. The coach therefore returns **one** sentence, ranked by
`count × mean gap`, or **nothing**. Longer analysis belongs in a debrief.

## 5. What would make it wrong

- The opponent model is wrong, so the "best" alternative is best against a
  fiction. Fix: calibrate on logged opponents; show uncertainty.
- Execution noise is mis-specified for this person. Fix: fit it per person from
  their own failures.
- Contexts are mis-measured (perception error). Fix: carry measurement
  uncertainty into the counterfactual instead of treating contexts as exact.
- Gaps are small relative to Monte Carlo noise. Fix: more samples, and report
  only gaps larger than the standard error.
