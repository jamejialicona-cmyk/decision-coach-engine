"""A small policy/value network in plain numpy, with manual backprop and Adam.

    features -> [tanh] -> [tanh] -+-> policy logits (one per action)
                                  +-> state value  (scalar)

No framework on purpose: the point is that every gradient is visible and
checked against finite differences (tests/test_nn.py). Swap in a framework when
the problem outgrows it; the interface (`probs`, `loss_and_grads`) will not change.
"""
from __future__ import annotations

import numpy as np


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class PolicyValueNet:
    def __init__(self, d_in: int, n_actions: int, hidden: int = 64, seed: int = 0):
        rng = np.random.default_rng(seed)
        def init(i, o, gain=1.0):
            return rng.normal(0, gain / np.sqrt(i), (i, o))
        self.p = {
            "W1": init(d_in, hidden), "b1": np.zeros(hidden),
            "W2": init(hidden, hidden), "b2": np.zeros(hidden),
            "Wp": init(hidden, n_actions, 0.1), "bp": np.zeros(n_actions),
            "Wv": init(hidden, 1, 0.1), "bv": np.zeros(1),
        }
        self.n_actions = n_actions
        self._m = {k: np.zeros_like(v) for k, v in self.p.items()}
        self._v = {k: np.zeros_like(v) for k, v in self.p.items()}
        self._t = 0

    # ------------------------------------------------------------ forward
    def forward(self, x: np.ndarray):
        p = self.p
        h1 = np.tanh(x @ p["W1"] + p["b1"])
        h2 = np.tanh(h1 @ p["W2"] + p["b2"])
        logits = h2 @ p["Wp"] + p["bp"]
        value = (h2 @ p["Wv"] + p["bv"])[:, 0]
        return logits, value, (x, h1, h2)

    def probs(self, x: np.ndarray) -> np.ndarray:
        return softmax(self.forward(x)[0])

    def act(self, x: np.ndarray, rng: np.random.Generator, greedy: bool = False) -> np.ndarray:
        pr = self.probs(x)
        if greedy:
            return pr.argmax(axis=1)
        c = pr.cumsum(axis=1)
        return (rng.random((len(x), 1)) > c).sum(axis=1).clip(max=pr.shape[1] - 1)

    # ----------------------------------------------------------- backward
    def backward(self, cache, dlogits: np.ndarray, dvalue: np.ndarray) -> dict:
        x, h1, h2 = cache
        p = self.p
        g = {"Wp": h2.T @ dlogits, "bp": dlogits.sum(0),
             "Wv": h2.T @ dvalue[:, None], "bv": dvalue.sum(0, keepdims=True)}
        dh2 = dlogits @ p["Wp"].T + dvalue[:, None] @ p["Wv"].T
        dz2 = dh2 * (1 - h2 ** 2)
        g["W2"], g["b2"] = h1.T @ dz2, dz2.sum(0)
        dz1 = (dz2 @ p["W2"].T) * (1 - h1 ** 2)
        g["W1"], g["b1"] = x.T @ dz1, dz1.sum(0)
        return g

    def loss_and_grads(self, x, *, actions, advantages=None, returns=None, teacher=None,
                       entropy_coef=0.0, value_coef=0.5):
        """Two losses share one network.

        policy-gradient (actions, advantages, returns): -A log pi(a|s) - beta H(pi) + c (V - R)^2 / 2
        imitation       (teacher):                      cross-entropy against teacher actions
        """
        logits, value, cache = self.forward(x)
        pr = softmax(logits)
        n = len(x)
        onehot = np.zeros_like(pr)
        if teacher is not None:
            onehot[np.arange(n), teacher] = 1.0
            loss = -np.log(pr[np.arange(n), teacher] + 1e-12).mean()
            return loss, self.backward(cache, (pr - onehot) / n, np.zeros(n))
        onehot[np.arange(n), actions] = 1.0
        logp = np.log(pr + 1e-12)
        ent = -(pr * logp).sum(axis=1)
        dlogits = (-advantages[:, None] * (onehot - pr)
                   + entropy_coef * pr * (logp + ent[:, None])) / n
        dvalue = value_coef * (value - returns) / n
        loss = (-(advantages * logp[np.arange(n), actions]).mean() - entropy_coef * ent.mean()
                + 0.5 * value_coef * ((value - returns) ** 2).mean())
        return loss, self.backward(cache, dlogits, dvalue)

    # ---------------------------------------------------------------- Adam
    def step(self, grads: dict, lr: float = 3e-3, b1: float = 0.9, b2: float = 0.999, clip: float = 5.0):
        norm = np.sqrt(sum((g ** 2).sum() for g in grads.values()))
        scale = min(1.0, clip / (norm + 1e-12))
        self._t += 1
        for k, g in grads.items():
            g = g * scale
            self._m[k] = b1 * self._m[k] + (1 - b1) * g
            self._v[k] = b2 * self._v[k] + (1 - b2) * g * g
            mh = self._m[k] / (1 - b1 ** self._t)
            vh = self._v[k] / (1 - b2 ** self._t)
            self.p[k] -= lr * mh / (np.sqrt(vh) + 1e-8)
