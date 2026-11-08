import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dce import coach, counterfactual, learn  # noqa: E402
from dce.arena import (ACTIONS, CONTINUE, ERROR, N_ACTIONS, N_FEATURES, WIN, States,  # noqa: E402
                       opponent_reach_prob, play, sample_states)
from dce.nn import PolicyValueNet, softmax  # noqa: E402
from dce.physics import (BodyParams, SurfaceParams, launch_velocity, simulate,  # noqa: E402
                         spin_vector)


def launch(speed, elev_deg, rpm=0.0, az=0.0, n=1, z0=1.0, **kw):
    sp = np.full(n, float(speed)); a = np.full(n, az); e = np.radians(np.full(n, float(elev_deg)))
    return simulate(np.tile([0.0, 0.0, z0], (n, 1)), launch_velocity(sp, a, e),
                    spin_vector(np.full(n, float(rpm)), a), **kw)


class Physics(unittest.TestCase):
    def test_matches_the_analytic_parabola_without_drag(self):
        f = launch(20, 10, body=BodyParams(cd=0.0), dt=1 / 480)
        vz = 20 * np.sin(np.radians(10))
        t = (vz + np.sqrt(vz ** 2 + 2 * 9.81)) / 9.81
        self.assertAlmostEqual(float(f.t_land[0]), t, places=3)
        self.assertAlmostEqual(float(f.pos_land[0, 1]), 20 * np.cos(np.radians(10)) * t, places=2)

    def test_drag_shortens_the_flight(self):
        free = launch(20, 10, body=BodyParams(cd=0.0))
        drag = launch(20, 10)
        self.assertLess(drag.pos_land[0, 1], free.pos_land[0, 1] - 0.5)

    def test_forward_spin_dips_and_backspin_floats(self):
        top, flat, back = (launch(20, 8, rpm).pos_land[0, 1] for rpm in (2500, 0, -2500))
        self.assertLess(top, flat)
        self.assertGreater(back, flat)

    def test_batch_equals_individual_runs(self):
        speeds = np.array([16.0, 20.0, 24.0])
        elev = np.radians([5.0, 8.0, 11.0])
        az = np.zeros(3)
        p0 = np.tile([0, 0, 1.0], (3, 1))
        batch = simulate(p0, launch_velocity(speeds, az, elev), spin_vector(np.full(3, 1000.0), az))
        for i in range(3):
            one = simulate(p0[i:i + 1], launch_velocity(speeds[i:i + 1], az[:1], elev[i:i + 1]),
                           spin_vector(np.array([1000.0]), az[:1]))
            np.testing.assert_allclose(batch.pos_land[i], one.pos_land[0], atol=1e-6)

    def test_bounce_restitution(self):
        s = SurfaceParams(restitution=0.6)
        f = launch(10, 0.0, z0=2.0, surface=s, body=BodyParams(cd=0.0))
        vz_in = -np.sqrt(2 * 9.81 * 2.0)
        self.assertAlmostEqual(float(f.vel_out[0, 2]), -0.6 * vz_in, places=1)

    def test_barrier_crossing_height_is_interpolated(self):
        f = launch(20, 10, body=BodyParams(cd=0.0), y_barrier=10.0, dt=1 / 480)
        t = 10.0 / (20 * np.cos(np.radians(10)))
        z = 1.0 + 20 * np.sin(np.radians(10)) * t - 0.5 * 9.81 * t * t
        self.assertTrue(f.crossed[0])
        self.assertAlmostEqual(float(f.z_cross[0]), z, places=2)

    def test_a_ball_that_never_reaches_the_barrier_is_not_marked_crossed(self):
        f = launch(10, 2, y_barrier=40.0)
        self.assertFalse(f.crossed[0])
        self.assertTrue(np.isnan(f.z_cross[0]))


class Arena(unittest.TestCase):
    def test_features_are_finite_and_sized(self):
        x = sample_states(50, np.random.default_rng(0)).features()
        self.assertEqual(x.shape, (50, N_FEATURES))
        self.assertTrue(np.isfinite(x).all())

    def test_reward_range_and_error_semantics(self):
        rng = np.random.default_rng(1)
        s = sample_states(400, rng)
        out = play(s, rng.integers(0, N_ACTIONS, 400), rng)
        self.assertTrue(((out.reward >= -1) & (out.reward <= 1)).all())
        self.assertTrue((out.reward[out.kind == ERROR] == -1).all())
        self.assertTrue(set(np.unique(out.kind)) <= {ERROR, CONTINUE, WIN})

    def test_time_pressure_raises_the_failure_rate(self):
        rng = np.random.default_rng(2)
        a = 22  # fast, high arc, centre, forward spin
        def fail_rate(t):
            s = sample_states(1500, rng)
            s.t_avail = np.full(1500, t)
            return (play(s, np.full(1500, a), rng).kind == ERROR).mean()
        self.assertGreater(fail_rate(0.4), fail_rate(1.6))

    def test_a_stronger_opponent_reaches_more(self):
        land = np.array([[1.0, 14.0, 0.0]] * 2)
        vel = np.array([[0.0, 12.0, 3.0]] * 2)
        p = opponent_reach_prob(land, vel, np.array([0.8, 0.8]), np.array([-1.5, -1.5]),
                                np.array([17.0, 17.0]), np.array([0, 2]))
        self.assertGreater(p[1], p[0])

    def test_a_farther_opponent_reaches_less(self):
        land = np.array([[0.0, 14.0, 0.0]] * 2)
        vel = np.array([[0.0, 12.0, 3.0]] * 2)
        p = opponent_reach_prob(land, vel, np.array([0.8, 0.8]), np.array([0.0, 2.5]),
                                np.array([15.0, 18.0]), np.array([1, 1]))
        self.assertGreater(p[0], p[1])


class Network(unittest.TestCase):
    def _numeric_check(self, **loss_kw):
        net = PolicyValueNet(4, 5, hidden=6, seed=1)
        rng = np.random.default_rng(0)
        x = rng.normal(size=(7, 4))
        extra = {k: (v(rng) if callable(v) else v) for k, v in loss_kw.items()}
        loss, grads = net.loss_and_grads(x, **extra)
        worst = 0.0
        for name, arr in net.p.items():
            flat = arr.reshape(-1)
            for i in rng.choice(flat.size, size=min(4, flat.size), replace=False):
                old = flat[i]
                flat[i] = old + 1e-6
                up = net.loss_and_grads(x, **extra)[0]
                flat[i] = old - 1e-6
                dn = net.loss_and_grads(x, **extra)[0]
                flat[i] = old
                num = (up - dn) / 2e-6
                ana = grads[name].reshape(-1)[i]
                worst = max(worst, abs(num - ana) / max(1e-8, abs(num) + abs(ana)))
        return worst

    def test_policy_gradient_gradients_match_finite_differences(self):
        w = self._numeric_check(actions=lambda r: r.integers(0, 5, 7), advantages=lambda r: r.normal(size=7),
                                returns=lambda r: r.normal(size=7), entropy_coef=0.02)
        self.assertLess(w, 1e-5)

    def test_imitation_gradients_match_finite_differences(self):
        w = self._numeric_check(actions=None, teacher=lambda r: r.integers(0, 5, 7))
        self.assertLess(w, 1e-5)

    def test_softmax_rows_sum_to_one(self):
        z = np.random.default_rng(0).normal(size=(5, 9)) * 50
        np.testing.assert_allclose(softmax(z).sum(axis=1), 1.0)

    def test_sampling_follows_the_probabilities(self):
        net = PolicyValueNet(2, 3, seed=0)
        x = np.zeros((20000, 2))
        counts = np.bincount(net.act(x, np.random.default_rng(0)), minlength=3) / 20000
        np.testing.assert_allclose(counts, net.probs(x[:1])[0], atol=0.02)

    def test_imitation_learns_a_toy_rule(self):
        rng = np.random.default_rng(0)
        net = PolicyValueNet(2, 2, hidden=16, seed=0)
        x = rng.normal(size=(300, 2))
        y = (x[:, 0] > 0).astype(int)
        for _ in range(300):
            _, g = net.loss_and_grads(x, actions=None, teacher=y)
            net.step(g, 1e-2)
        self.assertGreater((net.act(x, rng, greedy=True) == y).mean(), 0.95)


class Counterfactual(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rng = np.random.default_rng(3)
        cls.states = sample_states(60, cls.rng)
        cls.values = counterfactual.evaluate(cls.states, cls.rng, n_samples=12)

    def test_shapes_and_best_is_argmax(self):
        self.assertEqual(self.values.value.shape, (60, N_ACTIONS))
        np.testing.assert_array_equal(self.values.best, self.values.value.argmax(axis=1))

    def test_best_action_aims_away_from_the_opponent(self):
        az = np.array([ACTIONS[b][0] for b in self.values.best])
        self.assertLess(np.corrcoef(az, self.states.xb)[0, 1], -0.4)

    def test_best_beats_the_average_action_by_a_lot(self):
        self.assertGreater((self.values.best_value - self.values.value.mean(axis=1)).min(), 0.2)


class Learning(unittest.TestCase):
    def test_imitation_then_policy_gradient_beats_random_and_improves(self):
        rng = np.random.default_rng(5)
        net = PolicyValueNet(N_FEATURES, N_ACTIONS, seed=3)
        hist = learn.imitation(net, rng, n_states=120, n_samples=8, epochs=100)
        self.assertLess(np.mean(hist[-10:]), np.mean(hist[:10]))
        ev = lambda pol: learn.evaluate_policy(pol, np.random.default_rng(99), n=500)["reward"]
        base = ev(learn.random_policy)
        after_il = ev(learn.net_policy(net))
        log = learn.reinforce(net, rng, iters=30, batch=128)
        after_rl = ev(learn.net_policy(net))
        self.assertGreater(after_il, base + 0.5)
        self.assertGreater(after_rl, base + 0.5)
        self.assertTrue(np.isfinite(log.mean_reward).all())

    def test_policy_gradient_from_scratch_improves_its_own_reward(self):
        net = PolicyValueNet(N_FEATURES, N_ACTIONS, seed=3)
        log = learn.reinforce(net, np.random.default_rng(1), iters=80, batch=192)
        self.assertGreater(np.mean(log.mean_reward[-10:]), np.mean(log.mean_reward[:10]) + 0.4)


class Coach(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rng = np.random.default_rng(11)
        cls.states = sample_states(80, cls.rng)
        cls.values = counterfactual.evaluate(cls.states, cls.rng, n_samples=12)

    def review_of(self, chosen, kinds=None):
        kinds = np.full(len(chosen), CONTINUE) if kinds is None else kinds
        r = coach.Review(self.states, chosen, kinds, self.values)
        decision = r.gap >= 0.10
        r.verdict = np.where(decision, "decision", np.where(kinds == ERROR, "execution", "fine"))
        return r

    def test_choosing_the_best_action_has_no_gap_and_the_coach_stays_silent(self):
        r = self.review_of(self.values.best)
        self.assertTrue((r.gap == 0).all())
        self.assertIsNone(coach.one_thing(r))

    def test_a_miss_after_the_best_choice_is_execution_not_decision(self):
        kinds = np.full(80, ERROR)
        r = self.review_of(self.values.best, kinds)
        self.assertTrue((r.verdict == "execution").all())

    def test_aiming_at_the_opponent_is_flagged_as_a_repeated_decision_pattern(self):
        side = np.where(self.states.xb < 0, -6.0, 6.0)
        acts = np.array(ACTIONS)
        chosen = np.array([int(np.argmin(np.abs(acts - np.array((s, 10.0, 24.0, 2000.0))).sum(axis=1)))
                           for s in side])
        r = self.review_of(chosen)
        found = coach.insights(r)
        self.assertIn("into_opponent", [i.key for i in found])
        self.assertIn("aimed where the opponent already was", coach.one_thing(r))

    def test_the_coach_is_silent_when_a_pattern_does_not_repeat(self):
        side = np.where(self.states.xb < 0, -6.0, 6.0)
        acts = np.array(ACTIONS)
        chosen = self.values.best.copy()
        for i in range(2):  # only two bad decisions: below min_repeat
            chosen[i] = int(np.argmin(np.abs(acts - np.array((side[i], 10.0, 24.0, 2000.0))).sum(axis=1)))
        self.assertIsNone(coach.one_thing(self.review_of(chosen), min_repeat=3))


if __name__ == "__main__":
    unittest.main()
