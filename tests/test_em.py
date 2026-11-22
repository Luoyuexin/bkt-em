"""EM 实现的正确性测试。

最硬的一条是 ``test_loglikelihood_matches_enumeration``：
序列长度 3 时隐状态只有 2^3 = 8 条可能路径，直接把 8 条路径的概率加起来，
就等于前向算法算出来的似然。两者不相等，说明下标或者缩放写错了。
"""

from __future__ import annotations

import math
import unittest

from bkt_em import BKTModel, Params, summarize
from bkt_em.em import _accumulate, _filter, _m_step, _scan
from bkt_em.simulate import attach_rows, simulate


def enumerate_loglik(y, p: Params) -> float:
    """穷举所有隐状态路径，作为前向算法的标准答案。"""
    n = len(y)
    A = p.transition()
    e = p.emission()
    total = 0.0
    for bits in range(1 << n):
        state = [(bits >> t) & 1 for t in range(n)]
        prob = (1 - p.L0) if state[0] == 0 else p.L0
        for t in range(n):
            prob *= e[state[t]] if y[t] else 1.0 - e[state[t]]
            if t:
                prob *= A[state[t - 1]][state[t]]
        total += prob
    return math.log(total)


class TestForwardBackward(unittest.TestCase):
    def test_loglikelihood_matches_enumeration(self):
        p = Params(L0=0.3, S=0.1, G=0.2, T=0.25, F=0.05)
        for y in ([1, 1, 1], [0, 1, 0], [1, 0, 1], [0, 0, 1], [0, 0, 0], [1, 0, 0]):
            with self.subTest(y=y):
                got = _scan(y, p)["ll"]
                want = enumerate_loglik(y, p)
                self.assertAlmostEqual(got, want, places=9)

    def test_loglikelihood_matches_enumeration_short_seq(self):
        p = Params(L0=0.1, S=0.35, G=0.05, T=0.6, F=0.2)
        wants = [enumerate_loglik([y], p) for y in (0, 1)]
        for y, want in zip((0, 1), wants):
            self.assertAlmostEqual(_scan([y], p)["ll"], want, places=9)

    def test_empty_sequence_returns_none(self):
        self.assertIsNone(_scan([], Params()))

    def test_degenerate_params_return_none(self):
        # S=1 且 y=0：已掌握的人不可能答错，这条观测概率为 0
        p = Params(L0=0.5, S=1.0, G=0.0, T=0.1)
        self.assertIsNone(_scan([1, 0], p))

    def test_em_likelihood_is_monotone(self):
        """E 步 M 步反复跑，对数似然必须单调不降 —— 不降就说明闭式解写错了。"""
        seqs = simulate(n_users=40, n_skills=1, n_items=12, seed=3)
        p = Params()
        last = -math.inf
        for _ in range(25):
            stats = _accumulate(seqs, p)
            ll = stats["ll"]
            self.assertGreaterEqual(ll, last - 1e-6)
            last = ll
            p = _m_step(stats, p, forget=False, bounds=(1e-3, 1 - 1e-3))

    def test_gamma_xi_consistency(self):
        """Σ_k xi(j,k) 必须等于 gamma(j)，否则 T 的分母会算偏。"""
        seqs = simulate(n_users=10, n_skills=1, n_items=6, seed=5)
        p = Params(L0=0.4, S=0.15, G=0.2, T=0.3)
        stats = _scan(seqs[0].y, p)
        # 单条序列：T 的分子 + F 的分子 ≈ 总转移次数（每条序列 n-1 次）
        self.assertLessEqual(stats["t_num"] + stats["f_num"], len(seqs[0].y) - 1 + 1e-9)


class TestFilter(unittest.TestCase):
    def test_first_prediction_is_prior(self):
        p = Params(L0=0.3, S=0.1, G=0.2, T=0.25)
        probs, _ = _filter([1, 1, 1], p)
        want = p.L0 * (1 - p.S) + (1 - p.L0) * p.G
        self.assertAlmostEqual(probs[0], want, places=12)

    def test_no_answer_leakage(self):
        """预测第 1 题时不能偷看第 1 题的答案。"""
        p = Params(L0=0.3, S=0.1, G=0.2, T=0.25)
        right, _ = _filter([1, 1, 1], p)
        wrong, _ = _filter([0, 1, 1], p)
        self.assertAlmostEqual(right[0], wrong[0], places=12)

    def test_mastery_rises_with_correct_answers(self):
        p = Params(L0=0.2, S=0.1, G=0.25, T=0.2)
        _, good = _filter([1, 1, 1, 1], p)
        _, bad = _filter([0, 0, 0, 0], p)
        self.assertGreater(good[1], bad[1])
        self.assertTrue(0.0 <= bad[1] <= 1.0 and 0.0 <= good[1] <= 1.0)

    def test_forgetting_lowers_mastery(self):
        no_forget = Params(L0=0.9, S=0.05, G=0.1, T=0.1, F=0.0)
        forget = Params(L0=0.9, S=0.05, G=0.1, T=0.1, F=0.4)
        _, a = _filter([1] * 6, no_forget)
        _, b = _filter([1] * 6, forget)
        self.assertLess(b[1], a[1])


class TestRecovery(unittest.TestCase):
    """已知真值 → 模拟 → 拟合，看得不得回真值。"""

    TRUTH = Params(L0=0.30, S=0.08, G=0.20, T=0.25)

    @classmethod
    def setUpClass(cls):
        cls.seqs = simulate(n_users=300, n_skills=1, n_items=12,
                            params=cls.TRUTH, seed=7)
        cls.model = BKTModel(model="global", n_init=2, seed=1).fit(cls.seqs)

    def test_parameters_recovered(self):
        est = self.model.params()["global"]
        for name, want in self.TRUTH.as_dict().items():
            with self.subTest(param=name):
                self.assertAlmostEqual(est[name], want, delta=0.10)

    def test_fit_beats_prior(self):
        """拟合出来的模型，似然必须高于「文献默认值」，否则 EM 白跑。"""
        default = Params()
        naive = _accumulate(self.seqs, default)["ll"]
        self.assertGreater(self.model.log_likelihood_, naive)

    def test_auc_reasonable(self):
        m = self.model.metrics(self.seqs)
        self.assertGreater(m["auc"], 0.70)
        self.assertLess(m["auc"], 1.0)
        self.assertEqual(m["n"], 300 * 12)

    def test_per_skill_beats_global_likelihood(self):
        seqs = simulate(n_users=100, n_skills=4, n_items=10, params=self.TRUTH, seed=11)
        g = BKTModel(model="global", n_init=2, seed=2).fit(seqs)
        ps = BKTModel(model="per-skill", n_init=2, seed=2).fit(seqs)
        self.assertGreaterEqual(ps.log_likelihood_ + 1e-6, g.log_likelihood_)
        self.assertEqual(ps.n_params_, 4 * 4)
        self.assertEqual(g.n_params_, 4)

    def test_forgetting_recovered(self):
        truth = Params(L0=0.35, S=0.10, G=0.22, T=0.30, F=0.10)
        seqs = simulate(n_users=300, n_skills=1, n_items=12, params=truth, seed=13)
        est = BKTModel(model="global", forget=True, n_init=2, seed=3).fit(seqs).params()["global"]
        self.assertAlmostEqual(est["F"], 0.10, delta=0.06)

    def test_forgetting_off_stays_zero(self):
        est = self.model.params()["global"]
        self.assertEqual(est["F"], 0.0)


class TestModelApi(unittest.TestCase):
    def setUp(self):
        self.seqs = attach_rows(simulate(n_users=12, n_skills=2, n_items=6, seed=21))
        self.model = BKTModel(model="per-skill", n_init=1, seed=1).fit(self.seqs)

    def test_predict_length_matches_observations(self):
        pred = self.model.predict(self.seqs)
        self.assertEqual(len(pred), sum(len(s) for s in self.seqs))
        self.assertTrue(all(0.0 < p < 1.0 for p in pred))

    def test_predict_rows_aligns_with_input(self):
        rows = self.model.predict_rows(self.seqs)
        self.assertEqual(set(rows), {r for s in self.seqs for r in s.rows})

    def test_mastery_keys(self):
        m = self.model.mastery(self.seqs)
        self.assertEqual(len(m), len(self.seqs))
        self.assertIn("s01/K1", m)
        self.assertTrue(all(0.0 <= v <= 1.0 for v in m.values()))

    def test_bounds_respected(self):
        for params in self.model.params().values():
            for name, value in params.items():
                with self.subTest(param=name):
                    self.assertGreaterEqual(value, 0.0)
                    self.assertLessEqual(value, 1.0)

    def test_report_has_keys(self):
        rep = self.model.report(self.seqs)
        for key in ("model", "groups", "nParams", "logLikelihood", "aic", "bic", "auc"):
            self.assertIn(key, rep)
        self.assertGreater(rep["bic"], rep["aic"])   # ln(n) > 2 时 BIC 罚得更狠

    def test_unfitted_raises(self):
        with self.assertRaises(RuntimeError):
            BKTModel().predict(self.seqs)

    def test_bad_model_name(self):
        with self.assertRaises(ValueError):
            BKTModel(model="per-class")

    def test_empty_input_raises(self):
        with self.assertRaises(ValueError):
            BKTModel().fit([])

    def test_params_accepts_dict(self):
        model = BKTModel(model="global", initial={"L0": 0.4, "S": 0.2, "G": 0.2, "T": 0.1},
                         n_init=1).fit(self.seqs)
        self.assertTrue(0.0 < model.params()["global"]["L0"] < 1.0)

    def test_min_obs_fallback(self):
        """每个学生样本都不够，per-user 应该整体退回 per-skill。"""
        model = BKTModel(model="per-user", min_obs=1000, n_init=1, seed=1).fit(self.seqs)
        self.assertEqual(model._per_user, {})
        self.assertEqual(model.n_params_, 4 * 2)   # 两个知识点，各一套

    def test_summarize(self):
        info = summarize(self.seqs)
        self.assertEqual(info["users"], 12)
        self.assertEqual(info["skills"], 2)
        self.assertEqual(info["observations"], 12 * 2 * 6)


if __name__ == "__main__":
    unittest.main()
