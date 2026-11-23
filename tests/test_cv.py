"""交叉验证的测试：折怎么切、指标怎么汇总。"""

from __future__ import annotations

import unittest

from bkt_em import BKTModel, Params, compare_models, cross_validate
from bkt_em.simulate import simulate

TRUTH = Params(L0=0.35, S=0.10, G=0.20, T=0.25)


class TestCrossValidate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.seqs = simulate(n_users=24, n_skills=2, n_items=8, params=TRUTH,
                            seed=31, ability_sd=0.10)

    def test_fold_count(self):
        rep = cross_validate(self.seqs, k=4, n_init=1, seed=1)
        self.assertEqual(rep["k"], 4)
        self.assertEqual(len(rep["perFold"]), 4)
        self.assertEqual(sum(f["users"] for f in rep["perFold"]), 24)

    def test_metrics_present(self):
        rep = cross_validate(self.seqs, k=3, n_init=1, seed=1)
        for key in ("auc", "accuracy", "logLoss", "rmse", "n"):
            self.assertIn(key, rep)
        self.assertEqual(rep["n"], 24 * 2 * 8)
        self.assertTrue(0.0 < rep["auc"] <= 1.0)

    def test_k_larger_than_users_is_clamped(self):
        rep = cross_validate(self.seqs, k=999, n_init=1, seed=1)
        self.assertLessEqual(rep["k"], 24)

    def test_same_seed_same_result(self):
        a = cross_validate(self.seqs, k=3, n_init=1, seed=5)
        b = cross_validate(self.seqs, k=3, n_init=1, seed=5)
        self.assertAlmostEqual(a["logLoss"], b["logLoss"], places=12)

    def test_different_seed_moves_result(self):
        a = cross_validate(self.seqs, k=3, n_init=1, seed=5)
        b = cross_validate(self.seqs, k=3, n_init=1, seed=6)
        self.assertNotAlmostEqual(a["logLoss"], b["logLoss"], places=12)

    def test_needs_two_users(self):
        one = simulate(n_users=1, n_skills=1, n_items=5, seed=1)
        with self.assertRaises(ValueError):
            cross_validate(one, k=3)

    def test_unseen_users_fall_back(self):
        """per-user 模式碰到没见过的学生，只能退回知识点级参数（不是全班一套）。"""
        cv_user = cross_validate(self.seqs, k=3, model="per-user", min_obs=10_000,
                                 n_init=1, seed=1)
        cv_skill = cross_validate(self.seqs, k=3, model="per-skill", n_init=1, seed=1)
        self.assertAlmostEqual(cv_user["logLoss"], cv_skill["logLoss"], places=12)


class TestCompareModels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.seqs = simulate(n_users=20, n_skills=3, n_items=6, params=TRUTH, seed=41)
        cls.rep = compare_models(cls.seqs, k=2, n_init=1, seed=1)

    def test_reports_all_models(self):
        self.assertEqual(set(self.rep), {"global", "per-skill", "per-user"})
        for row in self.rep.values():
            for key in ("nParams", "logLikelihood", "aic", "bic", "cvAuc"):
                self.assertIn(key, row)

    def test_more_params_more_penalty(self):
        self.assertLess(self.rep["global"]["nParams"], self.rep["per-skill"]["nParams"])
        self.assertGreaterEqual(self.rep["per-skill"]["logLikelihood"],
                                self.rep["global"]["logLikelihood"])

    def test_train_likelihood_never_the_judge(self):
        """per-skill 训练集似然更高，但参数多了 AIC 未必更好，这正是要交叉验证的原因。"""
        g = BKTModel(model="global", n_init=1, seed=1).fit(self.seqs)
        ps = BKTModel(model="per-skill", n_init=1, seed=1).fit(self.seqs)
        self.assertGreaterEqual(ps.log_likelihood_, g.log_likelihood_)
        self.assertGreater(ps.n_params_, g.n_params_)


if __name__ == "__main__":
    unittest.main()
