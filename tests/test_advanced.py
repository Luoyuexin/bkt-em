"""0.2.0 新增能力的测试：题目难度、时间衰减。

这两条扩展最容易「看起来能用、其实没接上」，所以每个都要验两件事：
  1. 行为方向对不对（难题预测更低、隔得越久忘得越多）
  2. 真的进了模型（关掉就变回去、开了似然真的更高）
"""

from __future__ import annotations

import math
import unittest
import warnings

from bkt_em import BKTModel, Params, to_sequences
from bkt_em.data import _as_days
from bkt_em.em import DIFFICULTY_SCALE, _accumulate, _filter, _m_step, _scan
from bkt_em.simulate import simulate

TRUTH = Params(L0=0.30, S=0.10, G=0.20, T=0.25)
CYCLE = ("basic", "medium", "hard")


class TestDifficulty(unittest.TestCase):
    def test_harder_means_lower_prediction(self):
        """同一个学生、同样的作答历史，题目越难，预测做对的概率越低。"""
        p = Params(L0=0.5, S=0.10, G=0.20, T=0.20)
        first = {}
        for label in CYCLE:
            probs, _ = _filter([1, 1, 1], p, 0.0, [label] * 3, None)
            first[label] = probs[0]
        self.assertGreater(first["basic"], first["medium"])
        self.assertGreater(first["medium"], first["hard"])

    def test_difficulty_scales_are_monotone(self):
        g_scales = [DIFFICULTY_SCALE[d][1] for d in ("basic", "medium", "hard")]
        s_scales = [DIFFICULTY_SCALE[d][0] for d in ("basic", "medium", "hard")]
        self.assertEqual(g_scales, sorted(g_scales, reverse=True))   # 越难越难蒙对
        self.assertEqual(s_scales, sorted(s_scales))                 # 越难越容易失误

    def test_aliases_and_unknown(self):
        p = Params(L0=0.5, S=0.1, G=0.2)
        base = _filter([1], p, 0.0, ["medium"], None)[0][0]
        for label in ("medium", "normal", "中等", "MEDIUM", " Medium "):
            with self.subTest(label=label):
                self.assertAlmostEqual(_filter([1], p, 0.0, [label], None)[0][0], base, places=12)
        # 认不出来的标签按「不调整」处理，不报错 —— 不知道的就不该假装知道
        neutral = _filter([1], p, 0.0, None, None)[0][0]
        for label in ("???", "", "very-hard"):
            with self.subTest(label=label):
                self.assertAlmostEqual(
                    _filter([1], p, 0.0, [label], None)[0][0], neutral, places=12)

    def test_difficulty_enters_likelihood(self):
        """难度必须真的影响似然，否则等于没接上。"""
        seqs = simulate(n_users=150, n_skills=1, n_items=12, params=TRUTH,
                        difficulty_cycle=("basic", "hard"), seed=17)
        p = Params(L0=0.3, S=0.1, G=0.2, T=0.25)
        with_diff = _clip_all(seqs, p)
        no_diff = _clip_all([_strip_difficulty(s) for s in seqs], p)
        self.assertNotAlmostEqual(with_diff, no_diff, places=6)

    def test_difficulty_improves_likelihood(self):
        """按难度拟合，应该比忽略难度拟合拿到更高的似然与更好的 AUC。"""
        seqs = simulate(n_users=300, n_skills=1, n_items=12, params=TRUTH,
                        difficulty_cycle=("basic", "hard"), seed=17)
        flat = [_strip_difficulty(s) for s in seqs]
        with_diff = BKTModel(model="global", n_init=2, seed=1).fit(seqs)
        no_diff = BKTModel(model="global", n_init=2, seed=1).fit(flat)
        self.assertGreater(with_diff.log_likelihood_, no_diff.log_likelihood_)
        self.assertGreater(with_diff.metrics(seqs)["auc"], no_diff.metrics(flat)["auc"])

    def test_em_monotone_with_difficulty(self):
        """带难度时，加了似然守护的迭代过程必须单调不降。"""
        seqs = simulate(n_users=40, n_skills=1, n_items=12, params=TRUTH,
                        difficulty_cycle=CYCLE, seed=3)
        p, prev, prev_ll = Params(), -math.inf, -math.inf
        for _ in range(60):
            stats = _accumulate(seqs, p, 0.0)
            ll = stats["ll"]
            self.assertGreaterEqual(ll, prev_ll - 1e-12)
            prev_ll = ll
            cand = _m_step(stats, p, False, (1e-3, 1 - 1e-3))
            probe = _accumulate(seqs, cand, 0.0)
            if probe is None or probe["ll"] < ll - 1e-12:
                break
            p = cand
            if abs(ll - prev) < 1e-6:
                break
            prev = ll

    def test_parameters_recovered_with_difficulty(self):
        seqs = simulate(n_users=300, n_skills=1, n_items=12, params=TRUTH,
                        difficulty_cycle=CYCLE, seed=5)
        est = BKTModel(model="global", n_init=3, seed=1).fit(seqs).params()["global"]
        for name, want in TRUTH.as_dict().items():
            with self.subTest(param=name):
                self.assertAlmostEqual(est[name], want, delta=0.12)


class TestTimeDecay(unittest.TestCase):
    def test_longer_gap_means_less_mastery(self):
        """间隔越久，学完之后的掌握后验越低。"""
        p = Params(L0=0.9, S=0.05, G=0.10, T=0.10)
        posts = []
        for gap in (0.0, 2.0, 7.0, 30.0):
            _, post = _filter([1, 1, 1], p, 0.08, None, [0.0, gap, 2 * gap])
            posts.append(post[1])
        self.assertEqual(posts, sorted(posts, reverse=True))
        self.assertGreater(posts[0] - posts[-1], 0.2)

    def test_lambda_zero_is_no_op(self):
        seqs = simulate(n_users=200, n_skills=1, n_items=10, params=TRUTH, seed=21)
        a = BKTModel(model="global", n_init=1, seed=1).fit(seqs)
        b = BKTModel(model="global", n_init=1, seed=1, decay_lambda=0.0).fit(seqs)
        self.assertEqual(a.params(), b.params())
        self.assertEqual(a.log_likelihood_, b.log_likelihood_)

    def test_decay_improves_likelihood_when_present(self):
        """数据里真有时间衰减时，用对 λ 的似然必须明显高于当它不存在。"""
        seqs = simulate(n_users=250, n_skills=1, n_items=10, params=TRUTH,
                        gap_days=4.0, decay_lambda=0.08, seed=9)
        off = BKTModel(model="global", n_init=2, seed=1).fit(seqs)
        on = BKTModel(model="global", decay_lambda=0.08, n_init=2, seed=1).fit(seqs)
        self.assertGreater(on.log_likelihood_, off.log_likelihood_ + 10)

    def test_parameters_recovered_with_decay(self):
        truth = Params(L0=0.60, S=0.10, G=0.20, T=0.15)
        seqs = simulate(n_users=250, n_skills=1, n_items=10, params=truth,
                        gap_days=4.0, decay_lambda=0.08, seed=9)
        est = BKTModel(model="global", decay_lambda=0.08, n_init=2, seed=1)\
            .fit(seqs).params()["global"]
        for name in ("L0", "S", "G", "T"):
            with self.subTest(param=name):
                self.assertAlmostEqual(est[name], truth.as_dict()[name], delta=0.15)

    def test_decay_and_forget_combine(self):
        """F 和 λ 应该叠成 (1-F)·exp(-λΔt)，两个都能同时开。"""
        seqs = simulate(n_users=120, n_skills=1, n_items=10, params=TRUTH,
                        gap_days=3.0, decay_lambda=0.05, seed=11)
        model = BKTModel(model="global", forget=True, decay_lambda=0.05,
                         n_init=2, seed=1).fit(seqs)
        self.assertEqual(model.n_params_, 5)
        self.assertGreaterEqual(model.params()["global"]["F"], 0.0)

    def test_warns_when_lambda_without_time(self):
        seqs = simulate(n_users=20, n_skills=1, n_items=6, params=TRUTH, seed=2)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            BKTModel(model="global", decay_lambda=0.05, n_init=1).fit(seqs)
        self.assertTrue(any("时间衰减不会生效" in str(w.message) for w in caught))

    def test_negative_lambda_rejected(self):
        with self.assertRaises(ValueError):
            BKTModel(decay_lambda=-0.1)

    def test_report_exposes_lambda(self):
        seqs = simulate(n_users=20, n_skills=1, n_items=6, params=TRUTH, seed=3)
        model = BKTModel(model="global", n_init=1).fit(seqs)
        self.assertEqual(model.report(seqs)["decayLambda"], 0.0)


class TestParsing(unittest.TestCase):
    def test_difficulty_and_time_columns(self):
        rows = [
            {"user_id": "a", "skill_name": "K1", "correct": 1, "order_id": 1,
             "difficulty": "HARD", "answered_at": 0},
            {"user_id": "a", "skill_name": "K1", "correct": 0, "order_id": 2,
             "difficulty": "medium", "answered_at": 86_400_000},
        ]
        seq = to_sequences(rows, time_unit="ms")[0]
        self.assertEqual(seq.difficulty, ["hard", "medium"])
        self.assertEqual(seq.time, [0.0, 1.0])
        self.assertEqual(seq.gaps(), [1.0])

    def test_time_units(self):
        cases = [("day", 2.0, 2.0), ("hour", 48.0, 2.0), ("minute", 2880.0, 2.0),
                 ("second", 172_800.0, 2.0), ("ms", 172_800_000.0, 2.0)]
        for unit, raw, want_days in cases:
            with self.subTest(unit=unit):
                self.assertAlmostEqual(_as_days(raw, unit), want_days, places=9)

    def test_unknown_unit_raises(self):
        with self.assertRaises(ValueError):
            _as_days(1.0, "fortnight")

    def test_partial_difficulty_is_dropped(self):
        """一条序列里难度缺一半，就整条当成没有难度 —— 避免下游拿到错位列表。"""
        rows = [
            {"user_id": "a", "skill_name": "K1", "correct": 1, "order_id": 1, "difficulty": "hard"},
            {"user_id": "a", "skill_name": "K1", "correct": 0, "order_id": 2},
        ]
        self.assertEqual(to_sequences(rows)[0].difficulty, [])

    def test_no_time_means_empty(self):
        rows = [{"user_id": "a", "skill_name": "K1", "correct": 1, "order_id": 1}]
        seq = to_sequences(rows)[0]
        self.assertEqual(seq.time, [])
        self.assertEqual(seq.gaps(), [])

    def test_order_column_not_stolen_by_time(self):
        """timestamp 属于「顺序」列，不能被时间列抢走。"""
        rows = [
            {"user_id": "a", "skill_name": "K1", "correct": 1, "timestamp": 3},
            {"user_id": "a", "skill_name": "K1", "correct": 0, "timestamp": 1},
        ]
        seq = to_sequences(rows)[0]
        self.assertEqual(seq.y, [0, 1])     # 按 timestamp 升序排
        self.assertEqual(seq.time, [])      # 但不当作时间衰减的依据

    def test_scan_unchanged_without_extras(self):
        """没有难度/时间时，_scan 必须与 0.1.0 完全一致（穷举校验兜底）。"""
        p = Params(L0=0.3, S=0.1, G=0.2, T=0.25, F=0.05)
        y = [1, 0, 1]
        self.assertAlmostEqual(_scan(y, p)["ll"], _scan(y, p, 0.0, None, None)["ll"], places=12)


def _strip_difficulty(seq):
    """复制一条序列但抹掉难度标签。"""
    from bkt_em.data import Sequence
    return Sequence(user=seq.user, skill=seq.skill, y=list(seq.y),
                    rows=list(seq.rows), time=list(seq.time))


def _clip_all(seqs, p) -> float:
    total = 0.0
    for s in seqs:
        stats = _scan(s.y, p, 0.0, s.difficulty, s.time)
        if stats:
            total += stats["ll"]
    return total


if __name__ == "__main__":
    unittest.main()
