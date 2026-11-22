"""输入解析的测试：列名识别、排序、行号对齐。"""

from __future__ import annotations

import os
import tempfile
import unittest

from bkt_em import load_csv, to_sequences
from bkt_em.data import _as_bool01


def rows(*items):
    return [dict(zip(("user_id", "skill_name", "correct", "order_id"), it)) for it in items]


class TestToSequences(unittest.TestCase):
    def test_groups_by_user_and_skill(self):
        seqs = to_sequences([
            {"user_id": "a", "skill_name": "K1", "correct": 1},
            {"user_id": "a", "skill_name": "K1", "correct": 0},
            {"user_id": "a", "skill_name": "K2", "correct": 1},
            {"user_id": "b", "skill_name": "K1", "correct": 1},
        ])
        self.assertEqual(len(seqs), 3)
        got = {(s.user, s.skill): s.y for s in seqs}
        self.assertEqual(got[("a", "K1")], [1, 0])
        self.assertEqual(got[("a", "K2")], [1])
        self.assertEqual(got[("b", "K1")], [1])

    def test_sorts_by_order_column(self):
        seqs = to_sequences(rows(("a", "K1", 1, 3), ("a", "K1", 0, 1), ("a", "K1", 1, 2)))
        self.assertEqual(seqs[0].y, [0, 1, 1])

    def test_keeps_original_row_index(self):
        seqs = to_sequences(rows(("a", "K1", 1, 3), ("a", "K1", 0, 1), ("a", "K1", 1, 2)))
        self.assertEqual(seqs[0].rows, [1, 2, 0])

    def test_alias_column_names(self):
        seqs = to_sequences([
            {"student": "a", "kc": "K1", "is_correct": True},
            {"student": "a", "kc": "K1", "is_correct": False},
        ])
        self.assertEqual(seqs[0].y, [1, 0])

    def test_explicit_columns(self):
        seqs = to_sequences(
            [{"who": "a", "what": "K1", "hit": "对"}, {"who": "a", "what": "K1", "hit": "错"}],
            user_col="who", skill_col="what", correct_col="hit",
        )
        self.assertEqual(seqs[0].y, [1, 0])

    def test_drops_unparsable_rows(self):
        seqs = to_sequences([
            {"user_id": "a", "skill_name": "K1", "correct": "对"},
            {"user_id": "a", "skill_name": "K1", "correct": "不确定"},
            {"user_id": "a", "skill_name": "K1", "correct": None},
        ])
        self.assertEqual(len(seqs[0].y), 1)

    def test_min_length_filters(self):
        data = rows(("a", "K1", 1, 1), ("b", "K1", 1, 1), ("b", "K1", 0, 2))
        self.assertEqual(len(to_sequences(data)), 2)
        self.assertEqual(len(to_sequences(data, min_length=2)), 1)

    def test_stable_user_order(self):
        data = rows(*[(u, "K1", 1, i) for i, u in enumerate(["c", "a", "b"])])
        self.assertEqual([s.user for s in to_sequences(data)], ["a", "b", "c"])


class TestBool01(unittest.TestCase):
    CASES = [
        (True, 1), (False, 0), (1, 1), (0, 0), (1.0, 1), (0.0, 0),
        ("1", 1), ("0", 0), ("T", 1), ("f", 0), ("对", 1), ("错", 0),
        ("correct", 1), ("wrong", 0), ("0.9", 1), ("0.1", 0),
    ]

    def test_known_spellings(self):
        for raw, want in self.CASES:
            with self.subTest(raw=raw):
                self.assertEqual(_as_bool01(raw), want)

    def test_unknown_returns_none(self):
        for raw in ("", "不确定", "n/a", None, object()):
            with self.subTest(raw=raw):
                self.assertIsNone(_as_bool01(raw))


class TestLoadCsv(unittest.TestCase):
    def test_roundtrip_with_bom(self):
        text = "user_id,skill_name,correct,order_id\na,K1,1,1\na,K1,0,2\n"
        fd, path = tempfile.mkstemp(suffix=".csv")
        os.close(fd)
        try:
            with open(path, "w", encoding="utf-8-sig", newline="") as fh:
                fh.write(text)
            seqs = load_csv(path)
            self.assertEqual(len(seqs), 1)
            self.assertEqual(seqs[0].y, [1, 0])
        finally:
            os.remove(path)


if __name__ == "__main__":
    unittest.main()
