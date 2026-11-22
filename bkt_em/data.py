"""输入解析：把「一行一条作答」的长表切成「一个学生在一个知识点上的作答序列」。

BKT 建模的对象是**序列**，不是单条记录。一条记录只说明「这次答对/答错」，
连续几条才能看出「他是不是越练越对」——而这正是 BKT 要估的东西。

所以这个模块负责一件事：把长表整理成序列，并且**记住每一行在原表里的位置**，
这样预测结果能按原来的顺序对齐回去，调用方不用自己数行。
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence as TSequence

# 列名允许的几种写法 —— 不同人导出的表头叫法不一样，全接住省得来回改名
_ALIASES = {
    "user": ("user_id", "user", "student_id", "student", "uid", "learner_id"),
    "skill": ("skill_name", "skill", "kc", "knowledge", "knowledge_id", "kcs"),
    "correct": ("correct", "is_correct", "y", "label", "score", "outcome"),
    "order": ("order_id", "order", "step", "attempt", "seq", "timestamp", "time"),
}


@dataclass
class Sequence:
    """一个学生在同一个知识点上的一段连续作答。"""

    user: str
    skill: str
    y: list[int] = field(default_factory=list)          # 0/1 观测
    rows: list[int] = field(default_factory=list)        # 每行在原始输入里的下标

    def __len__(self) -> int:
        return len(self.y)

    def __repr__(self) -> str:  # pragma: no cover - 只为调试好看
        return f"<Sequence {self.user}/{self.skill} n={len(self.y)}>"


def _pick(row: Mapping[str, Any], names: TSequence[str]) -> Any:
    for name in names:
        if name in row and row[name] is not None and row[name] != "":
            return row[name]
    return None


def _as_bool01(value: Any) -> int | None:
    """把各种写法的「对/错」统一成 1/0。认不出来返回 None（该行丢弃）。"""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return 1 if float(value) >= 0.5 else 0
    text = str(value).strip().lower()
    if text in ("1", "true", "t", "yes", "y", "对", "正确", "right", "correct"):
        return 1
    if text in ("0", "false", "f", "no", "n", "错", "错误", "wrong", "incorrect"):
        return 0
    try:
        return 1 if float(text) >= 0.5 else 0
    except (TypeError, ValueError):
        return None


def to_sequences(rows: Iterable[Mapping[str, Any]],
                 user_col: str | None = None,
                 skill_col: str | None = None,
                 correct_col: str | None = None,
                 order_col: str | None = None,
                 min_length: int = 1) -> list[Sequence]:
    """长表 → 序列列表。

    rows 里每行至少要有用户、知识点、对错三样。列名默认按常见写法自动识别，
    也可以用 *_col 参数强制指定。

    排序：给了顺序列就按它排，没给就按原表顺序 —— 因为 BKT 对顺序敏感，
    顺序错了估出来的 T（学习率）就没有意义。
    """
    user_names = (user_col,) if user_col else _ALIASES["user"]
    skill_names = (skill_col,) if skill_col else _ALIASES["skill"]
    correct_names = (correct_col,) if correct_col else _ALIASES["correct"]
    order_names = (order_col,) if order_col else _ALIASES["order"]

    indexed: list[tuple[str, str, float, int, int]] = []  # (user, skill, order, idx, y)
    for idx, row in enumerate(rows):
        user = _pick(row, user_names)
        skill = _pick(row, skill_names)
        y = _as_bool01(_pick(row, correct_names))
        if user is None or skill is None or y is None:
            continue
        raw_order = _pick(row, order_names)
        try:
            order = float(raw_order)
        except (TypeError, ValueError):
            order = float(idx)  # 没有顺序列就用行号，至少是稳定的
        indexed.append((str(user), str(skill), order, idx, y))

    buckets: dict[tuple[str, str], list[tuple[float, int, int]]] = {}
    for user, skill, order, idx, y in indexed:
        buckets.setdefault((user, skill), []).append((order, idx, y))

    out: list[Sequence] = []
    for (user, skill), items in buckets.items():
        items.sort(key=lambda t: (t[0], t[1]))
        if len(items) < min_length:
            continue
        out.append(Sequence(
            user=user,
            skill=skill,
            y=[t[2] for t in items],
            rows=[t[1] for t in items],
        ))
    # 稳定排序：同一用户的序列挨在一起，方便 per-user 模式复用
    out.sort(key=lambda s: (s.user, s.skill))
    return out


def load_csv(path: str, **kwargs: Any) -> list[Sequence]:
    """从 CSV 读作答记录。用标准库 csv，不引入 pandas。"""
    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        return to_sequences(list(reader), **kwargs)


def summarize(sequences: TSequence[Sequence]) -> dict[str, Any]:
    """数据集概况 —— 拟合前先看一眼，数据不够就别指望估出四个参数。"""
    users = {s.user for s in sequences}
    skills = {s.skill for s in sequences}
    n_obs = sum(len(s) for s in sequences)
    correct = sum(sum(s.y) for s in sequences)
    lengths = [len(s) for s in sequences] or [0]
    return {
        "sequences": len(sequences),
        "users": len(users),
        "skills": len(skills),
        "observations": n_obs,
        "accuracy": round(correct / n_obs, 4) if n_obs else None,
        "meanLength": round(sum(lengths) / len(lengths), 2),
        "maxLength": max(lengths),
    }
