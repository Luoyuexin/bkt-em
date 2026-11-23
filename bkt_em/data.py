"""输入解析：把一行一条作答的长表切成一个学生在一个知识点上的作答序列。

BKT 建模的对象是序列，不是单条记录。这里做两件事：整理成序列，
并记住每一行在原表里的位置，预测结果才能按原顺序对齐回去。
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence as TSequence

# 常见表头写法，全接住，省得调用方来回改名
_ALIASES = {
    "user": ("user_id", "user", "student_id", "student", "uid", "learner_id"),
    "skill": ("skill_name", "skill", "kc", "knowledge", "knowledge_id", "kcs"),
    "correct": ("correct", "is_correct", "y", "label", "score", "outcome"),
    "order": ("order_id", "order", "step", "attempt", "seq", "timestamp", "time"),
    "difficulty": ("difficulty", "diff", "level", "difficulty_level", "难度"),
    # 注意：timestamp / time 归「顺序」列（那是 BKT 的题序），时间列只认下面这些，
    # 否则一列 timestamp 会被同时当成顺序和时间，语义就串了
    "time": ("answered_at", "created_at", "submit_time", "datetime", "答题时间", "时间"),
}

# 时间列的单位换算：统一成「天」，因为遗忘是按天走的
_TIME_UNITS = {
    "day": 1.0,
    "days": 1.0,
    "d": 1.0,
    "hour": 1.0 / 24,
    "hours": 1.0 / 24,
    "h": 1.0 / 24,
    "minute": 1.0 / 1440,
    "min": 1.0 / 1440,
    "second": 1.0 / 86400,
    "sec": 1.0 / 86400,
    "s": 1.0 / 86400,
    "ms": 1.0 / 86_400_000,
    "millisecond": 1.0 / 86_400_000,
}


@dataclass
class Sequence:
    """一个学生在同一个知识点上的一段连续作答。"""

    user: str
    skill: str
    y: list[int] = field(default_factory=list)          # 0/1 观测
    rows: list[int] = field(default_factory=list)        # 每行在原始输入里的下标
    difficulty: list[str] = field(default_factory=list)  # 每题的难度标签，可空
    time: list[float] = field(default_factory=list)      # 每题的时间，单位「天」，可空

    def __len__(self) -> int:
        return len(self.y)

    def gaps(self) -> list[float]:
        """相邻两题间隔了几天。长度 = len(y) - 1。没有时间列时返回空。"""
        if not self.time or len(self.time) != len(self.y):
            return []
        return [self.time[t] - self.time[t - 1] for t in range(1, len(self.y))]

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


def _as_days(value: Any, unit: str = "day") -> float | None:
    """把时间值统一换算成「天」。认不出来返回 None。"""
    try:
        raw = float(value)
    except (TypeError, ValueError):
        return None
    scale = _TIME_UNITS.get(str(unit).strip().lower())
    if scale is None:
        raise ValueError(
            f"不认识的 time_unit: {unit!r}，可用：{', '.join(sorted(_TIME_UNITS))}"
        )
    return raw * scale


def _normalize_difficulty(value: Any) -> str | None:
    """难度标签统一成小写字符串。空值返回 None（该题按「不调整」处理）。"""
    if value is None:
        return None
    text = str(value).strip().lower()
    return text or None


def to_sequences(rows: Iterable[Mapping[str, Any]],
                 user_col: str | None = None,
                 skill_col: str | None = None,
                 correct_col: str | None = None,
                 order_col: str | None = None,
                 difficulty_col: str | None = None,
                 time_col: str | None = None,
                 time_unit: str = "day",
                 min_length: int = 1) -> list[Sequence]:
    """长表 → 序列列表。

    rows 里每行至少要有用户、知识点、对错三样。列名默认按常见写法自动识别，
    也可以用 *_col 参数强制指定。

    排序：给了顺序列就按它排，否则按原表顺序。BKT 对顺序敏感，
    顺序错了估出来的 T（学习率）没有意义。

    difficulty_col 与 time_col 可选：
      · 给了难度，难题的「蒙对率」会被压低、「失误率」会被抬高
      · 给了时间，间隔越久掌握越可能退化（需要同时给 BKTModel 的 decay_lambda）
      · time_unit 指原始数据的时间单位（day / hour / second / ms …），
        内部统一换算成「天」
    """
    user_names = (user_col,) if user_col else _ALIASES["user"]
    skill_names = (skill_col,) if skill_col else _ALIASES["skill"]
    correct_names = (correct_col,) if correct_col else _ALIASES["correct"]
    order_names = (order_col,) if order_col else _ALIASES["order"]
    diff_names = (difficulty_col,) if difficulty_col else _ALIASES["difficulty"]
    time_names = (time_col,) if time_col else _ALIASES["time"]

    indexed: list[tuple[str, str, float, int, int, str | None, float | None]] = []
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
        diff = _normalize_difficulty(_pick(row, diff_names))
        raw_time = _pick(row, time_names)
        ts = _as_days(raw_time, time_unit) if raw_time is not None else None
        indexed.append((str(user), str(skill), order, idx, y, diff, ts))

    buckets: dict[tuple[str, str], list[tuple[float, int, int, str | None, float | None]]] = {}
    for user, skill, order, idx, y, diff, ts in indexed:
        buckets.setdefault((user, skill), []).append((order, idx, y, diff, ts))

    out: list[Sequence] = []
    for (user, skill), items in buckets.items():
        items.sort(key=lambda t: (t[0], t[1]))
        if len(items) < min_length:
            continue
        diffs = [t[3] for t in items]
        times = [t[4] for t in items]
        out.append(Sequence(
            user=user,
            skill=skill,
            y=[t[2] for t in items],
            rows=[t[1] for t in items],
            # 整条序列的难度/时间都齐了才带上，缺一半就当没有，
            # 免得下游拿到不等长的列表还要自己判断
            difficulty=diffs if all(d is not None for d in diffs) else [],
            time=times if all(v is not None for v in times) else [],
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
    """数据集概况。拟合前先看一眼，数据太少估不出四个参数。"""
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
