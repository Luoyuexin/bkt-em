"""模拟数据：已知真值灌进去，看 EM 能不能把真值捞回来。

这是评估「实现到底对不对」最直接的办法 —— 拿真实数据只能看 AUC 高不高，
但你永远不知道参数该是多少；拿模拟数据就有标准答案：

    真值 L0=0.30 S=0.08 G=0.20 T=0.25
      ↓ 按模型生成作答序列
    拟合 → 估出 L0=? S=? G=? T=?

估出来的和真值差得远，那一定是实现有问题（下标错位、gamma/xi 归一不一致
之类），而不是「数据不好」。

0.2.0 起还能顺带造「带难度」和「带时间间隔」的数据，用来验证这两条扩展
到底有没有按预期工作。
"""

from __future__ import annotations

import math
import random
from typing import Sequence as TSequence

from .data import Sequence
from .em import DIFFICULTY_SCALE, Params


def simulate(n_users: int = 60, n_skills: int = 4, n_items: int = 20, *,
             params: Params | dict[str, float] | None = None,
             seed: int = 0,
             ability_sd: float = 0.0,
             difficulty_cycle: TSequence[str] | None = None,
             gap_days: float = 0.0,
             gap_jitter: float = 0.5,
             decay_lambda: float = 0.0,
             prefix: str = "s") -> list[Sequence]:
    """按 BKT 生成作答序列。

    参数
    ----
    n_users / n_skills / n_items   学生数 / 知识点数 / 每人每知识点做几题
    params                         真值。给 Params 或 dict 都行
    seed                           随机种子，同种子必得同一份数据
    ability_sd                     学生之间初始水平的差异（L0 的标准差）。
                                   设 0 就是全班一个水平 —— 验证参数恢复时用这个，
                                   因为「每人 L0 不同」本身就不符合 per-skill 模型
    difficulty_cycle               难度标签循环，如 ("basic","medium","hard")。
                                   难度按同一套系数压低蒙对率、抬高失误率
    gap_days                       两题之间隔几天。0 表示不带时间列
    gap_jitter                     间隔的随机抖动幅度（0.5 表示 ±50%）
    decay_lambda                   生成时的真实时间衰减系数（1/天）。
                                   需要 gap_days > 0 才有意义
    prefix                         学生 id 前缀，生成 s01 / s02 ...

    返回按 (user, skill) 排好序的序列列表，可以直接丢给 ``BKTModel.fit``。

    注意：真值里的 F 与 decay_lambda 会叠加成 ``(1-F)·exp(-λΔt)`` ——
    生成的「练完就忘」和「隔久了会忘」是同一个式子的两个因子。
    """
    if n_users < 1 or n_skills < 1 or n_items < 1:
        raise ValueError("n_users / n_skills / n_items 都得是正整数")
    if gap_jitter < 0:
        raise ValueError("gap_jitter 不能为负")

    truth = params if isinstance(params, Params) else Params(**(params or {}))
    rng = random.Random(seed)
    width = max(2, len(str(n_users)))

    cycle = list(difficulty_cycle) if difficulty_cycle else []
    diffs = [cycle[i % len(cycle)] for i in range(n_items)] if cycle else []
    scales = [DIFFICULTY_SCALE.get((d or "").strip().lower(), (1.0, 1.0)) for d in diffs]

    out: list[Sequence] = []
    for u in range(n_users):
        user = f"{prefix}{u + 1:0{width}d}"
        l0 = truth.L0 + (rng.gauss(0.0, ability_sd) if ability_sd else 0.0)
        l0 = min(0.999, max(0.001, l0))
        for k in range(n_skills):
            skill = f"K{k + 1}"
            # 时间轴只在需要时生成 —— 否则会多消耗随机数，
            # 让「不带时间」的老用法跑出来的数据和 0.1.0 不一致
            times: list[float] = []
            if gap_days > 0:
                clock = 0.0
                for _ in range(n_items):
                    times.append(clock)
                    clock += gap_days * (1.0 + rng.uniform(-gap_jitter, gap_jitter))

            mastered = rng.random() < l0
            ys: list[int] = []
            for i in range(n_items):
                s_scale, g_scale = scales[i] if scales else (1.0, 1.0)
                p_hit = (1.0 - truth.S * s_scale) if mastered else min(1.0, truth.G * g_scale)
                ys.append(1 if rng.random() < p_hit else 0)
                # 答完这一题之后才发生状态转移：先可能记住/学会，再可能遗忘
                keep = 1.0 - truth.F
                if times and i < n_items - 1:
                    keep *= math.exp(-decay_lambda * (times[i + 1] - times[i]))
                if mastered:
                    mastered = rng.random() < keep
                else:
                    mastered = rng.random() < truth.T
            out.append(Sequence(user=user, skill=skill, y=ys, rows=[],
                                difficulty=diffs if diffs else [],
                                time=times if times else []))
    out.sort(key=lambda s: (s.user, s.skill))
    return out


def attach_rows(sequences: TSequence[Sequence]) -> list[Sequence]:
    """给模拟序列补上连续行号，方便测试 :meth:`BKTModel.predict_rows`。"""
    row = 0
    for s in sequences:
        s.rows = list(range(row, row + len(s)))
        row += len(s)
    return list(sequences)
