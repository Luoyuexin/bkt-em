"""模拟数据：已知真值灌进去，看 EM 能不能把真值捞回来。

这是评估「实现到底对不对」最直接的办法 —— 拿真实数据只能看 AUC 高不高，
但你永远不知道参数该是多少；拿模拟数据就有标准答案：

    真值 L0=0.30 S=0.08 G=0.20 T=0.25
      ↓ 按模型生成作答序列
    拟合 → 估出 L0=? S=? G=? T=?

估出来的和真值差得远，那一定是实现有问题（梯度写错、下标错位、
gamma/xi 归一不一致之类），而不是「数据不好」。
"""

from __future__ import annotations

import random
from typing import Sequence as TSequence

from .data import Sequence
from .em import Params


def simulate(n_users: int = 60, n_skills: int = 4, n_items: int = 20, *,
             params: Params | dict[str, float] | None = None,
             seed: int = 0,
             ability_sd: float = 0.0,
             prefix: str = "s") -> list[Sequence]:
    """按 BKT 生成作答序列。

    参数
    ----
    n_users / n_skills / n_items   学生数 / 知识点数 / 每人每知识点做几题
    params                         真值。给 Params 或 dict 都行
    ability_sd                     学生之间初始水平的差异（L0 的标准差）。
                                   设 0 就是全班一个水平 —— 验证参数恢复时用这个，
                                   因为「每人 L0 不同」本身就不符合 per-skill 模型
    prefix                         学生 id 前缀，生成 s01 / s02 ...

    返回按 (user, skill) 排好序的序列列表，可以直接丢给 ``BKTModel.fit``。
    """
    if n_users < 1 or n_skills < 1 or n_items < 1:
        raise ValueError("n_users / n_skills / n_items 都得是正整数")

    truth = params if isinstance(params, Params) else Params(**(params or {}))
    rng = random.Random(seed)
    a00, a01 = 1.0 - truth.T, truth.T
    a10, a11 = truth.F, 1.0 - truth.F
    p_correct_if_mastered = 1.0 - truth.S
    p_correct_if_not = truth.G

    out: list[Sequence] = []
    width = max(2, len(str(n_users)))
    for u in range(n_users):
        user = f"{prefix}{u + 1:0{width}d}"
        l0 = truth.L0 + (rng.gauss(0.0, ability_sd) if ability_sd else 0.0)
        l0 = min(0.999, max(0.001, l0))
        for k in range(n_skills):
            skill = f"K{k + 1}"
            mastered = rng.random() < l0
            ys: list[int] = []
            for _ in range(n_items):
                p_hit = p_correct_if_mastered if mastered else p_correct_if_not
                ys.append(1 if rng.random() < p_hit else 0)
                # 答完这一题之后才发生状态转移（学习/遗忘）
                mastered = rng.random() < (a11 if mastered else a01)
            out.append(Sequence(user=user, skill=skill, y=ys, rows=[]))
    out.sort(key=lambda s: (s.user, s.skill))
    return out


def attach_rows(sequences: TSequence[Sequence]) -> list[Sequence]:
    """给模拟序列补上连续行号，方便测试 :meth:`BKTModel.predict_rows`。"""
    row = 0
    for s in sequences:
        s.rows = list(range(row, row + len(s)))
        row += len(s)
    return list(sequences)
