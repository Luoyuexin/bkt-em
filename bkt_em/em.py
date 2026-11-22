"""EM 求解：拿作答序列，反推 L0 / S / G / T（可选 F）。

为什么用 EM 而不是梯度下降
--------------------------
BKT 的隐状态只有 0/1 两种，是一件非常「干净」的模型：

    · E 步能精确算出每一步处在「已掌握」状态的后验概率（前向-后向）
    · M 步能写成闭式解（就是加权计数），不需要调学习率、不会梯度爆炸

代价是似然面可能有多个局部极大，所以默认从几个随机起点各跑一次，
取对数似然最高的那组参数（``n_init`` 控制次数，``n_init=1`` 表示只用初始值）。

一个数值细节
------------
序列一长，前向概率会连乘到浮点下溢（几十道题就是 1e-40 量级）。
这里用标准做法：每步除掉一个缩放因子 c_t，最后把 log c_t 加起来就是
对数似然。既不下溢，算出来的似然还是准的。
"""

from __future__ import annotations

import math
import random
import warnings
from dataclasses import dataclass, replace
from typing import Any, Iterable, Iterator, Sequence as TSequence

from .data import Sequence

_EPS = 1e-12
_NEG_INF = float("-inf")

# 题目难度怎么进模型：不是在外层乘一个权重，而是直接改发射概率
#   S_eff = S × s_scale      难题更容易「会了却答错」
#   G_eff = G × g_scale      难题更难「不会却蒙对」
# 三档系数与教学侧的常识一致：基础题蒙对率高、失误率低；难题反过来。
# 这套数值和项目里 knowledge.py 的 DIFFICULTY_S / DIFFICULTY_G 完全一致，
# 两边算出来的掌握度才能对得上。
DIFFICULTY_SCALE: dict[str, tuple[float, float]] = {
    "basic": (0.90, 1.20),
    "easy": (0.90, 1.20),
    "简单": (0.90, 1.20),
    "medium": (1.05, 0.95),
    "normal": (1.05, 0.95),
    "中等": (1.05, 0.95),
    "hard": (1.25, 0.65),
    "困难": (1.25, 0.65),
    "难": (1.25, 0.65),
}
_DEFAULT_SCALE = (1.0, 1.0)


@dataclass(frozen=True)
class Params:
    """一个知识点（或者一个学生，或者全班）的四/五个参数。

    L0  初始掌握概率      pristine
    S   失误率（会了却答错）
    G   猜测率（不会却蒙对）
    T   学习转移率（练一次就从不会变成会）
    F   遗忘率（会了又忘了），默认 0，即标准 BKT
    """

    L0: float = 0.25
    S: float = 0.10
    G: float = 0.25
    T: float = 0.15
    F: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {"L0": self.L0, "S": self.S, "G": self.G, "T": self.T, "F": self.F}

    def emission(self, difficulty: str | None = None) -> tuple[float, float]:
        """(P(答对|未掌握), P(答对|已掌握))，可选按难度打折。

        难度直接改两端概率，不改变点估计的语义：S/G 估的是「medium 基准题」
        上的失误率与猜测率，难度系数是一层乘法偏移。
        """
        s_scale, g_scale = DIFFICULTY_SCALE.get(
            (difficulty or "").strip().lower(), _DEFAULT_SCALE
        )
        return (
            min(1.0 - _EPS, max(_EPS, self.G * g_scale)),
            min(1.0 - _EPS, max(_EPS, 1.0 - self.S * s_scale)),
        )

    def transition(self, dt_days: float = 0.0,
                   lam: float = 0.0) -> tuple[tuple[float, float], tuple[float, float]]:
        """A[j][k] = P(k_t = k | k_{t-1} = j)。

        把两种遗忘并进同一个式子：

            P(k_t=1 | k_{t-1}=1) = (1 - F) · exp(-λΔt)

        · λ=0 时退化为 1-F，就是标准 BKT（不遗忘）。
        · F=0 时只剩时间衰减，纯「隔久了会忘」。
        · 两个都开也行：练一次也可能忘，再加上时间越久忘得越多。
        · T（学会）不受影响 —— 做题才学得会，时间流逝不会让人突然学会。
        """
        keep = 1.0
        if lam > 0.0 and dt_days > 0.0:
            keep = math.exp(-lam * dt_days)
        keep *= 1.0 - self.F
        keep = min(1.0 - _EPS, max(_EPS, keep))
        return ((1.0 - self.T, self.T), (1.0 - keep, keep))

    def clamped(self, lo: float, hi: float, forget: bool = True) -> "Params":
        def cut(v: float, low: float = lo) -> float:
            return min(hi, max(low, v))

        return Params(
            L0=cut(self.L0),
            S=cut(self.S),
            G=cut(self.G),
            T=cut(self.T),
            F=cut(self.F, 0.0) if forget else 0.0,
        )


# ---------------------------------------------------------------- E 步


def _scale_vectors(difficulty: TSequence[str] | None,
                   n: int) -> tuple[list[float], list[float]]:
    """把难度标签翻译成两条与题等长的向量：(S 系数, G 系数)。"""
    if not difficulty or len(difficulty) != n:
        return [1.0] * n, [1.0] * n
    s_scale: list[float] = []
    g_scale: list[float] = []
    for label in difficulty:
        s_k, g_k = DIFFICULTY_SCALE.get((label or "").strip().lower(), _DEFAULT_SCALE)
        s_scale.append(s_k)
        g_scale.append(g_k)
    return s_scale, g_scale


def _gap_vector(times: TSequence[float] | None, n: int) -> list[float]:
    """相邻两题的间隔（天）：gaps[t] 是从 t-1 到 t 的间隔，gaps[0] 恒为 0。"""
    if not times or len(times) != n:
        return [0.0] * n
    gaps = [0.0] * n
    for t in range(1, n):
        gaps[t] = max(0.0, times[t] - times[t - 1])
    return gaps


def _scan(y: TSequence[int], p: Params, lam: float = 0.0,
          difficulty: TSequence[str] | None = None,
          times: TSequence[float] | None = None) -> dict[str, float] | None:
    """单条序列的前向-后向，返回 M 步要用的充分统计量。

    返回 None 表示这条序列在当前参数下概率全为 0（参数跑到边界上去了），
    上层跳过它即可。

    与标准 BKT 的两处差别，都体现在这里的「按题取参数」：
      · 难度 → 每道题有自己的发射概率
      · 时间 → 每道题有自己的转移矩阵（间隔越久越容易忘）
    """
    n = len(y)
    if n == 0:
        return None

    s_scale, g_scale = _scale_vectors(difficulty, n)
    gaps = _gap_vector(times, n)
    # trans[t] 是「从 t-1 走到 t」的矩阵；trans[0] 用不到
    # 没有时间列时所有转移矩阵都相同，算一次就够 —— 老用法能省一大截开销
    if times and len(times) == n:
        trans = [p.transition(gaps[t], lam) for t in range(n)]
    else:
        trans = [p.transition(0.0, lam)] * n
    if difficulty and len(difficulty) == n:
        emis = [p.emission(difficulty[t]) for t in range(n)]
    else:
        emis = [p.emission(None)] * n

    def like(t: int, k: int) -> float:
        e = emis[t]
        return e[k] if y[t] else 1.0 - e[k]

    # ---- 前向（带缩放）
    alpha: list[list[float]] = [[0.0, 0.0] for _ in range(n)]
    c: list[float] = [1.0] * n
    prior = (1.0 - p.L0, p.L0)
    for k in (0, 1):
        alpha[0][k] = prior[k] * like(0, k)
    c[0] = alpha[0][0] + alpha[0][1]
    if c[0] <= _EPS:
        return None
    alpha[0][0] /= c[0]
    alpha[0][1] /= c[0]
    loglik = math.log(c[0])

    for t in range(1, n):
        prev = alpha[t - 1]
        A = trans[t]
        for k in (0, 1):
            alpha[t][k] = (prev[0] * A[0][k] + prev[1] * A[1][k]) * like(t, k)
        ct = alpha[t][0] + alpha[t][1]
        if ct <= _EPS:
            return None
        alpha[t][0] /= ct
        alpha[t][1] /= ct
        c[t] = ct
        loglik += math.log(ct)

    # ---- 后向（同样带缩放，除 c_{t+1}）
    beta: list[list[float]] = [[1.0, 1.0] for _ in range(n)]
    for t in range(n - 2, -1, -1):
        A = trans[t + 1]
        for j in (0, 1):
            beta[t][j] = (
                A[j][0] * like(t + 1, 0) * beta[t + 1][0]
                + A[j][1] * like(t + 1, 1) * beta[t + 1][1]
            ) / c[t + 1]

    # ---- 把 gamma / xi 累加成计数
    stats = {
        "ll": loglik,
        "n": float(n),
        "l0_num": 0.0,
        "t_num": 0.0, "t_den": 0.0,
        "f_num": 0.0, "f_den": 0.0,
        "s_num": 0.0, "s_den": 0.0,
        "g_num": 0.0, "g_den": 0.0,
    }

    for t in range(n):
        g0_raw = alpha[t][0] * beta[t][0]
        g1_raw = alpha[t][1] * beta[t][1]
        z = g0_raw + g1_raw
        if z <= _EPS:
            continue
        g0 = g0_raw / z
        g1 = g1_raw / z

        if t == 0:
            stats["l0_num"] = g1
        # 分母带上难度系数：估的是「基准题」上的 S/G，难度只是乘法偏移。
        # 系数全为 1 时这里就退化成标准写法 Σγ，不改变原有行为。
        stats["s_den"] += g1 * s_scale[t]
        stats["g_den"] += g0 * g_scale[t]
        if y[t]:
            stats["g_num"] += g0        # 不会却蒙对
        else:
            stats["s_num"] += g1        # 会了却答错

        if t < n - 1:
            # xi 与 gamma 用同一个 z 归一，保证 Σ_k xi(j,k) == gamma(j)
            A = trans[t + 1]
            xi01 = alpha[t][0] * A[0][1] * like(t + 1, 1) * beta[t + 1][1] / c[t + 1] / z
            xi10 = alpha[t][1] * A[1][0] * like(t + 1, 0) * beta[t + 1][0] / c[t + 1] / z
            stats["t_num"] += xi01      # 不会 → 会
            stats["t_den"] += g0
            stats["f_num"] += xi10      # 会 → 不会
            stats["f_den"] += g1

    return stats


def _accumulate(seqs: Iterable[Sequence], p: Params,
                lam: float = 0.0) -> dict[str, float] | None:
    """把一组的统计量加起来。``_nseq`` 是成功参与计算的序列条数 —— L0 的分母。"""
    total: dict[str, float] | None = None
    nseq = 0.0
    for s in seqs:
        one = _scan(s.y, p, lam, s.difficulty, s.time)
        if one is None:
            continue
        nseq += 1.0
        if total is None:
            total = dict(one)
        else:
            for key, val in one.items():
                total[key] += val
    if total is None:
        return None
    total["_nseq"] = nseq
    return total


# ---------------------------------------------------------------- M 步


def _m_step(stats: dict[str, float], prev: Params, forget: bool,
            bounds: tuple[float, float]) -> Params:
    """闭式解：每个参数就是「发生了的次数 / 有机会发生的次数」。"""

    def ratio(num: str, den: str, fallback: float) -> float:
        d = stats.get(den, 0.0)
        return stats.get(num, 0.0) / d if d > _EPS else fallback

    # L0 的分母是「序列条数」，不是题数：每条序列只在第一题贡献一次先验
    n_seq = stats.get("_nseq", 0.0)
    L0 = stats.get("l0_num", 0.0) / n_seq if n_seq > _EPS else prev.L0

    cand = Params(
        L0=L0,
        S=ratio("s_num", "s_den", prev.S),
        G=ratio("g_num", "g_den", prev.G),
        T=ratio("t_num", "t_den", prev.T),
        F=ratio("f_num", "f_den", prev.F) if forget else 0.0,
    )
    return cand.clamped(bounds[0], bounds[1], forget=forget)


def _random_params(rng: random.Random, forget: bool) -> Params:
    return Params(
        L0=rng.uniform(0.10, 0.60),
        S=rng.uniform(0.02, 0.25),
        G=rng.uniform(0.02, 0.30),
        T=rng.uniform(0.03, 0.35),
        F=rng.uniform(0.0, 0.08) if forget else 0.0,
    )


def _fit_group(seqs: list[Sequence], init: Params, *, forget: bool, tol: float,
               max_iter: int, n_init: int, rng: random.Random,
               bounds: tuple[float, float],
               lam: float = 0.0, guard: bool = False) -> tuple[Params, float, int]:
    """对一组序列跑 EM，多个起点取最优。返回 (参数, 对数似然, 迭代次数)。

    ``guard`` 是给难度/时间这类「非标准 BKT」准备的：
    有难度系数时，S/G 的 M 步闭式解只是一阶近似（因为 1-S·s_t 没法直接解析），
    每轮可能让似然掉 1e-5 量级。开启守护后，一旦更新让似然下降就回退并收敛，
    保证对外表现仍是「迭代过程似然单调不降」。
    标准 BKT（无难度、无时间）不需要它，所以默认关闭、零开销。
    """
    starts = [init]
    for _ in range(max(0, n_init - 1)):
        starts.append(_random_params(rng, forget))

    best: tuple[Params, float, int] | None = None
    for start in starts:
        p = start
        prev_ll = _NEG_INF
        used = 0
        pending: dict[str, float] | None = None   # 上一轮 guard 已经算过的 stats
        for it in range(1, max_iter + 1):
            used = it
            stats = pending if pending is not None else _accumulate(seqs, p, lam)
            pending = None
            if stats is None:
                break
            ll = stats["ll"]
            cand = _m_step(stats, p, forget, bounds)
            if guard:
                probe = _accumulate(seqs, cand, lam)
                if probe is None or probe["ll"] < ll - 1e-12:
                    break           # 不再上升，就地收敛（p 保持当前值）
                pending = probe     # 下一轮直接用，别重算 —— 否则开销翻倍
            p = cand
            if abs(ll - prev_ll) < tol:
                break
            prev_ll = ll
        # M 步之后参数变了，重新算一次真实的似然用于比较
        stats = _accumulate(seqs, p, lam)
        ll_final = stats["ll"] if stats is not None else _NEG_INF
        if best is None or ll_final > best[1]:
            best = (p, ll_final, used)

    assert best is not None
    return best


# ---------------------------------------------------------------- 预测


def _filter(y: TSequence[int], p: Params, lam: float = 0.0,
            difficulty: TSequence[str] | None = None,
            times: TSequence[float] | None = None) -> tuple[list[float], list[float]]:
    """逐题的**事前**预测概率 + 全部作答完后的掌握后验。

    注意这里是「事前」：预测第 t 题时只用到 y_1..y_{t-1}，
    不然拿答案算出来的预测等于作弊，评估出来的 AUC 会虚高。
    """
    n = len(y)
    gaps = _gap_vector(times, n)
    has_diff = bool(difficulty) and len(difficulty) == n
    post = [1.0 - p.L0, p.L0]
    out: list[float] = []
    for t in range(n):
        if t:
            A = p.transition(gaps[t], lam)
            post = [
                post[0] * A[0][0] + post[1] * A[1][0],
                post[0] * A[0][1] + post[1] * A[1][1],
            ]
        e = p.emission(difficulty[t] if has_diff else None)
        out.append(min(1.0 - _EPS, max(_EPS, post[0] * e[0] + post[1] * e[1])))

        yt = y[t]
        n0 = post[0] * (e[0] if yt else 1.0 - e[0])
        n1 = post[1] * (e[1] if yt else 1.0 - e[1])
        z = n0 + n1
        if z > _EPS:
            post = [n0 / z, n1 / z]
    return out, post


# ---------------------------------------------------------------- 评估


def auc(y_true: TSequence[int], scores: TSequence[float]) -> float | None:
    """纯 Python 的 AUC（按秩算，处理好平分的情况）。只有一个类别时返回 None。"""
    pairs = sorted(zip(scores, y_true))
    n = len(pairs)
    n_pos = sum(y_true)
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    rank_sum = 0.0
    i = 0
    while i < n:
        j = i
        while j + 1 < n and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            if pairs[k][1] == 1:
                rank_sum += avg_rank
        i = j + 1
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _clip(x: float) -> float:
    return min(1.0 - 1e-9, max(1e-9, x))


# ---------------------------------------------------------------- 主类


class BKTModel:
    """贝叶斯知识追踪。用法就三步：fit → predict → metrics。

    >>> model = BKTModel(model="per-skill").fit(sequences)
    >>> model.metrics(sequences)["auc"]
    0.78...

    参数
    ----
    model         "global"（全班一套） / "per-skill"（每个知识点一套） /
                  "per-user"（每个学生一套，样本少的自动退回 per-skill）
    forget        是否估遗忘率 F。默认 False。
    decay_lambda  时间衰减系数 λ（单位 1/天）。默认 0 表示不看时间。
                  开了之后 P(k_t=1|k_{t-1}=1) = (1-F)·exp(-λΔt)，
                  也就是「隔得越久越可能忘」。需要序列里带时间列。
    n_init        EM 随机起点个数，默认 3。设 1 就只从 initial 那组跑。
    min_obs       per-user 模式下，样本量低于此值的学生不单独建模（默认 30）。
    seed          随机种子，保证同一份数据跑两次结果一样。
    bounds        参数裁剪区间，默认 (1e-3, 1 - 1e-3)。不裁剪容易出现
                  S=0 或 G=1 这种「完美观测」的退化解。
    """

    _MODELS = ("global", "per-skill", "per-user")

    def __init__(self, model: str = "per-skill", *, forget: bool = False,
                 decay_lambda: float = 0.0,
                 initial: Params | dict[str, float] | None = None,
                 tol: float = 1e-6, max_iter: int = 200, n_init: int = 3,
                 min_obs: int = 30, seed: int = 42,
                 bounds: tuple[float, float] = (1e-3, 1.0 - 1e-3)) -> None:
        if model not in self._MODELS:
            raise ValueError(f"model 只能是 {self._MODELS}，收到的是 {model!r}")
        if decay_lambda < 0:
            raise ValueError("decay_lambda 不能为负")
        self.model = model
        self.forget = forget
        self.decay_lambda = float(decay_lambda)
        self.tol = tol
        self.max_iter = max_iter
        self.n_init = n_init
        self.min_obs = min_obs
        self.seed = seed
        self.bounds = bounds
        self.initial = self._as_params(initial)

        self._pooled: Params = self.initial
        self._per_skill: dict[str, Params] = {}
        self._per_user: dict[str, Params] = {}
        self.log_likelihood_: float | None = None
        self.n_params_: int = 0
        self.iterations_: int = 0
        self._fitted = False

    # -- 小工具 --------------------------------------------------
    def _as_params(self, initial: Params | dict[str, float] | None) -> Params:
        if initial is None:
            base = Params()
        elif isinstance(initial, Params):
            base = initial
        else:
            base = Params(**{k: float(v) for k, v in initial.items()})
        return base if self.forget else replace(base, F=0.0)

    def _params_for(self, seq: Sequence) -> Params:
        if self.model == "global":
            return self._pooled
        if self.model == "per-skill":
            return self._per_skill.get(seq.skill, self._pooled)
        own = self._per_user.get(seq.user)
        if own is not None:
            return own
        return self._per_skill.get(seq.skill, self._pooled)

    @staticmethod
    def _group(seqs: Iterable[Sequence], key) -> Iterator[tuple[str, list[Sequence]]]:
        buckets: dict[str, list[Sequence]] = {}
        for s in seqs:
            buckets.setdefault(key(s), []).append(s)
        for name, group in buckets.items():
            yield name, group

    # -- 拟合 ----------------------------------------------------
    def fit(self, sequences: TSequence[Sequence]) -> "BKTModel":
        seqs = list(sequences)
        if not seqs:
            raise ValueError("没有序列可拟合：检查 to_sequences 的列名识别是否正常")

        if self.decay_lambda > 0 and not any(s.time for s in seqs):
            warnings.warn(
                "开了 decay_lambda 但序列里没有时间列，时间衰减不会生效。"
                "用 to_sequences(time_col=...) 或 load_csv(time_col=...) 带上时间。",
                stacklevel=2,
            )

        rng = random.Random(self.seed)
        # 只有「难度」或「时间」在场时才需要似然守护，标准 BKT 走原来的快路径
        guard = any(bool(s.difficulty) or bool(s.time) for s in seqs)
        kw = dict(forget=self.forget, tol=self.tol, max_iter=self.max_iter,
                  n_init=self.n_init, rng=rng, bounds=self.bounds,
                  lam=self.decay_lambda, guard=guard)

        self._pooled, _ll, iters = _fit_group(seqs, self.initial, **kw)
        self._per_skill, self._per_user = {}, {}

        if self.model in ("per-skill", "per-user"):
            for skill, group in self._group(seqs, lambda s: s.skill):
                p, _ll_s, it_s = _fit_group(group, self._pooled, **kw)
                self._per_skill[skill] = p
                iters = max(iters, it_s)

        if self.model == "per-user":
            for user, group in self._group(seqs, lambda s: s.user):
                if sum(len(s) for s in group) < self.min_obs:
                    continue          # 样本不够就别单独建模，退回 per-skill
                p, _ll_u, it_u = _fit_group(group, self._pooled, **kw)
                self._per_user[user] = p
                iters = max(iters, it_u)

        self.log_likelihood_, self.n_params_ = self._score_fit(seqs)
        self.iterations_ = iters
        self._fitted = True
        return self

    def _score_fit(self, sequences: TSequence[Sequence]) -> tuple[float, int]:
        """按**实际用到**的参数组算总对数似然与参数个数。

        不能把「pooled 的似然 + 各知识点的似然」加起来 —— 那是两个不同模型的
        似然相加，加起来既不是 A 也不是 B。这里统一按每条序列最终落到哪组参数
        去重统计，AIC/BIC 才站得住。
        """
        total = 0.0
        used: dict[int, Params] = {}
        for s in sequences:
            p = self._params_for(s)
            stats = _scan(s.y, p, self.decay_lambda, s.difficulty, s.time)
            if stats is None:
                continue
            total += stats["ll"]
            used[id(p)] = p
        return total, len(used) * (5 if self.forget else 4)

    # -- 输出 ----------------------------------------------------
    def params(self) -> dict[str, dict[str, float]]:
        """拟合出来的参数表。global 模式只有一条 ``"global"``。"""
        self._check_fitted()
        if self.model == "global":
            return {"global": self._pooled.as_dict()}
        if self.model == "per-skill":
            return {k: v.as_dict() for k, v in sorted(self._per_skill.items())}
        table = {k: v.as_dict() for k, v in sorted(self._per_skill.items())}
        table.update({f"user:{k}": v.as_dict() for k, v in sorted(self._per_user.items())})
        return table

    def predict(self, sequences: TSequence[Sequence]) -> list[float]:
        """逐条作答的预测正确率，顺序与 sequences 拼接后的顺序一致。"""
        self._check_fitted()
        out: list[float] = []
        for s in sequences:
            probs, _ = _filter(s.y, self._params_for(s), self.decay_lambda, s.difficulty, s.time)
            out.extend(probs)
        return out

    def predict_rows(self, sequences: TSequence[Sequence]) -> dict[int, float]:
        """按「原表行号」对齐的预测结果 —— 直接能跟输入 CSV 拼起来看。"""
        self._check_fitted()
        out: dict[int, float] = {}
        for s in sequences:
            probs, _ = _filter(s.y, self._params_for(s), self.decay_lambda, s.difficulty, s.time)
            for row, prob in zip(s.rows, probs):
                out[row] = prob
        return out

    def mastery(self, sequences: TSequence[Sequence]) -> dict[str, float]:
        """做完所有题之后，对每个 (学生, 知识点) 的掌握概率估计。

        返回的 key 形如 ``"s01/K3"``。这就是「知识追踪」最终要的东西：
        不是「他这次答对没」，而是「他现在到底会不会」。
        """
        self._check_fitted()
        out: dict[str, float] = {}
        for s in sequences:
            _, post = _filter(s.y, self._params_for(s), self.decay_lambda, s.difficulty, s.time)
            out[f"{s.user}/{s.skill}"] = post[1]
        return out

    def metrics(self, sequences: TSequence[Sequence]) -> dict[str, Any]:
        """一套常用指标。AUC 看排序能力，logLoss 看概率校准。"""
        self._check_fitted()
        y_true: list[int] = []
        y_pred: list[float] = []
        for s in sequences:
            probs, _ = _filter(s.y, self._params_for(s), self.decay_lambda, s.difficulty, s.time)
            y_true.extend(s.y)
            y_pred.extend(probs)
        if not y_true:
            return {"n": 0, "auc": None, "accuracy": None, "logLoss": None, "rmse": None}
        n = len(y_true)
        correct = sum(1 for t, p in zip(y_true, y_pred) if (p >= 0.5) == bool(t))
        ll = -sum(math.log(_clip(p)) if t else math.log(_clip(1.0 - p))
                  for t, p in zip(y_true, y_pred)) / n
        rmse = math.sqrt(sum((p - t) ** 2 for t, p in zip(y_true, y_pred)) / n)
        return {
            "n": n,
            "auc": auc(y_true, y_pred),
            "accuracy": correct / n,
            "logLoss": ll,
            "rmse": rmse,
        }

    def aic(self, n_obs: int) -> float | None:
        """AIC，用来比「该用哪种参数化」。参数多一组，AIC 要多付 2 的代价。"""
        if self.log_likelihood_ is None or not n_obs:
            return None
        return 2 * self.n_params_ - 2 * self.log_likelihood_

    def bic(self, n_obs: int) -> float | None:
        if self.log_likelihood_ is None or not n_obs:
            return None
        return self.n_params_ * math.log(n_obs) - 2 * self.log_likelihood_

    def report(self, sequences: TSequence[Sequence]) -> dict[str, Any]:
        """拟合 + 评估一次性打包，命令行和 README 都用它。"""
        m = self.metrics(sequences)
        n_obs = m["n"]
        return {
            "model": self.model,
            "forget": self.forget,
            "decayLambda": self.decay_lambda,
            "groups": len(self.params()),
            "nParams": self.n_params_,
            "iterations": self.iterations_,
            "logLikelihood": self.log_likelihood_,
            "aic": self.aic(n_obs),
            "bic": self.bic(n_obs),
            **m,
        }

    # -- 其他 ----------------------------------------------------
    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("还没 fit。先调 .fit(sequences)。")

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        if not self._fitted:
            return f"<BKTModel {self.model} (未拟合)>"
        return (f"<BKTModel {self.model} groups={self.n_params_ // (5 if self.forget else 4)} "
                f"ll={self.log_likelihood_:.1f}>")
