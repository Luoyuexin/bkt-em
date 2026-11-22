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
from dataclasses import dataclass, replace
from typing import Any, Iterable, Iterator, Sequence as TSequence

from .data import Sequence

_EPS = 1e-12
_NEG_INF = float("-inf")


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

    def emission(self) -> tuple[float, float]:
        """(P(答对|未掌握), P(答对|已掌握)) = (G, 1-S)"""
        return (self.G, 1.0 - self.S)

    def transition(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """A[j][k] = P(k_t = k | k_{t-1} = j)"""
        return ((1.0 - self.T, self.T), (self.F, 1.0 - self.F))

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


def _scan(y: TSequence[int], p: Params) -> dict[str, float] | None:
    """单条序列的前向-后向，返回 M 步要用的充分统计量。

    返回 None 表示这条序列在当前参数下概率全为 0（参数跑到边界上去了），
    上层跳过它即可。
    """
    n = len(y)
    if n == 0:
        return None

    A = p.transition()
    e = p.emission()

    def like(t: int, k: int) -> float:
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
        stats["s_den"] += g1
        stats["g_den"] += g0
        if y[t]:
            stats["g_num"] += g0        # 不会却蒙对
        else:
            stats["s_num"] += g1        # 会了却答错

        if t < n - 1:
            # xi 与 gamma 用同一个 z 归一，保证 Σ_k xi(j,k) == gamma(j)
            xi01 = alpha[t][0] * A[0][1] * like(t + 1, 1) * beta[t + 1][1] / c[t + 1] / z
            xi10 = alpha[t][1] * A[1][0] * like(t + 1, 0) * beta[t + 1][0] / c[t + 1] / z
            stats["t_num"] += xi01      # 不会 → 会
            stats["t_den"] += g0
            stats["f_num"] += xi10      # 会 → 不会
            stats["f_den"] += g1

    return stats


def _accumulate(seqs: Iterable[Sequence], p: Params) -> dict[str, float] | None:
    """把一组的统计量加起来。``_nseq`` 是成功参与计算的序列条数 —— L0 的分母。"""
    total: dict[str, float] | None = None
    nseq = 0.0
    for s in seqs:
        one = _scan(s.y, p)
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
               bounds: tuple[float, float]) -> tuple[Params, float, int]:
    """对一组序列跑 EM，多个起点取最优。返回 (参数, 对数似然, 迭代次数)。"""
    starts = [init]
    for _ in range(max(0, n_init - 1)):
        starts.append(_random_params(rng, forget))

    best: tuple[Params, float, int] | None = None
    for start in starts:
        p = start
        prev_ll = _NEG_INF
        used = 0
        for it in range(1, max_iter + 1):
            used = it
            stats = _accumulate(seqs, p)
            if stats is None:
                break
            ll = stats["ll"]
            p = _m_step(stats, p, forget, bounds)
            if abs(ll - prev_ll) < tol:
                break
            prev_ll = ll
        # M 步之后参数变了，重新算一次真实的似然用于比较
        stats = _accumulate(seqs, p)
        ll_final = stats["ll"] if stats is not None else _NEG_INF
        if best is None or ll_final > best[1]:
            best = (p, ll_final, used)

    assert best is not None
    return best


# ---------------------------------------------------------------- 预测


def _filter(y: TSequence[int], p: Params) -> tuple[list[float], list[float]]:
    """逐题的**事前**预测概率 + 全部作答完后的掌握后验。

    注意这里是「事前」：预测第 t 题时只用到 y_1..y_{t-1}，
    不然拿答案算出来的预测等于作弊，评估出来的 AUC 会虚高。
    """
    A = p.transition()
    e = p.emission()
    post = [1.0 - p.L0, p.L0]
    out: list[float] = []
    for t in range(len(y)):
        if t:
            post = [
                post[0] * A[0][0] + post[1] * A[1][0],
                post[0] * A[0][1] + post[1] * A[1][1],
            ]
        out.append(post[0] * e[0] + post[1] * e[1])

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
    model     "global"（全班一套） / "per-skill"（每个知识点一套） /
              "per-user"（每个学生一套，样本少的自动退回 per-skill）
    forget    是否估遗忘率 F。默认 False。
    n_init    EM 随机起点个数，默认 3。设 1 就只从 initial 那组跑。
    min_obs   per-user 模式下，样本量低于此值的学生不单独建模（默认 30）。
    seed      随机种子，保证同一份数据跑两次结果一样。
    bounds    参数裁剪区间，默认 (1e-3, 1 - 1e-3)。不裁剪容易出现
              S=0 或 G=1 这种「完美观测」的退化解。
    """

    _MODELS = ("global", "per-skill", "per-user")

    def __init__(self, model: str = "per-skill", *, forget: bool = False,
                 initial: Params | dict[str, float] | None = None,
                 tol: float = 1e-6, max_iter: int = 200, n_init: int = 3,
                 min_obs: int = 30, seed: int = 42,
                 bounds: tuple[float, float] = (1e-3, 1.0 - 1e-3)) -> None:
        if model not in self._MODELS:
            raise ValueError(f"model 只能是 {self._MODELS}，收到的是 {model!r}")
        self.model = model
        self.forget = forget
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

        rng = random.Random(self.seed)
        kw = dict(forget=self.forget, tol=self.tol, max_iter=self.max_iter,
                  n_init=self.n_init, rng=rng, bounds=self.bounds)

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
            stats = _scan(s.y, p)
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
            probs, _ = _filter(s.y, self._params_for(s))
            out.extend(probs)
        return out

    def predict_rows(self, sequences: TSequence[Sequence]) -> dict[int, float]:
        """按「原表行号」对齐的预测结果 —— 直接能跟输入 CSV 拼起来看。"""
        self._check_fitted()
        out: dict[int, float] = {}
        for s in sequences:
            probs, _ = _filter(s.y, self._params_for(s))
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
            _, post = _filter(s.y, self._params_for(s))
            out[f"{s.user}/{s.skill}"] = post[1]
        return out

    def metrics(self, sequences: TSequence[Sequence]) -> dict[str, Any]:
        """一套常用指标。AUC 看排序能力，logLoss 看概率校准。"""
        self._check_fitted()
        y_true: list[int] = []
        y_pred: list[float] = []
        for s in sequences:
            probs, _ = _filter(s.y, self._params_for(s))
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
