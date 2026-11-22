"""交叉验证：估出来的参数到底靠不靠谱。

BKT 有个特别容易骗自己的地方 —— 参数越多，训练集上的对数似然一定越高。
「每个学生一套参数」在训练集上永远比「每个知识点一套」好看，但那是因为
它把每个人的噪声也背下来了，换个学生就废。

所以这里按 **学生** 分组切折（同一学生的所有序列只会落在一折里，
不然等于拿他前几题训练、后几题测试，成绩必然虚高），
再在没见过的学生身上比 AUC / logLoss。
"""

from __future__ import annotations

import math
import random
from typing import Any, Sequence as TSequence

from .data import Sequence
from .em import BKTModel, Params, _clip, auc


def _score(y_true: TSequence[int], y_pred: TSequence[float]) -> dict[str, Any]:
    n = len(y_true)
    if not n:
        return {"n": 0, "auc": None, "accuracy": None, "logLoss": None, "rmse": None}
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


def _split_users(seqs: TSequence[Sequence], k: int, seed: int) -> list[list[str]]:
    users = sorted({s.user for s in seqs})
    random.Random(seed).shuffle(users)
    k = max(2, min(k, len(users)))
    return [users[i::k] for i in range(k)]


def cross_validate(sequences: TSequence[Sequence], k: int = 5, *,
                   model: str = "per-skill", seed: int = 42,
                   initial: Params | dict[str, float] | None = None,
                   **fit_kwargs: Any) -> dict[str, Any]:
    """按学生分组的 K 折交叉验证。

    返回的指标是把各折的预测拼在一起算的（不是各折平均）——
    拼接后再算，指标才对样本量加权，短折不会被高估。

    >>> report = cross_validate(sequences, k=5, model="per-skill")
    >>> report["auc"]
    0.71...

    其余参数原样透传给 :class:`bkt_em.em.BKTModel`（forget / n_init / seed 之外的那些）。
    """
    seqs = list(sequences)
    if len({s.user for s in seqs}) < 2:
        raise ValueError("交叉验证至少需要 2 个学生的数据")

    folds = _split_users(seqs, k, seed)
    all_true: list[int] = []
    all_pred: list[float] = []
    per_fold: list[dict[str, Any]] = []

    for i, held_out in enumerate(folds, start=1):
        hold = set(held_out)
        train = [s for s in seqs if s.user not in hold]
        test = [s for s in seqs if s.user in hold]
        if not train or not test:
            continue
        fitted = BKTModel(model=model, seed=seed + i, initial=initial, **fit_kwargs).fit(train)
        pred = fitted.predict(test)
        y_true = [v for s in test for v in s.y]
        fold = _score(y_true, pred)
        fold["fold"] = i
        fold["users"] = len(held_out)
        per_fold.append(fold)
        all_true.extend(y_true)
        all_pred.extend(pred)

    out: dict[str, Any] = {"k": len(per_fold), "model": model, "perFold": per_fold}
    out.update(_score(all_true, all_pred))
    return out


def compare_models(sequences: TSequence[Sequence], k: int = 5, *,
                   models: TSequence[str] = ("global", "per-skill", "per-user"),
                   seed: int = 42, **fit_kwargs: Any) -> dict[str, dict[str, Any]]:
    """把几种参数化放在一起比，顺带给出「参数多了到底值不值」。

    返回 ``{model: 报告}``，每份报告里含 ``aic``/``bic``（训练集拟合优度，
    参数越多越低）和 ``auc``/``logLoss``（交叉验证，才是真本事）。
    两者结论打架的时候，信交叉验证。
    """
    seqs = list(sequences)
    n_obs = sum(len(s) for s in seqs)
    out: dict[str, dict[str, Any]] = {}
    for name in models:
        fitted = BKTModel(model=name, seed=seed, **fit_kwargs).fit(seqs)
        cv = cross_validate(seqs, k=k, model=name, seed=seed, **fit_kwargs)
        out[name] = {
            "nParams": fitted.n_params_,
            "logLikelihood": fitted.log_likelihood_,
            "aic": fitted.aic(n_obs),
            "bic": fitted.bic(n_obs),
            "cvAuc": cv["auc"],
            "cvLogLoss": cv["logLoss"],
            "cvAccuracy": cv["accuracy"],
            "folds": cv["k"],
        }
    return out
