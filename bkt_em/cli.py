"""命令行入口。

    python -m bkt_em fit answers.csv --model per-skill --cv 5
    python -m bkt_em simulate --users 40 --skills 3 --items 8 --out sim.csv

不装包也能跑：把仓库根目录加进 sys.path 就行（python -m bkt_em 需要包已在路径上）。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from typing import Any, Sequence as TSequence

from .cv import compare_models, cross_validate
from .data import Sequence, load_csv, summarize
from .em import BKTModel, Params
from .simulate import simulate


def _truth_params(args: argparse.Namespace) -> Params | None:
    """把 --l0/--s/--g/--t/--f 拼成一组真值；一个都没给就返回 None（用默认值）。"""
    names = {"l0": "L0", "s": "S", "g": "G", "t": "T", "f": "F"}
    given = {field: getattr(args, key) for key, field in names.items()
             if getattr(args, key, None) is not None}
    return Params(**given) if given else None


def _fmt(value: Any, digits: int = 4, width: int = 0) -> str:
    if value is None:
        text = "—"
    elif isinstance(value, float):
        text = f"{value:.{digits}f}"
    else:
        text = str(value)
    return text.rjust(width) if width else text


def _table(headers: TSequence[str], rows: TSequence[TSequence[Any]],
           digits: int = 4) -> str:
    """固定宽度文本表。第一列左对齐（名字），其余右对齐（数字）。"""
    cells = [[_fmt(v, digits) for v in row] for row in rows]
    widths = [max(len(h), *(len(r[i]) for r in cells)) if cells else len(h)
              for i, h in enumerate(headers)]

    def line_of(values: TSequence[str]) -> str:
        return "  ".join(v.ljust(w) if i == 0 else v.rjust(w)
                         for i, (v, w) in enumerate(zip(values, widths)))

    out = [line_of(headers), "  ".join("-" * w for w in widths)]
    out.extend(line_of(row) for row in cells)
    return "\n".join(out)


def _print_summary(seqs: TSequence[Sequence]) -> None:
    info = summarize(seqs)
    print(f"序列 {info['sequences']} 条 | 学生 {info['users']} 人 | "
          f"知识点 {info['skills']} 个 | 作答 {info['observations']} 条 | "
          f"整体正确率 {info['accuracy']}")
    print(f"平均序列长度 {info['meanLength']}，最长 {info['maxLength']}")
    if info["observations"] < 200:
        print("⚠ 样本偏少，估四个参数容易飘；建议每个知识点至少 500 条作答。")


def _cmd_fit(args: argparse.Namespace) -> int:
    if args.simulate:
        truth = _truth_params(args)
        seqs = simulate(n_users=args.users, n_skills=args.skills, n_items=args.items,
                        params=truth, seed=args.seed)
        print(f"[模拟数据] {args.users} 人 × {args.skills} 知识点 × {args.items} 题")
        if truth is not None:
            print(f"[真值] {truth.as_dict()}")
    else:
        if not args.input:
            print("要么给一个 CSV 路径，要么加 --simulate 用模拟数据。", file=sys.stderr)
            return 2
        seqs = load_csv(args.input, user_col=args.user_col, skill_col=args.skill_col,
                        correct_col=args.correct_col, order_col=args.order_col,
                        min_length=args.min_length)

    _print_summary(seqs)
    print()

    model = BKTModel(model=args.model, forget=args.forget, n_init=args.n_init,
                     min_obs=args.min_obs, seed=args.seed).fit(seqs)

    head = ["group", "L0", "S", "G", "T"] + (["F"] if args.forget else [])
    body = []
    for name, params in model.params().items():
        body.append([name] + [params[k] for k in head[1:]])
    print(_table(head, body))

    print()
    rep = model.report(seqs)
    print(f"对数似然 {rep['logLikelihood']:.1f} | 参数 {rep['nParams']} 个 | "
          f"AIC {rep['aic']:.1f} | BIC {rep['bic']:.1f} | EM 迭代 {rep['iterations']}")
    print(f"训练集 AUC {_fmt(rep['auc'])} | 准确率 {_fmt(rep['accuracy'])} | "
          f"logLoss {_fmt(rep['logLoss'])}")

    if args.cv:
        print()
        cv = cross_validate(seqs, k=args.cv, model=args.model, forget=args.forget,
                            n_init=args.n_init, min_obs=args.min_obs, seed=args.seed)
        print(f"{cv['k']} 折交叉验证（按学生分折，预测的都是没见过的学生）："
              f"AUC {_fmt(cv['auc'])} | 准确率 {_fmt(cv['accuracy'])} | "
              f"logLoss {_fmt(cv['logLoss'])}")

    if args.compare:
        print()
        print("三种参数化对比（cvAuc 才是真本事，AIC/BIC 只看训练集）：")
        cmp = compare_models(seqs, k=args.cv or 3, n_init=args.n_init,
                             min_obs=args.min_obs, seed=args.seed)
        print(_table(
            ["model", "nParams", "logLik", "AIC", "cvAuc", "cvLogLoss"],
            [[name, row["nParams"], row["logLikelihood"], row["aic"],
              row["cvAuc"], row["cvLogLoss"]] for name, row in cmp.items()],
        ))

    if args.out:
        payload = {
            "model": model.model,
            "forget": model.forget,
            "params": model.params(),
            "report": rep,
            "summary": summarize(seqs),
        }
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        print(f"\n参数已写入 {args.out}")

    if args.mastery:
        table = model.mastery(seqs)
        with open(args.mastery, "w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["user", "skill", "mastery"])
            for key, value in sorted(table.items()):
                user, skill = key.rsplit("/", 1)
                writer.writerow([user, skill, f"{value:.6f}"])
        print(f"掌握度已写入 {args.mastery}（{len(table)} 行）")

    if args.predict:
        # 用序列自己记的行号回填，不靠猜列名 —— 换过列名的表也能对齐
        index = {row: (s.user, s.skill, y)
                 for s in seqs for row, y in zip(s.rows, s.y)}
        preds = model.predict_rows(seqs)
        with open(args.predict, "w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["row", "user", "skill", "correct", "predicted"])
            for row in sorted(index):
                user, skill, y = index[row]
                writer.writerow([row, user, skill, y, f"{preds[row]:.6f}"])
        print(f"逐题预测已写入 {args.predict}")

    return 0


def _cmd_simulate(args: argparse.Namespace) -> int:
    truth = _truth_params(args)
    seqs = simulate(n_users=args.users, n_skills=args.skills, n_items=args.items,
                    params=truth, seed=args.seed, ability_sd=args.ability_sd)
    rows = []
    for s in seqs:
        for order, y in enumerate(s.y):
            rows.append([s.user, s.skill, y, order])
    path = args.out or "simulated.csv"
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["user_id", "skill_name", "correct", "order_id"])
        writer.writerows(rows)
    print(f"已生成 {path}：{len(rows)} 行，{len(seqs)} 条序列")
    print(f"再用它反推参数：python -m bkt_em fit {path} --model per-skill")
    return 0


def _add_truth_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("模拟数据的真值（不填就用 0.25 / 0.10 / 0.25 / 0.15）")
    group.add_argument("--l0", type=float, help="初始掌握概率")
    group.add_argument("--s", type=float, help="失误率 slip")
    group.add_argument("--g", type=float, help="猜测率 guess")
    group.add_argument("--t", type=float, help="学习转移率 learn")
    group.add_argument("--f", type=float, help="遗忘率 forget（默认 0）")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bkt-em",
        description="贝叶斯知识追踪（BKT）的 EM 实现，零第三方依赖。",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    fit = sub.add_parser("fit", help="拟合参数并评估")
    fit.add_argument("input", nargs="?", help="作答记录 CSV（长表）")
    fit.add_argument("--model", default="per-skill",
                     choices=["global", "per-skill", "per-user"],
                     help="参数化方式，默认 per-skill")
    fit.add_argument("--forget", action="store_true", help="额外估遗忘率 F")
    fit.add_argument("--cv", type=int, default=0, help="做 K 折交叉验证（按学生分折）")
    fit.add_argument("--compare", action="store_true", help="对比三种参数化")
    fit.add_argument("--simulate", action="store_true", help="不用 CSV，直接生成模拟数据")
    fit.add_argument("--users", type=int, default=60)
    fit.add_argument("--skills", type=int, default=3)
    fit.add_argument("--items", type=int, default=10)
    fit.add_argument("--min-obs", dest="min_obs", type=int, default=30,
                     help="per-user 模式下单独建模的最小样本量")
    fit.add_argument("--min-length", dest="min_length", type=int, default=1,
                     help="丢弃短于此时长的序列")
    fit.add_argument("--n-init", dest="n_init", type=int, default=3,
                     help="EM 随机起点个数")
    fit.add_argument("--seed", type=int, default=42)
    fit.add_argument("--user-col", dest="user_col")
    fit.add_argument("--skill-col", dest="skill_col")
    fit.add_argument("--correct-col", dest="correct_col")
    fit.add_argument("--order-col", dest="order_col")
    fit.add_argument("--out", help="参数写入 JSON")
    fit.add_argument("--mastery", help="每个学生每个知识点的掌握度写入 CSV")
    fit.add_argument("--predict", help="逐题预测概率写入 CSV")
    _add_truth_args(fit)
    fit.set_defaults(func=_cmd_fit)

    sim = sub.add_parser("simulate", help="生成模拟作答数据")
    sim.add_argument("--users", type=int, default=60)
    sim.add_argument("--skills", type=int, default=3)
    sim.add_argument("--items", type=int, default=10)
    sim.add_argument("--ability-sd", dest="ability_sd", type=float, default=0.0,
                     help="学生之间初始水平的差异")
    sim.add_argument("--seed", type=int, default=0)
    sim.add_argument("--out", help="输出 CSV 路径")
    _add_truth_args(sim)
    sim.set_defaults(func=_cmd_simulate)

    return parser


def main(argv: TSequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
