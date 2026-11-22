"""最小可运行示例：把 README 里那段跑一遍，并演示「真值能不能捞回来」。

    python examples/quickstart.py

用的是 examples/answers_sample.csv（30 人 × 2 知识点 × 8 题，模拟数据）；
没有这个文件时会自动回退到内存里现场生成。
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bkt_em import BKTModel, Params, cross_validate, load_csv, summarize
from bkt_em.simulate import simulate

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLE = os.path.join(HERE, "answers_sample.csv")
TRUTH = Params(L0=0.30, S=0.10, G=0.22, T=0.20)


def main() -> int:
    if os.path.exists(SAMPLE):
        seqs = load_csv(SAMPLE)
        print(f"读入 {SAMPLE}")
    else:
        seqs = simulate(n_users=30, n_skills=2, n_items=8, params=TRUTH, seed=0)
        print("样例 CSV 不存在，改用内存里生成的模拟数据")

    info = summarize(seqs)
    print(f"{info['sequences']} 条序列 | {info['users']} 人 | "
          f"{info['skills']} 个知识点 | {info['observations']} 条作答 | "
          f"正确率 {info['accuracy']}")
    print()

    model = BKTModel(model="per-skill").fit(seqs)
    print("拟合出的参数：")
    for name, params in model.params().items():
        rounded = {k: round(v, 3) for k, v in params.items()}
        print(f"  {name}  {rounded}")

    report = model.report(seqs)
    print(f"\nAUC {report['auc']:.4f} | 准确率 {report['accuracy']:.4f} | "
          f"logLoss {report['logLoss']:.4f}")

    cv = cross_validate(seqs, k=5, model="per-skill")
    print(f"5 折交叉验证（按学生分折）AUC {cv['auc']:.4f} | logLoss {cv['logLoss']:.4f}")

    print("\n掌握度最低的三个（学生/知识点 → 掌握概率，这些才需要干预）：")
    table = sorted(model.mastery(seqs).items(), key=lambda kv: kv[1])[:3]
    for key, value in table:
        print(f"  {key}  {value:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
