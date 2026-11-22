"""bkt-em —— 贝叶斯知识追踪（BKT）的 EM 实现，零第三方依赖。

为什么会有这个包
----------------
标准 BKT 一共四个参数：

    L0  初始掌握概率      priori knowledge
    S   失误率            slip      —— 会了却答错
    G   猜测率            guess     —— 不会却蒙对
    T   学习转移率        learn     —— 练一次就从不会变成会

这四个参数要么拍文献常用值，要么从数据里估。拍值容易被问「凭什么」，
估就需要 EM。现成的 pyBKT 在我们那套环境（Python 3.14 / numpy 2.5 /
pandas 3 / scipy 1.18 / sklearn 1.9）上跑不通，于是重写了一个：

    · 纯标准库，零第三方依赖（连 numpy 都不用，装完就能跑）
    · 支持 global / per-skill / per-user 三种参数化
    · 支持按 user 分组的 K 折交叉验证
    · 命令行可用，也能当库调用

数学口径
--------
隐状态 k_t ∈ {0,1} 表示「第 t 题作答时是否已掌握」，观测 y_t ∈ {0,1} 是答对与否。

    发射：P(y=1|k=1) = 1-S      P(y=1|k=0) = G
    转移：P(k_t=1|k_{t-1}=1) = 1-F    P(k_t=1|k_{t-1}=0) = T
    初始：P(k_1=1) = L0

F 是遗忘率，默认 0（标准 BKT 不支持遗忘，这里留了开关）。
参数用标准前向-后向 + 缩放因子估计，M 步是闭式解，不需要梯度。

两条扩展（0.2.0 起）
--------------------
难度：不改变模型结构，只改发射概率
    S_eff = S × s_scale        G_eff = G × g_scale
三档系数见 DIFFICULTY_SCALE。难题「更难蒙对、更容易失误」，基础题反过来。
难度是**已知输入**（题目自带的标签），不从数据反推 —— 要估难度请用 IRT。

时间：把「练完就忘」和「隔久了会忘」并进同一个式子
    P(k_t=1 | k_{t-1}=1) = (1 - F) · exp(-λΔt)
λ=0 退化为标准 BKT；F=0 就只剩时间衰减；两个可以同时开。
"""

from .cv import compare_models, cross_validate
from .data import Sequence, load_csv, summarize, to_sequences
from .em import BKTModel, DIFFICULTY_SCALE, Params

__version__ = "0.2.0"
__all__ = [
    "BKTModel",
    "Params",
    "DIFFICULTY_SCALE",
    "Sequence",
    "to_sequences",
    "load_csv",
    "summarize",
    "cross_validate",
    "compare_models",
    "__version__",
]
