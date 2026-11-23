"""bkt-em：贝叶斯知识追踪（BKT）的 EM 实现，零第三方依赖。

标准 BKT 四个参数：

    L0  初始掌握概率
    S   失误率，会了却答错
    G   猜测率，不会却蒙对
    T   学习转移率，练一次从不会变会

四个参数要么按文献拍，要么从数据估。估就需要 EM。现成的 pyBKT 在我们那套环境
（Python 3.14 / numpy 2.5 / pandas 3 / scipy 1.18 / sklearn 1.9）上跑不通，
于是用标准库重写了一份，支持 global / per-skill / per-user 三种参数化，
带按学生分组的 K 折交叉验证，命令行和当库调用都可以。

数学口径：

    发射  P(y=1|k=1) = 1-S      P(y=1|k=0) = G
    转移  P(k_t=1|k_{t-1}=1) = 1-F    P(k_t=1|k_{t-1}=0) = T
    初始  P(k_1=1) = L0

0.2.0 起有两层扩展：难度改发射概率，时间衰减改转移概率。细节见 README。
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
