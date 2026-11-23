# bkt-em

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-3776ab?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-2ea44f)](LICENSE)
[![Dependencies](https://img.shields.io/badge/dependencies-0-brightgreen)](pyproject.toml)
[![Tests](https://img.shields.io/badge/tests-70%20passed-2ea44f)](tests)

贝叶斯知识追踪（BKT）的 EM 实现，纯 Python 标准库，零第三方依赖。

四个参数从数据里估，不用手调：

| 参数 | 含义 |
| --- | --- |
| `L0` | 初始掌握概率 |
| `S` | 失误率，会了却答错 |
| `G` | 猜测率，不会却蒙对 |
| `T` | 学习转移率，练一次从不会变会 |

算完之后能回答「这个学生现在会不会」，而不是「他这次答对没」。

## 相比 pyBKT 的优势

pyBKT 依赖 numpy / pandas / scipy / sklearn，这几个包换接口时它就容易崩。BKT 只有四个参数、隐状态只有 0/1 两种，EM 的 M 步有闭式解，不需要优化器。所以这里只用 `math` / `random` / `csv` / `json` 重写了一遍，装了 Python 3.9+ 就能跑。

## 安装

```bash
pip install -e .          # 本仓库
pip install bkt-em        # 版本相对稳定后发布到 PyPI
```

或者直接把 `bkt_em/` 目录拷进你自己的项目，没有依赖。

## 用法

```python
from bkt_em import BKTModel, Params, load_csv, summarize
from bkt_em.simulate import simulate

# 用已知真值造数据，验证能不能估回来
truth = Params(L0=0.30, S=0.08, G=0.20, T=0.25)
seqs = simulate(n_users=300, n_skills=1, n_items=12, params=truth, seed=7)

model = BKTModel(model="per-skill").fit(seqs)
print(model.params()["K1"])
# {'L0': 0.276, 'S': 0.082, 'G': 0.227, 'T': 0.27, 'F': 0.0}

print(model.metrics(seqs))
# {'n': 3600, 'auc': 0.7809, 'accuracy': 0.7981, 'logLoss': 0.4359, 'rmse': 0.3725}

print(model.mastery(seqs)["s252/K1"])   # 0.044，12 题几乎全错
```

换成自己的数据：

```python
seqs = load_csv("answers.csv")
```

完整示例见 `examples/quickstart.py`。

`predict()` 预测第 t 题时只用到前 t-1 题的答案，不会偷看当前题。想要「学完之后他到底会不会」，用 `mastery()`。

## 数据格式

一行一条作答。列名不挑，常见写法都认：

| 语义 | 列名 |
| --- | --- |
| 学生 | `user_id` `user` `student_id` `uid` `learner_id` |
| 知识点 | `skill_name` `skill` `kc` `knowledge_id` |
| 对错 | `correct` `is_correct` `y` `label` `score` |
| 顺序 | `order_id` `order` `step` `attempt` `timestamp` |
| 难度（可选） | `difficulty` `diff` `level` `难度` |
| 时间（可选） | `answered_at` `created_at` `datetime` `答题时间` |

```csv
user_id,skill_name,correct,order_id,difficulty,answered_at
s01,K1,1,1,hard,0
s01,K1,0,2,hard,3.5
```

对错列认 `1/0`、`true/false`、`对/错`，认不出来的行会丢掉。`timestamp` 算顺序列，时间衰减只认「时间」那一行。

题序不能乱。`T` 是「练一次之后会不会」，打乱顺序估出来的 T 没有意义。预测结果会带回原表行号，能拼回原始 CSV。

## 参数化

| `model` | 一套参数给谁 | 什么时候用 |
| --- | --- | --- |
| `global` | 全班共用 | 样本少，或者只想看整体 |
| `per-skill` | 每个知识点一套 | 默认 |
| `per-user` | 每个学生一套 | 每个学生有 30 条以上作答 |

`per-user` 里样本不够的学生会退回 `per-skill`。

## 难度与时间

标准 BKT 默认所有题一样难、两道题之间没有时间流逝。0.2.0 起这两件事都能进模型，数据里带了就生效，没带就完全不影响结果。

**难度**：给一列难度标签，按标签调整这道题的发射概率。

| 标签 | `S` 系数 | `G` 系数 |
| --- | --- | --- |
| `basic` / `easy` / `简单` | 0.90 | 1.20 |
| `medium` / `normal` / `中等` | 1.05 | 0.95 |
| `hard` / `困难` / `难` | 1.25 | 0.65 |

认不出的标签不调整。难度是已知输入，不从数据反推。要估题目难度用 IRT。

在有难度的数据上（300 人 × 12 题），AUC 0.7943，忽略难度是 0.7855。

**时间**：给一列时间，相邻两题的间隔进转移概率。

$$P(k_t=1 \mid k_{t-1}=1) = (1-F) \cdot e^{-\lambda \Delta t}$$

`decay_lambda` 单位是 1/天，默认 0 表示不看时间。和 `forget=True` 同时开时两个遗忘叠加。`T` 不受时间影响。

实测（250 人 × 10 题，间隔约 4 天）：用对 λ 时对数似然 -1625.2，估出的参数 0.663 / 0.110 / 0.194 / 0.162（真值 0.60 / 0.10 / 0.20 / 0.15）。当它不存在时似然 -1698.3，`T` 被压到 0.001，参数全歪。

小样本下 `L0` 和 `G` 会互相补偿，40 人 × 9 题的规模估不准，加到几百人就好。

## 交叉验证

参数越多，训练集似然一定越高。所以按学生分折做交叉验证：

```bash
python -m bkt_em fit --simulate --users 80 --skills 3 --items 10 --cv 5 --compare
```

```
model      nParams      logLik        AIC   cvAuc  cvLogLoss
---------  -------  ----------  ---------  ------  ---------
global           4  -1285.4768  2578.9535  0.7579     0.5384
per-skill       12  -1283.7893  2591.5785  0.7503     0.5418
per-user       320  -1115.6253  2871.2507  0.7503     0.5418
```

`per-user` 训练似然最好（-1115），但参数涨到 320 个，AIC 罚完最差，交叉验证也没提升。这份数据该用 `global` 或 `per-skill`。

同一学生的所有序列只落在一折里，否则等于拿他前几题训练、后几题测试。

## 命令行

```bash
python -m bkt_em fit examples/answers_sample.csv --cv 5
python -m bkt_em fit answers.csv --cv 5 --out params.json --mastery mastery.csv
python -m bkt_em fit answers.csv --decay 0.05 --time-unit ms

# 没有数据就先造一份
python -m bkt_em simulate --users 300 --skills 2 --items 12 \
    --difficulty-cycle basic,medium,hard --gap-days 3 --decay 0.05 --out sim.csv
```

样例数据跑出来是这样（按 `L0=0.30, S=0.10, G=0.22, T=0.20` 生成，30 人偏少，`L0` 估到 0.36 是正常抖动）：

```
group      L0       S       G       T
-----  ------  ------  ------  ------
K1     0.3627  0.1084  0.2450  0.2100
K2     0.1934  0.0890  0.2688  0.2354

训练集 AUC 0.7534 | 5 折交叉验证 AUC 0.7203
```

`--mastery` 输出每个学生每个知识点的掌握度：

```csv
user,skill,mastery
s01,K1,0.044584
s01,K2,0.911378
```

## 限制

- 知识点之间不建先修关系，每个知识点单独估。要做 DAG 上的推断得另写一层。
- 难度是已知标签，不从数据估。
- EM 只保证局部最优，`n_init` 调大能缓解。带难度或时间时 M 步是一阶近似，加了似然守护，更新后似然不升就退回。
- 参数可辨识性有边界。几乎没人答错时 `S` 估不准，几乎没人答对时 `G` 估不准。

## 数学

隐状态 $k_t \in \{0,1\}$ 表示第 t 题作答时是否已掌握，观测 $y_t \in \{0,1\}$：

$$
P(y=1 \mid k=1) = 1-S,\qquad P(y=1 \mid k=0) = G
$$
$$
P(k_t=1 \mid k_{t-1}=1) = 1-F,\qquad P(k_t=1 \mid k_{t-1}=0) = T,\qquad P(k_1=1) = L_0
$$

前向-后向带缩放因子，$\sum_t \log c_t$ 是对数似然，序列再长也不下溢。参数裁到 $[10^{-3}, 1-10^{-3}]$，避免 `S=0`、`G=1` 这种退化解。默认 3 个随机起点取最优，`seed` 固定后结果可复现。

## 测试

```bash
python -m unittest discover -s tests -v
```

70 项，不需要 pytest。其中 `test_loglikelihood_matches_enumeration` 用穷举路径校验前向算法，`test_em_likelihood_is_monotone` 抓 M 步写错。

## 许可

MIT，见 [LICENSE](LICENSE)。

## 参考文献

Corbett, A. T., & Anderson, J. R. (1995). Knowledge tracing: Modeling the acquisition of procedural knowledge. *User Modeling and User-Adapted Interaction*, 4(4), 253–278.

Beck, J. E., & Chang, K. (2007). Identifiability: A fundamental problem of student modeling. *UM 2007*.（论证「数据不够时参数估不准」）
