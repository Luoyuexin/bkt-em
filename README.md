# bkt-em

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-3776ab?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-2ea44f)](LICENSE)
[![Dependencies](https://img.shields.io/badge/dependencies-0-brightgreen)](pyproject.toml)
[![Tests](https://img.shields.io/badge/tests-48%20passed-2ea44f)](tests)

**贝叶斯知识追踪（BKT）的 EM 实现，纯 Python 标准库，零第三方依赖。**

四个参数全从数据里估出来，不用拍值：

| 参数 | 含义 | 直觉 |
| --- | --- | --- |
| `L0` | 初始掌握概率 | 上课前他就会了？ |
| `S` | 失误率 slip | 会了却答错 |
| `G` | 猜测率 guess | 不会却蒙对 |
| `T` | 学习转移率 learn | 练一次就从不会变成会 |

算完这些，你就能回答一个问题：**这个学生现在到底会不会** —— 而不是「他这次答对没」。

---

## 相比 pyBKT 的优势

pyBKT 拖着 numpy / pandas / scipy / sklearn 四条链。这四个包任意一个
换了接口，pyBKT 就崩，而且崩的位置一次比一次深。实测在
Python 3.14 + numpy 2.5 + pandas 3.0 + scipy 1.18 + sklearn 1.9 上暴露了三处不兼容：

```
sklearn 1.9 把指标函数的报错类型从 TypeError 改成 AttributeError  → pyBKT 的兼容层失效
scipy 1.18 的 minimize 返回形状变了                            → EM_fit 里赋值报错
调用参数表跟不上新版本                                          → _check_args 静默拒绝
```

而 BKT 本身只有四个参数、隐状态只有 0/1 两种，EM 的 M 步有闭式解 —— **完全不需要
优化器**。所以这里重写了一个：只用 `math` / `random` / `csv` / `json`，
`python -c "import bkt_em"` 在任何装了 Python 3.9+ 的机器上都能过。

---

## 上手

### 方式一：直接把包拷进你的项目

`bkt_em/` 这个目录不依赖任何东西，拷过去就能 `import`。

### 方式二：安装

```bash
pip install -e .          # 本仓库
pip install bkt-em        # 等待版本相对稳定 发布到 PyPI 后
```

### 三十秒跑通

```python
from bkt_em import BKTModel, Params, summarize
from bkt_em.simulate import simulate

# 1. 造一份「已知真值」的数据，好检查能不能把真值捞回来
truth = Params(L0=0.30, S=0.08, G=0.20, T=0.25)
seqs = simulate(n_users=300, n_skills=1, n_items=12, params=truth, seed=7)
print(summarize(seqs))
# {'sequences': 300, 'users': 300, 'skills': 1, 'observations': 3600,
#  'accuracy': 0.7664, 'meanLength': 12.0, 'maxLength': 12}

# 2. 拟合
model = BKTModel(model="per-skill").fit(seqs)
print(model.params()["K1"])
# {'L0': 0.276, 'S': 0.082, 'G': 0.227, 'T': 0.27, 'F': 0.0}
# 真值 0.30 / 0.08 / 0.20 / 0.25 —— 四个参数都捞回来了

# 3. 评估
print(model.metrics(seqs))
# {'n': 3600, 'auc': 0.7809, 'accuracy': 0.7981, 'logLoss': 0.4359, 'rmse': 0.3725}

# 4. 拿结论
table = model.mastery(seqs)
print(table["s252/K1"])   # 0.044 —— 12 题几乎全错，基本没学会
print(table["s001/K1"])   # 1.0   —— 12 题几乎全对，基本确定会了
```

换成自己的数据只改一行：

```python
from bkt_em import load_csv
seqs = load_csv("answers.csv")            # 列名自动识别，见下文「数据格式」
```

不想写代码就先跑一遍仓库里的示例：`python examples/quickstart.py`

---

## 数据格式

一行一条作答的长表，四列就够（列名不挑，常见写法都能认）：

| 语义 | 认得的列名 |
| --- | --- |
| 学生 | `user_id` `user` `student_id` `student` `uid` `learner_id` |
| 知识点 | `skill_name` `skill` `kc` `knowledge` `knowledge_id` `kcs` |
| 对错 | `correct` `is_correct` `y` `label` `score` `outcome` |
| 顺序 | `order_id` `order` `step` `attempt` `seq` `timestamp` `time` |

```csv
user_id,skill_name,correct,order_id
s01,K1,1,1
s01,K1,0,2
s01,K2,1,1
```

对错列认 `1/0`、`true/false`、`t/f`、`对/错`；认不出来的行会被丢掉（不猜）。

**顺序很重要。** BKT 的 `T` 是「练一次之后会不会」，把题序打乱了估出来的 T 就没有意义。
给了顺序列就按它排，没给就按原表顺序。预测结果会带回原表行号，能直接拼回原始 CSV。

---

## 三种参数化怎么选

| `model` | 一套参数给谁 | 参数个数 | 什么时候用 |
| --- | --- | --- | --- |
| `global` | 全班共用 | 4 | 样本少、只想看整体难度 · 兜底 |
| `per-skill` | 每个知识点一套 | 4 × 知识点数 | **默认**。知识点难度不同，这个差异真实存在 |
| `per-user` | 每个学生一套 | 4 × 达标学生数 | 想做个性化，且每个学生有 ≥30 条作答 |

`per-user` 里样本不够的学生会自动退回 `per-skill`，不会用一个学生的三条规定拟合四个参数。

---

## 交叉验证：为什么不能只看似然

参数越多，训练集上的对数似然一定越高 —— 这是数学事实，不是模型变好了。
所以 `compare_models` 同时给两个视角：

```bash
python -m bkt_em fit --simulate --users 80 --skills 3 --items 10 --cv 5 --compare
```

```
三种参数化对比（cvAuc 才是真本事，AIC/BIC 只看训练集）：
model      nParams      logLik        AIC   cvAuc  cvLogLoss
---------  -------  ----------  ---------  ------  ---------
global           4  -1285.4768  2578.9535  0.7579     0.5384
per-skill       12  -1283.7893  2591.5785  0.7503     0.5418
per-user       320  -1115.6253  2871.2507  0.7503     0.5418
```

怎么读这张表：

- `per-user` 的训练似然 `-1115` 明显好于 `per-skill` 的 `-1283` —— 它把每个人的噪声都背下来了
- 但参数从 12 个涨到 320 个，AIC 罚完之后 `2871` 反而最差
- 交叉验证 `cvAuc` 一点没涨 —— 对**没见过的学生**，个性化参数没有帮助
- 结论：这份数据该用 `global` 或 `per-skill`，`per-user` 在这里是过拟合

（交叉验证按**学生**分折，同一学生的所有序列只会落在一折里。否则等于拿他前几题训练、
后几题测试，指标必然虚高 —— 这是教育数据挖掘里最常见的作弊方式。）

---

## 命令行

```bash
# 用仓库自带的样例数据跑一遍（30 人 × 2 知识点 × 8 题）
python -m bkt_em fit examples/answers_sample.csv --cv 5

# 拟合 + 交叉验证，参数和掌握度落盘
python -m bkt_em fit answers.csv --model per-skill --cv 5 \
    --out params.json --mastery mastery.csv --predict pred.csv

# 没有数据？先造一份来验证流程
python -m bkt_em simulate --users 60 --skills 3 --items 10 --out sim.csv
python -m bkt_em fit sim.csv --cv 5 --compare
```

`fit examples/answers_sample.csv --cv 5` 的输出：

```
序列 60 条 | 学生 30 人 | 知识点 2 个 | 作答 480 条 | 整体正确率 0.6792
平均序列长度 8.0，最长 8

group      L0       S       G       T
-----  ------  ------  ------  ------
K1     0.3627  0.1084  0.2450  0.2100
K2     0.1934  0.0890  0.2688  0.2354

对数似然 -254.4 | 参数 8 个 | AIC 524.8 | BIC 558.1 | EM 迭代 26
训练集 AUC 0.7534 | 准确率 0.7208 | logLoss 0.5300

5 折交叉验证（按学生分折，预测的都是没见过的学生）：AUC 0.7203 | 准确率 0.6917 | logLoss 0.5537
```

（这份样例数据是按 `L0=0.30, S=0.10, G=0.22, T=0.20` 生成的 —— 30 人偏少，
`K1` 的 `L0` 估到 0.36 就属于小样本的正常抖动。样本到了几千条会稳得多。）

`--mastery` 输出的 CSV 就是知识追踪的最终产物：

```csv
user,skill,mastery
s01,K1,0.044584
s01,K2,0.911378
```

---

## 数学口径

隐状态 $k_t \in \{0,1\}$ 表示「第 $t$ 题作答时是否已掌握」，观测 $y_t \in \{0,1\}$ 是答对与否。

$$
P(y=1 \mid k=1) = 1-S,\qquad P(y=1 \mid k=0) = G
$$
$$
P(k_t=1 \mid k_{t-1}=1) = 1-F,\qquad P(k_t=1 \mid k_{t-1}=0) = T,\qquad P(k_1=1) = L_0
$$

`F` 是遗忘率，**默认 0**（标准 BKT 不支持遗忘），加 `--forget` 才估。

EM 的 M 步全是「发生了的次数 / 有机会发生的次数」，闭式解不用梯度：

| 参数 | 估计式 |
| --- | --- |
| `L0` | $\dfrac{\sum_{\text{序列}} \gamma_1(1)}{\#\text{序列}}$ |
| `T` | $\dfrac{\sum_t \xi_t(0\to1)}{\sum_t \gamma_t(0)}$ |
| `F` | $\dfrac{\sum_t \xi_t(1\to0)}{\sum_t \gamma_t(1)}$ |
| `S` | $\dfrac{\sum_t \gamma_t(1)(1-y_t)}{\sum_t \gamma_t(1)}$ |
| `G` | $\dfrac{\sum_t \gamma_t(0)\,y_t}{\sum_t \gamma_t(0)}$ |

前向概率用**缩放因子** $c_t$ 逐步归一，$\sum_t \log c_t$ 就是对数似然 —— 序列再长也不会
浮点下溢。参数裁到 $[10^{-3}, 1-10^{-3}]$，否则容易出现 `S=0` / `G=1` 这种「完美观测」
的退化解。

似然面可能有多个局部极大，所以默认从 3 个随机起点各跑一遍取最优（`n_init` 控制，
`seed` 固定后结果可复现）。

---

## 预测是「事前」的

`predict()` 预测第 $t$ 题时只用到 $y_1 \dots y_{t-1}$，绝不用第 $t$ 题的答案。
拿答案算出来的预测等于作弊，AUC 会虚高十几个点 —— 这个坑踩过的人不少。

想拿「学完之后他到底会不会」，用 `mastery()`，那是把全部作答都用上后的后验。

---

## 已知局限

- **知识点之间不建模先修关系。** 每个知识点单独估一套（除非用 `global`）。要做 DAG 上的
  推断得另写一层。
- **不含题目难度/区分度。** 那是 IRT 的活。题意难在 BKT 里会混进 `S` 里（难题答错看着像失误）。
- **不含作答时间间隔。** 真实遗忘跟「隔了多久」强相关，这里只有开关式的 `F`。
- **EM 只能保证局部最优。** `n_init` 调大能缓解，代价是慢。
- **参数可辨识性有边界。** 数据里几乎没人答错时 `S` 估不准，几乎没人答对时 `G` 估不准 ——
  这不是 bug，是信息不够。

---

## API 速查

```python
from bkt_em import BKTModel, Params, load_csv, to_sequences, summarize, \
                   cross_validate, compare_models
from bkt_em.simulate import simulate

BKTModel(model="per-skill", forget=False, n_init=3, min_obs=30, seed=42,
         tol=1e-6, max_iter=200, bounds=(1e-3, 1-1e-3))

.fit(sequences) -> self
.params()       -> {"K1": {"L0":..., "S":..., "G":..., "T":..., "F":...}}
.predict(seqs)  -> [0.62, 0.71, ...]         # 逐题事前预测，顺序同输入
.predict_rows(seqs) -> {原表行号: 概率}
.mastery(seqs)  -> {"s01/K1": 0.87, ...}     # 作答完后的掌握后验
.metrics(seqs)  -> {"auc", "accuracy", "logLoss", "rmse", "n"}
.report(seqs)   -> 上面这些 + AIC/BIC/参数个数/迭代次数
.aic(n_obs) / .bic(n_obs)

cross_validate(seqs, k=5, model="per-skill")     # 按学生分折
compare_models(seqs, k=5)                        # 三种参数化放一起比
```

---

## 仓库结构

```
bkt_em/
  data.py       长表 → 序列（列名自动识别、行号对齐）
  em.py         BKTModel：EM 拟合 / 预测 / 掌握度 / 指标 / AIC
  cv.py         按学生分组的 K 折交叉验证 + 三种参数化对比
  simulate.py   按已知参数生成作答序列（自测、教学、压测都用它）
  cli.py        命令行
examples/
  quickstart.py          README 里那段的可运行版本
  answers_sample.csv     30 人 × 2 知识点 × 8 题样例数据
tests/                   48 项 unittest
```

---

## 测试

48 项，全部标准库 `unittest`，不需要 pytest：

```bash
python -m unittest discover -s tests -v
```

最硬的一条是 `test_loglikelihood_matches_enumeration`：序列长度 3 时隐状态只有 8 条
可能路径，把 8 条路径的概率直接加起来，必须等于前向算法算出来的似然。两者不相等，
说明下标或者缩放写错了。另外还有「EM 迭代过程中似然必须单调不降」这条，用来抓 M 步闭式解
写错的情况 —— 这两个测试比任何对照实验都能更快定位问题。

---

## 许可

MIT。见 [LICENSE](LICENSE)。

## 参考文献

Corbett, A. T., & Anderson, J. R. (1995). Knowledge tracing: Modeling the acquisition of
procedural knowledge. *User Modeling and User-Adapted Interaction*, 4(4), 253–278.

Beck, J. E., & Chang, K. (2007). Identifiability: A fundamental problem of student
modeling. *UM 2007*.（解决「数据不够时参数估不准」）
