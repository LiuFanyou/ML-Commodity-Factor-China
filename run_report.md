# 实验报告（运行结果摘要）

**生成时间：** 2026-07-22

## 1. 研究背景与动机 (Introduction)

- **商品期货的独特性：** 商品期货除了存在显著的横截面差异外，还深受宏观与产业链因素影响，品种间相关性常随宏观环境波动而改变。
- **机器学习的引入：** 传统的 CTA 或单因子排序在复杂非线性关系面前容易失效。本文采用 1D-CNN 来捕捉多因子的时序和横截面非线性交互，以提高短窗口（20 日）内的预测稳定性。

## 2. 数据处理与合约映射 (Data & Preprocessing)

- **主力合约拼接：** 使用基于滞后活跃度（上日成交量与持仓量构成的 activity）判断主力合约，避免使用未来数据，从而有效防止前视偏差（look-ahead bias）。
- **换月逻辑与展期收益处理：** 优先使用同日近远月价格计算展期收益（roll yield）；当缺失时回退到合约字段中的直接 roll 字段、现货/基差或 settle/close 代理，保证在多种数据降级场景下都能获得展期特征。
- **数据预热（Burn-in）：** 对每个品种预留 90 天用于滚动统计（如动量、波动率等），确保模型训练与第一条样本均使用完整的滚动窗口。

## 3. 多维度因子工程 (Feature Engineering)

- **选取的核心因子（5 个）：**
  - 动量（20 日）：$\text{momentum}_{20}=\sum_{i=0}^{19}\log(1+R_{t-i})$。
  - 展期收益率（roll yield）：使用近远月价格按年化对数差进行计算，若近月价格 $P_n$、远月价格 $P_f$，且间隔 $D$ 天，则
  $$\text{roll} = \log\frac{P_n}{P_f}\times\frac{365}{D}.$$ 
  - 波动率（20 日）：年化样本标准差 $\sigma_{20}=\sqrt{252}\,\mathrm{std}(R_{t-19:t})$。
  - 流动性：$\text{liquidity}=\log\left(1+\frac{\text{volume}}{\text{open\_interest}}\right)$。
  - 偏度（20 日）：20 日滚动样本偏度。

- **去极值与标准化：** 先用绝对中位差（MAD）方法对异常值截断（scale 因子为 $1.4826\times\text{MAD}$），随后对横截面做 Z-Score 标准化，保证不同因子在同一量纲下输入网络。

## 4. 深度学习模型架构 (Model Architecture)

- **输入与输出：** 输入张量维度为 $[B,5,20]$（Batch，5 因子，20 日序列），输出为单值的下一持有期截面超额收益预测。
- **网络结构要点：** 多层 1D 卷积块（每块为两次 Conv1d + GroupNorm + GELU + MaxPool），全局平均池化后进入两层全连线回归头。
- **防过拟合：** 使用 Dropout 与 Huber Loss（对尖峰噪声更鲁棒），训练中采用早停（patience=7）与训练内时间验证来选择 epoch。

### 训练/验证损失曲线

![Training / Validation Loss](futures_ml_outputs/loss_curve.png)

*图：`futures_ml_outputs/loss_curve.png`，显示每个 Walk-Forward 折中的训练与验证 Loss。*

## 5. 滚动验证与回测体系 (Experimental Setup)

- **Walk-Forward 设置：** 每个测试年采用前 3 年数据训练、次年样本外测试（3 年训练 + 1 年测试），并在训练内使用时间序列验证（保留最近 20 天为验证），确保时间上严格前后分离。
- **截面多空构建：** 每个样本日对全品种按预测得分排序，做多预测前 20%（权重合计 0.5），做空预测后 20%（权重合计 -0.5），合约层合并权重并按合约换手计费。

## 6. 实验结果与绩效分析 (Empirical Results)

- 关键样本外绩效指标（摘自 `futures_ml_outputs/backtest_metrics.csv`）：

| 场景 | 单边费率 | 年化收益 | 夏普比率 | 最大回撤 | 胜率 |
|---:|---:|---:|---:|---:|---:|
| Gross (不计交易费) | 0.0 | -0.2627% | 0.00565 | -20.39% | 49.31% |
| 1 bp one-way | 0.0001 | -1.9955% | -0.2178 | -22.52% | 48.68% |
| 3 bp one-way | 0.0003 | -5.3717% | -0.6644 | -32.61% | 47.57% |
| 5 bp one-way | 0.0005 | -8.6321% | -1.1106 | -44.48% | 46.12% |

> 注：负的年化收益与低夏普提示在当前参数/样本下策略并未实现正向收益，但结果有助于诊断信号稳定性与交易摩擦敏感性。

### 累计净值曲线

![Cumulative Net Value](futures_ml_outputs/cumulative_return.png)

*图：`futures_ml_outputs/cumulative_return.png`，显示样本外净值与回撤走势。*

### 因子重要性（置换法）

![Feature Importance](futures_ml_outputs/feature_importance.png)

*图：`futures_ml_outputs/feature_importance.png`，展示置换重要性归因，便于判断模型依赖的主导因子。*

## 7. 摩擦成本敏感性与可行性 (Limitations & Feasibility)

- **费率敏感性图（1/3/5 bp）：**

![Fee Sensitivity](futures_ml_outputs/fee_sensitivity.png)

*图：`futures_ml_outputs/fee_sensitivity.png`，展示随着单边费用上升，净值的退化情况。*

- **实盘落地挑战：** 当前回测采用固定 bp 成本，但没有完整建模涨跌停无法成交、冲击成本、保证金约束与限仓等实盘要素，导致回测偏乐观或保守均有可能。

- **建议的下一步改进（简要）：**
  - 引入基于盘口深度的冲击成本模型或微观价差回归；
  - 在回测中加入保证金/持仓限制逻辑并测算实际资金占用；
  - 使用市场状态识别（例如 HMM）做动态仓位调整；
  - 考虑产业链或宏观因子残差，提升跨品种信息利用效率。

## 附录

- 关键输出目录：`futures_ml_outputs/`
  - `loss_curve.png`, `cumulative_return.png`, `fee_sensitivity.png`, `feature_importance.png`
  - `backtest_metrics.csv`, `daily_backtest.csv`, `oos_predictions.csv`, `training_history.csv`


