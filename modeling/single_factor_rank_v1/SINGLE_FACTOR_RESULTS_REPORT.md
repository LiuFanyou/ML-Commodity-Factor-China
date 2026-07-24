# 单因子排序模型结果报告

**模型版本**：`single_factor_rank_v1`  
**数据接口**：`training_handoff`（与 Ridge 线性回归同一套交接包）  
**报告日期**：2026-07-23  
**样本外区间**：2019–2024（扩展窗口）；2025 未参与开发评价  

---

## 1. 报告目的与结论摘要

本报告汇总单因子横截面排序模型在统一实验协议下的样本外结果。模型不对多因子做加权或回归，而是**逐字段**检验：若仅用该字段排序并按协议交易，预测力与组合表现如何。

### 核心结论

1. **评测覆盖完整**：注册表全部 **92** 个字段均完成单因子回测；其中经济含义解读以 **36 个 predictor** 为主。
2. **预测力弱正、分布分化**：predictor 中约 **61%**（22/36）样本外日均 RankIC > 0；RankIC 中位数约 **0.0065**。
3. **成本是主要约束**：在协议主成本（单边 **5bp**）下，predictor 中仅 **1** 个字段净 Sharpe > 0（`warehouse_tightness__csz`）；净 Sharpe 中位数约 **−0.58**。
4. **不宜用全量 RankIC 直接“选冠军”**：`quality_flag` / `__missing` 可因覆盖模式推高 RankIC，但不构成可投资的经济因子信号。
5. **正文短表建议**：在 predictor 宇宙内，用 **RankIC Top10** 作主表，用 **5bp 净 Sharpe Top10** 作成本对照；二者交集达 **9** 个字段，叙事一致性较好。

---

## 2. 实验口径（统一协议）

| 项目 | 设定 |
|---|---|
| 数据来源 | `training_handoff`：特征 / 标签 / 注册表 / `fold_definitions.csv` |
| 品种与面板 | 中国商品期货 30 品种；`trade_date × product` |
| 开发样本 | 2015-01-05 — 2024-12-31 |
| 主标签 | `future_return_5d`（同一真实合约开盘→开盘） |
| 分数变换 | 方向对齐后做同日横截面中心化排名 `cs_rank_centered` ∈ [−1, 1] |
| 方向 | 因子目录 `expected_direction`；若为 0，仅用该折训练窗 RankIC 符号冻结 |
| 时间切分 | 扩展窗口：`fold_2019` … `fold_2024` |
| 净化 | `label_end_date_5d < valid_start` |
| 交易执行 | T 收盘形成信号，T+1 开盘成交；持有/调仓 **5** 日 |
| 组合规则 | 预测最高/最低各 20%；多空各 0.5；净敞口 0、总敞口 1 |
| 成本 | 主口径单边 **5bp**；敏感性 0 / 3 / 5 / 10 bp |
| 年化 | 非重叠 5 日收益，`annualization = 50.4`（≈252/5） |

本模型**不负责**因子加工；更换 handoff 矩阵后可直接重跑。

---

## 3. 评测对象说明

### 3.1 字段构成（92）

| usage_type | 数量 | 在本报告中的角色 |
|---|---|---|
| predictor | 36 | **主解读对象**（经济预测信号） |
| quality_flag | 45 | 多为 `__missing`；全量审计保留，不进经济短表 |
| risk | 6 | 风险类；可观察，不作主叙事 |
| conditional | 5 | 条件变量；可观察，不作主叙事 |

与 Ridge 一致：训练/评测输入覆盖注册表全部字段；与 Ridge 不同：单因子模型是**逐字段单独排序**，不是 92 维联合回归。

### 3.2 展示分层原则

| 层级 | 内容 | 用途 |
|---|---|---|
| 全量表 | 92 字段各自指标 | 审计、可复现 |
| 经济短表 | predictor Top10（多指标） | 正文/汇报 |
| 图表 | RankIC 分布、成本热力图等 | 直观对照 |

**已关闭**自动 TopK 作为“唯一冠军输出”；短表用于分析，不等于删掉其余因子。

---

## 4. 总体样本外表现

### 4.1 Predictor 宇宙（36）

| 指标 | 结果 |
|---|---|
| RankIC > 0 占比 | **61.1%**（22/36） |
| RankIC 中位数 | **0.0065** |
| 毛 Sharpe > 0 占比 | **55.6%** |
| 毛 Sharpe 中位数 | **0.022** |
| 5bp 净 Sharpe > 0 占比 | **2.8%**（1/36） |
| 5bp 净 Sharpe 中位数 | **−0.577** |

解读：单因子层面存在弱正的横截面预测力，但在 **5 日调仓 + 单边 5bp** 的统一执行假设下，多数单因子多空组合难以覆盖成本。这与 Ridge 多因子联合预测在成本后转负的现象方向一致，说明问题更多来自**执行与成本口径**，而非“完全没有预测信息”。

### 4.2 按 usage_type 的 RankIC 中位数（全量 92）

| usage_type | RankIC 中位数 |
|---|---|
| risk | 0.0120 |
| predictor | 0.0065 |
| conditional | −0.0035 |
| quality_flag | −0.0064 |

注意：个别 `quality_flag` 字段可在全量榜上 RankIC 极高（例如部分 `__missing`），但其经济可解释性与可交易性通常较弱，故**不以全量榜第一名作为论文主结论**。

---

## 5. 对照短表结果

短表宇宙：`usage_type=predictor` 且排除 `__missing`。详细见 `outputs/tables/SHORTLISTS.md`。

### 5.1 A. 按 RankIC Top10（主短表）

| # | 字段 | RankIC | 胜率 | 毛 Sharpe | 5bp 净 Sharpe | 正 IC 年数 |
|---|---|---|---|---|---|---|
| 1 | `carry_annualized__csz` | 0.0291 | 0.547 | 0.279 | −0.272 | 5/6 |
| 2 | `basis_fresh__csz` | 0.0285 | 0.557 | 0.463 | −0.168 | 5/6 |
| 3 | `warehouse_tightness__csz` | 0.0230 | 0.540 | 0.831 | **0.053** | 5/6 |
| 4 | `information_divergence__csz` | 0.0227 | 0.539 | 0.284 | −0.387 | 5/6 |
| 5 | `inventory_price_divergence__csz` | 0.0214 | 0.546 | 0.313 | −0.551 | 5/6 |
| 6 | `term_structure_slope__csz` | 0.0204 | 0.546 | 0.107 | −0.388 | 4/6 |
| 7 | `nonlinear_tightness__csz` | 0.0197 | 0.524 | 0.539 | −0.150 | 4/6 |
| 8 | `tightness_composite__csz` | 0.0194 | 0.525 | 0.523 | −0.166 | 4/6 |
| 9 | `information_consistency__csz` | 0.0190 | 0.537 | 0.322 | −0.367 | 4/6 |
| 10 | `short_reversal_5__csz` | 0.0182 | 0.526 | 0.438 | −0.033 | 5/6 |

主题分布上，前列较多落在 **期限结构/基差（carry、basis）**、**库存/拥挤度（tightness）** 与 **信息类**，与课题“动量、期限结构、流动性”主线可对应讨论；经典趋势动量（如 `tsmom_60`）在本轮 5 日标签口径下并未进入 RankIC Top10。

### 5.2 B. 按 5bp 净 Sharpe Top10（成本对照）

| # | 字段 | RankIC | 毛 Sharpe | 5bp 净 Sharpe |
|---|---|---|---|---|
| 1 | `warehouse_tightness__csz` | 0.0230 | 0.831 | **0.053** |
| 2 | `short_reversal_5__csz` | 0.0182 | 0.438 | −0.033 |
| 3 | `nonlinear_tightness__csz` | 0.0197 | 0.539 | −0.150 |
| 4 | `tightness_composite__csz` | 0.0194 | 0.523 | −0.166 |
| 5 | `basis_fresh__csz` | 0.0285 | 0.463 | −0.168 |
| 6 | `sector_relative_value_60__csz` | 0.0109 | 0.317 | −0.220 |
| 7 | `carry_annualized__csz` | 0.0291 | 0.279 | −0.272 |
| 8 | `information_consistency__csz` | 0.0190 | 0.322 | −0.367 |
| 9 | `information_divergence__csz` | 0.0227 | 0.284 | −0.387 |
| 10 | `term_structure_slope__csz` | 0.0204 | 0.107 | −0.388 |

### 5.3 C. 稳定性短表（可选）

在 RankIC > 0 的 predictor 中，按「正 IC 年数 → 胜率 → RankIC」排序。前列多为 **5/6 年 RankIC 为正** 的字段（如 `basis_fresh`、`carry_annualized`、`warehouse_tightness`），与 A 表高度重叠，说明头部预测力并非单年偶然。

### 5.4 交集与错位（A vs B）

| 类型 | 字段 |
|---|---|
| **A∩B（9）** | `basis_fresh`，`carry_annualized`，`information_consistency`，`information_divergence`，`nonlinear_tightness`，`short_reversal_5`，`term_structure_slope`，`tightness_composite`，`warehouse_tightness` |
| 只在 A | `inventory_price_divergence__csz`（IC 靠前，成本后更弱） |
| 只在 B | `sector_relative_value_60__csz`（成本后更好，IC 并非最高） |

**三表均进入**的字段：`basis_fresh__csz`，`carry_annualized__csz`，`warehouse_tightness__csz`，`information_divergence__csz`，`short_reversal_5__csz`。若需在汇报中点名“优先讨论集合”，建议从该交集出发，而不是只报全量榜 Top1。

---

## 6. 与 Ridge 基准的对照（同协议）

| 项目 | 单因子排序 | Ridge（同 handoff） |
|---|---|---|
| 输入字段 | 92（逐字段单独评） | 92（联合回归） |
| 标签 / 切分 / 交易 | 相同 | 相同 |
| 典型样本外 RankIC | 单因子中位数约 0.007；头部约 0.02–0.03 | 组合预测平均 RankIC 约 **0.025** |
| 5bp 成本后 | 绝大多数单因子净 Sharpe < 0 | 组合净 Sharpe 亦为负 |

含义：

- Ridge 的平均 RankIC 接近单因子头部水平，说明**多因子线性组合能整合分散的弱信号**，但在本成本假设下仍难稳定盈利。
- 单因子报告的价值在于：**指出哪些经济主题自身有预测力**，并为后续等权多因子 / ML 模型提供可解释的参照，而不是宣称单因子即可实盘。

---

## 7. 图表与数据索引

### 7.1 图（`outputs/figures/`）

| 文件 | 内容 |
|---|---|
| `fig01_rankic_by_usage_boxplot.png` | 各 usage_type 的 RankIC 分布 |
| `fig02_predictor_rankic_bars.png` | 36 个 predictor 的 RankIC 条形图 |
| `fig03_rankic_vs_gross_sharpe.png` | RankIC–毛 Sharpe 散点 |
| `fig04_predictor_cost_sensitivity_heatmap.png` | RankIC 前列因子的成本敏感性 |
| `fig05_usage_type_counts.png` | 字段类型计数 |

### 7.2 表（`outputs/tables/`）

| 文件 | 内容 |
|---|---|
| `01`–`05_*.csv` | 全量/分组有序表 |
| `06_shortlist_top10_by_rank_ic.csv` | 短表 A |
| `07_shortlist_top10_by_net5bps_sharpe.csv` | 短表 B |
| `08_shortlist_top10_by_stability.csv` | 短表 C |
| `09_shortlist_overlap.csv` | 短表交集 |
| `SHORTLISTS.md` | 短表说明 |

### 7.3 其他

- 逐字段宽表：`outputs/factor_strategy_metrics.csv`、`outputs/leaderboard.csv`
- 分年明细：`outputs/yearly_factor_metrics.csv`、`outputs/fold_metrics.csv`
- 自动明细报告：`single_factor_rank_report.md`
- 重生图表：`py modeling/single_factor_rank_v1/make_visuals.py`

---

## 8. 使用边界与局限

1. **点时**：信号在 T 收盘后形成，仅可用于 T+1 开盘执行；切分与方向选择未使用测试段。
2. **成本**：0/3/5/10bp 为统一敏感性情景，**不是**已核验的交易所真实手续费与冲击。
3. **成交约束**：尚未系统纳入涨跌停、成交容量、合约乘数等实盘约束。
4. **缺失指示器**：可出现在全量 92 榜单，但不应单独作为“最佳因子”叙事。
5. **相关冗余**：协议要求高相关字段保留并提供聚类元数据；短表未做“每簇只留一个代表”，若论文需要消融相关，可再按 `correlation_cluster` 压缩。
6. **2025**：本 handoff 不含 2025，不得把本报告结果表述为封存样本外最终结论。

---

## 9. 对后续工作的建议

1. **正文表述**：主表用 RankIC Top10（predictor）；附表给 5bp 净 Sharpe 对照与 A∩B 名单。
2. **模型比较**：将本单因子结果作为基准，与等权多因子、Ridge/Lasso、树模型在同一协议下并表。
3. **成本诊断**：对 A∩B 字段单独报告换手与 0/3/5/10bp 曲线，区分“无预测力”与“有预测力但成本过重”。
4. **主题归纳**：可将头部字段归入期限结构/基差、库存拥挤、短端反转等主题，服务论文结构，而不是只列字段名。

---

## 10. 一句话总结

在统一协议下，单因子排序显示中国商品期货截面上存在**可检出的弱预测信号**（尤其期限结构、基差与拥挤度相关 predictor），但在 **5 日调仓 + 单边 5bp** 假设下，**绝大多数单因子策略无法在成本后稳定盈利**；完整 92 字段结果用于审计，经济讨论应聚焦 predictor 短表及其成本对照。
