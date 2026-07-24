# 中国商品期货 Alpha 因子科学筛选系统：完整实验报告

> 报告性质：因子研究与机器学习输入接口实验，不是最终预测模型提交，也不是实盘收益承诺。

## 0. 提交流程与差异声明

### 0.1 硬性差异声明

本次实验相对通用团队模板存在以下显式差异：

1. **数据基座**：只读取 `data_final_raw`。
2. **主力规则修订**：按用户最终确认，不使用“T−1可见持仓量重新选择主力”；使用 TuShare T 日连续主力行情，并确定性映射至真实合约。
3. **期限结构**：Carry、斜率和曲率使用主力及其后最近可交易合约/完整实际合约曲线，不能只用主力一个合约；单一主力无法定义期限结构。
4. **样本划分**：采用 2015 起点固定的 Expanding-Window；2019—2024逐年开发期样本外评价，2025为一次封存测试，不采用单次静态切分。
5. **标签**：T收盘形成因子，T+1开盘进入当时映射的固定真实合约，T+h+1开盘退出；主标签为5日Open-to-Open，不是Close-to-Close。
6. **成本**：筛选主口径为单边5bps，同时报告0/3/5/10bps；未用不完整的逐品种真实手续费替代统一成本。
7. **组合映射**：无3日EMA、无绝对信号阈值、无流动性底池剔除；固定为横截面前后20%、两侧各0.5总权重、5日非重叠调仓。
8. **特征保留**：相关性高的字段只聚类和标注，不按全样本标签删除；所有47个通过数据质量硬门的预注册因子进入ML接口。
9. **标准化**：多数品种特征使用同日横截面中位数/MAD稳健Z分数；没有统一采用普通均值/标准差Z-Score，也没有全局板块中性化。
10. **基本面时点**：缺少日内发布时间时保守延迟一个交易日；现货陈旧超过7日、仓单超过5日、库存超过14日即作缺失。
11. **净值口径**：筛选器只生成非重叠5日收益；随附“每日净值”按退出日确认该期收益，是会计阶梯净值，不是假装成逐日盯市净值。
12. **封存边界修正**：本报告生成前发现原开发输出有2个调仓日的标签结束于2025；已修正为标签结束日必须位于评价窗口内并从头重算，开发期跨2025记录现为0。

### 0.2 提交物

- `factor_screening_experiment_report.md`：本报告。
- `factor_backtest_metrics.csv`：每个可计算因子在开发期与2025、3/5bps下的标准化指标。
- `factor_portfolio_period_returns.csv`：每个因子的5日非重叠组合收益及信号/入场/退出日期。
- `factor_nav_accounting_daily.csv.gz`：按退出日确认收益的每日会计净值。
- `ml_feature_registry.csv`：92个最终模型输入字段的唯一清单。
- `feature_dictionary.md`：特征含义与变换说明。
- `fold_definitions.csv`：扩展窗口定义。
- `SHA256SUMS` 与 `submission_manifest.json`：文件完整性和数据血缘。

## 1. 基础信息

- 策略/模型名称：中国商品期货 Alpha 因子科学筛选系统（data_final_raw统一版）
- 开发负责人：`[待填写]`
- 提交日期：2026-07-23
- 项目版本：`cn_futures_factor_science_data_final_v1`
- 原始合约文件SHA256：`1eed683f05bbaf5dcca2e0750ff82fe9bcf5fa8ecec730a8e67bfc47c32d57f8`
- 研究目的：将57个概念因子进行点时安全计算、统计筛选、经济分类和ML输入加工。

## 2. 数据基座与数据审计

### 2.1 数据范围

- 品种数：30；板块：能源、金属、化工、农产品。
- 完整数据范围：2015-01-05至2025-12-31。
- 开发特征区：2015-01-05至2024-12-31。
- 开发样本外筛选：2019-01-01至2024-12-31，且标签结束日不晚于2024-12-31。
- 封存测试：2025-01-01至2025-12-31，且标签结束日不晚于2025-12-31。

### 2.2 审计结果

| 审计项 | 结果 |
|---|---:|
| 原始实际合约日记录 | 846,963 |
| 实际合约代码 | 3,753 |
| `trade_date × product`面板行 | 78,112 |
| 交易日 | 2,674 |
| 重复主键 | 0 |
| 开发期行数 | 70,822 |
| 2025封存行数 | 7,290 |
| 实际合约映射覆盖 | 100.00% |
| 完全字段匹配率 | 94.99% |
| 高质量或完全匹配率 | 99.00% |
| 产品收益覆盖 | 99.72% |
| 至少2点曲线覆盖 | 98.37% |
| 至少3点曲线覆盖 | 82.33% |
| 新鲜现货覆盖 | 73.41% |
| 新鲜仓单覆盖 | 98.74% |
| 新鲜库存覆盖 | 33.95% |
| 库存覆盖品种 | 10 |
| 1/5/20日标签覆盖 | 98.66% / 98.46% / 92.86% |
| 点时违规 | 0 |
| 开发标签跨入2025（修正后） | 0 |

### 2.3 主力映射

1. 读取30品种的TuShare精确连续主力序列。
2. 用T日`pre_close/pre_settle/open/high/low/close/settle`与实际合约逐字段匹配。
3. 并列时依次使用上日实际合约连续性、持仓量、成交量、较晚到期日打破平局。
4. 禁止在无明确证据时倒退到更早到期月份。
5. 得到的`mapped_actual_ts_code`既用于收益标签，也用于后续可成交性检查。

这项修订不是“用当日持仓量事后选主力”：连续主力代码由数据源给定，持仓量只在行情字段无法唯一映射时作确定性并列处理。

### 2.4 期限结构

- 候选实际合约剩余期限限制为10—365日。
- 合约价格必须为正，且持仓占比或成交占比至少1%。
- Carry使用最近两个合格期限点并按到期差年化。
- 曲率使用前三个期限点，相对首尾线性插值计算中间偏离，处理不等距到期。
- 斜率在至少3点时，对期限—对数价格做以`sqrt(OI)`为权重的线性拟合。

因此期限结构因子必须同时使用主力和后续合约；只保留主力会直接导致Carry、斜率、曲率无法定义。

### 2.5 基本面点时处理

- 所有现货、仓单、库存先按观察日期对齐，再整体延迟1个交易日。
- 只有`observation_date < trade_date`时才允许进入因子。
- 现货/仓单/库存最大陈旧期分别为7/5/14个自然日。
- FU、I、SC因现货单位或汇率口径不可直接同比，排除基差计算。
- 不把周/月数据向前填充后的每一天当作新信息事件；Surprise只在观察日期改变时产生。

### 2.6 数据限制

- 现货源止于2023-12-29，因此2024—2025基差相关特征按规则缺失，不能人为延长。
- 只有OHLC锁价代理，没有交易所逐日精确涨跌停表。
- 缺少盘口、买卖价差、逐笔冲击和容量数据，无法声称已核验真实实盘冲击。
- 实际手续费任一字段覆盖约73.16%，且2021年前较弱；统一bps只用于横向研究比较。
- 库存仅覆盖10个品种，库存类结果存在品种集中风险。

## 3. 数据预处理与标签

### 3.1 原始数据清洗

- 主键统一为`trade_date × product`，日期升序、品种内排序。
- 正价格、正成交量和合约存在性作为收益与曲线计算硬门。
- 不使用后向填充（bfill），不把未来观测填回过去。
- 原始因子计算阶段缺失保持NaN；只有进入ML加工阶段才变为中性0并附缺失标记。
- Inf由非法分母防护产生前即被置为缺失，不用任意大数替换。

### 3.2 标签定义

对信号日T映射的实际合约c，h日标签为：

$$y^{(h)}_{i,T}=\frac{Open_{c,T+h+1}}{Open_{c,T+1}}-1,\quad h\in\{1,5,20\}.$$

- T收盘后形成信号；T+1开盘才允许交易。
- 持有期间固定同一真实合约，不随主力切换偷换持仓。
- 入场或退出开盘无价格、无成交量或疑似锁死时，标签置缺失。
- 主筛选标签为5日；1日、20日只作期限稳健性。

### 3.3 封存边界

任一评价窗口必须同时满足：

$$valid\_start\le T\le valid\_end,\qquad label\_end\_date_h\le valid\_end.$$

训练折还必须满足`label_end_date_5d < valid_start`，形成5日purge。这样既防训练标签越过验证起点，也防开发期标签进入2025封存区。

## 4. 因子计算全流程

### 4.1 流程总览

```text
data_final_raw实际合约 + TuShare连续主力
  → 主力连续行情映射至真实合约
  → 完整期限结构与点时基本面
  → 1/5/20日固定合约标签
  → 57个预注册原始因子
  → 数据质量硬门（覆盖、方差、事件数、品种数）
  → 47个可计算因子
  → IC/Rank IC/ICIR/HAC/FDR/分组回测
  → 年份、品种、板块、期限、延迟、窗口稳健性
  → 相关聚类与固定Ridge增量诊断
  → A—F研究分类
  → 点时安全特征加工
  → 92字段ML接口
```

### 4.2 主要因子构造

- 趋势：20/60/120日对数累计收益、波动率调整、路径效率及多周期一致性。
- Carry：最近两个合格真实合约的对数价差除以年化到期差。
- 基差：新鲜且单位可比的现货相对主力期货价格差。
- 价值：当前对数价格相对仅使用历史的252日均值/波动的反向偏离。
- 库存/仓单：按品种与ISO周进行仅用历史同期观测的扩展季节Z分数。
- 新息：当前事件值相对该品种前60次历史条件分布的标准化异常，只在观察日变化时记值。
- 流动性：20日Amihud、成交/持仓换手、成交量与持仓异常。
- 风险：20日实现波动、60日偏度、剔除市场和板块共同收益后的60日特质波动。
- 交互：仅保留预注册的动量×Carry、非线性紧张度、信息一致/背离等有经济含义组合。
- 元因子：四个冻结基础因子的已实现收益合成后取20日持续性；只作时间条件变量。

所有滚动均值、波动、条件期望和标准化基准均使用T时点以前的历史；当公式要求当日横截面时，只使用同日各品种在T收盘已知的数据。

### 4.3 数据质量硬门

- 稠密因子开发覆盖率至少20%。
- 事件因子至少100个有效事件、至少5个品种。
- 因子必须具有非零横截面/时间变化。
- 未通过者不伪造数值，保留概念登记并标记F类。

| 因子 | 缺失数据/无法计算原因 |
|---|---|
| `hedging_pressure` | data_final_raw没有商业/非商业交易者分类持仓 |
| `industry_chain_residual` | 未提供经确认的产业链邻接表和生产传导系数 |
| `market_state_conditional` | 当前协议未定义和冻结市场状态概率引擎 |
| `product_state_interaction` | 当前协议未定义和冻结单品种状态引擎 |
| `cost_transmission_gap` | 未提供经确认的产业链邻接表和实际生产配比 |
| `processing_margin_anomaly` | 缺少经确认的生产配比和其他成本序列 |
| `transmission_residual` | 未提供经确认的产业链邻接和传导滞后 |
| `supply_demand_shock` | data_final_raw没有供给与需求多源实际值及预期字段 |
| `crowding` | data_final_raw没有会员或参与者分项持仓 |
| `commodity_network_factor` | 未提供经确认的商品网络邻接表与权重 |

## 5. 扩展窗口设计

| 折 | 训练区间 | 验证区间 | Purge |
|---|---|---|---:|
| 2019 | 2015-01-05—2018-12-31 | 2019全年 | 5日 |
| 2020 | 2015-01-05—2019-12-31 | 2020全年 | 5日 |
| 2021 | 2015-01-05—2020-12-31 | 2021全年 | 5日 |
| 2022 | 2015-01-05—2021-12-31 | 2022全年 | 5日 |
| 2023 | 2015-01-05—2022-12-31 | 2023全年 | 5日 |
| 2024 | 2015-01-05—2023-12-31 | 2024全年 | 5日 |
| 封存2025 | 2015-01-05—2024-12-31 | 2025全年 | 5日 |

扩展窗口每轮扩大训练集而不丢弃早期数据，适合样本有限的商品期货；每个验证年都只使用此前可见数据。单因子IC主表把2019—2024各验证日期汇总，年度表保留逐年分解；固定Ridge增量测试在每一折重新拟合。

## 6. 因子预处理与机器学习输入加工

### 6.1 连续因子

同日横截面稳健标准化：

$$z_{i,t}=clip\left(\frac{x_{i,t}-Median_t(x)}{1.4826\,MAD_t(x)},-5,5\right).$$

- 当MAD为0时回退同日横截面标准差。
- 缺失值用同日中位数对应的中性值0表示。
- 这种处理降低极端行情和不同量纲对模型的支配。

### 6.2 时间序列变量

`factor_momentum`没有同日品种差异，使用仅含过去日期的252日滚动Z分数，最少60日，截断±5；不能用普通横截面IC解释。

### 6.3 缺失信息

- 原始缺失率达到1%的因子增加`__missing`二元字段。
- 92个特征由47个数值特征和45个缺失/质量标志组成。
- 缺失标志不是Alpha，而是让模型识别“数据不可用”与“因子中性”不同。

### 6.4 相关性、残差化和正交化

- 在开发期计算特征相关矩阵，绝对相关系数≥0.90的字段用并查集聚为同一簇。
- 当前92字段形成65个相关簇；每簇登记代表字段，但所有字段保留。
- 不在全样本上按标签挑选簇内赢家，避免选择偏差。
- 不做全局目标驱动正交化；只允许预注册的板块相对价值、特质波动等经济残差。
- 普通OLS不适合直接使用全部相关字段；队友若用线性模型应采用Ridge/Elastic Net，或只在每个训练折内选簇代表。

### 6.5 最终接口

- 输入字段：92；数值/主特征47，缺失标志45。
- 用途分布：predictor=36，conditional=5，risk=6，quality_flag=45。
- 唯一取列规则：读取`ml_feature_registry.csv`的全部`feature_name`。
- 不按A—F分类删列；A—F是证据等级，不是当前全量接口的硬特征选择。
- 训练数据位于项目`training_handoff/ml_dataset_development.csv.gz`，队友仅训练模型时不需要原始数据。

## 7. 科学筛选统计

### 7.1 IC、Rank IC与ICIR

- 每个交易日在至少8个有效品种上计算横截面Pearson IC和Spearman Rank IC。
- ICIR与Rank ICIR均为“逐日均值/逐日标准差”的未年化口径，不能等同策略夏普。
- Rank IC胜率为`RankIC_t > 0`的日期比例。
- 5日标签重叠引起自相关，均值检验使用最多5阶Newey–West/HAC标准误。

### 7.2 多重检验

- 对所有可评价因子的Rank IC p值统一做Benjamini–Hochberg FDR修正。
- 多个FDR q相同是BH单调化的正常结果：排序后的`m·p(i)/i`从尾部取累计最小值，邻近检验可映射到同一q值。
- 因此不能因为多个q相同而认定程序复制了结果。

### 7.3 分组与回测

每个第5个交易日：

1. 按预注册方向调整因子正负。
2. 有效截面至少8个品种。
3. 做多最高20%、做空最低20%；两侧总绝对权重各0.5，组内等权。
4. 使用固定实际合约T+1开盘至T+6开盘标签。
5. 换手为`sum(abs(w_t-w_{t-1}))`。
6. 净收益为`gross - turnover × cost_bps / 10000`。
7. 每年期数按`252/5`年化；净值用复利，最大回撤由累计净值峰谷计算。

这是一套公平比较单因子的统一映射，不为每个因子单独调阈值、持有期或成本。

### 7.4 稳健性

- 期限：1/5/20日Rank IC。
- 延迟：信号再延迟1个交易日。
- 时间：2019—2024逐年Rank IC、正向年份率及年度波动。
- 横截面：30品种时间序列相关、四板块分解。
- 参数：趋势40/60/80日；Carry变化、基差动量、反转、波动、持仓增长、曲线动量10/20/40日。
- 成本：0/3/5/10bps。
- 相关性：高相关簇、最大绝对相关、代表字段。

参数邻域结果如下（诊断，不用于事后挑最佳窗口）：

| 家族 | 窗口 | Rank IC | HAC t | FDR q | 覆盖率 |
|---|---|---|---|---|---|
| tsmom | 40 | -0.0114 | -0.8941 | 0.6500 | 0.9991 |
| tsmom | 60 | -0.0069 | -0.5482 | 0.8755 | 0.9986 |
| tsmom | 80 | 0.0002 | 0.0154 | 0.9877 | 0.9981 |
| carry_change | 10 | 0.0010 | 0.1141 | 0.9546 | 0.9984 |
| basis_momentum | 10 | -0.0036 | -0.3067 | 0.8856 | 0.7420 |
| short_reversal | 10 | 0.0197 | 1.5704 | 0.6496 | 0.9998 |
| realized_volatility | 10 | 0.0166 | 1.1188 | 0.6500 | 0.9998 |
| oi_growth | 10 | 0.0170 | 1.8164 | 0.6496 | 0.9998 |
| term_structure_momentum | 10 | 0.0162 | 1.4240 | 0.6496 | 0.9051 |
| carry_change | 20 | -0.0090 | -0.9347 | 0.6500 | 0.9976 |
| basis_momentum | 20 | -0.0052 | -0.4043 | 0.8856 | 0.7418 |
| short_reversal | 20 | 0.0174 | 1.2864 | 0.6500 | 0.9995 |
| realized_volatility | 20 | 0.0128 | 0.8375 | 0.6502 | 0.9995 |
| oi_growth | 20 | 0.0145 | 1.4524 | 0.6496 | 0.9995 |
| term_structure_momentum | 20 | 0.0137 | 1.1491 | 0.6500 | 0.8929 |
| carry_change | 40 | 0.0035 | 0.3360 | 0.8856 | 0.9972 |
| basis_momentum | 40 | -0.0051 | -0.4295 | 0.8856 | 0.7414 |
| short_reversal | 40 | 0.0131 | 0.9471 | 0.6500 | 0.9991 |
| realized_volatility | 40 | 0.0152 | 0.9842 | 0.6500 | 0.9991 |
| oi_growth | 40 | 0.0261 | 2.7212 | 0.1382 | 0.9991 |
| term_structure_momentum | 40 | 0.0025 | 0.2184 | 0.9142 | 0.8837 |

### 7.5 增量价值

- 固定基线：TSMOM、Carry、Amihud、Turnover。
- 模型：Ridge，alpha=10，不调参。
- 每折比较加入一个候选前后OOS MSE和日度Rank IC。
- 训练样本严格满足标签结束日早于验证起点；验证标签结束日在验证年内。
- 共评价34个预测候选，其中14个平均增量Rank IC为正。
- 结果仅判断互补性，不反向改变特征注册表。

## 8. A—F分类依据

分类顺序与门槛：

1. 不可计算 → F。
2. 预注册用途为risk → D。
3. 预注册用途为conditional或quality flag → C。
4. 预测因子证据门：Rank IC≥0.02。
5. 经济门：单边5bps后年化收益>0。
6. 显著性门：全库FDR q≤0.10。
7. core角色同时通过4—6 → A。
8. 通过经济门，且通过证据门或显著性门 → B。
9. 其余可计算预测因子 → E。

该分类是保守的研究证据分级。年份、品种、板块、延迟和参数结果在报告中作为稳健性诊断，但当前代码没有把它们全部写成A类硬门；这点必须透明披露。

## 9. 筛选结论

- A=0、B=8、C=5、D=6、E=28、F=10。
- 没有A类不是“因子全部无效”，而是没有因子同时通过10%全库FDR、Rank IC和5bps经济门。
- B类代表辅助候选，不等于已验证商业Alpha。
- 开发期与2025可评价Rank IC同方向比例为37.5%；跨期稳定性仍弱。
- 47个可计算因子全部进入ML接口，是预注册+数据质量策略；模型必须用正则化与严格滚动验证控制弱信号和冗余。

### 9.1 因子家族

| 家族 | 概念数 | 可计算 | 平均Rank IC | 中位Rank IC | 平均5bps夏普 | 平均覆盖率 |
|---|---|---|---|---|---|---|
| basis_spot | 4 | 4 | 0.0087 | 0.0045 | -0.0243 | 0.7444 |
| factor_interaction | 2 | 2 | 0.0143 | 0.0143 | 0.0546 | 0.8902 |
| industry_chain | 6 | 1 | -0.0109 | -0.0109 | -0.4718 | 0.1664 |
| information_news | 8 | 7 | 0.0112 | 0.0139 | 0.0640 | 0.5521 |
| inventory_fundamental | 5 | 5 | 0.0183 | 0.0211 | 0.3489 | 0.5311 |
| meta_factor | 1 | 1 | — | — | — | 1.0000 |
| state_interaction | 2 | 0 | — | — | — | 0.0000 |
| term_structure | 6 | 6 | 0.0108 | 0.0139 | -0.1502 | 0.9350 |
| trading_liquidity | 9 | 7 | 0.0053 | 0.0063 | -0.0596 | 0.7774 |
| trend_momentum | 6 | 6 | -0.0046 | -0.0069 | -0.1343 | 0.9984 |
| value_reversion | 3 | 3 | 0.0057 | 0.0127 | -0.2071 | 0.9867 |
| volatility_risk | 5 | 5 | -0.0060 | -0.0128 | -0.2854 | 0.9987 |

### 9.2 B类候选及核心OOS指标

本项目没有预注册的“B类等权总策略”，因此不存在一个诚实的单一年化收益/回撤/夏普。下表逐因子报告团队3bps口径与项目5bps口径；禁止事后把表现最好的因子拼成总策略再宣称OOS。

| 因子 | Rank IC | Rank ICIR | FDR q | 3bps年化 | 3bps夏普 | 3bps最大回撤 | 胜率 | 日均等效换手 | 5bps年化 | 2025 Rank IC | 2025 5bps年化 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `carry_annualized` | 0.0293 | 0.1118 | 0.1931 | 3.7% | 0.4170 | -26.5% | 52.4% | 9.4% | 3.2% | 0.0323 | 2.9% |
| `basis_fresh` | 0.0309 | 0.1288 | 0.1926 | 7.3% | 0.8085 | -13.0% | 56.2% | 12.3% | 6.7% | — | — |
| `inventory_change_5` | 0.0211 | 0.0642 | 0.4592 | 7.1% | 0.7781 | -15.9% | 55.2% | 22.6% | 5.9% | -0.0557 | -9.0% |
| `warehouse_tightness` | 0.0236 | 0.1120 | 0.1926 | 4.4% | 0.5914 | -7.1% | 54.5% | 10.4% | 3.8% | 0.0653 | 3.7% |
| `information_divergence` | 0.0257 | 0.0939 | 0.2816 | 3.4% | 0.4267 | -14.2% | 54.3% | 13.0% | 2.7% | -0.1062 | -20.9% |
| `term_structure_slope` | 0.0241 | 0.0867 | 0.4274 | 4.0% | 0.3938 | -18.4% | 50.7% | 7.5% | 3.6% | 0.0165 | -2.3% |
| `inventory_acceleration` | 0.0316 | 0.0954 | 0.1926 | 6.0% | 0.5760 | -17.4% | 54.8% | 31.9% | 4.3% | 0.0136 | 2.7% |
| `inventory_price_divergence` | 0.0315 | 0.0938 | 0.2316 | 5.9% | 0.6530 | -12.5% | 52.8% | 23.0% | 4.7% | -0.0836 | -18.5% |

### 9.3 经济解释

- **Carry/期限结构斜率**：Backwardation可反映近端稀缺、便利收益和风险转移补偿；完整曲线比单点价差更稳定。
- **基差**：新鲜现货强于期货说明近端供需偏紧；但现货在2023后缺失，不能做2025确认。
- **仓单紧张度**：可交割库存的边际稀缺比总库存更贴近交割约束，覆盖也明显更高。
- **库存变化/加速度**：去库本身及去库速度加快可能传递供需趋紧，但仅10个品种有库存，必须防止样本集中。
- **信息背离**：供需紧张度相对价格趋势的差异可能代表信息未充分吸收；2025反向说明当前代理并非真正市场一致预期。
- **库存—价格背离**：库存趋紧而价格未跟随可能形成后续修复；同样受库存覆盖低和2025反向制约。
- **趋势家族整体偏弱**：公开中期趋势在本样本与当前open-to-open标签下未形成稳定横截面优势，不应仅凭传统声誉升级。
- **风险/流动性变量**：其任务是解释条件风险、可交易性和尾部暴露，单独多空收益不是主要入选标准。

## 10. 模型架构与训练细节

- 最终模型/算法类型：不适用；本提交不训练最终模型。
- 核心超参数：不适用。
- 损失函数：不适用。
- 防过拟合：通过预注册、扩展窗口、purge、FDR、参数邻域和不使用目标选特征实现。
- 唯一模型使用：固定Ridge(alpha=10)仅作增量诊断，不产生本报告最终交易信号。

## 11. 交易执行与净值口径

- 信号平滑：无。
- 入场：横截面前20%多、后20%空。
- 平仓：固定持有5个交易日，在T+6开盘退出。
- 底池过滤：只执行数据可用、价格/成交及疑似锁价可执行硬门；无额外流动性底20%剔除。
- 成本：0/3/5/10bps单边；主分类5bps，团队比较表3bps。
- 净值CSV按退出日确认整期5日收益；中间日收益填0以保持日历索引，不能用于日内风险或逐日VaR分析。

## 12. 核心评价指标填写说明

团队模板要求单一策略的年化、回撤、夏普、胜率和换手，但本实验产物是57个单因子诊断，不是一个最终组合：

- 年化收益率：不适用（无预注册总策略）。
- 最大回撤：不适用（无预注册总策略）。
- 夏普比率：不适用（无预注册总策略）。
- 胜率：不适用（无预注册总策略）。
- 日均换手率：不适用（无预注册总策略）。
- 5日/20日Rank IC：逐因子列于`factor_backtest_metrics.csv`。
- 多空分组单调性：逐因子列于同一CSV。

如果团队必须比较一个模型策略，应由建模负责人基于冻结的92字段、相同折和相同执行协议生成，不能由因子筛选负责人事后选因子代替。

## 13. 可复现性与运行方式

```bash
cd /Users/Wantree/Desktop/因子筛选系统/unified_factor_system
PYTHONPATH=src python scripts/run_pipeline.py
PYTHONPATH=src python scripts/build_icir_report.py
PYTHONPATH=src python scripts/build_factor_analysis_report.py
PYTHONPATH=src python scripts/build_factor_screening_submission.py
pytest -q
```

主流水线拒绝覆盖已有产物；如需完整重跑，应先把旧生成目录移动到备份位置。不能删除原始数据，也不能用旧结果覆盖本次统一版。

## 14. 最终判断

本轮工作的核心成果不是发现一个可立即商业化的强Alpha，而是建立了一个可审计的因子研究基座：57个概念完整登记、47个点时安全可计算、92个字段可直接交接建模、10个数据缺口明确记录、开发与封存边界隔离。最有希望的方向集中在期限结构、基差、仓单和库存边际变化，但没有任何因子通过10%全库FDR，且2025方向一致率不足一半。因此正确结论是“形成高质量研究输入和若干辅助候选”，不是“已证明稳定盈利”。

# 附录A：57个因子定义与数据来源

| 编号 | 因子 | 家族 | 数学/程序定义 | 数据来源 | 可计算 | 分类 |
|---:|---|---|---|---|---|---|
| 1 | `tsmom_60` 时间序列动量 | trend_momentum | `log_return_60/(vol_60*sqrt(60))` | 实际主力合约收益 | 是 | E 暂不使用因子 |
| 2 | `csmom_60` 横截面动量 | trend_momentum | `cross_sectional_rank(log_return_60)` | 实际主力合约收益 | 是 | E 暂不使用因子 |
| 3 | `trend_efficiency_60` 趋势效率 | trend_momentum | `abs(log_return_60)/sum(abs(daily_log_return),60)` | 实际主力合约收益 | 是 | E 暂不使用因子 |
| 4 | `multi_horizon_trend` 多周期趋势一致性 | trend_momentum | `mean(sign(return_20),sign(return_60),sign(return_120))` | 实际主力合约收益 | 是 | E 暂不使用因子 |
| 5 | `carry_annualized` Carry因子 | term_structure | `log(curve_near/curve_middle)*365/maturity_gap` | 完整实际合约曲线、收盘价、持仓量、成交量与到期日 | 是 | B 辅助预测因子 |
| 6 | `carry_change_20` Carry变化 | term_structure | `carry_t-carry_t-20` | 主力与次近月收盘价、到期日 | 是 | E 暂不使用因子 |
| 7 | `basis_fresh` 基差 | basis_spot | `(fresh_spot-main_close)/main_close` | 现货价格、观察日、主力收盘价 | 是 | B 辅助预测因子 |
| 8 | `basis_shock_20` 基差冲击 | basis_spot | `zscore(delta_basis; prior_20)` | 现货价格、观察日、主力收盘价 | 是 | E 暂不使用因子 |
| 9 | `basis_momentum_20` 基差动量 | basis_spot | `basis_t-basis_t-20` | 延迟一交易日可用的现货与主力收盘价 | 是 | E 暂不使用因子 |
| 10 | `curve_curvature` 期限结构曲率 | term_structure | `2*(log(middle)-interpolate(log(near),log(far)))` | 通过流动性和期限硬门的前三个实际合约 | 是 | E 暂不使用因子 |
| 11 | `value_252` 价值因子 | value_reversion | `-(log_price-rolling_mean_252)/rolling_std_252` | 主力收盘价 | 是 | E 暂不使用因子 |
| 12 | `short_reversal_5` 短期反转 | value_reversion | `-log_return_5` | 实际主力合约收益 | 是 | E 暂不使用因子 |
| 13 | `inventory_scarcity` 库存稀缺因子 | inventory_fundamental | `-seasonal_z(stock_current)` | 库存、观察日与数据年龄 | 是 | E 暂不使用因子 |
| 14 | `inventory_change_5` 库存变化因子 | inventory_fundamental | `-delta_5(stock_current)` | 库存、观察日与数据年龄 | 是 | B 辅助预测因子 |
| 15 | `inventory_surprise` 库存意外／库存新息 | information_news | `-innovation_z(stock_current; prior_60)` | 库存、观察日与数据年龄 | 是 | E 暂不使用因子 |
| 16 | `warehouse_tightness` 仓单紧张因子 | inventory_fundamental | `-seasonal_z(warehouse_receipt)` | 注册仓单及观察日 | 是 | B 辅助预测因子 |
| 17 | `tightness_composite` 供需紧张度 | inventory_fundamental | `mean(-inventory_z,-warehouse_z,basis_z,carry_z)` | 库存、仓单、现货、期限结构 | 是 | E 暂不使用因子 |
| 18 | `oi_growth_20` 持仓量增长 | trading_liquidity | `log(OI_t)-log(OI_t-20)` | 主力持仓量 | 是 | C 条件变量 |
| 19 | `price_oi_interaction_20` 价格—持仓量交互 | trading_liquidity | `log_return_20*oi_growth_20` | 实际主力合约收益、持仓量 | 是 | E 暂不使用因子 |
| 20 | `hedging_pressure` 套保压力 | trading_liquidity | `(commercial_long-commercial_short)/OI` | 可靠交易者分类持仓 | 否 | F 数据不足无法判断 |
| 21 | `volume_shock_20` 成交量冲击 | trading_liquidity | `zscore(log_volume; prior_20)` | 主力成交量 | 是 | C 条件变量 |
| 22 | `amihud_20` Amihud非流动性 | trading_liquidity | `mean(abs(return)/amount,20)` | 实际主力合约收益、成交额 | 是 | D 风险变量 |
| 23 | `turnover_oi` 成交量—持仓量换手率 | trading_liquidity | `volume/open_interest` | 主力成交量、持仓量 | 是 | C 条件变量 |
| 24 | `realized_volatility_20` 已实现波动率 | volatility_risk | `std(log_return,20)*sqrt(252)` | 实际主力合约收益 | 是 | D 风险变量 |
| 25 | `low_volatility_20` 低波动因子 | volatility_risk | `-cross_sectional_rank(realized_volatility_20)` | 实际主力合约收益 | 是 | D 风险变量 |
| 26 | `realized_skewness_60` 已实现偏度 | volatility_risk | `skew(log_return,60)` | 实际主力合约收益 | 是 | D 风险变量 |
| 27 | `seasonality_3y` 季节性因子 | value_reversion | `mean(trailing_return_5 shifted 252/504/756)` | 实际主力合约历史收益 | 是 | E 暂不使用因子 |
| 28 | `industry_chain_residual` 产业链残差 | industry_chain | `x-E[x\|industry_neighbors]` | 产业链邻接关系与传导系数 | 否 | F 数据不足无法判断 |
| 29 | `sector_relative_value_60` 相对价值残差 | industry_chain | `return_60-sector_mean_return_60` | 实际主力合约收益、固定板块映射 | 是 | E 暂不使用因子 |
| 30 | `market_state_conditional` 状态条件因子 | state_interaction | `factor*p_market_state` | 实时市场状态概率 | 否 | F 数据不足无法判断 |
| 31 | `product_state_interaction` 单品种状态交互因子 | state_interaction | `factor*product_state` | 实时单品种状态 | 否 | F 数据不足无法判断 |
| 32 | `factor_momentum` 因子动量 | meta_factor | `rolling_sum_20(mean(lagged_factor_portfolio_return))` | 严格滞后一期的动量、Carry、反转和板块相对价值组合收益 | 是 | C 条件变量 |
| 33 | `fundamental_composite_news` 基本面综合新息代理 | information_news | `mean(inventory_surprise,warehouse_surprise,basis_surprise)` | 延迟一交易日可用的库存、仓单和现货；不是市场一致预期 | 是 | E 暂不使用因子 |
| 34 | `fundamental_underreaction` 基本面反应不足代理 | information_news | `fundamental_composite_news-cs_z(initial_price_response)` | 基本面综合新息代理与主力收益；不是市场一致预期 | 是 | E 暂不使用因子 |
| 35 | `information_consistency` 信息一致性 | information_news | `mean(z_k)*abs(mean(sign(z_k)))` | 库存、仓单、基差、Carry | 是 | E 暂不使用因子 |
| 36 | `information_divergence` 信息背离 | information_news | `tightness_z-tsmom_z` | 供需紧张度与实际主力合约收益 | 是 | B 辅助预测因子 |
| 37 | `cost_transmission_gap` 成本传导缺口 | industry_chain | `sum(a_ij*upstream_return)-downstream_return` | 上下游关系与生产配比 | 否 | F 数据不足无法判断 |
| 38 | `processing_margin_anomaly` 加工利润异常 | industry_chain | `seasonal_z(output-sum(a*input)-other_cost)` | 产出投入价格、生产配比和其他成本 | 否 | F 数据不足无法判断 |
| 39 | `warehouse_surprise` 仓单新息 | information_news | `-innovation_z(warehouse_receipt; prior_60)` | 注册仓单及观察日 | 是 | E 暂不使用因子 |
| 40 | `transmission_residual` 产业链传导残差 | industry_chain | `r_i-sum(beta*lagged_upstream_return)-beta_sector*r_sector` | 产业链邻接、滞后与板块收益 | 否 | F 数据不足无法判断 |
| 41 | `momentum_quality_60` 动量质量因子 | trend_momentum | `tsmom_60*trend_efficiency_60` | 实际主力合约收益 | 是 | E 暂不使用因子 |
| 42 | `vol_managed_momentum` 波动率管理动量 | trend_momentum | `sign(return_60)/volatility_20` | 实际主力合约收益 | 是 | E 暂不使用因子 |
| 43 | `idiosyncratic_volatility_60` 特质波动因子 | volatility_risk | `std(residual(r_i~r_market+r_sector),60)` | 实际主力合约收益、固定板块映射 | 是 | D 风险变量 |
| 44 | `term_structure_slope` 期限结构斜率 | term_structure | `-weighted_OLS_slope(log_future_price~maturity)` | 至少三个通过流动性和期限硬门的实际合约 | 是 | B 辅助预测因子 |
| 45 | `term_structure_momentum` 期限结构动量 | term_structure | `slope_t-slope_t-20` | 完整曲线斜率历史 | 是 | E 暂不使用因子 |
| 46 | `term_structure_acceleration` 期限结构加速度 | term_structure | `delta20_slope_t-delta20_slope_t-20` | 完整曲线斜率历史 | 是 | E 暂不使用因子 |
| 47 | `basis_surprise` 基差意外 | basis_spot | `innovation_z(basis; prior_60)` | 新鲜现货价格、观察日、主力收盘价 | 是 | E 暂不使用因子 |
| 48 | `inventory_acceleration` 库存加速度 | inventory_fundamental | `-(delta_5_inventory_t-delta_5_inventory_t-5)` | 库存、观察日与数据年龄 | 是 | B 辅助预测因子 |
| 49 | `inventory_price_divergence` 库存—价格背离 | information_news | `z(-delta_inventory)-z(return_20)` | 库存与实际主力合约收益 | 是 | B 辅助预测因子 |
| 50 | `supply_demand_shock` 供需冲击因子 | information_news | `sum(demand_surprise)-sum(supply_surprise)` | 产量、需求、进口等实际与预期数据 | 否 | F 数据不足无法判断 |
| 51 | `crowding` 拥挤度因子 | trading_liquidity | `HHI(member_position_share)` | 会员或参与者分项持仓 | 否 | F 数据不足无法判断 |
| 52 | `oi_surprise_60` 持仓意外因子 | trading_liquidity | `zscore(delta_log_OI; prior_60)` | 主力持仓量 | 是 | C 条件变量 |
| 53 | `volume_price_divergence_20` 量价背离因子 | trading_liquidity | `z(return_20)-z(delta_log_volume_20)` | 实际主力合约收益、成交量 | 是 | E 暂不使用因子 |
| 54 | `momentum_carry_interaction` 因子交互因子 | factor_interaction | `z(tsmom_60)*z(carry)` | 动量与期限结构 | 是 | E 暂不使用因子 |
| 55 | `nonlinear_tightness` 非线性基本面因子 | factor_interaction | `sign(tightness_z)*abs(tightness_z)^2` | 供需紧张度 | 是 | E 暂不使用因子 |
| 56 | `commodity_network_factor` 商品网络因子 | industry_chain | `sum(w_ij*z(signal_j))` | 经确认的商品网络邻接与权重 | 否 | F 数据不足无法判断 |
| 57 | `correlation_risk_20_120` 相关性风险因子 | volatility_risk | `corr_20(r_i,r_market)-corr_120(r_i,r_market)` | 实际主力合约收益、商品市场收益 | 是 | D 风险变量 |

# 附录B：92个机器学习输入字段

| feature_name | source_factor | transformation | category | usage_type | missing_rate | cluster | representative |
|---|---|---|---|---|---:|---:|---|
| `tsmom_60__csz` | `tsmom_60` | same_date_cross_sectional_median_MAD_zscore_clip_5_missing_to_cross_sectional_median | trend_momentum | predictor | 0.0395 | 1 | `tsmom_60__csz` |
| `tsmom_60__missing` | `tsmom_60` | raw_missing_indicator | data_quality | quality_flag | 0.0000 | 2 | `tsmom_60__missing` |
| `csmom_60__csz` | `csmom_60` | same_date_cross_sectional_median_MAD_zscore_clip_5_missing_to_cross_sectional_median | trend_momentum | predictor | 0.0383 | 3 | `csmom_60__csz` |
| `csmom_60__missing` | `csmom_60` | raw_missing_indicator | data_quality | quality_flag | 0.0000 | 2 | `tsmom_60__missing` |
| `trend_efficiency_60__csz` | `trend_efficiency_60` | same_date_cross_sectional_median_MAD_zscore_clip_5_missing_to_cross_sectional_median | trend_momentum | predictor | 0.0395 | 4 | `trend_efficiency_60__csz` |
| `trend_efficiency_60__missing` | `trend_efficiency_60` | raw_missing_indicator | data_quality | quality_flag | 0.0000 | 2 | `tsmom_60__missing` |
| `multi_horizon_trend__csz` | `multi_horizon_trend` | same_date_cross_sectional_median_MAD_zscore_clip_5_missing_to_cross_sectional_median | trend_momentum | predictor | 0.0673 | 5 | `multi_horizon_trend__csz` |
| `multi_horizon_trend__missing` | `multi_horizon_trend` | raw_missing_indicator | data_quality | quality_flag | 0.0000 | 6 | `multi_horizon_trend__missing` |
| `carry_annualized__csz` | `carry_annualized` | same_date_cross_sectional_median_MAD_zscore_clip_5_missing_to_cross_sectional_median | term_structure | predictor | 0.0180 | 7 | `carry_annualized__csz` |
| `carry_annualized__missing` | `carry_annualized` | raw_missing_indicator | data_quality | quality_flag | 0.0000 | 8 | `carry_annualized__missing` |
| `carry_change_20__csz` | `carry_change_20` | same_date_cross_sectional_median_MAD_zscore_clip_5_missing_to_cross_sectional_median | term_structure | predictor | 0.0355 | 9 | `carry_change_20__csz` |
| `carry_change_20__missing` | `carry_change_20` | raw_missing_indicator | data_quality | quality_flag | 0.0000 | 10 | `carry_change_20__missing` |
| `basis_fresh__csz` | `basis_fresh` | same_date_cross_sectional_median_MAD_zscore_clip_5_missing_to_cross_sectional_median | basis_spot | predictor | 0.1922 | 11 | `basis_fresh__csz` |
| `basis_fresh__missing` | `basis_fresh` | raw_missing_indicator | data_quality | quality_flag | 0.0000 | 12 | `basis_fresh__missing` |
| `basis_shock_20__csz` | `basis_shock_20` | same_date_cross_sectional_median_MAD_zscore_clip_5_missing_to_cross_sectional_median | basis_spot | predictor | 0.2043 | 13 | `basis_shock_20__csz` |
| `basis_shock_20__missing` | `basis_shock_20` | raw_missing_indicator | data_quality | quality_flag | 0.0000 | 12 | `basis_fresh__missing` |
| `basis_momentum_20__csz` | `basis_momentum_20` | same_date_cross_sectional_median_MAD_zscore_clip_5_missing_to_cross_sectional_median | basis_spot | predictor | 0.2080 | 14 | `basis_momentum_20__csz` |
| `basis_momentum_20__missing` | `basis_momentum_20` | raw_missing_indicator | data_quality | quality_flag | 0.0000 | 12 | `basis_fresh__missing` |
| `curve_curvature__csz` | `curve_curvature` | same_date_cross_sectional_median_MAD_zscore_clip_5_missing_to_cross_sectional_median | term_structure | predictor | 0.1922 | 15 | `curve_curvature__csz` |
| `curve_curvature__missing` | `curve_curvature` | raw_missing_indicator | data_quality | quality_flag | 0.0000 | 16 | `curve_curvature__missing` |
| `value_252__csz` | `value_252` | same_date_cross_sectional_median_MAD_zscore_clip_5_missing_to_cross_sectional_median | value_reversion | predictor | 0.0553 | 17 | `value_252__csz` |
| `value_252__missing` | `value_252` | raw_missing_indicator | data_quality | quality_flag | 0.0000 | 18 | `value_252__missing` |
| `short_reversal_5__csz` | `short_reversal_5` | same_date_cross_sectional_median_MAD_zscore_clip_5_missing_to_cross_sectional_median | value_reversion | predictor | 0.0065 | 19 | `short_reversal_5__csz` |
| `inventory_scarcity__csz` | `inventory_scarcity` | same_date_cross_sectional_median_MAD_zscore_clip_5_missing_to_cross_sectional_median | inventory_fundamental | predictor | 0.7713 | 20 | `inventory_scarcity__csz` |
| `inventory_scarcity__missing` | `inventory_scarcity` | raw_missing_indicator | data_quality | quality_flag | 0.0000 | 21 | `inventory_scarcity__missing` |
| `inventory_change_5__csz` | `inventory_change_5` | same_date_cross_sectional_median_MAD_zscore_clip_5_missing_to_cross_sectional_median | inventory_fundamental | predictor | 0.6610 | 22 | `inventory_change_5__csz` |
| `inventory_change_5__missing` | `inventory_change_5` | raw_missing_indicator | data_quality | quality_flag | 0.0000 | 23 | `inventory_change_5__missing` |
| `inventory_surprise__csz` | `inventory_surprise` | same_date_cross_sectional_median_MAD_zscore_clip_5_missing_to_cross_sectional_median | information_news | predictor | 0.9465 | 24 | `inventory_surprise__csz` |
| `inventory_surprise__missing` | `inventory_surprise` | raw_missing_indicator | data_quality | quality_flag | 0.0000 | 25 | `inventory_surprise__missing` |
| `warehouse_tightness__csz` | `warehouse_tightness` | same_date_cross_sectional_median_MAD_zscore_clip_5_missing_to_cross_sectional_median | inventory_fundamental | predictor | 0.1852 | 26 | `warehouse_tightness__csz` |
| `warehouse_tightness__missing` | `warehouse_tightness` | raw_missing_indicator | data_quality | quality_flag | 0.0000 | 27 | `warehouse_tightness__missing` |
| `tightness_composite__csz` | `tightness_composite` | same_date_cross_sectional_median_MAD_zscore_clip_5_missing_to_cross_sectional_median | inventory_fundamental | predictor | 0.2944 | 28 | `tightness_composite__csz` |
| `tightness_composite__missing` | `tightness_composite` | raw_missing_indicator | data_quality | quality_flag | 0.0000 | 29 | `tightness_composite__missing` |
| `oi_growth_20__csz` | `oi_growth_20` | same_date_cross_sectional_median_MAD_zscore_clip_5_missing_to_cross_sectional_median | trading_liquidity | conditional | 0.0130 | 30 | `oi_growth_20__csz` |
| `oi_growth_20__missing` | `oi_growth_20` | raw_missing_indicator | data_quality | quality_flag | 0.0000 | 31 | `oi_growth_20__missing` |
| `price_oi_interaction_20__csz` | `price_oi_interaction_20` | same_date_cross_sectional_median_MAD_zscore_clip_5_missing_to_cross_sectional_median | trading_liquidity | predictor | 0.0195 | 32 | `price_oi_interaction_20__csz` |
| `price_oi_interaction_20__missing` | `price_oi_interaction_20` | raw_missing_indicator | data_quality | quality_flag | 0.0000 | 33 | `price_oi_interaction_20__missing` |
| `volume_shock_20__csz` | `volume_shock_20` | same_date_cross_sectional_median_MAD_zscore_clip_5_missing_to_cross_sectional_median | trading_liquidity | conditional | 0.0168 | 34 | `volume_shock_20__csz` |
| `volume_shock_20__missing` | `volume_shock_20` | raw_missing_indicator | data_quality | quality_flag | 0.0000 | 35 | `volume_shock_20__missing` |
| `amihud_20__csz` | `amihud_20` | same_date_cross_sectional_median_MAD_zscore_clip_5_missing_to_cross_sectional_median | trading_liquidity | risk | 0.0249 | 36 | `amihud_20__csz` |
| `amihud_20__missing` | `amihud_20` | raw_missing_indicator | data_quality | quality_flag | 0.0000 | 33 | `price_oi_interaction_20__missing` |
| `turnover_oi__csz` | `turnover_oi` | same_date_cross_sectional_median_MAD_zscore_clip_5_missing_to_cross_sectional_median | trading_liquidity | conditional | 0.0039 | 37 | `turnover_oi__csz` |
| `realized_volatility_20__csz` | `realized_volatility_20` | same_date_cross_sectional_median_MAD_zscore_clip_5_missing_to_cross_sectional_median | volatility_risk | risk | 0.0160 | 38 | `realized_volatility_20__csz` |
| `realized_volatility_20__missing` | `realized_volatility_20` | raw_missing_indicator | data_quality | quality_flag | 0.0000 | 33 | `price_oi_interaction_20__missing` |
| `low_volatility_20__csz` | `low_volatility_20` | same_date_cross_sectional_median_MAD_zscore_clip_5_missing_to_cross_sectional_median | volatility_risk | risk | 0.0160 | 38 | `realized_volatility_20__csz` |
| `low_volatility_20__missing` | `low_volatility_20` | raw_missing_indicator | data_quality | quality_flag | 0.0000 | 33 | `price_oi_interaction_20__missing` |
| `realized_skewness_60__csz` | `realized_skewness_60` | same_date_cross_sectional_median_MAD_zscore_clip_5_missing_to_cross_sectional_median | volatility_risk | risk | 0.0383 | 39 | `realized_skewness_60__csz` |
| `realized_skewness_60__missing` | `realized_skewness_60` | raw_missing_indicator | data_quality | quality_flag | 0.0000 | 2 | `tsmom_60__missing` |
| `seasonality_3y__csz` | `seasonality_3y` | same_date_cross_sectional_median_MAD_zscore_clip_5_missing_to_cross_sectional_median | value_reversion | predictor | 0.3291 | 40 | `seasonality_3y__csz` |
| `seasonality_3y__missing` | `seasonality_3y` | raw_missing_indicator | data_quality | quality_flag | 0.0000 | 41 | `seasonality_3y__missing` |
| `sector_relative_value_60__csz` | `sector_relative_value_60` | same_date_cross_sectional_median_MAD_zscore_clip_5_missing_to_cross_sectional_median | industry_chain | predictor | 0.0383 | 42 | `sector_relative_value_60__csz` |
| `sector_relative_value_60__missing` | `sector_relative_value_60` | raw_missing_indicator | data_quality | quality_flag | 0.0000 | 2 | `tsmom_60__missing` |
| `factor_momentum__tsz` | `factor_momentum` | prior_only_rolling_date_zscore_window_252_minimum_60_clip_5_missing_to_zero | meta_factor | conditional | 0.0262 | 43 | `factor_momentum__tsz` |
| `factor_momentum__missing` | `factor_momentum` | raw_missing_indicator | data_quality | quality_flag | 0.0000 | 44 | `factor_momentum__missing` |
| `fundamental_composite_news__csz` | `fundamental_composite_news` | same_date_cross_sectional_median_MAD_zscore_clip_5_missing_to_cross_sectional_median | information_news | predictor | 0.2119 | 45 | `fundamental_composite_news__csz` |
| `fundamental_composite_news__missing` | `fundamental_composite_news` | raw_missing_indicator | data_quality | quality_flag | 0.0000 | 12 | `basis_fresh__missing` |
| `fundamental_underreaction__csz` | `fundamental_underreaction` | same_date_cross_sectional_median_MAD_zscore_clip_5_missing_to_cross_sectional_median | information_news | predictor | 0.2121 | 46 | `fundamental_underreaction__csz` |
| `fundamental_underreaction__missing` | `fundamental_underreaction` | raw_missing_indicator | data_quality | quality_flag | 0.0000 | 12 | `basis_fresh__missing` |
| `information_consistency__csz` | `information_consistency` | same_date_cross_sectional_median_MAD_zscore_clip_5_missing_to_cross_sectional_median | information_news | predictor | 0.2944 | 28 | `tightness_composite__csz` |
| `information_consistency__missing` | `information_consistency` | raw_missing_indicator | data_quality | quality_flag | 0.0000 | 29 | `tightness_composite__missing` |
| `information_divergence__csz` | `information_divergence` | same_date_cross_sectional_median_MAD_zscore_clip_5_missing_to_cross_sectional_median | information_news | predictor | 0.2951 | 47 | `information_divergence__csz` |
| `information_divergence__missing` | `information_divergence` | raw_missing_indicator | data_quality | quality_flag | 0.0000 | 29 | `tightness_composite__missing` |
| `warehouse_surprise__csz` | `warehouse_surprise` | same_date_cross_sectional_median_MAD_zscore_clip_5_missing_to_cross_sectional_median | information_news | predictor | 0.0558 | 48 | `warehouse_surprise__csz` |
| `warehouse_surprise__missing` | `warehouse_surprise` | raw_missing_indicator | data_quality | quality_flag | 0.0000 | 49 | `warehouse_surprise__missing` |
| `momentum_quality_60__csz` | `momentum_quality_60` | same_date_cross_sectional_median_MAD_zscore_clip_5_missing_to_cross_sectional_median | trend_momentum | predictor | 0.0395 | 50 | `momentum_quality_60__csz` |
| `momentum_quality_60__missing` | `momentum_quality_60` | raw_missing_indicator | data_quality | quality_flag | 0.0000 | 2 | `tsmom_60__missing` |
| `vol_managed_momentum__csz` | `vol_managed_momentum` | same_date_cross_sectional_median_MAD_zscore_clip_5_missing_to_cross_sectional_median | trend_momentum | predictor | 0.0400 | 51 | `vol_managed_momentum__csz` |
| `vol_managed_momentum__missing` | `vol_managed_momentum` | raw_missing_indicator | data_quality | quality_flag | 0.0000 | 2 | `tsmom_60__missing` |
| `idiosyncratic_volatility_60__csz` | `idiosyncratic_volatility_60` | same_date_cross_sectional_median_MAD_zscore_clip_5_missing_to_cross_sectional_median | volatility_risk | risk | 0.0289 | 52 | `idiosyncratic_volatility_60__csz` |
| `idiosyncratic_volatility_60__missing` | `idiosyncratic_volatility_60` | raw_missing_indicator | data_quality | quality_flag | 0.0000 | 53 | `idiosyncratic_volatility_60__missing` |
| `term_structure_slope__csz` | `term_structure_slope` | same_date_cross_sectional_median_MAD_zscore_clip_5_missing_to_cross_sectional_median | term_structure | predictor | 0.1922 | 54 | `term_structure_slope__csz` |
| `term_structure_slope__missing` | `term_structure_slope` | raw_missing_indicator | data_quality | quality_flag | 0.0000 | 16 | `curve_curvature__missing` |
| `term_structure_momentum__csz` | `term_structure_momentum` | same_date_cross_sectional_median_MAD_zscore_clip_5_missing_to_cross_sectional_median | term_structure | predictor | 0.2663 | 55 | `term_structure_momentum__csz` |
| `term_structure_momentum__missing` | `term_structure_momentum` | raw_missing_indicator | data_quality | quality_flag | 0.0000 | 56 | `term_structure_momentum__missing` |
| `term_structure_acceleration__csz` | `term_structure_acceleration` | same_date_cross_sectional_median_MAD_zscore_clip_5_missing_to_cross_sectional_median | term_structure | predictor | 0.3235 | 57 | `term_structure_acceleration__csz` |
| `term_structure_acceleration__missing` | `term_structure_acceleration` | raw_missing_indicator | data_quality | quality_flag | 0.0000 | 58 | `term_structure_acceleration__missing` |
| `basis_surprise__csz` | `basis_surprise` | same_date_cross_sectional_median_MAD_zscore_clip_5_missing_to_cross_sectional_median | basis_spot | predictor | 0.2043 | 59 | `basis_surprise__csz` |
| `basis_surprise__missing` | `basis_surprise` | raw_missing_indicator | data_quality | quality_flag | 0.0000 | 12 | `basis_fresh__missing` |
| `inventory_acceleration__csz` | `inventory_acceleration` | same_date_cross_sectional_median_MAD_zscore_clip_5_missing_to_cross_sectional_median | inventory_fundamental | predictor | 0.6622 | 60 | `inventory_acceleration__csz` |
| `inventory_acceleration__missing` | `inventory_acceleration` | raw_missing_indicator | data_quality | quality_flag | 0.0000 | 23 | `inventory_change_5__missing` |
| `inventory_price_divergence__csz` | `inventory_price_divergence` | same_date_cross_sectional_median_MAD_zscore_clip_5_missing_to_cross_sectional_median | information_news | predictor | 0.6650 | 22 | `inventory_change_5__csz` |
| `inventory_price_divergence__missing` | `inventory_price_divergence` | raw_missing_indicator | data_quality | quality_flag | 0.0000 | 23 | `inventory_change_5__missing` |
| `oi_surprise_60__csz` | `oi_surprise_60` | same_date_cross_sectional_median_MAD_zscore_clip_5_missing_to_cross_sectional_median | trading_liquidity | conditional | 0.0144 | 61 | `oi_surprise_60__csz` |
| `oi_surprise_60__missing` | `oi_surprise_60` | raw_missing_indicator | data_quality | quality_flag | 0.0000 | 31 | `oi_growth_20__missing` |
| `volume_price_divergence_20__csz` | `volume_price_divergence_20` | same_date_cross_sectional_median_MAD_zscore_clip_5_missing_to_cross_sectional_median | trading_liquidity | predictor | 0.0236 | 62 | `volume_price_divergence_20__csz` |
| `volume_price_divergence_20__missing` | `volume_price_divergence_20` | raw_missing_indicator | data_quality | quality_flag | 0.0000 | 33 | `price_oi_interaction_20__missing` |
| `momentum_carry_interaction__csz` | `momentum_carry_interaction` | same_date_cross_sectional_median_MAD_zscore_clip_5_missing_to_cross_sectional_median | factor_interaction | predictor | 0.0501 | 63 | `momentum_carry_interaction__csz` |
| `momentum_carry_interaction__missing` | `momentum_carry_interaction` | raw_missing_indicator | data_quality | quality_flag | 0.0000 | 64 | `momentum_carry_interaction__missing` |
| `nonlinear_tightness__csz` | `nonlinear_tightness` | same_date_cross_sectional_median_MAD_zscore_clip_5_missing_to_cross_sectional_median | factor_interaction | predictor | 0.2944 | 28 | `tightness_composite__csz` |
| `nonlinear_tightness__missing` | `nonlinear_tightness` | raw_missing_indicator | data_quality | quality_flag | 0.0000 | 29 | `tightness_composite__missing` |
| `correlation_risk_20_120__csz` | `correlation_risk_20_120` | same_date_cross_sectional_median_MAD_zscore_clip_5_missing_to_cross_sectional_median | volatility_risk | risk | 0.0681 | 65 | `correlation_risk_20_120__csz` |
| `correlation_risk_20_120__missing` | `correlation_risk_20_120` | raw_missing_indicator | data_quality | quality_flag | 0.0000 | 6 | `multi_horizon_trend__missing` |

# 附录C：逐因子完整证据与经济分析

## 因子家族分析

### 趋势与动量

趋势与动量共6个概念因子、6个可计算，平均Rank IC -0.0046。家族内开发期Rank IC最高的是`trend_efficiency_60`（0.0050）；B类0个、F类0个。

传统趋势、横截面动量和动量质量在开发期方向偏弱，但部分指标在2025转正，表现更像状态依赖信号而非稳定无条件Alpha。

### 期限结构与Carry

期限结构与Carry共6个概念因子、6个可计算，平均Rank IC 0.0108。家族内开发期Rank IC最高的是`carry_annualized`（0.0293）；B类2个、F类0个。

该家族是当前最清晰的经济信息源。Carry在开发期和2025均为正，整曲线斜率的Rank IC也保持正向；曲线变化和加速度则明显弱于曲线水平。

### 基差与现货

基差与现货共4个概念因子、4个可计算，平均Rank IC 0.0087。家族内开发期Rank IC最高的是`basis_fresh`（0.0309）；B类1个、F类0个。

新鲜基差在开发期的ICIR和组合收益较好，但现货源止于2023年，2025无法确认，因此不能据此升级为核心Alpha。

### 价值与均值回复

价值与均值回复共3个概念因子、3个可计算，平均Rank IC 0.0057。家族内开发期Rank IC最高的是`short_reversal_5`（0.0182）；B类0个、F类0个。


### 库存与供需

库存与供需共5个概念因子、5个可计算，平均Rank IC 0.0183。家族内开发期Rank IC最高的是`inventory_acceleration`（0.0316）；B类3个、F类0个。

仓单紧张度的覆盖和稳定性优于库存类变量；库存加速度有预测迹象，但库存覆盖仅约三分之一，解释时必须防止少数品种主导。

### 信息冲击与反应

信息冲击与反应共8个概念因子、7个可计算，平均Rank IC 0.0112。家族内开发期Rank IC最高的是`inventory_price_divergence`（0.0315）；B类2个、F类1个。

信息背离与库存—价格背离在开发期较强，但2025明显反向；这提示当前代理新息并非市场一致预期，可能混合了慢频数据和价格趋势。

### 交易行为与流动性

交易行为与流动性共9个概念因子、7个可计算，平均Rank IC 0.0053。家族内开发期Rank IC最高的是`oi_growth_20`（0.0145）；B类0个、F类2个。

成交、持仓和非流动性变量的主要价值是解释交易状态、换手和风险；当前没有证据支持把它们单独作为主要方向信号。

### 波动率与风险

波动率与风险共5个概念因子、5个可计算，平均Rank IC -0.0060。家族内开发期Rank IC最高的是`idiosyncratic_volatility_60`（0.0150）；B类0个、F类0个。

这些变量应进入风险和条件模块。2025某些风险变量表现突出，不代表它们已经成为方向Alpha。

### 产业链与相对价值

产业链与相对价值共6个概念因子、1个可计算，平均Rank IC -0.0109。家族内开发期Rank IC最高的是`sector_relative_value_60`（-0.0109）；B类0个、F类5个。

当前最有潜在差异化价值，但生产配比、上下游映射和成本口径不足，多数概念因子只能列为F类，不能用价格代理冒充真实产业链信息。

### 市场与品种状态

市场与品种状态家族当前没有可评价因子；2个概念因子因数据或经济映射不足列为F类。


### 元因子

元因子共1个概念因子、1个可计算，但变量在同一日期没有足够横截面差异，不适用横截面Rank IC评价，应仅作为跨时间条件变量使用。


### 非线性与交互

非线性与交互共2个概念因子、2个可计算，平均Rank IC 0.0143。家族内开发期Rank IC最高的是`nonlinear_tightness`（0.0184）；B类0个、F类0个。


## 57个概念因子结果

| ID | 因子 | 家族 | 可计算 | 覆盖率 | IC | ICIR | Rank IC | Rank ICIR | 胜率 | FDR q | 分组多空 | 年化 | 夏普 | 年度正向率 | 最大相关 | 增量Rank IC | 分类 | 进入ML |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 时间序列动量 | 趋势与动量 | 是 | 99.9% | -0.0047 | -0.0167 | -0.0069 | -0.0256 | 49.2% | 0.7152 | -0.0005 | -3.4% | -0.2931 | 33.3% | 0.8984 | 0.004718 | E | 是 |
| 2 | 横截面动量 | 趋势与动量 | 是 | 99.9% | -0.0051 | -0.0169 | -0.0087 | -0.0302 | 49.7% | 0.6855 | -0.0003 | -3.1% | -0.2278 | 33.3% | 0.8758 | -0.002059 | E | 是 |
| 3 | 趋势效率 | 趋势与动量 | 是 | 99.9% | 0.0088 | 0.0343 | 0.0050 | 0.0205 | 50.9% | 0.7510 | 0.0005 | -0.5% | -0.0101 | 66.7% | 0.1757 | -0.002257 | E | 是 |
| 4 | 多周期趋势一致性 | 趋势与动量 | 是 | 99.7% | -0.0055 | -0.0217 | -0.0099 | -0.0389 | 49.5% | 0.6421 | -0.0001 | -2.2% | -0.2324 | 33.3% | 0.6187 | -0.000324 | E | 是 |
| 5 | Carry因子 | 期限结构与Carry | 是 | 99.9% | 0.0301 | 0.1026 | 0.0293 | 0.1118 | 54.7% | 0.1931 | 0.0018 | 3.2% | 0.3689 | 83.3% | 0.6994 | 0.010541 | B | 是 |
| 6 | Carry变化 | 期限结构与Carry | 是 | 99.8% | -0.0266 | -0.1075 | -0.0090 | -0.0416 | 48.0% | 0.6421 | -0.0018 | -6.9% | -0.7769 | 33.3% | 0.4466 | 0.007296 | E | 是 |
| 7 | 基差 | 基差与现货 | 是 | 74.9% | 0.0333 | 0.1219 | 0.0309 | 0.1288 | 55.1% | 0.1926 | 0.0030 | 6.7% | 0.7411 | 66.7% | 0.5521 | 0.004888 | B | 是 |
| 8 | 基差冲击 | 基差与现货 | 是 | 74.2% | 0.0031 | 0.0129 | 0.0012 | 0.0048 | 50.2% | 0.8890 | 0.0003 | -2.8% | -0.3163 | 50.0% | 0.4937 | -0.002618 | E | 是 |
| 9 | 基差动量 | 基差与现货 | 是 | 74.2% | -0.0113 | -0.0373 | -0.0052 | -0.0200 | 48.8% | 0.7696 | -0.0002 | -1.9% | -0.1603 | 50.0% | 0.6349 | 0.007194 | E | 是 |
| 10 | 期限结构曲率 | 期限结构与Carry | 是 | 92.7% | -0.0005 | -0.0021 | -0.0076 | -0.0352 | 48.6% | 0.6421 | -0.0016 | -5.2% | -0.6619 | 33.3% | 0.5262 | 0.002604 | E | 是 |
| 11 | 价值因子 | 价值与均值回复 | 是 | 99.7% | 0.0046 | 0.0165 | 0.0127 | 0.0459 | 50.8% | 0.6421 | -0.0007 | -2.8% | -0.2333 | 66.7% | 0.6617 | -0.001203 | E | 是 |
| 12 | 短期反转 | 价值与均值回复 | 是 | 100.0% | 0.0216 | 0.0711 | 0.0182 | 0.0664 | 52.6% | 0.4592 | 0.0011 | -0.7% | -0.0078 | 83.3% | 0.3165 | -0.009521 | E | 是 |
| 13 | 库存稀缺因子 | 库存与供需 | 是 | 26.6% | -0.0087 | -0.0237 | -0.0034 | -0.0095 | 48.5% | 0.8890 | -0.0007 | 0.2% | 0.0839 | 50.0% | 0.3308 | -0.004522 | E | 是 |
| 14 | 库存变化因子 | 库存与供需 | 是 | 33.4% | 0.0138 | 0.0453 | 0.0211 | 0.0642 | 51.2% | 0.4592 | 0.0035 | 5.9% | 0.6564 | 83.3% | 0.9289 | 0.000470 | B | 是 |
| 15 | 库存意外／库存新息 | 信息冲击与反应 | 是 | 5.5% | -0.0289 | -0.0740 | -0.0182 | -0.0466 | 45.6% | 0.6421 | -0.0061 | -13.4% | -0.6877 | 50.0% | 0.2048 | -0.000606 | E | 是 |
| 16 | 仓单紧张因子 | 库存与供需 | 是 | 94.0% | 0.0224 | 0.1043 | 0.0236 | 0.1120 | 54.2% | 0.1926 | 0.0022 | 3.8% | 0.5240 | 83.3% | 0.6232 | 0.007944 | B | 是 |
| 17 | 供需紧张度 | 库存与供需 | 是 | 78.3% | 0.0097 | 0.0333 | 0.0184 | 0.0677 | 52.6% | 0.4592 | 0.0009 | 0.0% | 0.0468 | 66.7% | 0.9403 | -0.001898 | E | 是 |
| 18 | 持仓量增长 | 交易行为与流动性 | 是 | 100.0% | 0.0075 | 0.0349 | 0.0145 | 0.0675 | 51.9% | 0.4592 | 0.0013 | 0.7% | 0.1328 | 66.7% | 0.2984 | — | C | 是 |
| 19 | 价格—持仓量交互 | 交易行为与流动性 | 是 | 100.0% | 0.0013 | 0.0049 | 0.0055 | 0.0242 | 51.0% | 0.7152 | 0.0014 | 1.1% | 0.1666 | 66.7% | 0.1783 | -0.003763 | E | 是 |
| 20 | 套保压力 | 交易行为与流动性 | 否 | 0.0% | — | — | — | — | — | — | — | — | — | — | — | — | F | 否 |
| 21 | 成交量冲击 | 交易行为与流动性 | 是 | 100.0% | 0.0079 | 0.0355 | 0.0063 | 0.0284 | 51.2% | 0.6421 | 0.0001 | -3.7% | -0.4510 | 66.7% | 0.4265 | — | C | 是 |
| 22 | Amihud非流动性 | 交易行为与流动性 | 是 | 99.9% | 0.0134 | 0.0565 | 0.0116 | 0.0540 | 52.7% | 0.5709 | 0.0007 | 1.4% | 0.2059 | 66.7% | 0.2381 | — | D | 是 |
| 23 | 成交量—持仓量换手率 | 交易行为与流动性 | 是 | 100.0% | 0.0077 | 0.0283 | 0.0073 | 0.0300 | 50.7% | 0.6855 | 0.0010 | 1.2% | 0.1814 | 66.7% | 0.5012 | — | C | 是 |
| 24 | 已实现波动率 | 波动率与风险 | 是 | 100.0% | 0.0137 | 0.0397 | 0.0128 | 0.0405 | 51.2% | 0.6421 | 0.0007 | -0.1% | 0.0540 | 83.3% | 0.9114 | — | D | 是 |
| 25 | 低波动因子 | 波动率与风险 | 是 | 100.0% | -0.0188 | -0.0590 | -0.0128 | -0.0405 | 48.8% | 0.6421 | -0.0007 | -3.2% | -0.1955 | 16.7% | 0.9114 | — | D | 是 |
| 26 | 已实现偏度 | 波动率与风险 | 是 | 99.9% | -0.0087 | -0.0393 | -0.0144 | -0.0639 | 45.9% | 0.4592 | -0.0009 | -3.8% | -0.4343 | 33.3% | 0.2222 | — | D | 是 |
| 27 | 季节性因子 | 价值与均值回复 | 是 | 96.3% | -0.0145 | -0.0549 | -0.0139 | -0.0553 | 48.5% | 0.4592 | 0.0004 | -3.7% | -0.3803 | 33.3% | 0.0539 | 0.005089 | E | 是 |
| 28 | 产业链残差 | 产业链与相对价值 | 否 | 0.0% | — | — | — | — | — | — | — | — | — | — | — | — | F | 否 |
| 29 | 相对价值残差 | 产业链与相对价值 | 是 | 99.9% | -0.0125 | -0.0450 | -0.0109 | -0.0436 | 47.5% | 0.6421 | -0.0012 | -4.4% | -0.4718 | 16.7% | 0.7875 | -0.001796 | E | 是 |
| 30 | 状态条件因子 | 市场与品种状态 | 否 | 0.0% | — | — | — | — | — | — | — | — | — | — | — | — | F | 否 |
| 31 | 单品种状态交互因子 | 市场与品种状态 | 否 | 0.0% | — | — | — | — | — | — | — | — | — | — | — | — | F | 否 |
| 32 | 因子动量 | 元因子 | 是 | 100.0% | — | — | — | — | 0.0% | — | — | — | — | 0.0% | 0.0953 | — | C | 是 |
| 33 | 基本面综合新息代理 | 信息冲击与反应 | 是 | 74.7% | 0.0063 | 0.0286 | 0.0049 | 0.0225 | 51.2% | 0.7510 | 0.0005 | -1.1% | -0.1355 | 33.3% | 0.6875 | -0.001600 | E | 是 |
| 34 | 基本面反应不足代理 | 信息冲击与反应 | 是 | 74.7% | 0.0171 | 0.0631 | 0.0139 | 0.0550 | 51.3% | 0.4592 | 0.0024 | 2.9% | 0.3684 | 66.7% | 0.6791 | -0.001535 | E | 是 |
| 35 | 信息一致性 | 信息冲击与反应 | 是 | 78.3% | 0.0055 | 0.0190 | 0.0182 | 0.0667 | 53.7% | 0.4592 | 0.0005 | 0.4% | 0.0880 | 66.7% | 0.9043 | -0.002822 | E | 是 |
| 36 | 信息背离 | 信息冲击与反应 | 是 | 78.3% | 0.0189 | 0.0663 | 0.0257 | 0.0939 | 54.7% | 0.2816 | 0.0018 | 2.7% | 0.3511 | 66.7% | 0.7250 | 0.002259 | B | 是 |
| 37 | 成本传导缺口 | 产业链与相对价值 | 否 | 0.0% | — | — | — | — | — | — | — | — | — | — | — | — | F | 否 |
| 38 | 加工利润异常 | 产业链与相对价值 | 否 | 0.0% | — | — | — | — | — | — | — | — | — | — | — | — | F | 否 |
| 39 | 仓单新息 | 信息冲击与反应 | 是 | 96.9% | -0.0002 | -0.0009 | 0.0023 | 0.0112 | 48.8% | 0.8841 | 0.0004 | -0.8% | -0.0670 | 66.7% | 0.6875 | -0.002907 | E | 是 |
| 40 | 产业链传导残差 | 产业链与相对价值 | 否 | 0.0% | — | — | — | — | — | — | — | — | — | — | — | — | F | 否 |
| 41 | 动量质量因子 | 趋势与动量 | 是 | 99.9% | -0.0042 | -0.0152 | -0.0070 | -0.0261 | 49.4% | 0.7152 | -0.0004 | -2.8% | -0.2355 | 33.3% | 0.8984 | -0.003612 | E | 是 |
| 42 | 波动率管理动量 | 趋势与动量 | 是 | 99.9% | -0.0025 | -0.0127 | -0.0004 | -0.0018 | 51.2% | 0.9685 | 0.0010 | 1.1% | 0.1932 | 33.3% | 0.5785 | -0.003381 | E | 是 |
| 43 | 特质波动因子 | 波动率与风险 | 是 | 99.9% | 0.0166 | 0.0549 | 0.0150 | 0.0570 | 53.4% | 0.5709 | 0.0008 | 1.1% | 0.1578 | 66.7% | 0.6450 | — | D | 是 |
| 44 | 期限结构斜率 | 期限结构与Carry | 是 | 92.7% | 0.0255 | 0.0830 | 0.0241 | 0.0867 | 54.7% | 0.4274 | 0.0019 | 3.6% | 0.3612 | 66.7% | 0.6994 | 0.005513 | B | 是 |
| 45 | 期限结构动量 | 期限结构与Carry | 是 | 89.3% | 0.0162 | 0.0554 | 0.0137 | 0.0534 | 53.9% | 0.5709 | 0.0007 | 0.2% | 0.0673 | 83.3% | 0.6486 | 0.005802 | E | 是 |
| 46 | 期限结构加速度 | 期限结构与Carry | 是 | 86.7% | 0.0130 | 0.0455 | 0.0142 | 0.0558 | 52.6% | 0.5529 | 0.0003 | -3.2% | -0.2598 | 83.3% | 0.6486 | 0.004712 | E | 是 |
| 47 | 基差意外 | 基差与现货 | 是 | 74.5% | 0.0075 | 0.0312 | 0.0078 | 0.0331 | 50.6% | 0.6651 | 0.0009 | -3.3% | -0.3617 | 50.0% | 0.6349 | -0.004012 | E | 是 |
| 48 | 库存加速度 | 库存与供需 | 是 | 33.3% | 0.0192 | 0.0599 | 0.0316 | 0.0954 | 54.0% | 0.1926 | 0.0041 | 4.3% | 0.4331 | 83.3% | 0.4045 | -0.000952 | B | 是 |
| 49 | 库存—价格背离 | 信息冲击与反应 | 是 | 33.4% | 0.0212 | 0.0688 | 0.0315 | 0.0938 | 53.3% | 0.2316 | 0.0033 | 4.7% | 0.5308 | 83.3% | 0.9289 | 0.001449 | B | 是 |
| 50 | 供需冲击因子 | 信息冲击与反应 | 否 | 0.0% | — | — | — | — | — | — | — | — | — | — | — | — | F | 否 |
| 51 | 拥挤度因子 | 交易行为与流动性 | 否 | 0.0% | — | — | — | — | — | — | — | — | — | — | — | — | F | 否 |
| 52 | 持仓意外因子 | 交易行为与流动性 | 是 | 100.0% | 0.0007 | 0.0033 | 0.0006 | 0.0029 | 49.2% | 0.9455 | 0.0011 | -1.1% | -0.1030 | 66.7% | 0.2227 | — | C | 是 |
| 53 | 量价背离因子 | 交易行为与流动性 | 是 | 100.0% | -0.0163 | -0.0585 | -0.0086 | -0.0337 | 48.0% | 0.6421 | -0.0010 | -5.3% | -0.5503 | 33.3% | 0.4265 | -0.014203 | E | 是 |
| 54 | 因子交互因子 | 非线性与交互 | 是 | 99.8% | 0.0182 | 0.0676 | 0.0101 | 0.0464 | 49.9% | 0.6421 | 0.0008 | 0.2% | 0.0624 | 66.7% | — | — | E | 是 |
| 55 | 非线性基本面因子 | 非线性与交互 | 是 | 78.3% | 0.0026 | 0.0087 | 0.0184 | 0.0677 | 52.6% | 0.4592 | 0.0009 | 0.0% | 0.0468 | 66.7% | — | — | E | 是 |
| 56 | 商品网络因子 | 产业链与相对价值 | 否 | 0.0% | — | — | — | — | — | — | — | — | — | — | — | — | F | 否 |
| 57 | 相关性风险因子 | 波动率与风险 | 是 | 99.7% | -0.0288 | -0.1214 | -0.0307 | -0.1250 | 44.0% | 0.1926 | -0.0027 | -8.7% | -1.0091 | 0.0% | 0.1154 | — | D | 是 |

## 逐因子经济解释与证据

### #1 时间序列动量：E类

- 经济原理：波动率调整后的中期趋势延续。
- 数据与实现：已计算；数据源为实际主力合约收益；开发期覆盖率99.9%，覆盖30个品种。
- 开发期证据：开发期Rank IC -0.0069、Rank ICIR -0.0256、胜率49.2%、FDR q 0.7152；五分组多空收益-0.0005，5bps后年化-3.4%、夏普-0.2931；正向年份率33.3%。
- 2025确认：2025 Rank IC 0.0252、5bps后年化2.4%、夏普0.3121；与开发期Rank IC方向不一致。
- 结论：开发期无条件预测、成本后收益或跨期稳定性不足；固定Ridge增量Rank IC为0.004718。机器学习输入：进入。

### #2 横截面动量：E类

- 经济原理：同日商品过去收益相对强弱。
- 数据与实现：已计算；数据源为实际主力合约收益；开发期覆盖率99.9%，覆盖30个品种。
- 开发期证据：开发期Rank IC -0.0087、Rank ICIR -0.0302、胜率49.7%、FDR q 0.6855；五分组多空收益-0.0003，5bps后年化-3.1%、夏普-0.2278；正向年份率33.3%。
- 2025确认：2025 Rank IC 0.0105、5bps后年化-3.0%、夏普-0.2155；与开发期Rank IC方向不一致。
- 结论：开发期无条件预测、成本后收益或跨期稳定性不足；固定Ridge增量Rank IC为-0.002059。机器学习输入：进入。

### #3 趋势效率：E类

- 经济原理：价格路径的单向和平滑程度。
- 数据与实现：已计算；数据源为实际主力合约收益；开发期覆盖率99.9%，覆盖30个品种。
- 开发期证据：开发期Rank IC 0.0050、Rank ICIR 0.0205、胜率50.9%、FDR q 0.7510；五分组多空收益0.0005，5bps后年化-0.5%、夏普-0.0101；正向年份率66.7%。
- 2025确认：2025 Rank IC 0.0140、5bps后年化-7.8%、夏普-1.0067；与开发期Rank IC同方向。
- 结论：开发期无条件预测、成本后收益或跨期稳定性不足；固定Ridge增量Rank IC为-0.002257。机器学习输入：进入。

### #4 多周期趋势一致性：E类

- 经济原理：20/60/120日趋势方向一致程度。
- 数据与实现：已计算；数据源为实际主力合约收益；开发期覆盖率99.7%，覆盖30个品种。
- 开发期证据：开发期Rank IC -0.0099、Rank ICIR -0.0389、胜率49.5%、FDR q 0.6421；五分组多空收益-0.0001，5bps后年化-2.2%、夏普-0.2324；正向年份率33.3%。
- 2025确认：2025 Rank IC 0.0497、5bps后年化5.6%、夏普0.7248；与开发期Rank IC方向不一致。
- 结论：开发期无条件预测、成本后收益或跨期稳定性不足；固定Ridge增量Rank IC为-0.000324。机器学习输入：进入。

### #5 Carry因子：B类

- 经济原理：完整可交易曲线最近两个合约之间的年化期限结构补偿。
- 数据与实现：已计算；数据源为完整实际合约曲线、收盘价、持仓量、成交量与到期日；开发期覆盖率99.9%，覆盖30个品种。
- 开发期证据：开发期Rank IC 0.0293、Rank ICIR 0.1118、胜率54.7%、FDR q 0.1931；五分组多空收益0.0018，5bps后年化3.2%、夏普0.3689；正向年份率83.3%。
- 2025确认：2025 Rank IC 0.0323、5bps后年化2.9%、夏普0.3220；与开发期Rank IC同方向。
- 结论：存在正向预测和组合证据，但FDR q为0.1931，未达到10%核心门槛；固定Ridge增量Rank IC为0.010541。机器学习输入：进入。

### #6 Carry变化：E类

- 经济原理：期限结构二十日边际变化。
- 数据与实现：已计算；数据源为主力与次近月收盘价、到期日；开发期覆盖率99.8%，覆盖30个品种。
- 开发期证据：开发期Rank IC -0.0090、Rank ICIR -0.0416、胜率48.0%、FDR q 0.6421；五分组多空收益-0.0018，5bps后年化-6.9%、夏普-0.7769；正向年份率33.3%。
- 2025确认：2025 Rank IC -0.0516、5bps后年化-12.0%、夏普-2.0268；与开发期Rank IC同方向。
- 结论：开发期无条件预测、成本后收益或跨期稳定性不足；固定Ridge增量Rank IC为0.007296。机器学习输入：进入。

### #7 基差：B类

- 经济原理：新鲜现货相对主力期货的强弱。
- 数据与实现：已计算；数据源为现货价格、观察日、主力收盘价；开发期覆盖率74.9%，覆盖27个品种。
- 开发期证据：开发期Rank IC 0.0309、Rank ICIR 0.1288、胜率55.1%、FDR q 0.1926；五分组多空收益0.0030，5bps后年化6.7%、夏普0.7411；正向年份率66.7%。
- 2025确认：2025没有可评价样本，不能完成封存确认。
- 结论：存在正向预测和组合证据，但FDR q为0.1926，未达到10%核心门槛；固定Ridge增量Rank IC为0.004888。机器学习输入：进入。

### #8 基差冲击：E类

- 经济原理：基差日变化相对历史条件分布的异常。
- 数据与实现：已计算；数据源为现货价格、观察日、主力收盘价；开发期覆盖率74.2%，覆盖27个品种。
- 开发期证据：开发期Rank IC 0.0012、Rank ICIR 0.0048、胜率50.2%、FDR q 0.8890；五分组多空收益0.0003，5bps后年化-2.8%、夏普-0.3163；正向年份率50.0%。
- 2025确认：2025没有可评价样本，不能完成封存确认。
- 结论：开发期无条件预测、成本后收益或跨期稳定性不足；固定Ridge增量Rank IC为-0.002618。机器学习输入：进入。

### #9 基差动量：E类

- 经济原理：新鲜可用基差的二十日边际变化。
- 数据与实现：已计算；数据源为延迟一交易日可用的现货与主力收盘价；开发期覆盖率74.2%，覆盖27个品种。
- 开发期证据：开发期Rank IC -0.0052、Rank ICIR -0.0200、胜率48.8%、FDR q 0.7696；五分组多空收益-0.0002，5bps后年化-1.9%、夏普-0.1603；正向年份率50.0%。
- 2025确认：2025没有可评价样本，不能完成封存确认。
- 结论：开发期无条件预测、成本后收益或跨期稳定性不足；固定Ridge增量Rank IC为0.007194。机器学习输入：进入。

### #10 期限结构曲率：E类

- 经济原理：曲线第二个合约相对第一和第三个合约线性插值的弯曲程度。
- 数据与实现：已计算；数据源为通过流动性和期限硬门的前三个实际合约；开发期覆盖率92.7%，覆盖30个品种。
- 开发期证据：开发期Rank IC -0.0076、Rank ICIR -0.0352、胜率48.6%、FDR q 0.6421；五分组多空收益-0.0016，5bps后年化-5.2%、夏普-0.6619；正向年份率33.3%。
- 2025确认：2025 Rank IC 0.0087、5bps后年化-7.8%、夏普-1.4618；与开发期Rank IC方向不一致。
- 结论：开发期无条件预测、成本后收益或跨期稳定性不足；固定Ridge增量Rank IC为0.002604。机器学习输入：进入。

### #11 价值因子：E类

- 经济原理：价格相对一年滚动历史锚点的反向偏离。
- 数据与实现：已计算；数据源为主力收盘价；开发期覆盖率99.7%，覆盖30个品种。
- 开发期证据：开发期Rank IC 0.0127、Rank ICIR 0.0459、胜率50.8%、FDR q 0.6421；五分组多空收益-0.0007，5bps后年化-2.8%、夏普-0.2333；正向年份率66.7%。
- 2025确认：2025 Rank IC -0.0477、5bps后年化-9.0%、夏普-1.0020；与开发期Rank IC方向不一致。
- 结论：开发期无条件预测、成本后收益或跨期稳定性不足；固定Ridge增量Rank IC为-0.001203。机器学习输入：进入。

### #12 短期反转：E类

- 经济原理：过去五日过度反应后的修复。
- 数据与实现：已计算；数据源为实际主力合约收益；开发期覆盖率100.0%，覆盖30个品种。
- 开发期证据：开发期Rank IC 0.0182、Rank ICIR 0.0664、胜率52.6%、FDR q 0.4592；五分组多空收益0.0011，5bps后年化-0.7%、夏普-0.0078；正向年份率83.3%。
- 2025确认：2025 Rank IC 0.0045、5bps后年化5.5%、夏普0.6092；与开发期Rank IC同方向。
- 结论：开发期无条件预测、成本后收益或跨期稳定性不足；固定Ridge增量Rank IC为-0.009521。机器学习输入：进入。

### #13 库存稀缺因子：E类

- 经济原理：库存相对历史同期的稀缺程度。
- 数据与实现：已计算；数据源为库存、观察日与数据年龄；开发期覆盖率26.6%，覆盖8个品种。
- 开发期证据：开发期Rank IC -0.0034、Rank ICIR -0.0095、胜率48.5%、FDR q 0.8890；五分组多空收益-0.0007，5bps后年化0.2%、夏普0.0839；正向年份率50.0%。
- 2025确认：2025 Rank IC -0.1303、5bps后年化-27.0%、夏普-2.1761；与开发期Rank IC同方向。
- 结论：开发期无条件预测、成本后收益或跨期稳定性不足；固定Ridge增量Rank IC为-0.004522。机器学习输入：进入。

### #14 库存变化因子：B类

- 经济原理：五日库存下降所表示的边际趋紧。
- 数据与实现：已计算；数据源为库存、观察日与数据年龄；开发期覆盖率33.4%，覆盖10个品种。
- 开发期证据：开发期Rank IC 0.0211、Rank ICIR 0.0642、胜率51.2%、FDR q 0.4592；五分组多空收益0.0035，5bps后年化5.9%、夏普0.6564；正向年份率83.3%。
- 2025确认：2025 Rank IC -0.0557、5bps后年化-9.0%、夏普-1.2309；与开发期Rank IC方向不一致。
- 结论：存在正向预测和组合证据，但FDR q为0.4592，未达到10%核心门槛；固定Ridge增量Rank IC为0.000470。机器学习输入：进入。

### #15 库存意外／库存新息：E类

- 经济原理：库存相对仅用历史信息形成的条件预期的标准化意外。
- 数据与实现：已计算；数据源为库存、观察日与数据年龄；开发期覆盖率5.5%，覆盖8个品种，属于低频事件特征。
- 开发期证据：开发期Rank IC -0.0182、Rank ICIR -0.0466、胜率45.6%、FDR q 0.6421；五分组多空收益-0.0061，5bps后年化-13.4%、夏普-0.6877；正向年份率50.0%。
- 2025确认：2025 Rank IC -0.0902、5bps后年化16.4%、夏普1.7779；与开发期Rank IC同方向。
- 结论：开发期无条件预测、成本后收益或跨期稳定性不足；固定Ridge增量Rank IC为-0.000606。机器学习输入：进入。

### #16 仓单紧张因子：B类

- 经济原理：注册仓单相对历史同期的紧张程度。
- 数据与实现：已计算；数据源为注册仓单及观察日；开发期覆盖率94.0%，覆盖30个品种。
- 开发期证据：开发期Rank IC 0.0236、Rank ICIR 0.1120、胜率54.2%、FDR q 0.1926；五分组多空收益0.0022，5bps后年化3.8%、夏普0.5240；正向年份率83.3%。
- 2025确认：2025 Rank IC 0.0653、5bps后年化3.7%、夏普0.7732；与开发期Rank IC同方向。
- 结论：存在正向预测和组合证据，但FDR q为0.1926，未达到10%核心门槛；固定Ridge增量Rank IC为0.007944。机器学习输入：进入。

### #17 供需紧张度：E类

- 经济原理：仓单、库存、基差和Carry的等权看多方向组合。
- 数据与实现：已计算；数据源为库存、仓单、现货、期限结构；开发期覆盖率78.3%，覆盖28个品种。
- 开发期证据：开发期Rank IC 0.0184、Rank ICIR 0.0677、胜率52.6%、FDR q 0.4592；五分组多空收益0.0009，5bps后年化0.0%、夏普0.0468；正向年份率66.7%。
- 2025确认：2025 Rank IC -0.1157、5bps后年化-25.3%、夏普-2.1315；与开发期Rank IC方向不一致。
- 结论：开发期无条件预测、成本后收益或跨期稳定性不足；固定Ridge增量Rank IC为-0.001898。机器学习输入：进入。

### #18 持仓量增长：C类

- 经济原理：未平仓合约数量的二十日增长。
- 数据与实现：已计算；数据源为主力持仓量；开发期覆盖率100.0%，覆盖30个品种。
- 开发期证据：开发期Rank IC 0.0145、Rank ICIR 0.0675、胜率51.9%、FDR q 0.4592；五分组多空收益0.0013，5bps后年化0.7%、夏普0.1328；正向年份率66.7%。
- 2025确认：2025 Rank IC -0.0324、5bps后年化-7.9%、夏普-1.1621；与开发期Rank IC方向不一致。
- 结论：经济角色主要是调节其他信号、刻画交易状态或形成交互，不要求其单独产生稳定多空收益。机器学习输入：进入。

### #19 价格—持仓量交互：E类

- 经济原理：价格方向和新增持仓方向的联合变化。
- 数据与实现：已计算；数据源为实际主力合约收益、持仓量；开发期覆盖率100.0%，覆盖30个品种。
- 开发期证据：开发期Rank IC 0.0055、Rank ICIR 0.0242、胜率51.0%、FDR q 0.7152；五分组多空收益0.0014，5bps后年化1.1%、夏普0.1666；正向年份率66.7%。
- 2025确认：2025 Rank IC -0.0216、5bps后年化-5.5%、夏普-0.6958；与开发期Rank IC方向不一致。
- 结论：开发期无条件预测、成本后收益或跨期稳定性不足；固定Ridge增量Rank IC为-0.003763。机器学习输入：进入。

### #20 套保压力：F类

- 经济原理：商业交易者净持仓压力。
- 数据与实现：不可计算；data_final_raw没有商业/非商业交易者分类持仓。
- 开发期证据：没有生成统计结果，不能以零收益或零IC替代缺失证据。
- 2025确认：2025没有可评价样本，不能完成封存确认。
- 结论：缺失数据为：data_final_raw没有商业/非商业交易者分类持仓。当前只能保留概念定义，不能形成数值特征。机器学习输入：不进入。

### #21 成交量冲击：C类

- 经济原理：成交量相对历史正常水平的异常。
- 数据与实现：已计算；数据源为主力成交量；开发期覆盖率100.0%，覆盖30个品种。
- 开发期证据：开发期Rank IC 0.0063、Rank ICIR 0.0284、胜率51.2%、FDR q 0.6421；五分组多空收益0.0001，5bps后年化-3.7%、夏普-0.4510；正向年份率66.7%。
- 2025确认：2025 Rank IC -0.0288、5bps后年化-13.2%、夏普-1.9675；与开发期Rank IC方向不一致。
- 结论：经济角色主要是调节其他信号、刻画交易状态或形成交互，不要求其单独产生稳定多空收益。机器学习输入：进入。

### #22 Amihud非流动性：D类

- 经济原理：单位成交额对应的绝对价格变动。
- 数据与实现：已计算；数据源为实际主力合约收益、成交额；开发期覆盖率99.9%，覆盖30个品种。
- 开发期证据：开发期Rank IC 0.0116、Rank ICIR 0.0540、胜率52.7%、FDR q 0.5709；五分组多空收益0.0007，5bps后年化1.4%、夏普0.2059；正向年份率66.7%。
- 2025确认：2025 Rank IC 0.1109、5bps后年化23.4%、夏普2.8996；与开发期Rank IC同方向。
- 结论：主要用途是风险、波动、流动性或尾部暴露刻画，不能因某一年方向收益较高而改列Alpha。机器学习输入：进入。

### #23 成交量—持仓量换手率：C类

- 经济原理：相对存量持仓的交易活跃程度。
- 数据与实现：已计算；数据源为主力成交量、持仓量；开发期覆盖率100.0%，覆盖30个品种。
- 开发期证据：开发期Rank IC 0.0073、Rank ICIR 0.0300、胜率50.7%、FDR q 0.6855；五分组多空收益0.0010，5bps后年化1.2%、夏普0.1814；正向年份率66.7%。
- 2025确认：2025 Rank IC 0.0260、5bps后年化11.2%、夏普1.4979；与开发期Rank IC同方向。
- 结论：经济角色主要是调节其他信号、刻画交易状态或形成交互，不要求其单独产生稳定多空收益。机器学习输入：进入。

### #24 已实现波动率：D类

- 经济原理：过去二十日收益波动风险。
- 数据与实现：已计算；数据源为实际主力合约收益；开发期覆盖率100.0%，覆盖30个品种。
- 开发期证据：开发期Rank IC 0.0128、Rank ICIR 0.0405、胜率51.2%、FDR q 0.6421；五分组多空收益0.0007，5bps后年化-0.1%、夏普0.0540；正向年份率83.3%。
- 2025确认：2025 Rank IC -0.0196、5bps后年化-2.1%、夏普-0.1669；与开发期Rank IC方向不一致。
- 结论：主要用途是风险、波动、流动性或尾部暴露刻画，不能因某一年方向收益较高而改列Alpha。机器学习输入：进入。

### #25 低波动因子：D类

- 经济原理：横截面低波动品种得分更高。
- 数据与实现：已计算；数据源为实际主力合约收益；开发期覆盖率100.0%，覆盖30个品种。
- 开发期证据：开发期Rank IC -0.0128、Rank ICIR -0.0405、胜率48.8%、FDR q 0.6421；五分组多空收益-0.0007，5bps后年化-3.2%、夏普-0.1955；正向年份率16.7%。
- 2025确认：2025 Rank IC 0.0196、5bps后年化-0.8%、夏普-0.0293；与开发期Rank IC方向不一致。
- 结论：主要用途是风险、波动、流动性或尾部暴露刻画，不能因某一年方向收益较高而改列Alpha。机器学习输入：进入。

### #26 已实现偏度：D类

- 经济原理：过去六十日收益分布非对称性。
- 数据与实现：已计算；数据源为实际主力合约收益；开发期覆盖率99.9%，覆盖30个品种。
- 开发期证据：开发期Rank IC -0.0144、Rank ICIR -0.0639、胜率45.9%、FDR q 0.4592；五分组多空收益-0.0009，5bps后年化-3.8%、夏普-0.4343；正向年份率33.3%。
- 2025确认：2025 Rank IC 0.0116、5bps后年化-6.0%、夏普-0.7063；与开发期Rank IC方向不一致。
- 结论：主要用途是风险、波动、流动性或尾部暴露刻画，不能因某一年方向收益较高而改列Alpha。机器学习输入：进入。

### #27 季节性因子：E类

- 经济原理：过去三年近似同期五日收益的平均。
- 数据与实现：已计算；数据源为实际主力合约历史收益；开发期覆盖率96.3%，覆盖30个品种。
- 开发期证据：开发期Rank IC -0.0139、Rank ICIR -0.0553、胜率48.5%、FDR q 0.4592；五分组多空收益0.0004，5bps后年化-3.7%、夏普-0.3803；正向年份率33.3%。
- 2025确认：2025 Rank IC -0.0103、5bps后年化4.5%、夏普0.5031；与开发期Rank IC同方向。
- 结论：开发期无条件预测、成本后收益或跨期稳定性不足；固定Ridge增量Rank IC为0.005089。机器学习输入：进入。

### #28 产业链残差：F类

- 经济原理：相对明确上下游和替代品关系的状态偏离。
- 数据与实现：不可计算；未提供经确认的产业链邻接表和生产传导系数。
- 开发期证据：没有生成统计结果，不能以零收益或零IC替代缺失证据。
- 2025确认：2025没有可评价样本，不能完成封存确认。
- 结论：缺失数据为：未提供经确认的产业链邻接表和生产传导系数。当前只能保留概念定义，不能形成数值特征。机器学习输入：不进入。

### #29 相对价值残差：E类

- 经济原理：品种六十日收益相对板块共同收益的偏离。
- 数据与实现：已计算；数据源为实际主力合约收益、固定板块映射；开发期覆盖率99.9%，覆盖30个品种。
- 开发期证据：开发期Rank IC -0.0109、Rank ICIR -0.0436、胜率47.5%、FDR q 0.6421；五分组多空收益-0.0012，5bps后年化-4.4%、夏普-0.4718；正向年份率16.7%。
- 2025确认：2025 Rank IC 0.0262、5bps后年化-3.4%、夏普-0.3454；与开发期Rank IC方向不一致。
- 结论：开发期无条件预测、成本后收益或跨期稳定性不足；固定Ridge增量Rank IC为-0.001796。机器学习输入：进入。

### #30 状态条件因子：F类

- 经济原理：市场状态概率调节基础因子。
- 数据与实现：不可计算；当前协议未定义和冻结市场状态概率引擎。
- 开发期证据：没有生成统计结果，不能以零收益或零IC替代缺失证据。
- 2025确认：2025没有可评价样本，不能完成封存确认。
- 结论：缺失数据为：当前协议未定义和冻结市场状态概率引擎。当前只能保留概念定义，不能形成数值特征。机器学习输入：不进入。

### #31 单品种状态交互因子：F类

- 经济原理：品种自身状态调节基础因子。
- 数据与实现：不可计算；当前协议未定义和冻结单品种状态引擎。
- 开发期证据：没有生成统计结果，不能以零收益或零IC替代缺失证据。
- 2025确认：2025没有可评价样本，不能完成封存确认。
- 结论：缺失数据为：当前协议未定义和冻结单品种状态引擎。当前只能保留概念定义，不能形成数值特征。机器学习输入：不进入。

### #32 因子动量：C类

- 经济原理：四个冻结基础因子组合最近二十日已实现收益的持续性。
- 数据与实现：已计算；数据源为严格滞后一期的动量、Carry、反转和板块相对价值组合收益；开发期覆盖率100.0%，覆盖30个品种。
- 开发期证据：该变量在同一日期缺少足够横截面差异，因此单日横截面IC不可解释；它只能作为跨时间变化的条件变量评价。
- 2025确认：2025没有可评价样本，不能完成封存确认。
- 结论：经济角色主要是调节其他信号、刻画交易状态或形成交互，不要求其单独产生稳定多空收益。机器学习输入：进入。

### #33 基本面综合新息代理：E类

- 经济原理：库存、仓单和基差中至少两类历史模型意外的统一方向组合。
- 数据与实现：已计算；数据源为延迟一交易日可用的库存、仓单和现货；不是市场一致预期；开发期覆盖率74.7%，覆盖28个品种，属于低频事件特征。
- 开发期证据：开发期Rank IC 0.0049、Rank ICIR 0.0225、胜率51.2%、FDR q 0.7510；五分组多空收益0.0005，5bps后年化-1.1%、夏普-0.1355；正向年份率33.3%。
- 2025确认：2025没有可评价样本，不能完成封存确认。
- 结论：开发期无条件预测、成本后收益或跨期稳定性不足；固定Ridge增量Rank IC为-0.001600。机器学习输入：进入。

### #34 基本面反应不足代理：E类

- 经济原理：模型隐含基本面新息扣除首个可交易日价格反应后的未吸收部分。
- 数据与实现：已计算；数据源为基本面综合新息代理与主力收益；不是市场一致预期；开发期覆盖率74.7%，覆盖28个品种，属于低频事件特征。
- 开发期证据：开发期Rank IC 0.0139、Rank ICIR 0.0550、胜率51.3%、FDR q 0.4592；五分组多空收益0.0024，5bps后年化2.9%、夏普0.3684；正向年份率66.7%。
- 2025确认：2025没有可评价样本，不能完成封存确认。
- 结论：开发期无条件预测、成本后收益或跨期稳定性不足；固定Ridge增量Rank IC为-0.001535。机器学习输入：进入。

### #35 信息一致性：E类

- 经济原理：仓单、库存、基差和Carry方向一致时放大的综合信号。
- 数据与实现：已计算；数据源为库存、仓单、基差、Carry；开发期覆盖率78.3%，覆盖28个品种。
- 开发期证据：开发期Rank IC 0.0182、Rank ICIR 0.0667、胜率53.7%、FDR q 0.4592；五分组多空收益0.0005，5bps后年化0.4%、夏普0.0880；正向年份率66.7%。
- 2025确认：2025 Rank IC -0.1113、5bps后年化-26.4%、夏普-2.2022；与开发期Rank IC方向不一致。
- 结论：开发期无条件预测、成本后收益或跨期稳定性不足；固定Ridge增量Rank IC为-0.002822。机器学习输入：进入。

### #36 信息背离：B类

- 经济原理：供需基本面信号相对价格趋势的未反映差异。
- 数据与实现：已计算；数据源为供需紧张度与实际主力合约收益；开发期覆盖率78.3%，覆盖28个品种。
- 开发期证据：开发期Rank IC 0.0257、Rank ICIR 0.0939、胜率54.7%、FDR q 0.2816；五分组多空收益0.0018，5bps后年化2.7%、夏普0.3511；正向年份率66.7%。
- 2025确认：2025 Rank IC -0.1062、5bps后年化-20.9%、夏普-1.8061；与开发期Rank IC方向不一致。
- 结论：存在正向预测和组合证据，但FDR q为0.2816，未达到10%核心门槛；固定Ridge增量Rank IC为0.002259。机器学习输入：进入。

### #37 成本传导缺口：F类

- 经济原理：上游成本冲击相对下游价格变化的未传导部分。
- 数据与实现：不可计算；未提供经确认的产业链邻接表和实际生产配比。
- 开发期证据：没有生成统计结果，不能以零收益或零IC替代缺失证据。
- 2025确认：2025没有可评价样本，不能完成封存确认。
- 结论：缺失数据为：未提供经确认的产业链邻接表和实际生产配比。当前只能保留概念定义，不能形成数值特征。机器学习输入：不进入。

### #38 加工利润异常：F类

- 经济原理：加工利润相对历史季节正常水平的异常。
- 数据与实现：不可计算；缺少经确认的生产配比和其他成本序列。
- 开发期证据：没有生成统计结果，不能以零收益或零IC替代缺失证据。
- 2025确认：2025没有可评价样本，不能完成封存确认。
- 结论：缺失数据为：缺少经确认的生产配比和其他成本序列。当前只能保留概念定义，不能形成数值特征。机器学习输入：不进入。

### #39 仓单新息：E类

- 经济原理：注册仓单相对历史条件预期的标准化意外。
- 数据与实现：已计算；数据源为注册仓单及观察日；开发期覆盖率96.9%，覆盖30个品种，属于低频事件特征。
- 开发期证据：开发期Rank IC 0.0023、Rank ICIR 0.0112、胜率48.8%、FDR q 0.8841；五分组多空收益0.0004，5bps后年化-0.8%、夏普-0.0670；正向年份率66.7%。
- 2025确认：2025 Rank IC -0.0159、5bps后年化-7.4%、夏普-1.4628；与开发期Rank IC方向不一致。
- 结论：开发期无条件预测、成本后收益或跨期稳定性不足；固定Ridge增量Rank IC为-0.002907。机器学习输入：进入。

### #40 产业链传导残差：F类

- 经济原理：剥离上游滞后冲击和板块共同收益后的残差。
- 数据与实现：不可计算；未提供经确认的产业链邻接和传导滞后。
- 开发期证据：没有生成统计结果，不能以零收益或零IC替代缺失证据。
- 2025确认：2025没有可评价样本，不能完成封存确认。
- 结论：缺失数据为：未提供经确认的产业链邻接和传导滞后。当前只能保留概念定义，不能形成数值特征。机器学习输入：不进入。

### #41 动量质量因子：E类

- 经济原理：以趋势效率过滤中期动量路径噪声。
- 数据与实现：已计算；数据源为实际主力合约收益；开发期覆盖率99.9%，覆盖30个品种。
- 开发期证据：开发期Rank IC -0.0070、Rank ICIR -0.0261、胜率49.4%、FDR q 0.7152；五分组多空收益-0.0004，5bps后年化-2.8%、夏普-0.2355；正向年份率33.3%。
- 2025确认：2025 Rank IC 0.0249、5bps后年化2.2%、夏普0.2849；与开发期Rank IC方向不一致。
- 结论：开发期无条件预测、成本后收益或跨期稳定性不足；固定Ridge增量Rank IC为-0.003612。机器学习输入：进入。

### #42 波动率管理动量：E类

- 经济原理：按短期风险缩放中期趋势方向。
- 数据与实现：已计算；数据源为实际主力合约收益；开发期覆盖率99.9%，覆盖30个品种。
- 开发期证据：开发期Rank IC -0.0004、Rank ICIR -0.0018、胜率51.2%、FDR q 0.9685；五分组多空收益0.0010，5bps后年化1.1%、夏普0.1932；正向年份率33.3%。
- 2025确认：2025 Rank IC 0.0113、5bps后年化-1.1%、夏普-0.1854；与开发期Rank IC方向不一致。
- 结论：开发期无条件预测、成本后收益或跨期稳定性不足；固定Ridge增量Rank IC为-0.003381。机器学习输入：进入。

### #43 特质波动因子：D类

- 经济原理：剔除商品市场及板块共同收益后的剩余波动。
- 数据与实现：已计算；数据源为实际主力合约收益、固定板块映射；开发期覆盖率99.9%，覆盖30个品种。
- 开发期证据：开发期Rank IC 0.0150、Rank ICIR 0.0570、胜率53.4%、FDR q 0.5709；五分组多空收益0.0008，5bps后年化1.1%、夏普0.1578；正向年份率66.7%。
- 2025确认：2025 Rank IC 0.0505、5bps后年化9.1%、夏普1.0814；与开发期Rank IC同方向。
- 结论：主要用途是风险、波动、流动性或尾部暴露刻画，不能因某一年方向收益较高而改列Alpha。机器学习输入：进入。

### #44 期限结构斜率：B类

- 经济原理：用完整可交易曲线加权拟合的年化斜率。
- 数据与实现：已计算；数据源为至少三个通过流动性和期限硬门的实际合约；开发期覆盖率92.7%，覆盖30个品种。
- 开发期证据：开发期Rank IC 0.0241、Rank ICIR 0.0867、胜率54.7%、FDR q 0.4274；五分组多空收益0.0019，5bps后年化3.6%、夏普0.3612；正向年份率66.7%。
- 2025确认：2025 Rank IC 0.0165、5bps后年化-2.3%、夏普-0.1343；与开发期Rank IC同方向。
- 结论：存在正向预测和组合证据，但FDR q为0.4274，未达到10%核心门槛；固定Ridge增量Rank IC为0.005513。机器学习输入：进入。

### #45 期限结构动量：E类

- 经济原理：整条期限结构斜率的二十日变化。
- 数据与实现：已计算；数据源为完整曲线斜率历史；开发期覆盖率89.3%，覆盖30个品种。
- 开发期证据：开发期Rank IC 0.0137、Rank ICIR 0.0534、胜率53.9%、FDR q 0.5709；五分组多空收益0.0007，5bps后年化0.2%、夏普0.0673；正向年份率83.3%。
- 2025确认：2025 Rank IC -0.0733、5bps后年化-13.2%、夏普-2.2728；与开发期Rank IC方向不一致。
- 结论：开发期无条件预测、成本后收益或跨期稳定性不足；固定Ridge增量Rank IC为0.005802。机器学习输入：进入。

### #46 期限结构加速度：E类

- 经济原理：整条期限结构二十日变化的再二十日变化。
- 数据与实现：已计算；数据源为完整曲线斜率历史；开发期覆盖率86.7%，覆盖30个品种。
- 开发期证据：开发期Rank IC 0.0142、Rank ICIR 0.0558、胜率52.6%、FDR q 0.5529；五分组多空收益0.0003，5bps后年化-3.2%、夏普-0.2598；正向年份率83.3%。
- 2025确认：2025 Rank IC -0.0783、5bps后年化-13.7%、夏普-2.3070；与开发期Rank IC方向不一致。
- 结论：开发期无条件预测、成本后收益或跨期稳定性不足；固定Ridge增量Rank IC为0.004712。机器学习输入：进入。

### #47 基差意外：E类

- 经济原理：基差相对历史条件预期的标准化意外。
- 数据与实现：已计算；数据源为新鲜现货价格、观察日、主力收盘价；开发期覆盖率74.5%，覆盖27个品种，属于低频事件特征。
- 开发期证据：开发期Rank IC 0.0078、Rank ICIR 0.0331、胜率50.6%、FDR q 0.6651；五分组多空收益0.0009，5bps后年化-3.3%、夏普-0.3617；正向年份率50.0%。
- 2025确认：2025没有可评价样本，不能完成封存确认。
- 结论：开发期无条件预测、成本后收益或跨期稳定性不足；固定Ridge增量Rank IC为-0.004012。机器学习输入：进入。

### #48 库存加速度：B类

- 经济原理：去库存速度的二阶变化。
- 数据与实现：已计算；数据源为库存、观察日与数据年龄；开发期覆盖率33.3%，覆盖10个品种。
- 开发期证据：开发期Rank IC 0.0316、Rank ICIR 0.0954、胜率54.0%、FDR q 0.1926；五分组多空收益0.0041，5bps后年化4.3%、夏普0.4331；正向年份率83.3%。
- 2025确认：2025 Rank IC 0.0136、5bps后年化2.7%、夏普0.3926；与开发期Rank IC同方向。
- 结论：存在正向预测和组合证据，但FDR q为0.1926，未达到10%核心门槛；固定Ridge增量Rank IC为-0.000952。机器学习输入：进入。

### #49 库存—价格背离：B类

- 经济原理：库存趋紧相对同期价格变化的未反映部分。
- 数据与实现：已计算；数据源为库存与实际主力合约收益；开发期覆盖率33.4%，覆盖10个品种。
- 开发期证据：开发期Rank IC 0.0315、Rank ICIR 0.0938、胜率53.3%、FDR q 0.2316；五分组多空收益0.0033，5bps后年化4.7%、夏普0.5308；正向年份率83.3%。
- 2025确认：2025 Rank IC -0.0836、5bps后年化-18.5%、夏普-1.8315；与开发期Rank IC方向不一致。
- 结论：存在正向预测和组合证据，但FDR q为0.2316，未达到10%核心门槛；固定Ridge增量Rank IC为0.001449。机器学习输入：进入。

### #50 供需冲击因子：F类

- 经济原理：需求正向新息减去供给正向新息。
- 数据与实现：不可计算；data_final_raw没有供给与需求多源实际值及预期字段。
- 开发期证据：没有生成统计结果，不能以零收益或零IC替代缺失证据。
- 2025确认：2025没有可评价样本，不能完成封存确认。
- 结论：缺失数据为：data_final_raw没有供给与需求多源实际值及预期字段。当前只能保留概念定义，不能形成数值特征。机器学习输入：不进入。

### #51 拥挤度因子：F类

- 经济原理：主要参与者持仓份额集中度。
- 数据与实现：不可计算；data_final_raw没有会员或参与者分项持仓。
- 开发期证据：没有生成统计结果，不能以零收益或零IC替代缺失证据。
- 2025确认：2025没有可评价样本，不能完成封存确认。
- 结论：缺失数据为：data_final_raw没有会员或参与者分项持仓。当前只能保留概念定义，不能形成数值特征。机器学习输入：不进入。

### #52 持仓意外因子：C类

- 经济原理：持仓变化相对历史条件预期的异常。
- 数据与实现：已计算；数据源为主力持仓量；开发期覆盖率100.0%，覆盖30个品种。
- 开发期证据：开发期Rank IC 0.0006、Rank ICIR 0.0029、胜率49.2%、FDR q 0.9455；五分组多空收益0.0011，5bps后年化-1.1%、夏普-0.1030；正向年份率66.7%。
- 2025确认：2025 Rank IC 0.0021、5bps后年化-8.0%、夏普-1.1517；与开发期Rank IC同方向。
- 结论：经济角色主要是调节其他信号、刻画交易状态或形成交互，不要求其单独产生稳定多空收益。机器学习输入：进入。

### #53 量价背离因子：E类

- 经济原理：价格趋势与成交量变化不一致程度。
- 数据与实现：已计算；数据源为实际主力合约收益、成交量；开发期覆盖率100.0%，覆盖30个品种。
- 开发期证据：开发期Rank IC -0.0086、Rank ICIR -0.0337、胜率48.0%、FDR q 0.6421；五分组多空收益-0.0010，5bps后年化-5.3%、夏普-0.5503；正向年份率33.3%。
- 2025确认：2025 Rank IC 0.0350、5bps后年化6.1%、夏普0.6270；与开发期Rank IC方向不一致。
- 结论：开发期无条件预测、成本后收益或跨期稳定性不足；固定Ridge增量Rank IC为-0.014203。机器学习输入：进入。

### #54 因子交互因子：E类

- 经济原理：动量和Carry同时确认的非加性信息。
- 数据与实现：已计算；数据源为动量与期限结构；开发期覆盖率99.8%，覆盖30个品种。
- 开发期证据：开发期Rank IC 0.0101、Rank ICIR 0.0464、胜率49.9%、FDR q 0.6421；五分组多空收益0.0008，5bps后年化0.2%、夏普0.0624；正向年份率66.7%。
- 2025确认：2025 Rank IC -0.0291、5bps后年化-2.5%、夏普-0.3720；与开发期Rank IC方向不一致。
- 结论：开发期无条件预测、成本后收益或跨期稳定性不足；没有可解释的固定Ridge增量结果。机器学习输入：进入。

### #55 非线性基本面因子：E类

- 经济原理：极端供需紧张状态的平方增强。
- 数据与实现：已计算；数据源为供需紧张度；开发期覆盖率78.3%，覆盖28个品种。
- 开发期证据：开发期Rank IC 0.0184、Rank ICIR 0.0677、胜率52.6%、FDR q 0.4592；五分组多空收益0.0009，5bps后年化0.0%、夏普0.0468；正向年份率66.7%。
- 2025确认：2025 Rank IC -0.1157、5bps后年化-25.3%、夏普-2.1315；与开发期Rank IC方向不一致。
- 结论：开发期无条件预测、成本后收益或跨期稳定性不足；没有可解释的固定Ridge增量结果。机器学习输入：进入。

### #56 商品网络因子：F类

- 经济原理：按产业链和替代关系聚合邻居冲击。
- 数据与实现：不可计算；未提供经确认的商品网络邻接表与权重。
- 开发期证据：没有生成统计结果，不能以零收益或零IC替代缺失证据。
- 2025确认：2025没有可评价样本，不能完成封存确认。
- 结论：缺失数据为：未提供经确认的商品网络邻接表与权重。当前只能保留概念定义，不能形成数值特征。机器学习输入：不进入。

### #57 相关性风险因子：D类

- 经济原理：品种与商品市场短期相关性相对长期基准的上升。
- 数据与实现：已计算；数据源为实际主力合约收益、商品市场收益；开发期覆盖率99.7%，覆盖30个品种。
- 开发期证据：开发期Rank IC -0.0307、Rank ICIR -0.1250、胜率44.0%、FDR q 0.1926；五分组多空收益-0.0027，5bps后年化-8.7%、夏普-1.0091；正向年份率0.0%。
- 2025确认：2025 Rank IC -0.0073、5bps后年化-10.7%、夏普-1.2875；与开发期Rank IC同方向。
- 结论：主要用途是风险、波动、流动性或尾部暴露刻画，不能因某一年方向收益较高而改列Alpha。机器学习输入：进入。

## 2025封存测试分析

- 2025共有40个因子可评价Rank IC；与开发期同方向比例37.5%。
- Carry、仓单紧张度和库存加速度保持正向，是当前最有价值的持续跟踪对象。
- 信息背离、库存—价格背离、期限结构动量和期限结构加速度在2025反向，说明开发期结果存在显著时间不稳定性。
- 基差及其衍生因子因现货数据在2023年底终止，2025没有评价样本；这属于数据缺口，不应记为通过或失败。
- 趋势因子在开发期偏弱、2025转正，支持把趋势看作可能的状态依赖特征，而不是当前无条件核心Alpha。
- 封存结果不反向改变A—F分类，也不用于重新选择窗口和方向。

## 相关性与机器学习输入建议

- 47个可计算因子已经加工为92个模型字段，其中包含数值特征和缺失/质量指示器；92个字段不等于92个独立Alpha。
- Carry与整曲线斜率、库存变化与库存—价格背离、趋势与动量质量之间存在较高相关，线性模型应使用Ridge或Elastic Net，不能直接用普通OLS解释单个系数。
- 不应根据当前样本收益删除全部E类变量；但必须用扩展折消融比较Carry基线、期限结构组、基本面组、趋势组和全量特征组。
- F类因子不得以全零列进入模型。未来补充真实数据后，应作为新版本特征重新冻结并重新运行开发/封存流程。

## 最终研究判断

当前因子库的工程质量和信息覆盖较好，但单因子Alpha证据仍属中等偏弱。可优先把Carry、仓单紧张度、库存加速度和整曲线斜率作为基准预测组；基差保留为数据恢复后的重点候选；信息背离类仅作为弱辅助；波动、流动性和持仓变量作为风险或条件信息。最终是否具有联合预测价值，必须由严格扩展窗口下的正则化模型相对Carry基线给出。

