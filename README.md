# 中国商品期货 59 因子 Conv-Transformer 多任务研究

本项目使用 2015—2025 年 30 个商品期货品种的整合合约日表，构造因果主力合约、未来 5 日截面超额收益标签和 59 个训练特征，训练 Conv1d 与 Transformer Encoder 共享底层的分类/回归双头网络，并进行扩展窗口样本外回测。

当前默认流程已经从“年度原始行情 + 旧 59 因子交接包”切换为“整合合约日表直接生成训练面板与 59 个因果特征”。旧交接包和年度行情仍保留为兼容与历史复现材料，但不会被默认流程读取。详细文件状态见 [`FILE_STATUS.md`](FILE_STATUS.md)。

## 当前默认数据契约

默认训练数据为：

```text
futures_contract_daily_30varieties_2015_2025.csv
```

字段说明为：

```text
整合后CSV字段说明.md
```

整合表共 53 个字段，粒度为“一份实际期货合约在一个交易日一行”，唯一键为 `trade_date + ts_code`。数据覆盖 2015-01-05 至 2025-12-31，包含 30 个品种的合约元数据、OHLC、结算价、成交量、持仓量、手续费、保证金、现货、仓单和库存信息。脚本会优先只读取这一个整合表，避免与年度行情重复加载。

`fut_basic_all.csv` 仍作为合约元数据补充来源。缺少该文件时脚本可以从合约代码推断品种和到期月份，但保留该文件更有利于复现主力换月与期限结构。

默认数据和训练起点均为 2015 年，首个样本外测试年为 2022 年，随后按年度扩展训练窗口至 2025 年。

## 特征与标签

整合模式从因果主力合约面板生成 59 个输入字段，其中包括 45 个连续特征和 14 个缺失质量标志。连续特征覆盖价格动量、趋势效率、波动与偏度、振幅、量仓变化、Amihud 非流动性、期限结构、现货、仓单及库存。连续特征按交易日做横截面 MAD 稳健标准化和板块去均值，缺失位置单独记录，最终模型输入为有限 `float32` 数值。

现货、仓单和库存特征会结合对应的 `*_age_days` 字段屏蔽过度陈旧的观测。该处理只使用决策日及更早数据，不使用未来信息。

信号在决策日 `t` 收盘后形成，`t+1` 开盘执行。主标签为从 `t+1` 开盘开始的未来 5 个 open-to-open 收益几何累计值，并在执行日进行截面去均值。方向头预测超额收益是否大于零，回归头预测未来 5 日超额收益幅度。

## 模型与训练

模型输入形状为 `[Batch, 59, 20]`。共享结构包括 Conv1d 局部模式嵌入、品种 Embedding、单层轻量 Transformer Encoder、LayerNorm 和 Global Average Pooling。分类头输出方向 logits，回归头输出收益幅度。

总损失为：

```text
0.7 × BCEWithLogitsLoss + 0.3 × HuberLoss
```

综合预测信号为：

```text
prediction = sigmoid(logits) × regression_prediction
```

模型采用扩展窗口验证。首个训练窗为 2015—2021 年，2022 年样本外测试；后续逐年扩展。未来 5 日标签存在重叠，因此训练期每 5 个交易日保留一个日期的完整品种截面，验证和测试仍逐日预测。

## 组合与风控

回测沿用既有策略，不因展示层更新而改变。预测信号先做 3 日 EMA；多头在当日信号上 30%且大于零时进入，空头在下 30%且小于零时进入，信号越过中位数后退出。组合按预测强度加权，并使用仅由历史数据拟合的两状态 GMM 识别系统性压力；高压力状态下组合毛杠杆降至 50%。

原回测文件继续报告 1、3、5 bps 单边费用。新增评价层另行报告统一的 0、3、5、10 bps 情景，不修改原回测收益列。

## 安装和运行

Windows 下推荐直接双击：

```text
run_integrated_training.bat
```

该脚本会设置 `FUTURES_FEATURE_MODE=integrated`，使用项目内 `futures_env`，把完整控制台输出写入 `integrated_training.log`，并将结果保存到 `futures_ml_outputs`。

也可以在终端运行：

```bat
cd /d "D:\大学\大二下\财大量化\tushare_theme7\tushare_theme7"
futures_env\Scripts\activate
pip install -r requirements_futures_ml.txt
set FUTURES_FEATURE_MODE=integrated
python china_futures_ml_kaggle.py
```

可通过以下环境变量切换数据位置或兼容模式：

```text
FUTURES_DATA_ROOT=<项目或数据目录>
FUTURES_INTEGRATED_CSV=<整合CSV路径>
FUTURES_FEATURE_MODE=auto|integrated|handoff
```

`auto` 会在发现整合表时选择 `integrated`；只有找不到整合表时才使用 `handoff`。当前旧交接矩阵与新 30 品种池不一致，因此不建议直接运行旧 `handoff` 模式，除非先更新对应特征矩阵。

## 原有输出

原有输出仍位于 `futures_ml_outputs`，主要包括：

```text
best_model.pth
backtest_metrics.csv
oos_predictions.csv
daily_backtest.csv
positions.csv
training_history.csv
roll_yield_sources.csv
feature_merge_missing_report.csv
model_feature_list.csv
systemic_stress_features.csv
systemic_stress_probability.csv
run_config_and_data_report.json
loss_curve.png
cumulative_return.png
fee_sensitivity.png
feature_importance.png
```

## 新增评价与审计输出

评价层只读取既有样本外预测、每日回测和品种面板，不会改变因子、模型、训练、信号、持仓或原结果。程序先保存原结果，再生成评价文件；即使评价层失败，原结果仍会保留。

```text
unified_metrics_panel.csv
prediction_ic_daily.csv
rank_ic_by_year.csv
cost_scenarios.csv
statistical_tests.csv
grouped_return_monotonicity.csv
robustness_holding_period.csv
robustness_sector.csv
robustness_liquidity.csv
robustness_signal_delay.csv
evaluation_methodology.json
```

统一面板包含累计收益、年化收益、Sharpe、最大回撤、Pearson IC、Rank IC、IC 标准差、未年化 ICIR、IR、IC 阈值概率、HAC t 统计量和 p 值、分组单调性，以及 Calmar、换手和年化波动。统计附表包含 HAC 标准误和圆形块自助法 95% 区间；稳健性附表覆盖 1/5/20 日持有期、四大板块、剔除最低流动性尾部和信号延迟一个市场交易日。

## 结果状态说明

`futures_ml_outputs` 和旧版 `run_report.pdf` 当前保存的是数据更新和代码重构前的历史运行结果，不能视为新整合数据流程的正式绩效。完成一次新的 2015—2025 全量训练后，这些输出会由当前脚本重新生成；届时再依据新文件更新正式绩效结论。

本项目用于研究验证，不构成投资建议。固定 bp 成本不能完整覆盖盘口冲击、涨跌停、容量、保证金变化和期货公司加收费用。
