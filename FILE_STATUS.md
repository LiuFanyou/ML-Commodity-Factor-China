# 项目文件状态清单

**更新时间：** 2026-07-23  
**原则：** 本清单只标注用途，不删除、不移动、不重命名任何文件。  
**判断基准：** 当前 `china_futures_ml_kaggle.py` 默认 `feature_mode=auto`，发现整合表后使用 `integrated` 流程。

## 1. 当前默认流程需要保留

| 文件或目录 | 状态 | 当前用途 |
| --- | --- | --- |
| `china_futures_ml_kaggle.py` | 当前核心 | 训练、样本外预测、组合回测、图表和统一评价审计主脚本。 |
| `futures_contract_daily_30varieties_2015_2025.csv` | 当前核心数据 | 默认整合合约日表；包含 2015—2025 年 30 品种的合约行情及现货、仓单、库存等字段。 |
| `整合后CSV字段说明.md` | 当前数据文档 | 说明整合表的 53 个字段、粒度、费用单位和建模注意事项。 |
| `fut_basic_all.csv` | 当前辅助数据 | 补充合约上市、退市、交易所和交割元数据；缺失时可降级推断，但建议保留。 |
| `requirements_futures_ml.txt` | 当前环境文件 | Python 依赖列表。 |
| `futures_env/` | 当前本地环境 | Windows 项目虚拟环境；`run_integrated_training.bat` 默认使用其中的解释器。 |
| `run_integrated_training.bat` | 当前运行入口 | 设置整合模式并运行主脚本，将控制台输出写入 `integrated_training.log`。 |
| `README.md` | 当前项目文档 | 安装、运行、数据契约、模型与输出说明。 |
| `UPDATE_HISTORY.md` | 当前项目文档 | 记录研究版本和迁移历史。 |
| `run_report.md` | 当前报告模板 | 描述当前设计与结果状态；新训练完成后填入正式结果。 |
| `FILE_STATUS.md` | 当前维护文档 | 本文件，用于标识当前、兼容、历史和可归档文件。 |

## 2. 兼容流程或历史复现需要，默认不读取

这些文件并非“完全无用”，但当前整合模式不会读取。若将 `FUTURES_FEATURE_MODE` 强制设为 `handoff`，或需要复现旧实验，它们仍有价值。

| 文件或目录 | 状态 | 说明 |
| --- | --- | --- |
| `ml_feature_registry.csv` | 旧 handoff 兼容 | 旧 59 因子注册表。当前整合模式自行生成 59 个新训练字段，不读取该文件。 |
| `ml_features_development.csv.gz` | 旧 handoff 兼容 | 旧品种日特征矩阵，截至 2024 年。其品种池与当前新 30 品种池不一致，不能直接用于当前默认训练。 |
| `training_handoff/` | 旧 handoff 归档 | 旧交接包、标签、注册表、审计报告及校验文件。用于追溯旧特征来源。 |
| `2015/`—`2025/` 中的 `fut_daily_raw.csv` | 原始行情备份 | 当前整合表存在时不会读取；只有找不到整合表并回退扫描行情时才可能使用。 |

旧 handoff 的已知品种差异如下：当前整合池有但旧矩阵没有 `CS/L/SF/SN`；旧矩阵有但当前整合池没有 `EG/FG/OI/RU`。因此不要通过关闭严格校验来混用两个研究池。

## 3. 上游数据整合材料：当前训练不直接读取，可考虑归档

这些文件看起来是生成整合表的上游材料或其他宽表版本。当前主脚本没有引用它们，但如果未来需要重建、检查或扩展整合 CSV，仍可能有用。建议先归档而不是删除。

| 文件或目录 | 建议状态 | 依据 |
| --- | --- | --- |
| `China_Futures_AllData_WIDE_30Varieties_2015_2025.csv` | 可归档：上游超宽表 | 含约 4,070 列的品种级宽表；当前主脚本不识别其中文字段结构，也不读取。 |
| `cangdan.kucun.xianhuo.csv` | 可归档：上游基本面材料 | 仓单、库存、现货整合中间数据；相关字段已进入当前 53 列合约日表，主脚本不直接读取该文件。 |
| `fut_settle_all.csv` | 可归档：上游结算参数 | 汇总结算、费用或保证金参数；当前训练直接使用整合表中的对应字段，不读取该文件。 |
| `fut_settle_by_date/` | 可归档：上游逐日结算参数 | 包含大量逐日 CSV。扫描函数会显式跳过该目录，当前训练不读取。 |
| `30个品种.md` | 可归档：研究池草稿 | 30 品种范围的历史说明或草稿；当前代码中的 `Config.products` 才是执行口径。 |
| `30个品种_clean.md` | 可归档：整理版研究池说明 | 与前一文件用途相近，当前训练不直接读取。若要保留一个人工品种说明，建议人工比较后选定权威版本。 |

“可归档”只表示当前训练运行不需要，不代表源数据可以永久删除。特别是 `fut_settle_all.csv`、`fut_settle_by_date/` 和 `cangdan.kucun.xianhuo.csv` 可能是重新生成整合表的重要来源。

## 4. 历史运行结果：等待新全量训练覆盖或另行归档

| 文件或目录 | 状态 | 说明 |
| --- | --- | --- |
| `futures_ml_outputs/` | 历史运行快照 | 现有 17 个文件来自当前整合数据与评价层完成前的旧运行，不能代表新版本绩效。新全量训练会在同目录生成或覆盖结果。 |
| `run_report.pdf` | 历史报告 | 与旧运行结果对应。本次按用户选择只更新 Markdown，没有重做 PDF。 |

若需要同时保存旧、新实验，建议在运行新版本前手动复制为带日期的归档目录，例如：

```text
futures_ml_outputs_archive_2026-07-23_old/
run_report_archive_2026-07-22_old.pdf
```

本项目未自动执行上述移动或重命名操作。

## 5. 自动生成或缓存，可随环境重建

| 文件或目录 | 状态 | 说明 |
| --- | --- | --- |
| `__pycache__/` | Python 缓存 | 由解释器自动生成，不是研究数据。 |
| `integrated_training.log` | 运行日志 | 运行批处理后生成，用于定位报错；当前可能尚不存在。 |
| `evaluation_audit_error.txt` | 条件错误记录 | 只有统一评价层失败时生成；原模型和回测结果仍会保存。 |

## 6. 当前结果目录预期新增的评价文件

新全量训练成功后，`futures_ml_outputs/` 除原文件外应新增：

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

若这些文件尚未出现，通常表示当前目录仍是旧运行结果，或新全量训练尚未完成。

## 7. 归档前建议检查

在人工归档任何上游数据前，建议确认以下事项：整合表是否可以从现有材料完整重建；是否需要保留手续费和保证金的逐日历史；是否需要对现货、仓单和库存映射进行重新审计；旧 handoff 是否仍需用于论文对照实验；旧 `futures_ml_outputs` 是否已备份。

本清单不会自动判断数据的唯一性或法律/课程提交要求，因此没有把任何数据标为“可以安全删除”。
