# 中国商品期货 Alpha 因子科学筛选系统

本项目基于 `CORE30_V1` 的30个中国商品期货品种，建立严格点时、可重复运行的因子筛选和机器学习特征加工流程。

> GitHub 仓库不包含原始/研究数据、压缩特征矩阵、标签矩阵或预测明细。克隆后需按 `DATA_DIRECTORY.md` 恢复本地数据，或重新运行数据与特征流水线；因子注册表和汇总报告仍保留在仓库中。

## 核心原则

- 原始数据和 `data_research/core30_v1/` 不修改。
- 因子在交易日收盘后形成，最早下一交易日开盘使用。
- 2015—2024用于开发；2025在代码和配置冻结后只评估一次。
- 没有真实发布时间、字段语义或交易成本单位的数据不会被补造。
- 单因子弱不等于直接删除；系统同时检验相关性、增量价值和风险/条件用途。

## 安装与测试

```bash
python3 -m pip install -e .
pytest
```

也可以不安装，直接使用：

```bash
PYTHONPATH=src python3 -m factor_screening development
```

## 运行顺序

### 1. 开发期分析（不会加载2025进入评价）

```bash
PYTHONPATH=src python3 -m factor_screening development
```

### 2. 冻结代码、配置和开发期结论

```bash
PYTHONPATH=src python3 -m factor_screening seal
```

### 3. 一次性运行2025封存测试

```bash
PYTHONPATH=src python3 -m factor_screening sealed
```

封存测试成功后，将最终矩阵严格对齐到开发期冻结的特征注册表（只补缺失质量掩码，不重算封存指标）：

```bash
PYTHONPATH=src python3 scripts/reconcile_ml_schema.py
```

封存结果禁止覆盖。若需要开始新的研究版本，应复制项目并建立新的版本号，而不是修改已有封存结果。

## 输出

- `reports/data_audit_report.md`
- `reports/factor_screening_report.md`
- `reports/factor_family_report.md`
- `reports/ml_feature_report.md`
- `reports/sealed_2025_report.md`
- `features/ml_feature_registry.csv`
- `features/feature_dictionary.md`
- `features/ml_features_development.csv.gz`
- `features/ml_features.csv.gz`
- `artifacts/development/`：可复核的中间统计结果
- `artifacts/FREEZE_MANIFEST.json`：封存哈希清单

## 分类

- A：核心预测因子
- B：辅助预测因子
- C：条件变量
- D：风险变量
- E：暂不使用
- F：数据不足无法判断

筛选结论是研究证据，不是盈利承诺。

## Ridge线性模型测试

使用全部59个注册特征，对下一交易日开盘至收盘收益的横截面排名进行三年滚动Ridge回归：

```bash
PYTHONPATH=src python3 -m factor_modeling.linear_ridge --config modeling/linear_ridge_v1/config.json
```

模型定义、策略规则和结果位于 `modeling/linear_ridge_v1/`。该模型只使用2015—2024开发数据，2025不参与模型选择或评价。

## 2015固定起点扩展窗口版本

新版本不覆盖上面的冻结结果。每个测试年前都在当时的扩展训练窗口内重新完成因子筛选、去重、增量检验、特征加工和Ridge训练：

```text
2015—2018训练 -> 2019测试
2015—2019训练 -> 2020测试
...
2015—2024训练 -> 2025测试
```

运行：

```bash
PYTHONPATH=src python3 -m factor_expanding --config config/screening_expanding_2015_v1.json
```

结果写入 `runs/expanding_2015_v1/`，成功后禁止覆盖。
