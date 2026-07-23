# 中国商品期货因子模型训练交接包

本目录用于训练单一机器学习收益预测模型。输入端统一使用 `ml_feature_registry.csv` 注册的全部字段，不再按 A/B/C/D 分层建模。

## 文件说明

- `ml_features_development.csv.gz`：2015-01-05 至 2024-12-31 的实际特征矩阵。
- `labels_development.csv.gz`：1日、5日和20日前瞻收益标签。
- `ml_feature_registry.csv`：模型输入字段清单和元数据；它不是特征数值文件。
- `feature_dictionary.md`：逐字段经济含义和加工方式。
- `screening.json`：筛选与特征加工配置。
- `factor_screening_report.md`：因子筛选证据和分类结果。
- `ml_feature_report.md`：机器学习输入端说明。
- `data_audit_report.md`：数据质量、可用性及限制。
- `training_contract.json`：供训练代码读取的固定接口约定。
- `SHA256SUMS`：文件完整性校验值。

## 固定训练接口

- 合并主键：`trade_date + product`。
- 默认目标：`future_return_5d`。
- 输入字段：`ml_feature_registry.csv` 中 `feature_name` 列的全部字段。
- `trade_date`、`product`、`sector` 是索引或分组字段，默认不作为数值特征。
- 特征已经完成同日横截面稳健标准化、缺失填0和缺失指示器加工。

```python
import pandas as pd

registry = pd.read_csv("ml_feature_registry.csv")
feature_names = registry["feature_name"].tolist()

features = pd.read_csv("ml_features_development.csv.gz", parse_dates=["trade_date"])
labels = pd.read_csv("labels_development.csv.gz", parse_dates=["trade_date"])

data = features.merge(
    labels,
    on=["trade_date", "product"],
    how="inner",
    validate="one_to_one",
).dropna(subset=["future_return_5d"])

X = data[feature_names].astype("float32")
y = data["future_return_5d"].astype("float32")
```

## 验证方式

不得随机拆分数据。采用三年训练、下一年验证：

```text
2015—2017训练 → 2018验证
2016—2018训练 → 2019验证
2017—2019训练 → 2020验证
2018—2020训练 → 2021验证
2019—2021训练 → 2022验证
2020—2022训练 → 2023验证
2021—2023训练 → 2024验证
```

使用5日标签时，训练集和验证集边界至少净化5个交易日。模型选择只能依据2018—2024滚动样本外结果。

## 重要限制

1. 当前5日标签复合未来五个每日开盘至收盘收益。如果目标是从首次开盘连续持有五日，需要重新定义标签。
2. 成本结果只是单边1/3/5基点敏感性，不代表已核验的真实手续费和冲击成本。
3. 2025已经用于因子级封存确认，本包不包含2025特征，避免其被用于模型调参。
4. 高相关特征全部保留时，线性模型必须正则化；树模型应限制深度、叶节点和特征采样。

