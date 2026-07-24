# 建模训练交接包

本文件夹可直接交给建模队友。训练不需要原始 `data_final_raw`，但若要审计或重新计算因子，仍需要原数据和源码。
`factor_registry.csv` 完整登记 V2.1 的57个概念因子；可计算且通过数据质量硬门的因子才会形成模型特征。

## 直接训练

1. 安装 `requirements.txt`。
2. 运行 `python example_train.py` 验证接口。
3. 正式模型读取 `ml_dataset_development.csv.gz`。
4. 模型特征严格取 `ml_feature_registry.csv` 的 `feature_name` 全部行。
5. 主标签为 `future_return_5d`。
6. 必须使用 `fold_definitions.csv` 的扩展窗口，禁止随机切分。
7. 训练行必须满足 `label_end_date_5d < valid_start`。

2025 数据不在本文件夹中，防止开发阶段反复窥视最终样本。
