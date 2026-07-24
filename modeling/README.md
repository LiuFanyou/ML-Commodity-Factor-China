# Modeling

本目录存放各模型版本的配置、报告与运行产物。源码在 `src/factor_modeling/`。

## 模型一览

| 模型 | 源码 | 配置与产物 | 运行 |
|------|------|------------|------|
| Ridge 线性回归 | `src/factor_modeling/linear_ridge.py` | [`linear_ridge_v1/`](linear_ridge_v1/) | `PYTHONPATH=src py -m factor_modeling.linear_ridge --config modeling/linear_ridge_v1/config.json` |
| 单因子排序 | `src/factor_modeling/single_factor_rank.py` | [`single_factor_rank_v1/`](single_factor_rank_v1/) | `PYTHONPATH=src py -m factor_modeling.single_factor_rank --config modeling/single_factor_rank_v1/config.json` |
| 等权多因子（全体等权） | `src/factor_modeling/equal_weight_multi.py` | [`equal_weight_multi_v1/`](equal_weight_multi_v1/) | `PYTHONPATH=src py -m factor_modeling.equal_weight_multi --config modeling/equal_weight_multi_v1/config.json` |
| Conv1d+Attention 机器学习 | `src/factor_modeling/ml_cnn.py` | [`ml_cnn_v1/`](ml_cnn_v1/) | `PYTHONPATH=src py -m factor_modeling.ml_cnn --config modeling/ml_cnn_v1/config.json` |

## 共享工具

各模型共用的回测与评价函数在 `src/factor_modeling/common.py`（横截面排名目标、purge、多空组合、收益统计、日度 IC 等）。  
单因子、Ridge、等权多因子、机器学习 **互不依赖**；Ridge 专属逻辑仅在 `linear_ridge.py`，神经网络打分仅在 `ml_cnn.py`。

## Ridge 附属

- 模型冻结：`src/factor_modeling/model_freeze.py`
- 2025 补充样本外：`src/factor_modeling/supplemental_2025.py`（产物在 `linear_ridge_v1/supplemental_2025/`）
