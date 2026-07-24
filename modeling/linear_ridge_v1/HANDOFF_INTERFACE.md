# Handoff 矩阵接口说明

本文件定义 Ridge 模型消费的固定数据接口。因子与标签的生成不在本模型内完成。
口径对齐当前 `training_handoff/training_contract.json`。

## 1. 目录

默认：`training_handoff/`  
可通过 `modeling/linear_ridge_v1/config.json` 的 `handoff_directory` 切换。

## 2. 必选文件

### 2.1 `ml_feature_registry.csv`

| 列 | 必填 | 说明 |
|----|------|------|
| `feature_name` | 是 | 模型输入字段名，须与特征矩阵列名一致 |
| `source_factor` | 否 | 来源因子中文名 |
| `usage_type` | 否 | predictor / risk / quality_flag 等 |
| `correlation_cluster` | 否 | 高相关聚类 ID（协议要求保留，不物理删除） |

模型使用注册表中的**全部** `feature_name`，不做相关因子物理删除。

### 2.2 特征矩阵（默认 `ml_features_development.csv.gz`）

| 列 | 必填 | 说明 |
|----|------|------|
| `trade_date` | 是 | 日期，可解析为 datetime |
| `product` | 是 | 品种代码 |
| `sector` | 是 | 板块 |
| `<feature_name>...` | 是 | 注册表列出的全部字段 |

要求：

- 主键 `trade_date + product` 唯一
- 特征应已完成建模侧预处理（如横截面稳健标准化、缺失填 0、缺失指示器）
- Ridge 折内仍会再做一次训练期 `StandardScaler`

### 2.3 标签矩阵（默认 `labels_development.csv.gz`）

| 列 | 必填 | 说明 |
|----|------|------|
| `trade_date` | 是 | 与特征对齐 |
| `product` | 是 | 与特征对齐 |
| `future_return_5d` | 是* | 主标签（*由 config.`raw_return_column` 指定） |
| `label_end_date_5d` | 是 | 训练净化：必须 `< valid_start` |
| `future_return_1d` / `20d` | 否 | 可并存，本模型默认不用 |

要求：主键唯一；与特征矩阵按 `trade_date + product` 内连接。

### 2.4 `fold_definitions.csv`（必须）

扩展窗口折定义。当前包：

- `fold_2019` … `fold_2024`
- 训练起点固定 2015-01-05，验证年逐年外推

### 2.5 `training_contract.json`（推荐）

记录主标签、净化规则、5 日持有/调仓、成本网格等契约；`config.json` 中的文件名优先覆盖。

## 3. 模型侧衍生列

Ridge 在内存中生成，不要求 handoff 预先提供：

- `target_cs_rank_5d`：由 `raw_return_column` 做同日横截面中心化排名得到，映射到 `[-1, 1]`

## 4. 执行协议摘要

1. T 日收盘形成信号，T+1 开盘成交
2. 持有 5 日，每 5 日调仓
3. 多空各 20%，组内等权，总敞口 1、净敞口 0
4. 单边成本主口径 5bp，敏感性 0/3/5/10bp

## 5. 更换 handoff 因子包

1. 按上述 schema 导出新的 registry / features / labels / folds
2. 放入新目录或覆盖旧文件
3. 修改 `handoff_directory`（如需要）后重跑 Ridge

加载代码：`src/factor_modeling/handoff.py`  
训练代码：`src/factor_modeling/linear_ridge.py`
