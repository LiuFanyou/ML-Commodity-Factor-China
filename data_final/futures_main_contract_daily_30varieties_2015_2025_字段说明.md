# `futures_main_contract_daily_30varieties_2015_2025.csv` 字段说明

## 1. 数据概况

- 数据范围：2015-01-05 至 2025-12-31
- 品种数量：30
- 数据粒度：一个期货品种在一个交易日对应一行
- 唯一主键：`trade_date + product_code`
- 行数：78,112
- 列数：78
- 行情来源：原 ZIP 中的 TuShare 主力连续序列
- 现货、仓单和库存：按品种合并
- 机器学习标签：未生成

## 2. 日期、品种与合约标识

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `trade_date` | 日期 | 交易日，格式为 `YYYY-MM-DD`。 |
| `ts_code` | 文本 | 原始主力连续代码，如 `RB.SHF`、`CU.SHF`。它不是可直接交易的月合约。 |
| `product_code` | 文本 | 期货品种代码，如 `RB`、`CU`、`M`。 |
| `product_name` | 文本 | 期货品种中文名称。 |
| `exchange` | 文本 | 交易所代码：`SHF`、`DCE`、`ZCE` 或 `INE`。 |
| `mapped_actual_ts_code` | 文本 | 连续序列当天映射到的实际合约代码，如 `RB2101.SHF`。实盘交易和换月分析应使用该字段。 |
| `mapping_quality` | 文本 | 连续行情反推实际合约的匹配质量，具体取值见下文。 |

`mapping_quality` 的主要取值：

- `exact_current_fields`：开高低收、结算价、成交量、成交额和持仓量全部精确匹配。
- `ambiguous_exact_current_fields_kept_previous`：多个实际合约完全相同，延续上一实际主力。
- `exact_without_amount`：除成交额外的当日字段精确匹配。
- `exact_prices_and_oi`：价格及持仓量精确匹配。
- `exact_prices`：价格字段精确匹配。

全部 78,112 行均成功映射到实际合约，没有无法映射的记录。

## 3. 映射实际合约元数据

以下字段对应 `mapped_actual_ts_code`，而不是连续代码本身。

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `contract` | 文本 | 不含交易所后缀的实际合约代码。 |
| `contract_month` | 文本 | 实际合约月份，通常为 `YYYYMM`。 |
| `list_date` | 日期 | 实际合约上市日期。 |
| `delist_date` | 日期 | 实际合约退市或最后交易日期。 |
| `last_delivery_date` | 日期 | 实际合约最后交割日期。 |
| `trade_unit` | 文本 | 合约标的单位，如吨。 |
| `contract_size` | 数值 | 每手合约对应的标的数量。 |
| `quote_unit` | 文本 | 期货报价单位，如人民币元/吨。 |
| `quote_unit_desc` | 文本 | 最小报价单位的文字说明。 |
| `delivery_mode` | 文本 | 交割方式，如实物交割。 |

## 4. 原主力连续日行情

以下字段直接来自原 ZIP 的主力连续序列，未进行价格复权。

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `pre_close` | 数值 | 连续序列的前收盘价。换月日可能来自上一实际主力。 |
| `pre_settle` | 数值 | 连续序列的前结算价。换月日可能来自上一实际主力。 |
| `open` | 数值 | 当日连续主力开盘价，来自当天映射的实际合约。 |
| `high` | 数值 | 当日最高价。 |
| `low` | 数值 | 当日最低价。 |
| `close` | 数值 | 当日收盘价。 |
| `settle` | 数值 | 当日结算价。 |
| `volume` | 数值 | 当日主力合约成交量。 |
| `amount` | 数值 | 当日主力合约成交金额，沿用原始数值和单位口径，未统一换算。 |
| `open_interest` | 数值 | 当日主力合约持仓量。 |

注意：换月日虽然可能衔接上一主力的 `pre_close` 和 `pre_settle`，但当日开高低收来自新主力，因此直接对相邻 `close` 计算收益仍可能包含换月跳变。

## 5. 实际合约手续费与保证金

以下字段按 `trade_date + mapped_actual_ts_code` 合并。

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `trading_fee_rate` | 数值 | 按成交金额计费的手续费率，单位为千分比 `‰`。例如 `0.1` 表示万分之一。 |
| `trading_fee` | 数值 | 按固定金额计费的每手手续费，通常单位为元/手。 |
| `delivery_fee` | 数值 | 交割手续费。 |
| `buy_hedging_margin_rate` | 数值 | 买入套期保值保证金比例。`0.1` 表示 10%。 |
| `sell_hedging_margin_rate` | 数值 | 卖出套期保值保证金比例。 |
| `long_margin_rate` | 数值 | 多头投机保证金比例。 |
| `short_margin_rate` | 数值 | 空头投机保证金比例。 |

单边每手手续费的基本计算：

```text
按成交金额计费 = 成交价 × contract_size × trading_fee_rate ÷ 1000
按固定金额计费 = trading_fee
```

两种计费方式通常互斥，不应直接相加。源数据没有对应结算参数时字段为空。

## 6. 统一现货价格

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `spot_price` | 数值 | 每个品种选定的统一现货基准价格。没有新报价时沿用最近一次按“日期”记录的价格。 |
| `spot_price_unit` | 文本 | 现货价格原始报价单位。不同品种可能不同。 |
| `spot_indicator_code` | 文本 | 被选作现货基准的原始指标代码。 |
| `spot_indicator_name` | 文本 | 被选作现货基准的原始指标名称。 |
| `spot_price_source` | 文本 | 现货指标数据来源。 |
| `spot_observation_date` | 日期 | 当前现货价格的实际观测日期。 |
| `spot_price_age_days` | 整数 | 交易日距离现货观测日期的自然日数。0 表示当日有新报价。 |

原始现货数据截至 2023 年，因此 2024—2025 年会按既定规则沿用最近价格。建模时应同时使用 `spot_price_age_days` 判断数据是否陈旧。

## 7. 仓单字段

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `warehouse_receipt_current` | 数值 | 当前仓单数量，采用日度仓单记录并向后沿用。 |
| `warehouse_receipt_change` | 数值 | 当前仓单相对上一条仓单观测的变化，由仓单当前值重新计算。 |
| `warehouse_receipt_unit` | 文本 | 仓单单位，如张、手或吨。 |
| `warehouse_receipt_observation_date` | 日期 | 当前仓单值的原始观测日期。 |
| `warehouse_receipt_age_days` | 整数 | 交易日距离仓单观测日期的自然日数。 |
| `warehouse_receipt_conflict` | 0/1 | 1 表示同日的日度仓单记录与周期库存记录所带仓单值不同。 |

冲突标记表示原始来源存在不同值，不代表当前 CSV 的主键重复。

## 8. 库存与可用库存

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `stock_current` | 数值 | 周期库存记录中的当前库存值，并沿用至下一次非空观测。 |
| `stock_unit` | 文本 | 库存单位。 |
| `stock_observation_date` | 日期 | 当前库存值的原始观测日期。 |
| `stock_age_days` | 整数 | 交易日距离库存观测日期的自然日数。 |
| `available_stock_current` | 数值 | 当前可用库存值，并沿用至下一次非空观测。 |
| `available_stock_unit` | 文本 | 可用库存单位。 |
| `available_stock_observation_date` | 日期 | 当前可用库存值的原始观测日期。 |
| `available_stock_age_days` | 整数 | 交易日距离可用库存观测日期的自然日数。 |

库存为空表示原始数据未提供对应序列，不代表库存为 0。

## 9. 原主力选择信息

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `main_selection_method` | 文本 | 固定为 `original_tushare_continuous_series`，表示直接采用原 ZIP 主力连续序列。 |
| `selection_signal_date` | 日期 | 原数据未提供主力选择信号日期，因此为空。 |
| `selection_signal_open_interest` | 数值 | 原数据未公开主力选择算法，因此为空。 |
| `selection_signal_volume` | 数值 | 原数据未公开主力选择算法，因此为空。 |
| `main_selection_fallback` | 0/1 | 0 表示全字段精确映射；1 表示实际合约映射使用了降级匹配或歧义消解。 |
| `main_initialization_flag` | 0/1 | 1 表示该品种第一条连续主力记录。 |

TuShare 提供连续合约映射接口，但没有公开主力选择算法或当天映射的确定时间。因此不能根据这些字段声称主力映射在当日开盘前已知。

## 10. 换月字段

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `main_contract_changed` | 0/1 | 1 表示映射实际合约相对上一条该品种记录发生变化。 |
| `previous_main_ts_code` | 文本 | 换月前的实际主力合约代码。 |
| `main_days_to_delist` | 整数 | 当前交易日距离映射实际合约退市日的自然日数。 |
| `old_contract_open_on_roll` | 数值 | 换月日旧主力实际合约的开盘价。 |
| `new_contract_open_on_roll` | 数值 | 换月日新主力实际合约的开盘价。 |
| `roll_open_spread` | 数值 | 新主力开盘价减旧主力开盘价。它是期限价差，不等同于交易成本。 |
| `estimated_roll_exchange_fee` | 数值 | 换月时平旧仓并开新仓的一手交易所手续费估计。 |

`estimated_roll_exchange_fee` 不包含期货公司加收费用、买卖价差、滑点和冲击成本。费率或旧合约行情不足时为空。

## 11. 涨跌停辅助字段

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `suspected_price_limit_lock` | 0/1 | 1 表示主力有成交、开高低收完全相同，且价格相对前结算价发生变化，属于疑似封板。 |
| `suspected_price_limit_direction` | 文本 | 疑似封板方向：`up` 或 `down`；未触发时为空。 |

原始数据没有交易所公布的每日涨停价和跌停价，因此这里只能做疑似判断，不能替代精确涨跌停数据。

## 12. 次近月合约字段

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `next_contract_ts_code` | 文本 | 当前实际主力之后到期时间最近、且当日有成交或持仓的实际合约代码。 |
| `next_contract_month` | 文本 | 次近月合约月份。 |
| `next_contract_delist_date` | 日期 | 次近月合约退市日期。 |
| `next_contract_close` | 数值 | 次近月合约当日收盘价。 |
| `next_contract_settle` | 数值 | 次近月合约当日结算价。 |
| `next_contract_volume` | 数值 | 次近月合约当日成交量。 |
| `next_contract_open_interest` | 数值 | 次近月合约当日持仓量。 |
| `next_contract_maturity_gap_days` | 整数 | 次近月与当前主力退市日期之间的自然日数。 |

这些字段用于构造期限结构、Carry 和展期收益。价格应使用未复权的真实合约价格。

## 13. 建模注意事项

- 不要直接对连续 `close` 求相邻收益，换月日可能包含机械价格跳变。
- 动量、波动率和偏度应使用处理过换月的品种收益序列。
- Carry 和期限结构应保留主力与次近月之间的真实价格差。
- 计算基差前必须检查 `spot_price_unit` 与 `quote_unit` 是否一致。
- 所有 `*_age_days` 字段都应纳入数据质量控制。
- 换月回测应使用实际合约字段，并另外加入滑点、冲击成本和无法成交处理。
- 本文件不包含未来收益标签。
