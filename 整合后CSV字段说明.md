# 整合后 CSV 字段说明

适用文件：`futures_contract_daily_30varieties_2015_2025.csv`

数据粒度为“一份实际期货合约在一个交易日一行”，唯一主键为 `trade_date + ts_code`。文件共 53 个字段。

## 1. 主键与品种标识

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `trade_date` | 日期 | 交易日，格式为 `YYYY-MM-DD`。 |
| `ts_code` | 文本 | 实际期货合约代码，如 `RB2510.SHF`。与 `trade_date` 共同构成唯一主键。 |
| `product_code` | 文本 | 期货品种代码，如 `RB`、`CU`、`M`。 |
| `product_name` | 文本 | 期货品种中文名称，如螺纹钢、铜、豆粕。 |
| `exchange` | 文本 | 交易所代码，如 `SHF`、`DCE`、`ZCE`、`INE`。 |

## 2. 合约元数据

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `contract` | 文本 | 不含交易所后缀的实际合约代码，如 `RB2510`。 |
| `contract_month` | 文本 | 从合约代码中提取的到期月份数字。郑商所早期合约可能使用三位月份编码。 |
| `list_date` | 日期 | 合约上市日期。 |
| `delist_date` | 日期 | 合约退市或最后交易日期。 |
| `last_delivery_date` | 日期 | 最后交割日期。 |
| `trade_unit` | 文本 | 合约交易单位的名称，如吨。 |
| `contract_size` | 数值 | 每手合约对应的标的数量，来自原始合约元数据的 `per_unit`。 |
| `quote_unit` | 文本 | 报价单位，如人民币元/吨。 |
| `quote_unit_desc` | 文本 | 最小报价单位的文字说明。 |
| `delivery_mode` | 文本 | 交割方式，如实物交割。 |

## 3. 期货日行情

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `pre_close` | 数值 | 前一交易日收盘价。 |
| `pre_settle` | 数值 | 前一交易日结算价。 |
| `open` | 数值 | 当日开盘价。 |
| `high` | 数值 | 当日最高价。 |
| `low` | 数值 | 当日最低价。 |
| `close` | 数值 | 当日收盘价。 |
| `settle` | 数值 | 当日结算价。 |
| `volume` | 数值 | 当日成交量，沿用原始期货行情口径。 |
| `amount` | 数值 | 当日成交金额，沿用原始数据的数值和单位口径，未进行统一换算。 |
| `open_interest` | 数值 | 当日持仓量。 |

价格字段的单位由 `quote_unit` 和 `quote_unit_desc` 说明。停牌、未成交或源数据缺失时，部分行情字段可能为空。

## 4. 手续费与保证金

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `trading_fee_rate` | 数值 | 按成交金额计费时使用的手续费率，单位为千分比（‰）。例如 `0.1` 表示 0.1‰，即万分之一。 |
| `trading_fee` | 数值 | 按固定金额计费时使用的每手手续费，通常单位为元/手。 |
| `delivery_fee` | 数值 | 交割手续费。 |
| `buy_hedging_margin_rate` | 数值 | 买入套期保值保证金比例。 |
| `sell_hedging_margin_rate` | 数值 | 卖出套期保值保证金比例。 |
| `long_margin_rate` | 数值 | 多头投机保证金比例。 |
| `short_margin_rate` | 数值 | 空头投机保证金比例。 |

这些字段按 `trade_date + ts_code` 从期货结算参数表合并。源表未提供对应记录时为空。

两种手续费字段通常对应两种不同的计费方式：

- 按成交金额计费：`单边每手手续费 = 成交价 × contract_size × trading_fee_rate ÷ 1000`
- 按固定金额计费：`单边每手手续费 = trading_fee`

例如螺纹钢的 `trading_fee_rate=0.1` 表示成交金额的 0.1‰；原油的 `trading_fee=20` 表示固定金额 20 元/手。二者通常是互斥口径，不应直接相加。实际交易成本还可能受到开仓、平仓、平今、交易所调整和期货公司加收费用的影响，当前字段不能完整表达这些差异。

## 5. 现货价格

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `spot_price` | 数值 | 对应品种的统一现货基准价格。没有当日报价时，沿用最近一次按“日期”记录的价格。 |
| `spot_price_unit` | 文本 | 现货价格的原始报价单位。不同品种可能不同，如元/吨、元/克或美元/吨。 |
| `spot_indicator_code` | 文本 | 被选作现货基准的原始指标代码。 |
| `spot_indicator_name` | 文本 | 被选作现货基准的原始指标名称。 |
| `spot_price_source` | 文本 | 现货指标的数据来源。 |
| `spot_observation_date` | 日期 | 当前 `spot_price` 在原始文件中的实际观测日期。 |
| `spot_price_age_days` | 整数 | `trade_date` 与 `spot_observation_date` 相差的自然日数。0 表示当日有新报价，数值越大表示价格越陈旧。 |

各品种使用的具体现货指标见 `selected_spot_benchmarks.csv`。原始现货数据截至 2023 年，因此 2024—2025 年的价格会继续沿用，建模时应结合 `spot_price_age_days` 使用。

## 6. 仓单

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `warehouse_receipt_current` | 数值 | 当前仓单数量，优先采用日度仓单记录，并向后沿用至下一次观测。 |
| `warehouse_receipt_change` | 数值 | 当前仓单数量相对上一条仓单观测的变化，由 `warehouse_receipt_current` 重新计算。 |
| `warehouse_receipt_unit` | 文本 | 仓单数量单位，如张、手或吨。 |
| `warehouse_receipt_observation_date` | 日期 | 当前仓单值的原始观测日期。 |
| `warehouse_receipt_age_days` | 整数 | 交易日距离仓单原始观测日期的自然日数。 |
| `warehouse_receipt_conflict` | 0/1 | 同一品种同一天的日度仓单记录与周期库存记录所带仓单值是否冲突。1 表示冲突，0 表示未发现冲突。 |

`warehouse_receipt_conflict` 会随被沿用的仓单观测一起保留。它表示源数据同日存在不同仓单值，并不表示当前 CSV 出现重复行。

## 7. 库存与可用库存

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `stock_current` | 数值 | 周期库存记录中的当前库存值，并向后沿用至下一次非空观测。 |
| `stock_unit` | 文本 | 当前库存的单位。 |
| `stock_observation_date` | 日期 | 当前库存值的原始观测日期。 |
| `stock_age_days` | 整数 | 交易日距离库存原始观测日期的自然日数。 |
| `available_stock_current` | 数值 | 周期库存记录中的当前可用库存值，并向后沿用。 |
| `available_stock_unit` | 文本 | 当前可用库存的单位。 |
| `available_stock_observation_date` | 日期 | 当前可用库存值的原始观测日期。 |
| `available_stock_age_days` | 整数 | 交易日距离可用库存原始观测日期的自然日数。 |

库存字段为空表示原始仓单库存文件没有为该品种提供对应库存序列，不代表库存为 0。

## 8. 建模使用提示

- CSV 不包含预测标签，需要在建模阶段根据实际交易规则另行生成。
- 同一品种同一天可能有多份实际合约，因此品种级的现货、仓单和库存值会出现在多个合约行中。
- 现货、仓单和库存均只使用观测日期不晚于交易日的数据。
- 建议将各类 `*_age_days` 字段同时用于缺失质量控制，避免把长期沿用的旧值视为新数据。
- `spot_price_unit` 和期货 `quote_unit` 不一定一致，计算基差前必须先检查并统一单位。
