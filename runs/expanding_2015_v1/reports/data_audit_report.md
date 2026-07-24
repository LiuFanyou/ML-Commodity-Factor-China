# 扩展窗口版本数据审计报告

> 本报告由项目真实数据自动生成，不补造缺失口径。

- 数据版本：`CORE30_V1`。
- 数据范围：2015-01-05 至 2025-12-31。
- 品种数：30；品种日记录：77,206。
- 可交易日收益覆盖率：99.36%。
- 期限结构品种日：76,875；至少2个合格合约占比：99.60%。
- 因子形成时间为T日收盘后，最早T+1开盘使用。
- 每个外层测试折只使用测试年前的数据筛选因子和训练模型。

## 数据可用性

| data_type          | source                           | strict_point_in_time_usable   | coverage   | note                                 |
|:-------------------|:---------------------------------|:------------------------------|:-----------|:-------------------------------------|
| 价格/OHLC          | market_contract_daily            | 是                            | 2015-2025  | 严格点时可用                         |
| 成交量             | market_contract_daily            | 是                            | 2015-2025  | 严格点时可用                         |
| 成交额             | market_contract_daily            | 是                            | 2015-2025  | 原始单位保留，适合品种内标准化       |
| 持仓量             | market_contract_daily            | 是                            | 2015-2025  | 严格点时可用                         |
| 全合约期限结构     | market_contract_daily            | 是                            | 2015-2025  | 动态流动性准入                       |
| 主力选择           | main_selection_daily             | 是                            | 2015-2025  | 当日收盘选择，下一交易日可用         |
| 可交易基础收益     | research_returns_daily           | 是                            | 2015-2025  | 前日选择合约的次日开收收益           |
| 真实现货价格       | 无                               | 否                            | 无         | 旧现货文件实为会员成本               |
| 新仓单             | warehouse_receipts_daily_new     | 否                            | 2015-2025  | 真实公布时间未确认                   |
| 新库存/可用量      | warehouse_inventory_periodic_new | 否                            | 2015-2025  | 公布时间及字段语义未确认             |
| 旧库存             | legacy_inventory_weekly          | 否                            | 2020-2025  | 仅有假定可用日，单位和语义部分未确认 |
| 旧仓单             | legacy_warehouse_receipts_weekly | 否                            | 2020-2025  | 覆盖8个品种且仅有假定可用日          |
| 商业交易者分类持仓 | 无                               | 否                            | 无         | 会员成本不能替代持仓份额             |
| 产业链配比/成本    | 无                               | 否                            | 无         | 缺少确认后的产业图、配比和成本       |
| 手续费与合约乘数   | contract_rules/metadata          | 否                            | 2015-2025  | 手续费单位及乘数未完成核验           |

## 仍未解决的边界

1. 仓单、库存和现货缺少可核验的真实发布时间或字段口径，因此相关概念因子仍为F类。
2. 手续费单位、合约乘数、涨跌停可成交性、盘口冲击和容量未核验，成本仅作1/3/5bp敏感性。
3. 产业链映射、生产配比和商业交易者分类持仓不足。

## 点时审计

```json
{
  "selection_available_strictly_after_signal_date": true,
  "selection_bad_row_count": 0,
  "panel_key_unique": true,
  "fundamental_tables_loaded_into_strict_panel": false,
  "phase": "development",
  "panel_date_min": "2015-01-05",
  "panel_date_max": "2025-12-31",
  "development_excludes_sealed_period": true
}
```
