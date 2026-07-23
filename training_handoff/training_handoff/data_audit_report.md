# 数据审计报告

> 本报告由项目内真实数据自动生成。没有真实点时数据的项目不会被估算或补造。

## 审计结论

- 数据版本：`CORE30_V1`。
- 开发数据范围：2015-01-05 至 2024-12-31。
- 动态品种数：30；品种日记录：69,916。
- 可交易日收益覆盖率：99.29%。
- 期限结构品种日：69,585；至少2个合格合约占比：99.55%；至少3个占比：98.80%。
- 开发流程没有加载仓单、库存或所谓现货表进入严格因子矩阵。
- 主力选择在当日收盘后形成，最早下一交易日使用；点时检查已通过。
- 2025 年封存期在开发阶段被排除。

## 数据可用性清单

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

## 影响结论的数据边界

1. 新仓单和库存的 `available_date` 为空，全部 `point_in_time_usable=False`。
2. 旧库存/仓单只有推定的下一样本交易日，并非核验后的真实发布时间。
3. 原‘现货价格’文件实际是会员多空平均成本，不能计算基差。
4. 手续费单位和合约乘数未核验，主报告只提供毛收益与1/3/5基点敏感性，不声称是真实交易所净收益。
5. 产业链映射、生产配比、加工成本及商业交易者分类持仓缺失。

## 点时检查

```json
{
  "selection_available_strictly_after_signal_date": true,
  "selection_bad_row_count": 0,
  "panel_key_unique": true,
  "fundamental_tables_loaded_into_strict_panel": false,
  "phase": "development",
  "panel_date_min": "2015-01-05",
  "panel_date_max": "2024-12-31",
  "development_excludes_sealed_period": true
}
```

## 57个概念因子的当前分类数量

| 分类   |   因子数 |
|:-------|---------:|
| A      |        2 |
| B      |       11 |
| C      |        4 |
| D      |        7 |
| E      |        9 |
| F      |       24 |
