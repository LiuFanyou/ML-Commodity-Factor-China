# data_final_raw 数据审计报告

## 审计结论

- 本轮只使用 `data_final_raw`，未读取 `data3` 或旧版 research 数据生成因子或标签。
- `trade_date × product` 主键通过唯一性检查。
- 按用户修订，读取 TuShare 精确连续代码并映射真实合约，不实施 T−1 持仓量重选主力。
- 完整期限结构来自全部实际合约，并与主力映射规则相互独立。
- 基本面统一延迟一个交易日后才可见；现货最大陈旧期7天。
- 2025 数据仅输出到独立评价目录，不进入开发训练交接包。

## 数据概况

| 项目 | 数值 |
|---|---:|
| `raw_contract_rows` | 846963 |
| `raw_contracts` | 3753 |
| `raw_product_days` | 78112 |
| `continuous_product_days` | 78112 |
| `point_in_time_violations` | {'spot_observation_date': 0, 'warehouse_receipt_observation_date': 0, 'stock_observation_date': 0, 'available_stock_observation_date': 0} |
| `spot_last_observation` | 2023-12-29 00:00:00 |
| `warehouse_last_observation` | 2025-12-31 00:00:00 |
| `stock_last_observation` | 2025-12-31 00:00:00 |
| `actual_fee_any_coverage` | 0.731564 |
| `mapped_rows` | 78112 |
| `columns` | 80 |
| `date_min` | 2015-01-05 |
| `date_max` | 2025-12-31 |
| `products` | 30 |
| `trade_dates` | 2674 |
| `duplicate_keys` | 0 |
| `development_rows` | 70822 |
| `sealed_2025_rows` | 7290 |
| `mapped_actual_contract_coverage` | 1.000000 |
| `mapping_exact_rate` | 0.949918 |
| `mapping_high_or_exact_rate` | 0.990040 |
| `mapping_min_score` | 2 |
| `mapping_signal_time_coverage` | 1.000000 |
| `product_return_coverage` | 0.997248 |
| `roll_rows` | 1500 |
| `roll_return_missing_rows` | 18 |
| `suspected_limit_rows` | 104 |
| `curve_two_point_coverage` | 0.983652 |
| `curve_three_point_coverage` | 0.823279 |
| `curve_median_contracts` | 4.000000 |
| `spot_fresh_usable_coverage` | 0.734125 |
| `warehouse_fresh_usable_coverage` | 0.987428 |
| `stock_fresh_usable_coverage` | 0.339487 |
| `stock_products` | 10 |
| `main_contract_rule` | direct_tushare_continuous_series_mapped_to_actual_contract |
| `t_minus_1_selection_required` | False |
| `fundamental_release_lag_trading_days` | 1 |
| `label_1d_coverage` | 0.986622 |
| `label_5d_coverage` | 0.984586 |
| `label_20d_coverage` | 0.928564 |

## 因子数据可用性

- 预注册原始因子：57 个。
- 通过覆盖率与非零方差硬门：47 个。
- 数据不足：10 个。

| 因子 | 覆盖率 | 原因 |
|---|---:|---|
| `hedging_pressure` | 0.0000 | data_final_raw没有商业/非商业交易者分类持仓 |
| `industry_chain_residual` | 0.0000 | 未提供经确认的产业链邻接表和生产传导系数 |
| `market_state_conditional` | 0.0000 | 当前协议未定义和冻结市场状态概率引擎 |
| `product_state_interaction` | 0.0000 | 当前协议未定义和冻结单品种状态引擎 |
| `cost_transmission_gap` | 0.0000 | 未提供经确认的产业链邻接表和实际生产配比 |
| `processing_margin_anomaly` | 0.0000 | 缺少经确认的生产配比和其他成本序列 |
| `transmission_residual` | 0.0000 | 未提供经确认的产业链邻接和传导滞后 |
| `supply_demand_shock` | 0.0000 | data_final_raw没有供给与需求多源实际值及预期字段 |
| `crowding` | 0.0000 | data_final_raw没有会员或参与者分项持仓 |
| `commodity_network_factor` | 0.0000 | 未提供经确认的商品网络邻接表与权重 |

## 已知数据限制

- 连续主力到真实合约依赖同日行情字段映射；极少数非完全匹配行通过冻结的确定性规则处理。
- `suspected_price_limit_lock` 是 OHLC 锁价代理，不等价于交易所逐日精确涨跌停价。
- 文件没有盘口深度、买卖价差和逐笔冲击数据，不能核验真实盘口冲击与容量。
- 2021年前真实手续费覆盖较弱，统一5bps只用于可比研究成本，不能声称为完整实盘净收益。
- 现货源截至2023年，2024—2025过期现货不会被继续当作有效基差。
