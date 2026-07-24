# 机器学习输入端报告

最终输入字段共 **92** 个。`ml_feature_registry.csv` 的 `feature_name` 列是唯一默认模型输入清单。

## 加工规则

- 品种级连续变量：同一交易日横截面中位数/MAD 标准化，截断后将缺失值置为横截面中性值 0。
- 市场级条件变量：只使用历史值形成滚动均值和标准差。
- 高于阈值的原始缺失率：附加独立缺失标记。
- 二元质量变量：保留 0/1。
- 交互项：只采用协议允许、经济含义预注册的少量乘积项。
- 不使用标签决定最终输入字段；相关聚类只提供诊断和代表字段，不物理删除。
- 产业链残差、板块残差和交互项按预注册经济公式实现；不实施全样本目标驱动的统一正交化。

## 全部模型输入

| feature_name | 来源 | 变换 | 类别 | 用途 | 相关簇 | 代表变量 | 经济含义 |
|---|---|---|---|---|---:|---|---|
| `tsmom_60__csz` | `tsmom_60` | same_date_cross_sectional_median_MAD_zscore_clip_5_missing_to_cross_sectional_median | trend_momentum | predictor | 1 | `tsmom_60__csz` | 波动率调整后的中期趋势延续 |
| `tsmom_60__missing` | `tsmom_60` | raw_missing_indicator | data_quality | quality_flag | 2 | `tsmom_60__missing` | 时间序列动量原始值不可用标记 |
| `csmom_60__csz` | `csmom_60` | same_date_cross_sectional_median_MAD_zscore_clip_5_missing_to_cross_sectional_median | trend_momentum | predictor | 3 | `csmom_60__csz` | 同日商品过去收益相对强弱 |
| `csmom_60__missing` | `csmom_60` | raw_missing_indicator | data_quality | quality_flag | 2 | `tsmom_60__missing` | 横截面动量原始值不可用标记 |
| `trend_efficiency_60__csz` | `trend_efficiency_60` | same_date_cross_sectional_median_MAD_zscore_clip_5_missing_to_cross_sectional_median | trend_momentum | predictor | 4 | `trend_efficiency_60__csz` | 价格路径的单向和平滑程度 |
| `trend_efficiency_60__missing` | `trend_efficiency_60` | raw_missing_indicator | data_quality | quality_flag | 2 | `tsmom_60__missing` | 趋势效率原始值不可用标记 |
| `multi_horizon_trend__csz` | `multi_horizon_trend` | same_date_cross_sectional_median_MAD_zscore_clip_5_missing_to_cross_sectional_median | trend_momentum | predictor | 5 | `multi_horizon_trend__csz` | 20/60/120日趋势方向一致程度 |
| `multi_horizon_trend__missing` | `multi_horizon_trend` | raw_missing_indicator | data_quality | quality_flag | 6 | `multi_horizon_trend__missing` | 多周期趋势一致性原始值不可用标记 |
| `carry_annualized__csz` | `carry_annualized` | same_date_cross_sectional_median_MAD_zscore_clip_5_missing_to_cross_sectional_median | term_structure | predictor | 7 | `carry_annualized__csz` | 完整可交易曲线最近两个合约之间的年化期限结构补偿 |
| `carry_annualized__missing` | `carry_annualized` | raw_missing_indicator | data_quality | quality_flag | 8 | `carry_annualized__missing` | Carry因子原始值不可用标记 |
| `carry_change_20__csz` | `carry_change_20` | same_date_cross_sectional_median_MAD_zscore_clip_5_missing_to_cross_sectional_median | term_structure | predictor | 9 | `carry_change_20__csz` | 期限结构二十日边际变化 |
| `carry_change_20__missing` | `carry_change_20` | raw_missing_indicator | data_quality | quality_flag | 10 | `carry_change_20__missing` | Carry变化原始值不可用标记 |
| `basis_fresh__csz` | `basis_fresh` | same_date_cross_sectional_median_MAD_zscore_clip_5_missing_to_cross_sectional_median | basis_spot | predictor | 11 | `basis_fresh__csz` | 新鲜现货相对主力期货的强弱 |
| `basis_fresh__missing` | `basis_fresh` | raw_missing_indicator | data_quality | quality_flag | 12 | `basis_fresh__missing` | 基差原始值不可用标记 |
| `basis_shock_20__csz` | `basis_shock_20` | same_date_cross_sectional_median_MAD_zscore_clip_5_missing_to_cross_sectional_median | basis_spot | predictor | 13 | `basis_shock_20__csz` | 基差日变化相对历史条件分布的异常 |
| `basis_shock_20__missing` | `basis_shock_20` | raw_missing_indicator | data_quality | quality_flag | 12 | `basis_fresh__missing` | 基差冲击原始值不可用标记 |
| `basis_momentum_20__csz` | `basis_momentum_20` | same_date_cross_sectional_median_MAD_zscore_clip_5_missing_to_cross_sectional_median | basis_spot | predictor | 14 | `basis_momentum_20__csz` | 新鲜可用基差的二十日边际变化 |
| `basis_momentum_20__missing` | `basis_momentum_20` | raw_missing_indicator | data_quality | quality_flag | 12 | `basis_fresh__missing` | 基差动量原始值不可用标记 |
| `curve_curvature__csz` | `curve_curvature` | same_date_cross_sectional_median_MAD_zscore_clip_5_missing_to_cross_sectional_median | term_structure | predictor | 15 | `curve_curvature__csz` | 曲线第二个合约相对第一和第三个合约线性插值的弯曲程度 |
| `curve_curvature__missing` | `curve_curvature` | raw_missing_indicator | data_quality | quality_flag | 16 | `curve_curvature__missing` | 期限结构曲率原始值不可用标记 |
| `value_252__csz` | `value_252` | same_date_cross_sectional_median_MAD_zscore_clip_5_missing_to_cross_sectional_median | value_reversion | predictor | 17 | `value_252__csz` | 价格相对一年滚动历史锚点的反向偏离 |
| `value_252__missing` | `value_252` | raw_missing_indicator | data_quality | quality_flag | 18 | `value_252__missing` | 价值因子原始值不可用标记 |
| `short_reversal_5__csz` | `short_reversal_5` | same_date_cross_sectional_median_MAD_zscore_clip_5_missing_to_cross_sectional_median | value_reversion | predictor | 19 | `short_reversal_5__csz` | 过去五日过度反应后的修复 |
| `inventory_scarcity__csz` | `inventory_scarcity` | same_date_cross_sectional_median_MAD_zscore_clip_5_missing_to_cross_sectional_median | inventory_fundamental | predictor | 20 | `inventory_scarcity__csz` | 库存相对历史同期的稀缺程度 |
| `inventory_scarcity__missing` | `inventory_scarcity` | raw_missing_indicator | data_quality | quality_flag | 21 | `inventory_scarcity__missing` | 库存稀缺因子原始值不可用标记 |
| `inventory_change_5__csz` | `inventory_change_5` | same_date_cross_sectional_median_MAD_zscore_clip_5_missing_to_cross_sectional_median | inventory_fundamental | predictor | 22 | `inventory_change_5__csz` | 五日库存下降所表示的边际趋紧 |
| `inventory_change_5__missing` | `inventory_change_5` | raw_missing_indicator | data_quality | quality_flag | 23 | `inventory_change_5__missing` | 库存变化因子原始值不可用标记 |
| `inventory_surprise__csz` | `inventory_surprise` | same_date_cross_sectional_median_MAD_zscore_clip_5_missing_to_cross_sectional_median | information_news | predictor | 24 | `inventory_surprise__csz` | 库存相对仅用历史信息形成的条件预期的标准化意外 |
| `inventory_surprise__missing` | `inventory_surprise` | raw_missing_indicator | data_quality | quality_flag | 25 | `inventory_surprise__missing` | 库存意外／库存新息原始值不可用标记 |
| `warehouse_tightness__csz` | `warehouse_tightness` | same_date_cross_sectional_median_MAD_zscore_clip_5_missing_to_cross_sectional_median | inventory_fundamental | predictor | 26 | `warehouse_tightness__csz` | 注册仓单相对历史同期的紧张程度 |
| `warehouse_tightness__missing` | `warehouse_tightness` | raw_missing_indicator | data_quality | quality_flag | 27 | `warehouse_tightness__missing` | 仓单紧张因子原始值不可用标记 |
| `tightness_composite__csz` | `tightness_composite` | same_date_cross_sectional_median_MAD_zscore_clip_5_missing_to_cross_sectional_median | inventory_fundamental | predictor | 28 | `tightness_composite__csz` | 仓单、库存、基差和Carry的等权看多方向组合 |
| `tightness_composite__missing` | `tightness_composite` | raw_missing_indicator | data_quality | quality_flag | 29 | `tightness_composite__missing` | 供需紧张度原始值不可用标记 |
| `oi_growth_20__csz` | `oi_growth_20` | same_date_cross_sectional_median_MAD_zscore_clip_5_missing_to_cross_sectional_median | trading_liquidity | conditional | 30 | `oi_growth_20__csz` | 未平仓合约数量的二十日增长 |
| `oi_growth_20__missing` | `oi_growth_20` | raw_missing_indicator | data_quality | quality_flag | 31 | `oi_growth_20__missing` | 持仓量增长原始值不可用标记 |
| `price_oi_interaction_20__csz` | `price_oi_interaction_20` | same_date_cross_sectional_median_MAD_zscore_clip_5_missing_to_cross_sectional_median | trading_liquidity | predictor | 32 | `price_oi_interaction_20__csz` | 价格方向和新增持仓方向的联合变化 |
| `price_oi_interaction_20__missing` | `price_oi_interaction_20` | raw_missing_indicator | data_quality | quality_flag | 33 | `price_oi_interaction_20__missing` | 价格—持仓量交互原始值不可用标记 |
| `volume_shock_20__csz` | `volume_shock_20` | same_date_cross_sectional_median_MAD_zscore_clip_5_missing_to_cross_sectional_median | trading_liquidity | conditional | 34 | `volume_shock_20__csz` | 成交量相对历史正常水平的异常 |
| `volume_shock_20__missing` | `volume_shock_20` | raw_missing_indicator | data_quality | quality_flag | 35 | `volume_shock_20__missing` | 成交量冲击原始值不可用标记 |
| `amihud_20__csz` | `amihud_20` | same_date_cross_sectional_median_MAD_zscore_clip_5_missing_to_cross_sectional_median | trading_liquidity | risk | 36 | `amihud_20__csz` | 单位成交额对应的绝对价格变动 |
| `amihud_20__missing` | `amihud_20` | raw_missing_indicator | data_quality | quality_flag | 33 | `price_oi_interaction_20__missing` | Amihud非流动性原始值不可用标记 |
| `turnover_oi__csz` | `turnover_oi` | same_date_cross_sectional_median_MAD_zscore_clip_5_missing_to_cross_sectional_median | trading_liquidity | conditional | 37 | `turnover_oi__csz` | 相对存量持仓的交易活跃程度 |
| `realized_volatility_20__csz` | `realized_volatility_20` | same_date_cross_sectional_median_MAD_zscore_clip_5_missing_to_cross_sectional_median | volatility_risk | risk | 38 | `realized_volatility_20__csz` | 过去二十日收益波动风险 |
| `realized_volatility_20__missing` | `realized_volatility_20` | raw_missing_indicator | data_quality | quality_flag | 33 | `price_oi_interaction_20__missing` | 已实现波动率原始值不可用标记 |
| `low_volatility_20__csz` | `low_volatility_20` | same_date_cross_sectional_median_MAD_zscore_clip_5_missing_to_cross_sectional_median | volatility_risk | risk | 38 | `realized_volatility_20__csz` | 横截面低波动品种得分更高 |
| `low_volatility_20__missing` | `low_volatility_20` | raw_missing_indicator | data_quality | quality_flag | 33 | `price_oi_interaction_20__missing` | 低波动因子原始值不可用标记 |
| `realized_skewness_60__csz` | `realized_skewness_60` | same_date_cross_sectional_median_MAD_zscore_clip_5_missing_to_cross_sectional_median | volatility_risk | risk | 39 | `realized_skewness_60__csz` | 过去六十日收益分布非对称性 |
| `realized_skewness_60__missing` | `realized_skewness_60` | raw_missing_indicator | data_quality | quality_flag | 2 | `tsmom_60__missing` | 已实现偏度原始值不可用标记 |
| `seasonality_3y__csz` | `seasonality_3y` | same_date_cross_sectional_median_MAD_zscore_clip_5_missing_to_cross_sectional_median | value_reversion | predictor | 40 | `seasonality_3y__csz` | 过去三年近似同期五日收益的平均 |
| `seasonality_3y__missing` | `seasonality_3y` | raw_missing_indicator | data_quality | quality_flag | 41 | `seasonality_3y__missing` | 季节性因子原始值不可用标记 |
| `sector_relative_value_60__csz` | `sector_relative_value_60` | same_date_cross_sectional_median_MAD_zscore_clip_5_missing_to_cross_sectional_median | industry_chain | predictor | 42 | `sector_relative_value_60__csz` | 品种六十日收益相对板块共同收益的偏离 |
| `sector_relative_value_60__missing` | `sector_relative_value_60` | raw_missing_indicator | data_quality | quality_flag | 2 | `tsmom_60__missing` | 相对价值残差原始值不可用标记 |
| `factor_momentum__tsz` | `factor_momentum` | prior_only_rolling_date_zscore_window_252_minimum_60_clip_5_missing_to_zero | meta_factor | conditional | 43 | `factor_momentum__tsz` | 四个冻结基础因子组合最近二十日已实现收益的持续性 |
| `factor_momentum__missing` | `factor_momentum` | raw_missing_indicator | data_quality | quality_flag | 44 | `factor_momentum__missing` | 因子动量原始值不可用标记 |
| `fundamental_composite_news__csz` | `fundamental_composite_news` | same_date_cross_sectional_median_MAD_zscore_clip_5_missing_to_cross_sectional_median | information_news | predictor | 45 | `fundamental_composite_news__csz` | 库存、仓单和基差中至少两类历史模型意外的统一方向组合 |
| `fundamental_composite_news__missing` | `fundamental_composite_news` | raw_missing_indicator | data_quality | quality_flag | 12 | `basis_fresh__missing` | 基本面综合新息代理原始值不可用标记 |
| `fundamental_underreaction__csz` | `fundamental_underreaction` | same_date_cross_sectional_median_MAD_zscore_clip_5_missing_to_cross_sectional_median | information_news | predictor | 46 | `fundamental_underreaction__csz` | 模型隐含基本面新息扣除首个可交易日价格反应后的未吸收部分 |
| `fundamental_underreaction__missing` | `fundamental_underreaction` | raw_missing_indicator | data_quality | quality_flag | 12 | `basis_fresh__missing` | 基本面反应不足代理原始值不可用标记 |
| `information_consistency__csz` | `information_consistency` | same_date_cross_sectional_median_MAD_zscore_clip_5_missing_to_cross_sectional_median | information_news | predictor | 28 | `tightness_composite__csz` | 仓单、库存、基差和Carry方向一致时放大的综合信号 |
| `information_consistency__missing` | `information_consistency` | raw_missing_indicator | data_quality | quality_flag | 29 | `tightness_composite__missing` | 信息一致性原始值不可用标记 |
| `information_divergence__csz` | `information_divergence` | same_date_cross_sectional_median_MAD_zscore_clip_5_missing_to_cross_sectional_median | information_news | predictor | 47 | `information_divergence__csz` | 供需基本面信号相对价格趋势的未反映差异 |
| `information_divergence__missing` | `information_divergence` | raw_missing_indicator | data_quality | quality_flag | 29 | `tightness_composite__missing` | 信息背离原始值不可用标记 |
| `warehouse_surprise__csz` | `warehouse_surprise` | same_date_cross_sectional_median_MAD_zscore_clip_5_missing_to_cross_sectional_median | information_news | predictor | 48 | `warehouse_surprise__csz` | 注册仓单相对历史条件预期的标准化意外 |
| `warehouse_surprise__missing` | `warehouse_surprise` | raw_missing_indicator | data_quality | quality_flag | 49 | `warehouse_surprise__missing` | 仓单新息原始值不可用标记 |
| `momentum_quality_60__csz` | `momentum_quality_60` | same_date_cross_sectional_median_MAD_zscore_clip_5_missing_to_cross_sectional_median | trend_momentum | predictor | 50 | `momentum_quality_60__csz` | 以趋势效率过滤中期动量路径噪声 |
| `momentum_quality_60__missing` | `momentum_quality_60` | raw_missing_indicator | data_quality | quality_flag | 2 | `tsmom_60__missing` | 动量质量因子原始值不可用标记 |
| `vol_managed_momentum__csz` | `vol_managed_momentum` | same_date_cross_sectional_median_MAD_zscore_clip_5_missing_to_cross_sectional_median | trend_momentum | predictor | 51 | `vol_managed_momentum__csz` | 按短期风险缩放中期趋势方向 |
| `vol_managed_momentum__missing` | `vol_managed_momentum` | raw_missing_indicator | data_quality | quality_flag | 2 | `tsmom_60__missing` | 波动率管理动量原始值不可用标记 |
| `idiosyncratic_volatility_60__csz` | `idiosyncratic_volatility_60` | same_date_cross_sectional_median_MAD_zscore_clip_5_missing_to_cross_sectional_median | volatility_risk | risk | 52 | `idiosyncratic_volatility_60__csz` | 剔除商品市场及板块共同收益后的剩余波动 |
| `idiosyncratic_volatility_60__missing` | `idiosyncratic_volatility_60` | raw_missing_indicator | data_quality | quality_flag | 53 | `idiosyncratic_volatility_60__missing` | 特质波动因子原始值不可用标记 |
| `term_structure_slope__csz` | `term_structure_slope` | same_date_cross_sectional_median_MAD_zscore_clip_5_missing_to_cross_sectional_median | term_structure | predictor | 54 | `term_structure_slope__csz` | 用完整可交易曲线加权拟合的年化斜率 |
| `term_structure_slope__missing` | `term_structure_slope` | raw_missing_indicator | data_quality | quality_flag | 16 | `curve_curvature__missing` | 期限结构斜率原始值不可用标记 |
| `term_structure_momentum__csz` | `term_structure_momentum` | same_date_cross_sectional_median_MAD_zscore_clip_5_missing_to_cross_sectional_median | term_structure | predictor | 55 | `term_structure_momentum__csz` | 整条期限结构斜率的二十日变化 |
| `term_structure_momentum__missing` | `term_structure_momentum` | raw_missing_indicator | data_quality | quality_flag | 56 | `term_structure_momentum__missing` | 期限结构动量原始值不可用标记 |
| `term_structure_acceleration__csz` | `term_structure_acceleration` | same_date_cross_sectional_median_MAD_zscore_clip_5_missing_to_cross_sectional_median | term_structure | predictor | 57 | `term_structure_acceleration__csz` | 整条期限结构二十日变化的再二十日变化 |
| `term_structure_acceleration__missing` | `term_structure_acceleration` | raw_missing_indicator | data_quality | quality_flag | 58 | `term_structure_acceleration__missing` | 期限结构加速度原始值不可用标记 |
| `basis_surprise__csz` | `basis_surprise` | same_date_cross_sectional_median_MAD_zscore_clip_5_missing_to_cross_sectional_median | basis_spot | predictor | 59 | `basis_surprise__csz` | 基差相对历史条件预期的标准化意外 |
| `basis_surprise__missing` | `basis_surprise` | raw_missing_indicator | data_quality | quality_flag | 12 | `basis_fresh__missing` | 基差意外原始值不可用标记 |
| `inventory_acceleration__csz` | `inventory_acceleration` | same_date_cross_sectional_median_MAD_zscore_clip_5_missing_to_cross_sectional_median | inventory_fundamental | predictor | 60 | `inventory_acceleration__csz` | 去库存速度的二阶变化 |
| `inventory_acceleration__missing` | `inventory_acceleration` | raw_missing_indicator | data_quality | quality_flag | 23 | `inventory_change_5__missing` | 库存加速度原始值不可用标记 |
| `inventory_price_divergence__csz` | `inventory_price_divergence` | same_date_cross_sectional_median_MAD_zscore_clip_5_missing_to_cross_sectional_median | information_news | predictor | 22 | `inventory_change_5__csz` | 库存趋紧相对同期价格变化的未反映部分 |
| `inventory_price_divergence__missing` | `inventory_price_divergence` | raw_missing_indicator | data_quality | quality_flag | 23 | `inventory_change_5__missing` | 库存—价格背离原始值不可用标记 |
| `oi_surprise_60__csz` | `oi_surprise_60` | same_date_cross_sectional_median_MAD_zscore_clip_5_missing_to_cross_sectional_median | trading_liquidity | conditional | 61 | `oi_surprise_60__csz` | 持仓变化相对历史条件预期的异常 |
| `oi_surprise_60__missing` | `oi_surprise_60` | raw_missing_indicator | data_quality | quality_flag | 31 | `oi_growth_20__missing` | 持仓意外因子原始值不可用标记 |
| `volume_price_divergence_20__csz` | `volume_price_divergence_20` | same_date_cross_sectional_median_MAD_zscore_clip_5_missing_to_cross_sectional_median | trading_liquidity | predictor | 62 | `volume_price_divergence_20__csz` | 价格趋势与成交量变化不一致程度 |
| `volume_price_divergence_20__missing` | `volume_price_divergence_20` | raw_missing_indicator | data_quality | quality_flag | 33 | `price_oi_interaction_20__missing` | 量价背离因子原始值不可用标记 |
| `momentum_carry_interaction__csz` | `momentum_carry_interaction` | same_date_cross_sectional_median_MAD_zscore_clip_5_missing_to_cross_sectional_median | factor_interaction | predictor | 63 | `momentum_carry_interaction__csz` | 动量和Carry同时确认的非加性信息 |
| `momentum_carry_interaction__missing` | `momentum_carry_interaction` | raw_missing_indicator | data_quality | quality_flag | 64 | `momentum_carry_interaction__missing` | 因子交互因子原始值不可用标记 |
| `nonlinear_tightness__csz` | `nonlinear_tightness` | same_date_cross_sectional_median_MAD_zscore_clip_5_missing_to_cross_sectional_median | factor_interaction | predictor | 28 | `tightness_composite__csz` | 极端供需紧张状态的平方增强 |
| `nonlinear_tightness__missing` | `nonlinear_tightness` | raw_missing_indicator | data_quality | quality_flag | 29 | `tightness_composite__missing` | 非线性基本面因子原始值不可用标记 |
| `correlation_risk_20_120__csz` | `correlation_risk_20_120` | same_date_cross_sectional_median_MAD_zscore_clip_5_missing_to_cross_sectional_median | volatility_risk | risk | 65 | `correlation_risk_20_120__csz` | 品种与商品市场短期相关性相对长期基准的上升 |
| `correlation_risk_20_120__missing` | `correlation_risk_20_120` | raw_missing_indicator | data_quality | quality_flag | 6 | `multi_horizon_trend__missing` | 相关性风险因子原始值不可用标记 |

线性回归不建议直接使用普通 OLS 解释所有高度相关系数；首选 Ridge，或仅在各训练折内使用相关簇代表变量。LightGBM/XGBoost 可使用注册表中全部特征，但仍应固定滚动折与早停规则。
