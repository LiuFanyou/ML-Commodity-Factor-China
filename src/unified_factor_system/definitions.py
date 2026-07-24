from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd


@dataclass(frozen=True)
class FactorDefinition:
    factor_no: int
    factor_id: str
    factor_name: str
    category: str
    protocol_role: str
    usage_type: str
    economic_meaning: str
    formula: str
    expected_direction: str
    data_source: str
    implementation_status: str = "computed"
    unavailable_reason: str = ""
    binary: bool = False


def _factor(
    number: int,
    factor_id: str,
    name: str,
    category: str,
    usage: str,
    meaning: str,
    formula: str,
    direction: str,
    source: str,
    status: str = "computed",
    reason: str = "",
) -> FactorDefinition:
    core_categories = {"trend_momentum", "term_structure", "trading_liquidity"}
    return FactorDefinition(
        factor_no=number,
        factor_id=factor_id,
        factor_name=name,
        category=category,
        protocol_role="core" if category in core_categories else "extension",
        usage_type=usage,
        economic_meaning=meaning,
        formula=formula,
        expected_direction=direction,
        data_source=source,
        implementation_status=status,
        unavailable_reason=reason,
    )


FACTOR_DEFINITIONS = [
    _factor(1, "tsmom_60", "时间序列动量", "trend_momentum", "predictor", "波动率调整后的中期趋势延续", "log_return_60/(vol_60*sqrt(60))", "positive", "实际主力合约收益"),
    _factor(2, "csmom_60", "横截面动量", "trend_momentum", "predictor", "同日商品过去收益相对强弱", "cross_sectional_rank(log_return_60)", "positive", "实际主力合约收益"),
    _factor(3, "trend_efficiency_60", "趋势效率", "trend_momentum", "predictor", "价格路径的单向和平滑程度", "abs(log_return_60)/sum(abs(daily_log_return),60)", "positive", "实际主力合约收益"),
    _factor(4, "multi_horizon_trend", "多周期趋势一致性", "trend_momentum", "predictor", "20/60/120日趋势方向一致程度", "mean(sign(return_20),sign(return_60),sign(return_120))", "positive", "实际主力合约收益"),
    _factor(5, "carry_annualized", "Carry因子", "term_structure", "predictor", "完整可交易曲线最近两个合约之间的年化期限结构补偿", "log(curve_near/curve_middle)*365/maturity_gap", "positive", "完整实际合约曲线、收盘价、持仓量、成交量与到期日"),
    _factor(6, "carry_change_20", "Carry变化", "term_structure", "predictor", "期限结构二十日边际变化", "carry_t-carry_t-20", "positive", "主力与次近月收盘价、到期日"),
    _factor(7, "basis_fresh", "基差", "basis_spot", "predictor", "新鲜现货相对主力期货的强弱", "(fresh_spot-main_close)/main_close", "positive", "现货价格、观察日、主力收盘价"),
    _factor(8, "basis_shock_20", "基差冲击", "basis_spot", "predictor", "基差日变化相对历史条件分布的异常", "zscore(delta_basis; prior_20)", "positive", "现货价格、观察日、主力收盘价"),
    _factor(9, "basis_momentum_20", "基差动量", "basis_spot", "predictor", "新鲜可用基差的二十日边际变化", "basis_t-basis_t-20", "positive", "延迟一交易日可用的现货与主力收盘价"),
    _factor(10, "curve_curvature", "期限结构曲率", "term_structure", "predictor", "曲线第二个合约相对第一和第三个合约线性插值的弯曲程度", "2*(log(middle)-interpolate(log(near),log(far)))", "mixed", "通过流动性和期限硬门的前三个实际合约"),
    _factor(11, "value_252", "价值因子", "value_reversion", "predictor", "价格相对一年滚动历史锚点的反向偏离", "-(log_price-rolling_mean_252)/rolling_std_252", "positive", "主力收盘价"),
    _factor(12, "short_reversal_5", "短期反转", "value_reversion", "predictor", "过去五日过度反应后的修复", "-log_return_5", "positive", "实际主力合约收益"),
    _factor(13, "inventory_scarcity", "库存稀缺因子", "inventory_fundamental", "predictor", "库存相对历史同期的稀缺程度", "-seasonal_z(stock_current)", "positive", "库存、观察日与数据年龄"),
    _factor(14, "inventory_change_5", "库存变化因子", "inventory_fundamental", "predictor", "五日库存下降所表示的边际趋紧", "-delta_5(stock_current)", "positive", "库存、观察日与数据年龄"),
    _factor(15, "inventory_surprise", "库存意外／库存新息", "information_news", "predictor", "库存相对仅用历史信息形成的条件预期的标准化意外", "-innovation_z(stock_current; prior_60)", "positive", "库存、观察日与数据年龄"),
    _factor(16, "warehouse_tightness", "仓单紧张因子", "inventory_fundamental", "predictor", "注册仓单相对历史同期的紧张程度", "-seasonal_z(warehouse_receipt)", "positive", "注册仓单及观察日"),
    _factor(17, "tightness_composite", "供需紧张度", "inventory_fundamental", "predictor", "仓单、库存、基差和Carry的等权看多方向组合", "mean(-inventory_z,-warehouse_z,basis_z,carry_z)", "positive", "库存、仓单、现货、期限结构"),
    _factor(18, "oi_growth_20", "持仓量增长", "trading_liquidity", "conditional", "未平仓合约数量的二十日增长", "log(OI_t)-log(OI_t-20)", "mixed", "主力持仓量"),
    _factor(19, "price_oi_interaction_20", "价格—持仓量交互", "trading_liquidity", "predictor", "价格方向和新增持仓方向的联合变化", "log_return_20*oi_growth_20", "mixed", "实际主力合约收益、持仓量"),
    _factor(20, "hedging_pressure", "套保压力", "trading_liquidity", "predictor", "商业交易者净持仓压力", "(commercial_long-commercial_short)/OI", "mixed", "可靠交易者分类持仓", "unavailable", "data_final_raw没有商业/非商业交易者分类持仓"),
    _factor(21, "volume_shock_20", "成交量冲击", "trading_liquidity", "conditional", "成交量相对历史正常水平的异常", "zscore(log_volume; prior_20)", "mixed", "主力成交量"),
    _factor(22, "amihud_20", "Amihud非流动性", "trading_liquidity", "risk", "单位成交额对应的绝对价格变动", "mean(abs(return)/amount,20)", "negative", "实际主力合约收益、成交额"),
    _factor(23, "turnover_oi", "成交量—持仓量换手率", "trading_liquidity", "conditional", "相对存量持仓的交易活跃程度", "volume/open_interest", "mixed", "主力成交量、持仓量"),
    _factor(24, "realized_volatility_20", "已实现波动率", "volatility_risk", "risk", "过去二十日收益波动风险", "std(log_return,20)*sqrt(252)", "mixed", "实际主力合约收益"),
    _factor(25, "low_volatility_20", "低波动因子", "volatility_risk", "risk", "横截面低波动品种得分更高", "-cross_sectional_rank(realized_volatility_20)", "positive", "实际主力合约收益"),
    _factor(26, "realized_skewness_60", "已实现偏度", "volatility_risk", "risk", "过去六十日收益分布非对称性", "skew(log_return,60)", "mixed", "实际主力合约收益"),
    _factor(27, "seasonality_3y", "季节性因子", "value_reversion", "predictor", "过去三年近似同期五日收益的平均", "mean(trailing_return_5 shifted 252/504/756)", "mixed", "实际主力合约历史收益"),
    _factor(28, "industry_chain_residual", "产业链残差", "industry_chain", "predictor", "相对明确上下游和替代品关系的状态偏离", "x-E[x|industry_neighbors]", "mixed", "产业链邻接关系与传导系数", "unavailable", "未提供经确认的产业链邻接表和生产传导系数"),
    _factor(29, "sector_relative_value_60", "相对价值残差", "industry_chain", "predictor", "品种六十日收益相对板块共同收益的偏离", "return_60-sector_mean_return_60", "negative", "实际主力合约收益、固定板块映射"),
    _factor(30, "market_state_conditional", "状态条件因子", "state_interaction", "conditional", "市场状态概率调节基础因子", "factor*p_market_state", "mixed", "实时市场状态概率", "unavailable", "当前协议未定义和冻结市场状态概率引擎"),
    _factor(31, "product_state_interaction", "单品种状态交互因子", "state_interaction", "conditional", "品种自身状态调节基础因子", "factor*product_state", "mixed", "实时单品种状态", "unavailable", "当前协议未定义和冻结单品种状态引擎"),
    _factor(32, "factor_momentum", "因子动量", "meta_factor", "conditional", "四个冻结基础因子组合最近二十日已实现收益的持续性", "rolling_sum_20(mean(lagged_factor_portfolio_return))", "mixed", "严格滞后一期的动量、Carry、反转和板块相对价值组合收益"),
    _factor(33, "fundamental_composite_news", "基本面综合新息代理", "information_news", "predictor", "库存、仓单和基差中至少两类历史模型意外的统一方向组合", "mean(inventory_surprise,warehouse_surprise,basis_surprise)", "positive", "延迟一交易日可用的库存、仓单和现货；不是市场一致预期"),
    _factor(34, "fundamental_underreaction", "基本面反应不足代理", "information_news", "predictor", "模型隐含基本面新息扣除首个可交易日价格反应后的未吸收部分", "fundamental_composite_news-cs_z(initial_price_response)", "positive", "基本面综合新息代理与主力收益；不是市场一致预期"),
    _factor(35, "information_consistency", "信息一致性", "information_news", "predictor", "仓单、库存、基差和Carry方向一致时放大的综合信号", "mean(z_k)*abs(mean(sign(z_k)))", "positive", "库存、仓单、基差、Carry"),
    _factor(36, "information_divergence", "信息背离", "information_news", "predictor", "供需基本面信号相对价格趋势的未反映差异", "tightness_z-tsmom_z", "positive", "供需紧张度与实际主力合约收益"),
    _factor(37, "cost_transmission_gap", "成本传导缺口", "industry_chain", "predictor", "上游成本冲击相对下游价格变化的未传导部分", "sum(a_ij*upstream_return)-downstream_return", "mixed", "上下游关系与生产配比", "unavailable", "未提供经确认的产业链邻接表和实际生产配比"),
    _factor(38, "processing_margin_anomaly", "加工利润异常", "industry_chain", "predictor", "加工利润相对历史季节正常水平的异常", "seasonal_z(output-sum(a*input)-other_cost)", "mixed", "产出投入价格、生产配比和其他成本", "unavailable", "缺少经确认的生产配比和其他成本序列"),
    _factor(39, "warehouse_surprise", "仓单新息", "information_news", "predictor", "注册仓单相对历史条件预期的标准化意外", "-innovation_z(warehouse_receipt; prior_60)", "positive", "注册仓单及观察日"),
    _factor(40, "transmission_residual", "产业链传导残差", "industry_chain", "predictor", "剥离上游滞后冲击和板块共同收益后的残差", "r_i-sum(beta*lagged_upstream_return)-beta_sector*r_sector", "mixed", "产业链邻接、滞后与板块收益", "unavailable", "未提供经确认的产业链邻接和传导滞后"),
    _factor(41, "momentum_quality_60", "动量质量因子", "trend_momentum", "predictor", "以趋势效率过滤中期动量路径噪声", "tsmom_60*trend_efficiency_60", "positive", "实际主力合约收益"),
    _factor(42, "vol_managed_momentum", "波动率管理动量", "trend_momentum", "predictor", "按短期风险缩放中期趋势方向", "sign(return_60)/volatility_20", "positive", "实际主力合约收益"),
    _factor(43, "idiosyncratic_volatility_60", "特质波动因子", "volatility_risk", "risk", "剔除商品市场及板块共同收益后的剩余波动", "std(residual(r_i~r_market+r_sector),60)", "mixed", "实际主力合约收益、固定板块映射"),
    _factor(44, "term_structure_slope", "期限结构斜率", "term_structure", "predictor", "用完整可交易曲线加权拟合的年化斜率", "-weighted_OLS_slope(log_future_price~maturity)", "positive", "至少三个通过流动性和期限硬门的实际合约"),
    _factor(45, "term_structure_momentum", "期限结构动量", "term_structure", "predictor", "整条期限结构斜率的二十日变化", "slope_t-slope_t-20", "positive", "完整曲线斜率历史"),
    _factor(46, "term_structure_acceleration", "期限结构加速度", "term_structure", "predictor", "整条期限结构二十日变化的再二十日变化", "delta20_slope_t-delta20_slope_t-20", "mixed", "完整曲线斜率历史"),
    _factor(47, "basis_surprise", "基差意外", "basis_spot", "predictor", "基差相对历史条件预期的标准化意外", "innovation_z(basis; prior_60)", "positive", "新鲜现货价格、观察日、主力收盘价"),
    _factor(48, "inventory_acceleration", "库存加速度", "inventory_fundamental", "predictor", "去库存速度的二阶变化", "-(delta_5_inventory_t-delta_5_inventory_t-5)", "positive", "库存、观察日与数据年龄"),
    _factor(49, "inventory_price_divergence", "库存—价格背离", "information_news", "predictor", "库存趋紧相对同期价格变化的未反映部分", "z(-delta_inventory)-z(return_20)", "positive", "库存与实际主力合约收益"),
    _factor(50, "supply_demand_shock", "供需冲击因子", "information_news", "predictor", "需求正向新息减去供给正向新息", "sum(demand_surprise)-sum(supply_surprise)", "positive", "产量、需求、进口等实际与预期数据", "unavailable", "data_final_raw没有供给与需求多源实际值及预期字段"),
    _factor(51, "crowding", "拥挤度因子", "trading_liquidity", "risk", "主要参与者持仓份额集中度", "HHI(member_position_share)", "negative", "会员或参与者分项持仓", "unavailable", "data_final_raw没有会员或参与者分项持仓"),
    _factor(52, "oi_surprise_60", "持仓意外因子", "trading_liquidity", "conditional", "持仓变化相对历史条件预期的异常", "zscore(delta_log_OI; prior_60)", "mixed", "主力持仓量"),
    _factor(53, "volume_price_divergence_20", "量价背离因子", "trading_liquidity", "predictor", "价格趋势与成交量变化不一致程度", "z(return_20)-z(delta_log_volume_20)", "mixed", "实际主力合约收益、成交量"),
    _factor(54, "momentum_carry_interaction", "因子交互因子", "factor_interaction", "predictor", "动量和Carry同时确认的非加性信息", "z(tsmom_60)*z(carry)", "positive", "动量与期限结构"),
    _factor(55, "nonlinear_tightness", "非线性基本面因子", "factor_interaction", "predictor", "极端供需紧张状态的平方增强", "sign(tightness_z)*abs(tightness_z)^2", "positive", "供需紧张度"),
    _factor(56, "commodity_network_factor", "商品网络因子", "industry_chain", "predictor", "按产业链和替代关系聚合邻居冲击", "sum(w_ij*z(signal_j))", "mixed", "经确认的商品网络邻接与权重", "unavailable", "未提供经确认的商品网络邻接表与权重"),
    _factor(57, "correlation_risk_20_120", "相关性风险因子", "volatility_risk", "risk", "品种与商品市场短期相关性相对长期基准的上升", "corr_20(r_i,r_market)-corr_120(r_i,r_market)", "negative", "实际主力合约收益、商品市场收益"),
]


def definitions_frame() -> pd.DataFrame:
    frame = pd.DataFrame([asdict(item) for item in FACTOR_DEFINITIONS])
    if len(frame) != 57 or frame["factor_no"].tolist() != list(range(1, 58)):
        raise AssertionError("V2.1 factor registry must contain exactly factors 1 through 57")
    if not frame["factor_id"].is_unique:
        raise AssertionError("factor_id must be unique")
    return frame
