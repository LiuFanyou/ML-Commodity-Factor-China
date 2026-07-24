# 中国商品期货 Alpha 因子库 V2.1（分类版）

本文档保留经典商品期货因子，并加入面向中国商品市场的基本面新息、信息反应、产业链传导、相对价值、曲线动态、交易拥挤与非线性交互因子。

所有滚动统计量、预期值和标准化参数均应仅使用时点 $t$ 之前真实可获得的信息。基本面因子应按实际发布时间对齐，不得按统计期提前使用。

## 1. 时间序列动量（Time-Series Momentum）

**数学定义**

$$
TSMOM_{i,t}^{(L)}
=
\frac{\ln P_{i,t}-\ln P_{i,t-L}}
{\widehat{\sigma}_{i,t}^{(L)}}
$$

**说明**：衡量品种 $i$ 过去 $L$ 期的波动率调整趋势方向和强度。

## 2. 横截面动量（Cross-Sectional Momentum）

**数学定义**

$$
CSMOM_{i,t}^{(L)}
=
\operatorname{Rank}_i
\left(
\ln P_{i,t}-\ln P_{i,t-L}
\right)
$$

**说明**：衡量同一时点各品种过去收益的相对强弱。

## 3. 趋势效率（Trend Efficiency）

**数学定义**

$$
TE_{i,t}^{(L)}
=
\frac{
\left|P_{i,t}-P_{i,t-L}\right|
}{
\sum_{k=1}^{L}
\left|P_{i,t-k+1}-P_{i,t-k}\right|
}
$$

**说明**：衡量价格路径的单向程度；越接近 $1$，趋势越平滑。

## 4. 多周期趋势一致性（Multi-Horizon Trend Consistency）

**数学定义**

$$
MHT_{i,t}
=
\frac{1}{K}
\sum_{k=1}^{K}
\operatorname{sign}
\left(
\ln P_{i,t}-\ln P_{i,t-L_k}
\right)
$$

**说明**：衡量多个回看周期的趋势方向是否一致。

## 5. Carry 因子

**数学定义**

$$
Carry_{i,t}
=
\frac{
\ln F_{i,t}^{near}-\ln F_{i,t}^{far}
}{
\tau_{far}-\tau_{near}
}
$$

**说明**：衡量年化期限结构斜率；按此定义，近月强于远月时 Carry 为正。

## 6. Carry 变化

**数学定义**

$$
\Delta Carry_{i,t}^{(L)}
=
Carry_{i,t}-Carry_{i,t-L}
$$

**说明**：衡量期限结构在过去 $L$ 期的边际变化。

## 7. 基差（Basis）

**数学定义**

$$
Basis_{i,t}
=
\frac{S_{i,t}-F_{i,t}^{near}}
{F_{i,t}^{near}}
$$

**说明**：衡量现货相对近月期货的强弱；现货价格须匹配品级、地区和发布时间。

## 8. 基差冲击（Basis Shock）

**数学定义**

$$
BasisShock_{i,t}^{(L)}
=
\frac{
\Delta Basis_{i,t}
-\widehat{\mu}_{i,t^-}^{(L)}(\Delta Basis)
}{
\widehat{\sigma}_{i,t^-}^{(L)}(\Delta Basis)
}
$$

**说明**：衡量基差变化相对于其历史正常水平的异常程度。

## 9. 基差动量（Basis Momentum）

**数学定义**

$$
BM_{i,t}^{(L)}
=
\left(
\ln F_{i,t}^{near}-\ln F_{i,t-L}^{near}
\right)
-
\left(
\ln F_{i,t}^{far}-\ln F_{i,t-L}^{far}
\right)
$$

**说明**：衡量近月合约相对远月合约的近期价格动量。

## 10. 期限结构曲率（Curve Curvature）

**数学定义**

$$
Curvature_{i,t}
=
2\ln F_{i,t}^{middle}
-\ln F_{i,t}^{near}
-\ln F_{i,t}^{far}
$$

**说明**：衡量期货曲线中部相对于近端和远端的弯曲程度。

## 11. 价值因子（Value）

**数学定义**

$$
Value_{i,t}^{(L)}
=
-\frac{
\ln P_{i,t}
-\operatorname{MA}_{L}(\ln P_{i,t})
}{
\operatorname{Std}_{L}(\ln P_{i,t})
}
$$

**说明**：衡量当前价格相对于滚动长期锚点的标准化偏离。

## 12. 短期反转（Short-Term Reversal）

**数学定义**

$$
Reversal_{i,t}^{(L)}
=
-\left(
\ln P_{i,t}-\ln P_{i,t-L}
\right)
$$

**说明**：检验短期价格过度反应后是否存在反向修复。

## 13. 库存稀缺因子（Inventory Scarcity）

**数学定义**

$$
Scarcity_{i,t}
=
-\frac{
Inventory_{i,t}
-\widehat{\mu}_{i,t}^{seasonal}
}{
\widehat{\sigma}_{i,t}^{seasonal}
}
$$

**说明**：衡量库存相对于品种自身季节正常水平的稀缺程度。

## 14. 库存变化因子（Inventory Change）

**数学定义**

$$
InventoryChange_{i,t}^{(L)}
=
-\left(
Inventory_{i,t}-Inventory_{i,t-L}
\right)
$$

**说明**：正值表示库存下降，即供需边际趋紧。

## 15. 库存意外／库存新息（Inventory Surprise）

**数学定义**

$$
InventorySurprise_{i,\tau}
=
-\frac{
Inventory_{i,\tau}^{actual}
-\widehat{\mathbb E}_{\tau^-}
\left[Inventory_{i,\tau}\right]
}{
\widehat{\sigma}_{i,\tau^-}^{surprise}
}
$$

**说明**：衡量实际公布库存相对公布前预期的标准化新息；$\tau$ 为真实发布时间。

## 16. 仓单紧张因子（Warehouse Receipt Tightness）

**数学定义**

$$
WarehouseTightness_{i,t}
=
-\frac{
Warehouse_{i,t}
-\widehat{\mu}_{i,t}^{seasonal}
}{
\widehat{\sigma}_{i,t}^{seasonal}
}
$$

**说明**：衡量注册仓单相对于季节正常水平的紧张程度。

## 17. 供需紧张度（Tightness）

**数学定义**

$$
Tightness_{i,t}
=
w_1(-InventoryZ_{i,t})
+w_2(-WarehouseZ_{i,t})
+w_3BasisZ_{i,t}
+w_4CarryZ_{i,t}
$$

**说明**：综合库存、仓单、基差和期限结构刻画供需紧张程度；权重须预先确定或仅在训练期估计。

## 18. 持仓量增长（Open-Interest Growth）

**数学定义**

$$
OIGrowth_{i,t}^{(L)}
=
\ln OI_{i,t}-\ln OI_{i,t-L}
$$

**说明**：衡量市场未平仓合约数量的增长速度。

## 19. 价格—持仓量交互（Price–OI Interaction）

**数学定义**

$$
PriceOI_{i,t}^{(L)}
=
\left(
\ln P_{i,t}-\ln P_{i,t-L}
\right)
\left(
\ln OI_{i,t}-\ln OI_{i,t-L}
\right)
$$

**说明**：联合刻画价格方向和新增持仓方向。

## 20. 套保压力（Hedging Pressure）

**数学定义**

$$
HP_{i,t}
=
\frac{
CommercialLong_{i,t}-CommercialShort_{i,t}
}{
OpenInterest_{i,t}
}
$$

**说明**：衡量商业交易者的净持仓压力；仅适用于存在可靠交易者分类的数据。

## 21. 成交量冲击（Volume Shock）

**数学定义**

$$
VolumeShock_{i,t}^{(L)}
=
\frac{
\ln Volume_{i,t}
-\operatorname{Mean}_{L}(\ln Volume_{i,t})
}{
\operatorname{Std}_{L}(\ln Volume_{i,t})
}
$$

**说明**：衡量当期成交量相对于近期正常水平的异常程度。

## 22. Amihud 非流动性因子

**数学定义**

$$
Illiquidity_{i,t}^{(L)}
=
\frac{1}{L}
\sum_{k=0}^{L-1}
\frac{|r_{i,t-k}|}{Amount_{i,t-k}}
$$

**说明**：衡量单位成交额对应的绝对价格变动，数值越高表示流动性越差。

## 23. 成交量—持仓量换手率

**数学定义**

$$
Turnover_{i,t}
=
\frac{Volume_{i,t}}{OpenInterest_{i,t}}
$$

**说明**：衡量相对于存量持仓的交易活跃程度。

## 24. 已实现波动率（Realized Volatility）

**数学定义**

$$
Volatility_{i,t}^{(L)}
=
\sqrt{
\sum_{k=1}^{L}
r_{i,t-k+1}^{2}
}
$$

**说明**：衡量过去 $L$ 期收益的累计波动风险。

## 25. 低波动因子（Low Volatility）

**数学定义**

$$
LowVol_{i,t}
=
-\operatorname{Rank}_i
\left(
Volatility_{i,t}^{(L)}
\right)
$$

**说明**：给予横截面低波动品种更高因子值。

## 26. 已实现偏度（Realized Skewness）

**数学定义**

$$
Skewness_{i,t}^{(L)}
=
\frac{
\frac{1}{L}
\sum_{k=1}^{L}
\left(r_{i,t-k+1}-\bar r_{i,t}^{(L)}\right)^3
}{
\left[
\frac{1}{L}
\sum_{k=1}^{L}
\left(r_{i,t-k+1}-\bar r_{i,t}^{(L)}\right)^2
\right]^{3/2}
}
$$

**说明**：衡量历史收益分布的不对称程度。

## 27. 季节性因子（Seasonality）

**数学定义**

$$
Seasonality_{i,t}^{(K,h)}
=
\frac{1}{K}
\sum_{y=1}^{K}
R_{i,t-y\,\mathrm{year}:\,t-y\,\mathrm{year}+h}
$$

**说明**：使用历史同期收益估计品种在当前季节位置的方向性规律。

## 28. 产业链残差（Industry-Chain Residual）

**数学定义**

$$
ChainResidual_{i,t}
=
x_{i,t}
-\widehat{\mathbb E}_{t^-}
\left[
x_{i,t}
\mid
x_{\mathcal N(i),t}
\right]
$$

**说明**：衡量品种状态相对于其上下游、替代品或共同成本品种所隐含合理状态的偏离。

## 29. 相对价值残差（Relative-Value Residual）

**数学定义**

$$
RelativeValue_{i,t}
=
x_{i,t}
-\widehat{\alpha}_{g(i),t}
-\widehat{\beta}_{i,t}x_{g(i),t}
$$

**说明**：衡量品种相对于所属板块、配对品种或基准价格关系的残余偏离。

## 30. 状态条件因子（Market-State Conditional Factor）

**数学定义**

$$
ConditionalFactor_{i,t}^{(s)}
=
Factor_{i,t}\,p_{t,s}
$$

**说明**：使用市场状态 $s$ 的实时概率调节基础因子强度。

## 31. 单品种状态交互因子

**数学定义**

$$
Interaction_{i,t}
=
Factor_{i,t}\cdot ProductState_{i,t}
$$

**说明**：检验品种自身供需、流动性或基本面状态是否改变基础因子的有效性。

## 32. 因子动量（Factor Momentum）

**数学定义**

$$
FactorMomentum_{j,t}^{(L)}
=
\sum_{k=1}^{L}
r_{j,t-k+1}^{factor}
$$

**说明**：衡量因子 $j$ 自身近期收益的持续性，用于研究因子择时。

## 33. 基本面综合新息（Fundamental Composite News）

**数学定义**

$$
FundamentalNews_{i,\tau}
=
\sum_{k=1}^{K}
w_k s_k
\frac{
X_{i,\tau,k}^{actual}
-\widehat{\mathbb E}_{\tau^-}
\left[X_{i,\tau,k}\right]
}{
\widehat{\sigma}_{i,\tau^-,k}^{surprise}
}
$$

**说明**：综合库存、仓单、开工率、产量、进口等公布新息；$s_k$ 用于统一各变量的看多方向。

## 34. 基本面反应不足（Fundamental Underreaction）

**数学定义**

$$
Underreaction_{i,\tau}^{(\delta)}
=
Z\left(FundamentalNews_{i,\tau}\right)
-\lambda
Z\left(
R_{i,\tau:\tau+\delta}^{initial}
\right)
$$

**说明**：衡量基本面新息中尚未被初始价格反应充分吸收的部分；$\delta$ 为预先设定的初始反应窗口。

## 35. 信息一致性（Information Consistency）

**数学定义**

$$
Consistency_{i,t}
=
\left(
\frac{1}{K}
\sum_{k=1}^{K}z_{i,t}^{(k)}
\right)
\left|
\frac{1}{K}
\sum_{k=1}^{K}
\operatorname{sign}
\left(z_{i,t}^{(k)}\right)
\right|
$$

**说明**：综合多个已统一看多方向的信息源；方向一致时放大综合信号，分歧时收缩。

## 36. 信息背离（Information Divergence）

**数学定义**

$$
Divergence_{i,t}
=
Z\left(FundamentalSignal_{i,t}\right)
-Z\left(PriceSignal_{i,t}\right)
$$

**说明**：衡量基本面信号与价格已反映信号之间的差异；正值表示基本面相对更偏多。

## 37. 成本传导缺口（Cost Transmission Gap）

**数学定义**

$$
TransmissionGap_{i,t}^{(L)}
=
\sum_{j\in\mathcal U(i)}
a_{ij}
\left(
\ln P_{j,t}-\ln P_{j,t-L}
\right)
-
\left(
\ln P_{i,t}-\ln P_{i,t-L}
\right)
$$

**说明**：衡量上游成本变化相对于下游价格实际变化尚未完成的传导部分；$a_{ij}$ 为实际生产配比或预先估计的传导系数。

## 38. 加工利润异常（Processing-Margin Anomaly）

**数学定义**

$$
MarginAnomaly_{i,t}
=
\frac{
Margin_{i,t}
-\widehat{\mu}_{i,t^-}^{(L,seasonal)}(Margin)
}{
\widehat{\sigma}_{i,t^-}^{(L,seasonal)}(Margin)
}
$$

其中：

$$
Margin_{i,t}
=
P_{i,t}^{output}
-\sum_{j=1}^{m}a_{ij}P_{j,t}^{input}
-OtherCost_{i,t}
$$

**说明**：衡量加工利润相对于历史及季节正常水平的异常程度。

## 39. 仓单新息（Warehouse-Receipt Surprise）

**数学定义**

$$
WarehouseSurprise_{i,\tau}
=
-\frac{
Warehouse_{i,\tau}^{actual}
-\widehat{\mathbb E}_{\tau^-}
\left[Warehouse_{i,\tau}\right]
}{
\widehat{\sigma}_{i,\tau^-}^{surprise}
}
$$

**说明**：衡量实际公布仓单相对于公布前预期的标准化新息。

## 40. 产业链传导残差（Transmission Residual）

**数学定义**

$$
TransmissionResidual_{i,t}
=
r_{i,t}
-\sum_{j\in\mathcal U(i)}
\widehat{\beta}_{ij,t^-}
r_{j,t-\ell_{ij}}
-\widehat{\beta}_{g(i),t^-}r_{g(i),t}
$$

**说明**：剥离上游滞后冲击和板块共同收益后，保留品种自身未被解释的价格变化。

## 41. 动量质量因子（Momentum Quality）

**数学定义**

$$
MomentumQuality_{i,t}^{(L)}
=
TSMOM_{i,t}^{(L)}
\cdot
TE_{i,t}^{(L)}
$$

**说明**：用趋势效率过滤路径噪声，使平滑、持续的趋势获得更高因子值。

## 42. 波动率管理动量（Volatility-Managed Momentum）

**数学定义**

$$
VMM_{i,t}^{(L_m,L_v)}
=
\operatorname{sign}
\left(
\ln P_{i,t}-\ln P_{i,t-L_m}
\right)
\frac{\sigma_{target}}
{\widehat{\sigma}_{i,t}^{(L_v)}}
$$

**说明**：根据预测波动率缩放趋势暴露，使不同品种和时期的风险预算更可比。

## 43. 特质波动因子（Idiosyncratic Volatility）

**数学定义**

先估计：

$$
r_{i,t}
=
\alpha_i
+\boldsymbol{\beta}_i^{\top}\boldsymbol{F}_t
+\varepsilon_{i,t}
$$

再定义：

$$
IVol_{i,t}^{(L)}
=
\operatorname{Std}_{L}
\left(
\widehat{\varepsilon}_{i,t}
\right)
$$

**说明**：衡量剔除商品市场、板块及其他预设共同因子后，品种自身的剩余波动风险。

## 44. 期限结构斜率（Term-Structure Slope）

**数学定义**

对同一品种多个到期合约拟合：

$$
\ln F_{i,t}^{(m)}
=
a_{i,t}
+b_{i,t}\tau_m
+\epsilon_{i,t}^{(m)}
$$

定义：

$$
TSSlope_{i,t}
=
-\widehat{b}_{i,t}
$$

**说明**：利用整条合约曲线估计年化斜率；按此符号约定，近端强于远端时因子为正。

## 45. 期限结构动量（Term-Structure Momentum）

**数学定义**

$$
TSMomentum_{i,t}^{(L)}
=
TSSlope_{i,t}
-TSSlope_{i,t-L}
$$

**说明**：衡量整条期限结构斜率的近期变化方向和幅度。

## 46. 期限结构加速度（Term-Structure Acceleration）

**数学定义**

$$
TSAcceleration_{i,t}^{(L)}
=
\left(
TSSlope_{i,t}-TSSlope_{i,t-L}
\right)
-
\left(
TSSlope_{i,t-L}-TSSlope_{i,t-2L}
\right)
$$

**说明**：衡量期限结构变化速度是否继续加快或开始减弱。

## 47. 基差意外（Basis Surprise）

**数学定义**

$$
BasisSurprise_{i,\tau}
=
\frac{
Basis_{i,\tau}^{actual}
-\widehat{\mathbb E}_{\tau^-}
\left[Basis_{i,\tau}\right]
}{
\widehat{\sigma}_{i,\tau^-}^{surprise}
}
$$

**说明**：衡量最新可得基差相对于公布前或更新前预期的标准化新息。

## 48. 库存加速度（Inventory Acceleration）

**数学定义**

$$
InventoryAcceleration_{i,t}^{(L)}
=
-\left[
\left(Inventory_{i,t}-Inventory_{i,t-L}\right)
-
\left(Inventory_{i,t-L}-Inventory_{i,t-2L}\right)
\right]
$$

**说明**：正值表示去库存速度正在加快，负值表示累库存速度正在加快。

## 49. 库存—价格背离（Inventory–Price Divergence）

**数学定义**

$$
IPD_{i,t}^{(L)}
=
Z\left(
-\Delta_L Inventory_{i,t}
\right)
-\lambda
Z\left(
R_{i,t-L:t}
\right)
$$

**说明**：衡量库存紧张信号中尚未被同期价格变化充分反映的部分；正值表示基本面相对价格更偏多。

## 50. 供需冲击因子（Supply–Demand Shock）

**数学定义**

$$
SupplyDemandShock_{i,\tau}
=
\sum_{k=1}^{K_D}
w_k^{D}Z\left(DemandSurprise_{i,\tau,k}\right)
-
\sum_{j=1}^{K_S}
w_j^{S}Z\left(SupplySurprise_{i,\tau,j}\right)
$$

**说明**：综合需求端正向新息与供给端负向新息；数值越高表示供需边际越偏紧。

## 51. 拥挤度因子（Crowding）

**数学定义**

$$
Crowding_{i,t}
=
\sum_{m=1}^{M}
\left(
\frac{|Position_{i,t}^{(m)}|}
{\sum_{n=1}^{M}|Position_{i,t}^{(n)}|}
\right)^2
$$

**说明**：使用主要参与者持仓份额的赫芬达尔指数衡量持仓集中度；数值越高表示交易越拥挤。

## 52. 持仓意外因子（Open-Interest Surprise）

**数学定义**

$$
OISurprise_{i,t}^{(L)}
=
\frac{
\Delta\ln OI_{i,t}
-\widehat{\mathbb E}_{t^-}
\left[\Delta\ln OI_{i,t}\right]
}{
\widehat{\sigma}_{i,t^-}^{(L)}
\left(\Delta\ln OI\right)
}
$$

**说明**：衡量持仓量变化相对于历史条件预期的异常程度。

## 53. 量价背离因子（Volume–Price Divergence）

**数学定义**

$$
VPD_{i,t}^{(L)}
=
Z\left(R_{i,t-L:t}\right)
-
Z\left(
\Delta_L\ln Volume_{i,t}
\right)
$$

**说明**：衡量价格趋势与成交量变化之间的不一致程度；方向解释须结合研究假设预先登记。

## 54. 因子交互因子（Factor Interaction）

**数学定义**

$$
FactorInteraction_{i,t}^{(a,b)}
=
Z\left(Factor_{i,t}^{(a)}\right)
\cdot
Z\left(Factor_{i,t}^{(b)}\right)
$$

**说明**：刻画两个基础因子同时处于特定方向或强度时的非加性信息。

## 55. 非线性基本面因子（Nonlinear Fundamental Factor）

**数学定义**

$$
NonlinearFundamental_{i,t}^{(\gamma)}
=
\operatorname{sign}\left(z_{i,t}\right)
\left|z_{i,t}\right|^{\gamma},
\qquad \gamma>1
$$

其中：

$$
z_{i,t}=Z\left(FundamentalSignal_{i,t}\right)
$$

**说明**：允许极端库存、基差或供需状态具有非线性影响；$\gamma$ 必须预先设定或仅在训练期估计。

## 56. 商品网络因子（Commodity Network Factor）

**数学定义**

$$
NetworkFactor_{i,t}
=
\sum_{j\in\mathcal N(i)}
w_{ij}
Z\left(Signal_{j,t}\right)
$$

**说明**：按产业链、替代关系、共同成本或统计网络权重，聚合关联商品的冲击信号。

## 57. 相关性风险因子（Correlation Risk Factor）

**数学定义**

$$
CorrelationRisk_{i,t}
=
\operatorname{Corr}_{L_s}
\left(r_{i},r_{market}\right)_t
-
\operatorname{Corr}_{L_l}
\left(r_{i},r_{market}\right)_t,
\qquad L_s<L_l
$$

**说明**：衡量品种与商品市场共同波动的短期相关性相对于长期基准的上升程度。

---

## 符号说明

- $i$：期货品种；$t$：研究时点；$\tau$：基本面数据真实发布时间。
- $L$：回看窗口；$h$：预测或持有期限；$K$：信号、周期或历史年份数量。
- $P$：价格；$F^{near}$、$F^{middle}$、$F^{far}$：近月、中月和远月期货价格；$S$：现货价格。
- $OI$：持仓量；$Amount$：成交额；$Warehouse$：注册仓单。
- $Z(\cdot)$：仅使用历史可得数据计算的标准化值。
- $p_{t,s}$：市场状态 $s$ 在时点 $t$ 的实时概率。
- $\mathcal N(i)$：品种 $i$ 的产业链或相对价值邻居；$\mathcal U(i)$：品种 $i$ 的上游原料集合。
- $\boldsymbol F_t$：预先确定的商品市场、板块或风格共同因子；$r_{market}$：商品市场基准收益。
- $\sigma_{target}$：目标波动率；$L_m$、$L_v$：动量与波动率回看窗口；$L_s$、$L_l$：短、长期相关性窗口。
- $Position^{(m)}$：参与者 $m$ 的持仓；$w$、$a$、$\beta$：预先确定或仅用训练期数据估计的权重及系数。
- 所有期货曲线因子应统一到可比剩余期限；所有真实收益应使用当时可交易合约并计入换月影响。

---

# 因子库分类总结

以下按照因子的主要研究逻辑分类，而不是按照编号顺序分类。部分因子同时具有多种经济含义，因此会在相关类别中重复出现；这类重复表示交叉属性，不表示重复实现。

## 一、趋势与动量因子

核心研究逻辑：价格冲击是否存在延续性，以及趋势路径是否足够稳定。

- 时间序列动量（#1）
- 横截面动量（#2）
- 趋势效率（#3）
- 多周期趋势一致性（#4）
- 动量质量因子（#41）
- 波动率管理动量（#42）

## 二、期限结构与 Carry 因子

核心研究逻辑：期货曲线所反映的持有收益、供需紧张程度及其动态变化是否具有预测信息。

- Carry 因子（#5）
- Carry 变化（#6）
- 基差动量（#9）
- 期限结构曲率（#10）
- 期限结构斜率（#44）
- 期限结构动量（#45）
- 期限结构加速度（#46）

## 三、价值与均值回复因子

核心研究逻辑：价格或收益是否偏离长期、季节性或经济均衡锚点，并存在修复倾向。

- 价值因子（#11）
- 短期反转（#12）
- 季节性因子（#27）
- 相对价值残差（#29）
- 加工利润异常（#38）

## 四、基差与现货市场因子

核心研究逻辑：现货与期货之间的价格关系是否反映即时供需、便利收益或尚未消化的信息。

- 基差（#7）
- 基差冲击（#8）
- 基差动量（#9）
- 供需紧张度中的基差分量（#17）
- 基差意外（#47）

## 五、库存与供需基本面因子

核心研究逻辑：库存、仓单及供需变量的水平、变化速度和综合紧张程度是否预示未来收益。

- 库存稀缺因子（#13）
- 库存变化因子（#14）
- 库存意外／库存新息（#15）
- 仓单紧张因子（#16）
- 供需紧张度（#17）
- 仓单新息（#39）
- 库存加速度（#48）
- 库存—价格背离（#49）
- 供需冲击因子（#50）

## 六、信息冲击与市场反应因子

核心研究逻辑：新信息是否超出市场预期，以及价格是否对信息反应不足、过度或出现内部矛盾。

- 库存意外／库存新息（#15）
- 基本面综合新息（#33）
- 基本面反应不足（#34）
- 信息一致性（#35）
- 信息背离（#36）
- 仓单新息（#39）
- 基差意外（#47）
- 库存—价格背离（#49）
- 供需冲击因子（#50）
- 量价背离因子（#53）

## 七、产业链与相对价值因子

核心研究逻辑：上下游、替代品、加工利润和板块共同定价关系是否出现暂时偏离或传导不充分。

- 产业链残差（#28）
- 相对价值残差（#29）
- 成本传导缺口（#37）
- 加工利润异常（#38）
- 产业链传导残差（#40）
- 商品网络因子（#56）

## 八、交易行为与资金流因子

核心研究逻辑：成交、持仓、参与者集中度和流动性变化是否揭示资金进入、退出、拥挤或交易压力。

- 持仓量增长（#18）
- 价格—持仓量交互（#19）
- 套保压力（#20）
- 成交量冲击（#21）
- Amihud 非流动性因子（#22）
- 成交量—持仓量换手率（#23）
- 拥挤度因子（#51）
- 持仓意外因子（#52）
- 量价背离因子（#53）

## 九、波动率与风险因子

核心研究逻辑：总波动、特质波动、偏度及相关性变化是否代表风险补偿、尾部暴露或仓位缩放依据。

- 已实现波动率（#24）
- 低波动因子（#25）
- 已实现偏度（#26）
- 波动率管理动量（#42）
- 特质波动因子（#43）
- 相关性风险因子（#57）

## 十、非线性交互与机器学习增强因子

核心研究逻辑：基础因子之间、因子与状态之间是否存在非加性关系，以及网络或元因子信息能否提高组合表达能力。

- 状态条件因子（#30）
- 单品种状态交互因子（#31）
- 因子动量（#32）
- 信息一致性（#35）
- 因子交互因子（#54）
- 非线性基本面因子（#55）
- 商品网络因子（#56）

> 分类说明：本章用于组织研究假设和因子注册。进入实证阶段后，还应按数据来源、更新频率、预测任务和与已有因子的增量信息进一步建立标签。
