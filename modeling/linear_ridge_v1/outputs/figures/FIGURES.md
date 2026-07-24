# 线性回归（Ridge）结果图说明

本目录由 `modeling/linear_ridge_v1/make_visuals.py` 根据 `outputs/` 回测结果自动生成，对应模型版本 `linear_ridge_v1`（样本外 2019–2024；2025 未纳入）。

## 图片一览

| 文件 | 内容 | 怎么读 |
|---|---|---|
| `01_equity_curves.png` | 毛收益与 0/3/5/10bp 净收益的累计净值曲线 | 起点为 1；粗线为协议主成本 **5bp**。若 5bp 线低于 1，说明成本后累计亏损。 |
| `02_drawdown_gross_vs_net5.png` | 毛收益 vs 净 5bp 回撤 | 阴影为回撤深度；对比成本如何放大回撤。 |
| `03_cost_sensitivity.png` | 各成本档的年化收益与 Sharpe | 左图年化、右图 Sharpe；标注「protocol base」对应 5bp。 |
| `04_fold_rankic_and_sharpe.png` | 分年（外层折）RankIC 与 Sharpe | 左：预测力是否稳定；右：毛/净 5bp 是否同号、哪几年拖累。 |
| `05_daily_rank_ic.png` | 日度 RankIC + 60 日滚动均值 | 灰线波动大属正常；蓝线看阶段性预测力；橙虚线为全样本均值。 |
| `06_selected_alpha_by_fold.png` | 各外层折选中的 Ridge α | 对数轴；α 变小表示内层更偏好较弱正则。 |
| `07_top20_coefficients.png` | 跨折平均系数绝对值 Top20 | 颜色区分 `usage_type`；系数是条件关联而非因果，`__missing` 出现不代表经济因子本身。 |
| `08_alpha_search_heatmap.png` | 内层验证 RankIC × α 热力图 | 每行一个外层折；格子越“暖/高”表示该 α 内层 RankIC 更好（选参依据）。 |

## 数据来源

- 净值 / 成本：`outputs/strategy_daily_returns.csv`
- 分折指标：`outputs/fold_metrics.csv`
- 日度 IC：`outputs/daily_prediction_ic.csv`
- 系数：`outputs/coefficient_summary.csv`
- α 搜索：`outputs/alpha_search.csv`
- 汇总：`outputs/run_summary.json`

## 重新生成

```bash
set PYTHONPATH=src
py modeling/linear_ridge_v1/make_visuals.py
```

## 阅读提示

1. 协议主口径是单边 **5bp**；提交模板常另报 3bp，两套数字不要混用。
2. 净值序列按 **每 5 日调仓日** 复利，不是日历日逐笔成交净值。
3. 图只服务开发期 OOS（2019–2024），不能当作 2025 封存结论。

## 本次生成文件

- `01_equity_curves.png`
- `02_drawdown_gross_vs_net5.png`
- `03_cost_sensitivity.png`
- `04_fold_rankic_and_sharpe.png`
- `05_daily_rank_ic.png`
- `06_selected_alpha_by_fold.png`
- `07_top20_coefficients.png`
- `08_alpha_search_heatmap.png`
