# Linear Ridge V2（筛弱 / 去重后再 Ridge）

与 [`equal_weight_multi_v2`](../equal_weight_multi_v2/) 同思路：

1. 先验宇宙：仅 `usage_type=predictor`
2. 每折训练窗：RankIC 剔弱 + 贪心相关去重
3. 对入选集做 Ridge（内层验证年选 `alpha`）
4. 同协议样本外多空评价

相对 [`linear_ridge_v1`](../linear_ridge_v1/)（92 字段全体入模）：不做 AIC 逐步回归；筛选规则与等权 v2 对齐，**不偷看 OOS**。

## 运行

```bash
set PYTHONPATH=src
py modeling/linear_ridge_v2/run.py
```

## 主要输出

- `线性回归模型_团队规范提交报告.md`（团队模板，含差异声明、OOS 与 v1 对照）
- `linear_ridge_report.md`（含与 v1 对照）
- `outputs/fold_selected_factors.csv`
- `outputs/fold_metrics.csv` / `run_summary.json` / 系数与 alpha 表
