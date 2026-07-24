from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from factor_modeling.common import _daily_correlations, return_statistics


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None or (isinstance(value, (float, np.floating)) and np.isnan(value)):
        return "—"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    return str(value)


def _markdown(frame: pd.DataFrame) -> str:
    return frame.to_markdown(index=False) if not frame.empty else "暂无记录。"


def write_data_audit(
    path: Path,
    dataset_version: str,
    panel: pd.DataFrame,
    curve: pd.DataFrame,
    availability: pd.DataFrame,
    pit: dict[str, Any],
) -> None:
    curve_days = curve[["trade_date", "product", "n_contracts"]].drop_duplicates()
    lines = [
        "# 扩展窗口版本数据审计报告",
        "",
        "> 本报告由项目真实数据自动生成，不补造缺失口径。",
        "",
        f"- 数据版本：`{dataset_version}`。",
        f"- 数据范围：{panel.trade_date.min().date()} 至 {panel.trade_date.max().date()}。",
        f"- 品种数：{panel['product'].nunique()}；品种日记录：{len(panel):,}。",
        f"- 可交易日收益覆盖率：{panel.tradable_return.notna().mean():.2%}。",
        f"- 期限结构品种日：{len(curve_days):,}；至少2个合格合约占比：{(curve_days.n_contracts >= 2).mean():.2%}。",
        "- 因子形成时间为T日收盘后，最早T+1开盘使用。",
        "- 每个外层测试折只使用测试年前的数据筛选因子和训练模型。",
        "",
        "## 数据可用性",
        "",
        _markdown(availability.assign(strict_point_in_time_usable=availability.strict_point_in_time_usable.map({True: "是", False: "否"}))),
        "",
        "## 仍未解决的边界",
        "",
        "1. 仓单、库存和现货缺少可核验的真实发布时间或字段口径，因此相关概念因子仍为F类。",
        "2. 手续费单位、合约乘数、涨跌停可成交性、盘口冲击和容量未核验，成本仅作1/3/5bp敏感性。",
        "3. 产业链映射、生产配比和商业交易者分类持仓不足。",
        "",
        "## 点时审计",
        "",
        "```json",
        __import__("json").dumps(pit, ensure_ascii=False, indent=2),
        "```",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_walk_forward_report(
    path: Path,
    fold_metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    strategy: pd.DataFrame,
    selection_history: pd.DataFrame,
    class_history: pd.DataFrame,
    config: dict[str, Any],
) -> None:
    daily_ic = _daily_correlations(predictions)
    annualization = int(config["model"]["annualization"])
    return_rows = [{"成本": "毛收益", **return_statistics(strategy.gross_return, annualization)}]
    for bps in config["model"]["one_way_cost_bps"]:
        label = str(int(float(bps)))
        return_rows.append({
            "成本": f"单边{float(bps):g}bp",
            **return_statistics(strategy[f"net_{label}bps_return"], annualization),
        })
    returns = pd.DataFrame(return_rows)
    folds = fold_metrics.copy()
    for column in folds.columns:
        if pd.api.types.is_float_dtype(folds[column]):
            folds[column] = folds[column].map(_fmt)
    for column in returns.columns[1:]:
        returns[column] = returns[column].map(_fmt)
    classes = class_history.pivot(index="test_year", columns="classification", values="count").fillna(0).astype(int).reset_index()
    stable = selection_history.sort_values(["selected_fold_share", "selected_folds", "feature_name"], ascending=[False, False, True]).head(30).copy()
    stable["selected_fold_share"] = stable["selected_fold_share"].map(_fmt)
    overall = pd.DataFrame([
        {"指标": "外层测试年份", "结果": "2019—2025"},
        {"指标": "预测记录", "结果": len(predictions)},
        {"指标": "测试交易日", "结果": predictions.trade_date.nunique()},
        {"指标": "平均Pearson IC", "结果": _fmt(daily_ic.pearson_ic.mean())},
        {"指标": "平均RankIC", "结果": _fmt(daily_ic.rank_ic.mean())},
        {"指标": "RankIC胜率", "结果": _fmt(daily_ic.rank_ic.gt(0).mean())},
    ])
    lines = [
        "# 2015固定起点扩展窗口：因子筛选至模型回测报告",
        "",
        "> 每个测试折均重新执行训练期因子筛选、相关性去重、增量检验、特征加工和Ridge训练。现有旧版本结果未被覆盖。",
        "",
        "## 窗口规则",
        "",
        "- 2015—2018训练，2019测试。",
        "- 2015—2019训练，2020测试；之后训练终点逐年扩展，直至2015—2024训练、2025测试。",
        "- 每折因子筛选主期限为5日；最终模型预测下一交易日开盘至收盘收益的横截面排名。",
        "- 每折只把该训练窗口内筛出的A/B/C/D字段全部输入同一个Ridge模型，不使用分层模型。",
        "- 2025此前已在旧研究中被查看，本报告中的2025只能视为历史补充折；整个2019—2025结果属于研究回测，不是全新封存验证。",
        "",
        "## 总体预测结果",
        "",
        _markdown(overall),
        "",
        "## 总体策略成本敏感性",
        "",
        _markdown(returns),
        "",
        "## 各测试折结果",
        "",
        _markdown(folds),
        "",
        "## 各折概念因子分类数量",
        "",
        _markdown(classes),
        "",
        "## 选择稳定度最高的输入字段",
        "",
        _markdown(stable),
        "",
        "## 结论边界",
        "",
        "1. 因子筛选发生在每个外层训练窗口内，避免将未来年份的因子选择倒灌到早期测试折。",
        "2. 训练期因子显著性仍属于模型选择环节；只有随后的单年外层结果属于该折样本外评价。",
        "3. 策略每天开仓和平仓，成本敏感度很高；未核验的真实手续费和冲击不能被当前bp情景替代。",
        "4. 如果根据本报告继续修改因子、标签或持仓规则，2019—2025都应视为开发数据，未来确认必须使用更新年份。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
