"""Ridge L2-regularized linear regression on the training-handoff matrix interface.

Aligned with ``training_handoff/training_contract.json`` and the unified protocol:

- Features / registry / labels / folds are loaded only via ``factor_modeling.handoff``.
- Primary label: ``future_return_5d`` (open→open on mapped real contract).
- Expanding windows from ``fold_definitions.csv`` (2019–2024 OOS).
- Train purge: ``label_end_date_5d < valid_start``.
- Execution: signal at T close, trade T+1 open, 5-day hold / 5-day rebalance.
- Cost: base 5bp one-way; sensitivity 0 / 3 / 5 / 10 bp.
- v1: all registered fields.
- v2 optional ``selection_mode=train_ic_corr``: predictor prior + train-window
  RankIC filter + greedy correlation dedup (same idea as equal_weight_multi_v2),
  then Ridge on the selected subset.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

from factor_modeling.common import (
    KEYS,
    _daily_correlations,
    build_strategy_returns,
    construct_daily_portfolio,
    cross_sectional_rank_target,
    mean_daily_rank_ic,
    purge_last_dates,
    return_statistics,
)
from factor_modeling.equal_weight_multi import resolve_directions, select_fold_universe
from factor_modeling.handoff import (
    apply_label_end_purge,
    handoff_summary,
    load_fold_definitions,
    load_handoff_bundle,
    merge_features_and_labels,
    split_fold,
)
from factor_modeling.single_factor_rank import select_candidates

__all__ = [
    "KEYS",
    "_daily_correlations",
    "_fit_ridge",
    "_fold_metrics",
    "apply_label_end_purge",
    "build_strategy_returns",
    "choose_alpha",
    "construct_daily_portfolio",
    "cross_sectional_rank_target",
    "date_equal_sample_weight",
    "main",
    "purge_last_dates",
    "return_statistics",
    "run",
    "split_fold",
    "write_report",
]


def date_equal_sample_weight(frame: pd.DataFrame) -> np.ndarray:
    counts = frame.groupby("trade_date")["product"].transform("count").astype(float)
    return (1.0 / counts).to_numpy()


def _fit_ridge(
    train: pd.DataFrame,
    predict: pd.DataFrame,
    feature_names: list[str],
    target_column: str,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train[feature_names].to_numpy(dtype=np.float64))
    x_predict = scaler.transform(predict[feature_names].to_numpy(dtype=np.float64))
    model = Ridge(alpha=alpha, fit_intercept=True, solver="lsqr")
    model.fit(
        x_train,
        train[target_column].to_numpy(dtype=np.float64),
        sample_weight=date_equal_sample_weight(train),
    )
    return model.predict(x_predict), model.coef_.copy(), float(model.intercept_)


def choose_alpha(
    outer_train: pd.DataFrame,
    feature_names: list[str],
    config: dict[str, Any],
) -> tuple[float, pd.DataFrame]:
    raw_return = config["raw_return_column"]
    target_column = config["model_target_column"]
    label_end_column = config.get("label_end_column", "label_end_date_5d")
    inner_year = int(outer_train["trade_date"].dt.year.max())
    inner_train_raw = outer_train.loc[outer_train["trade_date"].dt.year.lt(inner_year)].copy()
    inner_valid = outer_train.loc[outer_train["trade_date"].dt.year.eq(inner_year)].copy()
    inner_valid_start = pd.Timestamp(f"{inner_year}-01-01")
    inner_train = apply_label_end_purge(
        inner_train_raw,
        inner_valid_start,
        label_end_column=label_end_column,
        purge_trading_days=int(config.get("purge_trading_days", 5)),
    )
    rows: list[dict[str, Any]] = []
    for alpha in config["alpha_grid"]:
        if inner_train.empty or inner_valid.empty:
            break
        prediction, _, _ = _fit_ridge(
            inner_train,
            inner_valid,
            feature_names,
            target_column,
            float(alpha),
        )
        evaluated = inner_valid[["trade_date", "product", raw_return]].copy()
        evaluated["prediction"] = prediction
        rows.append({
            "inner_validation_year": inner_year,
            "alpha": float(alpha),
            "mean_rank_ic": mean_daily_rank_ic(evaluated, return_column=raw_return),
            "rows": len(inner_valid),
        })
    scores = pd.DataFrame(rows).sort_values(["mean_rank_ic", "alpha"], ascending=[False, True])
    if scores.empty or scores["mean_rank_ic"].isna().all():
        return float(config["fallback_alpha"]), scores
    return float(scores.iloc[0]["alpha"]), scores


def _fold_metrics(
    fold_id: str,
    test_year: int,
    alpha: float,
    train: pd.DataFrame,
    predictions: pd.DataFrame,
    strategy: pd.DataFrame,
    config: dict[str, Any],
    *,
    prior_universe_size: int | None = None,
    selected_universe_size: int | None = None,
) -> dict[str, Any]:
    target_column = config["model_target_column"]
    raw_return = config["raw_return_column"]
    daily_ic = _daily_correlations(predictions, return_column=raw_return)
    rank_std = float(daily_ic["rank_ic"].std(ddof=1)) if len(daily_ic) > 1 else np.nan
    result: dict[str, Any] = {
        "fold_id": fold_id,
        "test_year": test_year,
        "selected_alpha": alpha,
        "selection_mode": str(config.get("selection_mode", "none") or "none"),
        "prior_universe_size": int(prior_universe_size) if prior_universe_size is not None else np.nan,
        "universe_size": int(selected_universe_size) if selected_universe_size is not None else np.nan,
        "train_start": train["trade_date"].min().date(),
        "train_end": train["trade_date"].max().date(),
        "train_rows": len(train),
        "test_rows": len(predictions),
        "test_days": predictions["trade_date"].nunique(),
        "rebalance_days": int(strategy["trade_date"].nunique()) if not strategy.empty else 0,
        "pearson_ic_mean": float(daily_ic["pearson_ic"].mean()) if not daily_ic.empty else np.nan,
        "rank_ic_mean": float(daily_ic["rank_ic"].mean()) if not daily_ic.empty else np.nan,
        "rank_icir": float(daily_ic["rank_ic"].mean() / rank_std) if rank_std and rank_std > 0 else np.nan,
        "rank_ic_hit_rate": float(daily_ic["rank_ic"].gt(0).mean()) if not daily_ic.empty else np.nan,
        "target_r2": float(r2_score(predictions[target_column], predictions["prediction"])),
        "target_mae": float(mean_absolute_error(predictions[target_column], predictions["prediction"])),
    }
    for key, value in return_statistics(strategy["gross_return"], float(config["annualization"])).items():
        result[f"gross_{key}"] = value
    for bps in config["one_way_cost_bps"]:
        bps = float(bps)
        label = str(int(bps)) if bps.is_integer() else str(bps).replace(".", "p")
        column = f"net_{label}bps_return"
        for key, value in return_statistics(strategy[column], float(config["annualization"])).items():
            result[f"net_{label}bps_{key}"] = value
    return result


def _format(value: Any, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    return str(value)


def _markdown(frame: pd.DataFrame) -> str:
    return frame.to_markdown(index=False) if not frame.empty else "暂无记录。"


def write_report(
    path: Path,
    config: dict[str, Any],
    fold_metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    strategy: pd.DataFrame,
    coefficient_summary: pd.DataFrame,
    alpha_search: pd.DataFrame,
    registry: pd.DataFrame,
    bundle_info: dict[str, Any],
    fold_selected: pd.DataFrame | None = None,
    v1_summary: dict[str, Any] | None = None,
    prior_universe_size: int | None = None,
) -> None:
    target_column = config["model_target_column"]
    raw_return = config["raw_return_column"]
    daily_ic = _daily_correlations(predictions, return_column=raw_return)
    selection_mode = str(config.get("selection_mode", "none") or "none")
    is_v2 = selection_mode == "train_ic_corr"
    model_version = str(config.get("model_version", "linear_ridge"))
    n_prior = int(prior_universe_size if prior_universe_size is not None else len(registry))
    mean_selected = (
        float(fold_metrics["universe_size"].mean())
        if "universe_size" in fold_metrics.columns and fold_metrics["universe_size"].notna().any()
        else float(n_prior)
    )

    overall = {
        "模型版本": model_version,
        "筛选模式": selection_mode,
        "样本外年份": f"{int(fold_metrics.test_year.min())}—{int(fold_metrics.test_year.max())}",
        "先验宇宙大小": n_prior,
        "折均入选数": mean_selected,
        "样本外记录": len(predictions),
        "交易日": predictions.trade_date.nunique(),
        "调仓日": int(strategy.trade_date.nunique()) if not strategy.empty else 0,
        "平均Pearson IC": daily_ic.pearson_ic.mean() if not daily_ic.empty else np.nan,
        "平均RankIC": daily_ic.rank_ic.mean() if not daily_ic.empty else np.nan,
        "RankIC胜率": daily_ic.rank_ic.gt(0).mean() if not daily_ic.empty else np.nan,
        "目标R²": r2_score(predictions[target_column], predictions.prediction),
        "目标MAE": mean_absolute_error(predictions[target_column], predictions.prediction),
    }
    gross = return_statistics(strategy["gross_return"], float(config["annualization"]))
    strategy_rows = [{"成本": "毛收益", **gross}]
    for bps in config["one_way_cost_bps"]:
        bps = float(bps)
        label = str(int(bps)) if bps.is_integer() else str(bps).replace(".", "p")
        strategy_rows.append({
            "成本": f"单边{bps:g}bp",
            **return_statistics(strategy[f"net_{label}bps_return"], float(config["annualization"])),
        })
    strategy_table = pd.DataFrame(strategy_rows)

    fold_display = fold_metrics.copy()
    for column in fold_display.columns:
        if pd.api.types.is_float_dtype(fold_display[column]):
            fold_display[column] = fold_display[column].map(_format)
    overall_table = pd.DataFrame([{"指标": key, "结果": _format(value)} for key, value in overall.items()])
    strategy_display = strategy_table.copy()
    for column in strategy_display.columns[1:]:
        strategy_display[column] = strategy_display[column].map(_format)
    coefficients = coefficient_summary.head(20).copy()
    for column in ["coefficient_mean", "coefficient_std", "mean_abs_coefficient", "sign_consistency"]:
        if column in coefficients.columns:
            coefficients[column] = coefficients[column].map(_format)
    alpha_display = alpha_search.copy()
    for column in ["alpha", "mean_rank_ic"]:
        if column in alpha_display.columns:
            alpha_display[column] = alpha_display[column].map(_format)

    group_cols = [c for c in ["usage_type", "classification", "category"] if c in registry.columns]
    if len(group_cols) >= 2:
        registry_counts = (
            registry.groupby(group_cols[:2], as_index=False).size().rename(columns={"size": "字段数"})
        )
    elif group_cols:
        registry_counts = (
            registry.groupby(group_cols[0], as_index=False).size().rename(columns={"size": "字段数"})
        )
    else:
        registry_counts = pd.DataFrame({"字段数": [n_prior]})

    handoff_table = pd.DataFrame([
        {"项目": "handoff目录", "值": bundle_info.get("handoff_directory", "")},
        {"项目": "特征文件", "值": bundle_info.get("feature_file", "")},
        {"项目": "注册表", "值": bundle_info.get("registry_file", "")},
        {"项目": "标签文件", "值": bundle_info.get("label_file", "")},
        {"项目": "折定义", "值": config.get("fold_file", "fold_definitions.csv")},
        {"项目": "注册特征数(handoff)", "值": bundle_info.get("n_features", "")},
        {"项目": "本模型先验宇宙", "值": n_prior},
        {"项目": "特征行数", "值": bundle_info.get("feature_rows", "")},
        {"项目": "特征日期", "值": f"{bundle_info.get('feature_start')} — {bundle_info.get('feature_end')}"},
        {"项目": "品种数", "值": bundle_info.get("n_products", "")},
    ])

    selection_block = ""
    if is_v2 and fold_selected is not None and not fold_selected.empty:
        per_fold = (
            fold_selected.loc[fold_selected["selected"].astype(bool)]
            .groupby(["fold_id", "test_year"], as_index=False)
            .agg(
                n_selected=("feature_name", "size"),
                selected_features=("feature_name", lambda s: ", ".join(sorted(map(str, s)))),
            )
            .sort_values("test_year")
        )
        per_fold_display = per_fold.copy()
        per_fold_display["test_year"] = per_fold_display["test_year"].astype(int).astype(str)
        per_fold_display["n_selected"] = per_fold_display["n_selected"].astype(int).astype(str)
        reason_counts = (
            fold_selected.groupby(["selection_reason"], as_index=False)
            .size()
            .rename(columns={"size": "记录数"})
            .sort_values("记录数", ascending=False)
        )
        selection_block = "\n".join([
            "## 折内训练窗筛选摘要",
            "",
            f"- `selection_mode = {selection_mode}`：仅用该折训练窗；**未**使用样本外 IC/Sharpe 挑因子。",
            f"- 剔弱阈值 `min_train_rank_ic = {config.get('min_train_rank_ic', 0.01)}`。",
            f"- 去重阈值 `max_pairwise_corr = {config.get('max_pairwise_corr', 0.7)}`。",
            "- 筛选后在入选集上再选 `alpha` 并拟合 Ridge。",
            "",
            "各折入选数量与名单：",
            "",
            _markdown(per_fold_display),
            "",
            "筛选原因计数（跨折汇总）：",
            "",
            _markdown(reason_counts),
            "",
        ])

    compare_block = ""
    if is_v2 and v1_summary:
        v1_rank = v1_summary.get("mean_rank_ic")
        v1_gross = (v1_summary.get("gross") or {}).get("sharpe")
        v1_net5 = (v1_summary.get("net_base_cost") or {}).get("sharpe")
        compare = pd.DataFrame([
            {
                "版本": "v1 全体92 Ridge",
                "宇宙/入选": str(v1_summary.get("features", 92)),
                "RankIC": _format(v1_rank),
                "毛Sharpe": _format(v1_gross),
                "净5bp Sharpe": _format(v1_net5),
            },
            {
                "版本": "v2 筛弱/去重 Ridge",
                "宇宙/入选": f"{n_prior}先验 / 折均{_format(mean_selected, 1)}",
                "RankIC": _format(overall["平均RankIC"]),
                "毛Sharpe": _format(gross.get("sharpe")),
                "净5bp Sharpe": _format(
                    return_statistics(
                        strategy["net_5bps_return"], float(config["annualization"])
                    ).get("sharpe")
                    if "net_5bps_return" in strategy.columns else np.nan
                ),
            },
        ])
        compare_block = "\n".join([
            "## 与 v1（全体 Ridge）对照",
            "",
            "同协议、同 handoff；v2 仅改变先验宇宙与折内训练窗筛选，再对入选集做 Ridge。",
            "",
            _markdown(compare),
            "",
        ])

    cost_text = "/".join(str(int(float(b))) if float(b).is_integer() else str(b) for b in config["one_way_cost_bps"])
    if is_v2:
        title = "# Ridge线性回归模型测试报告（筛弱/去重 v2）"
        blurb = (
            f"> 本报告：先验 predictor（N={n_prior}）+ 折内训练窗剔弱/去相关后，"
            "对入选集做 L2 Ridge；2025不参与模型选择或评价。"
        )
        model_bullet = (
            f"- 模型：`{model_version}` / Ridge + `selection_mode=train_ic_corr`；"
            f"先验 `{config.get('candidate_usage_types')}`，折均入选 {_format(mean_selected, 1)}。"
        )
        boundary_extra = (
            "6. 入选名单允许折间变化；阈值写死，未做网格搜索；v1 仍为 92 全体输入地板基准。"
        )
    else:
        title = "# Ridge线性回归模型测试报告"
        blurb = f"> 本报告由 handoff 矩阵接口的 {n_prior} 个注册字段自动生成；2025不参与模型选择或评价。"
        model_bullet = f"- 模型：带 L2 正则的 Ridge 线性回归，全部 {n_prior} 个注册字段统一输入。"
        boundary_extra = "6. 本模型不负责因子加工；更换因子时只需替换 handoff 矩阵文件。"

    lines = [
        title,
        "",
        blurb,
        "",
        "## Handoff 数据接口",
        "",
        _markdown(handoff_table),
        "",
        "## 模型与标签选择",
        "",
        model_bullet,
        f"- 原始收益标签：`{raw_return}`（T 收盘形成信号后，同一真实合约从 T+1 开盘持有到 T+h+1 开盘）。",
        f"- 训练标签：同日横截面中心化收益排名 `{target_column}`，范围[-1,1]。",
        "- 正则参数：每个扩展训练窗口内部用最后一年验证，按日均 RankIC 选择（仅在入选集上）。",
        "- 外层验证：读取 `fold_definitions.csv` 的扩展窗口（2015 起点不变，2019–2024 逐年验证）；"
        f"训练净化规则为 `{config.get('label_end_column', 'label_end_date_5d')} < valid_start`。",
        "",
        "## 策略定义",
        "",
        f"- 每个调仓信号日按预测值排序，最高{float(config['long_short_fraction'])*100:.0f}%做多、"
        f"最低{float(config['long_short_fraction'])*100:.0f}%做空。",
        "- 多头和空头各占0.5风险资金，组合净敞口0、总敞口1。",
        f"- 持有 {config.get('holding_days', 5)} 个交易日，每 {config.get('rebalance_frequency_days', 5)} 个交易日调仓。",
        f"- 主成本口径单边 {config.get('base_cost_bps_one_way', 5)} bp；敏感性报告单边 {cost_text} bp。",
        "- 真实交易所手续费仅作辅助诊断，不替代统一成本口径。",
        "",
        "## 输入字段组成（先验宇宙）",
        "",
        _markdown(registry_counts),
        "",
        selection_block,
        "## 总体样本外预测结果",
        "",
        _markdown(overall_table),
        "",
        compare_block,
        "## 策略总体结果",
        "",
        _markdown(strategy_display),
        "",
        "## 分年度扩展窗口结果",
        "",
        _markdown(fold_display),
        "",
        "## 训练窗口内部正则参数选择",
        "",
        _markdown(alpha_display),
        "",
        "## 标准化系数绝对值前20名",
        "",
        "系数反映条件关联而非因果；相关特征可能相互分摊权重。",
        "",
        _markdown(coefficients),
        "",
        "## 使用边界",
        "",
        "1. 本测试验证的是协议主策略：T收盘信号 + T+1开盘成交 + 5日持有/调仓。",
        "2. 若成本后收益为负，不能用毛收益结果声称策略可实盘。",
        "3. 2025不在本 handoff 包内，不得作为开发阶段独立封存期窥视对象。",
        "4. 没有核验合约乘数、真实手续费、冲击、涨跌停和成交容量，结果只属于研究回测。",
        "5. 筛选与方向定号仅使用该折训练窗，未使用测试段。",
        boundary_extra,
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    root = config_path.parents[2]
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_dir = root / config["output_directory"]
    report_path = root / config["report_file"]
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    selection_mode = str(config.get("selection_mode", "none") or "none")
    bundle = load_handoff_bundle(root, config)
    bundle_info = handoff_summary(bundle)
    candidates = select_candidates(bundle.registry, config)
    feature_names = candidates["feature_name"].tolist()
    missing = sorted(set(feature_names) - set(bundle.feature_names))
    if missing:
        raise ValueError(f"registered features missing from handoff matrix: {missing}")
    registry = candidates.copy()
    folds = load_fold_definitions(bundle.directory, config)

    raw_return = config["raw_return_column"]
    target_column = config["model_target_column"]
    label_end_column = config.get("label_end_column", "label_end_date_5d")
    min_samples = int(config.get("min_ic_samples", 5))

    data = merge_features_and_labels(bundle, raw_return_column=raw_return)
    # Keep only keys + prior features + labels needed.
    label_meta = [c for c in data.columns if c.startswith("label_end_date_") or c == raw_return]
    keep = [c for c in KEYS + ["sector"] + feature_names + label_meta if c in data.columns]
    data = data[keep].copy()
    data[target_column] = cross_sectional_rank_target(data, raw_return)
    data = data.dropna(subset=[target_column, raw_return]).copy()

    development_end = config.get("development_end")
    if development_end:
        data = data.loc[data["trade_date"].le(pd.Timestamp(development_end))].copy()

    print(f"prior universe size N={len(feature_names)} selection_mode={selection_mode}")
    registry[
        [c for c in ["feature_name", "source_factor", "usage_type", "expected_direction", "category"] if c in registry.columns]
    ].rename(columns={"expected_direction": "catalog_direction"}).to_csv(
        output_dir / "universe_factors.csv", index=False
    )

    all_predictions: list[pd.DataFrame] = []
    all_coefficients: list[pd.DataFrame] = []
    all_fold_metrics: list[dict[str, Any]] = []
    all_alpha_search: list[pd.DataFrame] = []
    all_strategy: list[pd.DataFrame] = []
    all_selected: list[pd.DataFrame] = []

    for fold in folds.itertuples(index=False):
        outer_train, outer_test = split_fold(
            data,
            fold,
            target_column=target_column,
            label_end_column=label_end_column,
        )
        if outer_train.empty or outer_test.empty:
            raise ValueError(f"empty train/test split for fold {fold.fold_id}")

        test_year = int(pd.Timestamp(fold.valid_start).year)
        directions = resolve_directions(
            outer_train, candidates, raw_return, min_samples
        )
        selected_names, selection_audit = select_fold_universe(
            outer_train, candidates, directions, config
        )
        if selection_mode in {"none", "", "all_equal"}:
            selected_names = [n for n in feature_names if n in set(selected_names)]
        selection_audit = selection_audit.copy()
        selection_audit["fold_id"] = fold.fold_id
        selection_audit["test_year"] = test_year
        selection_audit["prior_universe_size"] = int(len(feature_names))
        selection_audit["selected_universe_size"] = int(len(selected_names))
        all_selected.append(selection_audit)

        # Ridge uses 0-fill only on the selected design matrix.
        train_fit = outer_train.copy()
        test_fit = outer_test.copy()
        train_fit[selected_names] = train_fit[selected_names].fillna(0.0)
        test_fit[selected_names] = test_fit[selected_names].fillna(0.0)

        alpha, alpha_search = choose_alpha(train_fit, selected_names, config)
        alpha_search.insert(0, "fold_id", fold.fold_id)
        alpha_search.insert(1, "outer_test_year", test_year)
        alpha_search.insert(2, "n_selected_features", len(selected_names))
        prediction, coefficients, intercept = _fit_ridge(
            train_fit,
            test_fit,
            selected_names,
            target_column,
            alpha,
        )
        keep_cols = ["trade_date", "product", "sector", raw_return, target_column]
        predicted = outer_test[keep_cols].copy()
        predicted["prediction"] = prediction
        predicted["fold_id"] = fold.fold_id
        predicted["test_year"] = test_year
        predicted["selected_alpha"] = alpha
        predicted["model_intercept"] = intercept
        predicted["n_features_used"] = len(selected_names)

        strategy = build_strategy_returns(predicted, config)
        strategy["fold_id"] = fold.fold_id
        strategy["test_year"] = test_year
        fold_metric = _fold_metrics(
            str(fold.fold_id),
            test_year,
            alpha,
            train_fit,
            predicted,
            strategy,
            config,
            prior_universe_size=len(feature_names),
            selected_universe_size=len(selected_names),
        )
        coefficient_frame = pd.DataFrame({
            "feature_name": selected_names,
            "fold_id": fold.fold_id,
            "test_year": test_year,
            "selected_alpha": alpha,
            "standardized_coefficient": coefficients,
            "selected": True,
        })
        all_predictions.append(predicted)
        all_coefficients.append(coefficient_frame)
        all_fold_metrics.append(fold_metric)
        all_alpha_search.append(alpha_search)
        all_strategy.append(strategy)
        print(
            f"{fold.fold_id}: selected={len(selected_names)}/{len(feature_names)} "
            f"alpha={alpha:g} rank_ic={fold_metric.get('rank_ic_mean'):.4f} "
            f"gross_sharpe={fold_metric.get('gross_sharpe'):.4f} "
            f"net_5bps_sharpe={fold_metric.get('net_5bps_sharpe'):.4f}"
        )

    predictions = pd.concat(all_predictions, ignore_index=True)
    coefficients = pd.concat(all_coefficients, ignore_index=True)
    fold_metrics = pd.DataFrame(all_fold_metrics)
    alpha_search = pd.concat(all_alpha_search, ignore_index=True)
    strategy = pd.concat(all_strategy, ignore_index=True)
    fold_selected = pd.concat(all_selected, ignore_index=True) if all_selected else pd.DataFrame()

    coefficient_summary = (
        coefficients.groupby("feature_name", as_index=False)
        .agg(
            coefficient_mean=("standardized_coefficient", "mean"),
            coefficient_std=("standardized_coefficient", "std"),
            mean_abs_coefficient=("standardized_coefficient", lambda values: float(np.abs(values).mean())),
            positive_share=("standardized_coefficient", lambda values: float((values > 0).mean())),
            folds=("test_year", "nunique"),
        )
    )
    coefficient_summary["sign_consistency"] = coefficient_summary["positive_share"].combine(
        1.0 - coefficient_summary["positive_share"], max
    )
    meta_cols = [
        c for c in ["feature_name", "source_factor", "usage_type", "classification", "category",
                     "correlation_cluster", "cluster_representative"]
        if c in registry.columns
    ]
    coefficient_summary = coefficient_summary.merge(
        registry[meta_cols],
        on="feature_name",
        how="left",
        validate="one_to_one",
    ).sort_values("mean_abs_coefficient", ascending=False)

    predictions.to_csv(output_dir / "oos_predictions.csv.gz", index=False, compression="gzip")
    strategy.to_csv(output_dir / "strategy_daily_returns.csv", index=False)
    fold_metrics.to_csv(output_dir / "fold_metrics.csv", index=False)
    alpha_search.to_csv(output_dir / "alpha_search.csv", index=False)
    coefficients.to_csv(output_dir / "coefficients_by_fold.csv", index=False)
    coefficient_summary.to_csv(output_dir / "coefficient_summary.csv", index=False)
    fold_selected.to_csv(output_dir / "fold_selected_factors.csv", index=False)
    daily_ic = _daily_correlations(predictions, return_column=raw_return)
    daily_ic.to_csv(output_dir / "daily_prediction_ic.csv", index=False)
    folds.to_csv(output_dir / "fold_definitions_used.csv", index=False)
    (output_dir / "handoff_summary.json").write_text(
        json.dumps(bundle_info, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    v1_summary: dict[str, Any] | None = None
    v1_summary_path = root / "modeling" / "linear_ridge_v1" / "outputs" / "run_summary.json"
    if selection_mode == "train_ic_corr" and v1_summary_path.exists():
        try:
            v1_summary = json.loads(v1_summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            v1_summary = None

    write_report(
        report_path,
        config,
        fold_metrics,
        predictions,
        strategy,
        coefficient_summary,
        alpha_search,
        registry,
        bundle_info,
        fold_selected=fold_selected,
        v1_summary=v1_summary,
        prior_universe_size=len(feature_names),
    )
    base_bps = float(config.get("base_cost_bps_one_way", 5))
    base_label = str(int(base_bps)) if base_bps.is_integer() else str(base_bps).replace(".", "p")
    base_net_col = f"net_{base_label}bps_return"
    mean_selected = (
        float(fold_metrics["universe_size"].mean())
        if "universe_size" in fold_metrics.columns and not fold_metrics.empty
        else float(len(feature_names))
    )
    summary = {
        "model": "Ridge",
        "model_version": config.get("model_version", "linear_ridge"),
        "selection_mode": selection_mode,
        "min_train_rank_ic": config.get("min_train_rank_ic"),
        "max_pairwise_corr": config.get("max_pairwise_corr"),
        "protocol": {
            "primary_target": raw_return,
            "split": "expanding_window_from_fold_definitions",
            "holding_days": config.get("holding_days", 5),
            "rebalance_frequency_days": config.get("rebalance_frequency_days", 5),
            "base_cost_bps_one_way": base_bps,
            "cost_scenarios_bps_one_way": config["one_way_cost_bps"],
        },
        "prior_universe_size": len(feature_names),
        "features": len(feature_names),
        "mean_selected_universe_size": mean_selected,
        "handoff": bundle_info,
        "folds": [str(x) for x in folds["fold_id"].tolist()],
        "oos_rows": len(predictions),
        "oos_days": int(predictions["trade_date"].nunique()),
        "rebalance_days": int(strategy["trade_date"].nunique()) if not strategy.empty else 0,
        "mean_rank_ic": float(daily_ic["rank_ic"].mean()) if not daily_ic.empty else None,
        "rank_ic_hit_rate": float(daily_ic["rank_ic"].gt(0).mean()) if not daily_ic.empty else None,
        "gross": return_statistics(strategy["gross_return"], float(config["annualization"])),
        "net_base_cost": return_statistics(strategy[base_net_col], float(config["annualization"]))
        if base_net_col in strategy.columns else None,
        "net_3bps": return_statistics(strategy["net_3bps_return"], float(config["annualization"]))
        if "net_3bps_return" in strategy.columns else None,
        "v1_comparison": {
            "mean_rank_ic": v1_summary.get("mean_rank_ic") if v1_summary else None,
            "gross_sharpe": (v1_summary.get("gross") or {}).get("sharpe") if v1_summary else None,
            "net_5bps_sharpe": (v1_summary.get("net_base_cost") or {}).get("sharpe") if v1_summary else None,
        } if selection_mode == "train_ic_corr" else None,
        "report": str(report_path),
        "note": (
            "Ridge after predictor prior + train-window RankIC filter and greedy correlation dedup"
            if selection_mode == "train_ic_corr"
            else "Ridge on configured prior universe; no train-window factor selection"
        ),
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Handoff矩阵接口上的Ridge线性回归（协议主策略）")
    parser.add_argument("--config", default="modeling/linear_ridge_v1/config.json")
    args = parser.parse_args(argv)
    print(json.dumps(run(args.config), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
