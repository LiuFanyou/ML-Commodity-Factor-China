from __future__ import annotations

import numpy as np
import pandas as pd

from .catalog import FACTOR_CATALOG
from .config import ProjectConfig
from .factors import FeatureDefinition


CLASS_ORDER = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5}


def classify_features(
    summary: pd.DataFrame,
    yearly: pd.DataFrame,
    product_metrics: pd.DataFrame,
    sector_metrics: pd.DataFrame,
    definitions: list[FeatureDefinition],
    incremental: pd.DataFrame,
    cluster_info: pd.DataFrame,
    correlation: pd.DataFrame,
    config: ProjectConfig,
) -> pd.DataFrame:
    primary = summary.loc[summary["horizon"].eq(config.primary_horizon)].copy()
    portfolio = summary.loc[summary["horizon"].eq(1)].set_index("feature_name")
    portfolio_columns = [
        "gross_annual_return",
        "gross_sharpe",
        "gross_max_drawdown",
        "average_one_way_turnover",
        "net_1bps_annual_return",
        "net_1bps_sharpe",
        "net_3bps_annual_return",
        "net_3bps_sharpe",
        "net_5bps_annual_return",
        "net_5bps_sharpe",
    ]
    for column in portfolio_columns:
        if column in portfolio:
            primary[column] = primary["feature_name"].map(portfolio[column])
    definition_frame = pd.DataFrame([definition.__dict__ for definition in definitions])
    catalog = pd.DataFrame([spec.__dict__ for spec in FACTOR_CATALOG])
    primary = primary.merge(definition_frame, on=["feature_name", "source_factor_id", "source_factor"], how="left")
    primary = primary.merge(
        catalog[["factor_id", "category", "usage_type", "expected_direction", "economic_meaning"]],
        left_on="source_factor_id",
        right_on="factor_id",
        how="left",
    ).drop(columns="factor_id")
    primary = primary.merge(incremental, on="feature_name", how="left")
    primary = primary.merge(cluster_info, on="feature_name", how="left", suffixes=("", "_cluster"))

    max_corr: dict[str, float] = {}
    for feature in correlation.columns:
        values = correlation[feature].drop(index=feature, errors="ignore").abs()
        max_corr[feature] = float(values.max()) if len(values) else np.nan
    primary["max_abs_factor_correlation"] = primary["feature_name"].map(max_corr)

    y = yearly.loc[yearly["horizon"].eq(config.primary_horizon)].copy()
    consistency_rows: list[dict[str, object]] = []
    for feature, part in y.groupby("feature_name"):
        overall = primary.loc[primary["feature_name"].eq(feature), "rank_ic_mean"]
        direction = primary.loc[primary["feature_name"].eq(feature), "expected_direction"]
        if overall.empty or direction.empty:
            continue
        desired = int(direction.iloc[0])
        if desired == 0:
            desired = 1 if float(overall.iloc[0]) >= 0 else -1
        valid = part["rank_ic_mean"].dropna()
        consistency_rows.append({
            "feature_name": feature,
            "year_direction_consistency": float((np.sign(valid) == desired).mean()) if len(valid) else np.nan,
            "year_count": int(len(valid)),
        })
    primary = primary.merge(pd.DataFrame(consistency_rows), on="feature_name", how="left")

    cross_section_rows: list[dict[str, object]] = []
    for feature in primary["feature_name"]:
        base = primary.loc[primary["feature_name"].eq(feature)].iloc[0]
        desired = int(base["expected_direction"])
        if desired == 0:
            desired = 1 if float(base["rank_ic_mean"]) >= 0 else -1
        product_part = product_metrics.loc[product_metrics["feature_name"].eq(feature), "correlation"].dropna()
        sector_part = sector_metrics.loc[sector_metrics["feature_name"].eq(feature), "correlation"].dropna()
        cross_section_rows.append({
            "feature_name": feature,
            "product_direction_consistency": float((np.sign(product_part) == desired).mean()) if len(product_part) else np.nan,
            "sector_direction_consistency": float((np.sign(sector_part) == desired).mean()) if len(sector_part) else np.nan,
            "worst_product_correlation": float((product_part * desired).min()) if len(product_part) else np.nan,
            "worst_sector_correlation": float((sector_part * desired).min()) if len(sector_part) else np.nan,
        })
    primary = primary.merge(pd.DataFrame(cross_section_rows), on="feature_name", how="left")

    minimum_coverage = float(config.raw["screening"]["minimum_factor_coverage"])
    classes: list[str] = []
    reasons: list[str] = []
    for row in primary.itertuples():
        if row.coverage < minimum_coverage:
            classes.append("F")
            reasons.append("可用覆盖率低于最低门槛")
            continue
        if row.usage_type == "risk":
            classes.append("D")
            reasons.append("主要用途是风险、波动或流动性刻画")
            continue
        if row.usage_type == "conditional":
            classes.append("C")
            reasons.append("经济角色主要是条件变量而非独立方向信号")
            continue
        desired = int(row.expected_direction)
        directional_ic = float(row.rank_ic_mean) * desired if desired else abs(float(row.rank_ic_mean))
        directional_spread = float(row.group_spread) * desired if desired else abs(float(row.group_spread))
        stable = pd.notna(row.year_direction_consistency) and row.year_direction_consistency >= 0.57
        strong = (
            desired != 0
            and pd.notna(row.rank_ic_q_bh)
            and row.rank_ic_q_bh <= 0.10
            and directional_ic > 0
            and directional_spread > 0
            and stable
        )
        auxiliary = (
            (pd.notna(row.rank_ic_q_bh) and row.rank_ic_q_bh <= 0.25 and directional_ic > 0)
            or (pd.notna(row.delta_rank_ic) and row.delta_rank_ic >= 0.001)
            or (directional_ic >= 0.01 and stable)
        )
        if strong:
            classes.append("A")
            reasons.append("FDR后显著、方向一致且分组收益同向")
        elif auxiliary:
            classes.append("B")
            if desired == 0:
                reasons.append("存在预测或增量信息，但经济方向未预先固定，暂列辅助")
            elif pd.notna(row.delta_rank_ic) and row.delta_rank_ic >= 0.001:
                reasons.append("相对固定基准模型提供样本外增量信息")
            else:
                reasons.append("具有较弱但方向一致的样本外预测证据")
        else:
            classes.append("E")
            reasons.append("开发期未发现足够稳定的方向或增量证据")
    primary["classification"] = classes
    primary["classification_reason"] = reasons
    representative = primary["cluster_representative"].fillna(True)
    incremental_ok = primary["delta_rank_ic"].fillna(0).ge(0.001)
    primary["enter_ml"] = primary["classification"].isin(["A", "B", "C", "D"]) & (representative | incremental_ok)
    return primary


def classify_conceptual_factors(feature_classes: pd.DataFrame) -> pd.DataFrame:
    catalog = pd.DataFrame([spec.__dict__ for spec in FACTOR_CATALOG])
    rows: list[dict[str, object]] = []
    for spec in FACTOR_CATALOG:
        candidates = feature_classes.loc[feature_classes["source_factor_id"].eq(spec.factor_id)].copy()
        if spec.implementation_status.startswith("missing"):
            classification = "F"
            reason = spec.missing_reason
            representative = ""
        elif spec.implementation_status == "deferred":
            classification = "E"
            reason = spec.missing_reason
            representative = ""
        elif candidates.empty:
            classification = "F"
            reason = "实现目录中没有对应特征实例"
            representative = ""
        else:
            candidates["class_order"] = candidates["classification"].map(CLASS_ORDER)
            candidates = candidates.sort_values(
                ["class_order", "rank_ic_q_bh", "coverage", "feature_name"],
                ascending=[True, True, False, True],
            )
            best = candidates.iloc[0]
            classification = str(best["classification"])
            reason = str(best["classification_reason"])
            representative = str(best["feature_name"])
        rows.append({
            "factor_id": spec.factor_id,
            "factor_name": spec.name_cn,
            "factor_name_en": spec.name_en,
            "category": spec.category,
            "data_source": spec.data_source,
            "implementation_status": spec.implementation_status,
            "economic_meaning": spec.economic_meaning,
            "classification": classification,
            "classification_reason": reason,
            "representative_feature": representative,
            "computable": not classification == "F" and spec.implementation_status != "deferred",
            "enter_ml": bool(candidates["enter_ml"].any()) if not candidates.empty else False,
        })
    return pd.DataFrame(rows)
