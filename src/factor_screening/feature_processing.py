from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ProjectConfig


def _robust_cross_sectional_z(data: pd.DataFrame, columns: list[str], clip: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    values = data[columns].copy()
    missing = values.isna()
    med = values.groupby(data["trade_date"]).transform("median")
    mad = (values - med).abs().groupby(data["trade_date"]).transform("median")
    scale = (1.4826 * mad).replace(0, np.nan)
    standardized = ((values - med) / scale).clip(-clip, clip)
    standardized = standardized.fillna(0.0)
    return standardized, missing


def _daily_orthogonal_residual(data: pd.DataFrame, candidate: str, representative: str) -> pd.Series:
    frame = data[["trade_date", candidate, representative]].copy()
    x_mean = frame.groupby("trade_date")[representative].transform("mean")
    y_mean = frame.groupby("trade_date")[candidate].transform("mean")
    xd = frame[representative] - x_mean
    yd = frame[candidate] - y_mean
    cov = (xd * yd).groupby(frame["trade_date"]).transform("sum")
    var = (xd * xd).groupby(frame["trade_date"]).transform("sum").replace(0, np.nan)
    beta = cov / var
    return yd - beta * xd


def build_ml_features(
    data: pd.DataFrame,
    feature_classes: pd.DataFrame,
    config: ProjectConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = feature_classes.loc[feature_classes["enter_ml"]].copy()
    selected_names = selected["feature_name"].tolist()
    clip = float(config.raw["features"]["mad_clip"])
    standardized, missing = _robust_cross_sectional_z(data, selected_names, clip)
    output = data[["trade_date", "product", "sector"]].copy()
    registry_rows: list[dict[str, object]] = []
    for name in selected_names:
        final_name = f"{name}__cs_robust_z"
        output[final_name] = standardized[name]
        row = selected.loc[selected["feature_name"].eq(name)].iloc[0]
        registry_rows.append({
            "feature_name": final_name,
            "source_factor": row["source_factor"],
            "source_factor_id": int(row["source_factor_id"]),
            "transformation": f"同日横截面中位数/MAD去极值({clip})并标准化；缺失填0；模型折内再次仅用训练期缩放",
            "category": row["category"],
            "economic_meaning": row["economic_meaning"],
            "available_date": "因子日T收盘后形成，T+1开盘可用",
            "missing_rate": float(missing[name].mean()),
            "usage_type": row["usage_type"],
            "classification": row["classification"],
            "correlation_cluster": row.get("correlation_cluster", np.nan),
        })
        if float(missing[name].mean()) >= float(config.raw["features"]["missing_indicator_threshold"]):
            indicator_name = f"{name}__missing"
            output[indicator_name] = missing[name].astype("int8")
            registry_rows.append({
                "feature_name": indicator_name,
                "source_factor": row["source_factor"],
                "source_factor_id": int(row["source_factor_id"]),
                "transformation": "原始特征缺失指示器",
                "category": "数据质量标签",
                "economic_meaning": "区分真实中性值与数据缺失造成的零填充",
                "available_date": "与源因子同步",
                "missing_rate": 0.0,
                "usage_type": "quality_flag",
                "classification": row["classification"],
                "correlation_cluster": row.get("correlation_cluster", np.nan),
            })

    cluster_reps = selected.loc[selected["cluster_representative"].fillna(False)].set_index("correlation_cluster")["feature_name"].to_dict()
    orth_candidates = selected.loc[
        ~selected["cluster_representative"].fillna(True)
        & selected["delta_rank_ic"].fillna(0).ge(0.001)
    ]
    for row in orth_candidates.itertuples():
        representative = cluster_reps.get(row.correlation_cluster)
        if not representative or representative not in selected_names:
            continue
        name = f"orth__{row.feature_name}__vs__{representative}"
        residual = _daily_orthogonal_residual(standardized.assign(trade_date=data["trade_date"]), row.feature_name, representative)
        output[name] = residual.fillna(0.0)
        registry_rows.append({
            "feature_name": name,
            "source_factor": row.source_factor,
            "source_factor_id": int(row.source_factor_id),
            "transformation": f"同日横截面对簇代表{representative}回归后的残差",
            "category": row.category,
            "economic_meaning": f"保留{row.feature_name}相对高相关代表因子的增量部分",
            "available_date": "因子日T收盘后形成，T+1开盘可用",
            "missing_rate": float(residual.isna().mean()),
            "usage_type": "orthogonalized_predictor",
            "classification": row.classification,
            "correlation_cluster": row.correlation_cluster,
        })

    output["feature_available_count"] = (~missing).sum(axis=1).astype("int16")
    registry_rows.append({
        "feature_name": "feature_available_count",
        "source_factor": "全体入选因子",
        "source_factor_id": 0,
        "transformation": "逐行统计原始可用特征数",
        "category": "数据质量标签",
        "economic_meaning": "刻画该品种日在模型输入端的信息完整度",
        "available_date": "与源因子同步",
        "missing_rate": 0.0,
        "usage_type": "quality_flag",
        "classification": "C",
        "correlation_cluster": np.nan,
    })
    return output, pd.DataFrame(registry_rows)
