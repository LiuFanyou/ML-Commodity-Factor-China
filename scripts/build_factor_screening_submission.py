from __future__ import annotations

import hashlib
import json
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SUBMISSION = ROOT / "submission" / "factor_screening_experiment_20260723"


def fmt(value: object, digits: int = 4) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value):.{digits}f}"


def pct(value: object, digits: int = 1) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{100.0 * float(value):.{digits}f}%"


def performance(values: pd.Series) -> dict[str, float]:
    returns = values.dropna().astype(float)
    if returns.empty:
        return {
            "total_return": np.nan,
            "annual_return": np.nan,
            "annual_volatility": np.nan,
            "sharpe": np.nan,
            "max_drawdown": np.nan,
            "calmar": np.nan,
            "win_rate": np.nan,
        }
    periods_per_year = 252.0 / 5.0
    equity = (1.0 + returns).cumprod()
    years = len(returns) / periods_per_year
    annual_return = (
        float(equity.iloc[-1] ** (1.0 / years) - 1.0)
        if years > 0 and equity.iloc[-1] > 0
        else np.nan
    )
    volatility = float(returns.std(ddof=1) * math.sqrt(periods_per_year))
    sharpe = (
        float(returns.mean() / returns.std(ddof=1) * math.sqrt(periods_per_year))
        if returns.std(ddof=1) > 0
        else np.nan
    )
    drawdown = equity / equity.cummax() - 1.0
    maximum_drawdown = float(drawdown.min())
    return {
        "total_return": float(equity.iloc[-1] - 1.0),
        "annual_return": annual_return,
        "annual_volatility": volatility,
        "sharpe": sharpe,
        "max_drawdown": maximum_drawdown,
        "calmar": (
            float(annual_return / abs(maximum_drawdown))
            if pd.notna(annual_return) and maximum_drawdown < 0
            else np.nan
        ),
        "win_rate": float(returns.gt(0).mean()),
    }


def markdown_table(frame: pd.DataFrame, columns: list[tuple[str, str]]) -> list[str]:
    lines = [
        "| " + " | ".join(label for _, label in columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for row in frame.itertuples(index=False):
        values: list[str] = []
        mapping = row._asdict()
        for column, _ in columns:
            value = mapping[column]
            if isinstance(value, (float, np.floating)):
                values.append(fmt(value))
            elif pd.isna(value):
                values.append("—")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def build_period_returns(
    development: pd.DataFrame,
    sealed: pd.DataFrame,
    labels: pd.DataFrame,
) -> pd.DataFrame:
    date_map = (
        labels[
            ["trade_date", "entry_date", "label_end_date_5d"]
        ]
        .dropna(subset=["label_end_date_5d"])
        .drop_duplicates()
    )
    uniqueness = date_map.groupby("trade_date").agg(
        entry_count=("entry_date", "nunique"),
        exit_count=("label_end_date_5d", "nunique"),
    )
    if not uniqueness["entry_count"].le(1).all() or not uniqueness["exit_count"].le(1).all():
        raise AssertionError("a portfolio signal date maps to multiple entry or exit dates")
    date_map = date_map.drop_duplicates("trade_date")
    development = development.assign(sample_segment="development_2019_2024")
    sealed = sealed.assign(sample_segment="sealed_2025")
    combined = pd.concat([development, sealed], ignore_index=True)
    combined["trade_date"] = pd.to_datetime(combined["trade_date"])
    combined = combined.merge(
        date_map,
        on="trade_date",
        how="left",
        validate="many_to_one",
    )
    combined = combined.rename(
        columns={
            "trade_date": "signal_date",
            "label_end_date_5d": "exit_date",
        }
    )
    if combined["exit_date"].isna().any():
        raise AssertionError("portfolio return is missing an exit date")
    crossing = (
        combined["sample_segment"].eq("development_2019_2024")
        & pd.to_datetime(combined["exit_date"]).gt(pd.Timestamp("2024-12-31"))
    )
    if crossing.any():
        raise AssertionError("2025 target leaked into development period returns")
    columns = [
        "sample_segment",
        "factor_id",
        "signal_date",
        "entry_date",
        "exit_date",
        "gross_return",
        "turnover",
        "long_count",
        "short_count",
        "net_return_0bps",
        "net_return_3bps",
        "net_return_5bps",
        "net_return_10bps",
    ]
    return combined[columns].sort_values(
        ["factor_id", "signal_date", "sample_segment"]
    )


def build_metrics(
    period_returns: pd.DataFrame,
    development_metrics: pd.DataFrame,
    sealed_metrics: pd.DataFrame,
    horizons: pd.DataFrame,
) -> pd.DataFrame:
    names = development_metrics.set_index("factor_id")
    sealed_lookup = sealed_metrics.set_index("factor_id")
    horizon_lookup = (
        horizons.assign(horizon_days=horizons["horizon_days"].astype(str))
        .pivot(index="factor_id", columns="horizon_days", values="rank_ic")
    )
    rows: list[dict[str, object]] = []
    available_factor_ids = names.loc[names["available"].astype(bool)].index.tolist()
    for segment in ["development_2019_2024", "sealed_2025"]:
        for factor_id in available_factor_ids:
            subset = period_returns.loc[
                period_returns["sample_segment"].eq(segment)
                & period_returns["factor_id"].eq(factor_id)
            ]
            source = (
                names.loc[factor_id]
                if segment.startswith("development")
                else sealed_lookup.loc[factor_id]
            )
            for cost in [3, 5]:
                stats = performance(subset[f"net_return_{cost}bps"])
                row = {
                    "sample_segment": segment,
                    "factor_no": int(names.at[factor_id, "factor_no"]),
                    "factor_id": factor_id,
                    "factor_name": names.at[factor_id, "factor_name"],
                    "screening_class": names.at[factor_id, "screening_class"],
                    "cost_bps_one_way": cost,
                    "period_count": int(len(subset)),
                    "signal_start": subset["signal_date"].min(),
                    "signal_end": subset["signal_date"].max(),
                    "exit_start": subset["exit_date"].min(),
                    "exit_end": subset["exit_date"].max(),
                    **stats,
                    "mean_turnover_per_rebalance": float(subset["turnover"].mean()),
                    "daily_equivalent_turnover": float(subset["turnover"].mean() / 5.0),
                    "mean_long_count": float(subset["long_count"].mean()),
                    "mean_short_count": float(subset["short_count"].mean()),
                    "rank_ic_5d": source.get("rank_ic", np.nan),
                    "rank_ic_ir_5d": source.get("rank_ic_ir", np.nan),
                    "rank_ic_hit_rate": source.get("rank_ic_hit_rate", np.nan),
                    "monotonicity": source.get("monotonicity", np.nan),
                    "rank_ic_fdr_q": source.get("rank_ic_fdr_q", np.nan),
                    "portfolio_fdr_q": source.get("portfolio_fdr_q", np.nan),
                }
                if segment.startswith("development") and factor_id in horizon_lookup.index:
                    row["rank_ic_1d"] = horizon_lookup.at[factor_id, "1"]
                    row["rank_ic_20d"] = horizon_lookup.at[factor_id, "20"]
                    row["delayed_1d_rank_ic_5d"] = horizon_lookup.at[
                        factor_id, "5d_signal_delayed_1d"
                    ]
                else:
                    row["rank_ic_1d"] = np.nan
                    row["rank_ic_20d"] = np.nan
                    row["delayed_1d_rank_ic_5d"] = np.nan
                rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["sample_segment", "factor_no", "cost_bps_one_way"]
    )


def build_daily_nav(
    period_returns: pd.DataFrame,
    labels: pd.DataFrame,
) -> pd.DataFrame:
    calendar = pd.Series(
        sorted(pd.to_datetime(labels["trade_date"].dropna().unique()))
    )
    start = pd.to_datetime(period_returns["signal_date"]).min()
    end = pd.to_datetime(period_returns["exit_date"]).max()
    calendar = calendar.loc[calendar.between(start, end)]
    rows: list[pd.DataFrame] = []
    for factor_id, factor_returns in period_returns.groupby("factor_id", sort=True):
        for cost in [3, 5]:
            value_column = f"net_return_{cost}bps"
            exits = factor_returns[
                ["exit_date", "signal_date", "sample_segment", value_column]
            ].copy()
            exits["exit_date"] = pd.to_datetime(exits["exit_date"])
            if exits["exit_date"].duplicated().any():
                raise AssertionError(f"duplicate exit dates for {factor_id}")
            daily = pd.DataFrame({"trade_date": calendar})
            daily = daily.merge(
                exits,
                left_on="trade_date",
                right_on="exit_date",
                how="left",
                validate="one_to_one",
            )
            daily["factor_id"] = factor_id
            daily["cost_bps_one_way"] = cost
            daily["is_exit_valuation_date"] = daily[value_column].notna()
            daily["period_net_return_recognized"] = daily[value_column].fillna(0.0)
            daily["nav"] = (1.0 + daily["period_net_return_recognized"]).cumprod()
            daily["valuation_method"] = (
                "exit_date_recognition_non_overlapping_5d_not_daily_mark_to_market"
            )
            rows.append(
                daily[
                    [
                        "trade_date",
                        "factor_id",
                        "cost_bps_one_way",
                        "sample_segment",
                        "signal_date",
                        "is_exit_valuation_date",
                        "period_net_return_recognized",
                        "nav",
                        "valuation_method",
                    ]
                ]
            )
    return pd.concat(rows, ignore_index=True).sort_values(
        ["factor_id", "cost_bps_one_way", "trade_date"]
    )


def build_report(
    manifest: dict[str, object],
    metrics: pd.DataFrame,
    factor_metrics: pd.DataFrame,
    sealed_metrics: pd.DataFrame,
    feature_registry: pd.DataFrame,
    factor_registry: pd.DataFrame,
    parameter_metrics: pd.DataFrame,
    incremental: pd.DataFrame,
    analysis_tail: str,
) -> str:
    class_counts = (
        factor_metrics["screening_class"].str.split().str[0].value_counts()
    )
    b_factors = factor_metrics.loc[
        factor_metrics["screening_class"].str.startswith("B")
    ].copy()
    dev_3 = metrics.loc[
        metrics["sample_segment"].eq("development_2019_2024")
        & metrics["cost_bps_one_way"].eq(3)
    ]
    dev_5 = metrics.loc[
        metrics["sample_segment"].eq("development_2019_2024")
        & metrics["cost_bps_one_way"].eq(5)
    ]
    sealed_5 = metrics.loc[
        metrics["sample_segment"].eq("sealed_2025")
        & metrics["cost_bps_one_way"].eq(5)
    ]
    b_summary = (
        b_factors[
            [
                "factor_no",
                "factor_id",
                "factor_name",
                "rank_ic",
                "rank_ic_ir",
                "rank_ic_fdr_q",
                "positive_year_rate",
            ]
        ]
        .merge(
            dev_3[
                [
                    "factor_id",
                    "annual_return",
                    "sharpe",
                    "max_drawdown",
                    "win_rate",
                    "daily_equivalent_turnover",
                ]
            ].rename(
                columns={
                    "annual_return": "annual_return_3bps",
                    "sharpe": "sharpe_3bps",
                    "max_drawdown": "max_drawdown_3bps",
                    "win_rate": "win_rate_3bps",
                }
            ),
            on="factor_id",
            how="left",
        )
        .merge(
            dev_5[
                ["factor_id", "annual_return", "sharpe"]
            ].rename(
                columns={
                    "annual_return": "annual_return_5bps",
                    "sharpe": "sharpe_5bps",
                }
            ),
            on="factor_id",
            how="left",
        )
        .merge(
            sealed_5[
                ["factor_id", "rank_ic_5d", "annual_return", "sharpe"]
            ].rename(
                columns={
                    "rank_ic_5d": "sealed_rank_ic",
                    "annual_return": "sealed_annual_return_5bps",
                    "sharpe": "sealed_sharpe_5bps",
                }
            ),
            on="factor_id",
            how="left",
        )
        .sort_values("factor_no")
    )
    b_table = [
        "| 因子 | Rank IC | Rank ICIR | FDR q | 3bps年化 | 3bps夏普 | 3bps最大回撤 | 胜率 | 日均等效换手 | 5bps年化 | 2025 Rank IC | 2025 5bps年化 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in b_summary.itertuples(index=False):
        b_table.append(
            f"| `{row.factor_id}` | {fmt(row.rank_ic)} | {fmt(row.rank_ic_ir)} | "
            f"{fmt(row.rank_ic_fdr_q)} | {pct(row.annual_return_3bps)} | "
            f"{fmt(row.sharpe_3bps)} | {pct(row.max_drawdown_3bps)} | "
            f"{pct(row.win_rate_3bps)} | {pct(row.daily_equivalent_turnover)} | "
            f"{pct(row.annual_return_5bps)} | {fmt(row.sealed_rank_ic)} | "
            f"{pct(row.sealed_annual_return_5bps)} |"
        )

    unavailable = factor_registry.loc[~factor_registry["available"].astype(bool)]
    unavailable_table = [
        "| 因子 | 缺失数据/无法计算原因 |",
        "|---|---|",
    ]
    for row in unavailable.itertuples(index=False):
        unavailable_table.append(
            f"| `{row.factor_id}` | {row.unavailable_reason} |"
        )

    family = (
        factor_metrics.groupby("category", as_index=False)
        .agg(
            concepts=("factor_id", "count"),
            computable=("available", "sum"),
            mean_rank_ic=("rank_ic", "mean"),
            median_rank_ic=("rank_ic", "median"),
            mean_5bps_sharpe=("sharpe_5bps", "mean"),
            mean_coverage=("coverage", "mean"),
        )
        .sort_values("category")
    )
    family_table = markdown_table(
        family,
        [
            ("category", "家族"),
            ("concepts", "概念数"),
            ("computable", "可计算"),
            ("mean_rank_ic", "平均Rank IC"),
            ("median_rank_ic", "中位Rank IC"),
            ("mean_5bps_sharpe", "平均5bps夏普"),
            ("mean_coverage", "平均覆盖率"),
        ],
    )

    parameter_table = markdown_table(
        parameter_metrics[
            [
                "factor_family",
                "lookback_window",
                "rank_ic",
                "rank_ic_t_stat",
                "rank_ic_fdr_q",
                "coverage",
            ]
        ],
        [
            ("factor_family", "家族"),
            ("lookback_window", "窗口"),
            ("rank_ic", "Rank IC"),
            ("rank_ic_t_stat", "HAC t"),
            ("rank_ic_fdr_q", "FDR q"),
            ("coverage", "覆盖率"),
        ],
    )

    feature_type_counts = feature_registry["usage_type"].value_counts()
    missing_features = int(feature_registry["is_missing_indicator"].astype(bool).sum())
    primary_features = len(feature_registry) - missing_features
    maximum_cluster = int(feature_registry["correlation_cluster"].max())
    positive_incremental = int(
        incremental["mean_oos_delta_rank_ic"].gt(0).sum()
    )
    same_direction = factor_metrics[
        ["factor_id", "rank_ic"]
    ].merge(
        sealed_metrics[["factor_id", "rank_ic"]],
        on="factor_id",
        suffixes=("_dev", "_2025"),
    ).dropna()
    direction_rate = float(
        (
            np.sign(same_direction["rank_ic_dev"])
            == np.sign(same_direction["rank_ic_2025"])
        ).mean()
    )

    definition_table = [
        "| 编号 | 因子 | 家族 | 数学/程序定义 | 数据来源 | 可计算 | 分类 |",
        "|---:|---|---|---|---|---|---|",
    ]
    class_lookup = factor_metrics.set_index("factor_id")["screening_class"]
    for row in factor_registry.sort_values("factor_no").itertuples(index=False):
        formula = str(row.formula).replace("|", "\\|")
        definition_table.append(
            f"| {row.factor_no} | `{row.factor_id}` {row.factor_name} | "
            f"{row.category} | `{formula}` | {row.data_source} | "
            f"{'是' if row.available else '否'} | {class_lookup.at[row.factor_id]} |"
        )

    feature_table = [
        "| feature_name | source_factor | transformation | category | usage_type | missing_rate | cluster | representative |",
        "|---|---|---|---|---|---:|---:|---|",
    ]
    for row in feature_registry.itertuples(index=False):
        feature_table.append(
            f"| `{row.feature_name}` | `{row.source_factor}` | "
            f"{row.transformation} | {row.category} | {row.usage_type} | "
            f"{fmt(row.missing_rate)} | {row.correlation_cluster} | "
            f"`{row.cluster_representative}` |"
        )

    lines = [
        "# 中国商品期货 Alpha 因子科学筛选系统：完整实验报告",
        "",
        "> 报告性质：因子研究与机器学习输入接口实验，不是最终预测模型提交，也不是实盘收益承诺。",
        "",
        "## 0. 提交流程与差异声明",
        "",
        "### 0.1 硬性差异声明",
        "",
        "本次实验相对通用团队模板存在以下显式差异：",
        "",
        "1. **研究对象不同**：本提交负责因子筛选与输入端建设，不训练最终模型；模型架构、损失函数、Early Stopping 等字段不适用。",
        "2. **数据基座**：只读取 `data_final_raw`；不读取旧 `data_research/core30_v2`、`data3` 或前两轮测试产物。",
        "3. **主力规则修订**：按用户最终确认，不使用“T−1可见持仓量重新选择主力”；使用 TuShare T 日连续主力行情，并确定性映射至真实合约。",
        "4. **期限结构**：Carry、斜率和曲率使用主力及其后最近可交易合约/完整实际合约曲线，不能只用主力一个合约；单一主力无法定义期限结构。",
        "5. **样本划分**：采用 2015 起点固定的 Expanding-Window；2019—2024逐年开发期样本外评价，2025为一次封存测试，不采用单次静态切分。",
        "6. **标签**：T收盘形成因子，T+1开盘进入当时映射的固定真实合约，T+h+1开盘退出；主标签为5日Open-to-Open，不是Close-to-Close。",
        "7. **成本**：筛选主口径为单边5bps，同时报告0/3/5/10bps；未用不完整的逐品种真实手续费替代统一成本。",
        "8. **组合映射**：无3日EMA、无绝对信号阈值、无流动性底池剔除；固定为横截面前后20%、两侧各0.5总权重、5日非重叠调仓。",
        "9. **特征保留**：相关性高的字段只聚类和标注，不按全样本标签删除；所有47个通过数据质量硬门的预注册因子进入ML接口。",
        "10. **标准化**：多数品种特征使用同日横截面中位数/MAD稳健Z分数；没有统一采用普通均值/标准差Z-Score，也没有全局板块中性化。",
        "11. **基本面时点**：缺少日内发布时间时保守延迟一个交易日；现货陈旧超过7日、仓单超过5日、库存超过14日即作缺失。",
        "12. **净值口径**：筛选器只生成非重叠5日收益；随附“每日净值”按退出日确认该期收益，是会计阶梯净值，不是假装成逐日盯市净值。",
        "13. **封存边界修正**：本报告生成前发现原开发输出有2个调仓日的标签结束于2025；已修正为标签结束日必须位于评价窗口内并从头重算，开发期跨2025记录现为0。",
        "",
        "### 0.2 提交物",
        "",
        "- `factor_screening_experiment_report.md`：本报告。",
        "- `factor_backtest_metrics.csv`：每个可计算因子在开发期与2025、3/5bps下的标准化指标。",
        "- `factor_portfolio_period_returns.csv`：每个因子的5日非重叠组合收益及信号/入场/退出日期。",
        "- `factor_nav_accounting_daily.csv.gz`：按退出日确认收益的每日会计净值。",
        "- `ml_feature_registry.csv`：92个最终模型输入字段的唯一清单。",
        "- `feature_dictionary.md`：特征含义与变换说明。",
        "- `fold_definitions.csv`：扩展窗口定义。",
        "- `SHA256SUMS` 与 `submission_manifest.json`：文件完整性和数据血缘。",
        "",
        "## 1. 基础信息",
        "",
        "- 策略/模型名称：中国商品期货 Alpha 因子科学筛选系统（data_final_raw统一版）",
        "- 开发负责人：`[待填写]`",
        "- 提交日期：2026-07-23",
        f"- 项目版本：`{manifest['project_version']}`",
        f"- 原始合约文件SHA256：`{manifest['data_sha256']}`",
        "- 研究目的：将57个概念因子进行点时安全计算、统计筛选、经济分类和ML输入加工。",
        "",
        "## 2. 数据基座与数据审计",
        "",
        "### 2.1 数据范围",
        "",
        "- 品种数：30；板块：能源、金属、化工、农产品。",
        "- 完整数据范围：2015-01-05至2025-12-31。",
        "- 开发特征区：2015-01-05至2024-12-31。",
        "- 开发样本外筛选：2019-01-01至2024-12-31，且标签结束日不晚于2024-12-31。",
        "- 封存测试：2025-01-01至2025-12-31，且标签结束日不晚于2025-12-31。",
        "",
        "### 2.2 审计结果",
        "",
        "| 审计项 | 结果 |",
        "|---|---:|",
        "| 原始实际合约日记录 | 846,963 |",
        "| 实际合约代码 | 3,753 |",
        "| `trade_date × product`面板行 | 78,112 |",
        "| 交易日 | 2,674 |",
        "| 重复主键 | 0 |",
        "| 开发期行数 | 70,822 |",
        "| 2025封存行数 | 7,290 |",
        "| 实际合约映射覆盖 | 100.00% |",
        "| 完全字段匹配率 | 94.99% |",
        "| 高质量或完全匹配率 | 99.00% |",
        "| 产品收益覆盖 | 99.72% |",
        "| 至少2点曲线覆盖 | 98.37% |",
        "| 至少3点曲线覆盖 | 82.33% |",
        "| 新鲜现货覆盖 | 73.41% |",
        "| 新鲜仓单覆盖 | 98.74% |",
        "| 新鲜库存覆盖 | 33.95% |",
        "| 库存覆盖品种 | 10 |",
        "| 1/5/20日标签覆盖 | 98.66% / 98.46% / 92.86% |",
        "| 点时违规 | 0 |",
        "| 开发标签跨入2025（修正后） | 0 |",
        "",
        "### 2.3 主力映射",
        "",
        "1. 读取30品种的TuShare精确连续主力序列。",
        "2. 用T日`pre_close/pre_settle/open/high/low/close/settle`与实际合约逐字段匹配。",
        "3. 并列时依次使用上日实际合约连续性、持仓量、成交量、较晚到期日打破平局。",
        "4. 禁止在无明确证据时倒退到更早到期月份。",
        "5. 得到的`mapped_actual_ts_code`既用于收益标签，也用于后续可成交性检查。",
        "",
        "这项修订不是“用当日持仓量事后选主力”：连续主力代码由数据源给定，持仓量只在行情字段无法唯一映射时作确定性并列处理。",
        "",
        "### 2.4 期限结构",
        "",
        "- 候选实际合约剩余期限限制为10—365日。",
        "- 合约价格必须为正，且持仓占比或成交占比至少1%。",
        "- Carry使用最近两个合格期限点并按到期差年化。",
        "- 曲率使用前三个期限点，相对首尾线性插值计算中间偏离，处理不等距到期。",
        "- 斜率在至少3点时，对期限—对数价格做以`sqrt(OI)`为权重的线性拟合。",
        "",
        "因此期限结构因子必须同时使用主力和后续合约；只保留主力会直接导致Carry、斜率、曲率无法定义。",
        "",
        "### 2.5 基本面点时处理",
        "",
        "- 所有现货、仓单、库存先按观察日期对齐，再整体延迟1个交易日。",
        "- 只有`observation_date < trade_date`时才允许进入因子。",
        "- 现货/仓单/库存最大陈旧期分别为7/5/14个自然日。",
        "- FU、I、SC因现货单位或汇率口径不可直接同比，排除基差计算。",
        "- 不把周/月数据向前填充后的每一天当作新信息事件；Surprise只在观察日期改变时产生。",
        "",
        "### 2.6 数据限制",
        "",
        "- 现货源止于2023-12-29，因此2024—2025基差相关特征按规则缺失，不能人为延长。",
        "- 只有OHLC锁价代理，没有交易所逐日精确涨跌停表。",
        "- 缺少盘口、买卖价差、逐笔冲击和容量数据，无法声称已核验真实实盘冲击。",
        "- 实际手续费任一字段覆盖约73.16%，且2021年前较弱；统一bps只用于横向研究比较。",
        "- 库存仅覆盖10个品种，库存类结果存在品种集中风险。",
        "",
        "## 3. 数据预处理与标签",
        "",
        "### 3.1 原始数据清洗",
        "",
        "- 主键统一为`trade_date × product`，日期升序、品种内排序。",
        "- 正价格、正成交量和合约存在性作为收益与曲线计算硬门。",
        "- 不使用后向填充（bfill），不把未来观测填回过去。",
        "- 原始因子计算阶段缺失保持NaN；只有进入ML加工阶段才变为中性0并附缺失标记。",
        "- Inf由非法分母防护产生前即被置为缺失，不用任意大数替换。",
        "",
        "### 3.2 标签定义",
        "",
        "对信号日T映射的实际合约c，h日标签为：",
        "",
        "$$y^{(h)}_{i,T}=\\frac{Open_{c,T+h+1}}{Open_{c,T+1}}-1,\\quad h\\in\\{1,5,20\\}.$$",
        "",
        "- T收盘后形成信号；T+1开盘才允许交易。",
        "- 持有期间固定同一真实合约，不随主力切换偷换持仓。",
        "- 入场或退出开盘无价格、无成交量或疑似锁死时，标签置缺失。",
        "- 主筛选标签为5日；1日、20日只作期限稳健性。",
        "",
        "### 3.3 封存边界",
        "",
        "任一评价窗口必须同时满足：",
        "",
        "$$valid\\_start\\le T\\le valid\\_end,\\qquad label\\_end\\_date_h\\le valid\\_end.$$",
        "",
        "训练折还必须满足`label_end_date_5d < valid_start`，形成5日purge。这样既防训练标签越过验证起点，也防开发期标签进入2025封存区。",
        "",
        "## 4. 因子计算全流程",
        "",
        "### 4.1 流程总览",
        "",
        "```text",
        "data_final_raw实际合约 + TuShare连续主力",
        "  → 主力连续行情映射至真实合约",
        "  → 完整期限结构与点时基本面",
        "  → 1/5/20日固定合约标签",
        "  → 57个预注册原始因子",
        "  → 数据质量硬门（覆盖、方差、事件数、品种数）",
        "  → 47个可计算因子",
        "  → IC/Rank IC/ICIR/HAC/FDR/分组回测",
        "  → 年份、品种、板块、期限、延迟、窗口稳健性",
        "  → 相关聚类与固定Ridge增量诊断",
        "  → A—F研究分类",
        "  → 点时安全特征加工",
        "  → 92字段ML接口",
        "```",
        "",
        "### 4.2 主要因子构造",
        "",
        "- 趋势：20/60/120日对数累计收益、波动率调整、路径效率及多周期一致性。",
        "- Carry：最近两个合格真实合约的对数价差除以年化到期差。",
        "- 基差：新鲜且单位可比的现货相对主力期货价格差。",
        "- 价值：当前对数价格相对仅使用历史的252日均值/波动的反向偏离。",
        "- 库存/仓单：按品种与ISO周进行仅用历史同期观测的扩展季节Z分数。",
        "- 新息：当前事件值相对该品种前60次历史条件分布的标准化异常，只在观察日变化时记值。",
        "- 流动性：20日Amihud、成交/持仓换手、成交量与持仓异常。",
        "- 风险：20日实现波动、60日偏度、剔除市场和板块共同收益后的60日特质波动。",
        "- 交互：仅保留预注册的动量×Carry、非线性紧张度、信息一致/背离等有经济含义组合。",
        "- 元因子：四个冻结基础因子的已实现收益合成后取20日持续性；只作时间条件变量。",
        "",
        "所有滚动均值、波动、条件期望和标准化基准均使用T时点以前的历史；当公式要求当日横截面时，只使用同日各品种在T收盘已知的数据。",
        "",
        "### 4.3 数据质量硬门",
        "",
        "- 稠密因子开发覆盖率至少20%。",
        "- 事件因子至少100个有效事件、至少5个品种。",
        "- 因子必须具有非零横截面/时间变化。",
        "- 未通过者不伪造数值，保留概念登记并标记F类。",
        "",
        *unavailable_table,
        "",
        "## 5. 扩展窗口设计",
        "",
        "| 折 | 训练区间 | 验证区间 | Purge |",
        "|---|---|---|---:|",
        "| 2019 | 2015-01-05—2018-12-31 | 2019全年 | 5日 |",
        "| 2020 | 2015-01-05—2019-12-31 | 2020全年 | 5日 |",
        "| 2021 | 2015-01-05—2020-12-31 | 2021全年 | 5日 |",
        "| 2022 | 2015-01-05—2021-12-31 | 2022全年 | 5日 |",
        "| 2023 | 2015-01-05—2022-12-31 | 2023全年 | 5日 |",
        "| 2024 | 2015-01-05—2023-12-31 | 2024全年 | 5日 |",
        "| 封存2025 | 2015-01-05—2024-12-31 | 2025全年 | 5日 |",
        "",
        "扩展窗口每轮扩大训练集而不丢弃早期数据，适合样本有限的商品期货；每个验证年都只使用此前可见数据。单因子IC主表把2019—2024各验证日期汇总，年度表保留逐年分解；固定Ridge增量测试在每一折重新拟合。",
        "",
        "## 6. 因子预处理与机器学习输入加工",
        "",
        "### 6.1 连续因子",
        "",
        "同日横截面稳健标准化：",
        "",
        "$$z_{i,t}=clip\\left(\\frac{x_{i,t}-Median_t(x)}{1.4826\\,MAD_t(x)},-5,5\\right).$$",
        "",
        "- 当MAD为0时回退同日横截面标准差。",
        "- 缺失值用同日中位数对应的中性值0表示。",
        "- 这种处理降低极端行情和不同量纲对模型的支配。",
        "",
        "### 6.2 时间序列变量",
        "",
        "`factor_momentum`没有同日品种差异，使用仅含过去日期的252日滚动Z分数，最少60日，截断±5；不能用普通横截面IC解释。",
        "",
        "### 6.3 缺失信息",
        "",
        "- 原始缺失率达到1%的因子增加`__missing`二元字段。",
        "- 92个特征由47个数值特征和45个缺失/质量标志组成。",
        "- 缺失标志不是Alpha，而是让模型识别“数据不可用”与“因子中性”不同。",
        "",
        "### 6.4 相关性、残差化和正交化",
        "",
        "- 在开发期计算特征相关矩阵，绝对相关系数≥0.90的字段用并查集聚为同一簇。",
        f"- 当前92字段形成{maximum_cluster}个相关簇；每簇登记代表字段，但所有字段保留。",
        "- 不在全样本上按标签挑选簇内赢家，避免选择偏差。",
        "- 不做全局目标驱动正交化；只允许预注册的板块相对价值、特质波动等经济残差。",
        "- 普通OLS不适合直接使用全部相关字段；队友若用线性模型应采用Ridge/Elastic Net，或只在每个训练折内选簇代表。",
        "",
        "### 6.5 最终接口",
        "",
        f"- 输入字段：{len(feature_registry)}；数值/主特征{primary_features}，缺失标志{missing_features}。",
        f"- 用途分布：predictor={feature_type_counts.get('predictor', 0)}，conditional={feature_type_counts.get('conditional', 0)}，risk={feature_type_counts.get('risk', 0)}，quality_flag={feature_type_counts.get('quality_flag', 0)}。",
        "- 唯一取列规则：读取`ml_feature_registry.csv`的全部`feature_name`。",
        "- 不按A—F分类删列；A—F是证据等级，不是当前全量接口的硬特征选择。",
        "- 训练数据位于项目`training_handoff/ml_dataset_development.csv.gz`，队友仅训练模型时不需要原始数据。",
        "",
        "## 7. 科学筛选统计",
        "",
        "### 7.1 IC、Rank IC与ICIR",
        "",
        "- 每个交易日在至少8个有效品种上计算横截面Pearson IC和Spearman Rank IC。",
        "- ICIR与Rank ICIR均为“逐日均值/逐日标准差”的未年化口径，不能等同策略夏普。",
        "- Rank IC胜率为`RankIC_t > 0`的日期比例。",
        "- 5日标签重叠引起自相关，均值检验使用最多5阶Newey–West/HAC标准误。",
        "",
        "### 7.2 多重检验",
        "",
        "- 对所有可评价因子的Rank IC p值统一做Benjamini–Hochberg FDR修正。",
        "- 多个FDR q相同是BH单调化的正常结果：排序后的`m·p(i)/i`从尾部取累计最小值，邻近检验可映射到同一q值。",
        "- 因此不能因为多个q相同而认定程序复制了结果。",
        "",
        "### 7.3 分组与回测",
        "",
        "每个第5个交易日：",
        "",
        "1. 按预注册方向调整因子正负。",
        "2. 有效截面至少8个品种。",
        "3. 做多最高20%、做空最低20%；两侧总绝对权重各0.5，组内等权。",
        "4. 使用固定实际合约T+1开盘至T+6开盘标签。",
        "5. 换手为`sum(abs(w_t-w_{t-1}))`。",
        "6. 净收益为`gross - turnover × cost_bps / 10000`。",
        "7. 每年期数按`252/5`年化；净值用复利，最大回撤由累计净值峰谷计算。",
        "",
        "这是一套公平比较单因子的统一映射，不为每个因子单独调阈值、持有期或成本。",
        "",
        "### 7.4 稳健性",
        "",
        "- 期限：1/5/20日Rank IC。",
        "- 延迟：信号再延迟1个交易日。",
        "- 时间：2019—2024逐年Rank IC、正向年份率及年度波动。",
        "- 横截面：30品种时间序列相关、四板块分解。",
        "- 参数：趋势40/60/80日；Carry变化、基差动量、反转、波动、持仓增长、曲线动量10/20/40日。",
        "- 成本：0/3/5/10bps。",
        "- 相关性：高相关簇、最大绝对相关、代表字段。",
        "",
        "参数邻域结果如下（诊断，不用于事后挑最佳窗口）：",
        "",
        *parameter_table,
        "",
        "### 7.5 增量价值",
        "",
        "- 固定基线：TSMOM、Carry、Amihud、Turnover。",
        "- 模型：Ridge，alpha=10，不调参。",
        "- 每折比较加入一个候选前后OOS MSE和日度Rank IC。",
        "- 训练样本严格满足标签结束日早于验证起点；验证标签结束日在验证年内。",
        f"- 共评价{len(incremental)}个预测候选，其中{positive_incremental}个平均增量Rank IC为正。",
        "- 结果仅判断互补性，不反向改变特征注册表。",
        "",
        "## 8. A—F分类依据",
        "",
        "分类顺序与门槛：",
        "",
        "1. 不可计算 → F。",
        "2. 预注册用途为risk → D。",
        "3. 预注册用途为conditional或quality flag → C。",
        "4. 预测因子证据门：Rank IC≥0.02。",
        "5. 经济门：单边5bps后年化收益>0。",
        "6. 显著性门：全库FDR q≤0.10。",
        "7. core角色同时通过4—6 → A。",
        "8. 通过经济门，且通过证据门或显著性门 → B。",
        "9. 其余可计算预测因子 → E。",
        "",
        "该分类是保守的研究证据分级。年份、品种、板块、延迟和参数结果在报告中作为稳健性诊断，但当前代码没有把它们全部写成A类硬门；这点必须透明披露。",
        "",
        "## 9. 筛选结论",
        "",
        f"- A={class_counts.get('A', 0)}、B={class_counts.get('B', 0)}、C={class_counts.get('C', 0)}、D={class_counts.get('D', 0)}、E={class_counts.get('E', 0)}、F={class_counts.get('F', 0)}。",
        "- 没有A类不是“因子全部无效”，而是没有因子同时通过10%全库FDR、Rank IC和5bps经济门。",
        "- B类代表辅助候选，不等于已验证商业Alpha。",
        f"- 开发期与2025可评价Rank IC同方向比例为{pct(direction_rate)}；跨期稳定性仍弱。",
        "- 47个可计算因子全部进入ML接口，是预注册+数据质量策略；模型必须用正则化与严格滚动验证控制弱信号和冗余。",
        "",
        "### 9.1 因子家族",
        "",
        *family_table,
        "",
        "### 9.2 B类候选及核心OOS指标",
        "",
        "本项目没有预注册的“B类等权总策略”，因此不存在一个诚实的单一年化收益/回撤/夏普。下表逐因子报告团队3bps口径与项目5bps口径；禁止事后把表现最好的因子拼成总策略再宣称OOS。",
        "",
        *b_table,
        "",
        "### 9.3 经济解释",
        "",
        "- **Carry/期限结构斜率**：Backwardation可反映近端稀缺、便利收益和风险转移补偿；完整曲线比单点价差更稳定。",
        "- **基差**：新鲜现货强于期货说明近端供需偏紧；但现货在2023后缺失，不能做2025确认。",
        "- **仓单紧张度**：可交割库存的边际稀缺比总库存更贴近交割约束，覆盖也明显更高。",
        "- **库存变化/加速度**：去库本身及去库速度加快可能传递供需趋紧，但仅10个品种有库存，必须防止样本集中。",
        "- **信息背离**：供需紧张度相对价格趋势的差异可能代表信息未充分吸收；2025反向说明当前代理并非真正市场一致预期。",
        "- **库存—价格背离**：库存趋紧而价格未跟随可能形成后续修复；同样受库存覆盖低和2025反向制约。",
        "- **趋势家族整体偏弱**：公开中期趋势在本样本与当前open-to-open标签下未形成稳定横截面优势，不应仅凭传统声誉升级。",
        "- **风险/流动性变量**：其任务是解释条件风险、可交易性和尾部暴露，单独多空收益不是主要入选标准。",
        "",
        "## 10. 模型架构与训练细节",
        "",
        "- 最终模型/算法类型：不适用；本提交不训练最终模型。",
        "- 核心超参数：不适用。",
        "- 损失函数：不适用。",
        "- 防过拟合：通过预注册、扩展窗口、purge、FDR、参数邻域和不使用目标选特征实现。",
        "- 唯一模型使用：固定Ridge(alpha=10)仅作增量诊断，不产生本报告最终交易信号。",
        "",
        "## 11. 交易执行与净值口径",
        "",
        "- 信号平滑：无。",
        "- 入场：横截面前20%多、后20%空。",
        "- 平仓：固定持有5个交易日，在T+6开盘退出。",
        "- 底池过滤：只执行数据可用、价格/成交及疑似锁价可执行硬门；无额外流动性底20%剔除。",
        "- 成本：0/3/5/10bps单边；主分类5bps，团队比较表3bps。",
        "- 净值CSV按退出日确认整期5日收益；中间日收益填0以保持日历索引，不能用于日内风险或逐日VaR分析。",
        "",
        "## 12. 核心评价指标填写说明",
        "",
        "团队模板要求单一策略的年化、回撤、夏普、胜率和换手，但本实验产物是57个单因子诊断，不是一个最终组合：",
        "",
        "- 年化收益率：不适用（无预注册总策略）。",
        "- 最大回撤：不适用（无预注册总策略）。",
        "- 夏普比率：不适用（无预注册总策略）。",
        "- 胜率：不适用（无预注册总策略）。",
        "- 日均换手率：不适用（无预注册总策略）。",
        "- 5日/20日Rank IC：逐因子列于`factor_backtest_metrics.csv`。",
        "- 多空分组单调性：逐因子列于同一CSV。",
        "",
        "如果团队必须比较一个模型策略，应由建模负责人基于冻结的92字段、相同折和相同执行协议生成，不能由因子筛选负责人事后选因子代替。",
        "",
        "## 13. 可复现性与运行方式",
        "",
        "```bash",
        "cd /Users/Wantree/Desktop/因子筛选系统/unified_factor_system",
        "PYTHONPATH=src python scripts/run_pipeline.py",
        "PYTHONPATH=src python scripts/build_icir_report.py",
        "PYTHONPATH=src python scripts/build_factor_analysis_report.py",
        "PYTHONPATH=src python scripts/build_factor_screening_submission.py",
        "pytest -q",
        "```",
        "",
        "主流水线拒绝覆盖已有产物；如需完整重跑，应先把旧生成目录移动到备份位置。不能删除原始数据，也不能用旧结果覆盖本次统一版。",
        "",
        "## 14. 最终判断",
        "",
        "本轮工作的核心成果不是发现一个可立即商业化的强Alpha，而是建立了一个可审计的因子研究基座：57个概念完整登记、47个点时安全可计算、92个字段可直接交接建模、10个数据缺口明确记录、开发与封存边界隔离。最有希望的方向集中在期限结构、基差、仓单和库存边际变化，但没有任何因子通过10%全库FDR，且2025方向一致率不足一半。因此正确结论是“形成高质量研究输入和若干辅助候选”，不是“已证明稳定盈利”。",
        "",
        "# 附录A：57个因子定义与数据来源",
        "",
        *definition_table,
        "",
        "# 附录B：92个机器学习输入字段",
        "",
        *feature_table,
        "",
        "# 附录C：逐因子完整证据与经济分析",
        "",
        analysis_tail,
        "",
    ]
    return "\n".join(lines)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    SUBMISSION.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(
        (ROOT / "outputs" / "RUN_MANIFEST.json").read_text(encoding="utf-8")
    )
    development_metrics = pd.read_csv(
        ROOT / "outputs" / "screening" / "factor_metrics_with_icir.csv"
    )
    sealed_metrics = pd.read_csv(
        ROOT / "sealed_evaluation" / "factor_metrics_2025.csv"
    )
    factor_registry = pd.read_csv(ROOT / "features" / "factor_registry.csv")
    feature_registry = pd.read_csv(
        ROOT / "features" / "ml_feature_registry.csv"
    )
    development_returns = pd.read_csv(
        ROOT / "outputs" / "screening" / "factor_portfolio_returns.csv",
        parse_dates=["trade_date"],
    )
    sealed_returns = pd.read_csv(
        ROOT / "sealed_evaluation" / "factor_portfolio_returns_2025.csv",
        parse_dates=["trade_date"],
    )
    labels = pd.read_parquet(ROOT / "outputs" / "intermediate" / "labels.parquet")
    horizons = pd.read_csv(
        ROOT / "outputs" / "screening" / "horizon_and_delay_metrics.csv"
    )
    parameter_metrics = pd.read_csv(
        ROOT / "outputs" / "screening" / "parameter_sensitivity_metrics.csv"
    )
    incremental = pd.read_csv(
        ROOT / "outputs" / "screening" / "incremental_ridge_tests.csv"
    )

    period_returns = build_period_returns(
        development_returns,
        sealed_returns,
        labels,
    )
    metric_export = build_metrics(
        period_returns,
        development_metrics,
        sealed_metrics,
        horizons,
    )
    daily_nav = build_daily_nav(period_returns, labels)

    period_returns.to_csv(
        SUBMISSION / "factor_portfolio_period_returns.csv",
        index=False,
        encoding="utf-8",
        date_format="%Y-%m-%d",
    )
    metric_export.to_csv(
        SUBMISSION / "factor_backtest_metrics.csv",
        index=False,
        encoding="utf-8",
        date_format="%Y-%m-%d",
    )
    daily_nav.to_csv(
        SUBMISSION / "factor_nav_accounting_daily.csv.gz",
        index=False,
        encoding="utf-8",
        compression="gzip",
        date_format="%Y-%m-%d",
    )
    shutil.copy2(
        ROOT / "features" / "ml_feature_registry.csv",
        SUBMISSION / "ml_feature_registry.csv",
    )
    shutil.copy2(
        ROOT / "features" / "feature_dictionary.md",
        SUBMISSION / "feature_dictionary.md",
    )
    shutil.copy2(
        ROOT / "training_handoff" / "fold_definitions.csv",
        SUBMISSION / "fold_definitions.csv",
    )

    analysis = (
        ROOT / "reports" / "factor_screening_report_analysis.md"
    ).read_text(encoding="utf-8")
    marker = "## 因子家族分析"
    analysis_tail = analysis[analysis.index(marker) :] if marker in analysis else analysis
    report = build_report(
        manifest,
        metric_export,
        development_metrics,
        sealed_metrics,
        feature_registry,
        factor_registry,
        parameter_metrics,
        incremental,
        analysis_tail,
    )
    (SUBMISSION / "factor_screening_experiment_report.md").write_text(
        report,
        encoding="utf-8",
    )
    readme = """# 因子筛选实验提交包

本目录是因子筛选负责人提交物，不是最终模型交接包。

- 先阅读 `factor_screening_experiment_report.md`。
- 因子表现见 `factor_backtest_metrics.csv`。
- 5日非重叠收益见 `factor_portfolio_period_returns.csv`。
- `factor_nav_accounting_daily.csv.gz` 是退出日确认收益的阶梯净值，不是逐日盯市净值。
- 模型字段严格取 `ml_feature_registry.csv` 的 `feature_name`。
- 完整可训练数据仍位于项目根目录的 `training_handoff/`。
"""
    (SUBMISSION / "README.md").write_text(readme, encoding="utf-8")

    files = sorted(
        path for path in SUBMISSION.iterdir()
        if path.is_file() and path.name not in {"SHA256SUMS", "submission_manifest.json"}
    )
    submission_manifest = {
        "project_version": manifest["project_version"],
        "source_run_data_sha256": manifest["data_sha256"],
        "report_date": "2026-07-23",
        "factor_concepts": int(len(factor_registry)),
        "computable_factors": int(factor_registry["available"].astype(bool).sum()),
        "ml_features": int(len(feature_registry)),
        "development_period_return_rows": int(
            period_returns["sample_segment"].eq("development_2019_2024").sum()
        ),
        "sealed_period_return_rows": int(
            period_returns["sample_segment"].eq("sealed_2025").sum()
        ),
        "daily_nav_rows": int(len(daily_nav)),
        "nav_method": "exit_date_recognition_non_overlapping_5d_not_daily_mark_to_market",
        "files": {path.name: sha256(path) for path in files},
    }
    (SUBMISSION / "submission_manifest.json").write_text(
        json.dumps(submission_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    checksum_files = sorted(
        path for path in SUBMISSION.iterdir()
        if path.is_file() and path.name != "SHA256SUMS"
    )
    (SUBMISSION / "SHA256SUMS").write_text(
        "\n".join(f"{sha256(path)}  {path.name}" for path in checksum_files) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {SUBMISSION}")
    print(
        f"factors={len(factor_registry)} computable={factor_registry['available'].sum()} "
        f"features={len(feature_registry)} nav_rows={len(daily_nav)}"
    )


if __name__ == "__main__":
    main()
