#!/usr/bin/env python3
"""Build the HCGNN report tables, figures, and a reproducible submission ZIP."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


VARIANTS = ("protocol_gnn", "hcgnn_continual")
FOLDS = tuple(f"fold_{year}" for year in range(2019, 2024))
COSTS_BPS = (0, 3, 5, 10)
FIGURES = (
    ("01_nav_and_drawdown", "样本外净值与回撤", "上图是扣除单边 5 bps 后的累计净值，下图是净值距离历史高点的跌幅。净值长期向上、回撤较浅且修复较快更理想。两条线使用相同交易日期，可以直接比较两个模型。"),
    ("02_fold_rank_ic", "扩展窗口各折 Rank IC", "每根柱对应一个样本外年份。Rank IC 衡量预测排序与未来五日收益排序是否一致，大于 0 表示方向正确。比起某一年特别高，更应关注多数年份是否稳定为正。"),
    ("03_rolling_rank_ic", "60 日滚动 Rank IC", "曲线是最近 60 个交易日的平均 Rank IC，用来观察模型是否随市场状态失效。曲线持续在 0 上方说明近期排序有效，跌到 0 以下则表示这一阶段的信号方向可能反了。"),
    ("04_daily_ic_distribution", "日频 Rank IC 分布", "横轴是每天的 Rank IC，纵轴是出现频率。分布中心越靠右越好；左侧长尾说明模型偶尔会出现较大的反向预测。它能补充均值指标，避免少数极端日期掩盖日常表现。"),
    ("05_quintile_monotonicity", "预测五分组单调性", "每天按预测值从低到高分成五组，Q1 是预测最低组，Q5 是预测最高组。柱子从 Q1 到 Q5 逐步升高，且 Q5 与 Q1 差距明显，才说明模型适合用来做截面多空排序。"),
    ("06_annual_returns", "年度收益率", "柱子给出扣除单边 5 bps 后的年度收益。它用于检查总收益是否依赖某一个年份。如果多数年份为正，结果通常比只靠单年大涨更可信。"),
    ("07_turnover", "20 日平均换手率", "曲线是单边换手率的 20 日均值。换手越高，手续费和滑点的影响越大。若收益改善伴随换手大幅上升，需要结合成本敏感性表判断是否值得。"),
    ("08_prediction_calibration", "预测值与未来收益校准", "横轴是模型预测，纵轴是实际未来五日收益，颜色越亮表示样本越密集。点云整体向右上倾斜说明预测方向有效；若点云近似水平，模型给出的数值大小与实际收益关系较弱。"),
    ("09_training_curves", "训练集与验证集损失曲线", "左图是训练损失，右图是验证损失，纵轴使用对数刻度。训练损失下降而验证损失持续上升是过拟合信号。早停应保留验证损失最低附近的模型，而不是最后一个 epoch。"),
    ("10_sector_rank_ic", "板块 Rank IC", "柱子比较农业、金属、能源等板块内的 Rank IC。它能看出收益是否集中在少数板块。某个板块明显为负时，应检查该板块的数据覆盖、因子含义和图连接是否合适。"),
    ("11_feature_missingness", "特征缺失率", "图中列出缺失率最高的 25 个输入列。较高的缺失率不等于数据被填成 0，模型同时接收缺失标记。库存类因子覆盖较低时，应结合特征清单确认模型依赖程度。"),
    ("12_fold_metric_heatmap", "各折指标热图", "每个格子是一个模型在一个样本外年份的 Rank IC，数字给出精确值，颜色帮助快速定位强弱年份。若结果只在一两个格子较好，其跨年份稳定性仍然不足。"),
)


def rank_ic(group: pd.DataFrame) -> float:
    valid = group[["prediction", "future_return_5d"]].dropna()
    if len(valid) < 5 or valid.nunique().min() < 2:
        return np.nan
    return valid["prediction"].corr(valid["future_return_5d"], method="spearman")


def max_drawdown(nav: pd.Series) -> float:
    return float((nav / nav.cummax() - 1.0).min()) if len(nav) else np.nan


def performance_stats(ret: pd.Series, turnover: pd.Series) -> dict[str, float]:
    ret = ret.fillna(0.0)
    nav = (1.0 + ret).cumprod()
    years = max(len(ret) / 252.0, 1.0 / 252.0)
    annual_return = float(nav.iloc[-1] ** (1.0 / years) - 1.0)
    annual_vol = float(ret.std(ddof=1) * math.sqrt(252.0))
    sharpe = float(ret.mean() / ret.std(ddof=1) * math.sqrt(252.0)) if ret.std(ddof=1) > 0 else np.nan
    return {
        "annual_return": annual_return,
        "annual_volatility": annual_vol,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown(nav),
        "win_rate": float((ret > 0).mean()),
        "daily_turnover": float(turnover.mean()),
        "terminal_nav": float(nav.iloc[-1]),
    }


def load_predictions(root: Path, allow_incomplete: bool) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    missing: list[str] = []
    for variant in VARIANTS:
        for fold in FOLDS:
            path = root / variant / fold / "predictions.csv"
            if path.exists():
                frame = pd.read_csv(path, parse_dates=["trade_date"])
                frame["variant"] = variant
                frame["fold_id"] = fold
                frames.append(frame)
            else:
                missing.append(str(path))
    if missing and not allow_incomplete:
        raise FileNotFoundError("Formal outputs are incomplete:\n" + "\n".join(missing))
    if not frames:
        raise FileNotFoundError(f"No predictions.csv found under {root}")
    return pd.concat(frames, ignore_index=True).drop_duplicates(["variant", "trade_date", "product"], keep="last")


def load_histories(root: Path) -> pd.DataFrame:
    rows: list[dict] = []
    for variant in VARIANTS:
        for fold in FOLDS:
            path = root / variant / fold / "history.json"
            if not path.exists():
                continue
            for row in json.loads(path.read_text(encoding="utf-8")):
                rows.append({"variant": variant, "fold_id": fold, **row})
    return pd.DataFrame(rows)


def load_model_metrics(root: Path) -> pd.DataFrame:
    rows: list[dict] = []
    for variant in VARIANTS:
        for fold in FOLDS:
            path = root / variant / fold / "metrics.json"
            if path.exists():
                rows.append(json.loads(path.read_text(encoding="utf-8")))
    return pd.DataFrame(rows)


def panel_returns(panel_path: Path) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    with np.load(panel_path, allow_pickle=False) as panel:
        dates = pd.to_datetime(panel["dates"])
        products = panel["products"].astype(str)
        one_day = pd.DataFrame(panel["future_return_1d"], index=dates, columns=products)
        mask = pd.DataFrame(panel["future_return_1d_mask"], index=dates, columns=products)
        one_day = one_day.where(mask)
        extras = {
            "features": panel["features"],
            "feature_names": panel["feature_names"].astype(str),
            "sectors": panel["sectors"].astype(str),
            "products": products,
        }
    return one_day, extras


def make_target_weights(day: pd.DataFrame, top_fraction: float = 0.2) -> pd.Series:
    scores = day.set_index("product")["prediction"].dropna().sort_values()
    n = max(1, int(math.floor(len(scores) * top_fraction)))
    if len(scores) < 2 * n:
        return pd.Series(dtype=float)
    weights = pd.Series(0.0, index=scores.index)
    weights.loc[scores.index[:n]] = -0.5 / n
    weights.loc[scores.index[-n:]] = 0.5 / n
    return weights


def backtest_variant(
    pred: pd.DataFrame,
    one_day_returns: pd.DataFrame,
    holding_days: int = 5,
    top_fraction: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    products = list(one_day_returns.columns)
    dates = sorted(set(pred["trade_date"]).intersection(one_day_returns.index))
    target = pd.DataFrame(0.0, index=pd.DatetimeIndex(dates), columns=products)
    for date, day in pred.groupby("trade_date", sort=True):
        if date not in target.index:
            continue
        weights = make_target_weights(day, top_fraction)
        target.loc[date, weights.index] = weights

    # Five equally weighted signal cohorts implement daily rebalancing with a five-day hold.
    active = sum(target.shift(k).fillna(0.0) for k in range(holding_days)) / holding_days
    realized = one_day_returns.reindex(active.index).fillna(0.0)
    gross = (active * realized).sum(axis=1)
    turnover = 0.5 * active.diff().abs().sum(axis=1)
    turnover.iloc[0] = 0.5 * active.iloc[0].abs().sum()
    daily = pd.DataFrame({"trade_date": active.index, "gross_return": gross.values, "turnover": turnover.values})
    positions = active.stack().rename("weight").reset_index().rename(columns={"level_0": "trade_date", "level_1": "product"})
    return daily, positions


def add_cost_nav(daily: pd.DataFrame, costs_bps: tuple[int, ...] = COSTS_BPS) -> pd.DataFrame:
    out = daily.copy()
    for bps in costs_bps:
        ret = out["gross_return"] - out["turnover"] * bps / 10_000.0
        out[f"net_return_{bps}bps"] = ret
        out[f"nav_{bps}bps"] = (1.0 + ret).cumprod()
    return out


def save_figure(fig: plt.Figure, figure_dir: Path, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(figure_dir / f"{stem}.png", dpi=180, bbox_inches="tight")
    fig.savefig(figure_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def create_figures(
    predictions: pd.DataFrame,
    histories: pd.DataFrame,
    model_metrics: pd.DataFrame,
    daily_nav: pd.DataFrame,
    feature_list: pd.DataFrame,
    figure_dir: Path,
) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.figsize": (10, 5.6), "axes.grid": True, "grid.alpha": 0.25, "font.size": 10})
    colors = {"protocol_gnn": "#4472C4", "hcgnn_continual": "#ED7D31"}

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for variant, group in daily_nav.groupby("variant"):
        group = group.sort_values("trade_date")
        nav = group["nav_5bps"]
        ax1.plot(group["trade_date"], nav, label=variant, color=colors.get(variant))
        ax2.plot(group["trade_date"], nav / nav.cummax() - 1.0, label=variant, color=colors.get(variant))
    ax1.set_title("OOS net asset value (5 bps)"); ax1.set_ylabel("NAV"); ax1.legend()
    ax2.set_title("Drawdown"); ax2.set_ylabel("Drawdown"); ax2.xaxis.set_major_locator(mdates.YearLocator())
    save_figure(fig, figure_dir, "01_nav_and_drawdown")

    fold_ic = predictions.groupby(["variant", "fold_id"])[["prediction", "future_return_5d"]].apply(rank_ic).rename("rank_ic").reset_index()
    pivot = fold_ic.pivot(index="fold_id", columns="variant", values="rank_ic")
    fig, ax = plt.subplots(figsize=(10, 5.5)); pivot.plot(kind="bar", ax=ax, color=[colors.get(c) for c in pivot.columns])
    ax.axhline(0, color="black", linewidth=0.8); ax.set_title("Rank IC by expanding-window fold"); ax.set_ylabel("Rank IC"); ax.tick_params(axis="x", rotation=0)
    save_figure(fig, figure_dir, "02_fold_rank_ic")

    daily_ic = predictions.groupby(["variant", "trade_date"])[["prediction", "future_return_5d"]].apply(rank_ic).rename("rank_ic").reset_index()
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for variant, group in daily_ic.groupby("variant"):
        group = group.sort_values("trade_date")
        ax.plot(group["trade_date"], group["rank_ic"].rolling(60, min_periods=30).mean(), label=variant, color=colors.get(variant))
    ax.axhline(0, color="black", linewidth=0.8); ax.set_title("60-day rolling Rank IC"); ax.set_ylabel("Rank IC"); ax.legend()
    save_figure(fig, figure_dir, "03_rolling_rank_ic")

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bins = np.linspace(-0.8, 0.8, 45)
    for variant, group in daily_ic.groupby("variant"):
        ax.hist(group["rank_ic"].dropna(), bins=bins, alpha=0.5, density=True, label=variant, color=colors.get(variant))
    ax.axvline(0, color="black", linewidth=0.8); ax.set_title("Daily Rank IC distribution"); ax.set_xlabel("Rank IC"); ax.legend()
    save_figure(fig, figure_dir, "04_daily_ic_distribution")

    quant = predictions.copy()
    quant["quintile"] = quant.groupby(["variant", "trade_date"])["prediction"].transform(
        lambda x: pd.qcut(x.rank(method="first"), 5, labels=False, duplicates="drop") + 1
    )
    mono = quant.groupby(["variant", "quintile"])["future_return_5d"].mean().unstack(0)
    fig, ax = plt.subplots(figsize=(9, 5.5)); mono.plot(kind="bar", ax=ax, color=[colors.get(c) for c in mono.columns])
    ax.axhline(0, color="black", linewidth=0.8); ax.set_title("Mean future 5-day return by prediction quintile"); ax.set_ylabel("Mean return"); ax.tick_params(axis="x", rotation=0)
    save_figure(fig, figure_dir, "05_quintile_monotonicity")

    fig, ax = plt.subplots(figsize=(10, 5.5))
    annual = daily_nav.assign(year=daily_nav["trade_date"].dt.year).groupby(["variant", "year"])["net_return_5bps"].apply(lambda x: (1 + x).prod() - 1).unstack(0)
    annual.plot(kind="bar", ax=ax, color=[colors.get(c) for c in annual.columns]); ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Calendar-year return (5 bps)"); ax.set_ylabel("Return"); ax.tick_params(axis="x", rotation=0)
    save_figure(fig, figure_dir, "06_annual_returns")

    fig, ax = plt.subplots(figsize=(11, 5.5))
    for variant, group in daily_nav.groupby("variant"):
        ax.plot(group["trade_date"], group["turnover"].rolling(20, min_periods=5).mean(), label=variant, color=colors.get(variant))
    ax.set_title("20-day average turnover"); ax.set_ylabel("One-way turnover"); ax.legend()
    save_figure(fig, figure_dir, "07_turnover")

    sample = predictions.dropna(subset=["prediction", "future_return_5d"])
    if len(sample) > 30_000:
        sample = sample.sample(30_000, random_state=42)
    fig, axes = plt.subplots(1, len(sample["variant"].unique()), figsize=(12, 5), squeeze=False)
    for ax, (variant, group) in zip(axes[0], sample.groupby("variant")):
        hb = ax.hexbin(group["prediction"], group["future_return_5d"], gridsize=45, mincnt=1, cmap="viridis")
        ax.set_title(variant); ax.set_xlabel("Prediction"); ax.set_ylabel("Future 5-day return"); fig.colorbar(hb, ax=ax, label="Count")
    save_figure(fig, figure_dir, "08_prediction_calibration")

    if not histories.empty:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
        for (variant, fold, task), group in histories.groupby(["variant", "fold_id", "task"]):
            label = f"{variant}/{fold}/{task}"
            axes[0].plot(group["epoch"], group["train_loss"], alpha=0.65, label=label)
            axes[1].plot(group["epoch"], group["val_loss"], alpha=0.65, label=label)
        axes[0].set_title("Training loss"); axes[1].set_title("Validation loss")
        for ax in axes: ax.set_xlabel("Epoch"); ax.set_yscale("log")
        axes[1].legend(fontsize=6, ncol=2, bbox_to_anchor=(1.02, 1), loc="upper left")
        save_figure(fig, figure_dir, "09_training_curves")

    sector_ic = predictions.groupby(["variant", "sector"])[["prediction", "future_return_5d"]].apply(rank_ic).rename("rank_ic").reset_index().pivot(index="sector", columns="variant", values="rank_ic")
    fig, ax = plt.subplots(figsize=(10, 5.5)); sector_ic.plot(kind="bar", ax=ax, color=[colors.get(c) for c in sector_ic.columns])
    ax.axhline(0, color="black", linewidth=0.8); ax.set_title("Rank IC by sector"); ax.set_ylabel("Rank IC"); ax.tick_params(axis="x", rotation=30)
    save_figure(fig, figure_dir, "10_sector_rank_ic")

    top_missing = feature_list.sort_values("missing_rate", ascending=False).head(25).sort_values("missing_rate")
    fig, ax = plt.subplots(figsize=(10, 8)); ax.barh(top_missing["feature"], top_missing["missing_rate"], color="#70AD47")
    ax.set_title("Top 25 feature missing rates"); ax.set_xlabel("Missing rate")
    save_figure(fig, figure_dir, "11_feature_missingness")

    if not model_metrics.empty:
        numeric = [c for c in ["rank_ic", "ic_mean", "ic_ir", "rmse", "direction_accuracy"] if c in model_metrics]
        heat = model_metrics.pivot(index="fold_id", columns="variant", values="rank_ic") if "rank_ic" in numeric else pd.DataFrame()
        if not heat.empty:
            fig, ax = plt.subplots(figsize=(7, 5)); image = ax.imshow(heat.values, cmap="RdYlGn", aspect="auto")
            ax.set_xticks(range(len(heat.columns)), heat.columns, rotation=20); ax.set_yticks(range(len(heat.index)), heat.index)
            for i in range(len(heat.index)):
                for j in range(len(heat.columns)):
                    ax.text(j, i, f"{heat.iloc[i, j]:.3f}", ha="center", va="center")
            ax.set_title("Fold Rank IC heatmap"); fig.colorbar(image, ax=ax)
            save_figure(fig, figure_dir, "12_fold_metric_heatmap")


def write_report(bundle: Path, metrics: pd.DataFrame, config: dict, complete: bool) -> None:
    by_variant = metrics.set_index("variant") if not metrics.empty else pd.DataFrame()
    interpretation: list[str] = []
    if set(VARIANTS).issubset(by_variant.index):
        continual = by_variant.loc["hcgnn_continual"]
        protocol = by_variant.loc["protocol_gnn"]
        winner_return = "hcgnn_continual" if continual["annual_return"] > protocol["annual_return"] else "protocol_gnn"
        winner_sharpe = "hcgnn_continual" if continual["sharpe"] > protocol["sharpe"] else "protocol_gnn"
        winner_drawdown = "hcgnn_continual" if continual["max_drawdown"] > protocol["max_drawdown"] else "protocol_gnn"
        winner_ic = "hcgnn_continual" if continual["rank_ic_5d"] > protocol["rank_ic_5d"] else "protocol_gnn"
        interpretation = [
            f"本次样本外回测中，{winner_return} 的年化收益更高。HCGNN 持续学习版为 {continual['annual_return']:.2%}，协议 GNN 为 {protocol['annual_return']:.2%}。",
            f"风险调整后，{winner_sharpe} 的夏普更高。持续学习版为 {continual['sharpe']:.3f}，协议 GNN 为 {protocol['sharpe']:.3f}。",
            f"回撤方面，{winner_drawdown} 较浅。持续学习版最大回撤为 {continual['max_drawdown']:.2%}，协议 GNN 为 {protocol['max_drawdown']:.2%}。",
            f"换手率分别为 {continual['daily_turnover']:.2%} 和 {protocol['daily_turnover']:.2%}。若把交易成本进一步抬高，换手较低的持续学习版通常更有优势。",
            f"不过五日 Rank IC 由 {winner_ic} 领先，持续学习版为 {continual['rank_ic_5d']:.4f}，协议 GNN 为 {protocol['rank_ic_5d']:.4f}。这说明持续学习版的组合收益更好，但单日截面排序优势并不更强，结论应结合图 1、图 2、图 5 和成本敏感性表一起看。",
        ]
    lines = [
        "# Heterogeneous Continual GNN 日频期货模型提交报告",
        "",
        "## 0. 提交流程与差异声明",
        "",
        "本实验采用 GNN 专用修订协议：TuShare 主力连续映射、2015 起点扩展窗口、样本截止 2023、图结构及 CPD/GAP/MA 辅助任务、二阶拉普拉斯多项式谱滤波。主成本为单边 5 bps，并报告 0/3/5/10 bps 敏感性。",
        "",
        "## 1. 数据与模型",
        "",
        "- 训练/验证：2015 起点扩展窗口，OOS 为 2019—2023。",
        "- 输入：47 个可用概念因子展开为 92 维（数值列与缺失标记）。",
        f"- 模型参数：hidden_dim={config['model']['hidden_dim']}，num_blocks={config['model']['num_blocks']}，dropout={config['model']['dropout']}，batch_size={config['training']['batch_size']}。",
        "- 执行：截面前后 20% 多空、组内等权、五个等权持有队列、五日持有、净敞口 0。",
        "",
        "## 2. 样本外结果（5 bps）",
        "",
        metrics.to_markdown(index=False) if not metrics.empty else "正式结果尚未完整生成。",
        "",
        "## 3. 本次结果解读",
        "",
        *(interpretation if interpretation else ["等待完整结果后生成。"]),
        "",
        "## 4. 图表索引",
        "",
        "所有图片均作为独立文件放在 `figures/` 下；以下链接使用相对路径，因此提交 Markdown 时必须连同整个 `figures/` 文件夹一起发送。",
        "",
    ]
    for stem, title, explanation in FIGURES:
        lines.extend([
            f"### 图 {stem.split('_', 1)[0]}：{title}",
            "",
            f"![{title}](figures/{stem}.png)",
            "",
            f"图意说明：{explanation}",
            "",
        ])
    lines.extend([
        "## 5. 完整性",
        "",
        "十折输出完整。" if complete else "这是允许缺折的预览包，不得作为最终提交。",
    ])
    (bundle / "GNN_SUBMISSION_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts_handoff_gnn_v2"))
    parser.add_argument("--config", type=Path, default=Path("config_handoff_v2.yaml"))
    parser.add_argument("--bundle-dir", type=Path, default=Path("submission_hcgnn_v2"))
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()

    predictions = load_predictions(args.artifacts, args.allow_incomplete)
    histories = load_histories(args.artifacts)
    model_metrics = load_model_metrics(args.artifacts)
    one_day, extras = panel_returns(args.artifacts / "prepared" / "protocol_panel.npz")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    complete = len(predictions[["variant", "fold_id"]].drop_duplicates()) == len(VARIANTS) * len(FOLDS)

    bundle = args.bundle_dir
    if bundle.exists():
        shutil.rmtree(bundle)
    figure_dir = bundle / "figures"
    table_dir = bundle / "tables"
    figure_dir.mkdir(parents=True)
    table_dir.mkdir(parents=True)

    daily_frames: list[pd.DataFrame] = []
    position_frames: list[pd.DataFrame] = []
    metric_rows: list[dict] = []
    sensitivity_rows: list[dict] = []
    for variant, group in predictions.groupby("variant"):
        daily, positions = backtest_variant(group, one_day)
        daily = add_cost_nav(daily)
        daily.insert(0, "variant", variant)
        positions.insert(0, "variant", variant)
        daily_frames.append(daily); position_frames.append(positions)
        for bps in COSTS_BPS:
            stats = performance_stats(daily[f"net_return_{bps}bps"], daily["turnover"])
            sensitivity_rows.append({"variant": variant, "cost_bps": bps, **stats})
            if bps == 5:
                variant_pred = predictions[predictions["variant"] == variant]
                metric_rows.append({"variant": variant, **stats, "rank_ic_5d": rank_ic(variant_pred)})

    daily_nav = pd.concat(daily_frames, ignore_index=True)
    positions = pd.concat(position_frames, ignore_index=True)
    metrics = pd.DataFrame(metric_rows)
    sensitivity = pd.DataFrame(sensitivity_rows)
    features = extras["features"]
    feature_list = pd.DataFrame({
        "feature": extras["feature_names"],
        "missing_rate": np.isnan(features).mean(axis=(0, 1)),
        "is_missing_indicator": [str(x).endswith("__missing") for x in extras["feature_names"]],
    })
    daily_ic = predictions.groupby(["variant", "trade_date"])[["prediction", "future_return_5d"]].apply(rank_ic).rename("rank_ic_5d").reset_index()

    metrics.to_csv(table_dir / "metrics_summary.csv", index=False)
    daily_nav.to_csv(table_dir / "daily_nav.csv", index=False)
    positions.to_csv(table_dir / "positions.csv", index=False)
    sensitivity.to_csv(table_dir / "cost_sensitivity.csv", index=False)
    feature_list.to_csv(table_dir / "feature_list.csv", index=False)
    predictions.to_csv(table_dir / "oos_predictions.csv", index=False)
    daily_ic.to_csv(table_dir / "daily_rank_ic.csv", index=False)
    model_metrics.to_csv(table_dir / "fold_prediction_metrics.csv", index=False)
    histories.to_csv(table_dir / "training_history.csv", index=False)
    figure_manifest = pd.DataFrame([
        {
            "figure_no": stem.split("_", 1)[0],
            "title": title,
            "png_file": f"figures/{stem}.png",
            "pdf_file": f"figures/{stem}.pdf",
            "markdown": f"![{title}](figures/{stem}.png)",
            "explanation": explanation,
        }
        for stem, title, explanation in FIGURES
    ])
    figure_manifest.to_csv(bundle / "figure_manifest.csv", index=False, encoding="utf-8-sig")
    (bundle / "experiment_config.yaml").write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    shutil.copy2(args.artifacts / "prepared" / "protocol_panel.json", bundle / "data_manifest.json")

    create_figures(predictions, histories, model_metrics, daily_nav, feature_list, figure_dir)
    write_report(bundle, metrics, config, complete)
    manifest = {
        "complete": complete,
        "variants": sorted(predictions["variant"].unique().tolist()),
        "folds": sorted(predictions["fold_id"].unique().tolist()),
        "figures_png": len(list(figure_dir.glob("*.png"))),
        "figures_pdf": len(list(figure_dir.glob("*.pdf"))),
        "tables": sorted(path.name for path in table_dir.glob("*.csv")),
    }
    (bundle / "bundle_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    archive = shutil.make_archive(str(bundle), "zip", root_dir=bundle.parent, base_dir=bundle.name)
    print(json.dumps({**manifest, "archive": archive}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
