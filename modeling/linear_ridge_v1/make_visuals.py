"""Build figures for linear Ridge OOS results."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CONFIG = json.loads((ROOT / "modeling/linear_ridge_v1/config.json").read_text(encoding="utf-8"))
OUTPUT = ROOT / CONFIG["output_directory"]
FIG_DIR = OUTPUT / "figures"


def _style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#333333",
        "axes.labelcolor": "#222222",
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "text.color": "#222222",
        "font.size": 10,
        "axes.grid": True,
        "grid.color": "#dddddd",
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


USAGE_COLORS = {
    "predictor": "#1f4e79",
    "quality_flag": "#7a7a7a",
    "risk": "#8b4513",
    "conditional": "#2f6f4e",
}


def equity_and_drawdown(returns: pd.Series) -> tuple[pd.Series, pd.Series]:
    equity = (1.0 + returns.astype(float)).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    return equity, drawdown


def fig_01_equity_curves(daily: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(10, 5))
    series = {
        "Gross": daily["gross_return"],
        "Net 0bp": daily["net_0bps_return"],
        "Net 3bp": daily["net_3bps_return"],
        "Net 5bp (base)": daily["net_5bps_return"],
        "Net 10bp": daily["net_10bps_return"],
    }
    colors = ["#1f4e79", "#5b8fbf", "#2f6f4e", "#b85c38", "#7a7a7a"]
    dates = pd.to_datetime(daily["trade_date"])
    for (label, rets), color in zip(series.items(), colors):
        eq, _ = equity_and_drawdown(rets)
        lw = 2.2 if "5bp" in label else 1.4
        ax.plot(dates, eq.values, label=label, color=color, linewidth=lw)
    ax.axhline(1.0, color="#999999", linewidth=0.8, linestyle="--")
    ax.set_title("Linear Ridge — OOS Equity Curves (2019–2024)")
    ax.set_ylabel("Cumulative NAV (start = 1)")
    ax.set_xlabel("Rebalance date")
    ax.legend(frameon=False, ncol=3, loc="upper left")
    fig.tight_layout()
    out = FIG_DIR / "01_equity_curves.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def fig_02_drawdown(daily: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(10, 4.5))
    dates = pd.to_datetime(daily["trade_date"])
    for col, label, color in [
        ("gross_return", "Gross", "#1f4e79"),
        ("net_5bps_return", "Net 5bp (base)", "#b85c38"),
    ]:
        _, dd = equity_and_drawdown(daily[col])
        ax.fill_between(dates, dd.values, 0, alpha=0.25, color=color)
        ax.plot(dates, dd.values, color=color, linewidth=1.5, label=label)
    ax.set_title("Linear Ridge — Drawdown (Gross vs Net 5bp)")
    ax.set_ylabel("Drawdown")
    ax.set_xlabel("Rebalance date")
    ax.legend(frameon=False)
    fig.tight_layout()
    out = FIG_DIR / "02_drawdown_gross_vs_net5.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def fig_03_cost_sensitivity(summary: dict) -> Path:
    # Prefer recomputing from daily if available; fall back to report table via fold aggregate.
    daily = pd.read_csv(OUTPUT / "strategy_daily_returns.csv")
    labels = ["Gross / 0bp", "3bp", "5bp", "10bp"]
    cols = ["gross_return", "net_3bps_return", "net_5bps_return", "net_10bps_return"]
    ann = 50.4
    sharpes = []
    ann_rets = []
    for c in cols:
        s = daily[c].astype(float)
        vol = float(s.std(ddof=1))
        sharpes.append(float(s.mean() / vol * np.sqrt(ann)) if vol > 0 else np.nan)
        ann_rets.append(float(s.mean() * ann))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    x = np.arange(len(labels))
    colors = ["#1f4e79", "#2f6f4e", "#b85c38", "#7a7a7a"]
    axes[0].bar(x, [r * 100 for r in ann_rets], color=colors)
    axes[0].axhline(0, color="#666666", linewidth=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("Annual return (%)")
    axes[0].set_title("Cost sensitivity — Annual return")

    axes[1].bar(x, sharpes, color=colors)
    axes[1].axhline(0, color="#666666", linewidth=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("Sharpe")
    axes[1].set_title("Cost sensitivity — Sharpe")
    # annotate base cost
    axes[1].annotate(
        "protocol base",
        xy=(2, sharpes[2]),
        xytext=(2.35, sharpes[2] - 0.25),
        fontsize=8,
        color="#b85c38",
        arrowprops=dict(arrowstyle="->", color="#b85c38", lw=0.8),
    )
    fig.suptitle("Linear Ridge — One-way cost scenarios", y=1.02)
    fig.tight_layout()
    out = FIG_DIR / "03_cost_sensitivity.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_04_fold_rank_ic_sharpe(folds: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    years = folds["test_year"].astype(int)
    axes[0].bar(years, folds["rank_ic_mean"], color="#1f4e79", width=0.7)
    axes[0].axhline(0, color="#666666", linewidth=0.8)
    axes[0].set_title("OOS RankIC by fold")
    axes[0].set_xlabel("Test year")
    axes[0].set_ylabel("Mean RankIC")

    axes[1].bar(years - 0.2, folds["gross_sharpe"], width=0.4, label="Gross", color="#1f4e79")
    axes[1].bar(years + 0.2, folds["net_5bps_sharpe"], width=0.4, label="Net 5bp", color="#b85c38")
    axes[1].axhline(0, color="#666666", linewidth=0.8)
    axes[1].set_title("OOS Sharpe by fold")
    axes[1].set_xlabel("Test year")
    axes[1].set_ylabel("Sharpe")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    out = FIG_DIR / "04_fold_rankic_and_sharpe.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def fig_05_rolling_rank_ic(ic: pd.DataFrame) -> Path:
    ic = ic.copy()
    ic["trade_date"] = pd.to_datetime(ic["trade_date"])
    ic = ic.sort_values("trade_date")
    roll = ic["rank_ic"].rolling(60, min_periods=20).mean()

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(ic["trade_date"], ic["rank_ic"], color="#b0b8c1", linewidth=0.6, alpha=0.7, label="Daily RankIC")
    ax.plot(ic["trade_date"], roll, color="#1f4e79", linewidth=2.0, label="60-day rolling mean")
    ax.axhline(0, color="#666666", linewidth=0.8, linestyle="--")
    ax.axhline(float(ic["rank_ic"].mean()), color="#b85c38", linewidth=1.2, linestyle=":", label=f"OOS mean = {ic['rank_ic'].mean():.4f}")
    ax.set_title("Linear Ridge — Daily RankIC (OOS)")
    ax.set_ylabel("RankIC")
    ax.set_xlabel("Date")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    out = FIG_DIR / "05_daily_rank_ic.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def fig_06_selected_alpha(folds: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4))
    years = folds["test_year"].astype(int)
    alphas = folds["selected_alpha"].astype(float)
    ax.plot(years, alphas, marker="o", color="#1f4e79", linewidth=1.8)
    for x, y in zip(years, alphas):
        ax.annotate(f"{y:g}", (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)
    ax.set_yscale("log")
    ax.set_title("Selected Ridge α by outer fold (inner RankIC)")
    ax.set_xlabel("Test year")
    ax.set_ylabel("Selected α (log scale)")
    ax.set_xticks(list(years))
    fig.tight_layout()
    out = FIG_DIR / "06_selected_alpha_by_fold.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def fig_07_top_coefficients(coef: pd.DataFrame, top_n: int = 20) -> Path:
    top = coef.sort_values("mean_abs_coefficient", ascending=False).head(top_n).iloc[::-1].copy()
    colors = [USAGE_COLORS.get(u, "#444444") for u in top["usage_type"]]
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(top["feature_name"], top["coefficient_mean"], color=colors)
    ax.axvline(0, color="#666666", linewidth=0.8)
    ax.set_title(f"Linear Ridge — Top {top_n} mean coefficients (by |mean|)")
    ax.set_xlabel("Mean coefficient across folds")
    # legend
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=c, label=u)
        for u, c in USAGE_COLORS.items()
        if u in set(top["usage_type"])
    ]
    ax.legend(handles=handles, frameon=False, loc="lower right", title="usage_type")
    fig.tight_layout()
    out = FIG_DIR / "07_top20_coefficients.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def fig_08_alpha_search_heatmap(alpha_search: pd.DataFrame) -> Path:
    # expect columns: fold_id, alpha, mean_rank_ic (or similar)
    cols = {c.lower(): c for c in alpha_search.columns}
    fold_col = cols.get("fold_id") or "fold_id"
    alpha_col = cols.get("alpha") or "alpha"
    ic_col = None
    for cand in ["mean_rank_ic", "rank_ic_mean", "mean_rankic"]:
        if cand in cols:
            ic_col = cols[cand]
            break
    if ic_col is None:
        # try fuzzy
        for c in alpha_search.columns:
            if "rank" in c.lower() and "ic" in c.lower():
                ic_col = c
                break
    if ic_col is None:
        raise KeyError(f"Cannot find RankIC column in alpha_search: {list(alpha_search.columns)}")

    pivot = alpha_search.pivot_table(index=fold_col, columns=alpha_col, values=ic_col, aggfunc="mean")
    pivot = pivot.reindex(sorted(pivot.columns, key=float), axis=1)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdBu_r", vmin=-0.02, vmax=0.06)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{float(a):g}" for a in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(list(pivot.index))
    ax.set_xlabel("α")
    ax.set_ylabel("Outer fold")
    ax.set_title("Inner-validation RankIC by α (heatmap)")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=8, color="#111111")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Mean RankIC")
    fig.tight_layout()
    out = FIG_DIR / "08_alpha_search_heatmap.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def write_readme(paths: list[Path]) -> Path:
    text = """# 线性回归（Ridge）结果图说明

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
"""
    out = FIG_DIR / "FIGURES.md"
    out.write_text(text, encoding="utf-8")
    # also list generated files at end for audit
    listing = "\n## 本次生成文件\n\n" + "\n".join(f"- `{p.name}`" for p in paths) + "\n"
    out.write_text(text + listing, encoding="utf-8")
    return out


def main() -> None:
    _style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    daily = pd.read_csv(OUTPUT / "strategy_daily_returns.csv")
    folds = pd.read_csv(OUTPUT / "fold_metrics.csv")
    ic = pd.read_csv(OUTPUT / "daily_prediction_ic.csv")
    coef = pd.read_csv(OUTPUT / "coefficient_summary.csv")
    alpha_search = pd.read_csv(OUTPUT / "alpha_search.csv")
    summary = json.loads((OUTPUT / "run_summary.json").read_text(encoding="utf-8"))

    paths = [
        fig_01_equity_curves(daily),
        fig_02_drawdown(daily),
        fig_03_cost_sensitivity(summary),
        fig_04_fold_rank_ic_sharpe(folds),
        fig_05_rolling_rank_ic(ic),
        fig_06_selected_alpha(folds),
        fig_07_top_coefficients(coef),
        fig_08_alpha_search_heatmap(alpha_search),
    ]
    readme = write_readme(paths)
    print("Wrote figures to", FIG_DIR)
    for p in paths:
        print(" -", p.relative_to(ROOT))
    print(" -", readme.relative_to(ROOT))


if __name__ == "__main__":
    main()
