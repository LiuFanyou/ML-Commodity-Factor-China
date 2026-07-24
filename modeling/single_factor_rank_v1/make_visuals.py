"""Build presentation tables and figures for the 92-field single-factor results."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CONFIG = json.loads((ROOT / "modeling/single_factor_rank_v1/config.json").read_text(encoding="utf-8"))
OUTPUT = ROOT / CONFIG["output_directory"]
FIG_DIR = ROOT / CONFIG.get("figures_directory", OUTPUT / "figures")
TABLE_DIR = ROOT / CONFIG.get("tables_directory", OUTPUT / "tables")


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


def _predictor_universe(lb: pd.DataFrame) -> pd.DataFrame:
    """Economic shortlist universe: predictors only, no missing indicators."""
    out = lb.loc[lb["usage_type"].eq("predictor")].copy()
    out = out.loc[~out["feature_name"].astype(str).str.endswith("__missing")].copy()
    return out


def write_shortlists(lb: pd.DataFrame, top_n: int = 10) -> list[str]:
    """Build compact contrast shortlists under different ranking keys."""
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    pred = _predictor_universe(lb)
    cols = [
        c
        for c in [
            "shortlist_rank",
            "rank",
            "feature_name",
            "source_factor",
            "category",
            "rank_ic_mean",
            "rank_ic_hit_rate",
            "rank_icir",
            "gross_annual_return",
            "gross_sharpe",
            "net_0bps_sharpe",
            "net_3bps_sharpe",
            "net_5bps_sharpe",
            "net_10bps_sharpe",
            "years_evaluated",
            "years_rank_ic_positive",
            "years_gross_sharpe_positive",
        ]
        if c == "shortlist_rank" or c in pred.columns
    ]
    written: list[str] = []

    def save_shortlist(name: str, frame: pd.DataFrame, sort_cols: list[str]) -> pd.DataFrame:
        ranked = frame.sort_values(sort_cols, ascending=[False] * len(sort_cols), na_position="last").head(top_n).copy()
        ranked.insert(0, "shortlist_rank", range(1, len(ranked) + 1))
        keep = [c for c in cols if c in ranked.columns]
        out = TABLE_DIR / name
        ranked[keep].to_csv(out, index=False)
        written.append(str(out.relative_to(ROOT)))
        return ranked

    a = save_shortlist(
        "06_shortlist_top10_by_rank_ic.csv",
        pred,
        ["rank_ic_mean", "rank_ic_hit_rate", "gross_sharpe"],
    )
    b = save_shortlist(
        "07_shortlist_top10_by_net5bps_sharpe.csv",
        pred,
        ["net_5bps_sharpe", "rank_ic_mean", "rank_ic_hit_rate"],
    )
    robust_pool = pred.loc[pred["rank_ic_mean"].gt(0)].copy()
    c = save_shortlist(
        "08_shortlist_top10_by_stability.csv",
        robust_pool,
        ["years_rank_ic_positive", "rank_ic_hit_rate", "rank_ic_mean"],
    )

    set_a = set(a["feature_name"])
    set_b = set(b["feature_name"])
    set_c = set(c["feature_name"])
    overlap_rows = []
    for name in sorted(set_a | set_b | set_c):
        overlap_rows.append({
            "feature_name": name,
            "in_rank_ic_top10": name in set_a,
            "in_net5_top10": name in set_b,
            "in_stability_top10": name in set_c,
            "n_shortlists": int(name in set_a) + int(name in set_b) + int(name in set_c),
        })
    overlap = pd.DataFrame(overlap_rows).sort_values(
        ["n_shortlists", "feature_name"], ascending=[False, True]
    )
    overlap_path = TABLE_DIR / "09_shortlist_overlap.csv"
    overlap.to_csv(overlap_path, index=False)
    written.append(str(overlap_path.relative_to(ROOT)))

    def _fmt(v: float, digits: int = 4) -> str:
        if pd.isna(v):
            return "—"
        return f"{float(v):.{digits}f}"

    def _md_table(frame: pd.DataFrame, title: str, rule: str) -> list[str]:
        lines = [
            f"## {title}",
            "",
            f"规则：{rule}",
            "",
            "| # | overall_rank | feature | RankIC | hit | gross Sharpe | net5 Sharpe | +IC years |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for _, row in frame.iterrows():
            lines.append(
                f"| {int(row['shortlist_rank'])} | {int(row['rank'])} | `{row['feature_name']}` | "
                f"{_fmt(row['rank_ic_mean'])} | {_fmt(row['rank_ic_hit_rate'], 3)} | "
                f"{_fmt(row['gross_sharpe'], 3)} | {_fmt(row['net_5bps_sharpe'], 3)} | "
                f"{int(row['years_rank_ic_positive'])}/{int(row['years_evaluated'])} |"
            )
        lines.append("")
        return lines

    md: list[str] = [
        "# Predictor 对照短表（Top10 × 多指标）",
        "",
        "- 宇宙：`usage_type=predictor` 且排除 `__missing`",
        f"- 每张短表取前 {top_n} 名",
        "- 全量 92 字段仍以 `01_`–`05_` 表为准；本文件只服务正文/汇报可读性",
        "",
        "## 怎么读",
        "",
        "- **A∩B**：预测力与 5bp 成本后都靠前 → 优先讨论",
        "- **只在 A**：RankIC 好但成本后一般 → 关注换手/持有期",
        "- **只在 B**：成本后靠前但 RankIC 未必最高 → 谨慎解读",
        "- **C**：在 RankIC>0 子集里按「正 IC 年数 / 胜率」看稳定性",
        "",
    ]
    md += _md_table(a, "A. Top10 by RankIC（主短表）", "rank_ic_mean ↓, hit ↓, gross Sharpe ↓")
    md += _md_table(b, "B. Top10 by net 5bp Sharpe（成本对照）", "net_5bps_sharpe ↓, rank_ic_mean ↓")
    md += _md_table(
        c,
        "C. Top10 by stability（可选）",
        "universe: RankIC>0; sort by years_rank_ic_positive ↓, hit ↓, RankIC ↓",
    )

    both = sorted(set_a & set_b)
    only_a = sorted(set_a - set_b)
    only_b = sorted(set_b - set_a)
    md += [
        "## 交集与错位（A vs B）",
        "",
        f"- A∩B（{len(both)}）：" + (", ".join(f"`{x}`" for x in both) if both else "无"),
        f"- 只在 A（{len(only_a)}）：" + (", ".join(f"`{x}`" for x in only_a) if only_a else "无"),
        f"- 只在 B（{len(only_b)}）：" + (", ".join(f"`{x}`" for x in only_b) if only_b else "无"),
        "",
        "## 文件",
        "",
    ]
    for rel in written:
        md.append(f"- `{rel}`")
    md_path = TABLE_DIR / "SHORTLISTS.md"
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    written.append(str(md_path.relative_to(ROOT)))
    return written


def write_tables(lb: pd.DataFrame) -> list[str]:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    cols = [
        c for c in [
            "rank", "feature_name", "source_factor", "usage_type", "category",
            "rank_ic_mean", "rank_ic_hit_rate", "rank_icir",
            "gross_annual_return", "gross_sharpe", "gross_max_drawdown",
            "net_0bps_sharpe", "net_3bps_sharpe", "net_5bps_sharpe", "net_10bps_sharpe",
            "years_evaluated", "years_rank_ic_positive", "years_gross_sharpe_positive",
        ] if c in lb.columns
    ]
    written: list[str] = []

    def save(name: str, frame: pd.DataFrame) -> None:
        out = TABLE_DIR / name
        frame[cols].to_csv(out, index=False)
        written.append(str(out.relative_to(ROOT)))

    save("01_all_by_rank_ic.csv", lb.sort_values("rank"))
    save(
        "02_predictor_by_rank_ic.csv",
        lb.loc[lb["usage_type"].eq("predictor")].sort_values(
            ["rank_ic_mean", "rank_ic_hit_rate"], ascending=[False, False]
        ),
    )
    save(
        "03_all_by_gross_sharpe.csv",
        lb.sort_values(["gross_sharpe", "rank_ic_mean"], ascending=[False, False]),
    )
    save(
        "04_predictor_by_net5bps_sharpe.csv",
        lb.loc[lb["usage_type"].eq("predictor")].sort_values(
            ["net_5bps_sharpe", "rank_ic_mean"], ascending=[False, False]
        ),
    )
    for usage, part in lb.groupby("usage_type", sort=True):
        safe = str(usage).replace("/", "_")
        save(
            f"05_by_usage_{safe}_rank_ic.csv",
            part.sort_values(["rank_ic_mean", "rank_ic_hit_rate"], ascending=[False, False]),
        )

    written.extend(write_shortlists(lb, top_n=10))

    # Compact markdown overview for quick reading.
    md_lines = [
        "# 单因子排序有序结果表",
        "",
        f"- 字段数：{len(lb)}",
        f"- 样本外口径：future_return_5d，扩展窗口 2019–2024，5 日持有/调仓",
        "- 对照短表说明见 `SHORTLISTS.md`",
        "",
        "## 文件清单",
        "",
    ]
    for rel in written:
        md_lines.append(f"- `{rel}`")

    pred = _predictor_universe(lb).sort_values(
        ["rank_ic_mean", "rank_ic_hit_rate"], ascending=[False, False]
    ).head(15)
    md_lines += [
        "",
        "## Predictor 按 RankIC 前 15（摘录）",
        "",
        "| rank | feature | RankIC | hit | gross Sharpe | net5 Sharpe |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for _, row in pred.iterrows():
        md_lines.append(
            f"| {int(row['rank'])} | `{row['feature_name']}` | {row['rank_ic_mean']:.4f} | "
            f"{row['rank_ic_hit_rate']:.3f} | {row['gross_sharpe']:.3f} | {row['net_5bps_sharpe']:.3f} |"
        )
    (TABLE_DIR / "README.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    written.append(str((TABLE_DIR / "README.md").relative_to(ROOT)))
    return written


def write_figures(lb: pd.DataFrame) -> list[str]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    _style()
    written: list[str] = []
    order = ["predictor", "risk", "conditional", "quality_flag"]

    # 1) RankIC distribution by usage type
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    data = [lb.loc[lb["usage_type"].eq(u), "rank_ic_mean"].dropna().to_numpy() for u in order]
    bp = ax.boxplot(
        data,
        tick_labels=order,
        patch_artist=True,
        medianprops={"color": "#111111", "linewidth": 1.4},
        whiskerprops={"color": "#555555"},
        capprops={"color": "#555555"},
        boxprops={"color": "#555555"},
        flierprops={"marker": "o", "markersize": 3, "markerfacecolor": "#888888", "markeredgecolor": "none"},
    )
    for patch, usage in zip(bp["boxes"], order):
        patch.set_facecolor(USAGE_COLORS.get(usage, "#aaaaaa"))
        patch.set_alpha(0.55)
    ax.axhline(0.0, color="#666666", linewidth=0.8, linestyle="--")
    ax.set_title("OOS RankIC distribution by usage_type (92 fields)")
    ax.set_xlabel("usage_type")
    ax.set_ylabel("mean daily RankIC")
    fig.tight_layout()
    out = FIG_DIR / "01_rankic_by_usage_boxplot.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    written.append(out)

    # 2) Predictor RankIC bar chart
    pred = lb.loc[lb["usage_type"].eq("predictor")].sort_values("rank_ic_mean")
    fig, ax = plt.subplots(figsize=(9.5, 8.5))
    colors = ["#1f4e79" if v >= 0 else "#9c2f2f" for v in pred["rank_ic_mean"]]
    ax.barh(pred["feature_name"], pred["rank_ic_mean"], color=colors, height=0.72)
    ax.axvline(0.0, color="#666666", linewidth=0.8)
    ax.set_title("Predictor fields: OOS mean RankIC (sorted)")
    ax.set_xlabel("mean daily RankIC")
    ax.set_ylabel("feature_name")
    fig.tight_layout()
    out = FIG_DIR / "02_predictor_rankic_bars.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    written.append(out)

    # 3) RankIC vs gross Sharpe scatter
    fig, ax = plt.subplots(figsize=(7.8, 5.8))
    for usage, part in lb.groupby("usage_type"):
        ax.scatter(
            part["rank_ic_mean"],
            part["gross_sharpe"],
            s=28 if usage == "predictor" else 18,
            alpha=0.8 if usage == "predictor" else 0.45,
            c=USAGE_COLORS.get(usage, "#888888"),
            label=usage,
            edgecolors="none",
        )
    ax.axhline(0.0, color="#888888", linewidth=0.7, linestyle="--")
    ax.axvline(0.0, color="#888888", linewidth=0.7, linestyle="--")
    ax.set_title("OOS RankIC vs gross Sharpe (all 92 fields)")
    ax.set_xlabel("mean daily RankIC")
    ax.set_ylabel("gross Sharpe")
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    out = FIG_DIR / "03_rankic_vs_gross_sharpe.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    written.append(out)

    # 4) Cost sensitivity heatmap (RankIC top 12 predictors)
    top = (
        lb.loc[lb["usage_type"].eq("predictor")]
        .sort_values("rank_ic_mean", ascending=False)
        .head(12)
    )
    cost_cols = ["net_0bps_sharpe", "net_3bps_sharpe", "net_5bps_sharpe", "net_10bps_sharpe"]
    mat = top[cost_cols].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    im = ax.imshow(mat, aspect="auto", cmap="RdBu", vmin=-1.0, vmax=1.0)
    ax.set_yticks(np.arange(len(top)))
    ax.set_yticklabels(top["feature_name"])
    ax.set_xticks(np.arange(len(cost_cols)))
    ax.set_xticklabels(["0bp", "3bp", "5bp", "10bp"])
    ax.set_title("Predictor cost sensitivity: Sharpe by one-way cost (RankIC top 12)")
    ax.set_xlabel("one-way cost scenario")
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Sharpe")
    fig.tight_layout()
    out = FIG_DIR / "04_predictor_cost_sensitivity_heatmap.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    written.append(out)

    # 5) Usage composition
    counts = lb["usage_type"].value_counts().reindex(order).fillna(0)
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.bar(counts.index.astype(str), counts.to_numpy(), color=[USAGE_COLORS[u] for u in counts.index])
    ax.set_title("Registered fields by usage_type")
    ax.set_xlabel("usage_type")
    ax.set_ylabel("count")
    for i, v in enumerate(counts.to_numpy()):
        ax.text(i, v + 0.4, str(int(v)), ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    out = FIG_DIR / "05_usage_type_counts.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    written.append(out)

    # 6) Predictor net 5bp Sharpe bars
    pred_net = (
        lb.loc[lb["usage_type"].eq("predictor")]
        .sort_values("net_5bps_sharpe")
        .copy()
    )
    fig, ax = plt.subplots(figsize=(9.5, 8.5))
    colors = ["#2f6f4e" if v >= 0 else "#9c2f2f" for v in pred_net["net_5bps_sharpe"]]
    ax.barh(pred_net["feature_name"], pred_net["net_5bps_sharpe"], color=colors, height=0.72)
    ax.axvline(0.0, color="#666666", linewidth=0.8)
    ax.set_title("Predictor fields: OOS net 5bp Sharpe (protocol base)")
    ax.set_xlabel("net Sharpe (one-way 5bp)")
    ax.set_ylabel("feature_name")
    fig.tight_layout()
    out = FIG_DIR / "06_predictor_net5bps_sharpe_bars.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    written.append(out)

    # 7) Equity curves for highlight factors
    daily_path = OUTPUT / "strategy_daily_returns.csv"
    highlight = [
        "warehouse_tightness__csz",
        "carry_annualized__csz",
        "basis_fresh__csz",
        "short_reversal_5__csz",
    ]
    if daily_path.exists():
        daily = pd.read_csv(daily_path)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharex=True)
        palette = {
            "warehouse_tightness__csz": "#1f4e79",
            "carry_annualized__csz": "#2f6f4e",
            "basis_fresh__csz": "#b85c38",
            "short_reversal_5__csz": "#6b4c9a",
        }
        for feat in highlight:
            part = daily.loc[daily["feature_name"].eq(feat)].sort_values("trade_date")
            if part.empty:
                continue
            dates = pd.to_datetime(part["trade_date"])
            for ax, col, title in [
                (axes[0], "gross_return", "Gross NAV"),
                (axes[1], "net_5bps_return", "Net 5bp NAV"),
            ]:
                eq = (1.0 + part[col].astype(float)).cumprod()
                ax.plot(dates, eq.values, color=palette.get(feat, "#444"), linewidth=1.6, label=feat)
                ax.axhline(1.0, color="#999999", linewidth=0.8, linestyle="--")
                ax.set_title(title)
                ax.set_ylabel("Cumulative NAV")
        axes[0].legend(frameon=False, fontsize=8, loc="upper left")
        axes[1].set_xlabel("Rebalance date")
        axes[0].set_xlabel("Rebalance date")
        fig.suptitle("Single-factor highlight equity curves (OOS 2019–2024)", y=1.02)
        fig.tight_layout()
        out = FIG_DIR / "07_highlight_equity_curves.png"
        fig.savefig(out, dpi=160, bbox_inches="tight")
        plt.close(fig)
        written.append(out)

    # 8) Shortlist overlap counts
    overlap_path = TABLE_DIR / "09_shortlist_overlap.csv"
    if overlap_path.exists():
        overlap = pd.read_csv(overlap_path)
        trip = overlap.loc[overlap["n_shortlists"].eq(3)].sort_values("feature_name")
        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        counts_ov = overlap["n_shortlists"].value_counts().reindex([3, 2, 1]).fillna(0)
        ax.bar(
            ["In 3 shortlists", "In 2 shortlists", "In 1 shortlist"],
            counts_ov.to_numpy(),
            color=["#1f4e79", "#5b8fbf", "#b0b8c1"],
        )
        ax.set_ylabel("Number of predictors")
        ax.set_title("Predictor shortlist overlap (RankIC / net5 / stability Top10)")
        for i, v in enumerate(counts_ov.to_numpy()):
            ax.text(i, v + 0.1, str(int(v)), ha="center", va="bottom", fontsize=9)
        if not trip.empty:
            note = "Triple-overlap: " + ", ".join(
                f.replace("__csz", "") for f in trip["feature_name"].tolist()
            )
            ax.text(0.0, -0.22, note, transform=ax.transAxes, fontsize=8, color="#333333")
        fig.tight_layout()
        out = FIG_DIR / "08_shortlist_overlap.png"
        fig.savefig(out, dpi=160, bbox_inches="tight")
        plt.close(fig)
        written.append(out)

    return [str(p.relative_to(ROOT)) for p in written]


def write_visuals_guide(figure_rels: list[str], table_rels: list[str]) -> str:
    """Write a Chinese guide covering both figures and tables."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "# 单因子排序模型 — 图表说明",
        "",
        "本说明对应 `single_factor_rank_v1` 样本外结果（2019–2024，5 日持有/调仓，主成本单边 5bp）。",
        "",
        "- **图片目录**：`modeling/single_factor_rank_v1/outputs/visuals/`",
        "- **表格目录**：`modeling/single_factor_rank_v1/outputs/tables/`",
        "- **生成脚本**：`modeling/single_factor_rank_v1/make_visuals.py`",
        "",
        "## 一、图片说明",
        "",
        "| 文件 | 内容 | 怎么读 |",
        "|---|---|---|",
        "| `01_rankic_by_usage_boxplot.png` | 92 字段按 `usage_type` 的 RankIC 箱线图 | 比较 predictor / risk / conditional / quality_flag 的预测力分布；`quality_flag` 偏高时勿直接当经济结论。 |",
        "| `02_predictor_rankic_bars.png` | 36 个 predictor 的 RankIC 横条图 | 蓝为正、红为负；正文主叙事看此图头部。 |",
        "| `03_rankic_vs_gross_sharpe.png` | RankIC vs 毛 Sharpe 散点 | 右上更理想；颜色区分 usage_type。 |",
        "| `04_predictor_cost_sensitivity_heatmap.png` | RankIC Top12 的 0/3/5/10bp Sharpe 热力图 | 看成本抬升后颜色是否迅速变冷（变负）。 |",
        "| `05_usage_type_counts.png` | 注册字段构成 | 审计用：确认 36/45/6/5 构成。 |",
        "| `06_predictor_net5bps_sharpe_bars.png` | 36 predictor 净 5bp Sharpe | 协议主成本下谁还能为正；通常极少。 |",
        "| `07_highlight_equity_curves.png` | 重点因子毛/净 5bp 净值曲线 | 左毛右净；对比 `warehouse_tightness` / `carry` / `basis_fresh` / `short_reversal_5`。 |",
        "| `08_shortlist_overlap.png` | 三张 Top10 短表的交集计数 | 进入 3 张短表的因子优先写进汇报。 |",
        "",
        "## 二、表格说明",
        "",
        "### 2.1 全量 / 分组有序表",
        "",
        "| 文件 | 内容 | 推荐用途 |",
        "|---|---|---|",
        "| `01_all_by_rank_ic.csv` | 全部 92 字段按 RankIC 总榜 | 审计、复现；注意前列可能是 `__missing`。 |",
        "| `02_predictor_by_rank_ic.csv` | 仅 predictor，按 RankIC | **经济主表**。 |",
        "| `03_all_by_gross_sharpe.csv` | 全部字段按毛 Sharpe | 毛收益视角对照。 |",
        "| `04_predictor_by_net5bps_sharpe.csv` | predictor 按净 5bp Sharpe | **成本后主表**。 |",
        "| `05_by_usage_*_rank_ic.csv` | 按 usage_type 拆分的 RankIC 表 | 分类排查。 |",
        "",
        "### 2.2 对照短表（汇报用）",
        "",
        "| 文件 | 内容 | 怎么读 |",
        "|---|---|---|",
        "| `06_shortlist_top10_by_rank_ic.csv` | A：RankIC Top10 | 预测力短名单。 |",
        "| `07_shortlist_top10_by_net5bps_sharpe.csv` | B：净 5bp Sharpe Top10 | 成本后短名单。 |",
        "| `08_shortlist_top10_by_stability.csv` | C：稳定性 Top10（RankIC>0） | 正 IC 年数 / 胜率。 |",
        "| `09_shortlist_overlap.csv` | A/B/C 是否入围及入围次数 | `n_shortlists=3` 为三表交集。 |",
        "| `SHORTLISTS.md` | 短表 Markdown 摘要 | 直接可贴进报告。 |",
        "| `README.md` | 表格目录索引 + predictor 前 15 摘录 | 快速浏览入口。 |",
        "",
        "## 三、阅读建议",
        "",
        "1. **先图后表**：用 `02`/`06` 看预测力，用 `04`/`06`/`07` 看成本约束，再用 `08` 定优先讨论名单。",
        "2. **全量 92 与 predictor 36 分层**：全量榜可被缺失标志虚高；正文只用 predictor。",
        "3. **主成本是 5bp**：3bp 仅作敏感性，不要和 5bp 结论混用。",
        "4. **净值按调仓日复利**：`07` 图与 `strategy_daily_returns.csv` 均为每 5 日一点。",
        "",
        "## 四、重新生成",
        "",
        "```bash",
        "set PYTHONPATH=src",
        "py modeling/single_factor_rank_v1/make_visuals.py",
        "```",
        "",
        "## 五、本次生成文件清单",
        "",
        "### 图片",
        "",
    ]
    for rel in figure_rels:
        lines.append(f"- `{Path(rel).name}`")
    lines += ["", "### 表格", ""]
    for rel in table_rels:
        name = Path(rel).name
        if name:
            lines.append(f"- `{name}`")
    guide = FIG_DIR / "VISUALS_AND_TABLES.md"
    guide.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(guide.relative_to(ROOT))


def refresh_report_and_summary(lb: pd.DataFrame) -> None:
    from factor_modeling.single_factor_rank import write_report

    config = CONFIG
    report_path = ROOT / config["report_file"]
    fold_metrics = pd.read_csv(OUTPUT / "fold_metrics.csv")
    factor_metrics = pd.read_csv(OUTPUT / "factor_metrics.csv")
    bundle_info = json.loads((OUTPUT / "handoff_summary.json").read_text(encoding="utf-8"))
    candidates = lb[["feature_name", "source_factor", "usage_type"]].copy()
    if "catalog_direction" in lb.columns:
        candidates["expected_direction"] = lb["catalog_direction"]
    empty_top = lb.iloc[0:0].copy()
    write_report(
        report_path,
        config,
        candidates,
        factor_metrics,
        fold_metrics,
        lb,
        empty_top,
        bundle_info,
    )

    summary_path = OUTPUT / "run_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["select_top_factors"] = False
    summary.pop("top3_factors", None)
    summary.pop("top_k_factors", None)
    summary.pop("selection_rule", None)
    summary["figures_directory"] = str(FIG_DIR)
    summary["tables_directory"] = str(TABLE_DIR)
    summary["note"] = (
        "Primary outputs are per-factor metrics for all registered fields; "
        "top-k shortlist disabled. Presentation assets in visuals/ and tables/."
    )
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    for obsolete in ("top3_factors.csv", "top3_factors.json"):
        p = OUTPUT / obsolete
        if p.exists():
            p.unlink()


def main() -> int:
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    lb = pd.read_csv(OUTPUT / "leaderboard.csv")
    tables = write_tables(lb)
    figures = write_figures(lb)
    guide = write_visuals_guide(figures, tables)
    refresh_report_and_summary(lb)
    print(json.dumps({"tables": tables, "figures": figures, "guide": guide}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
