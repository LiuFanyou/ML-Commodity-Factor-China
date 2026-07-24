#!/usr/bin/env python3
"""Generate academic-style figures for unified_factor_system screening results."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
SCREEN = ROOT / "outputs" / "screening"
SEALED = ROOT / "sealed_evaluation"
SUB = ROOT / "submission" / "factor_screening_experiment_20260723"
OUT = ROOT / "reports" / "figures"

# Okabe–Ito-inspired academic palette (colorblind-friendly)
C = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "red": "#D55E00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "yellow": "#F0E442",
    "black": "#000000",
    "gray": "#7A7A7A",
    "lightgray": "#D0D0D0",
    "bg": "#FFFFFF",
}

CLASS_COLOR = {
    "A": "#0072B2",
    "B": "#0072B2",
    "C": "#009E73",
    "D": "#E69F00",
    "E": "#B0B0B0",
    "F": "#E0E0E0",
}

CLASS_SHORT = {
    "A 核心预测因子": "A",
    "B 辅助预测因子": "B",
    "C 条件变量": "C",
    "D 风险变量": "D",
    "E 暂不使用因子": "E",
    "F 数据不足无法判断": "F",
}

FAMILY_LABEL = {
    "trend_momentum": "Trend / Momentum",
    "term_structure": "Term Structure",
    "basis_spot": "Basis / Spot",
    "value_reversion": "Value / Reversion",
    "inventory_fundamental": "Inventory",
    "information_news": "Information",
    "trading_liquidity": "Liquidity",
    "volatility_risk": "Volatility / Risk",
    "industry_chain": "Industry Chain",
    "factor_interaction": "Interaction",
    "state_interaction": "State Interaction",
    "meta_factor": "Meta",
}

B_FOCUS = [
    "carry_annualized",
    "basis_fresh",
    "warehouse_tightness",
    "term_structure_slope",
    "inventory_acceleration",
    "inventory_change_5",
    "information_divergence",
    "inventory_price_divergence",
]


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.dpi": 300,
            "figure.dpi": 120,
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        path = OUT / f"{name}.{ext}"
        fig.savefig(path, bbox_inches="tight", dpi=300)
        print(f"wrote {path}")
    plt.close(fig)


def class_letter(s: str) -> str:
    return CLASS_SHORT.get(s, str(s)[:1])


def load_tables() -> dict[str, pd.DataFrame]:
    cls = pd.read_csv(SCREEN / "factor_classification.csv")
    cls["class_letter"] = cls["screening_class"].map(class_letter)
    reg = pd.read_csv(SCREEN / "factor_registry.csv")
    if "category" not in cls.columns:
        cls = cls.merge(reg[["factor_id", "category"]], on="factor_id", how="left")
    sealed = pd.read_csv(SEALED / "factor_metrics_2025.csv")
    yearly = pd.read_csv(SCREEN / "yearly_metrics.csv")
    yearly25 = pd.read_csv(SEALED / "yearly_metrics_2025.csv")
    sector = pd.read_csv(SCREEN / "sector_metrics.csv")
    quantile = pd.read_csv(SCREEN / "quantile_returns.csv")
    param = pd.read_csv(SCREEN / "parameter_sensitivity_metrics.csv")
    incr = pd.read_csv(SCREEN / "incremental_ridge_tests.csv")
    corr = pd.read_csv(SCREEN / "feature_correlation.csv", index_col=0)
    port = pd.read_csv(SCREEN / "factor_portfolio_returns.csv", parse_dates=["trade_date"])
    horizon = pd.read_csv(SCREEN / "horizon_and_delay_metrics.csv")
    return {
        "cls": cls,
        "reg": reg,
        "sealed": sealed,
        "yearly": yearly,
        "yearly25": yearly25,
        "sector": sector,
        "quantile": quantile,
        "param": param,
        "incr": incr,
        "corr": corr,
        "port": port,
        "horizon": horizon,
    }


# ---------------------------------------------------------------------------
# Fig 01: coverage bars
# ---------------------------------------------------------------------------
def fig01_coverage(data: dict) -> None:
    reg = data["reg"]
    # Audit KPIs from report + registry coverage
    audit = [
        ("Mapped actual contract", 1.000),
        ("Mapping exact rate", 0.950),
        ("Curve ≥2 contracts", 0.984),
        ("Warehouse fresh usable", 0.987),
        ("Spot fresh usable", 0.734),
        ("Actual fee any coverage", 0.732),
        ("Stock fresh usable", 0.339),
        ("Label 5d coverage", 0.985),
    ]
    labels, vals = zip(*audit)
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    y = np.arange(len(labels))
    colors = [C["red"] if v < 0.5 else (C["orange"] if v < 0.8 else C["blue"]) for v in vals]
    ax.barh(y, vals, color=colors, height=0.7, edgecolor="none")
    ax.axvline(0.5, color=C["gray"], ls="--", lw=0.8, label="0.5 threshold")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Coverage rate")
    ax.set_title("Data availability audit (core30 panel)")
    for i, v in enumerate(vals):
        ax.text(v + 0.015, i, f"{v:.3f}", va="center", fontsize=7.5)
    ax.invert_yaxis()
    save(fig, "fig01_data_coverage")


# ---------------------------------------------------------------------------
# Fig 02: class ring / donut
# ---------------------------------------------------------------------------
def fig02_class_donut(data: dict) -> None:
    cls = data["cls"]
    order = ["A", "B", "C", "D", "E", "F"]
    counts = {k: 0 for k in order}
    for letter, n in cls["class_letter"].value_counts().items():
        counts[letter] = int(n)
    # ensure A=0 appears
    labels = [f"{k} (n={counts[k]})" for k in order]
    sizes = [counts[k] for k in order]
    colors = [CLASS_COLOR[k] for k in order]
    # explode A slightly for emphasis even if zero — skip zero slices visually but keep legend
    plot_sizes = [max(s, 0) for s in sizes]
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    wedges, _ = ax.pie(
        [s if s > 0 else 1e-9 for s in plot_sizes],
        colors=colors,
        startangle=90,
        wedgeprops=dict(width=0.42, edgecolor="white", linewidth=1.2),
    )
    for w, s in zip(wedges, sizes):
        w.set_alpha(0.15 if s == 0 else 1.0)
    ax.legend(wedges, labels, loc="center left", bbox_to_anchor=(1.02, 0.5), title="Screening class")
    ax.text(0, 0.08, "A = 0", ha="center", va="center", fontsize=16, fontweight="bold", color=C["blue"])
    ax.text(0, -0.12, "57 factors\nNo confirmed core alpha", ha="center", va="center", fontsize=8, color=C["gray"])
    ax.set_title("Factor screening taxonomy (A–F)")
    save(fig, "fig02_class_donut")


# ---------------------------------------------------------------------------
# Fig 03: Rank IC horizontal bars
# ---------------------------------------------------------------------------
def fig03_rank_ic_bars(data: dict) -> None:
    cls = data["cls"].dropna(subset=["rank_ic"]).sort_values("rank_ic")
    fig, ax = plt.subplots(figsize=(7.0, 9.5))
    y = np.arange(len(cls))
    colors = [CLASS_COLOR[c] for c in cls["class_letter"]]
    ax.barh(y, cls["rank_ic"], color=colors, height=0.78, edgecolor="none")
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(cls["factor_id"], fontsize=6.5)
    ax.set_xlabel("Development Rank IC (2019–2024 OOS expanding folds)")
    ax.set_title("Cross-sectional Rank IC leaderboard")
    # legend proxies
    from matplotlib.patches import Patch

    handles = [Patch(color=CLASS_COLOR[k], label=k) for k in ["B", "C", "D", "E"]]
    ax.legend(handles=handles, title="Class", loc="lower right")
    save(fig, "fig03_rank_ic_leaderboard")


# ---------------------------------------------------------------------------
# Fig 04: Rank IC vs Sharpe scatter
# ---------------------------------------------------------------------------
def fig04_ic_sharpe_scatter(data: dict) -> None:
    cls = data["cls"].dropna(subset=["rank_ic", "sharpe_5bps"])
    fig, ax = plt.subplots(figsize=(6.0, 5.2))
    for letter in ["E", "D", "C", "B"]:
        sub = cls[cls["class_letter"] == letter]
        ax.scatter(
            sub["rank_ic"],
            sub["sharpe_5bps"],
            s=np.clip(sub.get("coverage", pd.Series(0.7, index=sub.index)).fillna(0.5) * 80, 25, 90),
            c=CLASS_COLOR[letter],
            alpha=0.85 if letter == "B" else 0.55,
            edgecolors="black" if letter == "B" else "none",
            linewidths=0.6,
            label=f"Class {letter} (n={len(sub)})",
            zorder=3 if letter == "B" else 2,
        )
    ax.axhline(0, color=C["gray"], lw=0.7)
    ax.axvline(0, color=C["gray"], lw=0.7)
    for _, r in cls[cls["class_letter"] == "B"].iterrows():
        ax.annotate(
            r["factor_id"],
            (r["rank_ic"], r["sharpe_5bps"]),
            textcoords="offset points",
            xytext=(4, 3),
            fontsize=6.2,
            color=C["blue"],
        )
    ax.set_xlabel("Rank IC")
    ax.set_ylabel("Net Sharpe (5 bps one-way)")
    ax.set_title("Predictive power vs. tradability")
    ax.legend(loc="upper left")
    save(fig, "fig04_ic_vs_sharpe")


# ---------------------------------------------------------------------------
# Fig 05: quintile bars for B-class
# ---------------------------------------------------------------------------
def fig05_quantile_bars(data: dict) -> None:
    q = data["quantile"]
    factors = [f for f in B_FOCUS if f in set(q["factor_id"])]
    n = len(factors)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(10.5, 2.4 * nrows), sharey=True)
    axes = np.atleast_1d(axes).ravel()
    for i, fid in enumerate(factors):
        ax = axes[i]
        sub = q[q["factor_id"] == fid]
        g = sub.groupby("quantile", as_index=True)["return_5d"].mean().reindex([1, 2, 3, 4, 5])
        se = sub.groupby("quantile")["return_5d"].apply(lambda x: x.std(ddof=1) / np.sqrt(len(x)))
        se = se.reindex([1, 2, 3, 4, 5])
        x = np.arange(1, 6)
        ax.bar(x, g.values, color=C["blue"], alpha=0.85, width=0.7, yerr=se.values, capsize=2, ecolor=C["gray"])
        ax.axhline(0, color="black", lw=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels([f"Q{k}" for k in x])
        ax.set_title(fid, fontsize=8.5)
        if i % ncols == 0:
            ax.set_ylabel("Mean 5d return")
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    fig.suptitle("Quintile portfolio monotonicity (B-class candidates)", y=1.01, fontsize=11)
    fig.tight_layout()
    save(fig, "fig05_quantile_monotonicity")


# ---------------------------------------------------------------------------
# Fig 06: family average Rank IC
# ---------------------------------------------------------------------------
def fig06_family_bars(data: dict) -> None:
    cls = data["cls"].dropna(subset=["rank_ic"])
    g = (
        cls.groupby("category")
        .agg(mean_rank_ic=("rank_ic", "mean"), n=("rank_ic", "count"), std=("rank_ic", "std"))
        .sort_values("mean_rank_ic")
    )
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    y = np.arange(len(g))
    colors = [C["blue"] if v >= 0 else C["red"] for v in g["mean_rank_ic"]]
    se = (g["std"] / np.sqrt(g["n"])).fillna(0)
    ax.barh(y, g["mean_rank_ic"], xerr=se, color=colors, height=0.72, ecolor=C["gray"], capsize=2)
    ax.axvline(0, color="black", lw=0.8)
    labels = [f"{FAMILY_LABEL.get(i, i)} (n={int(g.loc[i, 'n'])})" for i in g.index]
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Mean Rank IC")
    ax.set_title("Factor-family average predictive strength")
    save(fig, "fig06_family_rank_ic")


# ---------------------------------------------------------------------------
# Fig 07: sector heatmap (B + selected)
# ---------------------------------------------------------------------------
def fig07_sector_heatmap(data: dict) -> None:
    sector = data["sector"]
    focus = [f for f in B_FOCUS if f in set(sector["factor_id"])]
    # add a few E/D contrasts
    extra = ["tsmom_60", "amihud_20", "correlation_risk_20_120"]
    rows = focus + [e for e in extra if e in set(sector["factor_id"])]
    mat = (
        sector[sector["factor_id"].isin(rows)]
        .pivot(index="factor_id", columns="sector", values="rank_ic")
        .reindex(index=rows)
    )
    # order columns
    cols = [c for c in ["energy", "metals", "chemicals", "agriculture"] if c in mat.columns]
    mat = mat[cols]
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    vmax = np.nanmax(np.abs(mat.values))
    vmax = max(float(vmax), 0.02)
    im = ax.imshow(mat.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(mat.columns)))
    ax.set_xticklabels(mat.columns, rotation=0)
    ax.set_yticks(range(len(mat.index)))
    ax.set_yticklabels(mat.index, fontsize=8)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=7, color="black")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Rank IC")
    ax.set_title("Sector robustness of Rank IC")
    save(fig, "fig07_sector_heatmap")


# ---------------------------------------------------------------------------
# Fig 08: year x factor heatmap + 2025
# ---------------------------------------------------------------------------
def fig08_yearly_heatmap(data: dict) -> None:
    yearly = data["yearly"].copy()
    y25 = data["yearly25"].copy()
    focus = [f for f in B_FOCUS if f in set(yearly["factor_id"])]
    extra = ["tsmom_60", "csmom_60", "amihud_20", "multi_horizon_trend"]
    rows = focus + [e for e in extra if e in set(yearly["factor_id"])]
    mat = yearly[yearly["factor_id"].isin(rows)].pivot(index="factor_id", columns="year", values="rank_ic")
    mat = mat.reindex(index=rows)
    if not y25.empty:
        m25 = y25[y25["factor_id"].isin(rows)].set_index("factor_id")["rank_ic"]
        mat[2025] = m25
    years = sorted([c for c in mat.columns if c != 2025]) + ([2025] if 2025 in mat.columns else [])
    mat = mat[years]
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    vmax = np.nanmax(np.abs(mat.values))
    vmax = max(float(vmax), 0.03)
    im = ax.imshow(mat.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(mat.columns)))
    ax.set_xticklabels([str(int(c)) if c != 2025 else "2025*" for c in mat.columns])
    ax.set_yticks(range(len(mat.index)))
    ax.set_yticklabels(mat.index, fontsize=8)
    # separator before 2025
    if 2025 in mat.columns:
        ax.axvline(len(mat.columns) - 1.5, color="black", lw=1.2)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat.values[i, j]
            txt = "NA" if not np.isfinite(v) else f"{v:.2f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=6.5)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Rank IC")
    ax.set_title("Yearly Rank IC stability (2019–2024) + sealed 2025")
    ax.text(0.0, -0.12, "*2025 is a one-shot sealed evaluation; not used for selection.", transform=ax.transAxes, fontsize=7, color=C["gray"])
    save(fig, "fig08_yearly_rank_ic_heatmap")


# ---------------------------------------------------------------------------
# Fig 09: parameter sensitivity
# ---------------------------------------------------------------------------
def fig09_param_sensitivity(data: dict) -> None:
    param = data["param"].copy()
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    families = list(param["factor_family"].unique())
    cmap = plt.cm.tab10
    for i, fam in enumerate(families):
        sub = param[param["factor_family"] == fam].sort_values("lookback_window")
        se = sub["rank_ic_std"] / np.sqrt(sub["date_count"].clip(lower=1))
        color = cmap(i % 10)
        ax.plot(sub["lookback_window"], sub["rank_ic"], "-o", ms=4, color=color, label=fam, lw=1.4)
        ax.fill_between(
            sub["lookback_window"],
            sub["rank_ic"] - 1.96 * se,
            sub["rank_ic"] + 1.96 * se,
            color=color,
            alpha=0.15,
            linewidth=0,
        )
    ax.axhline(0, color="black", lw=0.7)
    ax.set_xlabel("Lookback window (days)")
    ax.set_ylabel("Rank IC")
    ax.set_title("Lookback-window sensitivity (diagnostic)")
    ax.legend(ncol=2, fontsize=7)
    save(fig, "fig09_parameter_sensitivity")


# ---------------------------------------------------------------------------
# Fig 10: cost sensitivity for B-class
# ---------------------------------------------------------------------------
def fig10_cost_sensitivity(data: dict) -> None:
    cls = data["cls"]
    b = cls[cls["class_letter"] == "B"].copy()
    costs = [0, 3, 5, 10]
    n = len(b)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(10.5, 2.5 * nrows), sharex=True)
    axes = np.atleast_1d(axes).ravel()
    for i, (_, r) in enumerate(b.sort_values("rank_ic", ascending=False).iterrows()):
        ax = axes[i]
        rets = [r.get(f"annual_return_{c}bps", np.nan) for c in costs]
        sharpes = [r.get(f"sharpe_{c}bps", np.nan) for c in costs]
        ax.bar(costs, rets, width=1.6, color=C["lightgray"], edgecolor=C["gray"], label="Ann. return")
        ax2 = ax.twinx()
        ax2.plot(costs, sharpes, "-o", color=C["blue"], ms=4, lw=1.4, label="Sharpe")
        ax.axvline(5, color=C["red"], ls="--", lw=0.8, alpha=0.7)
        ax.set_title(r["factor_id"], fontsize=8)
        ax.set_xticks(costs)
        if i >= n - ncols:
            ax.set_xlabel("Cost (bps)")
        if i % ncols == 0:
            ax.set_ylabel("Ann. return")
        ax2.set_ylabel("Sharpe", fontsize=7, color=C["blue"])
        ax2.tick_params(axis="y", labelcolor=C["blue"], labelsize=7)
        ax.spines["right"].set_visible(True)
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    fig.suptitle("Cost-scenario sensitivity of B-class factors", y=1.01, fontsize=11)
    fig.tight_layout()
    save(fig, "fig10_cost_sensitivity")


# ---------------------------------------------------------------------------
# Fig 11: cumulative NAV for B-class (5bps)
# ---------------------------------------------------------------------------
def fig11_nav_curves(data: dict) -> None:
    port = data["port"]
    factors = [f for f in B_FOCUS if f in set(port["factor_id"])][:6]
    fig, (ax, axz) = plt.subplots(
        1, 2, figsize=(10.5, 4.0), gridspec_kw={"width_ratios": [2.2, 1.0]}
    )
    colors = [C["blue"], C["orange"], C["green"], C["red"], C["purple"], C["sky"]]
    for fid, color in zip(factors, colors):
        sub = port[port["factor_id"] == fid].sort_values("trade_date")
        nav = (1.0 + sub["net_return_5bps"].fillna(0)).cumprod()
        ax.plot(sub["trade_date"], nav, color=color, lw=1.3, label=fid)
        # zoom last 2y
        mask = sub["trade_date"] >= pd.Timestamp("2023-01-01")
        if mask.any():
            nav_z = (1.0 + sub.loc[mask, "net_return_5bps"].fillna(0)).cumprod()
            axz.plot(sub.loc[mask, "trade_date"], nav_z, color=color, lw=1.3)
    ax.axhline(1.0, color=C["gray"], lw=0.7)
    axz.axhline(1.0, color=C["gray"], lw=0.7)
    ax.set_ylabel("Cumulative net NAV (5 bps)")
    ax.set_title("Long–short accounting path (development)")
    ax.legend(fontsize=7, loc="upper left")
    axz.set_title("Inset: 2023–2024")
    fig.autofmt_xdate(rotation=30, ha="right")
    fig.tight_layout()
    save(fig, "fig11_nav_curves")


# ---------------------------------------------------------------------------
# Fig 12: Dev vs 2025 scatter
# ---------------------------------------------------------------------------
def fig12_dev_vs_2025(data: dict) -> None:
    cls = data["cls"][["factor_id", "rank_ic", "class_letter"]].rename(columns={"rank_ic": "rank_ic_dev"})
    sealed = data["sealed"][["factor_id", "rank_ic"]].rename(columns={"rank_ic": "rank_ic_2025"})
    m = cls.merge(sealed, on="factor_id", how="inner").dropna(subset=["rank_ic_dev", "rank_ic_2025"])
    m["same_sign"] = np.sign(m["rank_ic_dev"]) * np.sign(m["rank_ic_2025"]) > 0
    rate = m["same_sign"].mean()
    fig, ax = plt.subplots(figsize=(5.8, 5.4))
    for same, color, label in [
        (True, C["blue"], "Same sign"),
        (False, C["red"], "Sign flip"),
    ]:
        sub = m[m["same_sign"] == same]
        ax.scatter(sub["rank_ic_dev"], sub["rank_ic_2025"], c=color, s=36, alpha=0.8, label=label, edgecolors="none")
    # highlight B
    b = m[m["class_letter"] == "B"]
    ax.scatter(b["rank_ic_dev"], b["rank_ic_2025"], facecolors="none", edgecolors="black", s=70, lw=1.0, label="Class B", zorder=4)
    lim = max(np.nanmax(np.abs(m[["rank_ic_dev", "rank_ic_2025"]].values)), 0.05) * 1.1
    ax.plot([-lim, lim], [-lim, lim], ls="--", color=C["gray"], lw=0.9, label="y = x")
    ax.axhline(0, color=C["gray"], lw=0.6)
    ax.axvline(0, color=C["gray"], lw=0.6)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("Development Rank IC")
    ax.set_ylabel("Sealed 2025 Rank IC")
    ax.set_title(f"OOS sign persistence (same-sign rate = {rate:.1%})")
    # annotate flips of interest
    for fid in ["information_divergence", "inventory_price_divergence", "warehouse_tightness", "carry_annualized"]:
        row = m[m["factor_id"] == fid]
        if len(row):
            r = row.iloc[0]
            ax.annotate(fid, (r["rank_ic_dev"], r["rank_ic_2025"]), fontsize=6, xytext=(4, 4), textcoords="offset points")
    ax.legend(loc="upper left", fontsize=7)
    save(fig, "fig12_dev_vs_2025_scatter")


# ---------------------------------------------------------------------------
# Fig 13: B-class Dev vs 2025 grouped bars
# ---------------------------------------------------------------------------
def fig13_bclass_dev_2025_bars(data: dict) -> None:
    cls = data["cls"].set_index("factor_id")
    sealed = data["sealed"].set_index("factor_id")
    factors = [f for f in B_FOCUS if f in cls.index]
    dev = [cls.loc[f, "rank_ic"] if pd.notna(cls.loc[f, "rank_ic"]) else np.nan for f in factors]
    oos = [
        sealed.loc[f, "rank_ic"] if f in sealed.index and pd.notna(sealed.loc[f, "rank_ic"]) else np.nan
        for f in factors
    ]
    y = np.arange(len(factors))
    h = 0.36
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    ax.barh(y + h / 2, dev, height=h, color=C["blue"], label="Development")
    ax.barh(y - h / 2, oos, height=h, color="white", edgecolor=C["red"], hatch="////", label="Sealed 2025")
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(factors)
    ax.set_xlabel("Rank IC")
    ax.set_title("B-class candidates: development vs. sealed 2025")
    ax.legend(loc="lower right")
    # NA markers
    for i, v in enumerate(oos):
        if not np.isfinite(v):
            ax.text(0.002, y[i] - h / 2, "NA", va="center", fontsize=7, color=C["gray"])
    ax.invert_yaxis()
    save(fig, "fig13_bclass_dev_2025_bars")


# ---------------------------------------------------------------------------
# Fig 14: feature correlation heatmap (cluster-ordered, predictors only)
# ---------------------------------------------------------------------------
def fig14_feature_corr(data: dict) -> None:
    corr = data["corr"].copy()
    mlreg = pd.read_csv(SCREEN / "ml_feature_registry.csv")
    # prefer continuous predictors (csz), drop missing flags for clarity
    preds = mlreg[mlreg["feature_name"].str.endswith("__csz")]["feature_name"].tolist()
    preds = [p for p in preds if p in corr.columns]
    # order by correlation_cluster if available
    if "correlation_cluster" in mlreg.columns:
        order = (
            mlreg[mlreg["feature_name"].isin(preds)]
            .sort_values(["correlation_cluster", "feature_name"])["feature_name"]
            .tolist()
        )
    else:
        order = preds
    mat = corr.loc[order, order].astype(float)
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    im = ax.imshow(mat.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(f"ML features (__csz only, n={len(order)}), cluster-ordered")
    ax.set_title("Feature correlation matrix (redundancy diagnosis)")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Pearson ρ")
    # annotate a few high-corr pairs in caption area
    abs_mat = mat.abs().copy()
    np.fill_diagonal(abs_mat.values, 0)
    # top pair
    flat = abs_mat.stack().sort_values(ascending=False)
    top = flat.head(3)
    notes = "; ".join([f"{a.replace('__csz','')}–{b.replace('__csz','')}={v:.2f}" for (a, b), v in top.items()])
    ax.text(0.0, -0.06, f"Top |ρ|: {notes}", transform=ax.transAxes, fontsize=6.5, color=C["gray"])
    save(fig, "fig14_feature_correlation")


# ---------------------------------------------------------------------------
# Fig 15: Ridge incremental bars
# ---------------------------------------------------------------------------
def fig15_ridge_incremental(data: dict) -> None:
    incr = data["incr"].dropna(subset=["mean_oos_delta_rank_ic"]).sort_values("mean_oos_delta_rank_ic")
    fig, ax = plt.subplots(figsize=(6.5, max(3.5, 0.22 * len(incr) + 1)))
    y = np.arange(len(incr))
    colors = [C["green"] if v >= 0 else C["red"] for v in incr["mean_oos_delta_rank_ic"]]
    ax.barh(y, incr["mean_oos_delta_rank_ic"], color=colors, height=0.75)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(incr["factor_id"], fontsize=7)
    ax.set_xlabel(r"Mean OOS $\Delta$Rank IC vs. fixed Ridge baseline")
    ax.set_title("Incremental predictive value (diagnostic only)")
    save(fig, "fig15_ridge_incremental")


# ---------------------------------------------------------------------------
# Fig 16: horizon robustness for B-class
# ---------------------------------------------------------------------------
def fig16_horizon(data: dict) -> None:
    hz = data["horizon"].copy()
    # normalize horizon label
    factors = [f for f in B_FOCUS if f in set(hz["factor_id"])]
    # map horizon labels
    order = ["1", "5", "20", "5d_signal_delayed_1d"]
    label_map = {"1": "1d", "5": "5d", "20": "20d", "5d_signal_delayed_1d": "5d+1delay"}
    hz["h"] = hz["horizon_days"].astype(str)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    x = np.arange(len(order))
    width = 0.1
    for i, fid in enumerate(factors):
        sub = hz[hz["factor_id"] == fid].set_index("h")
        vals = [sub.loc[h, "rank_ic"] if h in sub.index else np.nan for h in order]
        ax.plot(x, vals, "-o", ms=4, lw=1.2, label=fid)
    ax.set_xticks(x)
    ax.set_xticklabels([label_map[h] for h in order])
    ax.axhline(0, color="black", lw=0.7)
    ax.set_ylabel("Rank IC")
    ax.set_xlabel("Label horizon / delay")
    ax.set_title("Horizon and signal-delay robustness (B-class)")
    ax.legend(fontsize=6.5, ncol=2, loc="best")
    save(fig, "fig16_horizon_delay")


# ---------------------------------------------------------------------------
# Fig 17: sealed 2025 NAV for persistent B factors
# ---------------------------------------------------------------------------
def fig17_sealed_nav() -> None:
    path = SEALED / "factor_portfolio_returns_2025.csv"
    if not path.exists():
        return
    port = pd.read_csv(path, parse_dates=["trade_date"])
    keep = ["carry_annualized", "warehouse_tightness", "inventory_acceleration", "term_structure_slope"]
    keep = [f for f in keep if f in set(port["factor_id"])]
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    colors = [C["blue"], C["orange"], C["green"], C["purple"]]
    for fid, color in zip(keep, colors):
        sub = port[port["factor_id"] == fid].sort_values("trade_date")
        nav = (1 + sub["net_return_5bps"].fillna(0)).cumprod()
        ax.plot(sub["trade_date"], nav, color=color, lw=1.4, label=fid)
    ax.axhline(1.0, color=C["gray"], lw=0.7)
    ax.set_ylabel("Cumulative net NAV (5 bps)")
    ax.set_title("Sealed 2025 long–short paths (sign-persistent B candidates)")
    ax.legend(fontsize=7)
    fig.autofmt_xdate(rotation=30, ha="right")
    save(fig, "fig17_sealed_2025_nav")


# ---------------------------------------------------------------------------
# Fig 18: summary dashboard of key metrics table-as-heatmap for B
# ---------------------------------------------------------------------------
def fig18_bclass_scorecard(data: dict) -> None:
    cls = data["cls"].set_index("factor_id")
    sealed = data["sealed"].set_index("factor_id")
    factors = [f for f in B_FOCUS if f in cls.index]
    metrics = {
        "Rank IC": [cls.loc[f, "rank_ic"] for f in factors],
        "Rank ICIR": [cls.loc[f, "rank_ic_ir"] for f in factors],
        "Sharpe 5bps": [cls.loc[f, "sharpe_5bps"] for f in factors],
        "Ann. ret 5bps": [cls.loc[f, "annual_return_5bps"] for f in factors],
        "FDR q": [cls.loc[f, "rank_ic_fdr_q"] for f in factors],
        "2025 Rank IC": [sealed.loc[f, "rank_ic"] if f in sealed.index else np.nan for f in factors],
    }
    mat = pd.DataFrame(metrics, index=factors)
    # z-score within column for color (except FDR inverted)
    color_mat = mat.copy()
    for c in color_mat.columns:
        s = color_mat[c]
        if c == "FDR q":
            color_mat[c] = -(s - s.mean()) / (s.std(ddof=0) + 1e-12)
        else:
            color_mat[c] = (s - s.mean()) / (s.std(ddof=0) + 1e-12)
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    vmax = np.nanmax(np.abs(color_mat.values))
    im = ax.imshow(color_mat.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(mat.columns)))
    ax.set_xticklabels(mat.columns, rotation=25, ha="right")
    ax.set_yticks(range(len(mat.index)))
    ax.set_yticklabels(mat.index)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat.values[i, j]
            txt = "NA" if not np.isfinite(v) else (f"{v:.3f}" if abs(v) < 10 else f"{v:.2f}")
            ax.text(j, i, txt, ha="center", va="center", fontsize=7)
    ax.set_title("B-class multi-metric scorecard (color = within-column z-score)")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Column z-score")
    save(fig, "fig18_bclass_scorecard")


def write_index() -> None:
    files = sorted(OUT.glob("fig*.png"))
    lines = [
        "# Screening figures",
        "",
        "Generated by `scripts/generate_figures.py`.",
        "",
        "| Figure | File | Narrative |",
        "|---|---|---|",
    ]
    desc = {
        "fig01_data_coverage": "Data coverage audit",
        "fig02_class_donut": "A–F taxonomy (A=0)",
        "fig03_rank_ic_leaderboard": "Full Rank IC leaderboard",
        "fig04_ic_vs_sharpe": "Rank IC vs 5bps Sharpe",
        "fig05_quantile_monotonicity": "Quintile monotonicity (B-class)",
        "fig06_family_rank_ic": "Family-average Rank IC",
        "fig07_sector_heatmap": "Sector robustness heatmap",
        "fig08_yearly_rank_ic_heatmap": "Yearly + sealed 2025 Rank IC",
        "fig09_parameter_sensitivity": "Lookback sensitivity",
        "fig10_cost_sensitivity": "Cost-scenario sensitivity",
        "fig11_nav_curves": "Development NAV curves",
        "fig12_dev_vs_2025_scatter": "Dev vs 2025 sign persistence",
        "fig13_bclass_dev_2025_bars": "B-class Dev vs 2025 bars",
        "fig14_feature_correlation": "ML feature correlation",
        "fig15_ridge_incremental": "Ridge incremental ΔRank IC",
        "fig16_horizon_delay": "Horizon / delay robustness",
        "fig17_sealed_2025_nav": "Sealed 2025 NAV",
        "fig18_bclass_scorecard": "B-class scorecard",
    }
    for p in files:
        stem = p.stem
        lines.append(f"| {stem} | `{p.name}` | {desc.get(stem, '')} |")
        lines.append("")
        lines.append(f"![{stem}]({p.name})")
        lines.append("")
    (OUT / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT / 'README.md'}")


def main() -> None:
    setup_style()
    OUT.mkdir(parents=True, exist_ok=True)
    data = load_tables()
    fig01_coverage(data)
    fig02_class_donut(data)
    fig03_rank_ic_bars(data)
    fig04_ic_sharpe_scatter(data)
    fig05_quantile_bars(data)
    fig06_family_bars(data)
    fig07_sector_heatmap(data)
    fig08_yearly_heatmap(data)
    fig09_param_sensitivity(data)
    fig10_cost_sensitivity(data)
    fig11_nav_curves(data)
    fig12_dev_vs_2025(data)
    fig13_bclass_dev_2025_bars(data)
    fig14_feature_corr(data)
    fig15_ridge_incremental(data)
    fig16_horizon(data)
    fig17_sealed_nav()
    fig18_bclass_scorecard(data)
    write_index()
    print(f"Done. Figures in {OUT}")


if __name__ == "__main__":
    main()
