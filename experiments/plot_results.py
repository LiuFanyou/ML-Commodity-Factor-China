"""Create aggregate-only figures for the three rolling OOS experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


STUDY_LABELS = {"baseline": "All features", "selected": "Selected features", "complex": "Complex trees"}
MODEL_LABELS = {"lightgbm": "LightGBM", "xgboost": "XGBoost", "random_forest": "Random forest"}
COLORS = {"lightgbm": "#1f77b4", "xgboost": "#ff7f0e", "random_forest": "#2ca02c"}


def find_csv(folder: Path, required: set[str], exclude: set[str] | None = None) -> Path:
    exclude = exclude or set()
    for path in folder.glob("*.csv"):
        columns = set(pd.read_csv(path, nrows=1).columns)
        if required.issubset(columns) and not columns.intersection(exclude):
            return path
    raise FileNotFoundError(f"No CSV in {folder} contains {required}")


def load_study(folder: Path) -> dict[str, pd.DataFrame]:
    return {
        "metrics": pd.read_csv(find_csv(folder, {"stable_oos_criteria_met", "annual_return", "rank_ic"})),
        "costs": pd.read_csv(find_csv(folder, {"cost_bps_one_way", "annual_return", "sharpe"})),
        "daily": pd.read_csv(find_csv(folder, {"trade_date", "net_return", "nav", "cost_bps_one_way"}), parse_dates=["trade_date"]),
        "predictions": pd.read_csv(find_csv(folder, {"prediction", "future_return_5d"}, {"weight"}), parse_dates=["trade_date"]),
    }


def configure() -> None:
    plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 180, "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.alpha": 0.25, "axes.unicode_minus": False})


def save(fig: plt.Figure, output: Path, filename: str, tight: bool = True) -> None:
    if tight:
        fig.tight_layout()
    fig.savefig(output / filename, bbox_inches="tight")
    plt.close(fig)


def plot_comparison(studies: dict[str, dict[str, pd.DataFrame]], output: Path) -> None:
    metrics = ["annual_return", "sharpe", "rank_ic"]
    titles = ["Annual return at 5 bps", "Sharpe at 5 bps", "5-day Rank IC"]
    models = list(MODEL_LABELS)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    width = 0.23
    for axis, metric, title in zip(axes, metrics, titles):
        for index, (study, data) in enumerate(studies.items()):
            values = data["metrics"].set_index("model").reindex(models)[metric]
            axis.bar(np.arange(len(models)) + (index - 1) * width, values, width, label=STUDY_LABELS[study])
        axis.axhline(0, color="#444444", linewidth=0.8)
        axis.set_xticks(range(len(models)), [MODEL_LABELS[m] for m in models])
        axis.set_title(title)
        axis.yaxis.set_major_formatter(lambda x, _: f"{x:.1%}" if metric != "sharpe" else f"{x:.2f}")
    axes[0].legend(frameon=False, loc="best")
    save(fig, output, "model-comparison-5bps.png")


def plot_costs(studies: dict[str, dict[str, pd.DataFrame]], output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
    for axis, (study, data) in zip(axes, studies.items()):
        for model, group in data["costs"].groupby("model"):
            group = group.sort_values("cost_bps_one_way")
            axis.plot(group["cost_bps_one_way"], group["annual_return"], marker="o", label=MODEL_LABELS[model], color=COLORS[model])
        axis.axhline(0, color="#444444", linewidth=0.8)
        axis.set_title(STUDY_LABELS[study]); axis.set_xlabel("One-way cost (bps)")
        axis.yaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    axes[0].set_ylabel("Annual return")
    axes[-1].legend(frameon=False, loc="best")
    save(fig, output, "cost-sensitivity.png")


def plot_nav(studies: dict[str, dict[str, pd.DataFrame]], output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for axis, (study, data) in zip(axes, studies.items()):
        daily = data["daily"].loc[data["daily"]["cost_bps_one_way"].eq(5)]
        for model, group in daily.groupby("model"):
            group = group.sort_values("trade_date")
            axis.plot(group["trade_date"], group["nav"], label=MODEL_LABELS[model], color=COLORS[model], linewidth=1.2)
        axis.axhline(1, color="#444444", linewidth=0.8)
        axis.set_title(f"{STUDY_LABELS[study]} at 5 bps")
        axis.set_xlabel("Date"); axis.set_ylabel("NAV")
    axes[-1].legend(frameon=False, loc="best")
    save(fig, output, "nav-curves-5bps.png")


def plot_annual_rank_ic(studies: dict[str, dict[str, pd.DataFrame]], output: Path) -> None:
    years = list(range(2019, 2025))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
    for axis, (study, data) in zip(axes, studies.items()):
        values = []
        for model in MODEL_LABELS:
            rows = []
            subset = data["predictions"].loc[data["predictions"]["model"].eq(model)]
            for _, group in subset.groupby("trade_date"):
                if group["prediction"].nunique() > 1 and group["future_return_5d"].nunique() > 1:
                    rows.append((group["trade_date"].iloc[0].year, group["prediction"].corr(group["future_return_5d"], method="spearman")))
            values.append(pd.DataFrame(rows, columns=["year", "rank_ic"]).groupby("year")["rank_ic"].mean().reindex(years).to_numpy())
        image = axis.imshow(np.asarray(values), cmap="RdYlGn", vmin=-0.05, vmax=0.05, aspect="auto")
        axis.set_xticks(range(len(years)), years)
        axis.set_yticks(range(len(MODEL_LABELS)), [MODEL_LABELS[m] for m in MODEL_LABELS])
        axis.set_title(STUDY_LABELS[study])
        for row in range(len(MODEL_LABELS)):
            for col in range(len(years)):
                value = values[row][col]
                axis.text(col, row, "NA" if np.isnan(value) else f"{value:.3f}", ha="center", va="center", fontsize=8)
    fig.subplots_adjust(right=0.91, wspace=0.08)
    fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.8, pad=0.02, label="Mean annual 5-day Rank IC")
    save(fig, output, "annual-rank-ic-heatmap.png", tight=False)


def plot_selection_stability(folder: Path, output: Path) -> None:
    selection = pd.read_csv(find_csv(folder, {"feature_name", "selected", "fold_id"})).query("selected == True")
    frequency = selection.groupby("feature_name")["fold_id"].nunique().sort_values(ascending=False).head(20).sort_values()
    fig, axis = plt.subplots(figsize=(9, 6.5))
    axis.barh(frequency.index, frequency.values, color="#1f77b4")
    axis.set_xlabel("Selected folds (out of 6)"); axis.set_ylabel("Feature")
    axis.set_title("Training-only Top-30 feature selection stability")
    axis.set_xlim(0, 6.4)
    for y, value in enumerate(frequency.values):
        axis.text(value + 0.08, y, str(value), va="center")
    save(fig, output, "factor-selection-stability.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--selected", type=Path, required=True)
    parser.add_argument("--complex", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("figures"))
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True); configure()
    studies = {"baseline": load_study(args.baseline), "selected": load_study(args.selected), "complex": load_study(args.complex)}
    plot_comparison(studies, args.output); plot_costs(studies, args.output); plot_nav(studies, args.output)
    plot_annual_rank_ic(studies, args.output); plot_selection_stability(args.selected, args.output)
    print(f"wrote figures to {args.output}")


if __name__ == "__main__":
    main()
