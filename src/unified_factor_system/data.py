from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import ProjectConfig


DATE_COLUMNS = [
    "trade_date",
    "list_date",
    "delist_date",
    "last_delivery_date",
    "spot_observation_date",
    "warehouse_receipt_observation_date",
    "stock_observation_date",
    "available_stock_observation_date",
]

CONTRACT_COLUMNS = [
    "trade_date",
    "ts_code",
    "product_code",
    "product_name",
    "exchange",
    "contract",
    "list_date",
    "delist_date",
    "last_delivery_date",
    "contract_size",
    "quote_unit",
    "pre_close",
    "pre_settle",
    "open",
    "high",
    "low",
    "close",
    "settle",
    "volume",
    "amount",
    "open_interest",
    "trading_fee_rate",
    "trading_fee",
    "spot_price",
    "spot_price_unit",
    "spot_indicator_code",
    "spot_indicator_name",
    "spot_price_source",
    "spot_observation_date",
    "spot_price_age_days",
    "warehouse_receipt_current",
    "warehouse_receipt_change",
    "warehouse_receipt_unit",
    "warehouse_receipt_observation_date",
    "warehouse_receipt_age_days",
    "warehouse_receipt_conflict",
    "stock_current",
    "stock_unit",
    "stock_observation_date",
    "stock_age_days",
    "available_stock_current",
    "available_stock_unit",
    "available_stock_observation_date",
    "available_stock_age_days",
]

CONTINUOUS_COLUMNS = [
    "ts_code",
    "trade_date",
    "pre_close",
    "pre_settle",
    "open",
    "high",
    "low",
    "close",
    "settle",
    "vol",
    "amount",
    "oi",
]


@dataclass
class DataBundle:
    panel: pd.DataFrame
    contracts: pd.DataFrame
    continuous: pd.DataFrame
    source_audit: dict[str, Any]


def _sector_lookup(config: ProjectConfig) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for sector, products in config.raw["sector_map"].items():
        for product in products:
            if product in mapping:
                raise ValueError(f"product appears in multiple sectors: {product}")
            mapping[product] = sector
    return mapping


def _read_contracts(config: ProjectConfig) -> pd.DataFrame:
    if not config.data_file.is_file():
        raise FileNotFoundError(config.data_file)
    frame = pd.read_csv(
        config.data_file,
        usecols=CONTRACT_COLUMNS,
        parse_dates=[name for name in DATE_COLUMNS if name in CONTRACT_COLUMNS],
        encoding="utf-8-sig",
        low_memory=False,
    )
    frame = frame.rename(columns={"product_code": "product"})
    frame["product"] = frame["product"].astype(str).str.upper()
    if frame.duplicated(["trade_date", "ts_code"]).any():
        raise ValueError("duplicate trade_date-ts_code keys in contract data")
    frame["days_to_delist"] = (frame["delist_date"] - frame["trade_date"]).dt.days
    price_equal = (
        frame[["open", "high", "low", "close"]]
        .notna()
        .all(axis=1)
        & frame["open"].eq(frame["high"])
        & frame["open"].eq(frame["low"])
        & frame["open"].eq(frame["close"])
        & frame["volume"].gt(0)
    )
    frame["suspected_price_limit_lock"] = price_equal.astype("int8")
    frame["suspected_price_limit_direction"] = np.sign(
        frame["open"] - frame["pre_settle"]
    ).where(price_equal)
    return frame.sort_values(["product", "trade_date", "delist_date", "ts_code"]).reset_index(
        drop=True
    )


def _continuous_code_map(
    contracts: pd.DataFrame,
    config: ProjectConfig,
) -> dict[str, str]:
    suffixes = config.raw["data"]["continuous_exchange_suffix"]
    product_exchange = contracts[["product", "exchange"]].drop_duplicates()
    if product_exchange["product"].duplicated().any():
        raise ValueError("one product maps to multiple exchanges")
    return {
        f"{row.product}.{suffixes[row.exchange]}": row.product
        for row in product_exchange.itertuples(index=False)
    }


def _read_continuous(
    contracts: pd.DataFrame,
    config: ProjectConfig,
) -> pd.DataFrame:
    code_map = _continuous_code_map(contracts, config)
    files = config.continuous_daily_files
    if not files:
        raise FileNotFoundError(
            f"no continuous daily files match {config.raw['data']['continuous_daily_glob']}"
        )
    frames: list[pd.DataFrame] = []
    for path in files:
        part = pd.read_csv(
            path,
            usecols=CONTINUOUS_COLUMNS,
            encoding="utf-8-sig",
            low_memory=False,
        )
        part = part.loc[part["ts_code"].isin(code_map)].copy()
        if part.empty:
            continue
        part["product"] = part["ts_code"].map(code_map)
        part["trade_date"] = pd.to_datetime(
            part["trade_date"].astype("Int64").astype(str),
            format="%Y%m%d",
            errors="coerce",
        )
        frames.append(part)
    if not frames:
        raise ValueError("continuous source contains none of the required exact codes")
    continuous = pd.concat(frames, ignore_index=True)
    continuous = continuous.rename(columns={"vol": "volume", "oi": "open_interest"})
    if continuous.duplicated(["trade_date", "product"]).any():
        duplicates = int(continuous.duplicated(["trade_date", "product"]).sum())
        raise ValueError(f"continuous source has {duplicates} duplicate product-days")
    return continuous.sort_values(["product", "trade_date"]).reset_index(drop=True)


def _mapping_scores(
    candidates: pd.DataFrame,
    match_fields: list[str],
) -> pd.DataFrame:
    scored = candidates.copy()
    score = np.zeros(len(scored), dtype=np.int16)
    comparable = np.zeros(len(scored), dtype=np.int16)
    for field in match_fields:
        continuous_values = scored[f"continuous_{field}"].to_numpy(dtype=float)
        actual_values = scored[field].to_numpy(dtype=float)
        valid = np.isfinite(continuous_values) & np.isfinite(actual_values)
        equal = np.isclose(
            continuous_values,
            actual_values,
            rtol=1e-10,
            atol=1e-8,
            equal_nan=False,
        )
        comparable += valid.astype(np.int16)
        score += (valid & equal).astype(np.int16)
    scored["mapping_comparable_fields"] = comparable
    scored["mapping_score"] = score
    return scored


def _select_mapped_contracts(
    contracts: pd.DataFrame,
    continuous: pd.DataFrame,
    config: ProjectConfig,
) -> pd.DataFrame:
    match_fields = list(config.raw["main_contract"]["mapping_match_fields"])
    continuous_fields = [
        "trade_date",
        "product",
        *match_fields,
    ]
    renamed = continuous[continuous_fields].rename(
        columns={field: f"continuous_{field}" for field in match_fields}
    )
    candidates = contracts.merge(
        renamed,
        on=["trade_date", "product"],
        how="inner",
        validate="many_to_one",
    )
    candidates = _mapping_scores(candidates, match_fields)
    selected_indices: list[int] = []
    previous_contract: dict[str, str] = {}
    previous_delist: dict[str, pd.Timestamp] = {}
    prevent_backtracking = bool(
        config.raw["main_contract"]["prohibit_maturity_backtracking"]
    )

    for (product, _), group in candidates.groupby(
        ["product", "trade_date"],
        sort=True,
    ):
        best_score = int(group["mapping_score"].max())
        top = group.loc[group["mapping_score"].eq(best_score)].copy()
        prior_code = previous_contract.get(product)
        if prior_code is not None and top["ts_code"].eq(prior_code).any():
            chosen = top.loc[top["ts_code"].eq(prior_code)].iloc[0]
        else:
            if prevent_backtracking and product in previous_delist:
                forward = top.loc[top["delist_date"].ge(previous_delist[product])]
                if not forward.empty:
                    top = forward
            chosen = top.sort_values(
                ["open_interest", "volume", "delist_date", "ts_code"],
                ascending=[False, False, False, True],
                na_position="last",
            ).iloc[0]
        selected_indices.append(int(chosen.name))
        previous_contract[product] = str(chosen["ts_code"])
        previous_delist[product] = pd.Timestamp(chosen["delist_date"])

    selected = candidates.loc[selected_indices].copy()
    selected = selected.rename(columns={"ts_code": "mapped_actual_ts_code"})
    selected["mapping_quality"] = np.select(
        [
            selected["mapping_score"].eq(selected["mapping_comparable_fields"])
            & selected["mapping_comparable_fields"].ge(5),
            selected["mapping_score"].ge(5),
        ],
        ["exact_current_fields", "high_score_current_fields"],
        default="fallback_current_fields",
    )
    selected["selection_signal_date"] = selected["trade_date"]
    selected["main_mapping_time_verified"] = True
    selected = selected.sort_values(["product", "trade_date"]).reset_index(drop=True)
    return selected


def _curve_summary(
    contracts: pd.DataFrame,
    config: ProjectConfig,
) -> pd.DataFrame:
    curve = contracts.copy()
    curve_config = config.raw["curve"]
    group_keys = ["trade_date", "product"]
    total_oi = curve["open_interest"].clip(lower=0).groupby(
        [curve["trade_date"], curve["product"]],
        sort=False,
    ).transform("sum")
    total_volume = curve["volume"].clip(lower=0).groupby(
        [curve["trade_date"], curve["product"]],
        sort=False,
    ).transform("sum")
    curve["oi_share"] = curve["open_interest"].clip(lower=0).div(
        total_oi.where(total_oi.gt(0))
    )
    curve["volume_share"] = curve["volume"].clip(lower=0).div(
        total_volume.where(total_volume.gt(0))
    )
    eligible = (
        curve["close"].gt(0)
        & curve["settle"].gt(0)
        & curve["days_to_delist"].between(
            int(curve_config["minimum_days_to_delist"]),
            int(curve_config["maximum_days_to_delist"]),
        )
        & (curve["volume"].gt(0) | curve["open_interest"].gt(0))
        & (
            curve["oi_share"].ge(float(curve_config["minimum_open_interest_share"]))
            | curve["volume_share"].ge(float(curve_config["minimum_volume_share"]))
        )
    )
    curve = curve.loc[eligible].copy()
    curve = curve.sort_values([*group_keys, "days_to_delist", "ts_code"])
    curve["curve_position"] = curve.groupby(group_keys, sort=False).cumcount() + 1
    curve["curve_contract_count"] = curve.groupby(group_keys, sort=False)[
        "ts_code"
    ].transform("size")
    curve["maturity_years"] = curve["days_to_delist"] / 365.0
    curve["log_curve_price"] = np.log(curve["close"])
    curve["curve_weight"] = np.sqrt(curve["open_interest"].clip(lower=1.0))

    first_three = curve.loc[curve["curve_position"].le(3)].copy()
    wide_parts: list[pd.DataFrame] = []
    labels = {1: "near", 2: "middle", 3: "far"}
    for position, label in labels.items():
        part = first_three.loc[
            first_three["curve_position"].eq(position),
            [*group_keys, "ts_code", "close", "days_to_delist"],
        ].rename(
            columns={
                "ts_code": f"curve_{label}_contract",
                "close": f"curve_{label}_close",
                "days_to_delist": f"curve_{label}_days",
            }
        )
        wide_parts.append(part)
    summary = curve[group_keys + ["curve_contract_count"]].drop_duplicates(group_keys)
    for part in wide_parts:
        summary = summary.merge(part, on=group_keys, how="left", validate="one_to_one")

    grouped = curve.groupby(group_keys, sort=False)
    curve["w"] = curve["curve_weight"]
    curve["wx"] = curve["w"] * curve["maturity_years"]
    curve["wy"] = curve["w"] * curve["log_curve_price"]
    curve["wxx"] = curve["w"] * curve["maturity_years"].pow(2)
    curve["wxy"] = curve["w"] * curve["maturity_years"] * curve["log_curve_price"]
    sums = grouped[["w", "wx", "wy", "wxx", "wxy"]].sum().reset_index()
    denominator = sums["wxx"] - sums["wx"].pow(2).div(sums["w"])
    slope = (
        sums["wxy"] - sums["wx"].mul(sums["wy"]).div(sums["w"])
    ).div(denominator.where(denominator.gt(0)))
    counts = summary[group_keys + ["curve_contract_count"]]
    sums = sums.merge(counts, on=group_keys, how="left", validate="one_to_one")
    sums["term_structure_slope"] = (-slope).where(
        sums["curve_contract_count"].ge(
            int(curve_config["minimum_points_for_slope"])
        )
    )
    summary = summary.merge(
        sums[group_keys + ["term_structure_slope"]],
        on=group_keys,
        how="left",
        validate="one_to_one",
    )

    near = summary["curve_near_close"]
    middle = summary["curve_middle_close"]
    far = summary["curve_far_close"]
    near_days = summary["curve_near_days"]
    middle_days = summary["curve_middle_days"]
    far_days = summary["curve_far_days"]
    gap = middle_days - near_days
    summary["carry_annualized"] = (
        np.log(near / middle) * 365.0 / gap.where(gap.gt(0))
    )
    interpolation_weight = (middle_days - near_days).div(
        (far_days - near_days).where(far_days.gt(near_days))
    )
    expected_middle = np.log(near) + interpolation_weight * (
        np.log(far) - np.log(near)
    )
    summary["curve_curvature"] = 2.0 * (
        np.log(middle) - expected_middle
    )
    return summary


def _add_tradable_returns(
    panel: pd.DataFrame,
    contracts: pd.DataFrame,
) -> pd.DataFrame:
    out = panel.sort_values(["product", "trade_date"]).copy()
    grouped = out.groupby("product", sort=False)
    previous_close = grouped["close"].shift(1)
    previous_contract = grouped["mapped_actual_ts_code"].shift(1)
    first = grouped.cumcount().eq(0)
    changed = out["mapped_actual_ts_code"].ne(previous_contract) & ~first
    out["main_contract_changed"] = changed.astype("int8")

    open_lookup = contracts.set_index(["trade_date", "ts_code"])["open"]
    old_keys = pd.MultiIndex.from_arrays(
        [out["trade_date"], previous_contract],
        names=["trade_date", "ts_code"],
    )
    out["old_contract_open_on_roll"] = open_lookup.reindex(old_keys).to_numpy()
    out["new_contract_open_on_roll"] = out["open"]

    same_contract = ~changed & ~first & previous_close.gt(0) & out["close"].gt(0)
    normal_return = out["close"] / previous_close - 1.0
    valid_roll = (
        changed
        & previous_close.gt(0)
        & out["old_contract_open_on_roll"].gt(0)
        & out["new_contract_open_on_roll"].gt(0)
        & out["close"].gt(0)
    )
    roll_return = (
        out["old_contract_open_on_roll"] / previous_close
        * out["close"] / out["new_contract_open_on_roll"]
        - 1.0
    )
    out["product_return_1d"] = np.select(
        [same_contract, valid_roll],
        [normal_return, roll_return],
        default=np.nan,
    )
    out["return_quality"] = np.select(
        [first, same_contract, valid_roll, changed],
        ["initialization", "same_contract", "roll_at_open", "roll_data_missing"],
        default="price_missing",
    )
    out["roll_return_available"] = (~changed | valid_roll).astype("int8")
    return out


def load_data_final(config: ProjectConfig) -> DataBundle:
    contracts = _read_contracts(config)
    continuous = _read_continuous(contracts, config)
    panel = _select_mapped_contracts(contracts, continuous, config)
    curve = _curve_summary(contracts, config)
    panel = panel.merge(
        curve,
        on=["trade_date", "product"],
        how="left",
        validate="one_to_one",
    )

    expected_start = pd.Timestamp(config.raw["sample"]["start"])
    expected_end = pd.Timestamp(config.raw["sample"]["end"])
    if panel["trade_date"].min() != expected_start or panel["trade_date"].max() != expected_end:
        raise ValueError("mapped panel sample dates do not match the frozen sample")
    if panel["product"].nunique() != 30:
        raise ValueError("mapped panel does not contain exactly 30 products")
    if panel.duplicated(["trade_date", "product"]).any():
        raise ValueError("duplicate trade_date-product keys in mapped panel")

    sector_lookup = _sector_lookup(config)
    if set(panel["product"]) != set(sector_lookup):
        raise ValueError("sector map and mapped products differ")
    panel["sector"] = panel["product"].map(sector_lookup)
    calendar = pd.Index(sorted(panel["trade_date"].unique()), name="trade_date")
    calendar_index = pd.Series(np.arange(len(calendar)), index=calendar)
    panel["calendar_index"] = panel["trade_date"].map(calendar_index).astype(int)
    panel = _add_tradable_returns(panel, contracts)

    observation_columns = [
        "spot_observation_date",
        "warehouse_receipt_observation_date",
        "stock_observation_date",
        "available_stock_observation_date",
    ]
    point_in_time_violations = {
        column: int(
            (
                contracts[column].notna()
                & contracts[column].gt(contracts["trade_date"])
            ).sum()
        )
        for column in observation_columns
    }
    source_audit = {
        "raw_contract_rows": int(len(contracts)),
        "raw_contracts": int(contracts["ts_code"].nunique()),
        "raw_product_days": int(
            contracts[["trade_date", "product"]].drop_duplicates().shape[0]
        ),
        "continuous_product_days": int(len(continuous)),
        "point_in_time_violations": point_in_time_violations,
        "spot_last_observation": contracts["spot_observation_date"].max(),
        "warehouse_last_observation": contracts[
            "warehouse_receipt_observation_date"
        ].max(),
        "stock_last_observation": contracts["stock_observation_date"].max(),
        "actual_fee_any_coverage": float(
            contracts[["trading_fee_rate", "trading_fee"]].notna().any(axis=1).mean()
        ),
    }
    return DataBundle(
        panel=panel,
        contracts=contracts,
        continuous=continuous,
        source_audit=source_audit,
    )


def data_audit(bundle: DataBundle, config: ProjectConfig) -> dict[str, Any]:
    panel = bundle.panel
    development_end = pd.Timestamp(config.raw["sample"]["development_end"])
    sealed_start = pd.Timestamp(config.raw["sample"]["sealed_start"])
    grouped = panel.groupby("product", sort=False)
    lag = int(config.raw["data"]["point_in_time"]["release_lag_trading_days"])

    def lagged_fresh(
        value_column: str,
        observation_column: str,
        maximum_age: int,
    ) -> pd.Series:
        values = grouped[value_column].shift(lag)
        observations = grouped[observation_column].shift(lag)
        age = (panel["trade_date"] - observations).dt.days
        return (
            values.notna()
            & observations.notna()
            & observations.lt(panel["trade_date"])
            & age.between(0, maximum_age)
        )

    spot_fresh = lagged_fresh(
        "spot_price",
        "spot_observation_date",
        int(config.raw["data"]["point_in_time"]["spot_max_age_calendar_days"]),
    ) & ~panel["product"].isin(config.raw["data"]["spot_unit_excluded_products"])
    warehouse_fresh = lagged_fresh(
        "warehouse_receipt_current",
        "warehouse_receipt_observation_date",
        int(config.raw["data"]["point_in_time"]["warehouse_max_age_calendar_days"]),
    )
    stock_fresh = lagged_fresh(
        "stock_current",
        "stock_observation_date",
        int(config.raw["data"]["point_in_time"]["stock_max_age_calendar_days"]),
    )
    return {
        **bundle.source_audit,
        "mapped_rows": int(len(panel)),
        "columns": int(len(panel.columns)),
        "date_min": panel["trade_date"].min().date().isoformat(),
        "date_max": panel["trade_date"].max().date().isoformat(),
        "products": int(panel["product"].nunique()),
        "trade_dates": int(panel["trade_date"].nunique()),
        "duplicate_keys": int(panel.duplicated(["trade_date", "product"]).sum()),
        "development_rows": int(panel["trade_date"].le(development_end).sum()),
        "sealed_2025_rows": int(panel["trade_date"].ge(sealed_start).sum()),
        "mapped_actual_contract_coverage": float(
            panel["mapped_actual_ts_code"].notna().mean()
        ),
        "mapping_exact_rate": float(
            panel["mapping_quality"].eq("exact_current_fields").mean()
        ),
        "mapping_high_or_exact_rate": float(
            panel["mapping_quality"].isin(
                ["exact_current_fields", "high_score_current_fields"]
            ).mean()
        ),
        "mapping_min_score": int(panel["mapping_score"].min()),
        "mapping_signal_time_coverage": float(
            panel["selection_signal_date"].notna().mean()
        ),
        "product_return_coverage": float(panel["product_return_1d"].notna().mean()),
        "roll_rows": int(panel["main_contract_changed"].eq(1).sum()),
        "roll_return_missing_rows": int(
            (
                panel["main_contract_changed"].eq(1)
                & panel["product_return_1d"].isna()
            ).sum()
        ),
        "suspected_limit_rows": int(panel["suspected_price_limit_lock"].eq(1).sum()),
        "curve_two_point_coverage": float(
            panel["curve_contract_count"].ge(2).mean()
        ),
        "curve_three_point_coverage": float(
            panel["curve_contract_count"].ge(3).mean()
        ),
        "curve_median_contracts": float(panel["curve_contract_count"].median()),
        "spot_fresh_usable_coverage": float(spot_fresh.mean()),
        "warehouse_fresh_usable_coverage": float(warehouse_fresh.mean()),
        "stock_fresh_usable_coverage": float(stock_fresh.mean()),
        "stock_products": int(panel.loc[stock_fresh, "product"].nunique()),
        "main_contract_rule": config.raw["main_contract"]["rule"],
        "t_minus_1_selection_required": False,
        "fundamental_release_lag_trading_days": int(
            config.raw["data"]["point_in_time"]["release_lag_trading_days"]
        ),
    }
