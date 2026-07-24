from __future__ import annotations

import numpy as np
import pandas as pd


def _lookup_contract_field(
    contracts: pd.DataFrame,
    dates: pd.Series,
    codes: pd.Series,
    field: str,
) -> np.ndarray:
    lookup = contracts.set_index(["trade_date", "ts_code"])[field]
    keys = pd.MultiIndex.from_arrays(
        [dates, codes],
        names=["trade_date", "ts_code"],
    )
    return lookup.reindex(keys).to_numpy()


def add_forward_labels(
    panel: pd.DataFrame,
    contracts: pd.DataFrame,
    horizons: list[int],
    exclude_suspected_locks: bool = True,
) -> pd.DataFrame:
    """Create fixed-actual-contract T+1-open to T+h+1-open labels."""
    out = panel.copy()
    calendar = pd.Series(
        sorted(out["trade_date"].unique()),
        index=np.arange(out["trade_date"].nunique()),
    )
    signal_index = out["calendar_index"].to_numpy(dtype=int)
    maximum_index = int(calendar.index.max())
    codes = out["mapped_actual_ts_code"]

    entry_indices = signal_index + 1
    entry_dates = pd.Series(
        pd.to_datetime(
            [
                calendar.get(index, pd.NaT)
                if index <= maximum_index
                else pd.NaT
                for index in entry_indices
            ]
        ),
        index=out.index,
    )
    entry_open = _lookup_contract_field(
        contracts,
        entry_dates,
        codes,
        "open",
    )
    entry_volume = _lookup_contract_field(
        contracts,
        entry_dates,
        codes,
        "volume",
    )
    entry_lock = _lookup_contract_field(
        contracts,
        entry_dates,
        codes,
        "suspected_price_limit_lock",
    )
    entry_direction = _lookup_contract_field(
        contracts,
        entry_dates,
        codes,
        "suspected_price_limit_direction",
    )
    entry_available = (
        np.isfinite(entry_open)
        & (entry_open > 0)
        & np.isfinite(entry_volume)
        & (entry_volume > 0)
    )
    if exclude_suspected_locks:
        entry_available &= np.nan_to_num(entry_lock, nan=0.0) == 0
    out["entry_date"] = entry_dates
    out["entry_execution_available"] = entry_available.astype("int8")
    out["entry_suspected_limit_lock"] = (
        pd.Series(entry_lock, index=out.index).fillna(0).astype("int8")
    )
    out["entry_suspected_limit_direction"] = entry_direction

    for horizon in horizons:
        exit_indices = signal_index + horizon + 1
        exit_dates = pd.Series(
            pd.to_datetime(
                [
                    calendar.get(index, pd.NaT)
                    if index <= maximum_index
                    else pd.NaT
                    for index in exit_indices
                ]
            ),
            index=out.index,
        )
        exit_open = _lookup_contract_field(
            contracts,
            exit_dates,
            codes,
            "open",
        )
        exit_volume = _lookup_contract_field(
            contracts,
            exit_dates,
            codes,
            "volume",
        )
        exit_lock = _lookup_contract_field(
            contracts,
            exit_dates,
            codes,
            "suspected_price_limit_lock",
        )
        exit_direction = _lookup_contract_field(
            contracts,
            exit_dates,
            codes,
            "suspected_price_limit_direction",
        )
        exit_available = (
            np.isfinite(exit_open)
            & (exit_open > 0)
            & np.isfinite(exit_volume)
            & (exit_volume > 0)
        )
        if exclude_suspected_locks:
            exit_available &= np.nan_to_num(exit_lock, nan=0.0) == 0
        complete = entry_available & exit_available & exit_dates.notna().to_numpy()
        label = exit_open / entry_open - 1.0
        out[f"future_return_{horizon}d"] = pd.Series(
            label,
            index=out.index,
        ).where(complete)
        out[f"label_end_date_{horizon}d"] = exit_dates.where(complete)
        out[f"exit_execution_available_{horizon}d"] = exit_available.astype("int8")
        out[f"exit_suspected_limit_lock_{horizon}d"] = (
            pd.Series(exit_lock, index=out.index).fillna(0).astype("int8")
        )
        out[f"exit_suspected_limit_direction_{horizon}d"] = exit_direction
    return out


def assert_forward_labels(panel: pd.DataFrame, horizons: list[int]) -> None:
    for horizon in horizons:
        label = f"future_return_{horizon}d"
        end = f"label_end_date_{horizon}d"
        valid = panel[label].notna()
        if not panel.loc[valid, end].gt(panel.loc[valid, "trade_date"]).all():
            raise AssertionError(f"{label} is not strictly forward")
        if not (
            panel.loc[valid, "entry_date"].gt(panel.loc[valid, "trade_date"])
        ).all():
            raise AssertionError(f"{label} entry is not after signal time")
        expected_steps = horizon + 1
        actual_steps = (
            panel.loc[valid, end]
            .map(
                pd.Series(
                    np.arange(panel["trade_date"].nunique()),
                    index=sorted(panel["trade_date"].unique()),
                )
            )
            - panel.loc[valid, "calendar_index"]
        )
        if not actual_steps.eq(expected_steps).all():
            raise AssertionError(f"{label} does not use the frozen open-to-open horizon")
