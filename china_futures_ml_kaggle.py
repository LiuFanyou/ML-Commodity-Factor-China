"""
中国商品期货机器学习因子投资：扩展窗口验证与可行性评估
=======================================================

本地/Kaggle 通用脚本。核心设计：
1) 因果主力合约与近远月期限结构；
2) 原始OHLC相对价格、收盘收益、量仓变化率与商品展期收益；
3) 20日原始量价序列输入 Conv1d + Self-Attention 双头网络；
4) 预测未来 5 个交易日累计截面超额收益；
5) 2018年起的 Expanding-Window：2018-2021 -> 2022，随后逐年扩展；
6) 预测值=上涨概率×回归幅度，经3日EMA后按动态分位/零阈值交易；
7) 输出训练诊断、净值、成本敏感性、因子重要性、模型和审计文件。

时间定义：因子在决策日 t 收盘后可得，t+1 开盘执行。5 日标签覆盖
从 t+1 开盘开始的五个交易持有期；实际回测仍逐日按 open-to-open 收益记账，
避免把重叠的 5 日标签错误当成每日组合收益。
"""

from __future__ import annotations

import copy
import json
import math
import os
import random
import re
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

try:
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler
except ImportError as exc:
    raise ImportError("GMM 压力风控需要 scikit-learn，请运行: pip install scikit-learn") from exc

try:
    from tqdm.auto import tqdm
except ImportError:  # 无 tqdm 时仍可运行，并保留阶段 print
    def tqdm(iterable, *args, **kwargs):
        return iterable


# =============================================================================
# 0. 配置：Kaggle 中可通过环境变量 FUTURES_DATA_ROOT 指定数据集目录；
#    未指定时自动递归搜索 /kaggle/input。
# =============================================================================


@dataclass
class Config:
    # 默认使用你的本地数据目录。需要切换 Kaggle 时，可在运行前设置
    # 环境变量 FUTURES_DATA_ROOT=/kaggle/input/data-2015-2025。
    data_root: Optional[str] = os.getenv(
        "FUTURES_DATA_ROOT",
        r"D:\大学\大二下\财大量化\tushare_theme7\tushare_theme7",
    )
    local_windows_data_root: str = r"D:\大学\大二下\财大量化\tushare_theme7\tushare_theme7"
    output_dir: str = (
        "/kaggle/working/futures_ml_outputs"
        if Path("/kaggle/working").exists()
        else str(Path.cwd() / "futures_ml_outputs")
    )

    # 固定的代表性基础池，避免利用全样本未来流动性进行事后选品。
    # 28 个品种覆盖黑色、能化、有色、贵金属、油脂油料和农产品。
    products: Tuple[str, ...] = (
        "RB", "HC", "I", "J", "JM",
        "TA", "MA", "SA", "PP", "L", "V", "EG", "RU", "BU",
        "CU", "AL", "ZN", "NI", "AU", "AG",
        "M", "Y", "P", "C", "CF", "SR", "RM", "OI",
    )
    min_cross_section: int = 10
    min_days_to_expiry: int = 7
    roll_switch_ratio: float = 1.10

    factor_window: int = 20
    sequence_length: int = 20
    burn_in_days: int = 90  # 应位于 60~120 个交易日范围内
    mad_n: float = 3.0

    data_start_year: int = 2018
    data_end_year: int = 2025

    # Expanding Window：2018-2021 训练，2022 首个样本外测试，随后逐年扩展。
    train_start_year: int = 2018
    first_test_year: int = 2022
    last_test_year: int = 2025
    forecast_horizon: int = 5
    validation_fraction: float = 0.15
    purge_days: int = 25  # 20日重叠序列 + 5日标签的保护间隔

    # 卷积局部嵌入 + 单层轻量 Transformer。
    model_dim: int = 48
    attention_heads: int = 4
    transformer_ff_dim: int = 96
    transformer_layers: int = 1
    dropout: float = 0.10
    multitask_alpha: float = 0.70
    train_date_stride: int = 5
    batch_size: int = 512
    max_epochs: int = 40
    patience: int = 7
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    huber_delta: float = 1.0
    num_workers: int = 0

    signal_ema_span: int = 3
    entry_quantile: float = 0.70  # 做多前30%，做空后30%
    absolute_signal_threshold: float = 0.0
    weighting_method: str = "signal"  # 当前实现：预测强度加权
    stress_threshold: float = 0.80
    stress_leverage_multiplier: float = 0.50
    gmm_components: int = 2
    gmm_max_iter: int = 100
    gmm_min_history_days: int = 120
    one_way_fees: Tuple[float, ...] = (0.0001, 0.0003, 0.0005)
    base_fee: float = 0.0003
    annualization: int = 252
    liquidate_at_end: bool = False

    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


CFG = Config()
# 原始日频量价特征 + 商品期限结构；全部按 trade_date 截面 MAD/Z-Score。
FACTOR_COLS = [
    "log_open_close",
    "log_high_close",
    "log_low_close",
    "log_close_prev_close",
    "volume_change",
    "open_interest_change",
    "roll_yield",
]


# =============================================================================
# 1. 通用工具、数据路径与字段适配
# =============================================================================


ALIASES: Dict[str, Tuple[str, ...]] = {
    "contract": ("ts_code", "contract", "symbol", "instrument", "instrumentid", "code"),
    "trade_date": ("trade_date", "date", "datetime", "trading_day", "tradingdate"),
    "open": ("open", "open_price", "openprice"),
    "high": ("high", "high_price", "highestprice"),
    "low": ("low", "low_price", "lowestprice"),
    "close": ("close", "close_price", "closeprice"),
    "settle": ("settle", "settlement", "settle_price", "settlementprice"),
    "volume": ("vol", "volume", "trade_volume", "tradingvolume"),
    "open_interest": ("oi", "open_interest", "openinterest", "position"),
    "amount": ("amount", "turnover", "turnover_value"),
    "product": ("fut_code", "product", "product_code", "variety"),
    "exchange": ("exchange", "exch", "market"),
    "list_date": ("list_date", "listed_date", "listdate"),
    "delist_date": ("delist_date", "last_trade_date", "expire_date", "delistdate"),
    "delivery_month": ("d_month", "delivery_month", "maturity_month"),
    "direct_roll_yield": ("roll_yield", "rollyield", "carry"),
    "basis": ("basis", "basis_rate", "basisrate"),
    "spot": ("spot", "spot_price", "spotprice"),
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def normalize_column_name(name: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "", str(name).strip().lower().replace(" ", "_"))


def read_csv_header(path: Path) -> Optional[List[str]]:
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return list(pd.read_csv(path, nrows=0, encoding=encoding).columns)
        except Exception:
            continue
    return None


def source_to_canonical(columns: Iterable[str]) -> Dict[str, str]:
    normalized_to_source = {normalize_column_name(c): c for c in columns}
    mapping: Dict[str, str] = {}
    for canonical, aliases in ALIASES.items():
        for alias in aliases:
            if alias in normalized_to_source:
                mapping[normalized_to_source[alias]] = canonical
                break
    return mapping


def read_csv_adaptive(path: Path, required: Sequence[str]) -> Optional[pd.DataFrame]:
    header = read_csv_header(path)
    if not header:
        return None
    mapping = source_to_canonical(header)
    available = set(mapping.values())
    if not set(required).issubset(available):
        return None
    usecols = list(mapping.keys())
    last_error: Optional[Exception] = None
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            frame = pd.read_csv(path, usecols=usecols, encoding=encoding, low_memory=False)
            frame = frame.rename(columns=mapping)
            frame = frame.loc[:, ~frame.columns.duplicated()]
            frame["_source_file"] = str(path)
            return frame
        except Exception as exc:
            last_error = exc
    warnings.warn(f"跳过无法读取的 CSV: {path}; 原因: {last_error}")
    return None


def resolve_data_root(cfg: Config) -> Path:
    candidates: List[Path] = []
    if cfg.data_root:
        configured = Path(cfg.data_root)
        candidates.append(configured)

        # 用户有时会复制 Kaggle Dataset 网页式路径：
        # /kaggle/input/datasets/<owner>/<slug>。
        # Notebook 内真正的挂载路径通常是 /kaggle/input/<slug>。
        parts = configured.parts
        if "datasets" in parts:
            dataset_pos = parts.index("datasets")
            if len(parts) > dataset_pos + 2:
                slug = parts[dataset_pos + 2]
                candidates.append(Path("/kaggle/input") / slug)
        elif str(configured).startswith("/kaggle/input/"):
            candidates.append(Path("/kaggle/input") / configured.name)

    candidates.append(Path(cfg.local_windows_data_root))

    # 保序去重，优先采用用户明确配置的有效路径。
    unique_candidates = list(dict.fromkeys(candidates))
    for candidate in unique_candidates:
        if candidate.exists():
            if cfg.data_root and candidate != Path(cfg.data_root):
                warnings.warn(f"配置路径未直接挂载，已自动改用 Kaggle 实际路径：{candidate}")
            return candidate

    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        basic_files = list(kaggle_input.rglob("fut_basic_all.csv"))
        if basic_files:
            # 选择同时最接近年度文件的父目录。
            return basic_files[0].parent
        csv_files = list(kaggle_input.rglob("*.csv"))
        if csv_files:
            warnings.warn("未找到 fut_basic_all.csv，将使用 /kaggle/input 并依赖合约代码解析品种。")
            return kaggle_input

    raise FileNotFoundError(
        "找不到数据。Kaggle 请把数据作为 Dataset 添加，并设置环境变量 "
        "FUTURES_DATA_ROOT=/kaggle/input/<dataset-name>；也可保持为空让脚本自动搜索。"
    )


def find_named_file(root: Path, filename: str) -> Optional[Path]:
    exact = root / filename
    if exact.exists():
        return exact
    matches = list(root.rglob(filename))
    return matches[0] if matches else None


def parse_product_from_contract(contract: pd.Series) -> pd.Series:
    return (
        contract.astype(str)
        .str.upper()
        .str.extract(r"^([A-Z]+)", expand=False)
        .replace({"ME": "MA"})  # 郑商所甲醇历史代码的常见衔接
    )


def parse_date(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
    return pd.to_datetime(text, errors="coerce")


def load_basic(root: Path) -> pd.DataFrame:
    path = find_named_file(root, "fut_basic_all.csv")
    if path is None:
        warnings.warn("缺少 fut_basic_all.csv：到期日将从合约代码推断，期限结构可靠性降低。")
        return pd.DataFrame(columns=["contract", "product", "exchange", "list_date", "delist_date"])
    basic = read_csv_adaptive(path, required=["contract"])
    if basic is None:
        raise ValueError(f"无法识别 {path} 中的合约代码字段。")
    basic["contract"] = basic["contract"].astype(str).str.strip().str.upper()
    if "product" not in basic:
        basic["product"] = parse_product_from_contract(basic["contract"])
    else:
        basic["product"] = basic["product"].astype(str).str.upper().replace({"ME": "MA"})
    for col in ("list_date", "delist_date"):
        basic[col] = parse_date(basic[col]) if col in basic else pd.NaT
    keep = [c for c in ["contract", "product", "exchange", "list_date", "delist_date", "delivery_month"] if c in basic]
    return basic[keep].drop_duplicates("contract", keep="last")


def year_from_path(path: Path) -> Optional[int]:
    """从目录名或文件名提取 4 位年份，用于读 CSV 前直接瘦身。"""
    for part in reversed(path.parts):
        match = re.search(r"(?<!\d)(20\d{2})(?!\d)", part)
        if match:
            return int(match.group(1))
    return None


def discover_market_files(root: Path, start_year: int, end_year: int) -> List[Path]:
    """只发现目标年份的完整日行情文件，避免读取无关年度和逐日结算参数。"""
    files: List[Path] = []
    for path in tqdm(list(root.rglob("*.csv")), desc="[Data] 扫描CSV表头", unit="文件"):
        relative_parts = [part.lower() for part in path.relative_to(root).parts]
        if path.name.lower() == "fut_basic_all.csv" or "fut_settle_by_date" in relative_parts:
            continue
        path_year = year_from_path(path)
        if path_year is not None and not start_year <= path_year <= end_year:
            continue
        header = read_csv_header(path)
        if not header:
            continue
        canon = set(source_to_canonical(header).values())
        if {"contract", "trade_date", "close"}.issubset(canon):
            files.append(path)
    if not files:
        raise FileNotFoundError("没有发现同时含合约、交易日和收盘价字段的行情 CSV。")
    return sorted(files)


def load_market_data(root: Path, cfg: Config) -> Tuple[pd.DataFrame, Dict[str, object]]:
    files = discover_market_files(root, cfg.data_start_year, cfg.data_end_year)
    print(
        f"[Data] 仅加载 {cfg.data_start_year}-{cfg.data_end_year}："
        f"发现 {len(files)} 个完整日行情文件，开始读取……",
        flush=True,
    )
    frames: List[pd.DataFrame] = []
    for i, path in enumerate(tqdm(files, desc="[Data] 读取日行情", unit="文件"), 1):
        frame = read_csv_adaptive(path, required=["contract", "trade_date", "close"])
        if frame is not None and not frame.empty:
            frames.append(frame)
    if not frames:
        raise ValueError("行情文件存在，但没有成功读取任何数据。")

    data = pd.concat(frames, ignore_index=True, sort=False)
    data["contract"] = data["contract"].astype(str).str.strip().str.upper()
    data["trade_date"] = parse_date(data["trade_date"])
    data = data.dropna(subset=["contract", "trade_date", "close"])
    data = data[data["trade_date"].dt.year.between(cfg.data_start_year, cfg.data_end_year)].copy()

    if "product" not in data:
        data["product"] = parse_product_from_contract(data["contract"])
    else:
        inferred = parse_product_from_contract(data["contract"])
        data["product"] = data["product"].astype(str).str.upper().replace({"ME": "MA"})
        data["product"] = data["product"].where(data["product"].ne("NAN"), inferred)

    numeric_cols = [
        "open", "high", "low", "close", "settle", "volume", "open_interest",
        "amount", "direct_roll_yield", "basis", "spot",
    ]
    for col in numeric_cols:
        if col in data:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    # 某些精简数据只有 close；允许降级运行，但记录执行价近似。
    execution_price_fallback = "open" not in data or data["open"].notna().mean() < 0.5
    if "open" not in data:
        data["open"] = data["close"]
    else:
        data["open"] = data["open"].fillna(data["close"])
    for col in ("high", "low", "settle"):
        if col not in data:
            data[col] = data["close"]
    for col in ("volume", "open_interest"):
        if col not in data:
            data[col] = np.nan

    before = len(data)
    data = data.sort_values(["contract", "trade_date", "_source_file"])
    duplicate_mask = data.duplicated(["contract", "trade_date"], keep=False)
    duplicate_rows = int(duplicate_mask.sum())
    data = data.drop_duplicates(["contract", "trade_date"], keep="last")
    data = data[data["product"].isin(cfg.products)].copy()
    data = data[(data["close"] > 0) & (data["open"] > 0)]

    report = {
        "market_files": len(files),
        "data_years": [cfg.data_start_year, cfg.data_end_year],
        "raw_rows": before,
        "duplicate_rows_seen": duplicate_rows,
        "filtered_rows": len(data),
        "products_found": sorted(data["product"].dropna().unique().tolist()),
        "execution_price_fallback_close": bool(execution_price_fallback),
    }
    if execution_price_fallback:
        warnings.warn("open 字段缺失较多：脚本使用 close 近似执行价；此版本仅适合信号验证。")
    return data, report


# =============================================================================
# 2. 到期日、因果主力合约与期限结构
# =============================================================================


def infer_expiry_from_code(contract: str, trade_date: pd.Timestamp) -> pd.Timestamp:
    core = str(contract).split(".")[0].upper()
    match = re.search(r"(\d{3,4})$", core)
    if not match or pd.isna(trade_date):
        return pd.NaT
    digits = match.group(1)
    try:
        if len(digits) == 4:
            year, month = 2000 + int(digits[:2]), int(digits[2:])
        else:  # 郑商所常见 yMM，结合交易年份推断最接近的交割十年
            year_digit, month = int(digits[0]), int(digits[1:])
            candidates = [y for y in range(trade_date.year - 2, trade_date.year + 9) if y % 10 == year_digit]
            year = min(candidates, key=lambda y: abs((y * 12 + month) - (trade_date.year * 12 + trade_date.month)))
        if not 1 <= month <= 12:
            return pd.NaT
        return pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)
    except Exception:
        return pd.NaT


def attach_metadata_and_returns(market: pd.DataFrame, basic: pd.DataFrame) -> pd.DataFrame:
    data = market.copy()
    if not basic.empty:
        meta = basic.rename(columns={"product": "meta_product"})
        data = data.merge(meta, on="contract", how="left", suffixes=("", "_meta"))
        if "meta_product" in data:
            data["product"] = data["meta_product"].fillna(data["product"])
        if "exchange_meta" in data and "exchange" not in data:
            data["exchange"] = data["exchange_meta"]

    if "delist_date" not in data:
        data["delist_date"] = pd.NaT
    data["delist_date"] = pd.to_datetime(data["delist_date"], errors="coerce")
    missing_expiry = data["delist_date"].isna()
    if missing_expiry.any():
        data.loc[missing_expiry, "delist_date"] = [
            infer_expiry_from_code(c, d)
            for c, d in zip(data.loc[missing_expiry, "contract"], data.loc[missing_expiry, "trade_date"])
        ]

    data = data.sort_values(["contract", "trade_date"]).copy()
    data["contract_close_return"] = data.groupby("contract", sort=False)["close"].pct_change(fill_method=None)
    next_open = data.groupby("contract", sort=False)["open"].shift(-1)
    next_date = data.groupby("contract", sort=False)["trade_date"].shift(-1)
    data["contract_fwd_open_return"] = next_open / data["open"] - 1.0
    # 周末可跨越，但异常长缺口不能当作一日收益。
    gap_days = (next_date - data["trade_date"]).dt.days
    data.loc[gap_days > 7, "contract_fwd_open_return"] = np.nan

    volume = data["volume"].fillna(0).clip(lower=0)
    oi = data["open_interest"].fillna(0).clip(lower=0)
    data["activity"] = oi + 0.25 * volume
    data["lag_activity"] = data.groupby("contract", sort=False)["activity"].shift(1)
    return data


def choose_main_contracts(data: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """状态化主力选择：只用滞后活跃度，设换月门槛，并禁止滚回更近月份。"""
    selected: List[Tuple[str, pd.Timestamp, str]] = []
    total_products = int(data["product"].nunique())
    for product, product_df in tqdm(data.groupby("product", sort=True), desc="[Main] 因果主力合约", total=total_products, unit="品种"):
        current_contract: Optional[str] = None
        current_expiry = pd.NaT
        for date, day in product_df.groupby("trade_date", sort=True):
            candidates = day[
                day["lag_activity"].gt(0)
                & day["close"].gt(0)
                & (day["delist_date"].isna() | (day["delist_date"] > date + pd.Timedelta(days=cfg.min_days_to_expiry)))
            ].copy()
            if candidates.empty:
                continue
            candidates = candidates.sort_values(["lag_activity", "delist_date"], ascending=[False, True])

            current_row = candidates[candidates["contract"].eq(current_contract)] if current_contract else candidates.iloc[0:0]
            must_roll = current_row.empty
            if not current_row.empty and pd.notna(current_expiry):
                must_roll = current_expiry <= date + pd.Timedelta(days=cfg.min_days_to_expiry)

            if must_roll:
                if pd.notna(current_expiry):
                    later = candidates[candidates["delist_date"].ge(current_expiry)]
                    chosen = later.iloc[0] if not later.empty else candidates.iloc[0]
                else:
                    chosen = candidates.iloc[0]
            else:
                chosen = current_row.iloc[0]
                current_score = float(chosen["lag_activity"])
                later = candidates[
                    candidates["delist_date"].gt(current_expiry)
                    & candidates["lag_activity"].gt(current_score * cfg.roll_switch_ratio)
                ]
                if not later.empty:
                    chosen = later.sort_values(["delist_date", "lag_activity"], ascending=[True, False]).iloc[0]

            current_contract = str(chosen["contract"])
            current_expiry = chosen["delist_date"]
            selected.append((product, date, current_contract))

    return pd.DataFrame(selected, columns=["product", "trade_date", "contract"])


def compute_roll_yield_for_day(day: pd.DataFrame, near_row: pd.Series) -> Tuple[float, str]:
    """优先同日近远月；再用已有 roll/basis；最后 settle/close 代理并明确标记。"""
    near_expiry = near_row.get("delist_date", pd.NaT)
    near_price = near_row.get("settle", np.nan)
    if not np.isfinite(near_price) or near_price <= 0:
        near_price = near_row.get("close", np.nan)

    if pd.notna(near_expiry) and np.isfinite(near_price) and near_price > 0:
        far = day[
            day["delist_date"].gt(near_expiry)
            & day["lag_activity"].gt(0)
            & day["settle"].fillna(day["close"]).gt(0)
        ].copy()
        if not far.empty:
            far = far.sort_values(["delist_date", "lag_activity"], ascending=[True, False]).iloc[0]
            far_price = far["settle"] if np.isfinite(far["settle"]) and far["settle"] > 0 else far["close"]
            days = (far["delist_date"] - near_expiry).days
            if 20 <= days <= 365 and far_price > 0:
                return float(np.log(near_price / far_price) * 365.0 / days), "near_far"

    direct = near_row.get("direct_roll_yield", np.nan)
    if np.isfinite(direct):
        return float(direct), "direct_field"

    spot = near_row.get("spot", np.nan)
    if np.isfinite(spot) and spot > 0 and near_price > 0:
        # 现货高于期货表示正 carry/backwardation。
        return float(np.log(spot / near_price)), "spot_basis"

    basis = near_row.get("basis", np.nan)
    if np.isfinite(basis):
        return float(basis), "basis_field"

    settle = near_row.get("settle", np.nan)
    close = near_row.get("close", np.nan)
    if np.isfinite(settle) and np.isfinite(close) and settle > 0 and close > 0 and settle != close:
        return float(np.log(settle / close)), "settle_close_proxy"
    return np.nan, "missing"


def build_product_panel(
    data: pd.DataFrame,
    selected: pd.DataFrame,
    forecast_horizon: int = 5,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """构造品种面板及未来 H 日累计标签。

    决策日 t 的信号在 t 收盘后形成，并于 t+1 开盘执行。target_raw 是从
    t+1 开盘开始连续 H 个 open-to-open 收益的几何累计值。若持有期内主力
    换月，则每一天使用当时因果选出的实际主力合约收益，避免跨合约价格跳空。
    """
    if forecast_horizon < 1:
        raise ValueError("forecast_horizon 必须至少为 1")
    key = selected.set_index(["product", "trade_date"])["contract"]
    rows: List[pd.Series] = []

    groups = data.groupby(["product", "trade_date"], sort=True)
    total_days = int(groups.ngroups)
    for (product, date), day in tqdm(groups, desc="[Panel] 主力与期限结构", total=total_days, unit="日"):
        try:
            contract = key.loc[(product, date)]
        except KeyError:
            continue
        near = day[day["contract"].eq(contract)]
        if near.empty:
            continue
        near_row = near.iloc[0].copy()
        roll, source = compute_roll_yield_for_day(day, near_row)
        near_row["roll_yield"] = roll
        near_row["roll_source"] = source
        rows.append(near_row)

    if not rows:
        raise ValueError("主力合约选择后没有可用的品种日数据。请检查字段和合约代码。")
    panel = pd.DataFrame(rows).sort_values(["product", "trade_date"]).reset_index(drop=True)

    def add_forward_targets(group: pd.DataFrame) -> pd.DataFrame:
        group = group.sort_values("trade_date").copy()
        one_day = group["contract_fwd_open_return"].to_numpy(dtype=float)
        contracts = group["contract"].astype(str).to_numpy()
        dates = group["trade_date"].to_numpy()
        n = len(group)
        target = np.full(n, np.nan)
        execution_returns = np.full(n, np.nan)
        sample_dates = np.full(n, np.datetime64("NaT"), dtype="datetime64[ns]")
        exit_dates = np.full(n, np.datetime64("NaT"), dtype="datetime64[ns]")
        entry_contracts = np.full(n, None, dtype=object)

        for i in range(n):
            start = i + 1
            # H 个收益分别位于 start ... start+H-1；最后一个收益的终点是
            # stop 对应的开盘，因此必须保证 stop < n。
            stop = start + forecast_horizon
            if stop >= n:
                continue
            holding_returns = one_day[start:stop]
            if not np.isfinite(holding_returns).all():
                continue
            target[i] = float(np.prod(1.0 + holding_returns) - 1.0)
            execution_returns[i] = float(one_day[start])
            sample_dates[i] = dates[start]
            exit_dates[i] = dates[stop]
            entry_contracts[i] = contracts[start]

        group["sample_date"] = pd.to_datetime(sample_dates)
        group["exit_date"] = pd.to_datetime(exit_dates)
        group["target_raw"] = target
        group["execution_return"] = execution_returns
        group["target_contract"] = entry_contracts
        return group

    product_groups = panel.groupby("product", group_keys=False, sort=False)
    panel = pd.concat(
        [add_forward_targets(group) for _, group in tqdm(product_groups, desc="[Labels] 未来5日累计收益", unit="品种")],
        ignore_index=True,
    )
    # 同一执行日做截面去均值，标签为未来 H 日累计超额收益。
    panel["target_excess"] = panel["target_raw"] - panel.groupby("sample_date")["target_raw"].transform("mean")
    panel["target_label"] = np.where(
        panel["target_excess"].notna(),
        (panel["target_excess"] > 0).astype(float),
        np.nan,
    )

    source_report = panel["roll_source"].value_counts(dropna=False).rename_axis("roll_source").reset_index(name="rows")
    return panel, source_report


# =============================================================================
# 3. 模块化因子工程
# =============================================================================


class FactorEngine:
    """将模型输入转换为原始日频量价变化，并逐日做稳健截面标准化。"""

    def __init__(self, window: int = 20, burn_in_days: int = 90, mad_n: float = 3.0):
        self.window = window
        self.burn_in_days = burn_in_days
        self.mad_n = mad_n

    @staticmethod
    def _safe_log_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
        valid = numerator.gt(0) & denominator.gt(0)
        result = pd.Series(np.nan, index=numerator.index, dtype=float)
        result.loc[valid] = np.log(numerator.loc[valid] / denominator.loc[valid])
        return result

    def add_raw_price_volume_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        frame = frame.sort_values(["product", "trade_date"]).copy()
        frame["log_open_close"] = self._safe_log_ratio(frame["open"], frame["close"])
        frame["log_high_close"] = self._safe_log_ratio(frame["high"], frame["close"])
        frame["log_low_close"] = self._safe_log_ratio(frame["low"], frame["close"])
        previous_close = frame.groupby("product", sort=False)["close"].shift(1)
        frame["log_close_prev_close"] = self._safe_log_ratio(frame["close"], previous_close)
        frame["volume_change"] = frame.groupby("product", sort=False)["volume"].pct_change(fill_method=None)
        frame["open_interest_change"] = frame.groupby("product", sort=False)["open_interest"].pct_change(fill_method=None)
        # 对零分母、合约切换和异常跳变产生的 inf 统一交由缺失与 MAD 处理。
        frame[["volume_change", "open_interest_change"]] = frame[
            ["volume_change", "open_interest_change"]
        ].replace([np.inf, -np.inf], np.nan)
        return frame

    def add_roll_yield_basis(self, frame: pd.DataFrame) -> pd.DataFrame:
        """强制保留商品近远月展期收益/基差属性。"""
        if "roll_yield" not in frame.columns:
            raise KeyError("缺少 roll_yield：无法构建商品期限结构/基差特征")
        frame["roll_yield"] = pd.to_numeric(frame["roll_yield"], errors="coerce")
        return frame

    @staticmethod
    def _mad_zscore(series: pd.Series, mad_n: float) -> pd.Series:
        values = series.replace([np.inf, -np.inf], np.nan).astype(float)
        valid = values.dropna()
        if len(valid) < 3:
            return pd.Series(np.nan, index=series.index)
        median = valid.median()
        mad = (valid - median).abs().median()
        if not np.isfinite(mad) or mad < 1e-12:
            clipped = values.copy()
        else:
            scale = 1.4826 * mad
            clipped = values.clip(median - mad_n * scale, median + mad_n * scale)
        mean, std = clipped.mean(), clipped.std(ddof=0)
        if not np.isfinite(std) or std < 1e-12:
            result = pd.Series(np.nan, index=series.index)
            result.loc[clipped.notna()] = 0.0
            return result
        return (clipped - mean) / std

    def transform(self, panel: pd.DataFrame) -> pd.DataFrame:
        frame = self.add_raw_price_volume_features(panel)
        frame = self.add_roll_yield_basis(frame)
        frame["observation_no"] = frame.groupby("product").cumcount()
        frame = frame[frame["observation_no"].ge(self.burn_in_days)].copy()
        for factor in tqdm(FACTOR_COLS, desc="[Factors] 每日MAD/Z-Score", unit="特征"):
            frame[factor] = frame.groupby("trade_date", group_keys=False)[factor].transform(
                lambda s: self._mad_zscore(s, self.mad_n)
            )
        return frame.replace([np.inf, -np.inf], np.nan)



# =============================================================================
# 4. 序列样本：[Batch, Num_Factors, Sequence_Length]
# =============================================================================


class NumpySequenceDataset(Dataset):
    def __init__(self, x: np.ndarray, y_reg: np.ndarray, y_cls: np.ndarray, indices: np.ndarray):
        self.x = x
        self.y_reg = y_reg
        self.y_cls = y_cls
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        j = self.indices[idx]
        return (
            torch.from_numpy(self.x[j]),
            torch.tensor(self.y_cls[j], dtype=torch.float32),
            torch.tensor(self.y_reg[j], dtype=torch.float32),
        )


def build_sequences(
    factors: pd.DataFrame,
    factor_cols: Sequence[str],
    sequence_length: int,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    xs: List[np.ndarray] = []
    ys: List[float] = []
    metadata: List[Dict[str, object]] = []

    total_products = int(factors["product"].nunique())
    for product, group in tqdm(factors.groupby("product", sort=True), desc="[Samples] 构造20日序列", total=total_products, unit="品种"):
        group = group.sort_values("trade_date").reset_index(drop=True)
        values = group[list(factor_cols)].to_numpy(dtype=np.float32)
        for i in range(sequence_length - 1, len(group)):
            current = group.iloc[i]
            seq = values[i - sequence_length + 1 : i + 1]
            if not np.isfinite(seq).all():
                continue
            if not np.isfinite(current["target_excess"]) or not np.isfinite(current["target_raw"]):
                continue
            if not np.isfinite(current["execution_return"]):
                continue
            if pd.isna(current["sample_date"]) or pd.isna(current["target_contract"]):
                continue
            # Conv1d 要求 channels=factors, length=time。
            xs.append(seq.T.copy())
            ys.append(float(current["target_excess"]))
            metadata.append({
                "product": product,
                "decision_date": current["trade_date"],
                "sample_date": current["sample_date"],
                "exit_date": current["exit_date"],
                "contract": current["target_contract"],
                "target_label": float(current["target_label"]),
                "raw_return": float(current["execution_return"]),
                "forward_5d_return": float(current["target_raw"]),
            })

    if not xs:
        raise ValueError("没有形成有效序列。请检查因子缺失率、预热长度及期限结构字段。")
    x = np.stack(xs).astype(np.float32)
    y = np.asarray(ys, dtype=np.float32)
    meta = pd.DataFrame(metadata)
    meta["sample_date"] = pd.to_datetime(meta["sample_date"])
    meta["decision_date"] = pd.to_datetime(meta["decision_date"])
    meta["exit_date"] = pd.to_datetime(meta["exit_date"])
    assert x.shape[1:] == (len(factor_cols), sequence_length)
    assert (meta["decision_date"] < meta["sample_date"]).all(), "特征截止日必须早于执行日"
    return x, y, meta


# =============================================================================
# 5. Conv1d 局部嵌入 + 轻量 Self-Attention + 多任务双头
# =============================================================================


class FuturesCNN1D(nn.Module):
    """输入 [B,F,20]；返回 (方向logits, 未来5日超额收益回归值)。"""

    def __init__(
        self,
        num_factors: int,
        model_dim: int,
        dropout: float,
        attention_heads: int = 4,
        transformer_ff_dim: int = 96,
        transformer_layers: int = 1,
    ):
        super().__init__()
        if model_dim % attention_heads != 0:
            raise ValueError("model_dim 必须能被 attention_heads 整除")
        self.local_embedding = nn.Sequential(
            nn.Conv1d(num_factors, model_dim, kernel_size=3, padding=1, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=attention_heads,
            dim_feedforward=transformer_ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=transformer_layers,
            norm=nn.LayerNorm(model_dim),
        )
        self.shared_norm = nn.LayerNorm(model_dim)
        self.shared_dropout = nn.Dropout(dropout)
        self.cls_head = nn.Linear(model_dim, 1)
        self.reg_head = nn.Linear(model_dim, 1)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.local_embedding(x)      # [B, D, T]
        x = x.transpose(1, 2)            # [B, T, D]
        x = self.transformer(x)          # 全局20日时序依赖
        shared = self.shared_dropout(self.shared_norm(x.mean(dim=1)))
        logits = self.cls_head(shared).squeeze(-1)
        reg_out = self.reg_head(shared).squeeze(-1)
        return logits, reg_out

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# 6. 时间序列训练、训练内验证与 Walk-Forward
# =============================================================================


def make_loader(
    x: np.ndarray,
    y_reg: np.ndarray,
    y_cls: np.ndarray,
    indices: np.ndarray,
    cfg: Config,
) -> DataLoader:
    return DataLoader(
        NumpySequenceDataset(x, y_reg, y_cls, indices),
        batch_size=cfg.batch_size,
        shuffle=False,  # 明确禁止乱序
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def combined_multitask_loss(
    logits: torch.Tensor,
    reg_out: torch.Tensor,
    y_cls: torch.Tensor,
    y_reg: torch.Tensor,
    alpha: float,
    bce: nn.Module,
    huber: nn.Module,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cls_loss = bce(logits, y_cls)
    reg_loss = huber(reg_out, y_reg)
    total = alpha * cls_loss + (1.0 - alpha) * reg_loss
    return total, cls_loss, reg_loss


def evaluate_loss(
    model: nn.Module,
    loader: DataLoader,
    bce: nn.Module,
    huber: nn.Module,
    alpha: float,
    device: str,
) -> Tuple[float, float, float]:
    model.eval(); totals: List[float] = []; cls_values: List[float] = []; reg_values: List[float] = []; counts: List[int] = []
    with torch.no_grad():
        for xb, y_cls, y_reg in loader:
            xb, y_cls, y_reg = xb.to(device), y_cls.to(device), y_reg.to(device)
            logits, reg_out = model(xb)
            total, cls_loss, reg_loss = combined_multitask_loss(logits, reg_out, y_cls, y_reg, alpha, bce, huber)
            totals.append(float(total.item())); cls_values.append(float(cls_loss.item())); reg_values.append(float(reg_loss.item())); counts.append(len(y_reg))
    if not totals:
        return np.nan, np.nan, np.nan
    return (
        float(np.average(totals, weights=counts)),
        float(np.average(cls_values, weights=counts)),
        float(np.average(reg_values, weights=counts)),
    )


def make_multitask_model(num_factors: int, cfg: Config) -> FuturesCNN1D:
    return FuturesCNN1D(
        num_factors=num_factors,
        model_dim=cfg.model_dim,
        dropout=cfg.dropout,
        attention_heads=cfg.attention_heads,
        transformer_ff_dim=cfg.transformer_ff_dim,
        transformer_layers=cfg.transformer_layers,
    )


def downsample_overlapping_dates(indices: np.ndarray, meta: pd.DataFrame, stride: int) -> np.ndarray:
    """保留每第 stride 个交易日的全部截面样本；同日品种截面不拆散。"""
    if stride <= 1 or len(indices) == 0:
        return indices
    unique_dates = np.array(sorted(meta.loc[indices, "sample_date"].dropna().unique()))
    selected_dates = unique_dates[::stride]
    mask = meta.loc[indices, "sample_date"].isin(selected_dates).to_numpy()
    return indices[mask]


def fit_with_validation(
    x: np.ndarray,
    y_reg: np.ndarray,
    y_cls: np.ndarray,
    fit_idx: np.ndarray,
    val_idx: np.ndarray,
    cfg: Config,
    fold_seed: int,
) -> Tuple[nn.Module, List[Dict[str, float]], int, float]:
    set_seed(fold_seed)
    model = make_multitask_model(x.shape[1], cfg).to(cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    bce = nn.BCEWithLogitsLoss(); huber = nn.HuberLoss(delta=cfg.huber_delta)
    fit_loader = make_loader(x, y_reg, y_cls, fit_idx, cfg)
    val_loader = make_loader(x, y_reg, y_cls, val_idx, cfg)
    history: List[Dict[str, float]] = []; best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf"); best_epoch = 1; stale = 0
    for epoch in range(1, cfg.max_epochs + 1):
        model.train(); totals: List[float] = []; cls_values: List[float] = []; reg_values: List[float] = []; counts: List[int] = []
        for xb, batch_cls, batch_reg in fit_loader:
            xb, batch_cls, batch_reg = xb.to(cfg.device), batch_cls.to(cfg.device), batch_reg.to(cfg.device)
            optimizer.zero_grad(set_to_none=True)
            logits, reg_out = model(xb)
            total, cls_loss, reg_loss = combined_multitask_loss(
                logits, reg_out, batch_cls, batch_reg, cfg.multitask_alpha, bce, huber
            )
            total.backward(); nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0); optimizer.step()
            totals.append(float(total.item())); cls_values.append(float(cls_loss.item())); reg_values.append(float(reg_loss.item())); counts.append(len(batch_reg))
        train_total = float(np.average(totals, weights=counts)); train_cls = float(np.average(cls_values, weights=counts)); train_reg = float(np.average(reg_values, weights=counts))
        val_total, val_cls, val_reg = evaluate_loss(model, val_loader, bce, huber, cfg.multitask_alpha, cfg.device)
        history.append({"epoch": epoch, "train_loss": train_total, "val_loss": val_total, "train_cls_loss": train_cls, "val_cls_loss": val_cls, "train_reg_loss": train_reg, "val_reg_loss": val_reg})
        print(f"[Train] epoch {epoch:02d}/{cfg.max_epochs} | total {train_total:.6f}/{val_total:.6f} | cls {train_cls:.6f}/{val_cls:.6f} | reg {train_reg:.6f}/{val_reg:.6f}", flush=True)
        if val_total < best_val - 1e-8:
            best_val = val_total; best_epoch = epoch; best_state = copy.deepcopy(model.state_dict()); stale = 0
        else:
            stale += 1
            if stale >= cfg.patience: break
    model.load_state_dict(best_state)
    return model, history, best_epoch, best_val


def refit_for_epochs(
    x: np.ndarray,
    y_reg: np.ndarray,
    y_cls: np.ndarray,
    train_idx: np.ndarray,
    epochs: int,
    cfg: Config,
    fold_seed: int,
) -> nn.Module:
    set_seed(fold_seed); model = make_multitask_model(x.shape[1], cfg).to(cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    bce = nn.BCEWithLogitsLoss(); huber = nn.HuberLoss(delta=cfg.huber_delta)
    loader = make_loader(x, y_reg, y_cls, train_idx, cfg)
    for _ in tqdm(range(max(1, epochs)), desc="[Refit] 多任务完整训练窗", unit="epoch"):
        model.train()
        for xb, batch_cls, batch_reg in loader:
            xb, batch_cls, batch_reg = xb.to(cfg.device), batch_cls.to(cfg.device), batch_reg.to(cfg.device)
            optimizer.zero_grad(set_to_none=True); logits, reg_out = model(xb)
            total, _, _ = combined_multitask_loss(logits, reg_out, batch_cls, batch_reg, cfg.multitask_alpha, bce, huber)
            total.backward(); nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0); optimizer.step()
    return model


def predict(model: nn.Module, x: np.ndarray, indices: np.ndarray, cfg: Config) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    dummy = np.zeros(len(x), dtype=np.float32); loader = make_loader(x, dummy, dummy, indices, cfg)
    probabilities: List[np.ndarray] = []; regressions: List[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for xb, _, _ in loader:
            logits, reg_out = model(xb.to(cfg.device))
            probabilities.append(torch.sigmoid(logits).cpu().numpy()); regressions.append(reg_out.cpu().numpy())
    if not probabilities:
        empty = np.array([], dtype=float); return empty, empty, empty
    prob = np.concatenate(probabilities); reg = np.concatenate(regressions)
    return prob * reg, prob, reg

def temporal_train_val_split(
    train_idx: np.ndarray,
    meta: pd.DataFrame,
    cfg: Config,
) -> Tuple[np.ndarray, np.ndarray]:
    dates = np.array(sorted(meta.loc[train_idx, "sample_date"].dropna().unique()))
    if len(dates) < 100:
        raise ValueError("训练窗有效日期太少，无法进行训练内时间验证。")
    val_n = max(20, int(math.ceil(len(dates) * cfg.validation_fraction)))
    val_dates = dates[-val_n:]
    val_start_pos = len(dates) - val_n
    fit_end_pos = max(1, val_start_pos - cfg.purge_days)
    fit_dates = dates[:fit_end_pos]
    fit_idx = train_idx[meta.loc[train_idx, "sample_date"].isin(fit_dates).to_numpy()]
    val_idx = train_idx[meta.loc[train_idx, "sample_date"].isin(val_dates).to_numpy()]
    if len(fit_idx) == 0 or len(val_idx) == 0:
        raise ValueError("purge 后训练集或验证集为空。")
    return fit_idx, val_idx


def permutation_importance(
    model: nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    indices: np.ndarray,
    cfg: Config,
    seed: int,
) -> np.ndarray:
    if len(indices) == 0:
        return np.zeros(x.shape[1])
    base_pred, _, _ = predict(model, x, indices, cfg)
    base_mse = float(np.mean((base_pred - y[indices]) ** 2))
    rng = np.random.default_rng(seed)
    importance = np.zeros(x.shape[1], dtype=float)
    for factor_idx in tqdm(range(x.shape[1]), desc="[Explain] 置换重要性", unit="因子"):
        perturbed = x[indices].copy()
        order = rng.permutation(len(indices))
        perturbed[:, factor_idx, :] = perturbed[order, factor_idx, :]
        perm_pred, _, _ = predict(model, perturbed, np.arange(len(indices)), cfg)
        importance[factor_idx] = max(0.0, float(np.mean((perm_pred - y[indices]) ** 2)) - base_mse)
    return importance


def walk_forward(
    x: np.ndarray,
    y: np.ndarray,
    meta: pd.DataFrame,
    cfg: Config,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, Dict[str, object]]:
    predictions: List[pd.DataFrame] = []
    histories: List[pd.DataFrame] = []
    importances: List[np.ndarray] = []
    best_checkpoint: Optional[Dict[str, object]] = None
    best_global_val = float("inf")

    sample_year = meta["sample_date"].dt.year
    y_cls = meta["target_label"].to_numpy(dtype=np.float32)
    for test_year in range(cfg.first_test_year, cfg.last_test_year + 1):
        # Expanding Window：训练起点固定为 2015，终点随测试年向前扩展。
        train_start = cfg.train_start_year
        train_mask = sample_year.between(train_start, test_year - 1)
        test_mask = sample_year.eq(test_year)
        train_idx = np.flatnonzero(train_mask.to_numpy())
        test_idx = np.flatnonzero(test_mask.to_numpy())
        if len(train_idx) == 0 or len(test_idx) == 0:
            print(f"[WalkForward] 跳过 {test_year}: 训练或测试样本为空")
            continue

        print(
            f"\n[WalkForward] 扩展窗 {train_start}-{test_year - 1} -> {test_year} | "
            f"train={len(train_idx):,}, test={len(test_idx):,}",
            flush=True,
        )
        fit_idx, val_idx = temporal_train_val_split(train_idx, meta, cfg)
        raw_fit_count, raw_train_count = len(fit_idx), len(train_idx)
        fit_idx = downsample_overlapping_dates(fit_idx, meta, cfg.train_date_stride)
        train_idx_sampled = downsample_overlapping_dates(train_idx, meta, cfg.train_date_stride)
        print(
            f"[Overlap] 每{cfg.train_date_stride}日取样：fit {raw_fit_count:,}->{len(fit_idx):,}，"
            f"refit {raw_train_count:,}->{len(train_idx_sampled):,}；验证/测试保持逐日",
            flush=True,
        )
        fold_seed = cfg.seed + test_year
        _, history, best_epoch, best_val = fit_with_validation(
            x, y, y_cls, fit_idx, val_idx, cfg, fold_seed
        )
        model = refit_for_epochs(x, y, y_cls, train_idx_sampled, best_epoch, cfg, fold_seed)
        test_pred, test_prob, test_reg = predict(model, x, test_idx, cfg)

        fold_pred = meta.iloc[test_idx].copy()
        fold_pred["prediction"] = test_pred
        fold_pred["classification_probability"] = test_prob
        fold_pred["regression_prediction"] = test_reg
        fold_pred["target_excess"] = y[test_idx]
        fold_pred["target_label"] = y_cls[test_idx]
        fold_pred["fold"] = f"{train_start}-{test_year - 1}->{test_year}"
        predictions.append(fold_pred)

        fold_history = pd.DataFrame(history)
        fold_history["test_year"] = test_year
        histories.append(fold_history)
        importances.append(permutation_importance(model, x, y, test_idx, cfg, fold_seed))

        if best_val < best_global_val:
            best_global_val = best_val
            best_checkpoint = {
                "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "config": asdict(cfg),
                "factor_names": FACTOR_COLS,
                "test_year": test_year,
                "train_years": [train_start, test_year - 1],
                "training_scheme": "expanding_window",
                "forecast_horizon": cfg.forecast_horizon,
                "selected_epoch": best_epoch,
                "inner_validation_loss": best_val,
                "model_class": "ConvTransformerDualHead",
                "multitask_alpha": cfg.multitask_alpha,
                "train_date_stride": cfg.train_date_stride,
                "input_shape": [len(FACTOR_COLS), cfg.sequence_length],
            }

        print(
            f"[WalkForward] 完成 {train_start}-{test_year - 1} -> {test_year} | "
            f"train={len(train_idx):,}, test={len(test_idx):,}, "
            f"epoch={best_epoch}, val={best_val:.6g}"
        )

    if not predictions or best_checkpoint is None:
        raise RuntimeError("没有完成任何 Walk-Forward 折。请检查年份范围和数据覆盖。")
    mean_importance = np.mean(np.vstack(importances), axis=0)
    return (
        pd.concat(predictions, ignore_index=True).sort_values(["sample_date", "product"]),
        pd.concat(histories, ignore_index=True),
        mean_importance,
        best_checkpoint,
    )


# =============================================================================
# 7. 截面多空回测、合约层换手与单边费用敏感性
# =============================================================================


def build_systemic_stress_features(panel: pd.DataFrame) -> pd.DataFrame:
    """由市场波动、平均相关性和流动性压力构造每日系统性压力特征。

    全部特征只使用当日及更早数据；相关性使用过去 20 日品种收益矩阵，
    流动性压力使用截面成交量/持仓量比的上升程度。
    """
    frame = panel.sort_values(["trade_date", "product"]).copy()
    returns = frame.pivot(index="trade_date", columns="product", values="contract_close_return").sort_index()
    market_vol = returns.std(axis=1, skipna=True).rolling(20, min_periods=10).mean()

    corr_values: List[float] = []
    for i in tqdm(range(len(returns)), desc="[Stress] 20日平均相关性", unit="日"):
        if i < 19:
            corr_values.append(np.nan)
            continue
        corr = returns.iloc[i - 19 : i + 1].corr(min_periods=10).to_numpy(dtype=float)
        upper = corr[np.triu_indices_from(corr, k=1)]
        upper = upper[np.isfinite(upper)]
        corr_values.append(float(upper.mean()) if len(upper) else np.nan)
    average_correlation = pd.Series(corr_values, index=returns.index)

    liquidity_ratio = frame["volume"] / frame["open_interest"].where(frame["open_interest"] > 0)
    frame["_liq_ratio"] = np.log1p(liquidity_ratio.clip(lower=0))
    liq_level = frame.groupby("trade_date")["_liq_ratio"].median().reindex(returns.index)
    baseline = liq_level.rolling(60, min_periods=20).median()
    liquidity_stress = (baseline - liq_level).rolling(5, min_periods=1).mean()

    features = pd.DataFrame({
        "date": returns.index,
        "market_volatility": market_vol.to_numpy(),
        "average_correlation": average_correlation.to_numpy(),
        "liquidity_stress": liquidity_stress.to_numpy(),
    })
    return features.replace([np.inf, -np.inf], np.nan)


def estimate_causal_gmm_stress(stress_features: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """逐测试年因果拟合两状态 GMM，并返回高压力状态概率 s_t。

    对测试年 Y，仅使用日期早于 Y-01-01 的历史特征拟合 scaler 和 GMM；
    测试年数据只做 transform/predict_proba，杜绝用未来压力分布反向拟合。
    """
    features = stress_features.sort_values("date").copy()
    feature_cols = ["market_volatility", "average_correlation", "liquidity_stress"]
    output: List[pd.DataFrame] = []

    for test_year in tqdm(range(cfg.first_test_year, cfg.last_test_year + 1), desc="[Stress] 因果GMM年度", unit="年"):
        cutoff = pd.Timestamp(test_year, 1, 1)
        train = features[features["date"].lt(cutoff)].dropna(subset=feature_cols)
        test = features[features["date"].dt.year.eq(test_year)].dropna(subset=feature_cols).copy()
        if len(test) == 0:
            continue
        if len(train) < cfg.gmm_min_history_days:
            warnings.warn(f"{test_year} 年 GMM 历史不足，压力概率回退为 0")
            test["systemic_stress_probability"] = 0.0
            output.append(test[["date", "systemic_stress_probability"]])
            continue

        scaler = StandardScaler().fit(train[feature_cols])
        train_x = scaler.transform(train[feature_cols])
        test_x = scaler.transform(test[feature_cols])
        gmm = GaussianMixture(
            n_components=cfg.gmm_components,
            covariance_type="full",
            max_iter=cfg.gmm_max_iter,
            n_init=5,
            reg_covar=1e-5,
            random_state=cfg.seed + test_year,
        ).fit(train_x)

        # 在标准化空间中，三项特征均越高越压力；均值总和最大的状态为高压态。
        stress_component = int(np.argmax(gmm.means_.sum(axis=1)))
        test["systemic_stress_probability"] = gmm.predict_proba(test_x)[:, stress_component]
        output.append(test[["date", "systemic_stress_probability"]])

    if not output:
        return pd.DataFrame(columns=["date", "systemic_stress_probability"])
    result = pd.concat(output, ignore_index=True).drop_duplicates("date", keep="last")
    result["systemic_stress_probability"] = result["systemic_stress_probability"].clip(0.0, 1.0)
    return result.sort_values("date")


def normalized_signal_weights(scores: pd.Series, gross_side: float) -> pd.Series:
    """按预测强度分配单侧仓位；减去侧内边界后归一化，避免退化为等权。"""
    strength = scores.abs().astype(float)
    shifted = strength - strength.min() + 1e-6
    total = shifted.sum()
    if not np.isfinite(total) or total <= 0:
        shifted = pd.Series(1.0, index=scores.index)
        total = shifted.sum()
    return gross_side * shifted / total


def build_daily_portfolio(
    predictions: pd.DataFrame,
    cfg: Config,
    stress_probabilities: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """3日 EMA + 动态分位与绝对阈值入场；越过中位数立即退出。

    多头：信号位于上30%且大于绝对阈值才入场，跌破当日中位数即退出。
    空头完全对称：位于下30%且小于负阈值才入场，升破中位数即退出。
    不强制固定持仓数量；已有费用、信号加权和GMM门控保持不变。
    """
    position_rows: List[Dict[str, object]] = []
    daily_rows: List[Dict[str, object]] = []
    previous_contract_weights: Dict[str, float] = {}
    long_products: set = set()
    short_products: set = set()

    scored = predictions.sort_values(["product", "sample_date"]).copy()
    scored["smoothed_prediction"] = scored.groupby("product", sort=False)["prediction"].transform(
        lambda s: s.ewm(span=cfg.signal_ema_span, adjust=False, min_periods=1).mean()
    )
    if stress_probabilities is None or stress_probabilities.empty:
        scored["systemic_stress_probability"] = 0.0
    else:
        stress = stress_probabilities.rename(columns={"date": "sample_date"}).copy()
        stress["sample_date"] = pd.to_datetime(stress["sample_date"])
        scored = scored.merge(stress, on="sample_date", how="left")
        scored["systemic_stress_probability"] = scored["systemic_stress_probability"].fillna(0.0).clip(0.0, 1.0)

    groups = scored.groupby("sample_date", sort=True)
    total_days = int(groups.ngroups)
    for date, day in tqdm(groups, desc="[Backtest] EMA+缓冲区", total=total_days, unit="日"):
        day = day.dropna(subset=["smoothed_prediction", "raw_return", "contract"]).copy()
        if len(day) < cfg.min_cross_section:
            continue
        day = day.sort_values(["smoothed_prediction", "product"], ascending=[False, True]).reset_index(drop=True)
        n_assets = len(day)
        upper_threshold = float(day["smoothed_prediction"].quantile(cfg.entry_quantile))
        lower_threshold = float(day["smoothed_prediction"].quantile(1.0 - cfg.entry_quantile))
        median_signal = float(day["smoothed_prediction"].median())
        signal_by_product = day.set_index("product")["smoothed_prediction"].to_dict()
        available = set(signal_by_product)

        # 中位数是即时退出线，不再使用固定40%缓冲区。
        long_products = {
            p for p in long_products
            if p in available and signal_by_product[p] >= median_signal
        }
        short_products = {
            p for p in short_products
            if p in available and signal_by_product[p] <= median_signal
        }

        long_entries = day[
            day["smoothed_prediction"].ge(upper_threshold)
            & day["smoothed_prediction"].gt(cfg.absolute_signal_threshold)
        ]["product"]
        short_entries = day[
            day["smoothed_prediction"].le(lower_threshold)
            & day["smoothed_prediction"].lt(-cfg.absolute_signal_threshold)
        ]["product"]
        long_products.update(long_entries.tolist())
        short_products.update(short_entries.tolist())
        # 理论上零阈值已使两侧互斥；此处仍显式保护。
        overlap = long_products & short_products
        long_products -= overlap; short_products -= overlap

        # 某一侧无有效信号时允许另一侧继续持仓，现金权重自然保留。
        long_side = day[day["product"].isin(long_products)].copy()
        short_side = day[day["product"].isin(short_products)].copy()

        stress_probability = float(day["systemic_stress_probability"].iloc[0])
        leverage_multiplier = (
            cfg.stress_leverage_multiplier
            if stress_probability > cfg.stress_threshold
            else 1.0
        )
        side_gross = 0.5 * leverage_multiplier
        if cfg.weighting_method != "signal":
            raise ValueError(f"当前仅实现 weighting_method='signal'，收到: {cfg.weighting_method}")
        if not long_side.empty:
            long_side["weight"] = normalized_signal_weights(long_side["smoothed_prediction"], side_gross)
        if not short_side.empty:
            short_side["weight"] = -normalized_signal_weights(short_side["smoothed_prediction"], side_gross)
        positions = pd.concat([long_side, short_side], ignore_index=True)
        if positions.empty:
            current_contract_weights = {}
        else:
            current_contract_weights = positions.groupby("contract")["weight"].sum().to_dict()

        all_contracts = set(previous_contract_weights) | set(current_contract_weights)
        absolute_traded = sum(
            abs(current_contract_weights.get(c, 0.0) - previous_contract_weights.get(c, 0.0))
            for c in all_contracts
        )
        turnover = 0.5 * absolute_traded
        gross = float((positions["weight"] * positions["raw_return"]).sum())

        row: Dict[str, object] = {
            "date": date,
            "gross_return": gross,
            "absolute_traded": absolute_traded,
            "turnover": turnover,
            "n_assets": n_assets,
            "n_long": len(long_side),
            "n_short": len(short_side),
            "upper_entry_threshold": upper_threshold,
            "lower_entry_threshold": lower_threshold,
            "median_exit_threshold": median_signal,
            "systemic_stress_probability": stress_probability,
            "leverage_multiplier": leverage_multiplier,
            "stress_gate_active": bool(stress_probability > cfg.stress_threshold),
            "net_exposure": float(positions["weight"].sum()),
            "gross_exposure": float(positions["weight"].abs().sum()),
        }
        for fee in cfg.one_way_fees:
            row[f"net_return_{int(round(fee * 10000))}bp"] = gross - fee * absolute_traded
        daily_rows.append(row)

        for record in positions.to_dict("records"):
            position_rows.append({
                "date": date,
                "product": record["product"],
                "contract": record["contract"],
                "weight": record["weight"],
                "prediction": record["prediction"],
                "smoothed_prediction": record["smoothed_prediction"],
                "systemic_stress_probability": stress_probability,
                "leverage_multiplier": leverage_multiplier,
                "realized_return": record["raw_return"],
            })
        previous_contract_weights = current_contract_weights

    daily = pd.DataFrame(daily_rows).sort_values("date").reset_index(drop=True)
    if daily.empty:
        raise RuntimeError("没有形成有效回测日期，请降低 min_cross_section 或检查预测缺失。")

    if cfg.liquidate_at_end and previous_contract_weights:
        final_absolute = sum(abs(w) for w in previous_contract_weights.values())
        daily.loc[daily.index[-1], "absolute_traded"] += final_absolute
        daily.loc[daily.index[-1], "turnover"] += 0.5 * final_absolute
        for fee in cfg.one_way_fees:
            col = f"net_return_{int(round(fee * 10000))}bp"
            daily.loc[daily.index[-1], col] -= fee * final_absolute

    return daily, pd.DataFrame(position_rows)


def performance_metrics(returns: pd.Series, annualization: int = 252) -> Dict[str, float]:
    returns = pd.Series(returns).replace([np.inf, -np.inf], np.nan).dropna()
    if returns.empty:
        return {"annual_return": np.nan, "sharpe_ratio": np.nan, "max_drawdown": np.nan, "win_rate": np.nan}
    wealth = (1.0 + returns).cumprod()
    years = len(returns) / annualization
    annual_return = wealth.iloc[-1] ** (1.0 / years) - 1.0 if years > 0 and wealth.iloc[-1] > 0 else np.nan
    vol = returns.std(ddof=1)
    sharpe = returns.mean() / vol * np.sqrt(annualization) if vol > 1e-12 else np.nan
    drawdown = wealth / wealth.cummax() - 1.0
    return {
        "annual_return": float(annual_return),
        "sharpe_ratio": float(sharpe),
        "max_drawdown": float(drawdown.min()),
        "win_rate": float((returns > 0).mean()),
        "total_return": float(wealth.iloc[-1] - 1.0),
        "annual_volatility": float(vol * np.sqrt(annualization)) if np.isfinite(vol) else np.nan,
    }


def make_metrics_table(daily: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    scenarios = [("gross", "gross_return", 0.0)] + [
        (f"{int(round(f * 10000))}bp_one_way", f"net_return_{int(round(f * 10000))}bp", f)
        for f in cfg.one_way_fees
    ]
    for name, col, fee in scenarios:
        metrics = performance_metrics(daily[col], cfg.annualization)
        metrics.update({
            "scenario": name,
            "one_way_fee": fee,
            "observations": len(daily),
            "average_daily_turnover": float(daily["turnover"].mean()),
            "average_absolute_traded": float(daily["absolute_traded"].mean()),
        })
        rows.append(metrics)
    return pd.DataFrame(rows)[
        ["scenario", "one_way_fee", "annual_return", "sharpe_ratio", "max_drawdown", "win_rate",
         "total_return", "annual_volatility", "average_daily_turnover", "average_absolute_traded", "observations"]
    ]


# =============================================================================
# 8. 可视化与输出
# =============================================================================


def setup_plot_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({"figure.dpi": 130, "axes.titlesize": 12, "axes.labelsize": 10})


def plot_loss(history: pd.DataFrame, path: Path) -> None:
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(9, 5))
    for year, fold in history.groupby("test_year"):
        ax.plot(fold["epoch"], fold["train_loss"], alpha=0.65, label=f"Train -> {year}")
        ax.plot(fold["epoch"], fold["val_loss"], linestyle="--", alpha=0.75, label=f"Val -> {year}")
    ax.set(title="Expanding-Window Training / Validation Loss", xlabel="Epoch", ylabel="Huber Loss")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def plot_cumulative_and_drawdown(daily: pd.DataFrame, cfg: Config, path: Path) -> None:
    setup_plot_style()
    bp = int(round(cfg.base_fee * 10000)); col = f"net_return_{bp}bp"
    wealth = (1.0 + daily[col]).cumprod(); drawdown = wealth / wealth.cummax() - 1.0
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    axes[0].plot(daily["date"], wealth, color="#0b6e4f", linewidth=1.6)
    axes[0].set(title=f"OOS Equity Curve ({bp} bp one-way, GMM gated)", ylabel="Net Asset Value")
    axes[1].fill_between(daily["date"], drawdown.to_numpy(), 0, color="#b23a48", alpha=0.65)
    axes[1].set(ylabel="Drawdown", xlabel="Date")
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def plot_fee_sensitivity(daily: pd.DataFrame, cfg: Config, path: Path) -> None:
    setup_plot_style(); fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(daily["date"], (1 + daily["gross_return"]).cumprod(), label="Gross", linewidth=1.5, color="black")
    colors = ["#247ba0", "#f4a261", "#c1121f"]
    for fee, color in zip(cfg.one_way_fees, colors):
        bp = int(round(fee * 10000))
        ax.plot(daily["date"], (1 + daily[f"net_return_{bp}bp"]).cumprod(), label=f"{bp} bp one-way", color=color)
    ax.set(title="Transaction Cost Sensitivity", xlabel="Date", ylabel="Net Asset Value"); ax.legend()
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def plot_feature_importance(importance: np.ndarray, path: Path) -> None:
    setup_plot_style(); total = importance.sum(); normalized = importance / total if total > 0 else importance
    order = np.argsort(normalized); fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.barh(np.array(FACTOR_COLS)[order], normalized[order], color="#5c6bc0")
    ax.set(title="OOS Permutation Feature Importance", xlabel="Normalized increase in MSE")
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def run_assertions(daily: pd.DataFrame, cfg: Config) -> None:
    leverage_cap = np.where(
        daily["systemic_stress_probability"].to_numpy() > cfg.stress_threshold,
        cfg.stress_leverage_multiplier, 1.0,
    )
    assert (daily["gross_exposure"].to_numpy() <= leverage_cap + 1e-8).all(), "组合毛敞口超过GMM门控上限"
    assert (daily["net_exposure"].abs().to_numpy() <= 0.5 * leverage_cap + 1e-8).all(), "单边现金状态净敞口超限"
    both_sides = daily["n_long"].gt(0) & daily["n_short"].gt(0)
    assert np.allclose(daily.loc[both_sides, "net_exposure"], 0.0, atol=1e-8), "双边组合净敞口不为零"
    assert np.allclose(daily.loc[both_sides, "gross_exposure"], leverage_cap[both_sides], atol=1e-8), "双边组合毛敞口不正确"
    net_cols = [f"net_return_{int(round(f * 10000))}bp" for f in cfg.one_way_fees]
    for fee, col in zip(cfg.one_way_fees, net_cols):
        expected = daily["gross_return"] - fee * daily["absolute_traded"]
        assert np.allclose(daily[col], expected, atol=1e-12), f"{col} 成本公式不一致"
    final_wealth = [(1 + daily[col]).prod() for col in net_cols]
    assert all(a >= b - 1e-10 for a, b in zip(final_wealth, final_wealth[1:])), "费用敏感性顺序异常"


# =============================================================================
# 9. 主流程
# =============================================================================


def main(cfg: Config = CFG) -> Dict[str, object]:
    set_seed(cfg.seed); output = Path(cfg.output_dir); output.mkdir(parents=True, exist_ok=True)
    print("\n========== 中国商品期货 Conv-Transformer 多任务研究开始 ==========", flush=True)
    print(f"[Config] device={cfg.device}, output={output}", flush=True)
    print(
        f"[Config] 原始特征={FACTOR_COLS} | data={cfg.data_start_year}-{cfg.data_end_year} | "
        f"horizon={cfg.forecast_horizon}日 | expanding={cfg.train_start_year}-{cfg.first_test_year - 1}->{cfg.first_test_year}",
        flush=True,
    )
    root = resolve_data_root(cfg); print(f"[1/9 数据] root={root}", flush=True)
    basic = load_basic(root); market, data_report = load_market_data(root, cfg)
    print(f"[2/9 数据] 行情过滤后 {len(market):,} 行，开始合并合约元数据……", flush=True)
    market = attach_metadata_and_returns(market, basic)
    print("[3/9 主力] 开始构造仅使用滞后活跃度的因果主力……", flush=True)
    selected = choose_main_contracts(market, cfg)
    print("[4/9 面板] 开始构造期限结构与未来5日标签……", flush=True)
    panel, roll_report = build_product_panel(market, selected, cfg.forecast_horizon)
    print("[5/9 特征] 构造原始OHLC/量仓变化与roll yield，并做每日MAD/Z-Score……", flush=True)
    factors = FactorEngine(cfg.factor_window, cfg.burn_in_days, cfg.mad_n).transform(panel)
    missing_report = factors[FACTOR_COLS].isna().mean().rename("missing_rate").rename_axis("factor").reset_index()
    valid_counts = factors.dropna(subset=FACTOR_COLS).groupby("trade_date")["product"].nunique()
    data_report.update({
        "selected_main_rows": len(selected), "panel_rows": len(panel),
        "factor_rows_after_burn_in": len(factors),
        "median_valid_products_per_day": float(valid_counts.median()) if len(valid_counts) else np.nan,
    })
    print(f"[6/9 样本] 构造 [样本, {len(FACTOR_COLS)}特征, {cfg.sequence_length}日] 张量……", flush=True)
    x, y, meta = build_sequences(factors, FACTOR_COLS, cfg.sequence_length)
    print(f"[Samples] X={x.shape}, y={y.shape}, range={meta.sample_date.min().date()}~{meta.sample_date.max().date()}", flush=True)
    example_model = make_multitask_model(len(FACTOR_COLS), cfg)
    print(f"[Model] Conv-Transformer Dual-Head parameters={example_model.count_parameters():,}", flush=True)
    print("[7/9 训练] 开始 Expanding-Window 训练与样本外预测……", flush=True)
    predictions, history, importance, checkpoint = walk_forward(x, y, meta, cfg)
    print("[Stress] 构造市场波动率、平均相关性和流动性压力特征……", flush=True)
    stress_features = build_systemic_stress_features(panel)
    stress_probabilities = estimate_causal_gmm_stress(stress_features, cfg)
    print(
        f"[Stress] 已生成 {len(stress_probabilities):,} 个样本外压力概率；"
        f"阈值>{cfg.stress_threshold:.2f}时毛杠杆×{cfg.stress_leverage_multiplier:.2f}", flush=True,
    )
    print("[8/9 回测] 开始 EMA、动态阈值、信号加权与 GMM 门控……", flush=True)
    daily, positions = build_daily_portfolio(predictions, cfg, stress_probabilities)
    run_assertions(daily, cfg); metrics = make_metrics_table(daily, cfg)
    print("[9/9 输出] 生成图表、模型和审计文件……", flush=True)
    plot_loss(history, output / "loss_curve.png")
    plot_cumulative_and_drawdown(daily, cfg, output / "cumulative_return.png")
    plot_fee_sensitivity(daily, cfg, output / "fee_sensitivity.png")
    plot_feature_importance(importance, output / "feature_importance.png")
    torch.save(checkpoint, output / "best_model.pth")
    metrics.to_csv(output / "backtest_metrics.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(output / "oos_predictions.csv", index=False, encoding="utf-8-sig")
    daily.to_csv(output / "daily_backtest.csv", index=False, encoding="utf-8-sig")
    positions.to_csv(output / "positions.csv", index=False, encoding="utf-8-sig")
    history.to_csv(output / "training_history.csv", index=False, encoding="utf-8-sig")
    roll_report.to_csv(output / "roll_yield_sources.csv", index=False, encoding="utf-8-sig")
    missing_report.to_csv(output / "factor_missing_report.csv", index=False, encoding="utf-8-sig")
    stress_features.to_csv(output / "systemic_stress_features.csv", index=False, encoding="utf-8-sig")
    stress_probabilities.to_csv(output / "systemic_stress_probability.csv", index=False, encoding="utf-8-sig")
    with open(output / "run_config_and_data_report.json", "w", encoding="utf-8") as file:
        json.dump({"config": asdict(cfg), "data_report": data_report}, file, ensure_ascii=False, indent=2, default=str)
    print("\n[Done] 样本外回测统计："); print(metrics.to_string(index=False))
    print(f"\n[Done] 所有结果已保存到: {output}")
    return {
        "metrics": metrics, "daily": daily, "predictions": predictions,
        "stress_probabilities": stress_probabilities,
        "feature_importance": pd.Series(importance, index=FACTOR_COLS), "output_dir": output,
    }


if __name__ == "__main__":
    RESULTS = main(CFG)
