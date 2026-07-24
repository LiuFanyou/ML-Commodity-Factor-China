"""
中国商品期货机器学习因子投资：扩展窗口验证与可行性评估
=======================================================

本地/Kaggle 通用脚本。核心设计：
1) 因果主力合约与近远月期限结构；
2) 原始OHLC相对价格、收盘收益、量仓变化率与商品展期收益；
3) 20日原始量价序列输入 Conv1d + Self-Attention 双头网络；
4) 预测未来 20 个交易日累计截面超额收益；
5) 2018年起的 Expanding-Window：2018-2021 -> 2022，随后逐年扩展；
6) 预测值=上涨概率×回归幅度，经3日EMA后按动态分位/零阈值交易；
7) 输出训练诊断、净值、成本敏感性、因子重要性、模型和审计文件。

时间定义：因子在决策日 t 收盘后可得，t+1 开盘执行。20 日标签覆盖
从 t+1 开盘开始的二十个交易持有期；实际回测仍逐日按 open-to-open 收益记账，
避免把重叠的 20 日标签错误当成每日组合收益。
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
    feature_registry_path: Optional[str] = None
    feature_matrix_path: Optional[str] = None
    integrated_market_path: Optional[str] = os.getenv("FUTURES_INTEGRATED_CSV")
    # auto: 发现整合表时由整合表重建因果特征，否则使用旧59因子交接包；
    # integrated/handoff 可强制指定路径。
    feature_mode: str = os.getenv("FUTURES_FEATURE_MODE", "auto").strip().lower()
    expected_feature_count: int = 59
    output_dir: str = (
        "/kaggle/working/futures_ml_outputs"
        if Path("/kaggle/working").exists()
        else str(Path.cwd() / "futures_ml_outputs")
    )

    # 固定30品种研究池。交接特征必须覆盖全部品种，否则严格报错停止。
    products: Tuple[str, ...] = (
        "RB", "HC", "I", "JM", "J", "SF",
        "CU", "AL", "ZN", "NI", "SN",
        "SC", "FU", "BU", "TA", "MA", "L", "PP", "V",
        "M", "Y", "P", "C", "CS", "SR", "CF", "RM",
        "AU", "AG", "SA",
    )
    product_embedding_dim: int = 8
    min_cross_section: int = 10
    min_days_to_expiry: int = 7
    roll_switch_ratio: float = 1.10

    factor_window: int = 20
    sequence_length: int = 20
    burn_in_days: int = 90  # 应位于 60~120 个交易日范围内
    mad_n: float = 3.0

    data_start_year: int = 2015
    data_end_year: int = 2025

    # Expanding Window：2015-2021 训练，2022 首个样本外测试，随后逐年扩展。
    train_start_year: int = 2015
    first_test_year: int = 2022
    last_test_year: int = 2025
    forecast_horizon: int = 20
    validation_fraction: float = 0.15
    purge_days: int = 40  # 20日输入序列 + 20日标签的防重叠保护间隔

    # 卷积局部嵌入 + 单层轻量 Transformer。
    model_dim: int = 48
    attention_heads: int = 4
    transformer_ff_dim: int = 96
    transformer_layers: int = 1
    dropout: float = 0.40
    multitask_alpha: float = 0.70
    train_date_stride: int = 5
    batch_size: int = 512
    max_epochs: int = 40
    patience: int = 7
    learning_rate: float = 1e-3
    weight_decay: float = 1e-3
    huber_delta: float = 1.0
    num_workers: int = 0

    signal_ema_span: int = 3
    entry_quantile: float = 0.70  # 做多前30%，做空后30%
    long_exit_quantile: float = 0.40  # 多头跌破截面40%分位才退出
    short_exit_quantile: float = 0.60  # 空头升破截面60%分位才退出
    liquidity_entry_quantile: float = 0.20  # 最低20%流动性品种禁止新开仓
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
# 运行时由 ml_feature_registry.csv 的 feature_name 列动态覆盖。
FACTOR_COLS: List[str] = []

PRODUCT_METADATA: Dict[str, Tuple[str, str]] = {
    "RB": ("黑色产业链", "螺纹钢"), "HC": ("黑色产业链", "热卷"),
    "I": ("黑色产业链", "铁矿石"), "JM": ("黑色产业链", "焦煤"),
    "J": ("黑色产业链", "焦炭"), "SF": ("黑色产业链", "硅铁"),
    "CU": ("有色金属", "铜"), "AL": ("有色金属", "铝"),
    "ZN": ("有色金属", "锌"), "NI": ("有色金属", "镍"), "SN": ("有色金属", "锡"),
    "SC": ("能源化工", "原油"), "FU": ("能源化工", "燃料油"),
    "BU": ("能源化工", "沥青"), "TA": ("能源化工", "PTA"),
    "MA": ("能源化工", "甲醇"), "L": ("能源化工", "聚乙烯"),
    "PP": ("能源化工", "聚丙烯"), "V": ("能源化工", "PVC"),
    "M": ("农产品", "豆粕"), "Y": ("农产品", "豆油"), "P": ("农产品", "棕榈油"),
    "C": ("农产品", "玉米"), "CS": ("农产品", "淀粉"), "SR": ("农产品", "白糖"),
    "CF": ("农产品", "棉花"), "RM": ("农产品", "菜粕"),
    "AU": ("贵金属", "黄金"), "AG": ("贵金属", "白银"),
    "SA": ("建材/化工补充", "纯碱"),
}
PRODUCT_TO_ID: Dict[str, int] = {symbol: idx for idx, symbol in enumerate(PRODUCT_METADATA)}


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
    "spot_age": ("spot_price_age_days", "spot_age_days"),
    "warehouse_receipt": ("warehouse_receipt_current", "warehouse_receipt"),
    "warehouse_receipt_change": ("warehouse_receipt_change",),
    "warehouse_receipt_age": ("warehouse_receipt_age_days",),
    "warehouse_receipt_conflict": ("warehouse_receipt_conflict",),
    "stock": ("stock_current", "inventory", "stock"),
    "stock_age": ("stock_age_days",),
    "available_stock": ("available_stock_current", "available_inventory"),
    "available_stock_age": ("available_stock_age_days",),
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


def find_integrated_market_file(root: Path, cfg: Config) -> Optional[Path]:
    """定位整合后的合约日表；显式配置优先，其次使用约定文件名。"""
    if cfg.integrated_market_path:
        path = Path(cfg.integrated_market_path)
        if not path.is_absolute():
            path = root / path
        if not path.exists():
            raise FileNotFoundError(f"配置的整合行情文件不存在: {path}")
        return path
    return find_named_file(root, "futures_contract_daily_30varieties_2015_2025.csv")


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
    integrated = find_integrated_market_file(root, cfg)
    if integrated is not None:
        files = [integrated]
        market_source = "integrated_contract_daily"
        print(f"[Data] 优先使用整合合约日表：{integrated}", flush=True)
    else:
        files = discover_market_files(root, cfg.data_start_year, cfg.data_end_year)
        market_source = "discovered_raw_csvs"
        print(
            f"[Data] 未发现整合表，仅加载 {cfg.data_start_year}-{cfg.data_end_year}："
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
        "amount", "direct_roll_yield", "basis", "spot", "spot_age",
        "warehouse_receipt", "warehouse_receipt_change", "warehouse_receipt_age",
        "warehouse_receipt_conflict", "stock", "stock_age", "available_stock",
        "available_stock_age",
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
        "market_data_source": market_source,
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
    forecast_horizon: int = 20,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """构造品种面板及未来 H 日累计标签。

    决策日 t 的信号在 t 收盘后形成，并于 t+1 开盘执行。target_raw 是从
    t+1 开盘开始连续 H 个 open-to-open 收益的几何累计值。若持有期内主力
    换月，则每一天使用当时因果选出的实际主力合约收益，避免跨合约价格跳空。
    """
    panel_fields = [
        "product", "trade_date", "contract", "open", "high", "low", "close",
        "settle", "volume", "open_interest", "amount", "delist_date",
        "contract_close_return", "contract_fwd_open_return", "lag_activity",
        "direct_roll_yield", "basis", "spot", "spot_age", "warehouse_receipt",
        "warehouse_receipt_change", "warehouse_receipt_age", "warehouse_receipt_conflict",
        "stock", "stock_age", "available_stock", "available_stock_age",
    ]
    panel_fields = [name for name in panel_fields if name in data.columns]
    working = data[panel_fields].copy()
    if forecast_horizon < 1:
        raise ValueError("forecast_horizon 必须至少为 1")
    key = selected.set_index(["product", "trade_date"])["contract"]
    rows: List[pd.Series] = []

    groups = working.groupby(["product", "trade_date"], sort=True)
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
        [add_forward_targets(group) for _, group in tqdm(
            product_groups, desc=f"[Labels] 未来{forecast_horizon}日累计收益", unit="品种"
        )],
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
# 3. 工业级特征交接包加载（特征已经预处理，禁止二次标准化）
# =============================================================================


def find_handoff_file(root: Path, configured_path: Optional[str], filename: str) -> Path:
    if configured_path:
        path = Path(configured_path)
        if path.exists():
            return path
        raise FileNotFoundError(f"配置的交接文件不存在: {path}")
    direct = root / filename
    if direct.exists():
        return direct
    candidates = [
        path for path in root.rglob(filename)
        if "__MACOSX" not in path.parts and not path.name.startswith("._")
    ]
    if not candidates:
        raise FileNotFoundError(f"项目中找不到交接文件 {filename}")
    # 优先尺寸较大的真实文件，再优先目录层级较浅者。
    return sorted(candidates, key=lambda x: (-x.stat().st_size, len(x.parts)))[0]


def identify_fundamental_features(registry: pd.DataFrame) -> List[str]:
    """依据注册表元数据识别低频基本面/库存/仓单/期限结构类连续特征。"""
    text_cols = [c for c in ["source_factor", "category", "economic_meaning", "transformation"] if c in registry]
    text = registry[text_cols].fillna("").astype(str).agg(" ".join, axis=1)
    pattern = r"库存|仓单|现货|基差|期限结构|carry|便利收益|供需|inventory|warehouse|spot|basis|fundamental"
    continuous = registry.get("usage_type", pd.Series("", index=registry.index)).ne("quality_flag")
    return registry.loc[continuous & text.str.contains(pattern, case=False, regex=True), "feature_name"].tolist()


def causal_ffill_fundamentals(
    features: pd.DataFrame,
    registry: pd.DataFrame,
    fundamental_names: Sequence[str],
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """只按品种向前填充低频基本面值；绝不使用 bfill。

    交接包已把缺失填0，因此优先利用同一 source_factor 的 quality_flag 还原
    缺失位置，再按 trade_date 升序 ffill，首个有效公布日前仍填0。
    """
    result = features.sort_values(["product", "trade_date"]).copy()
    restored_missing = 0
    filled_values = 0
    for name in fundamental_names:
        source = registry.loc[registry["feature_name"].eq(name), "source_factor"]
        if source.empty:
            continue
        source_name = source.iloc[0]
        flags = registry.loc[
            registry.get("usage_type", pd.Series("", index=registry.index)).eq("quality_flag")
            & registry["source_factor"].eq(source_name),
            "feature_name",
        ].tolist()
        missing_mask = pd.Series(False, index=result.index)
        for flag in flags:
            if flag in result:
                missing_mask |= result[flag].gt(0)
        restored_missing += int(missing_mask.sum())
        series = result[name].mask(missing_mask)
        forwarded = series.groupby(result["product"], sort=False).ffill()
        filled_values += int((series.isna() & forwarded.notna()).sum())
        result[name] = forwarded.fillna(0.0)
    return result, {"fundamental_missing_restored": restored_missing, "fundamental_values_ffilled": filled_values}


def sector_neutralize_features(
    features: pd.DataFrame,
    continuous_names: Sequence[str],
) -> pd.DataFrame:
    """逐日逐板块减均值；quality_flag 不参与中性化。"""
    result = features.copy()
    group_keys = [result["trade_date"], result["sector"]]
    for name in tqdm(continuous_names, desc="[Features] 行业中性化", unit="因子"):
        result[name] = result[name] - result.groupby(group_keys, sort=False)[name].transform("mean")
    return result


def load_feature_handoff(root: Path, cfg: Config) -> Tuple[List[str], pd.DataFrame, Dict[str, object]]:
    registry_path = find_handoff_file(root, cfg.feature_registry_path, "ml_feature_registry.csv")
    matrix_path = find_handoff_file(root, cfg.feature_matrix_path, "ml_features_development.csv.gz")
    print(f"[Features] registry={registry_path}", flush=True)
    print(f"[Features] matrix={matrix_path}", flush=True)

    registry = pd.read_csv(registry_path)
    if "feature_name" not in registry.columns:
        raise ValueError("ml_feature_registry.csv 缺少 feature_name 列")
    feature_names = registry["feature_name"].dropna().astype(str).str.strip().tolist()
    if not feature_names or len(feature_names) != len(set(feature_names)):
        raise ValueError("feature_name 为空或存在重复")
    if cfg.expected_feature_count and len(feature_names) != cfg.expected_feature_count:
        raise ValueError(f"注册表特征数应为 {cfg.expected_feature_count}，实际为 {len(feature_names)}")

    usecols = ["trade_date", "product", *feature_names]
    features = pd.read_csv(matrix_path, usecols=usecols, parse_dates=["trade_date"], compression="infer")
    features["product"] = features["product"].astype(str).str.upper().str.strip()
    features = features[features["trade_date"].dt.year.between(cfg.data_start_year, cfg.data_end_year)].copy()
    if features.duplicated(["trade_date", "product"]).any():
        raise ValueError("ml_features_development.csv.gz 的 trade_date+product 不是唯一键")

    expected_products = set(cfg.products)
    available_products = set(features["product"].unique())
    missing_products = sorted(expected_products - available_products)
    unexpected_products = sorted(available_products - expected_products)
    if missing_products:
        raise ValueError(
            "59因子矩阵未覆盖严格30品种池。缺失=" + str(missing_products)
            + "；矩阵额外品种=" + str(unexpected_products)
            + "。请补齐缺失品种特征后再训练，禁止静默替换或缩减品种池。"
        )
    features = features[features["product"].isin(cfg.products)].copy()
    features["sector"] = features["product"].map(lambda p: PRODUCT_METADATA[p][0])

    fundamental_names = identify_fundamental_features(registry)
    features, ffill_report = causal_ffill_fundamentals(features, registry, fundamental_names)
    quality_flags = registry.loc[
        registry.get("usage_type", pd.Series("", index=registry.index)).eq("quality_flag"), "feature_name"
    ].tolist()
    continuous_names = [name for name in feature_names if name not in set(quality_flags)]
    features = sector_neutralize_features(features, continuous_names)

    features[feature_names] = features[feature_names].astype(np.float32)
    values = features[feature_names].to_numpy(dtype=np.float32, copy=False)
    if not np.isfinite(values).all():
        raise ValueError("处理后的交接特征包含 NaN/inf")
    report = {
        "feature_registry_path": str(registry_path), "feature_matrix_path": str(matrix_path),
        "registered_feature_count": len(feature_names), "feature_matrix_rows": len(features),
        "feature_matrix_date_min": str(features["trade_date"].min().date()),
        "feature_matrix_date_max": str(features["trade_date"].max().date()),
        "feature_matrix_products": int(features["product"].nunique()),
        "strict_product_pool": list(cfg.products), "missing_products": missing_products,
        "unexpected_products_excluded": unexpected_products,
        "sector_neutralized_feature_count": len(continuous_names),
        "quality_flags_not_neutralized": len(quality_flags),
        "fundamental_features_ffilled": fundamental_names,
        **ffill_report,
    }
    return feature_names, features, report


def merge_handoff_features(
    panel: pd.DataFrame,
    features: pd.DataFrame,
    feature_names: Sequence[str],
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    overlap = set(feature_names) & set(panel.columns)
    if overlap:
        panel = panel.drop(columns=sorted(overlap))
    before = len(panel)
    merged = panel.merge(
        features[["trade_date", "product", *feature_names]],
        on=["trade_date", "product"],
        how="left",
        validate="one_to_one",
        indicator="_feature_merge",
    )
    matched = int(merged["_feature_merge"].eq("both").sum())
    coverage = matched / before if before else 0.0
    unmatched_by_year = (
        merged.loc[merged["_feature_merge"].ne("both"), "trade_date"].dt.year.value_counts().sort_index().to_dict()
    )
    merged = merged.drop(columns="_feature_merge")
    # 交接包只到2024；2025无特征的行自然不能构造模型样本，但仍保留面板供审计。
    report = {
        "panel_rows_before_feature_merge": before,
        "feature_rows_matched": matched,
        "feature_merge_coverage": coverage,
        "unmatched_feature_rows_by_year": {str(k): int(v) for k, v in unmatched_by_year.items()},
    }
    print(f"[Features] left merge 覆盖率={coverage:.2%} ({matched:,}/{before:,})", flush=True)
    return merged, report


def resolve_feature_mode(root: Path, cfg: Config) -> str:
    mode = cfg.feature_mode
    if mode not in {"auto", "integrated", "handoff"}:
        raise ValueError("feature_mode 必须是 auto、integrated 或 handoff")
    if mode == "auto":
        return "integrated" if find_integrated_market_file(root, cfg) is not None else "handoff"
    if mode == "integrated" and find_integrated_market_file(root, cfg) is None:
        raise FileNotFoundError("feature_mode=integrated，但没有找到整合合约日表")
    return mode


def build_integrated_features(panel: pd.DataFrame) -> Tuple[List[str], pd.DataFrame, Dict[str, object]]:
    """从整合合约日表的因果主力面板重建 59 个可训练特征。

    连续特征只使用决策日及更早数据，先按品种做时间序列变换，再逐交易日
    做 MAD 稳健标准化和行业去均值。低频字段会结合 age_days 屏蔽陈旧值；
    缺失位置以独立质量标志保留，标准化值最终填 0。
    """
    frame = panel.sort_values(["product", "trade_date"]).copy()
    frame["sector"] = frame["product"].map(lambda p: PRODUCT_METADATA[p][0])
    grouped = frame.groupby("product", sort=False)

    def gpct(column: str, periods: int) -> pd.Series:
        if column not in frame:
            return pd.Series(np.nan, index=frame.index, dtype=float)
        return grouped[column].transform(lambda s: s.pct_change(periods, fill_method=None))

    def gdiff(column: str, periods: int = 1) -> pd.Series:
        if column not in frame:
            return pd.Series(np.nan, index=frame.index, dtype=float)
        return grouped[column].transform(lambda s: s.diff(periods))

    def rolling_stat(series: pd.Series, window: int, statistic: str, min_periods: Optional[int] = None) -> pd.Series:
        minimum = min_periods or max(3, window // 2)
        by_product = series.groupby(frame["product"], sort=False)
        if statistic == "std":
            return by_product.transform(lambda s: s.rolling(window, min_periods=minimum).std())
        if statistic == "skew":
            return by_product.transform(lambda s: s.rolling(window, min_periods=minimum).skew())
        if statistic == "sum":
            return by_product.transform(lambda s: s.rolling(window, min_periods=minimum).sum())
        if statistic == "mean":
            return by_product.transform(lambda s: s.rolling(window, min_periods=minimum).mean())
        raise ValueError(f"不支持的 rolling statistic: {statistic}")

    def rolling_z(series: pd.Series, window: int = 60) -> pd.Series:
        mean = rolling_stat(series, window, "mean")
        std = rolling_stat(series, window, "std")
        return (series - mean) / std.where(std.gt(1e-12))

    def fresh(column: str, age_column: str, max_age: int) -> pd.Series:
        if column not in frame:
            return pd.Series(np.nan, index=frame.index, dtype=float)
        value = pd.to_numeric(frame[column], errors="coerce")
        if age_column in frame:
            age = pd.to_numeric(frame[age_column], errors="coerce")
            value = value.mask(age.gt(max_age))
        return value

    close = pd.to_numeric(frame["close"], errors="coerce")
    ret1 = gpct("close", 1)
    abs_move = ret1.abs()
    log_volume = np.log1p(pd.to_numeric(frame.get("volume"), errors="coerce").clip(lower=0))
    log_oi = np.log1p(pd.to_numeric(frame.get("open_interest"), errors="coerce").clip(lower=0))

    continuous: Dict[str, pd.Series] = {
        "return_1": ret1,
        "momentum_5": gpct("close", 5),
        "momentum_10": gpct("close", 10),
        "momentum_20": gpct("close", 20),
        "momentum_60": gpct("close", 60),
        "momentum_120": gpct("close", 120),
    }
    for window in (20, 60, 120):
        numerator = gpct("close", window).abs()
        denominator = rolling_stat(abs_move, window, "sum", max(5, window // 2))
        continuous[f"trend_efficiency_{window}"] = numerator / denominator.where(denominator.gt(1e-12))
    for window in (5, 10, 20, 60, 120):
        continuous[f"realized_vol_{window}"] = rolling_stat(ret1, window, "std") * np.sqrt(252.0)
    downside = ret1.where(ret1.lt(0), 0.0)
    continuous.update({
        "downside_vol_20": rolling_stat(downside, 20, "std") * np.sqrt(252.0),
        "realized_skew_20": rolling_stat(ret1, 20, "skew"),
        "realized_skew_60": rolling_stat(ret1, 60, "skew"),
        "high_low_range": (frame["high"] - frame["low"]) / close.where(close.gt(0)),
        "candle_body": (frame["close"] - frame["open"]) / frame["open"].where(frame["open"].gt(0)),
        "close_settle_spread": (frame["close"] - frame["settle"]) / frame["settle"].where(frame["settle"].gt(0)),
        "log_volume": log_volume,
        "volume_change_1": gpct("volume", 1),
        "volume_change_5": gpct("volume", 5),
        "volume_z_20": rolling_z(log_volume, 20),
        "volume_z_60": rolling_z(log_volume, 60),
        "log_open_interest": log_oi,
        "oi_change_1": gpct("open_interest", 1),
        "oi_change_5": gpct("open_interest", 5),
        "oi_z_20": rolling_z(log_oi, 20),
        "oi_z_60": rolling_z(log_oi, 60),
        "volume_oi_turnover": frame["volume"] / frame["open_interest"].where(frame["open_interest"].gt(0)),
    })
    amount = pd.to_numeric(frame.get("amount"), errors="coerce")
    amihud_daily = ret1.abs() / amount.where(amount.gt(0))
    continuous["amihud_20"] = rolling_stat(amihud_daily, 20, "mean")

    roll = pd.to_numeric(frame.get("roll_yield"), errors="coerce")
    continuous.update({
        "roll_yield": roll,
        "roll_change_5": roll.groupby(frame["product"], sort=False).transform(lambda s: s.diff(5)),
        "roll_change_20": roll.groupby(frame["product"], sort=False).transform(lambda s: s.diff(20)),
    })

    spot = fresh("spot", "spot_age", 30)
    continuous["spot_momentum_5"] = spot.groupby(frame["product"], sort=False).transform(
        lambda s: s.pct_change(5, fill_method=None)
    )
    continuous["spot_momentum_20"] = spot.groupby(frame["product"], sort=False).transform(
        lambda s: s.pct_change(20, fill_method=None)
    )

    warehouse = fresh("warehouse_receipt", "warehouse_receipt_age", 45)
    stock = fresh("stock", "stock_age", 60)
    available_stock = fresh("available_stock", "available_stock_age", 60)
    for prefix, series in (("warehouse", warehouse), ("stock", stock), ("available_stock", available_stock)):
        continuous[f"{prefix}_level"] = rolling_z(np.log1p(series.clip(lower=0)), 60)
        continuous[f"{prefix}_change_5"] = series.groupby(frame["product"], sort=False).transform(
            lambda s: s.pct_change(5, fill_method=None)
        )
    # 为仓单与库存分别保留一个更短周期的供需冲击，共计 45 个连续特征。
    continuous["warehouse_change_1"] = warehouse.groupby(frame["product"], sort=False).transform(
        lambda s: s.pct_change(1, fill_method=None)
    )
    continuous["stock_change_1"] = stock.groupby(frame["product"], sort=False).transform(
        lambda s: s.pct_change(1, fill_method=None)
    )

    if len(continuous) != 45:
        raise AssertionError(f"整合表连续特征设计应为45个，实际为{len(continuous)}个")
    quality_bases = [
        "momentum_120", "trend_efficiency_120", "realized_vol_120", "realized_skew_60",
        "amihud_20", "roll_yield", "spot_momentum_20", "warehouse_level",
        "warehouse_change_5", "stock_level", "stock_change_5", "available_stock_level",
        "available_stock_change_5", "volume_oi_turnover",
    ]

    continuous_frame = pd.DataFrame({
        name: pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
        for name, values in continuous.items()
    }, index=frame.index)
    raw_missing = continuous_frame.isna()

    # 向量化逐日 MAD 稳健标准化，避免 45 个特征逐日执行 Python 回调。
    date_keys = frame["trade_date"]
    daily_median = continuous_frame.groupby(date_keys, sort=False).transform("median")
    absolute_deviation = (continuous_frame - daily_median).abs()
    daily_mad = absolute_deviation.groupby(date_keys, sort=False).transform("median")
    daily_std = continuous_frame.groupby(date_keys, sort=False).transform("std")
    scale = (1.4826 * daily_mad).where(daily_mad.gt(1e-12), daily_std)
    scale = scale.where(scale.gt(1e-12))
    standardized_frame = ((continuous_frame - daily_median) / scale).clip(-8.0, 8.0)
    sector_keys = [frame["trade_date"], frame["sector"]]
    standardized_frame = standardized_frame - standardized_frame.groupby(
        sector_keys, sort=False
    ).transform("mean")

    output = frame[["trade_date", "product", "sector"]].copy()
    feature_names: List[str] = []
    for name in continuous:
        feature_name = f"{name}__cs_robust_z"
        output[feature_name] = standardized_frame[name].fillna(0.0).astype(np.float32)
        feature_names.append(feature_name)

    for name in quality_bases:
        feature_name = f"{name}__missing"
        output[feature_name] = raw_missing[name].astype(np.float32)
        feature_names.append(feature_name)

    if len(feature_names) != 59 or output.duplicated(["trade_date", "product"]).any():
        raise AssertionError("整合表特征必须恰为59列且 trade_date+product 唯一")
    values = output[feature_names].to_numpy(dtype=np.float32, copy=False)
    if not np.isfinite(values).all():
        raise ValueError("整合表生成的特征仍包含 NaN/inf")
    report = {
        "feature_mode": "integrated",
        "registered_feature_count": len(feature_names),
        "feature_matrix_rows": len(output),
        "feature_matrix_date_min": str(output["trade_date"].min().date()),
        "feature_matrix_date_max": str(output["trade_date"].max().date()),
        "feature_matrix_products": int(output["product"].nunique()),
        "strict_product_pool": sorted(output["product"].unique().tolist()),
        "feature_note": "由整合合约日表的因果主力面板重建；45个连续特征+14个缺失质量标志",
    }
    return feature_names, output, report


# =============================================================================
# 4. 序列样本：[Batch, Num_Factors, Sequence_Length]
# =============================================================================


class NumpySequenceDataset(Dataset):
    def __init__(self, x: np.ndarray, product_ids: np.ndarray, y_reg: np.ndarray, y_cls: np.ndarray, indices: np.ndarray):
        self.x = x
        self.product_ids = product_ids
        self.y_reg = y_reg
        self.y_cls = y_cls
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        j = self.indices[idx]
        return (
            torch.from_numpy(self.x[j]),
            torch.tensor(self.product_ids[j], dtype=torch.long),
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
                "product_id": PRODUCT_TO_ID[product],
                "sector": PRODUCT_METADATA[product][0],
                "decision_date": current["trade_date"],
                "sample_date": current["sample_date"],
                "exit_date": current["exit_date"],
                "contract": current["target_contract"],
                "target_label": float(current["target_label"]),
                "raw_return": float(current["execution_return"]),
                "forward_horizon_return": float(current["target_raw"]),
                "volume": float(current["volume"]) if np.isfinite(current.get("volume", np.nan)) else np.nan,
                "amount": float(current["amount"]) if np.isfinite(current.get("amount", np.nan)) else np.nan,
                "open_interest": float(current["open_interest"]) if np.isfinite(current.get("open_interest", np.nan)) else np.nan,
                "lag_activity": float(current["lag_activity"]) if np.isfinite(current.get("lag_activity", np.nan)) else np.nan,
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
    """输入 [B,F,20]；返回 (方向logits, 未来配置期超额收益回归值)。"""

    def __init__(
        self,
        num_factors: int,
        model_dim: int,
        dropout: float,
        attention_heads: int = 4,
        transformer_ff_dim: int = 96,
        transformer_layers: int = 1,
        num_products: int = 30,
        product_embedding_dim: int = 8,
    ):
        super().__init__()
        if model_dim % attention_heads != 0:
            raise ValueError("model_dim 必须能被 attention_heads 整除")
        self.local_embedding = nn.Sequential(
            nn.Conv1d(num_factors, model_dim, kernel_size=3, padding=1, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.product_embedding = nn.Embedding(num_products, product_embedding_dim)
        self.product_fusion = nn.Linear(model_dim + product_embedding_dim, model_dim)
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

    def forward(
        self,
        x: torch.Tensor,
        product_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.local_embedding(x)      # [B, D, T]
        x = x.transpose(1, 2)            # [B, T, D]
        if product_ids is None:
            product_ids = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        product_context = self.product_embedding(product_ids).unsqueeze(1).expand(-1, x.shape[1], -1)
        x = self.product_fusion(torch.cat([x, product_context], dim=-1))
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
    product_ids: np.ndarray,
    y_reg: np.ndarray,
    y_cls: np.ndarray,
    indices: np.ndarray,
    cfg: Config,
) -> DataLoader:
    return DataLoader(
        NumpySequenceDataset(x, product_ids, y_reg, y_cls, indices),
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
        for xb, product_id, y_cls, y_reg in loader:
            xb = xb.to(device)
            product_id = product_id.to(device)
            y_cls, y_reg = y_cls.to(device), y_reg.to(device)
            logits, reg_out = model(xb, product_id)
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
    product_ids: np.ndarray,
    fit_idx: np.ndarray,
    val_idx: np.ndarray,
    cfg: Config,
    fold_seed: int,
) -> Tuple[nn.Module, List[Dict[str, float]], int, float]:
    set_seed(fold_seed)
    model = make_multitask_model(x.shape[1], cfg).to(cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    bce = nn.BCEWithLogitsLoss(); huber = nn.HuberLoss(delta=cfg.huber_delta)
    fit_loader = make_loader(x, product_ids, y_reg, y_cls, fit_idx, cfg)
    val_loader = make_loader(x, product_ids, y_reg, y_cls, val_idx, cfg)
    history: List[Dict[str, float]] = []; best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf"); best_epoch = 1; stale = 0
    for epoch in range(1, cfg.max_epochs + 1):
        model.train(); totals: List[float] = []; cls_values: List[float] = []; reg_values: List[float] = []; counts: List[int] = []
        for xb, product_id, batch_cls, batch_reg in fit_loader:
            xb = xb.to(cfg.device)
            product_id = product_id.to(cfg.device)
            batch_cls, batch_reg = batch_cls.to(cfg.device), batch_reg.to(cfg.device)
            optimizer.zero_grad(set_to_none=True)
            logits, reg_out = model(xb, product_id)
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
    product_ids: np.ndarray,
    train_idx: np.ndarray,
    epochs: int,
    cfg: Config,
    fold_seed: int,
) -> nn.Module:
    set_seed(fold_seed); model = make_multitask_model(x.shape[1], cfg).to(cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    bce = nn.BCEWithLogitsLoss(); huber = nn.HuberLoss(delta=cfg.huber_delta)
    loader = make_loader(x, product_ids, y_reg, y_cls, train_idx, cfg)
    for _ in tqdm(range(max(1, epochs)), desc="[Refit] 多任务完整训练窗", unit="epoch"):
        model.train()
        for xb, product_id, batch_cls, batch_reg in loader:
            xb = xb.to(cfg.device)
            product_id = product_id.to(cfg.device)
            batch_cls, batch_reg = batch_cls.to(cfg.device), batch_reg.to(cfg.device)
            optimizer.zero_grad(set_to_none=True); logits, reg_out = model(xb, product_id)
            total, _, _ = combined_multitask_loss(logits, reg_out, batch_cls, batch_reg, cfg.multitask_alpha, bce, huber)
            total.backward(); nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0); optimizer.step()
    return model


def predict(
    model: nn.Module,
    x: np.ndarray,
    product_ids: np.ndarray,
    indices: np.ndarray,
    cfg: Config,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    dummy = np.zeros(len(x), dtype=np.float32)
    loader = make_loader(x, product_ids, dummy, dummy, indices, cfg)
    probabilities: List[np.ndarray] = []; regressions: List[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for xb, product_id, _, _ in loader:
            logits, reg_out = model(xb.to(cfg.device), product_id.to(cfg.device))
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
    product_ids: np.ndarray,
    indices: np.ndarray,
    cfg: Config,
    seed: int,
) -> np.ndarray:
    if len(indices) == 0:
        return np.zeros(x.shape[1])
    base_pred, _, _ = predict(model, x, product_ids, indices, cfg)
    base_mse = float(np.mean((base_pred - y[indices]) ** 2))
    rng = np.random.default_rng(seed)
    importance = np.zeros(x.shape[1], dtype=float)
    for factor_idx in tqdm(range(x.shape[1]), desc="[Explain] 置换重要性", unit="因子"):
        perturbed = x[indices].copy()
        order = rng.permutation(len(indices))
        perturbed[:, factor_idx, :] = perturbed[order, factor_idx, :]
        perm_pred, _, _ = predict(
            model, perturbed, product_ids[indices], np.arange(len(indices)), cfg
        )
        importance[factor_idx] = max(0.0, float(np.mean((perm_pred - y[indices]) ** 2)) - base_mse)
    return importance


def walk_forward(
    x: np.ndarray,
    y: np.ndarray,
    meta: pd.DataFrame,
    cfg: Config,
    factor_names: Sequence[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, Dict[str, object]]:
    predictions: List[pd.DataFrame] = []
    histories: List[pd.DataFrame] = []
    importances: List[np.ndarray] = []
    best_checkpoint: Optional[Dict[str, object]] = None
    best_global_val = float("inf")

    sample_year = meta["sample_date"].dt.year
    y_cls = meta["target_label"].to_numpy(dtype=np.float32)
    product_ids = meta["product_id"].to_numpy(dtype=np.int64)
    for test_year in range(cfg.first_test_year, cfg.last_test_year + 1):
        # Expanding Window：训练起点固定为 2015，终点随测试年向前扩展。
        train_start = cfg.train_start_year
        test_start = pd.Timestamp(test_year, 1, 1)
        # 20日标签必须在测试年开始前完全实现，禁止年末训练标签跨入测试期。
        train_mask = sample_year.between(train_start, test_year - 1) & meta["exit_date"].lt(test_start)
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
            x, y, y_cls, product_ids, fit_idx, val_idx, cfg, fold_seed
        )
        model = refit_for_epochs(
            x, y, y_cls, product_ids, train_idx_sampled, best_epoch, cfg, fold_seed
        )
        test_pred, test_prob, test_reg = predict(model, x, product_ids, test_idx, cfg)

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
        importances.append(permutation_importance(model, x, y, product_ids, test_idx, cfg, fold_seed))

        if best_val < best_global_val:
            best_global_val = best_val
            best_checkpoint = {
                "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "config": asdict(cfg),
                "factor_names": list(factor_names),
                "test_year": test_year,
                "train_years": [train_start, test_year - 1],
                "training_scheme": "expanding_window",
                "forecast_horizon": cfg.forecast_horizon,
                "selected_epoch": best_epoch,
                "inner_validation_loss": best_val,
                "model_class": "ConvTransformerDualHead",
                "multitask_alpha": cfg.multitask_alpha,
                "train_date_stride": cfg.train_date_stride,
                "input_shape": [len(factor_names), cfg.sequence_length],
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
    """3日 EMA + 30%分位入场 + 40%/60%宽退出缓冲 + 流动性准入。

    入场仍为多头前30%、空头后30%。退出放宽为：多头跌破40%分位才退出，
    空头升破60%分位才退出。决策日流动性最低20%的品种不得新开仓，但
    已有持仓仍按退出线正常保留或平仓。费用、信号加权和GMM门控不变。
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
        # 压力特征在 decision_date 收盘后可得；用决策日概率控制下一 sample_date 开盘仓位。
        stress = stress_probabilities.rename(columns={"date": "decision_date"}).copy()
        stress["decision_date"] = pd.to_datetime(stress["decision_date"])
        scored = scored.merge(stress, on="decision_date", how="left")
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
        long_exit_threshold = float(day["smoothed_prediction"].quantile(cfg.long_exit_quantile))
        short_exit_threshold = float(day["smoothed_prediction"].quantile(cfg.short_exit_quantile))
        signal_by_product = day.set_index("product")["smoothed_prediction"].to_dict()
        available = set(signal_by_product)

        # 已有仓位先按宽缓冲带判断退出，流动性排名不强制平掉旧仓。
        long_products = {
            p for p in long_products
            if p in available and signal_by_product[p] >= long_exit_threshold
        }
        short_products = {
            p for p in short_products
            if p in available and signal_by_product[p] <= short_exit_threshold
        }

        liquidity_fields = [name for name in ("volume", "amount", "lag_activity") if name in day]
        day["entry_liquidity"] = np.nan
        liquidity_source = "missing"
        for field in liquidity_fields:
            values = pd.to_numeric(day[field], errors="coerce")
            if values.notna().sum() >= cfg.min_cross_section:
                day["entry_liquidity"] = values
                liquidity_source = field
                break
        valid_liquidity = day["entry_liquidity"].replace([np.inf, -np.inf], np.nan).dropna()
        if valid_liquidity.empty:
            entry_eligible = pd.Series(True, index=day.index)
            liquidity_cutoff = np.nan
            liquidity_excluded_count = 0
        else:
            # 稳定排序严格剔除有效流动性截面的底部20%；并列时按product打破平局。
            liquidity_order = day.loc[valid_liquidity.index].sort_values(
                ["entry_liquidity", "product"], ascending=[True, True], kind="mergesort"
            )
            liquidity_excluded_count = int(math.ceil(len(liquidity_order) * cfg.liquidity_entry_quantile))
            excluded_indices = liquidity_order.index[:liquidity_excluded_count]
            entry_eligible = day["entry_liquidity"].notna()
            entry_eligible.loc[excluded_indices] = False
            liquidity_cutoff = (
                float(liquidity_order.iloc[liquidity_excluded_count - 1]["entry_liquidity"])
                if liquidity_excluded_count else np.nan
            )

        # 低流动性只限制新开仓，不影响满足缓冲条件的旧仓继续持有。
        long_entries = day[
            entry_eligible
            & day["smoothed_prediction"].ge(upper_threshold)
            & day["smoothed_prediction"].gt(cfg.absolute_signal_threshold)
        ]["product"]
        short_entries = day[
            entry_eligible
            & day["smoothed_prediction"].le(lower_threshold)
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
            "long_exit_threshold_q40": long_exit_threshold,
            "short_exit_threshold_q60": short_exit_threshold,
            "liquidity_entry_cutoff_q20": liquidity_cutoff,
            "liquidity_source": liquidity_source,
            "liquidity_excluded_assets": liquidity_excluded_count,
            "entry_eligible_assets": int(entry_eligible.sum()),
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
                "entry_liquidity": record.get("entry_liquidity", np.nan),
                "liquidity_source": liquidity_source,
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
# 7.1 只读评价与审计层：不改变模型、信号、持仓或原回测结果
# =============================================================================


def safe_correlation(x: pd.Series, y: pd.Series, rank: bool = False) -> float:
    pair = pd.concat([pd.to_numeric(x, errors="coerce"), pd.to_numeric(y, errors="coerce")], axis=1).dropna()
    if len(pair) < 3 or pair.iloc[:, 0].nunique() < 2 or pair.iloc[:, 1].nunique() < 2:
        return np.nan
    if rank:
        pair = pair.rank(method="average")
    return float(pair.iloc[:, 0].corr(pair.iloc[:, 1]))


def calculate_daily_ic(
    predictions: pd.DataFrame,
    signal_col: str = "prediction",
    target_col: str = "target_excess",
) -> pd.DataFrame:
    columns = ["date", "pearson_ic", "rank_ic", "n_assets", "signal_column", "target_column"]
    if predictions.empty or not {"sample_date", signal_col, target_col}.issubset(predictions.columns):
        return pd.DataFrame(columns=columns)
    rows: List[Dict[str, object]] = []
    for date, day in predictions.groupby("sample_date", sort=True):
        valid = day[[signal_col, target_col]].replace([np.inf, -np.inf], np.nan).dropna()
        rows.append({
            "date": date,
            "pearson_ic": safe_correlation(valid[signal_col], valid[target_col]),
            "rank_ic": safe_correlation(valid[signal_col], valid[target_col], rank=True),
            "n_assets": len(valid),
            "signal_column": signal_col,
            "target_column": target_col,
        })
    return pd.DataFrame(rows)


def hac_mean_test(values: pd.Series, max_lag: Optional[int] = None) -> Dict[str, float]:
    """Newey-West/Bartlett HAC：检验日度序列均值是否为0。"""
    x = pd.Series(values).replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    n = len(x)
    if n < 3:
        return {"mean": np.nan, "hac_standard_error": np.nan, "t_stat": np.nan, "p_value": np.nan, "hac_lag": np.nan}
    lag = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))) if max_lag is None else int(max_lag)
    lag = max(0, min(lag, n - 1))
    centered = x - x.mean()
    long_run_variance = float(np.dot(centered, centered) / n)
    for k in range(1, lag + 1):
        covariance = float(np.dot(centered[k:], centered[:-k]) / n)
        long_run_variance += 2.0 * (1.0 - k / (lag + 1.0)) * covariance
    standard_error = math.sqrt(max(long_run_variance, 0.0) / n)
    t_stat = float(x.mean() / standard_error) if standard_error > 1e-15 else np.nan
    # 大样本正态近似双侧 p 值，不新增 scipy 硬依赖。
    p_value = float(math.erfc(abs(t_stat) / math.sqrt(2.0))) if np.isfinite(t_stat) else np.nan
    return {
        "mean": float(x.mean()), "hac_standard_error": standard_error,
        "t_stat": t_stat, "p_value": p_value, "hac_lag": lag,
    }


def circular_block_bootstrap_mean_ci(
    values: pd.Series,
    block_length: int = 20,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> Dict[str, float]:
    """固定随机种子的圆形块自助法均值95%区间，仅用于结果审计。"""
    x = pd.Series(values).replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    n = len(x)
    if n < 3:
        return {"bootstrap_mean": np.nan, "ci_lower": np.nan, "ci_upper": np.nan}
    block = max(1, min(int(block_length), n))
    blocks_needed = int(math.ceil(n / block))
    offsets = np.arange(block)
    rng = np.random.default_rng(seed)
    estimates = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        starts = rng.integers(0, n, size=blocks_needed)
        indices = ((starts[:, None] + offsets[None, :]) % n).ravel()[:n]
        estimates[i] = float(x[indices].mean())
    lower, upper = np.quantile(estimates, [0.025, 0.975])
    return {"bootstrap_mean": float(estimates.mean()), "ci_lower": float(lower), "ci_upper": float(upper)}


def quantile_spread_and_monotonicity(
    frame: pd.DataFrame,
    signal_col: str,
    target_col: str,
    groups: int = 5,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """按日分组后汇总收益；单调性=相邻组平均收益严格递增的比例。"""
    records: List[pd.DataFrame] = []
    spread_rows: List[Dict[str, object]] = []
    for date, day in frame.groupby("sample_date", sort=True):
        valid = day[[signal_col, target_col]].replace([np.inf, -np.inf], np.nan).dropna().copy()
        if len(valid) < groups or valid[signal_col].nunique() < groups:
            continue
        ranked = valid[signal_col].rank(method="first")
        valid["quantile_group"] = pd.qcut(ranked, groups, labels=False, duplicates="drop") + 1
        group_return = valid.groupby("quantile_group", as_index=False)[target_col].mean()
        group_return["date"] = date
        records.append(group_return)
        if group_return["quantile_group"].nunique() == groups:
            ordered = group_return.sort_values("quantile_group")[target_col].to_numpy(dtype=float)
            spread_rows.append({"date": date, "quantile_spread": ordered[-1] - ordered[0]})
    if not records:
        empty = pd.DataFrame(columns=["quantile_group", target_col, "date"])
        return empty, {"monotonicity": np.nan, "quantile_spread_mean": np.nan}
    grouped = pd.concat(records, ignore_index=True)
    average_by_group = grouped.groupby("quantile_group")[target_col].mean().sort_index()
    differences = np.diff(average_by_group.to_numpy(dtype=float))
    monotonicity = float((differences > 0).mean()) if len(differences) else np.nan
    spread = pd.DataFrame(spread_rows)
    return grouped, {
        "monotonicity": monotonicity,
        "quantile_spread_mean": float(spread["quantile_spread"].mean()) if not spread.empty else np.nan,
    }


def add_forward_horizon_returns(predictions: pd.DataFrame, horizons: Sequence[int] = (1, 5, 20)) -> pd.DataFrame:
    """从OOS逐日可交易收益构造展示用前向收益，不参与训练或原策略。"""
    result = predictions.sort_values(["product", "sample_date"]).reset_index(drop=True).copy()
    for horizon in horizons:
        values = np.full(len(result), np.nan, dtype=float)
        for _, positions in result.groupby("product", sort=False).indices.items():
            positions = np.asarray(positions, dtype=int)
            returns = result.iloc[positions]["raw_return"].to_numpy(dtype=float)
            dates = pd.to_datetime(result.iloc[positions]["sample_date"]).to_numpy()
            for i in range(0, len(positions) - horizon + 1):
                window = returns[i:i + horizon]
                calendar_gap = (pd.Timestamp(dates[i + horizon - 1]) - pd.Timestamp(dates[i])).days
                max_calendar_gap = max(7, int(math.ceil(horizon * 2.25)))
                if np.isfinite(window).all() and calendar_gap <= max_calendar_gap:
                    values[positions[i]] = float(np.prod(1.0 + window) - 1.0)
        result[f"audit_return_{horizon}d"] = values
    return result


def summarize_ic_scope(
    frame: pd.DataFrame,
    scope: str,
    signal_col: str,
    target_col: str,
) -> Dict[str, object]:
    daily_ic = calculate_daily_ic(frame, signal_col, target_col)
    valid_pearson = daily_ic["pearson_ic"].dropna()
    valid_rank = daily_ic["rank_ic"].dropna()
    return {
        "scope": scope,
        "signal_column": signal_col,
        "target_column": target_col,
        "observations": int(len(valid_pearson)),
        "calendar_dates_considered": int(len(daily_ic)),
        "ic_mean": float(valid_pearson.mean()),
        "rank_ic": float(valid_rank.mean()),
        "ic_std": float(valid_pearson.std(ddof=1)),
        "rank_ic_std": float(valid_rank.std(ddof=1)),
    }


def build_evaluation_audit(
    predictions: pd.DataFrame,
    daily: pd.DataFrame,
    panel: pd.DataFrame,
    cfg: Config,
) -> Dict[str, object]:
    """生成统一结果展示和稳健性审计；所有输入均为既有模型的既有OOS结果。"""
    pred = add_forward_horizon_returns(predictions)
    ic_daily = calculate_daily_ic(pred, "prediction", "target_excess")
    rank_ic_by_year = (
        ic_daily.assign(year=pd.to_datetime(ic_daily["date"]).dt.year)
        .groupby("year", as_index=False)
        .agg(rank_ic=("rank_ic", "mean"), pearson_ic=("pearson_ic", "mean"),
             rank_ic_std=("rank_ic", "std"), observations=("rank_ic", "count"))
    )

    cost_rows: List[Dict[str, object]] = []
    audit_returns: Dict[int, pd.Series] = {}
    for bps in (0, 3, 5, 10):
        fee = bps / 10000.0
        returns = daily["gross_return"] - fee * daily["absolute_traded"]
        audit_returns[bps] = returns
        metrics = performance_metrics(returns, cfg.annualization)
        annual_return = metrics.get("annual_return", np.nan)
        max_drawdown = metrics.get("max_drawdown", np.nan)
        calmar = annual_return / abs(max_drawdown) if np.isfinite(max_drawdown) and abs(max_drawdown) > 1e-12 else np.nan
        cost_rows.append({
            "scenario": f"{bps}bps_one_way", "one_way_fee_bps": bps,
            "factor_return": metrics.get("total_return", np.nan),
            "annual_return": annual_return, "annual_volatility": metrics.get("annual_volatility", np.nan),
            "sharpe": metrics.get("sharpe_ratio", np.nan), "max_drawdown": max_drawdown,
            "calmar": calmar, "turnover": float(daily["turnover"].mean()),
            "observations": len(returns),
        })
    cost_scenarios = pd.DataFrame(cost_rows)

    # 主文统计对象：统一执行口径下3bps日度多空收益。
    base_returns = audit_returns[3]
    return_hac = hac_mean_test(base_returns)
    return_bootstrap = circular_block_bootstrap_mean_ci(base_returns, seed=cfg.seed)
    ic_hac = hac_mean_test(ic_daily["pearson_ic"], max_lag=max(1, cfg.forecast_horizon - 1))
    ic_bootstrap = circular_block_bootstrap_mean_ci(ic_daily["pearson_ic"], seed=cfg.seed + 1)
    statistical_tests = pd.DataFrame([
        {"test_object": "daily_long_short_return_3bps", **return_hac, **return_bootstrap,
         "bootstrap_block_length": 20, "bootstrap_replications": 1000},
        {"test_object": f"daily_pearson_ic_{cfg.forecast_horizon}d", **ic_hac, **ic_bootstrap,
         "bootstrap_block_length": 20, "bootstrap_replications": 1000},
    ])

    quantile_returns, monotonicity_result = quantile_spread_and_monotonicity(
        pred, "prediction", "target_excess", groups=5
    )
    grouped_return_summary = (
        quantile_returns.groupby("quantile_group", as_index=False)["target_excess"].mean()
        .rename(columns={"target_excess": "mean_realized_excess_return"})
    )
    grouped_return_summary["method"] = "daily_prediction_quintiles; monotonicity=positive_adjacent_mean_return_differences/4"
    grouped_return_summary["monotonicity"] = monotonicity_result["monotonicity"]
    grouped_return_summary["quantile_spread_mean"] = monotonicity_result["quantile_spread_mean"]

    base_metrics = cost_scenarios.loc[cost_scenarios["one_way_fee_bps"].eq(3)].iloc[0]
    valid_ic = ic_daily["pearson_ic"].dropna()
    valid_rank_ic = ic_daily["rank_ic"].dropna()
    ic_mean = float(valid_ic.mean())
    ic_std = float(valid_ic.std(ddof=1))
    unified_panel = pd.DataFrame([{
        "model": "ConvTransformerDualHead",
        "execution_scenario": "3bps_one_way",
        "factor_return": base_metrics["factor_return"],
        "annual_return": base_metrics["annual_return"],
        "sharpe": base_metrics["sharpe"],
        "max_drawdown": base_metrics["max_drawdown"],
        "ic_mean": ic_mean,
        "rank_ic": float(valid_rank_ic.mean()),
        "ic_std": ic_std,
        "ic_ir": ic_mean / ic_std if np.isfinite(ic_std) and ic_std > 1e-12 else np.nan,
        "ic_ir_annualized": False,
        "ir": base_metrics["sharpe"],
        "ir_definition": "annualized mean/std versus zero baseline; numerically identical to reported Sharpe",
        "ir_benchmark": "zero_baseline; equal_weight_multi_factor unavailable in this experiment",
        "p_ic_gt_002": float(valid_ic.gt(0.02).mean()),
        "p_ic_lt_neg002": float(valid_ic.lt(-0.02).mean()),
        "ic_valid_observations": int(len(valid_ic)),
        "t_stat": return_hac["t_stat"],
        "p_value": return_hac["p_value"],
        "hac_test_object": "daily_long_short_return_3bps",
        "monotonicity": monotonicity_result["monotonicity"],
        "monotonicity_method": "daily prediction quintiles; fraction of 4 adjacent full-sample mean returns that increase",
        "calmar": base_metrics["calmar"],
        "turnover": base_metrics["turnover"],
        "annual_volatility": base_metrics["annual_volatility"],
    }])

    holding_rows = [
        summarize_ic_scope(pred, f"holding_period_{h}d", "prediction", f"audit_return_{h}d")
        for h in (1, 5, 20)
    ]
    robustness_holding = pd.DataFrame(holding_rows)

    audit_sector_map = {
        "SC": "能源", "FU": "能源", "BU": "能源",
        "RB": "金属", "HC": "金属", "I": "金属", "JM": "金属", "J": "金属", "SF": "金属",
        "CU": "金属", "AL": "金属", "ZN": "金属", "NI": "金属", "SN": "金属", "AU": "金属", "AG": "金属",
        "TA": "化工", "MA": "化工", "L": "化工", "PP": "化工", "V": "化工", "SA": "化工",
        "M": "农产品", "Y": "农产品", "P": "农产品", "C": "农产品", "CS": "农产品",
        "SR": "农产品", "CF": "农产品", "RM": "农产品",
    }
    pred["audit_sector"] = pred["product"].map(audit_sector_map)
    robustness_sector = pd.DataFrame([
        summarize_ic_scope(group, f"sector_{sector}", "prediction", "target_excess")
        for sector, group in pred.groupby("audit_sector", sort=True)
    ])

    liquidity = panel[["trade_date", "product", "volume", "open_interest"]].copy()
    liquidity["audit_liquidity"] = np.log1p(liquidity["volume"].clip(lower=0)) + np.log1p(
        liquidity["open_interest"].clip(lower=0)
    )
    pred_liq = pred.merge(
        liquidity.rename(columns={"trade_date": "decision_date"})[["decision_date", "product", "audit_liquidity"]],
        on=["decision_date", "product"], how="left", validate="many_to_one",
    )
    liquidity_match_rate = float(pred_liq["audit_liquidity"].notna().mean()) if len(pred_liq) else np.nan
    cutoff = pred_liq.groupby("sample_date")["audit_liquidity"].transform(lambda s: s.quantile(0.20))
    liquid_only = pred_liq[
        pred_liq["audit_liquidity"].notna() & pred_liq["audit_liquidity"].ge(cutoff)
    ].copy()
    robustness_liquidity = pd.DataFrame([
        summarize_ic_scope(pred_liq, "all_products", "prediction", "target_excess"),
        summarize_ic_scope(liquid_only, "exclude_bottom_20pct_among_valid_daily_liquidity", "prediction", "target_excess"),
    ])
    robustness_liquidity["liquidity_definition"] = "log1p(volume)+log1p(open_interest) at decision_date"
    robustness_liquidity["liquidity_match_rate"] = liquidity_match_rate
    robustness_liquidity["rows_after_filter"] = [len(pred_liq), len(liquid_only)]

    delayed = pred.sort_values(["product", "sample_date"]).copy()
    delayed["prediction_delayed_1d"] = delayed.groupby("product", sort=False)["prediction"].shift(1)
    delayed["previous_sample_date"] = delayed.groupby("product", sort=False)["sample_date"].shift(1)
    observed_date_order = {date: i for i, date in enumerate(sorted(pred["sample_date"].dropna().unique()))}
    current_order = delayed["sample_date"].map(observed_date_order)
    previous_order = delayed["previous_sample_date"].map(observed_date_order)
    delayed.loc[current_order.sub(previous_order).ne(1), "prediction_delayed_1d"] = np.nan
    robustness_delay = pd.DataFrame([
        summarize_ic_scope(pred, "original_signal", "prediction", "target_excess"),
        summarize_ic_scope(delayed, "signal_delayed_1_observed_market_trading_day", "prediction_delayed_1d", "target_excess"),
    ])

    methodology = {
        "evaluation_only": True,
        "model_strategy_features_unchanged": True,
        "main_execution_scenario": "3bps one-way; return = existing gross_return - 0.0003 * existing absolute_traded",
        "ic_definition": "daily cross-sectional correlation between existing OOS prediction and configured-horizon target_excess",
        "ic_ir_annualized": False,
        "ir_benchmark": "zero baseline because equal_weight_multi_factor is not produced by this experiment",
        "hac": "Newey-West Bartlett kernel; return test uses automatic lag=floor(4*(T/100)^(2/9)); overlapping-horizon IC test uses forecast_horizon-1 lags; asymptotic normal two-sided p-value",
        "bootstrap": "circular block bootstrap; block=20; 1000 replications; fixed seed; 95% percentile interval",
        "monotonicity": "daily prediction quintiles; full-sample mean return by quintile; fraction of four adjacent differences > 0",
        "holding_periods": "1d/5d/20d all compound the corresponding number of consecutive available OOS raw_return observations; these are evaluation-only audit horizons and do not alter the configured training target",
        "sector_audit_only": audit_sector_map,
        "liquidity_tail": "among rows with valid decision-date liquidity, exclude each day's bottom 20% by log1p(volume)+log1p(open_interest); output records match rate",
        "signal_delay": "within each product use previous prediction only when its sample_date is the immediately previous observed market date in the OOS panel",
        "cost_scenarios_bps": [0, 3, 5, 10],
    }
    return {
        "unified_metrics_panel": unified_panel,
        "prediction_ic_daily": ic_daily,
        "rank_ic_by_year": rank_ic_by_year,
        "cost_scenarios": cost_scenarios,
        "statistical_tests": statistical_tests,
        "grouped_return_monotonicity": grouped_return_summary,
        "robustness_holding_period": robustness_holding,
        "robustness_sector": robustness_sector,
        "robustness_liquidity": robustness_liquidity,
        "robustness_signal_delay": robustness_delay,
        "methodology": methodology,
    }


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
    global FACTOR_COLS
    set_seed(cfg.seed); output = Path(cfg.output_dir); output.mkdir(parents=True, exist_ok=True)
    print("\n========== 中国商品期货 59因子 Conv-Transformer 多任务研究开始 ==========", flush=True)
    print(f"[Config] device={cfg.device}, output={output}, dropout={cfg.dropout:.2f}", flush=True)
    root = resolve_data_root(cfg); print(f"[1/10 数据] root={root}", flush=True)

    feature_mode = resolve_feature_mode(root, cfg)
    print(f"[Features] mode={feature_mode}", flush=True)
    feature_matrix: Optional[pd.DataFrame] = None
    feature_report: Dict[str, object] = {}
    if feature_mode == "handoff":
        FACTOR_COLS, feature_matrix, feature_report = load_feature_handoff(root, cfg)
        print(f"[Features] 交接包 FACTOR_COLS 数量={len(FACTOR_COLS)}", flush=True)
        print(f"[Features] 前5个={FACTOR_COLS[:5]}", flush=True)

    basic = load_basic(root); market, data_report = load_market_data(root, cfg)
    data_report.update(feature_report)
    print(f"[2/10 数据] 行情过滤后 {len(market):,} 行，开始合并合约元数据……", flush=True)
    market = attach_metadata_and_returns(market, basic)
    print("[3/10 主力] 构造因果主力合约……", flush=True); selected = choose_main_contracts(market, cfg)
    print(f"[4/10 面板] 构造期限结构与未来{cfg.forecast_horizon}日双任务标签……", flush=True)
    panel, roll_report = build_product_panel(market, selected, cfg.forecast_horizon)
    if feature_mode == "integrated":
        print("[5/10 特征] 由整合表主力面板生成59个因果训练特征……", flush=True)
        FACTOR_COLS, feature_matrix, feature_report = build_integrated_features(panel)
        print(f"[Features] 整合表 FACTOR_COLS 数量={len(FACTOR_COLS)}", flush=True)
    else:
        print("[5/10 特征] 将标签面板与旧59因子交接包 left merge……", flush=True)
        if feature_matrix is None:
            raise RuntimeError("handoff 模式缺少特征矩阵")
    factors, merge_report = merge_handoff_features(panel, feature_matrix, FACTOR_COLS)
    data_report.update(feature_report)
    data_report.update(merge_report)
    valid_feature_rows = factors[FACTOR_COLS].notna().all(axis=1)
    missing_report = factors[FACTOR_COLS].isna().mean().rename("missing_rate_after_merge").rename_axis("factor").reset_index()
    valid_counts = factors.loc[valid_feature_rows].groupby("trade_date")["product"].nunique()
    data_report.update({
        "selected_main_rows": len(selected), "panel_rows": len(panel),
        "feature_ready_rows": int(valid_feature_rows.sum()),
        "median_valid_products_per_day": float(valid_counts.median()) if len(valid_counts) else np.nan,
    })

    print(f"[6/10 样本] 构造 [样本, {len(FACTOR_COLS)}因子, {cfg.sequence_length}日] 张量……", flush=True)
    x, y, meta = build_sequences(factors, FACTOR_COLS, cfg.sequence_length)
    print(f"[Samples] X={x.shape}, y={y.shape}, range={meta.sample_date.min().date()}~{meta.sample_date.max().date()}", flush=True)
    example_model = make_multitask_model(len(FACTOR_COLS), cfg)
    print(f"[Model] num_factors={len(FACTOR_COLS)}, 参数量={example_model.count_parameters():,}", flush=True)

    print("[7/10 训练] Expanding-Window 多任务训练与样本外预测……", flush=True)
    predictions, history, importance, checkpoint = walk_forward(x, y, meta, cfg, FACTOR_COLS)
    print("[8/10 风控] 因果GMM系统性压力概率……", flush=True)
    stress_features = build_systemic_stress_features(panel)
    stress_probabilities = estimate_causal_gmm_stress(stress_features, cfg)
    print("[9/10 回测] EMA、动态阈值、信号加权与GMM门控……", flush=True)
    daily, positions = build_daily_portfolio(predictions, cfg, stress_probabilities)
    run_assertions(daily, cfg); metrics = make_metrics_table(daily, cfg)

    print("[10/10 输出] 生成原有图表、模型与结果文件……", flush=True)
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
    missing_report.to_csv(output / "feature_merge_missing_report.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"feature_name": FACTOR_COLS}).to_csv(output / "model_feature_list.csv", index=False, encoding="utf-8-sig")
    stress_features.to_csv(output / "systemic_stress_features.csv", index=False, encoding="utf-8-sig")
    stress_probabilities.to_csv(output / "systemic_stress_probability.csv", index=False, encoding="utf-8-sig")
    with open(output / "run_config_and_data_report.json", "w", encoding="utf-8") as file:
        json.dump({"config": asdict(cfg), "data_report": data_report}, file, ensure_ascii=False, indent=2, default=str)

    # 评价层与原结果落盘隔离：审计失败只告警，不影响模型和原结果文件。
    audit_outputs: Dict[str, object] = {}
    print("[Audit] 生成统一指标面板、IC、统计检验及稳健性附表（不改变原结果）……", flush=True)
    try:
        audit_outputs = build_evaluation_audit(predictions, daily, panel, cfg)
        for name, table in audit_outputs.items():
            if isinstance(table, pd.DataFrame):
                table.to_csv(output / f"{name}.csv", index=False, encoding="utf-8-sig")
        with open(output / "evaluation_methodology.json", "w", encoding="utf-8") as file:
            json.dump(audit_outputs["methodology"], file, ensure_ascii=False, indent=2, default=str)
    except Exception as exc:
        warnings.warn(f"评价审计层生成失败，但原模型和原结果已保存：{exc}")
        with open(output / "evaluation_audit_error.txt", "w", encoding="utf-8") as file:
            file.write(f"{type(exc).__name__}: {exc}\n")
    print("\n[Done] 样本外回测统计："); print(metrics.to_string(index=False))
    print(f"\n[Done] 所有结果已保存到: {output}")
    return {
        "metrics": metrics, "daily": daily, "predictions": predictions,
        "stress_probabilities": stress_probabilities,
        "feature_importance": pd.Series(importance, index=FACTOR_COLS),
        "evaluation_audit": audit_outputs, "output_dir": output,
    }


if __name__ == "__main__":
    RESULTS = main(CFG)
