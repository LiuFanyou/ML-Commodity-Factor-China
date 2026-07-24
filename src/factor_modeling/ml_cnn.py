"""Conv1d + Self-Attention ML model under the shared factor-modeling protocol.

Protocol matches linear_ridge / single_factor_rank / equal_weight_multi:
- data: training_handoff features + labels
- label: future_return_1d; train target: target_cs_rank_1d
- outer CV: rolling 3y train -> 1y test, years 2018-2024
- purge: 1 trading day at train end
- strategy: top/bottom 20% long/short, 0.5/0.5, 1/3/5bp costs via common.py

Only the scoring model differs: a light Conv1d + Transformer encoder on a
per-product lookback window of registered ML features.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

from factor_modeling.common import (
    KEYS,
    build_strategy_returns,
    cross_sectional_rank_target,
    daily_correlations,
    purge_last_dates,
    return_statistics,
)

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "ml_cnn 需要 PyTorch。请安装: pip install torch"
    ) from exc


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SequenceDataset(Dataset):
    def __init__(
        self,
        x: np.ndarray,
        product_ids: np.ndarray,
        y: np.ndarray,
        sample_weights: np.ndarray,
        indices: np.ndarray,
    ):
        self.x = x
        self.product_ids = product_ids
        self.y = y
        self.sample_weights = sample_weights
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        j = int(self.indices[idx])
        return (
            torch.from_numpy(self.x[j]),
            torch.tensor(self.product_ids[j], dtype=torch.long),
            torch.tensor(self.y[j], dtype=torch.float32),
            torch.tensor(self.sample_weights[j], dtype=torch.float32),
        )


class FuturesCNN1D(nn.Module):
    """Input [B, F, T] -> regression score for target_cs_rank_1d."""

    def __init__(
        self,
        num_factors: int,
        model_dim: int,
        dropout: float,
        attention_heads: int,
        transformer_ff_dim: int,
        transformer_layers: int,
        num_products: int,
        product_embedding_dim: int,
    ):
        super().__init__()
        if model_dim % attention_heads != 0:
            raise ValueError("model_dim must be divisible by attention_heads")
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

    def forward(self, x: torch.Tensor, product_ids: torch.Tensor) -> torch.Tensor:
        x = self.local_embedding(x)
        x = x.transpose(1, 2)
        product_context = self.product_embedding(product_ids).unsqueeze(1).expand(-1, x.shape[1], -1)
        x = self.product_fusion(torch.cat([x, product_context], dim=-1))
        x = self.transformer(x)
        shared = self.shared_dropout(self.shared_norm(x.mean(dim=1)))
        return self.reg_head(shared).squeeze(-1)


def build_sequences(
    data: pd.DataFrame,
    feature_names: list[str],
    sequence_length: int,
    product_to_id: dict[str, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """Build per-product lookback tensors; label/target aligned to sequence end date."""
    xs: list[np.ndarray] = []
    ys: list[float] = []
    product_ids: list[int] = []
    rows: list[dict[str, Any]] = []

    for product, group in data.groupby("product", sort=True):
        group = group.sort_values("trade_date").reset_index(drop=True)
        values = group[feature_names].to_numpy(dtype=np.float32)
        targets = group["target_cs_rank_1d"].to_numpy(dtype=np.float64)
        returns = group["future_return_1d"].to_numpy(dtype=np.float64)
        for i in range(sequence_length - 1, len(group)):
            if not np.isfinite(targets[i]) or not np.isfinite(returns[i]):
                continue
            seq = values[i - sequence_length + 1 : i + 1]
            if not np.isfinite(seq).all():
                continue
            current = group.iloc[i]
            xs.append(seq.T.copy())
            ys.append(float(targets[i]))
            product_ids.append(int(product_to_id[str(product)]))
            rows.append({
                "trade_date": current["trade_date"],
                "product": product,
                "sector": current.get("sector", ""),
                "future_return_1d": float(returns[i]),
                "target_cs_rank_1d": float(targets[i]),
                "product_id": int(product_to_id[str(product)]),
            })

    if not xs:
        raise ValueError("no valid sequences; check feature missingness and sequence_length")
    x = np.stack(xs).astype(np.float32)
    y = np.asarray(ys, dtype=np.float32)
    pids = np.asarray(product_ids, dtype=np.int64)
    meta = pd.DataFrame(rows)
    meta["trade_date"] = pd.to_datetime(meta["trade_date"])
    assert x.shape[1:] == (len(feature_names), sequence_length)
    return x, y, pids, meta


def date_equal_sample_weights(meta: pd.DataFrame) -> np.ndarray:
    counts = meta.groupby("trade_date")["product"].transform("count").astype(float)
    return (1.0 / counts).to_numpy(dtype=np.float32)


def scale_features_by_train(
    x: np.ndarray,
    train_idx: np.ndarray,
    feature_names: list[str],
) -> tuple[np.ndarray, StandardScaler]:
    """Fit scaler on train sequences flattened over time; apply to all rows."""
    # x: [N, F, T] -> flatten last dim for per-feature mean/std
    train_panel = x[train_idx].transpose(0, 2, 1).reshape(-1, len(feature_names))
    scaler = StandardScaler()
    scaler.fit(train_panel)
    flat = x.transpose(0, 2, 1).reshape(-1, len(feature_names))
    scaled = scaler.transform(flat).astype(np.float32)
    scaled = scaled.reshape(x.shape[0], x.shape[2], x.shape[1]).transpose(0, 2, 1)
    return scaled, scaler


def make_loader(
    x: np.ndarray,
    product_ids: np.ndarray,
    y: np.ndarray,
    sample_weights: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    return DataLoader(
        SequenceDataset(x, product_ids, y, sample_weights, indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def temporal_train_val_split(
    train_idx: np.ndarray,
    meta: pd.DataFrame,
    validation_fraction: float,
    purge_trading_days: int,
) -> tuple[np.ndarray, np.ndarray]:
    dates = np.array(sorted(meta.loc[train_idx, "trade_date"].dropna().unique()))
    if len(dates) < 40:
        raise ValueError("training window has too few dates for inner validation")
    val_n = max(10, int(math.ceil(len(dates) * validation_fraction)))
    val_dates = set(dates[-val_n:])
    fit_end = max(1, len(dates) - val_n - max(0, purge_trading_days))
    fit_dates = set(dates[:fit_end])
    fit_idx = train_idx[meta.loc[train_idx, "trade_date"].isin(fit_dates).to_numpy()]
    val_idx = train_idx[meta.loc[train_idx, "trade_date"].isin(val_dates).to_numpy()]
    if len(fit_idx) == 0 or len(val_idx) == 0:
        raise ValueError("empty fit/val after temporal split and purge")
    return fit_idx, val_idx


def make_model(num_factors: int, num_products: int, config: dict[str, Any], device: str) -> FuturesCNN1D:
    model = FuturesCNN1D(
        num_factors=num_factors,
        model_dim=int(config["model_dim"]),
        dropout=float(config["dropout"]),
        attention_heads=int(config["attention_heads"]),
        transformer_ff_dim=int(config["transformer_ff_dim"]),
        transformer_layers=int(config["transformer_layers"]),
        num_products=num_products,
        product_embedding_dim=int(config["product_embedding_dim"]),
    )
    return model.to(device)


def evaluate_loss(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: str,
) -> float:
    model.eval()
    totals: list[float] = []
    weights: list[float] = []
    with torch.no_grad():
        for xb, product_id, yb, wb in loader:
            pred = model(xb.to(device), product_id.to(device))
            yb = yb.to(device)
            wb = wb.to(device)
            per = loss_fn(pred, yb)
            totals.append(float((per * wb).sum().item()))
            weights.append(float(wb.sum().item()))
    if not weights or sum(weights) <= 0:
        return float("nan")
    return float(sum(totals) / sum(weights))


def fit_with_validation(
    x: np.ndarray,
    y: np.ndarray,
    product_ids: np.ndarray,
    sample_weights: np.ndarray,
    fit_idx: np.ndarray,
    val_idx: np.ndarray,
    num_products: int,
    config: dict[str, Any],
    fold_seed: int,
) -> tuple[nn.Module, list[dict[str, float]], int, float]:
    device = str(config["device"])
    set_seed(fold_seed)
    model = make_model(x.shape[1], num_products, config, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    loss_fn = nn.HuberLoss(delta=float(config["huber_delta"]), reduction="none")
    fit_loader = make_loader(
        x, product_ids, y, sample_weights, fit_idx,
        int(config["batch_size"]), int(config["num_workers"]),
    )
    val_loader = make_loader(
        x, product_ids, y, sample_weights, val_idx,
        int(config["batch_size"]), int(config["num_workers"]),
    )

    history: list[dict[str, float]] = []
    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    best_epoch = 1
    stale = 0
    max_epochs = int(config["max_epochs"])
    patience = int(config["patience"])

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_totals: list[float] = []
        train_weights: list[float] = []
        for xb, product_id, yb, wb in fit_loader:
            xb = xb.to(device)
            product_id = product_id.to(device)
            yb = yb.to(device)
            wb = wb.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb, product_id)
            per = loss_fn(pred, yb)
            loss = (per * wb).sum() / wb.sum().clamp_min(1e-8)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_totals.append(float((per * wb).sum().item()))
            train_weights.append(float(wb.sum().item()))
        train_loss = float(sum(train_totals) / max(sum(train_weights), 1e-8))
        val_loss = evaluate_loss(model, val_loader, loss_fn, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(
            f"[Train] epoch {epoch:02d}/{max_epochs} | train {train_loss:.6f} | val {val_loss:.6f}",
            flush=True,
        )
        if val_loss < best_val - 1e-8:
            best_val = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    return model, history, best_epoch, best_val


def refit_for_epochs(
    x: np.ndarray,
    y: np.ndarray,
    product_ids: np.ndarray,
    sample_weights: np.ndarray,
    train_idx: np.ndarray,
    epochs: int,
    num_products: int,
    config: dict[str, Any],
    fold_seed: int,
) -> nn.Module:
    device = str(config["device"])
    set_seed(fold_seed)
    model = make_model(x.shape[1], num_products, config, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    loss_fn = nn.HuberLoss(delta=float(config["huber_delta"]), reduction="none")
    loader = make_loader(
        x, product_ids, y, sample_weights, train_idx,
        int(config["batch_size"]), int(config["num_workers"]),
    )
    for _ in range(max(1, epochs)):
        model.train()
        for xb, product_id, yb, wb in loader:
            xb = xb.to(device)
            product_id = product_id.to(device)
            yb = yb.to(device)
            wb = wb.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb, product_id)
            per = loss_fn(pred, yb)
            loss = (per * wb).sum() / wb.sum().clamp_min(1e-8)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
    return model


def predict(
    model: nn.Module,
    x: np.ndarray,
    product_ids: np.ndarray,
    sample_weights: np.ndarray,
    indices: np.ndarray,
    config: dict[str, Any],
) -> np.ndarray:
    device = str(config["device"])
    dummy_y = np.zeros(len(x), dtype=np.float32)
    loader = make_loader(
        x, product_ids, dummy_y, sample_weights, indices,
        int(config["batch_size"]), int(config["num_workers"]),
    )
    outs: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for xb, product_id, _, _ in loader:
            pred = model(xb.to(device), product_id.to(device))
            outs.append(pred.cpu().numpy())
    if not outs:
        return np.array([], dtype=float)
    return np.concatenate(outs)


def _fold_metrics(
    test_year: int,
    selected_epoch: int,
    train_meta: pd.DataFrame,
    predictions: pd.DataFrame,
    strategy: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    daily_ic = daily_correlations(predictions)
    rank_std = float(daily_ic["rank_ic"].std(ddof=1)) if len(daily_ic) > 1 else np.nan
    result: dict[str, Any] = {
        "test_year": test_year,
        "selected_epoch": selected_epoch,
        "train_start": train_meta["trade_date"].min().date() if not train_meta.empty else None,
        "train_end": train_meta["trade_date"].max().date() if not train_meta.empty else None,
        "train_rows": len(train_meta),
        "test_rows": len(predictions),
        "test_days": predictions["trade_date"].nunique(),
        "pearson_ic_mean": float(daily_ic["pearson_ic"].mean()) if not daily_ic.empty else np.nan,
        "rank_ic_mean": float(daily_ic["rank_ic"].mean()) if not daily_ic.empty else np.nan,
        "rank_icir": (
            float(daily_ic["rank_ic"].mean() / rank_std)
            if not daily_ic.empty and rank_std and rank_std > 0
            else np.nan
        ),
        "rank_ic_hit_rate": float(daily_ic["rank_ic"].gt(0).mean()) if not daily_ic.empty else np.nan,
        "target_r2": float(r2_score(predictions["target_cs_rank_1d"], predictions["prediction"])),
        "target_mae": float(mean_absolute_error(predictions["target_cs_rank_1d"], predictions["prediction"])),
    }
    for key, value in return_statistics(strategy["gross_return"], int(config["annualization"])).items():
        result[f"gross_{key}"] = value
    for bps in config["one_way_cost_bps"]:
        bps = float(bps)
        label = str(int(bps)) if bps.is_integer() else str(bps).replace(".", "p")
        column = f"net_{label}bps_return"
        for key, value in return_statistics(strategy[column], int(config["annualization"])).items():
            result[f"net_{label}bps_{key}"] = value
    return result


def _format(value: Any, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    return str(value)


def _markdown(frame: pd.DataFrame) -> str:
    return frame.to_markdown(index=False) if not frame.empty else "暂无记录。"


def write_report(
    path: Path,
    config: dict[str, Any],
    fold_metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    strategy: pd.DataFrame,
    train_history: pd.DataFrame,
    registry: pd.DataFrame,
) -> None:
    daily_ic = daily_correlations(predictions)
    overall = {
        "样本外年份": f"{int(fold_metrics.test_year.min())}—{int(fold_metrics.test_year.max())}",
        "样本外记录": len(predictions),
        "交易日": predictions.trade_date.nunique(),
        "平均Pearson IC": daily_ic.pearson_ic.mean() if not daily_ic.empty else np.nan,
        "平均RankIC": daily_ic.rank_ic.mean() if not daily_ic.empty else np.nan,
        "RankIC胜率": daily_ic.rank_ic.gt(0).mean() if not daily_ic.empty else np.nan,
        "目标R²": r2_score(predictions.target_cs_rank_1d, predictions.prediction),
        "目标MAE": mean_absolute_error(predictions.target_cs_rank_1d, predictions.prediction),
    }
    gross = return_statistics(strategy["gross_return"], int(config["annualization"]))
    strategy_rows = [{"成本": "毛收益", **gross}]
    for bps in config["one_way_cost_bps"]:
        bps = float(bps)
        label = str(int(bps)) if bps.is_integer() else str(bps).replace(".", "p")
        strategy_rows.append({
            "成本": f"单边{bps:g}bp",
            **return_statistics(strategy[f"net_{label}bps_return"], int(config["annualization"])),
        })
    strategy_table = pd.DataFrame(strategy_rows)

    fold_display = fold_metrics.copy()
    for column in fold_display.columns:
        if pd.api.types.is_float_dtype(fold_display[column]):
            fold_display[column] = fold_display[column].map(_format)
    overall_table = pd.DataFrame([{"指标": key, "结果": _format(value)} for key, value in overall.items()])
    strategy_display = strategy_table.copy()
    for column in strategy_display.columns[1:]:
        strategy_display[column] = strategy_display[column].map(_format)

    epoch_summary = (
        train_history.groupby("test_year", as_index=False)
        .agg(best_epoch=("epoch", "max"), min_val_loss=("val_loss", "min"))
        if not train_history.empty
        else pd.DataFrame()
    )
    if not epoch_summary.empty:
        for column in ["min_val_loss"]:
            epoch_summary[column] = epoch_summary[column].map(_format)

    registry_counts = (
        registry.groupby(["usage_type", "classification"], as_index=False)
        .size()
        .rename(columns={"size": "字段数"})
    )
    lines = [
        "# Conv1d+Attention 机器学习模型测试报告",
        "",
        "> 本报告由冻结后的59字段特征接口和真实开发数据自动生成；2025不参与模型选择或评价。",
        "",
        "## 模型与标签选择",
        "",
        "- 模型：Conv1d局部嵌入 + 轻量Self-Attention，对每个品种最近"
        f"{int(config['sequence_length'])}个交易日的注册特征序列回归。",
        "- 原始收益标签：因子形成后的下一交易日开盘至收盘收益。",
        "- 训练标签：同日横截面中心化收益排名，范围[-1,1]；与Ridge及其他横截面策略一致。",
        "- 训练超参：每个三年训练窗口内部按时间切分验证集，用验证损失早停选择epoch，再在净化后的完整训练窗重训。",
        "- 外层验证：三年训练、下一年预测，覆盖2018—2024；边界净化1个交易日。",
        "",
        "## 策略定义",
        "",
        "- 每个信号日按预测值排序，最高20%做多、最低20%做空。",
        "- 多头和空头各占0.5风险资金，组合净敞口0、总敞口1。",
        "- 次日开盘建仓、当日收盘平仓，因此每日往返换手按2计算。",
        "- 净收益仅提供单边1/3/5基点敏感性，不代表已核验的真实手续费与冲击成本。",
        "",
        "## 输入字段组成",
        "",
        _markdown(registry_counts),
        "",
        "## 总体样本外预测结果",
        "",
        _markdown(overall_table),
        "",
        "## 策略总体结果",
        "",
        _markdown(strategy_display),
        "",
        "## 分年度滚动结果",
        "",
        _markdown(fold_display),
        "",
        "## 训练窗口内部早停摘要",
        "",
        _markdown(epoch_summary),
        "",
        "## 使用边界",
        "",
        "1. 本测试验证的是次日开盘至收盘的横截面策略，不代表连续隔夜持仓策略。",
        "2. 若成本后收益为负，不能用毛收益结果声称策略可实盘。",
        "3. 2025因子结果已经被查看，因此没有作为本模型的独立封存期；新的模型级确认应使用2026及以后数据。",
        "4. 没有核验合约乘数、真实手续费、冲击、涨跌停和成交容量，结果只属于研究回测。",
        "5. 本版已去掉原Kaggle脚本中的5日标签、扩展窗、信号EMA、动态分位与GMM压力缩放，仅保留神经网络打分器。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def resolve_device(config: dict[str, Any]) -> str:
    requested = str(config.get("device", "auto")).lower()
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return requested


def run(config_path: str | Path, years: list[int] | None = None) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    root = config_path.parents[2]
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["device"] = resolve_device(config)

    data_dir = root / config["data_directory"]
    output_dir = root / config["output_directory"]
    report_path = root / config["report_file"]
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    registry = pd.read_csv(data_dir / config["registry_file"])
    feature_names = registry["feature_name"].tolist()
    features = pd.read_csv(data_dir / config["feature_file"], parse_dates=["trade_date"])
    labels = pd.read_csv(data_dir / config["label_file"], parse_dates=["trade_date"])
    missing = sorted(set(feature_names) - set(features.columns))
    if missing:
        raise ValueError(f"registered features missing from matrix: {missing}")

    data = features.merge(labels, on=KEYS, how="inner", validate="one_to_one")
    data[config["model_target_column"]] = cross_sectional_rank_target(data, config["raw_return_column"])
    data = data.dropna(subset=[config["model_target_column"], config["raw_return_column"]]).copy()
    data["target_cs_rank_1d"] = data[config["model_target_column"]]
    data["future_return_1d"] = data[config["raw_return_column"]]

    products = sorted(data["product"].astype(str).unique().tolist())
    product_to_id = {name: idx for idx, name in enumerate(products)}
    print(
        f"[Data] rows={len(data):,} products={len(products)} features={len(feature_names)} "
        f"device={config['device']}",
        flush=True,
    )

    x_raw, y, product_ids, meta = build_sequences(
        data,
        feature_names,
        int(config["sequence_length"]),
        product_to_id,
    )
    sample_weights = date_equal_sample_weights(meta)
    print(f"[Samples] sequences={len(meta):,} shape={tuple(x_raw.shape)}", flush=True)

    test_years = [int(y) for y in (years if years is not None else config["outer_test_years"])]
    all_predictions: list[pd.DataFrame] = []
    all_fold_metrics: list[dict[str, Any]] = []
    all_strategy: list[pd.DataFrame] = []
    all_history: list[pd.DataFrame] = []

    for test_year in test_years:
        train_start = int(test_year) - int(config["rolling_train_years"])
        year = meta["trade_date"].dt.year
        train_mask = year.between(train_start, int(test_year) - 1)
        test_mask = year.eq(int(test_year))
        train_idx = np.flatnonzero(train_mask.to_numpy())
        test_idx = np.flatnonzero(test_mask.to_numpy())
        if len(train_idx) == 0 or len(test_idx) == 0:
            print(f"[WalkForward] skip {test_year}: empty train/test")
            continue

        train_meta = meta.iloc[train_idx].copy()
        purged_dates = set(
            purge_last_dates(train_meta, int(config["purge_trading_days"]))["trade_date"]
        )
        train_idx = train_idx[meta.loc[train_idx, "trade_date"].isin(purged_dates).to_numpy()]
        if len(train_idx) == 0:
            print(f"[WalkForward] skip {test_year}: empty after purge")
            continue

        print(
            f"\n[WalkForward] {train_start}-{test_year - 1} -> {test_year} | "
            f"train={len(train_idx):,} test={len(test_idx):,}",
            flush=True,
        )

        # Fold-local scaling: fit on train only, transform all for this fold.
        x_fold, _ = scale_features_by_train(x_raw, train_idx, feature_names)
        fit_idx, val_idx = temporal_train_val_split(
            train_idx,
            meta,
            float(config["validation_fraction"]),
            int(config["purge_trading_days"]),
        )
        fold_seed = int(config["seed"]) + int(test_year)
        _, history, best_epoch, best_val = fit_with_validation(
            x_fold, y, product_ids, sample_weights, fit_idx, val_idx,
            len(products), config, fold_seed,
        )
        model = refit_for_epochs(
            x_fold, y, product_ids, sample_weights, train_idx, best_epoch,
            len(products), config, fold_seed,
        )
        prediction = predict(model, x_fold, product_ids, sample_weights, test_idx, config)

        predicted = meta.iloc[test_idx][
            ["trade_date", "product", "sector", "future_return_1d", "target_cs_rank_1d"]
        ].copy()
        predicted["prediction"] = prediction
        predicted["test_year"] = int(test_year)
        predicted["selected_epoch"] = int(best_epoch)
        strategy = build_strategy_returns(predicted, config)
        strategy["test_year"] = int(test_year)
        fold = _fold_metrics(
            int(test_year),
            int(best_epoch),
            meta.iloc[train_idx],
            predicted,
            strategy,
            config,
        )
        fold["inner_val_loss"] = float(best_val)

        hist = pd.DataFrame(history)
        hist["test_year"] = int(test_year)
        all_predictions.append(predicted)
        all_fold_metrics.append(fold)
        all_strategy.append(strategy)
        all_history.append(hist)

        # Persist fold checkpoint for audit.
        torch.save(
            {
                "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "test_year": int(test_year),
                "selected_epoch": int(best_epoch),
                "feature_names": feature_names,
                "product_to_id": product_to_id,
                "config": {k: v for k, v in config.items() if k != "device"},
            },
            output_dir / f"model_fold_{test_year}.pt",
        )

    if not all_predictions:
        raise RuntimeError("no outer folds produced predictions")

    predictions = pd.concat(all_predictions, ignore_index=True)
    fold_metrics = pd.DataFrame(all_fold_metrics)
    strategy = pd.concat(all_strategy, ignore_index=True)
    train_history = pd.concat(all_history, ignore_index=True)
    daily_ic = daily_correlations(predictions)

    predictions.to_csv(output_dir / "oos_predictions.csv.gz", index=False, compression="gzip")
    strategy.to_csv(output_dir / "strategy_daily_returns.csv", index=False)
    fold_metrics.to_csv(output_dir / "fold_metrics.csv", index=False)
    train_history.to_csv(output_dir / "train_history.csv", index=False)
    daily_ic.to_csv(output_dir / "daily_prediction_ic.csv", index=False)
    write_report(
        report_path,
        config,
        fold_metrics,
        predictions,
        strategy,
        train_history,
        registry,
    )
    summary = {
        "model": "Conv1d+Attention",
        "features": len(feature_names),
        "sequence_length": int(config["sequence_length"]),
        "outer_years": test_years,
        "oos_rows": len(predictions),
        "oos_days": int(predictions["trade_date"].nunique()),
        "mean_rank_ic": float(daily_ic["rank_ic"].mean()) if not daily_ic.empty else float("nan"),
        "gross": return_statistics(strategy["gross_return"], int(config["annualization"])),
        "device": config["device"],
        "report": str(report_path),
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Conv1d+Attention 机器学习模型与日内多空测试")
    parser.add_argument("--config", default="modeling/ml_cnn_v1/config.json")
    parser.add_argument(
        "--years",
        default="",
        help="可选：逗号分隔测试年，例如 2018,2019；默认跑配置中全部外层年份",
    )
    args = parser.parse_args(argv)
    years = [int(x) for x in args.years.split(",") if x.strip()] or None
    print(json.dumps(run(args.config, years=years), ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
