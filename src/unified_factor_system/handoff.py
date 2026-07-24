from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd

from .config import ProjectConfig


KEY_COLUMNS = ["trade_date", "product", "sector"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(
        path,
        index=False,
        encoding="utf-8",
        compression="gzip" if path.suffix == ".gz" else None,
        date_format="%Y-%m-%d",
    )


def _dictionary_markdown(registry: pd.DataFrame) -> str:
    lines = [
        "# 机器学习特征字典",
        "",
        "模型字段以 `ml_feature_registry.csv` 的 `feature_name` 列为准。",
        "",
        "| 字段 | 来源因子 | 变换 | 类别 | 用途 | 经济含义 |",
        "|---|---|---|---|---|---|",
    ]
    for row in registry.itertuples(index=False):
        lines.append(
            f"| `{row.feature_name}` | `{row.source_factor}` | {row.transformation} | "
            f"{row.category} | {row.usage_type} | {row.economic_meaning} |"
        )
    return "\n".join(lines) + "\n"


def _example_train_script() -> str:
    return '''"""最小可运行示例：按统一滚动折训练 Ridge。"""
from pathlib import Path

import pandas as pd
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parent
data = pd.read_csv(
    ROOT / "ml_dataset_development.csv.gz",
    parse_dates=["trade_date", "label_end_date_5d"],
    low_memory=False,
)
registry = pd.read_csv(ROOT / "ml_feature_registry.csv")
folds = pd.read_csv(ROOT / "fold_definitions.csv", parse_dates=["train_start", "train_end", "valid_start", "valid_end"])
features = registry["feature_name"].tolist()
target = "future_return_5d"

predictions = []
for fold in folds.itertuples(index=False):
    train = data.loc[
        data["trade_date"].between(fold.train_start, fold.train_end)
        & data["label_end_date_5d"].lt(fold.valid_start)
        & data[target].notna()
    ]
    valid = data.loc[
        data["trade_date"].between(fold.valid_start, fold.valid_end)
        & data["label_end_date_5d"].le(fold.valid_end)
        & data[target].notna()
    ]
    model = Ridge(alpha=10.0)
    model.fit(train[features], train[target])
    part = valid[["trade_date", "product", "sector", target]].copy()
    part["prediction"] = model.predict(valid[features])
    part["fold_id"] = fold.fold_id
    predictions.append(part)

pd.concat(predictions, ignore_index=True).to_csv(ROOT / "example_predictions.csv", index=False)
print("wrote example_predictions.csv")
'''


def build_training_handoff(
    config: ProjectConfig,
    features: pd.DataFrame,
    labelled_panel: pd.DataFrame,
    registry: pd.DataFrame,
    factor_registry: pd.DataFrame,
) -> dict[str, object]:
    handoff = config.handoff_root
    handoff.mkdir(parents=True, exist_ok=False)
    development_end = pd.Timestamp(config.raw["sample"]["development_end"])
    development_features = features.loc[
        features["trade_date"].le(development_end)
    ].copy()
    label_columns = [
        "trade_date",
        "product",
        "mapped_actual_ts_code",
        "entry_date",
        "future_return_1d",
        "label_end_date_1d",
        "future_return_5d",
        "label_end_date_5d",
        "future_return_20d",
        "label_end_date_20d",
        "entry_execution_available",
        "entry_suspected_limit_lock",
        "entry_suspected_limit_direction",
        "exit_suspected_limit_lock_5d",
        "exit_suspected_limit_direction_5d",
        "exit_execution_available_5d",
    ]
    development_labels = labelled_panel.loc[
        labelled_panel["trade_date"].le(development_end),
        label_columns,
    ].copy()
    for horizon in config.raw["labels"]["horizons"]:
        end_column = f"label_end_date_{horizon}d"
        target_column = f"future_return_{horizon}d"
        future_mask = development_labels[end_column].gt(development_end)
        development_labels.loc[future_mask, [target_column, end_column]] = pd.NA
    dataset = development_features.merge(
        development_labels,
        on=["trade_date", "product"],
        how="left",
        validate="one_to_one",
    )
    if dataset["trade_date"].max() > development_end:
        raise AssertionError("2025 leaked into development handoff")
    feature_names = registry["feature_name"].tolist()
    if dataset[feature_names].isna().any().any():
        raise AssertionError("development handoff contains missing ML features")

    _write_csv(development_features, handoff / "ml_features_development.csv.gz")
    _write_csv(development_labels, handoff / "labels_development.csv.gz")
    _write_csv(dataset, handoff / "ml_dataset_development.csv.gz")
    registry.to_csv(handoff / "ml_feature_registry.csv", index=False, encoding="utf-8")
    factor_registry.to_csv(handoff / "factor_registry.csv", index=False, encoding="utf-8")
    folds = pd.DataFrame(config.raw["folds"])
    folds.to_csv(handoff / "fold_definitions.csv", index=False, encoding="utf-8")
    (handoff / "feature_dictionary.md").write_text(
        _dictionary_markdown(registry),
        encoding="utf-8",
    )
    (handoff / "example_train.py").write_text(
        _example_train_script(),
        encoding="utf-8",
    )
    (handoff / "requirements.txt").write_text(
        "pandas>=2.2\nscikit-learn>=1.6\n",
        encoding="utf-8",
    )
    training_contract = {
        "project_version": config.raw["project_version"],
        "development_end": str(development_end.date()),
        "sealed_year_included": False,
        "key_columns": KEY_COLUMNS,
        "feature_registry": "ml_feature_registry.csv",
        "feature_rule": "use every feature_name row",
        "primary_target": config.raw["labels"]["primary"],
        "target_file": "labels_development.csv.gz",
        "one_file_dataset": "ml_dataset_development.csv.gz",
        "split_file": "fold_definitions.csv",
        "purge_rule": "train label_end_date_5d must be earlier than valid_start",
        "random_split_forbidden": True,
        "feature_selection_uses_target": False,
        "main_contract_rule_amendment": (
            "direct data_final_raw TuShare continuous main series mapped to an "
            "actual contract at T close; T-1 OI reselection explicitly waived"
        ),
        "label_rule": (
            "signal at T close; hold the mapped actual contract from T+1 open "
            "to T+h+1 open; suspected locked entry/exit is excluded"
        ),
        "strategy_protocol": config.raw["execution"],
    }
    (handoff / "training_contract.json").write_text(
        json.dumps(training_contract, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    protocol_source = config.resolve_repo_path(config.raw["protocol_markdown"])
    shutil.copy2(protocol_source, handoff / "UNIFIED_EXPERIMENT_PROTOCOL_SNAPSHOT.md")
    factor_library_source = config.resolve_repo_path(config.raw["factor_library"])
    shutil.copy2(
        factor_library_source,
        handoff / "FACTOR_LIBRARY_V2.1_SNAPSHOT.md",
    )
    (handoff / "PROTOCOL_AMENDMENT.md").write_text(
        "# 协议修订\n\n"
        "经用户明确确认，本轮不采用“用 T−1 可见持仓量重新选择主力”。"
        "模型数据读取 `data_final_raw` 的 TuShare 主力连续序列，并用 T 日连续行情"
        "映射到真实合约，T+1 开盘执行。训练折改为2015起点不变的扩展窗口，"
        "2019—2024逐年验证，2025独立最终测试。全部57个概念因子均登记，"
        "高相关字段保留并提供聚类元数据。基本面统一延迟一个交易日可用。\n",
        encoding="utf-8",
    )
    (handoff / "DATA_PROVENANCE.md").write_text(
        "# 数据血缘\n\n"
        "- 原始合约行情：`data_final_raw/futures_contract_daily_30varieties_2015_2025.csv`。\n"
        "- 主力来源：TuShare 年度 `fut_daily_raw.csv` 中的精确连续代码。\n"
        "- 当前文件夹包含完成计算的开发期特征和标签；训练模型不需要原始数据。\n"
        "- 如需重算因子、改变主力映射、标签或点时规则，必须取得原始数据及完整源码。\n",
        encoding="utf-8",
    )
    readme = """# 建模训练交接包

本文件夹可直接交给建模队友。训练不需要原始 `data_final_raw`，但若要审计或重新计算因子，仍需要原数据和源码。
`factor_registry.csv` 完整登记 V2.1 的57个概念因子；可计算且通过数据质量硬门的因子才会形成模型特征。

## 直接训练

1. 安装 `requirements.txt`。
2. 运行 `python example_train.py` 验证接口。
3. 正式模型读取 `ml_dataset_development.csv.gz`。
4. 模型特征严格取 `ml_feature_registry.csv` 的 `feature_name` 全部行。
5. 主标签为 `future_return_5d`。
6. 必须使用 `fold_definitions.csv` 的扩展窗口，禁止随机切分。
7. 训练行必须满足 `label_end_date_5d < valid_start`。

2025 数据不在本文件夹中，防止开发阶段反复窥视最终样本。
"""
    (handoff / "README.md").write_text(readme, encoding="utf-8")

    manifest_paths = sorted(
        path for path in handoff.iterdir() if path.name != "SHA256SUMS"
    )
    sha_lines = [f"{_sha256(path)}  {path.name}" for path in manifest_paths if path.is_file()]
    (handoff / "SHA256SUMS").write_text("\n".join(sha_lines) + "\n", encoding="utf-8")
    return {
        "handoff_files": len(manifest_paths) + 1,
        "development_rows": int(len(dataset)),
        "feature_count": int(len(feature_names)),
        "development_date_min": str(dataset["trade_date"].min().date()),
        "development_date_max": str(dataset["trade_date"].max().date()),
    }


def build_sealed_evaluation(
    config: ProjectConfig,
    features: pd.DataFrame,
    labelled_panel: pd.DataFrame,
    registry: pd.DataFrame,
) -> dict[str, object]:
    sealed = config.sealed_root
    sealed.mkdir(parents=True, exist_ok=False)
    sealed_start = pd.Timestamp(config.raw["sample"]["sealed_start"])
    sealed_features = features.loc[features["trade_date"].ge(sealed_start)].copy()
    label_columns = [
        "trade_date",
        "product",
        "mapped_actual_ts_code",
        "entry_date",
        "future_return_1d",
        "label_end_date_1d",
        "future_return_5d",
        "label_end_date_5d",
        "future_return_20d",
        "label_end_date_20d",
        "entry_execution_available",
        "entry_suspected_limit_lock",
        "entry_suspected_limit_direction",
        "exit_execution_available_5d",
        "exit_suspected_limit_lock_5d",
        "exit_suspected_limit_direction_5d",
    ]
    sealed_labels = labelled_panel.loc[
        labelled_panel["trade_date"].ge(sealed_start),
        label_columns,
    ].copy()
    _write_csv(sealed_features, sealed / "ml_features_2025.csv.gz")
    _write_csv(sealed_labels, sealed / "labels_2025_evaluator_only.csv.gz")
    registry.to_csv(sealed / "ml_feature_registry.csv", index=False, encoding="utf-8")
    (sealed / "README.md").write_text(
        "# 2025 独立评价区\n\n本目录不得用于开发期调参。`labels_2025_evaluator_only.csv.gz` 只供最终评价脚本读取。\n",
        encoding="utf-8",
    )
    return {
        "sealed_rows": int(len(sealed_features)),
        "sealed_feature_count": int(len(registry)),
        "sealed_date_min": str(sealed_features["trade_date"].min().date()),
        "sealed_date_max": str(sealed_features["trade_date"].max().date()),
    }
