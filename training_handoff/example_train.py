"""最小可运行示例：按统一滚动折训练 Ridge。"""
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
