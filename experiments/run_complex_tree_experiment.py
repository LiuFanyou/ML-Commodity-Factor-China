"""More expressive, but still regularized, tree-model extension."""

from pathlib import Path

import run_oos_ml_experiment as base


ROOT = Path(__file__).resolve().parents[1]
RANDOM_STATE = base.RANDOM_STATE
base.MODEL_CONFIGS = {
    "lightgbm": {"params": {"n_estimators": 320, "learning_rate": 0.03, "num_leaves": 31,
        "max_depth": 6, "min_child_samples": 50, "subsample": 0.8, "subsample_freq": 1,
        "colsample_bytree": 0.8, "reg_alpha": 0.2, "reg_lambda": 5.0, "min_split_gain": 0.005,
        "random_state": RANDOM_STATE, "n_jobs": 4, "verbosity": -1}},
    "xgboost": {"params": {"n_estimators": 320, "learning_rate": 0.03, "max_depth": 6,
        "min_child_weight": 20, "gamma": 0.1, "subsample": 0.8, "colsample_bytree": 0.8,
        "reg_alpha": 0.2, "reg_lambda": 5.0, "objective": "reg:squarederror",
        "random_state": RANDOM_STATE, "n_jobs": 4, "verbosity": 0, "tree_method": "hist"}},
    "random_forest": {"params": {"n_estimators": 120, "max_depth": 12, "min_samples_split": 60,
        "min_samples_leaf": 30, "max_features": 0.8, "bootstrap": True, "max_samples": 0.8,
        "ccp_alpha": 1e-6, "random_state": RANDOM_STATE, "n_jobs": 4}},
}
base.OUT = ROOT / "delivery" / "complex_trees"

if __name__ == "__main__":
    base.main()
