"""
Step 3a: Train and evaluate the XGBoost baseline.

What this script does:
  1. Loads the Step 2 features (data/delta/features) for train, valid and test
  2. Scores two simple baselines so we know what "good" has to beat:
       - PaySim's own rule (isFlaggedFraud)
       - "Flag every transfer to a brand-new receiver"
  3. Trains XGBoost twice: with all 14 features, and without hour_of_day
     (to see how much the model leans on PaySim's night-time quirk)
  4. Picks an alert threshold on the VALIDATION set, then reports
     precision and recall on the TEST set
  5. Logs every run to MLflow and saves the models to models/

Why these metrics?
  Fraud is about 0.3% of transactions, so accuracy is useless. We use:
    - PR-AUC: one score for how well fraud is ranked above normal transactions.
      A model guessing at random scores the base fraud rate, so we also
      report "lift" = PR-AUC / base fraud rate.
    - Precision: of the transactions we flag, how many are really fraud.
    - Recall: of all the fraud, how much we catch.

Run from the project root:
    uv run python src/train.py
"""

from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import xgboost as xgb
from deltalake import DeltaTable
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)

# ---------------------------------------------------------------------------
# Paths and settings
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURES_PATH = PROJECT_ROOT / "data" / "delta" / "features"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
MLFLOW_DB = PROJECT_ROOT / "mlflow.db"
EXPERIMENT = "fraud-detection"

LABEL = "isFraud"
ALL_FEATURES = [
    "hour_of_day",
    "is_transfer",
    "log_amount",
    "dest_count_1h",
    "dest_count_6h",
    "dest_count_24h",
    "dest_amount_1h",
    "dest_amount_6h",
    "dest_amount_24h",
    "dest_count_prior",
    "dest_mean_prior",
    "dest_first_seen",
    "amount_vs_dest_mean",
    "hours_since_dest_last",
]

# Two feature sets to compare
FEATURE_SETS = {
    "xgb_all_features": ALL_FEATURES,
    "xgb_no_hour": [f for f in ALL_FEATURES if f != "hour_of_day"],
}

XGB_PARAMS = {
    "n_estimators": 2000,  # upper limit; early stopping picks the real number
    "learning_rate": 0.05,
    "max_depth": 6,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "tree_method": "hist",  # fast on a laptop CPU
    "eval_metric": "aucpr",  # tune for PR-AUC, not accuracy
    "early_stopping_rounds": 100,  # stop when validation stops improving
    "random_state": 42,
    "n_jobs": -1,
}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_split(split: str) -> pd.DataFrame:
    """Read one split from the Step 2 Delta table (no Spark needed)."""
    columns = ["step", "type", "isFlaggedFraud", LABEL, *ALL_FEATURES]
    df = DeltaTable(FEATURES_PATH.as_posix()).to_pandas(
        columns=columns, filters=[("split", "=", split)]
    )
    return df.sort_values("step").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def pick_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    """The score cut-off that gives the best F1 (balance of precision and recall)."""
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    # The last precision/recall pair has no threshold, so skip it
    return float(thresholds[np.argmax(f1[:-1])])


def evaluate(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    """Ranking quality plus what happens if we alert above the threshold."""
    flagged = scores >= threshold
    true_pos = int((flagged & (y_true == 1)).sum())
    alerts = int(flagged.sum())
    fraud = int(y_true.sum())
    base_rate = fraud / len(y_true)

    precision = true_pos / alerts if alerts else 0.0
    recall = true_pos / fraud if fraud else 0.0
    # A score that is the same for every row (a 0/1 rule) still works here
    pr_auc = average_precision_score(y_true, scores)

    return {
        "pr_auc": pr_auc,
        "lift": pr_auc / base_rate,
        "roc_auc": roc_auc_score(y_true, scores),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if true_pos else 0.0,
        "alerts": alerts,
        "fraud_caught": true_pos,
        "fraud_total": fraud,
        "base_rate": base_rate,
    }


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
def run_rule(name: str, rule, splits: dict) -> dict:
    """A yes/no rule. No training and no threshold to pick."""
    with mlflow.start_run(run_name=name):
        mlflow.log_param("model", "rule")
        results = {}
        for split in ("valid", "test"):
            df = splits[split]
            scores = rule(df).astype(float).to_numpy()
            metrics = evaluate(df[LABEL].to_numpy(), scores, threshold=0.5)
            mlflow.log_metrics({f"{split}_{k}": v for k, v in metrics.items()})
            results[split] = metrics
    return results


def run_xgboost(name: str, features: list[str], splits: dict) -> dict:
    train, valid, test = splits["train"], splits["valid"], splits["test"]

    with mlflow.start_run(run_name=name):
        mlflow.log_params({k: v for k, v in XGB_PARAMS.items() if k != "n_jobs"})
        mlflow.log_param("model", "xgboost")
        mlflow.log_param("features", ",".join(features))
        mlflow.log_param("n_features", len(features))

        model = xgb.XGBClassifier(**XGB_PARAMS)
        model.fit(
            train[features],
            train[LABEL],
            eval_set=[(valid[features], valid[LABEL])],
            verbose=False,
        )
        mlflow.log_metric("best_iteration", model.best_iteration)

        # Choose the alert threshold on validation, never on test
        valid_scores = model.predict_proba(valid[features])[:, 1]
        threshold = pick_threshold(valid[LABEL].to_numpy(), valid_scores)
        mlflow.log_metric("threshold", threshold)

        results = {}
        for split, df in (("valid", valid), ("test", test)):
            scores = model.predict_proba(df[features])[:, 1]
            metrics = evaluate(df[LABEL].to_numpy(), scores, threshold)
            mlflow.log_metrics({f"{split}_{k}": v for k, v in metrics.items()})
            results[split] = metrics

        # Which features did the model rely on most?
        importance = (
            pd.Series(
                model.get_booster().get_score(importance_type="gain"), name="gain"
            )
            .reindex(features)
            .fillna(0.0)
            .sort_values(ascending=False)
        )
        importance = (importance / importance.sum()).rename("share_of_gain")
        REPORTS_DIR.mkdir(exist_ok=True)
        importance_path = REPORTS_DIR / f"{name}_feature_importance.csv"
        importance.to_csv(importance_path, index_label="feature")
        mlflow.log_artifact(str(importance_path))

        # Save the model for Step 4 (serving)
        MODELS_DIR.mkdir(exist_ok=True)
        model_path = MODELS_DIR / f"{name}.json"
        model.save_model(model_path)
        mlflow.log_artifact(str(model_path))

        results["importance"] = importance
        results["threshold"] = threshold
        results["best_iteration"] = model.best_iteration
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB.as_posix()}")
    mlflow.set_experiment(EXPERIMENT)

    print("Loading features ...")
    splits = {s: load_split(s) for s in ("train", "valid", "test")}
    for s, df in splits.items():
        print(f"  {s:<5} {len(df):>10,} rows  {df[LABEL].sum():>6,} fraud")

    all_results = {
        "rule_paysim_flag": run_rule(
            "rule_paysim_flag", lambda df: df["isFlaggedFraud"] == 1, splits
        ),
        "rule_new_receiver_transfer": run_rule(
            "rule_new_receiver_transfer",
            lambda df: (df["dest_first_seen"] == 1) & (df["is_transfer"] == 1),
            splits,
        ),
    }

    for name, features in FEATURE_SETS.items():
        print(f"Training {name} ({len(features)} features) ...")
        all_results[name] = run_xgboost(name, features, splits)

    # Summary table
    rows = []
    for name, res in all_results.items():
        for split in ("valid", "test"):
            m = res[split]
            rows.append(
                {
                    "model": name,
                    "split": split,
                    "base_rate_%": round(m["base_rate"] * 100, 3),
                    "pr_auc": round(m["pr_auc"], 4),
                    "lift": round(m["lift"], 1),
                    "roc_auc": round(m["roc_auc"], 4),
                    "precision": round(m["precision"], 3),
                    "recall": round(m["recall"], 3),
                    "alerts": m["alerts"],
                    "fraud_caught": f"{m['fraud_caught']}/{m['fraud_total']}",
                }
            )
    summary = pd.DataFrame(rows)
    REPORTS_DIR.mkdir(exist_ok=True)
    summary.to_csv(REPORTS_DIR / "step3_results.csv", index=False)

    pd.set_option("display.width", 200)
    print("\n" + "=" * 100)
    print("RESULTS (threshold chosen on valid; test is the honest score)")
    print("=" * 100)
    print(summary.to_string(index=False))

    for name in FEATURE_SETS:
        res = all_results[name]
        print(
            f"\n{name}: stopped at tree {res['best_iteration']}, "
            f"alert threshold {res['threshold']:.3f}"
        )
        print("Top features (share of total gain):")
        print(res["importance"].head(8).round(3).to_string())

    print(f"\nSaved: {REPORTS_DIR / 'step3_results.csv'}, models in {MODELS_DIR}")
    print("Browse runs with:  uv run mlflow ui --backend-store-uri sqlite:///mlflow.db")
    print("Done.")


if __name__ == "__main__":
    main()
