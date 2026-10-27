"""
Step 3b: Transformer sequence model.

The idea: instead of hand-made summary features (counts, averages), let a
Transformer read the receiver's recent transactions one by one, the way a
language model reads words, and decide if the current transaction is fraud.

What the model sees for each transaction (a "sequence" of up to 9 tokens):
  token 0     = the current transaction
  tokens 1..8 = the receiver's 8 most recent transactions from EARLIER hours
                (same rule as Step 2: never the current hour, never the future)

Each token describes one transaction:
  - amount (log scale, standardised)
  - is it a TRANSFER?
  - hour of day (as a point on a clock, so 23:00 sits next to 00:00)
  - how many hours before the current transaction it happened
  - is this the current transaction?
  - how much history the receiver has (only on the current token)

What this script does:
  1. Loads the Step 2 table and builds each transaction's history
  2. Trains a small Transformer (PyTorch, CPU is fine)
  3. Stops early when validation PR-AUC stops improving
  4. Picks the alert threshold on validation, reports on test
  5. Logs to MLflow and compares with the XGBoost results from train.py

Run from the project root (after train.py, so the comparison is available):
    uv run python src/train_transformer.py
"""

import time
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
from deltalake import DeltaTable
from sklearn.metrics import average_precision_score
from torch import nn

from train import EXPERIMENT, LABEL, MLFLOW_DB, evaluate, pick_threshold

# ---------------------------------------------------------------------------
# Paths and settings
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURES_PATH = PROJECT_ROOT / "data" / "delta" / "features"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

RUN_NAME = "transformer_seq"
HISTORY_LEN = 8  # receivers average ~5 transactions, so 8 covers most
USE_HOUR = True  # set False to test the model without the night-time signal

# Training only: keep every fraud case but a random 20% of normal ones each
# epoch. This makes training ~5x faster. It shifts the scores' scale but not
# their order, and the alert threshold is chosen on full validation data anyway.
NEG_SAMPLE_RATE = 0.2

D_MODEL = 32
N_HEADS = 4
N_LAYERS = 2
DROPOUT = 0.1
BATCH_SIZE = 512
LEARNING_RATE = 1e-3
MAX_EPOCHS = 10
PATIENCE = 3  # stop after 3 epochs with no validation improvement
SEED = 123
MAX_STEP = 744


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_data() -> pd.DataFrame:
    columns = [
        "step",
        "nameDest",
        "amount",
        "is_transfer",
        "hour_of_day",
        LABEL,
        "split",
    ]
    df = DeltaTable(FEATURES_PATH.as_posix()).to_pandas(columns=columns)
    return df.reset_index(drop=True)


def build_history(df: pd.DataFrame) -> np.ndarray:
    """
    For every row, the row numbers of the receiver's most recent earlier
    transactions (column 0 = most recent). -1 means "no transaction".

    All splits are used together: a test transaction may look back at a
    training-period transaction, because that history really was known.
    """
    dest = pd.factorize(df["nameDest"])[0]
    steps = df["step"].to_numpy()
    order = np.lexsort((steps, dest))  # sort by receiver, then by hour
    d, s = dest[order], steps[order]
    n = len(order)
    position = np.arange(n)

    # Where each receiver's rows start, and where each (receiver, hour) block starts
    new_receiver = np.r_[True, d[1:] != d[:-1]]
    receiver_start = np.maximum.accumulate(np.where(new_receiver, position, 0))
    new_hour = new_receiver | np.r_[True, s[1:] != s[:-1]]
    hour_start = np.maximum.accumulate(np.where(new_hour, position, 0))

    # The k-th most recent earlier transaction sits k places before the hour block
    history_sorted = np.full((n, HISTORY_LEN), -1, dtype=np.int64)
    for k in range(1, HISTORY_LEN + 1):
        idx = hour_start - k
        ok = idx >= receiver_start
        history_sorted[ok, k - 1] = order[idx[ok]]

    history = np.empty_like(history_sorted)
    history[order] = history_sorted
    return history


def build_tokens(df: pd.DataFrame, amount_mean: float, amount_std: float) -> np.ndarray:
    """The per-transaction part of each token."""
    log_amount = (np.log1p(df["amount"].to_numpy()) - amount_mean) / amount_std
    columns = [log_amount, df["is_transfer"].to_numpy()]
    if USE_HOUR:
        angle = 2 * np.pi * df["hour_of_day"].to_numpy() / 24
        columns += [np.sin(angle), np.cos(angle)]
    return np.column_stack(columns).astype(np.float32)


def make_batch(rows, history, tokens, steps):
    """Turn row numbers into a (batch, 1 + HISTORY_LEN, features) tensor + padding mask."""
    b = len(rows)
    hist = history[rows]  # (b, K)
    padding = hist < 0
    hist_safe = np.where(padding, 0, hist)

    # History tokens: transaction features + hours before now + "not current" + 0
    gap = steps[rows][:, None] - steps[hist_safe]
    gap_feature = np.log1p(np.maximum(gap, 0)) / np.log1p(MAX_STEP)
    history_tokens = np.concatenate(
        [
            tokens[hist_safe],
            gap_feature[..., None],
            np.zeros((b, HISTORY_LEN, 2), dtype=np.float32),
        ],
        axis=-1,
    )
    history_tokens[padding] = 0.0

    # Current token: transaction features + gap 0 + "is current" + history size
    history_size = (~padding).sum(axis=1) / HISTORY_LEN
    current_token = np.concatenate(
        [
            tokens[rows],
            np.zeros((b, 1), dtype=np.float32),
            np.ones((b, 1), dtype=np.float32),
            history_size[:, None],
        ],
        axis=-1,
    )[:, None, :]

    x = np.concatenate([current_token, history_tokens], axis=1).astype(np.float32)
    pad = np.concatenate([np.zeros((b, 1), dtype=bool), padding], axis=1)
    return torch.from_numpy(x), torch.from_numpy(pad)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class FraudTransformer(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.embed = nn.Linear(n_features, D_MODEL)
        self.position = nn.Embedding(HISTORY_LEN + 1, D_MODEL)
        layer = nn.TransformerEncoderLayer(
            D_MODEL,
            N_HEADS,
            dim_feedforward=4 * D_MODEL,
            dropout=DROPOUT,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, N_LAYERS, enable_nested_tensor=False
        )
        self.head = nn.Sequential(
            nn.LayerNorm(D_MODEL),
            nn.Linear(D_MODEL, D_MODEL),
            nn.ReLU(),
            nn.Linear(D_MODEL, 1),
        )

    def forward(self, x, pad):
        positions = torch.arange(x.size(1), device=x.device)
        h = self.embed(x) + self.position(positions)
        h = self.encoder(h, src_key_padding_mask=pad)
        # Read the answer off the current transaction's token
        return self.head(h[:, 0]).squeeze(-1)


@torch.no_grad()
def predict(model, rows, history, tokens, steps, batch_size=8192) -> np.ndarray:
    model.eval()
    scores = []
    for start in range(0, len(rows), batch_size):
        x, pad = make_batch(rows[start : start + batch_size], history, tokens, steps)
        scores.append(torch.sigmoid(model(x, pad)).numpy())
    return np.concatenate(scores)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB.as_posix()}")
    mlflow.set_experiment(EXPERIMENT)

    print("Loading features ...")
    df = load_data()
    labels = df[LABEL].to_numpy().astype(np.float32)
    steps = df["step"].to_numpy()
    rows_by_split = {
        s: np.flatnonzero(df["split"].to_numpy() == s)
        for s in ("train", "valid", "test")
    }

    print("Building receiver histories ...")
    history = build_history(df)
    has_history = (history[:, 0] >= 0).mean()
    print(f"  {has_history:.1%} of transactions have at least one earlier transaction")

    train_rows = rows_by_split["train"]
    log_amount_train = np.log1p(df["amount"].to_numpy()[train_rows])
    tokens = build_tokens(df, log_amount_train.mean(), log_amount_train.std())
    n_features = tokens.shape[1] + 3
    del df

    model = FraudTransformer(n_features)
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4
    )
    loss_fn = nn.BCEWithLogitsLoss()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} parameters, {n_features} features per token")

    train_fraud = train_rows[labels[train_rows] == 1]
    train_normal = train_rows[labels[train_rows] == 0]
    valid_rows, test_rows = rows_by_split["valid"], rows_by_split["test"]

    with mlflow.start_run(run_name=RUN_NAME if USE_HOUR else f"{RUN_NAME}_no_hour"):
        mlflow.log_params(
            {
                "model": "transformer",
                "history_len": HISTORY_LEN,
                "use_hour": USE_HOUR,
                "neg_sample_rate": NEG_SAMPLE_RATE,
                "d_model": D_MODEL,
                "n_heads": N_HEADS,
                "n_layers": N_LAYERS,
                "dropout": DROPOUT,
                "batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "n_parameters": n_params,
            }
        )

        best_pr_auc, best_state, best_epoch, waited = -1.0, None, 0, 0
        for epoch in range(1, MAX_EPOCHS + 1):
            started = time.time()
            model.train()
            sampled = rng.choice(
                train_normal, int(len(train_normal) * NEG_SAMPLE_RATE), replace=False
            )
            epoch_rows = rng.permutation(np.concatenate([train_fraud, sampled]))

            total_loss = 0.0
            for start in range(0, len(epoch_rows), BATCH_SIZE):
                batch = epoch_rows[start : start + BATCH_SIZE]
                x, pad = make_batch(batch, history, tokens, steps)
                y = torch.from_numpy(labels[batch])
                optimiser.zero_grad()
                loss = loss_fn(model(x, pad), y)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimiser.step()
                total_loss += loss.item() * len(batch)

            valid_scores = predict(model, valid_rows, history, tokens, steps)
            valid_pr_auc = average_precision_score(labels[valid_rows], valid_scores)
            train_loss = total_loss / len(epoch_rows)
            mlflow.log_metrics(
                {"train_loss": train_loss, "valid_pr_auc_epoch": valid_pr_auc},
                step=epoch,
            )
            print(
                f"  epoch {epoch:>2}  loss {train_loss:.4f}  "
                f"valid PR-AUC {valid_pr_auc:.4f}  ({time.time() - started:.0f}s)"
            )

            if valid_pr_auc > best_pr_auc:
                best_pr_auc, best_epoch, waited = valid_pr_auc, epoch, 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                waited += 1
                if waited >= PATIENCE:
                    print(f"  stopping: no improvement for {PATIENCE} epochs")
                    break

        model.load_state_dict(best_state)
        mlflow.log_metric("best_epoch", best_epoch)

        # Same evaluation as train.py: threshold from validation, honest score on test
        valid_scores = predict(model, valid_rows, history, tokens, steps)
        threshold = pick_threshold(labels[valid_rows], valid_scores)
        mlflow.log_metric("threshold", threshold)

        results = {}
        for split, rows in (("valid", valid_rows), ("test", test_rows)):
            scores = (
                valid_scores
                if split == "valid"
                else predict(model, rows, history, tokens, steps)
            )
            metrics = evaluate(labels[rows].astype(int), scores, threshold)
            mlflow.log_metrics({f"{split}_{k}": v for k, v in metrics.items()})
            results[split] = metrics

        MODELS_DIR.mkdir(exist_ok=True)
        model_path = MODELS_DIR / f"{RUN_NAME}{'' if USE_HOUR else '_no_hour'}.pt"
        torch.save(model.state_dict(), model_path)
        mlflow.log_artifact(str(model_path))

    # Results table, with the XGBoost runs alongside if train.py has been run
    name = RUN_NAME if USE_HOUR else f"{RUN_NAME}_no_hour"
    rows = []
    for split in ("valid", "test"):
        m = results[split]
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
    summary.to_csv(REPORTS_DIR / f"step3_{name}_results.csv", index=False)

    xgb_path = REPORTS_DIR / "step3_results.csv"
    if xgb_path.exists():
        summary = pd.concat([pd.read_csv(xgb_path), summary], ignore_index=True)

    pd.set_option("display.width", 200)
    print("\n" + "=" * 100)
    print(
        f"RESULTS (best epoch {best_epoch}, threshold {threshold:.3f} chosen on valid)"
    )
    print("=" * 100)
    print(summary.to_string(index=False))
    print("Done.")


if __name__ == "__main__":
    main()
