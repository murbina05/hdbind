"""
03_train_evaluate.py — Group-stratified CV for all embeddings × all models.

Uses GroupKFold(k=5) split by pdb_id to prevent data leakage:
poses from the same complex are never split across train/test.
This is critical because protein-pocket embeddings are identical for
all poses of the same complex.

Output:
    results/metrics.csv        — per-fold metrics for all (embedding, model, fold)
    results/predictions.csv    — per-fold predictions for all poses
    models/{emb}_{model}.pkl   — retrained model + scaler (via joblib)

Usage:
    python 03_train_evaluate.py
    python 03_train_evaluate.py --embedding pocket_esmc600m
    python 03_train_evaluate.py --no-save-models
"""

import argparse
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
OUT_DIR     = Path(__file__).parent
EMB_DIR     = OUT_DIR / "embeddings"
DATA_DIR    = OUT_DIR / "data"
RESULTS_DIR = OUT_DIR / "results"
MODELS_DIR  = OUT_DIR / "models"

N_FOLDS     = 5
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_labels() -> pd.DataFrame:
    """Load data/labels.csv indexed by pose_id."""
    df = pd.read_csv(DATA_DIR / "labels.csv")
    return df.set_index("pose_id")


def get_models() -> dict:
    models = {
        "logreg": LogisticRegression(
            max_iter=1000, random_state=RANDOM_SEED, C=1.0,
            class_weight="balanced"),
        "rf":     RandomForestClassifier(
            n_estimators=200, random_state=RANDOM_SEED, n_jobs=-1,
            class_weight="balanced"),
        "mlp":    MLPClassifier(
            hidden_layer_sizes=(256, 128), max_iter=500,
            random_state=RANDOM_SEED, early_stopping=True, n_iter_no_change=15),
    }
    try:
        from xgboost import XGBClassifier
        models["xgb"] = XGBClassifier(
            n_estimators=200, random_state=RANDOM_SEED,
            use_label_encoder=False, eval_metric="logloss", verbosity=0,
            scale_pos_weight=1,   # set to n_neg/n_pos per fold for best results
        )
    except ImportError:
        pass
    return models


def align_to_labels(
    labels_df: pd.DataFrame,
    pose_ids: np.ndarray,
    pdb_ids_arr: np.ndarray,
    X: np.ndarray,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    """
    Intersect embedding pose_ids with labels_df.
    Returns (common_ids, X_aligned, y_aligned, groups_aligned)
    where groups_aligned is the pdb_id per pose — for GroupKFold.
    """
    pid_to_idx = {pid: i for i, pid in enumerate(pose_ids.tolist())}
    common = [pid for pid in pose_ids.tolist() if pid in labels_df.index]
    if not common:
        raise ValueError("No overlapping pose_ids between embedding and labels")

    X_aligned  = np.stack([X[pid_to_idx[pid]] for pid in common])
    y_aligned  = labels_df.loc[common, "label"].values.astype(int)
    groups     = labels_df.loc[common, "pdb_id"].values  # for GroupKFold

    return common, X_aligned, y_aligned, groups


def compute_metrics(y_true, y_pred, y_prob) -> dict:
    multi = len(np.unique(y_true)) > 1
    return {
        "auroc":    roc_auc_score(y_true, y_prob)           if multi else float("nan"),
        "auprc":    average_precision_score(y_true, y_prob) if multi else float("nan"),
        "mcc":      matthews_corrcoef(y_true, y_pred),
        "f1":       f1_score(y_true, y_pred, zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
    }


# ---------------------------------------------------------------------------
# Core training loop
# ---------------------------------------------------------------------------
def train_evaluate_embedding(
    emb_name: str,
    X: np.ndarray,
    common_ids: list[str],
    y: np.ndarray,
    groups: np.ndarray,
    models: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    5-fold GROUP CV (split by pdb_id) over all models for one embedding.
    Returns (metrics_df, predictions_df).
    """
    gkf = GroupKFold(n_splits=N_FOLDS)
    metrics_rows: list[dict] = []
    pred_rows:    list[dict] = []

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=groups)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        pids_test  = [common_ids[i] for i in test_idx]
        grps_test  = groups[test_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s  = scaler.transform(X_test)

        for model_name, clf in models.items():
            try:
                fitted = clone(clf)
                fitted.fit(X_train_s, y_train)
                y_prob = fitted.predict_proba(X_test_s)[:, 1]
                y_pred = fitted.predict(X_test_s)
            except Exception as e:
                print(f"    [FAIL] fold={fold_idx} {model_name}: {e}")
                continue

            m = compute_metrics(y_test, y_pred, y_prob)
            metrics_rows.append({
                "embedding": emb_name, "model": model_name, "fold": fold_idx,
                **m,
                "n_train": len(y_train), "n_test": len(y_test),
                "n_pos_train": int(y_train.sum()), "n_pos_test": int(y_test.sum()),
                "n_groups_test": len(np.unique(grps_test)),
            })

            for pid, pdb, prob, pred, true in zip(
                pids_test, grps_test, y_prob, y_pred, y_test
            ):
                pred_rows.append({
                    "pose_id":   pid,
                    "pdb_id":    pdb,
                    "embedding": emb_name,
                    "model":     model_name,
                    "fold":      fold_idx,
                    "y_true":    int(true),
                    "y_pred":    int(pred),
                    "y_prob":    float(prob),
                })

    return pd.DataFrame(metrics_rows), pd.DataFrame(pred_rows)


def save_models(
    emb_name: str,
    X: np.ndarray,
    common_ids: list[str],
    y: np.ndarray,
    models: dict,
) -> None:
    """Retrain each model on all data and save via joblib."""
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    for model_name, clf in models.items():
        try:
            fitted = clone(clf)
            fitted.fit(X_s, y)
            joblib.dump(
                {"model": fitted, "scaler": scaler,
                 "pose_ids": np.array(common_ids),
                 "n_samples": len(y), "n_pos": int(y.sum())},
                MODELS_DIR / f"{emb_name}_{model_name}.pkl",
            )
        except Exception as e:
            print(f"    [WARN] Could not save {emb_name}/{model_name}: {e}")


def print_summary(emb_name: str, metrics_df: pd.DataFrame) -> None:
    if metrics_df.empty:
        return
    summary = metrics_df.groupby("model")[["auroc", "auprc", "mcc", "f1"]].mean().round(3)
    print(f"\n  CV summary for {emb_name}:")
    print(summary.to_string())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train and evaluate classifiers")
    parser.add_argument("--embedding", type=str, default=None)
    parser.add_argument("--no-save-models", action="store_true")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    labels_df = load_labels()
    n_pos = labels_df["label"].sum()
    n_neg = (labels_df["label"] == 0).sum()
    print(f"Labels: {len(labels_df)} poses  "
          f"(pos={n_pos} [{n_pos/len(labels_df)*100:.1f}%], "
          f"neg={n_neg} [{n_neg/len(labels_df)*100:.1f}%])")
    print(f"Complexes: {labels_df['pdb_id'].nunique()}   "
          f"GroupKFold k={N_FOLDS} → ~{labels_df['pdb_id'].nunique()//N_FOLDS} complexes/fold")

    models = get_models()
    print(f"Models: {list(models.keys())}")

    npz_files = (
        [EMB_DIR / f"{args.embedding}.npz"]
        if args.embedding
        else sorted(EMB_DIR.glob("*.npz"))
    )
    if not npz_files or not npz_files[0].exists():
        raise FileNotFoundError(f"No .npz files found in {EMB_DIR}")

    all_metrics: list[pd.DataFrame] = []
    all_preds:   list[pd.DataFrame] = []

    for npz_path in npz_files:
        emb_name = npz_path.stem
        print(f"\n{'='*60}")
        print(f"  Embedding: {emb_name}")
        print(f"{'='*60}")

        try:
            d       = np.load(npz_path, allow_pickle=True)
            X_raw   = d["X"].astype(np.float32)
            pose_ids = d["pose_ids"].astype(str)
            pdb_ids_arr = d["pdb_ids"].astype(str)

            common_ids, X, y, groups = align_to_labels(
                labels_df, pose_ids, pdb_ids_arr, X_raw)

            n_pos_emb = y.sum()
            n_grp     = len(np.unique(groups))
            print(f"  n={len(common_ids)}, d={X.shape[1]}, "
                  f"pos={n_pos_emb} ({n_pos_emb/len(y)*100:.1f}%), "
                  f"complexes={n_grp}")

            if n_pos_emb < N_FOLDS or (y == 0).sum() < N_FOLDS:
                print(f"  [SKIP] Too few samples in one class")
                continue
            if n_grp < N_FOLDS:
                print(f"  [SKIP] Fewer complexes ({n_grp}) than folds ({N_FOLDS})")
                continue

            metrics_df, preds_df = train_evaluate_embedding(
                emb_name, X, common_ids, y, groups, models)
            all_metrics.append(metrics_df)
            all_preds.append(preds_df)
            print_summary(emb_name, metrics_df)

            if not args.no_save_models:
                save_models(emb_name, X, common_ids, y, models)

        except Exception as e:
            print(f"  [ERROR] {emb_name}: {e}")
            import traceback; traceback.print_exc()

    if not all_metrics:
        print("\nNo results to save.")
        return

    pd.concat(all_metrics, ignore_index=True).to_csv(RESULTS_DIR / "metrics.csv", index=False)
    pd.concat(all_preds,   ignore_index=True).to_csv(RESULTS_DIR / "predictions.csv", index=False)
    print(f"\nSaved: {RESULTS_DIR}/metrics.csv")
    print(f"Saved: {RESULTS_DIR}/predictions.csv")
    print("Done.")


if __name__ == "__main__":
    main()
