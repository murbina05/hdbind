"""
07_robustness.py — Robustness analysis for the pose-level docking benchmark.

Three analyses:

1. RMSD cutoff sweep [1.0, 1.5, 2.0, 2.5, 3.0 Å]
   Re-binarises pose labels at each threshold, re-runs GroupKFold LogReg CV.
   → plots/robustness/cutoff_sweep.png

2. Learning curves (# training complexes vs AUROC)
   Uses Pipeline(StandardScaler → LogisticRegression) + GroupKFold to avoid leakage.
   → plots/robustness/learning_curves.png

3. CV variance (mean ± std AUROC per embedding/model from results/metrics.csv)
   → plots/robustness/cv_variance.png

Usage:
    python 07_robustness.py
    python 07_robustness.py --skip-learning-curves   # fast mode
"""

import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, learning_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
OUT_DIR     = Path(__file__).parent
EMB_DIR     = OUT_DIR / "embeddings"
DATA_DIR    = OUT_DIR / "data"
RESULTS_DIR = OUT_DIR / "results"
ROB_DIR     = OUT_DIR / "plots" / "robustness"

CUTOFFS     = [1.0, 1.5, 2.0, 2.5, 3.0]
RANDOM_SEED = 42
N_FOLDS     = 5
MIN_CLASS_N = 5

LEARNING_CURVE_EMBEDDINGS = [
    "pocket_esmc600m", "pocket_esm2_3b", "pocket_comp",
    "ligand_morgan", "pose_3d",
]


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------
def load_labels() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "labels.csv").set_index("pose_id")


def align_embedding_pose(
    labels_df: pd.DataFrame,
    npz_path: Path,
) -> tuple[str, np.ndarray, np.ndarray, np.ndarray] | None:
    """
    Load embedding, intersect with labels_df.
    Returns (emb_name, X, rmsd_array, groups_pdb_id) or None.
    """
    d        = np.load(npz_path, allow_pickle=True)
    X_all    = d["X"].astype(np.float32)
    pose_ids = d["pose_ids"].astype(str).tolist()
    pdb_ids  = d["pdb_ids"].astype(str).tolist()

    pid_to_i  = {pid: i for i, pid in enumerate(pose_ids)}
    pdb_to_i  = {pid: i for i, pid in enumerate(pose_ids)}  # same index
    common    = [pid for pid in pose_ids if pid in labels_df.index]
    if not common:
        return None

    X      = np.stack([X_all[pid_to_i[pid]] for pid in common])
    rmsd   = labels_df.loc[common, "rmsd"].values
    groups = labels_df.loc[common, "pdb_id"].values

    return npz_path.stem, X, rmsd, groups


def gkf_auroc(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
              n_folds: int = N_FOLDS) -> float:
    """GroupKFold LogReg CV — returns mean AUROC (NaN if degenerate)."""
    gkf  = GroupKFold(n_splits=n_folds)
    aucs: list[float] = []
    scaler = StandardScaler()

    for train_idx, test_idx in gkf.split(X, y, groups=groups):
        y_te = y[test_idx]
        if len(np.unique(y_te)) < 2:
            continue
        X_tr_s = scaler.fit_transform(X[train_idx])
        X_te_s = scaler.transform(X[test_idx])
        clf    = LogisticRegression(max_iter=500, class_weight="balanced",
                                    random_state=RANDOM_SEED)
        clf.fit(X_tr_s, y[train_idx])
        aucs.append(roc_auc_score(y_te, clf.predict_proba(X_te_s)[:, 1]))

    return float(np.mean(aucs)) if aucs else float("nan")


# ---------------------------------------------------------------------------
# 1. RMSD cutoff sweep
# ---------------------------------------------------------------------------
def run_cutoff_sweep(labels_df: pd.DataFrame) -> pd.DataFrame:
    print("\n--- RMSD cutoff sweep ---")
    rows: list[dict] = []

    for npz_path in sorted(EMB_DIR.glob("*.npz")):
        result = align_embedding_pose(labels_df, npz_path)
        if result is None:
            continue
        emb_name, X, rmsd, groups = result

        for cutoff in CUTOFFS:
            y     = (rmsd <= cutoff).astype(int)
            n_pos = y.sum()
            n_neg = (y == 0).sum()
            n_grp = len(np.unique(groups))

            if n_pos < MIN_CLASS_N or n_neg < MIN_CLASS_N or n_grp < N_FOLDS:
                continue

            mean_auroc = gkf_auroc(X, y, groups)
            rows.append({
                "embedding":  emb_name,
                "cutoff":     cutoff,
                "mean_auroc": mean_auroc,
                "n":          len(y),
                "n_pos":      int(n_pos),
                "n_neg":      int(n_neg),
            })
            print(f"  {emb_name:42s}  cutoff={cutoff:.1f}Å  "
                  f"pos={n_pos:5d}  neg={n_neg:5d}  AUROC={mean_auroc:.3f}")

    return pd.DataFrame(rows)


def plot_cutoff_sweep(df: pd.DataFrame, out_path: Path) -> None:
    if df.empty:
        print("  [SKIP] No data")
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(data=df, x="cutoff", y="mean_auroc", hue="embedding",
                 marker="o", dashes=False, ax=ax, palette="colorblind")
    ax.axvline(2.0, color="black", linestyle="--", linewidth=1.2, alpha=0.6,
               label="Default cutoff (2.0 Å)")
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1.0, alpha=0.6)
    ax.set_xlabel("RMSD Cutoff (Å)", fontsize=11)
    ax.set_ylabel("Mean CV AUROC (GroupKFold LogReg)", fontsize=11)
    ax.set_title("Sensitivity to RMSD Cutoff — CASF-2016 Docking Poses", fontsize=12)
    ax.set_ylim(0.3, 1.02)
    ax.legend(title="Embedding", fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path.name}")


# ---------------------------------------------------------------------------
# 2. Learning curves
# ---------------------------------------------------------------------------
def run_learning_curves(labels_df: pd.DataFrame, out_path: Path) -> None:
    print("\n--- Learning curves ---")
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("lr",     LogisticRegression(max_iter=500, class_weight="balanced",
                                      random_state=RANDOM_SEED)),
    ])

    available = [e for e in LEARNING_CURVE_EMBEDDINGS
                 if (EMB_DIR / f"{e}.npz").exists()]
    if not available:
        print("  [SKIP] None of the target embeddings found")
        return

    n_cols = len(available)
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 4), sharey=True)
    if n_cols == 1:
        axes = [axes]

    for ax, emb_name in zip(axes, available):
        result = align_embedding_pose(labels_df, EMB_DIR / f"{emb_name}.npz")
        if result is None:
            ax.set_title(f"{emb_name}\n(no overlap)")
            continue

        _, X, rmsd, groups = result
        y      = (rmsd <= 2.0).astype(int)
        n_grp  = len(np.unique(groups))
        n_pos  = y.sum()

        if n_pos < MIN_CLASS_N or (y == 0).sum() < MIN_CLASS_N or n_grp < N_FOLDS:
            ax.set_title(f"{emb_name}\n(degenerate)")
            continue

        print(f"  {emb_name}: n={len(y)}, d={X.shape[1]}, pos={n_pos}, groups={n_grp}")

        # Sizes expressed as fractions of training-fold size
        train_sizes_frac = np.linspace(0.10, 1.0, 8)

        try:
            train_sizes, train_scores, val_scores = learning_curve(
                clf, X, y,
                cv=GroupKFold(N_FOLDS),
                groups=groups,
                scoring="roc_auc",
                train_sizes=train_sizes_frac,
                n_jobs=-1,
            )
        except Exception as e:
            print(f"    [FAIL] {e}")
            ax.set_title(f"{emb_name}\n(error)")
            continue

        t_mean, t_std = train_scores.mean(1), train_scores.std(1)
        v_mean, v_std = val_scores.mean(1),   val_scores.std(1)

        ax.plot(train_sizes, t_mean, "o-", color="#2196F3", label="Train", lw=1.8)
        ax.fill_between(train_sizes, t_mean - t_std, t_mean + t_std, alpha=0.15, color="#2196F3")
        ax.plot(train_sizes, v_mean, "s-", color="#E64A19", label="Validation", lw=1.8)
        ax.fill_between(train_sizes, v_mean - v_std, v_mean + v_std, alpha=0.15, color="#E64A19")

        ax.axhline(0.5, color="gray", linestyle=":", lw=1.0, alpha=0.6)
        ax.set_xlabel("Training set size (poses)", fontsize=10)
        ax.set_ylabel("AUROC", fontsize=10)
        ax.set_title(f"{emb_name}\n(d={X.shape[1]})", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0.3, 1.05)

    fig.suptitle("Learning Curves — LogReg (GroupKFold) — CASF-2016 Poses", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path.name}")


# ---------------------------------------------------------------------------
# 3. CV variance
# ---------------------------------------------------------------------------
def plot_cv_variance(out_path: Path) -> None:
    print("\n--- CV variance ---")
    metrics_path = RESULTS_DIR / "metrics.csv"
    if not metrics_path.exists():
        print(f"  [SKIP] {metrics_path} not found")
        return

    metrics = pd.read_csv(metrics_path)
    summary = (
        metrics.groupby(["embedding", "model"])["auroc"]
        .agg(mean_auroc="mean", std_auroc="std")
        .reset_index()
    )
    order = (
        summary.groupby("embedding")["mean_auroc"]
        .max().sort_values(ascending=False).index.tolist()
    )

    fig, ax = plt.subplots(figsize=(max(10, len(order) * 1.2), 5))
    sns.barplot(data=summary, x="embedding", y="mean_auroc", hue="model",
                order=order, ax=ax, palette="colorblind", capsize=0.05)
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1.0, alpha=0.5)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Embedding", fontsize=11)
    ax.set_ylabel("Mean CV AUROC ± SD (GroupKFold 5-fold)", fontsize=11)
    ax.set_title("CV Variance — AUROC per Embedding and Model", fontsize=12)
    ax.tick_params(axis="x", rotation=45)
    ax.legend(title="Model", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Robustness analysis (pose-level)")
    parser.add_argument("--skip-learning-curves", action="store_true")
    args = parser.parse_args()

    ROB_DIR.mkdir(parents=True, exist_ok=True)

    if not (DATA_DIR / "labels.csv").exists():
        raise FileNotFoundError("data/labels.csv not found — run 01_parse_dataset.py first")

    labels_df = load_labels()

    cutoff_df = run_cutoff_sweep(labels_df)
    if not cutoff_df.empty:
        cutoff_df.to_csv(ROB_DIR / "cutoff_sweep.csv", index=False)
    plot_cutoff_sweep(cutoff_df, ROB_DIR / "cutoff_sweep.png")

    if not args.skip_learning_curves:
        run_learning_curves(labels_df, ROB_DIR / "learning_curves.png")

    plot_cv_variance(ROB_DIR / "cv_variance.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
