"""
06_error_analysis.py — Per-embedding pose-level error analysis.

For the best model (by mean CV AUROC) of each embedding:
  - Confusion matrix over all CV folds
  - Pose RMSD vs predicted probability scatter
    (each point is one docking pose; colour = correct/incorrect)

Output:
    plots/error/{emb}_{model}_cm.png
    plots/error/{emb}_{model}_rmsd_vs_prob.png

Usage:
    python 06_error_analysis.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
OUT_DIR     = Path(__file__).parent
DATA_DIR    = OUT_DIR / "data"
RESULTS_DIR = OUT_DIR / "results"
ERROR_DIR   = OUT_DIR / "plots" / "error"

RMSD_CUTOFF  = 2.0
MAX_SCATTER  = 8000    # subsample large pose sets for scatter readability


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_all_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    labels  = pd.read_csv(DATA_DIR    / "labels.csv").set_index("pose_id")
    metrics = pd.read_csv(RESULTS_DIR / "metrics.csv")
    preds   = pd.read_csv(RESULTS_DIR / "predictions.csv")
    return labels, metrics, preds


def find_best_model_per_embedding(metrics: pd.DataFrame) -> dict[str, str]:
    mean_m = metrics.groupby(["embedding", "model"])["auroc"].mean().reset_index()
    best: dict[str, str] = {}
    for emb, grp in mean_m.groupby("embedding"):
        best[emb] = grp.loc[grp["auroc"].idxmax(), "model"]
    return best


def _safe_stem(name: str) -> str:
    return name.replace("/", "_").replace(" ", "_")


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------
def plot_confusion_matrix(
    preds_sub: pd.DataFrame,
    emb_name: str,
    model_name: str,
    out_path: Path,
) -> None:
    y_true = preds_sub["y_true"].values
    y_pred = preds_sub["y_pred"].values

    cm   = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Decoy (0)", "Near-native (1)"])

    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(cmap="Blues", ax=ax, colorbar=False)
    ax.set_title(
        f"{emb_name} — {model_name}\n"
        f"All CV folds  (n={len(y_true)}, "
        f"acc={np.mean(y_true==y_pred):.3f})",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# ---------------------------------------------------------------------------
# RMSD vs predicted probability
# ---------------------------------------------------------------------------
def plot_rmsd_vs_prob(
    preds_sub: pd.DataFrame,
    labels: pd.DataFrame,
    emb_name: str,
    model_name: str,
    out_path: Path,
) -> None:
    merged = preds_sub.merge(
        labels[["rmsd"]],
        left_on="pose_id",
        right_index=True,
        how="inner",
    )
    if merged.empty:
        print(f"  [SKIP] No RMSD data for {emb_name}")
        return

    merged["correct"] = merged["y_true"] == merged["y_pred"]

    # Subsample if too many points
    if len(merged) > MAX_SCATTER:
        merged = merged.sample(MAX_SCATTER, random_state=42)
        note = f"  (showing {MAX_SCATTER}/{len(preds_sub)})"
    else:
        note = ""

    cmap = {True: "#4CAF50", False: "#F44336"}

    fig, ax = plt.subplots(figsize=(8, 5))
    for correct, lbl in [(True, "Correct"), (False, "Incorrect")]:
        m = merged[merged["correct"] == correct]
        ax.scatter(m["rmsd"], m["y_prob"],
                   c=cmap[correct], label=f"{lbl} (n={len(m)})",
                   s=8, alpha=0.4, linewidths=0)

    ax.axvline(RMSD_CUTOFF, color="black", linestyle="--", linewidth=1.5,
               label=f"RMSD cutoff ({RMSD_CUTOFF} Å)")
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1.0, alpha=0.7,
               label="Decision boundary (0.5)")

    ax.set_xlabel("Pose RMSD (Å)", fontsize=11)
    ax.set_ylabel("Predicted Probability (Near-native)", fontsize=11)
    ax.set_title(
        f"{emb_name} — {model_name}\nRMSD vs Predicted Probability{note}",
        fontsize=11,
    )
    ax.legend(fontsize=9, framealpha=0.8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ERROR_DIR.mkdir(parents=True, exist_ok=True)

    if not (RESULTS_DIR / "predictions.csv").exists():
        raise FileNotFoundError("results/predictions.csv not found — run 03_train_evaluate.py first")

    labels, metrics, preds = load_all_data()
    best_models = find_best_model_per_embedding(metrics)
    print(f"Analysing {len(best_models)} embeddings...")

    for emb_name, model_name in best_models.items():
        preds_sub = preds[
            (preds["embedding"] == emb_name) &
            (preds["model"]     == model_name)
        ].copy()
        if preds_sub.empty:
            continue

        print(f"\n{emb_name} / {model_name}  (n={len(preds_sub)} poses)")
        safe = _safe_stem(emb_name)

        plot_confusion_matrix(
            preds_sub, emb_name, model_name,
            ERROR_DIR / f"{safe}_{model_name}_cm.png",
        )
        plot_rmsd_vs_prob(
            preds_sub, labels, emb_name, model_name,
            ERROR_DIR / f"{safe}_{model_name}_rmsd_vs_prob.png",
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
