"""
05_performance_plots.py — Visualise CV classification results.

Reads results/metrics.csv and results/predictions.csv produced by
03_train_evaluate.py and generates:

    plots/heatmaps/auroc_grouped_bar.png  — grouped bar chart (embedding × model)
    plots/heatmaps/auroc_heatmap.png      — AUROC heatmap
    plots/heatmaps/mcc_heatmap.png        — MCC heatmap
    plots/roc/roc_curves_all.png          — ROC curves (best model per embedding)
    plots/pr/pr_curves_all.png            — PR curves  (best model per embedding)

Usage:
    python 05_performance_plots.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_curve, auc

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
OUT_DIR     = Path(__file__).parent
RESULTS_DIR = OUT_DIR / "results"
ROC_DIR     = OUT_DIR / "plots" / "roc"
PR_DIR      = OUT_DIR / "plots" / "pr"
HEAT_DIR    = OUT_DIR / "plots" / "heatmaps"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_results() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics  = pd.read_csv(RESULTS_DIR / "metrics.csv")
    preds    = pd.read_csv(RESULTS_DIR / "predictions.csv")
    mean_m   = (
        metrics
        .groupby(["embedding", "model"])[["auroc", "auprc", "mcc", "f1", "accuracy"]]
        .mean()
        .reset_index()
    )
    return metrics, mean_m, preds


def find_best_model_per_embedding(mean_metrics: pd.DataFrame) -> dict[str, str]:
    """Return {embedding: model_name} with highest mean AUROC."""
    best: dict[str, str] = {}
    for emb, grp in mean_metrics.groupby("embedding"):
        best[emb] = grp.loc[grp["auroc"].idxmax(), "model"]
    return best


def _embedding_order(mean_metrics: pd.DataFrame) -> list[str]:
    """Sort embeddings by mean AUROC (descending) for consistent plotting."""
    order = (
        mean_metrics.groupby("embedding")["auroc"]
        .max()
        .sort_values(ascending=False)
        .index.tolist()
    )
    return order


# ---------------------------------------------------------------------------
# Grouped bar chart
# ---------------------------------------------------------------------------
def plot_auroc_grouped_bar(mean_metrics: pd.DataFrame, out_path: Path) -> None:
    order = _embedding_order(mean_metrics)
    fig, ax = plt.subplots(figsize=(max(10, len(order) * 1.2), 5))
    sns.barplot(
        data=mean_metrics,
        x="embedding", y="auroc", hue="model",
        order=order, ax=ax,
        palette="colorblind",
    )
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1.0, alpha=0.6, label="random")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Embedding", fontsize=11)
    ax.set_ylabel("Mean CV AUROC", fontsize=11)
    ax.set_title("AUROC by Embedding and Model — CASF-2016 Docking", fontsize=12)
    ax.tick_params(axis="x", rotation=45)
    ax.legend(title="Model", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path.name}")


# ---------------------------------------------------------------------------
# Heatmaps
# ---------------------------------------------------------------------------
def _heatmap(
    mean_metrics: pd.DataFrame,
    metric: str,
    title: str,
    cmap: str,
    center: float | None,
    out_path: Path,
) -> None:
    order = _embedding_order(mean_metrics)
    pivot = mean_metrics.pivot(index="embedding", columns="model", values=metric)
    pivot = pivot.reindex(order)

    fig, ax = plt.subplots(figsize=(max(6, len(pivot.columns) * 1.5), max(4, len(pivot) * 0.5 + 1)))
    sns.heatmap(
        pivot, annot=True, fmt=".3f", cmap=cmap, center=center,
        linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.8},
    )
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Model", fontsize=10)
    ax.set_ylabel("Embedding", fontsize=10)
    ax.tick_params(axis="x", rotation=30)
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path.name}")


# ---------------------------------------------------------------------------
# ROC curves
# ---------------------------------------------------------------------------
def plot_roc_curves(
    preds: pd.DataFrame,
    best_models: dict[str, str],
    out_path: Path,
) -> None:
    colors  = sns.color_palette("colorblind", len(best_models))
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.5, label="Random")

    for (emb, model_name), color, marker in zip(best_models.items(), colors, markers):
        sub = preds[(preds["embedding"] == emb) & (preds["model"] == model_name)]
        if sub.empty:
            continue
        fpr, tpr, _ = roc_curve(sub["y_true"], sub["y_prob"])
        auroc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, linewidth=1.8,
                label=f"{emb} ({model_name})  AUROC={auroc:.3f}")

    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("ROC Curves — Best Model per Embedding\nCASF-2016 Docking", fontsize=11)
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path.name}")


# ---------------------------------------------------------------------------
# PR curves
# ---------------------------------------------------------------------------
def plot_pr_curves(
    preds: pd.DataFrame,
    best_models: dict[str, str],
    out_path: Path,
) -> None:
    colors = sns.color_palette("colorblind", len(best_models))

    # Class-balance baseline
    all_labels = preds.drop_duplicates(["embedding", "pdb_id", "fold"])
    pos_rate   = all_labels["y_true"].mean() if len(all_labels) else 0.5

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.axhline(pos_rate, color="black", linestyle="--", linewidth=0.8, alpha=0.5,
               label=f"Random (prevalence={pos_rate:.2f})")

    for (emb, model_name), color in zip(best_models.items(), colors):
        sub = preds[(preds["embedding"] == emb) & (preds["model"] == model_name)]
        if sub.empty:
            continue
        precision, recall, _ = precision_recall_curve(sub["y_true"], sub["y_prob"])
        ap = auc(recall, precision)
        ax.plot(recall, precision, color=color, linewidth=1.8,
                label=f"{emb} ({model_name})  AUPRC={ap:.3f}")

    ax.set_xlabel("Recall", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_title("Precision-Recall Curves — Best Model per Embedding\nCASF-2016 Docking",
                 fontsize=11)
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    for d in [ROC_DIR, PR_DIR, HEAT_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    if not (RESULTS_DIR / "metrics.csv").exists():
        raise FileNotFoundError("results/metrics.csv not found — run 03_train_evaluate.py first")

    metrics, mean_metrics, preds = load_results()
    best_models = find_best_model_per_embedding(mean_metrics)

    print("Best model per embedding (by mean CV AUROC):")
    for emb, mdl in best_models.items():
        row = mean_metrics[(mean_metrics["embedding"] == emb) & (mean_metrics["model"] == mdl)]
        print(f"  {emb:40s}  {mdl:8s}  AUROC={row['auroc'].values[0]:.3f}  "
              f"MCC={row['mcc'].values[0]:.3f}")

    print("\n--- Plotting ---")
    plot_auroc_grouped_bar(mean_metrics, HEAT_DIR / "auroc_grouped_bar.png")

    _heatmap(mean_metrics, "auroc", "Mean CV AUROC — Embedding × Model",
             "YlOrRd", None, HEAT_DIR / "auroc_heatmap.png")
    _heatmap(mean_metrics, "mcc",  "Mean CV MCC  — Embedding × Model",
             "RdYlGn", 0.0, HEAT_DIR / "mcc_heatmap.png")

    plot_roc_curves(preds, best_models, ROC_DIR / "roc_curves_all.png")
    plot_pr_curves( preds, best_models, PR_DIR  / "pr_curves_all.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
