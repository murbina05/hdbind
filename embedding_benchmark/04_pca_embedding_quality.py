"""
04_pca_embedding_quality.py — PCA visualisation of each pose-level embedding.

For each embedding in embeddings/*.npz:
  - 2-D PCA scatter coloured by binary docking label (0=decoy, 1=near-native)
  - Scree + cumulative explained variance plot

Joint figure:
  - Overlaid scree curves for all embeddings

Output:
    plots/pca/{name}_pca_binary.png
    plots/pca/{name}_scree.png
    plots/pca/all_embeddings_scree.png

Usage:
    python 04_pca_embedding_quality.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
OUT_DIR   = Path(__file__).parent
EMB_DIR   = OUT_DIR / "embeddings"
DATA_DIR  = OUT_DIR / "data"
PLOTS_DIR = OUT_DIR / "plots" / "pca"

RANDOM_SEED = 42
N_SCREE     = 50
MAX_PLOT_N  = 5000   # subsample for scatter to keep plots readable


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_labels() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "labels.csv").set_index("pose_id")


def align_binary_labels(
    labels_df: pd.DataFrame,
    pose_ids: np.ndarray,
    X: np.ndarray,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    pid_to_idx = {pid: i for i, pid in enumerate(pose_ids.tolist())}
    common    = [pid for pid in pose_ids.tolist() if pid in labels_df.index]
    X_aligned = np.stack([X[pid_to_idx[pid]] for pid in common])
    y         = labels_df.loc[common, "label"].values.astype(int)
    return common, X_aligned, y


# ---------------------------------------------------------------------------
# Per-embedding plots
# ---------------------------------------------------------------------------
def pca_scatter_binary(
    X: np.ndarray,
    y: np.ndarray,
    label: str,
    out_path: Path,
) -> None:
    n, d = X.shape
    Xs   = StandardScaler().fit_transform(X)
    pca  = PCA(n_components=2, random_state=RANDOM_SEED)
    coords = pca.fit_transform(Xs)
    ve   = pca.explained_variance_ratio_ * 100

    # Subsample for scatter readability
    if n > MAX_PLOT_N:
        rng  = np.random.default_rng(RANDOM_SEED)
        idx  = rng.choice(n, MAX_PLOT_N, replace=False)
        coords, y_plot = coords[idx], y[idx]
        note = f"  (showing {MAX_PLOT_N}/{n})"
    else:
        y_plot = y
        note = ""

    fig, ax = plt.subplots(figsize=(7, 5))
    for cls, color, lbl in [
        (0, "#9E9E9E", f"Decoy (0)  n={(y==0).sum()}"),
        (1, "#E64A19", f"Near-native (1)  n={(y==1).sum()}"),
    ]:
        mask = y_plot == cls
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=color, label=lbl, s=8, alpha=0.45, linewidths=0)

    ax.set_xlabel(f"PC1 ({ve[0]:.1f}%)", fontsize=11)
    ax.set_ylabel(f"PC2 ({ve[1]:.1f}%)", fontsize=11)
    ax.set_title(f"{label}\n(n={n}{note}, d={d})", fontsize=11)
    ax.legend(fontsize=9, framealpha=0.8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def scree_and_cumvar(
    X: np.ndarray,
    label: str,
    out_path: Path,
    n_show: int = N_SCREE,
) -> np.ndarray:
    n, d = X.shape
    Xs = StandardScaler().fit_transform(X)
    n_components = min(n, d)
    pca = PCA(n_components=n_components, random_state=RANDOM_SEED)
    pca.fit(Xs)

    evr    = pca.explained_variance_ratio_
    cumvar = np.cumsum(evr)
    n_plot = min(n_show, len(evr))

    for t in (0.50, 0.80, 0.90, 0.95):
        k = int(np.searchsorted(cumvar, t)) + 1
        print(f"    {int(t*100)}%→{k}pc", end="")
    print()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(range(1, n_plot + 1), evr[:n_plot], "o-", markersize=4, linewidth=1.5,
             color="#2196F3", label=f"{label} (n={n}, d={d})")
    ax1.set_xlabel("Principal Component")
    ax1.set_ylabel("Explained Variance Ratio")
    ax1.set_title(f"Scree Plot — {label}\nCASF-2016  (n={n}, d={d})")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0.5, n_plot + 0.5)

    ax2.plot(range(1, n_plot + 1), cumvar[:n_plot], "s-", markersize=4, linewidth=1.5,
             color="#E91E63")
    for threshold in (0.80, 0.90, 0.95):
        ax2.axhline(threshold, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
        ax2.text(n_plot * 0.98, threshold + 0.005, f"{int(threshold*100)}%",
                 ha="right", va="bottom", fontsize=8, color="gray")
    ax2.set_xlabel("Principal Component")
    ax2.set_ylabel("Cumulative Explained Variance")
    ax2.set_title(f"Cumulative Variance — {label}\nCASF-2016  (n={n}, d={d})")
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0.5, n_plot + 0.5)
    ax2.set_ylim(0, 1.05)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return evr


# ---------------------------------------------------------------------------
# Joint scree overlay
# ---------------------------------------------------------------------------
def run_joint_scree(evr_dict: dict, out_path: Path, n_show: int = N_SCREE) -> None:
    colors  = sns.color_palette("colorblind", len(evr_dict))
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for (label, evr), color, marker in zip(evr_dict.items(), colors, markers):
        cumvar = np.cumsum(evr)
        n_plot = min(n_show, len(evr))
        xs = range(1, n_plot + 1)
        ax1.plot(xs, evr[:n_plot],    f"{marker}-", markersize=3, linewidth=1.3,
                 color=color, label=label)
        ax2.plot(xs, cumvar[:n_plot], f"{marker}-", markersize=3, linewidth=1.3,
                 color=color, label=label)

    for t in (0.80, 0.90, 0.95):
        ax2.axhline(t, color="gray", linestyle="--", linewidth=0.7, alpha=0.6)
        ax2.text(n_show * 0.99, t + 0.004, f"{int(t*100)}%",
                 ha="right", va="bottom", fontsize=7, color="gray")

    for ax, title, ylabel in [
        (ax1, "Scree Plot — All Embeddings", "Explained Variance Ratio"),
        (ax2, "Cumulative Variance — All Embeddings", "Cumulative Explained Variance"),
    ]:
        ax.set_xlabel("Principal Component")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title}\nCASF-2016 coreset (pose-level)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0.5, n_show + 0.5)
    ax2.set_ylim(0, 1.05)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Joint scree saved: {out_path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    if not (DATA_DIR / "labels.csv").exists():
        raise FileNotFoundError("data/labels.csv not found — run 01_parse_dataset.py first")

    labels_df = load_labels()
    npz_files = sorted(EMB_DIR.glob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No .npz files in {EMB_DIR} — run 02_featurize.py first")

    evr_dict: dict[str, np.ndarray] = {}

    for npz_path in npz_files:
        emb_name = npz_path.stem
        print(f"\n{emb_name}")

        d        = np.load(npz_path, allow_pickle=True)
        X_raw    = d["X"].astype(np.float32)
        pose_ids = d["pose_ids"].astype(str)

        try:
            _, X, y = align_binary_labels(labels_df, pose_ids, X_raw)
        except Exception as e:
            print(f"  [SKIP] {e}")
            continue

        n, dim = X.shape
        print(f"  n={n}, d={dim}, pos={y.sum()}, neg={(y==0).sum()}", end="")

        pca_scatter_binary(X, y, emb_name, PLOTS_DIR / f"{emb_name}_pca_binary.png")
        evr = scree_and_cumvar(X, emb_name, PLOTS_DIR / f"{emb_name}_scree.png")
        evr_dict[emb_name] = evr

    if evr_dict:
        run_joint_scree(evr_dict, PLOTS_DIR / "all_embeddings_scree.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
