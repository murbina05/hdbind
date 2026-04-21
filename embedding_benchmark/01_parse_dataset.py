"""
01_parse_dataset.py — CASF-2016 pose-level docking label extraction.

Each docking POSE is a separate data point:
    label = 1  if  pose RMSD <= RMSD_CUTOFF  (near-native)
    label = 0  otherwise

This gives ~25 000 data points with natural class imbalance
(only a small fraction of docking poses are near-native).

The native crystal-structure row ({pdb_id}_ligand, RMSD=0.00) is excluded
from every file — only actual docking decoy poses are included.

Output:
    data/labels.csv              — pose_id, pdb_id, rmsd, label, logKa, …
    plots/rmsd_distribution.png  — histogram of ALL pose RMSDs with cutoff line

Usage:
    python 01_parse_dataset.py
    python 01_parse_dataset.py --cutoff 1.0
    python 01_parse_dataset.py --cutoff 2.0   # default; ~10–15% positive rate
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CORESET_DIR = Path("/home/maurbina/datasets/CASF-2016/coreset")
DECOYS_DIR  = Path("/home/maurbina/datasets/CASF-2016/decoys_docking")
CORESET_DAT = Path("/home/maurbina/datasets/CASF-2016/power_scoring/CoreSet.dat")

OUT_DIR   = Path(__file__).parent
DATA_DIR  = OUT_DIR / "data"
PLOTS_DIR = OUT_DIR / "plots"

RMSD_CUTOFF = 2.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_coreset_meta() -> pd.DataFrame:
    """Load CoreSet.dat → DataFrame indexed by lowercase pdb_id."""
    df = pd.read_csv(
        CORESET_DAT,
        sep=r"\s+",
        comment="#",
        header=None,
        names=["pdb_id", "resolution", "year", "logKa", "Ka_str", "cluster"],
    )
    df["pdb_id"] = df["pdb_id"].str.lower()
    return df.set_index("pdb_id")


def parse_rmsd_file_poses(path: Path, pdb_id: str) -> list[dict]:
    """
    Read {pdb_id}_rmsd.dat and return one dict per docking pose.

    Skips:
      - comment lines (starting with #)
      - the native crystal-structure reference row ({pdb_id}_ligand)

    Returns list of {"pose_id": str, "rmsd": float}.
    """
    rows: list[dict] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            pose_id = parts[0]
            if pose_id.endswith("_ligand"):      # skip native crystal pose
                continue
            try:
                rmsd = float(parts[1])
            except ValueError:
                continue
            rows.append({"pose_id": pose_id, "pdb_id": pdb_id, "rmsd": rmsd})
    return rows


def build_pose_labels(meta: pd.DataFrame, cutoff: float = RMSD_CUTOFF) -> pd.DataFrame:
    """
    Iterate over coreset complexes, collect all docking poses, binarise at cutoff.
    Returns DataFrame with columns:
        pose_id, pdb_id, rmsd, label, logKa, cluster, year, resolution
    """
    all_rows: list[dict] = []
    failed: list[str] = []

    for pdb_id in meta.index:
        rmsd_file = DECOYS_DIR / f"{pdb_id}_rmsd.dat"
        if not rmsd_file.exists():
            print(f"  [SKIP] {pdb_id}: RMSD file not found")
            failed.append(pdb_id)
            continue
        try:
            pose_rows = parse_rmsd_file_poses(rmsd_file, pdb_id)
        except Exception as e:
            print(f"  [SKIP] {pdb_id}: {e}")
            failed.append(pdb_id)
            continue
        if not pose_rows:
            print(f"  [WARN] {pdb_id}: 0 valid poses")
            continue
        for row in pose_rows:
            row["label"] = int(row["rmsd"] <= cutoff)
        all_rows.extend(pose_rows)

    if failed:
        print(f"\n  Skipped {len(failed)} complexes")

    df = pd.DataFrame(all_rows)
    # Join with metadata
    df = df.merge(
        meta[["logKa", "cluster", "year", "resolution"]],
        left_on="pdb_id",
        right_index=True,
        how="left",
    )
    return df


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_rmsd_distribution(df: pd.DataFrame, cutoff: float, out_path: Path) -> None:
    n_pos = (df["label"] == 1).sum()
    n_neg = (df["label"] == 0).sum()
    n_total = len(df)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(df["rmsd"], bins=60, color="#2196F3", edgecolor="white", alpha=0.85)
    ax.axvline(
        cutoff, color="red", linestyle="--", linewidth=2,
        label=(f"Cutoff {cutoff} Å  "
               f"[near-native: {n_pos} ({n_pos/n_total*100:.1f}%),  "
               f"decoy: {n_neg} ({n_neg/n_total*100:.1f}%)]"),
    )
    ax.set_xlabel("Docking Pose RMSD (Å)", fontsize=12)
    ax.set_ylabel("Number of Poses", fontsize=12)
    ax.set_title(
        f"CASF-2016 Docking Pose RMSD Distribution\n"
        f"{n_total} poses from {df['pdb_id'].nunique()} complexes",
        fontsize=11,
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Parse CASF-2016 pose-level docking labels")
    parser.add_argument("--cutoff", type=float, default=RMSD_CUTOFF,
                        help=f"RMSD threshold for label=1 (default: {RMSD_CUTOFF})")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print("--- Loading CoreSet metadata ---")
    meta = load_coreset_meta()
    print(f"  Coreset: {len(meta)} complexes")

    print("\n--- Parsing pose RMSD files ---")
    df = build_pose_labels(meta, cutoff=args.cutoff)

    n = len(df)
    n_complexes = df["pdb_id"].nunique()
    n_pos = (df["label"] == 1).sum()
    n_neg = (df["label"] == 0).sum()
    poses_per_complex = n / n_complexes

    print(f"\nPose-level dataset summary:")
    print(f"  Complexes:             {n_complexes}")
    print(f"  Total poses:           {n}  (~{poses_per_complex:.0f} per complex)")
    print(f"  Near-native (1):       {n_pos}  ({n_pos/n*100:.1f}%)   [RMSD <= {args.cutoff} Å]")
    print(f"  Decoy       (0):       {n_neg}  ({n_neg/n*100:.1f}%)")
    print(f"  RMSD — min: {df['rmsd'].min():.2f}, max: {df['rmsd'].max():.2f}, "
          f"mean: {df['rmsd'].mean():.2f}, median: {df['rmsd'].median():.2f}")

    out_csv = DATA_DIR / "labels.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nLabels saved: {out_csv}  ({n} rows)")

    plot_rmsd_distribution(df, args.cutoff, PLOTS_DIR / "rmsd_distribution.png")


if __name__ == "__main__":
    main()
