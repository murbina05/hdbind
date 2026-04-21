"""
02_featurize.py — Pose-level featurization of CASF-2016 docking decoys.

Reads data/labels.csv (one row per docking pose) and builds per-pose
feature matrices saved to embeddings/ as:

    embeddings/{name}.npz
        X        float32  (n_poses, d)
        pose_ids str      (n_poses,)   e.g. "1a30_128"
        pdb_ids  str      (n_poses,)   e.g. "1a30"   ← for GroupKFold in 03

Embeddings produced:

    pocket_esmc600m    — ESM-C 600M pocket emb broadcast to all poses  (n, 1152)
    pocket_esm2_3b     — ESM-2 3B   pocket emb broadcast               (n, 2560)
    pocket_comp        — Residue composition broadcast                  (n,   24)
    ligand_morgan      — ECFP4 1024-bit, same per complex, broadcast   (n, 2048)
    pose_3d            — RDKit 3D shape + pocket-relative centroid     (n,   14)
    complex_esmc600m_pose3d          — pocket + pose_3d                (n, 1166)
    complex_esmc600m_morgan_pose3d   — pocket + Morgan + pose_3d       (n, 3214)

Usage:
    python 02_featurize.py
    python 02_featurize.py --skip-complex   # skip concatenated embeddings
    python 02_featurize.py --skip-pose3d    # skip mol2 parsing (faster, fewer features)
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
NOTEBOOKS_DIR = Path("/home/maurbina/hdbind/notebooks")
CORESET_DIR   = Path("/home/maurbina/datasets/CASF-2016/coreset")
DECOYS_DIR    = Path("/home/maurbina/datasets/CASF-2016/decoys_docking")
CORESET_DAT   = Path("/home/maurbina/datasets/CASF-2016/power_scoring/CoreSet.dat")

OUT_DIR  = Path(__file__).parent
DATA_DIR = OUT_DIR / "data"
EMB_DIR  = OUT_DIR / "embeddings"

# Pre-computed ESM sources: name → (npz_path, embedding_key)
ESM_SOURCES = {
    "pocket_esmc600m": (NOTEBOOKS_DIR / "esmc_600m_casf2016_embeddings.npz", "mean_embeddings"),
    "pocket_esmc6b":   (NOTEBOOKS_DIR / "esmc_6b_casf2016_embeddings.npz",   "mean_embeddings"),
    "pocket_esm2_3b":  (NOTEBOOKS_DIR / "esm2_casf2016_embeddings.npz",      "mean_embeddings"),
}

# Residue list for pocket composition
_RESIDUES = [
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "PYL", "SER", "SEC", "THR", "TRP",
    "TYR", "VAL", "ASX", "GLX",
]
_RES_TO_IDX = {r: i for i, r in enumerate(_RESIDUES)}

# 3D shape descriptors (14 features = 10 shape + 3 relative centroid + 1 distance)
_SHAPE3D_NAMES = [
    "Asphericity", "Eccentricity", "InertialShapeFactor",
    "NPR1", "NPR2", "PMI1", "PMI2", "PMI3",
    "RadiusOfGyration", "SpherocityIndex",
    "rel_x", "rel_y", "rel_z", "centroid_dist",
]


# ---------------------------------------------------------------------------
# I/O utilities
# ---------------------------------------------------------------------------
def save_embedding(name: str, X: np.ndarray,
                   pose_ids: list[str], pdb_ids: list[str]) -> None:
    path = EMB_DIR / f"{name}.npz"
    np.savez(
        path,
        X=X.astype(np.float32),
        pose_ids=np.array(pose_ids, dtype=str),
        pdb_ids=np.array(pdb_ids,  dtype=str),
    )
    print(f"  Saved {name:42s}  X={str(X.shape):18s}  n_poses={len(pose_ids)}")


def load_embedding(name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (X, pose_ids, pdb_ids)."""
    d = np.load(EMB_DIR / f"{name}.npz", allow_pickle=True)
    return d["X"].astype(np.float32), d["pose_ids"].astype(str), d["pdb_ids"].astype(str)


def load_labels() -> pd.DataFrame:
    """Load data/labels.csv — pose_id, pdb_id, rmsd, label, …"""
    df = pd.read_csv(DATA_DIR / "labels.csv")
    return df.set_index("pose_id")


def load_coreset_pdb_ids() -> list[str]:
    df = pd.read_csv(
        CORESET_DAT, sep=r"\s+", comment="#", header=None,
        names=["pdb_id", "resolution", "year", "logKa", "Ka_str", "cluster"],
    )
    return df["pdb_id"].str.lower().tolist()


# ---------------------------------------------------------------------------
# Pocket geometry helpers
# ---------------------------------------------------------------------------
def compute_pocket_centroid(pocket_pdb: Path) -> np.ndarray:
    """Mean Cα position of pocket residues — reference frame for pose 3D features."""
    ca_coords: list[list[float]] = []
    with open(pocket_pdb) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            if line[12:16].strip() != "CA":
                continue
            try:
                ca_coords.append([
                    float(line[30:38]),
                    float(line[38:46]),
                    float(line[46:54]),
                ])
            except ValueError:
                continue
    return np.mean(ca_coords, axis=0) if ca_coords else np.zeros(3)


def _featurize_pocket_pdb(path: Path) -> np.ndarray:
    """Residue count vector (d=24) — copied from compare_featurizers_casf2016.py."""
    counts = np.zeros(len(_RESIDUES), dtype=np.float32)
    seen: set[tuple] = set()
    with open(path) as fh:
        for line in fh:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            res_name = line[17:20].strip()
            key = (res_name, line[22:26].strip(), line[21])
            if key in seen:
                continue
            seen.add(key)
            if res_name in _RES_TO_IDX:
                counts[_RES_TO_IDX[res_name]] += 1
    return counts


# ---------------------------------------------------------------------------
# mol2 parsing
# ---------------------------------------------------------------------------
def parse_decoys_mol2(mol2_path: Path) -> dict[str, object]:
    """
    Parse a multi-mol2 decoys file into {pose_id: RDKit mol}.
    Splits on @<TRIPOS>MOLECULE blocks.  Skips the native pose (_ligand).
    Returns only successfully parsed mols with a 3-D conformer.
    """
    from rdkit import Chem

    with open(mol2_path) as f:
        content = f.read()

    # Split on the section header, keeping non-empty blocks
    raw_blocks = content.split("@<TRIPOS>MOLECULE")
    result: dict[str, object] = {}

    for raw in raw_blocks:
        raw = raw.strip()
        if not raw:
            continue
        lines = raw.split("\n")
        pose_id = lines[0].strip()
        if not pose_id or pose_id.endswith("_ligand"):
            continue

        full_block = "@<TRIPOS>MOLECULE\n" + raw
        mol = Chem.MolFromMol2Block(full_block, removeHs=True, sanitize=False)
        if mol is None:
            continue
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            pass
        if mol.GetNumConformers() == 0:
            continue
        result[pose_id] = mol

    return result


# ---------------------------------------------------------------------------
# 3D pose features
# ---------------------------------------------------------------------------
def compute_pose_3d_features(mol, pocket_centroid: np.ndarray) -> np.ndarray:
    """
    14-dimensional pose-specific feature vector:
        [0-9]  RDKit 3D shape descriptors (Asphericity … SpherocityIndex)
        [10-12] ligand centroid – pocket centroid  (pocket-relative position)
        [13]   Euclidean distance from ligand centroid to pocket centroid
    """
    from rdkit.Chem import Descriptors3D

    features: list[float] = []

    # 3D shape descriptors
    shape_funcs = [
        Descriptors3D.Asphericity,
        Descriptors3D.Eccentricity,
        Descriptors3D.InertialShapeFactor,
        Descriptors3D.NPR1,
        Descriptors3D.NPR2,
        Descriptors3D.PMI1,
        Descriptors3D.PMI2,
        Descriptors3D.PMI3,
        Descriptors3D.RadiusOfGyration,
        Descriptors3D.SpherocityIndex,
    ]
    for fn in shape_funcs:
        try:
            val = float(fn(mol))
            features.append(val if np.isfinite(val) else 0.0)
        except Exception:
            features.append(0.0)

    # Pocket-relative centroid
    positions = mol.GetConformer().GetPositions()       # (n_atoms, 3)
    lig_centroid = positions.mean(axis=0)               # (3,)
    relative = lig_centroid - pocket_centroid           # (3,)
    features.extend(relative.tolist())
    features.append(float(np.linalg.norm(relative)))   # scalar distance

    return np.array(features, dtype=np.float32)


# ---------------------------------------------------------------------------
# Per-complex protein feature lookup
# ---------------------------------------------------------------------------
def build_pdb_to_embedding(npz_path: Path, arr_key: str) -> dict[str, np.ndarray]:
    """Load an ESM NPZ and return {pdb_id: embedding_vector}."""
    d = np.load(npz_path, allow_pickle=True)
    X = d[arr_key].astype(np.float32)
    ids = d["pdb_ids"].astype(str).tolist()
    return {pid: X[i] for i, pid in enumerate(ids)}


# ---------------------------------------------------------------------------
# Main featurization routines
# ---------------------------------------------------------------------------
def process_pocket_broadcast(
    name: str,
    emb_by_pdb: dict[str, np.ndarray],
    labels: pd.DataFrame,
) -> None:
    """
    Broadcast a per-complex embedding to every pose of that complex.
    labels must be indexed by pose_id and have a 'pdb_id' column.
    """
    X_rows: list[np.ndarray] = []
    pose_ids: list[str] = []
    pdb_ids: list[str] = []

    for pose_id, row in labels.iterrows():
        pdb_id = row["pdb_id"]
        if pdb_id not in emb_by_pdb:
            continue
        X_rows.append(emb_by_pdb[pdb_id])
        pose_ids.append(pose_id)
        pdb_ids.append(pdb_id)

    if not X_rows:
        print(f"  [SKIP] {name}: no valid poses")
        return
    save_embedding(name, np.stack(X_rows), pose_ids, pdb_ids)


def process_esm_embeddings(labels: pd.DataFrame) -> None:
    print("\n--- ESM pocket embeddings (broadcast to poses) ---")
    for name, (npz_path, arr_key) in ESM_SOURCES.items():
        if not npz_path.exists():
            print(f"  [SKIP] {name}: {npz_path} not found")
            continue
        emb_by_pdb = build_pdb_to_embedding(npz_path, arr_key)
        process_pocket_broadcast(name, emb_by_pdb, labels)


def process_pocket_composition(labels: pd.DataFrame, pdb_ids_all: list[str]) -> None:
    print("\n--- Pocket composition (broadcast to poses) ---")
    # Prefer pre-computed cache; fall back to computing from PDB
    cache = Path(str(NOTEBOOKS_DIR)) / "dc_pocket_casf2016.npz"
    if cache.exists():
        d = np.load(cache, allow_pickle=True)
        X_cache = d["X"].astype(np.float32)
        if len(X_cache) == len(pdb_ids_all):
            emb_by_pdb = {pid: X_cache[i] for i, pid in enumerate(pdb_ids_all)}
            process_pocket_broadcast("pocket_comp", emb_by_pdb, labels)
            return
    # Compute from PDB files
    emb_by_pdb: dict[str, np.ndarray] = {}
    for pdb_id in pdb_ids_all:
        pdb_path = CORESET_DIR / pdb_id / f"{pdb_id}_pocket.pdb"
        if not pdb_path.exists():
            continue
        vec = _featurize_pocket_pdb(pdb_path)
        if vec.sum() > 0:
            emb_by_pdb[pdb_id] = vec
    process_pocket_broadcast("pocket_comp", emb_by_pdb, labels)


def process_ligand_morgan(labels: pd.DataFrame) -> None:
    print("\n--- Morgan fingerprints (ECFP4 broadcast to poses) ---")
    from rdkit import Chem
    from rdkit.Chem import AllChem

    # Compute Morgan FP once per complex (same ligand for all poses)
    emb_by_pdb: dict[str, np.ndarray] = {}
    failed = 0
    for pdb_id in labels["pdb_id"].unique():
        sdf = CORESET_DIR / pdb_id / f"{pdb_id}_ligand.sdf"
        if not sdf.exists():
            failed += 1
            continue
        try:
            suppl = Chem.SDMolSupplier(str(sdf), removeHs=True, sanitize=True)
            mol = next((m for m in suppl if m is not None), None)
            if mol is None:
                raise ValueError("no valid mol")
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
            emb_by_pdb[pdb_id] = np.array(fp, dtype=np.float32)
        except Exception:
            failed += 1

    print(f"  Morgan FP: {len(emb_by_pdb)} complexes, {failed} failed")
    process_pocket_broadcast("ligand_morgan", emb_by_pdb, labels)


def process_pose_3d(labels: pd.DataFrame) -> None:
    """
    Parse mol2 decoys files and compute 14-D pose-specific features for each pose.
    """
    print("\n--- Pose 3D features (shape + pocket-relative centroid) ---")

    X_rows:   list[np.ndarray] = []
    pose_ids: list[str] = []
    pdb_ids:  list[str] = []
    n_complexes = 0
    n_poses_ok  = 0
    n_poses_fail = 0

    for pdb_id, grp in labels.groupby("pdb_id"):
        mol2_path   = DECOYS_DIR / f"{pdb_id}_decoys.mol2"
        pocket_path = CORESET_DIR / pdb_id / f"{pdb_id}_pocket.pdb"

        if not mol2_path.exists() or not pocket_path.exists():
            continue

        # Parse all poses for this complex
        try:
            mol_dict = parse_decoys_mol2(mol2_path)
        except Exception as e:
            print(f"  [SKIP] {pdb_id} mol2: {e}")
            continue

        pocket_centroid = compute_pocket_centroid(pocket_path)
        n_complexes += 1

        for pose_id in grp.index:
            if pose_id not in mol_dict:
                n_poses_fail += 1
                continue
            try:
                feat = compute_pose_3d_features(mol_dict[pose_id], pocket_centroid)
                X_rows.append(feat)
                pose_ids.append(pose_id)
                pdb_ids.append(pdb_id)
                n_poses_ok += 1
            except Exception:
                n_poses_fail += 1

    print(f"  pose_3d: {n_complexes} complexes, "
          f"{n_poses_ok} poses OK, {n_poses_fail} failed")
    if not X_rows:
        print("  [ERROR] No valid pose 3D features")
        return
    save_embedding("pose_3d", np.stack(X_rows), pose_ids, pdb_ids)

    # Save feature names
    np.save(EMB_DIR / "pose_3d_feature_names.npy", np.array(_SHAPE3D_NAMES))


# ---------------------------------------------------------------------------
# Complex embeddings (concatenation)
# ---------------------------------------------------------------------------
def process_complex_embeddings() -> None:
    print("\n--- Complex embeddings (concatenation) ---")

    combos = [
        ("complex_esmc600m_pose3d",
         ["pocket_esmc600m", "pose_3d"]),
        ("complex_esmc600m_morgan_pose3d",
         ["pocket_esmc600m", "ligand_morgan", "pose_3d"]),
    ]

    for out_name, parts in combos:
        # Check all parts exist
        missing = [p for p in parts if not (EMB_DIR / f"{p}.npz").exists()]
        if missing:
            print(f"  [SKIP] {out_name}: missing {missing}")
            continue

        # Load each part and find intersection of pose_ids
        loaded = []
        for p in parts:
            X_p, pose_ids_p, pdb_ids_p = load_embedding(p)
            pid_to_i = {pid: i for i, pid in enumerate(pose_ids_p.tolist())}
            loaded.append((X_p, pid_to_i, pdb_ids_p))

        # Intersection (preserve order of first part)
        X0, pi0, pb0 = loaded[0]
        common_set = set(pi0.keys())
        for _, pid_to_i, _ in loaded[1:]:
            common_set &= set(pid_to_i.keys())
        common = [pid for pid in loaded[0][1].keys() if pid in common_set]

        if not common:
            print(f"  [SKIP] {out_name}: no common pose_ids")
            continue

        X_parts = [
            np.stack([X_p[pid_to_i[pid]] for pid in common])
            for X_p, pid_to_i, _ in loaded
        ]
        X_concat = np.concatenate(X_parts, axis=1)
        pdb_ids_common = [pb0[pi0[pid]] for pid in common]

        save_embedding(out_name, X_concat, common, pdb_ids_common)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Pose-level featurization")
    parser.add_argument("--skip-complex",  action="store_true",
                        help="Skip concatenated complex embeddings")
    parser.add_argument("--skip-pose3d",   action="store_true",
                        help="Skip mol2 parsing / 3D pose features (faster)")
    args = parser.parse_args()

    EMB_DIR.mkdir(parents=True, exist_ok=True)

    if not (DATA_DIR / "labels.csv").exists():
        raise FileNotFoundError("data/labels.csv not found — run 01_parse_dataset.py first")

    labels = load_labels()   # indexed by pose_id, has 'pdb_id' column
    pdb_ids_all = load_coreset_pdb_ids()
    print(f"Loaded {len(labels)} poses from {labels['pdb_id'].nunique()} complexes")

    process_esm_embeddings(labels)
    process_pocket_composition(labels, pdb_ids_all)
    process_ligand_morgan(labels)

    if not args.skip_pose3d:
        process_pose_3d(labels)

    if not args.skip_complex:
        process_complex_embeddings()

    print("\n--- Summary of saved embeddings ---")
    for f in sorted(EMB_DIR.glob("*.npz")):
        d = np.load(f, allow_pickle=True)
        print(f"  {f.name:52s}  X={str(d['X'].shape):20s}  n_poses={len(d['pose_ids'])}")

    print("\nDone.")


if __name__ == "__main__":
    main()
