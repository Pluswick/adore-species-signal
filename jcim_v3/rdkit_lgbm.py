from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

from jcim_v3.paths import CC_MPNN_DATA, RESULTS_ROOT, add_ccmpnn_to_path
from jcim_v3.featurizer import bemis_murcko_scaffold
from jcim_v3.species_controls import apply_species_control
from jcim_v3.stratum import fit_stratum_effect
from jcim_v3.stratum import remove as stratum_remove
from jcim_v3.stratum import restore as stratum_restore

add_ccmpnn_to_path()

from ccmpnn.metrics import perf_metrics, species_binned_rmse  # noqa: E402

DESCRIPTOR_NAMES = (
    "MolLogP",
    "TPSA",
    "MolWt",
    "NumHDonors",
    "NumHAcceptors",
    "NumRotatableBonds",
)

LIGHTGBM_BASELINES = {
    "LightGBM_RDKit_no_species": {
        "species_mode": "no_species",
        "species_control": "none",
        "include_species": False,
    },
    "LightGBM_RDKit_species_categorical": {
        "species_mode": "species_categorical",
        "species_control": "true",
        "include_species": True,
    },
    "LightGBM_RDKit_zero_species_categorical": {
        "species_mode": "zero_species_categorical",
        "species_control": "zero",
        "include_species": True,
    },
    "LightGBM_RDKit_shuffled_species_categorical": {
        "species_mode": "shuffled_species_categorical",
        "species_control": "shuffled",
        "include_species": True,
    },
    "LightGBM_RDKit_dummy_species_categorical": {
        "species_mode": "dummy_species_categorical",
        "species_control": "dummy",
        "include_species": True,
    },
    # GAP item 7 — Tier 3 taxonomy (rank-wise categorical). Scope EXPANSION (not a correction):
    # the manuscript scope excludes taxonomy from performance; this adds it as a tier.
    "LightGBM_RDKit_taxonomy_original": {
        "species_mode": "taxonomy_original", "species_control": "true",
        "include_species": True, "species_repr": "taxonomy_original",
    },
    "LightGBM_RDKit_shuffled_taxonomy_original": {
        "species_mode": "shuffled_taxonomy_original", "species_control": "shuffled",
        "include_species": True, "species_repr": "taxonomy_original",
    },
    "LightGBM_RDKit_taxonomy_ncbi": {
        "species_mode": "taxonomy_ncbi", "species_control": "true",
        "include_species": True, "species_repr": "taxonomy_ncbi",
    },
    "LightGBM_RDKit_shuffled_taxonomy_ncbi": {
        "species_mode": "shuffled_taxonomy_ncbi", "species_control": "shuffled",
        "include_species": True, "species_repr": "taxonomy_ncbi",
    },
    # rank-truncation study (native ranks, tier-3a lineage; main control only, exploratory family):
    "LightGBM_RDKit_taxonomy_genus": {
        "species_mode": "taxonomy_genus", "species_control": "true",
        "include_species": True, "species_repr": "taxonomy_genus",
    },
    "LightGBM_RDKit_taxonomy_genusfamily": {
        "species_mode": "taxonomy_genusfamily", "species_control": "true",
        "include_species": True, "species_repr": "taxonomy_genusfamily",
    },
}

# rank columns per taxonomy variant (present in the split CSVs). Unresolved ranks collapse to an
# "unknown" code; rows are never dropped (all tiers share one test set).
# ADORE §A-1: 4-rank (class/order/family/genus). kingdom/phylum absent (tax_group-degenerate) -> not used.
# original = ADORE native taxonomy (tax_*, 100% coverage); ncbi = self-resolved via taxdump (ncbi_*).
TAX_RANKS = {
    "taxonomy_original": ["tax_class", "tax_order", "tax_family", "tax_genus"],
    "taxonomy_ncbi": ["ncbi_class", "ncbi_order", "ncbi_family", "ncbi_genus"],
    # rank-truncation study (native ranks): where does the taxonomy signal saturate by depth?
    "taxonomy_genus": ["tax_genus"],
    "taxonomy_genusfamily": ["tax_family", "tax_genus"],
}


@dataclass(frozen=True)
class RDKitLGBMConfig:
    baseline: str
    split: str = "scaffold"
    seed: int = 0
    n_estimators: int = 300
    learning_rate: float = 0.05
    num_leaves: int = 31
    min_child_samples: int = 20
    val_frac: float = 0.1
    limit_train: int | None = None
    limit_test: int | None = None
    data_dir: str = str(CC_MPNN_DATA)
    out_root: str = str(RESULTS_ROOT / "lgbm_rdkit")


def _features(df: pd.DataFrame, include_species: bool, species_repr: str = "species_idx",
              tax_cols: list | None = None) -> tuple[pd.DataFrame, np.ndarray]:
    cache = {}
    rows = []
    for smi in df["smiles"]:
        if smi not in cache:
            mol = Chem.MolFromSmiles(str(smi))
            if mol is None:
                raise ValueError(f"Invalid SMILES for RDKit descriptor baseline: {smi!r}")
            cache[smi] = [
                Descriptors.MolLogP(mol),
                Descriptors.TPSA(mol),
                Descriptors.MolWt(mol),
                Descriptors.NumHDonors(mol),
                Descriptors.NumHAcceptors(mol),
                Descriptors.NumRotatableBonds(mol),
            ]
        rows.append(cache[smi])
    X = pd.DataFrame(rows, columns=list(DESCRIPTOR_NAMES))
    # SPEC 4-0b(A): endpoint/duration are NOT features. They are controlled uniformly for
    # every tier and backbone by additive target residualization (jcim_v3.stratum), so no
    # tier can exploit an endpoint x structure interaction that others cannot.
    if include_species:
        if species_repr == "species_idx":
            X["species_idx"] = df["species_idx"].astype("int32").to_numpy()
        else:  # GAP item 7: taxonomy rank codes (already integer-coded on the frame)
            for c in (tax_cols or []):
                X[c] = df[c].astype("int32").to_numpy()
    return X, df["target_log10"].to_numpy(np.float64)


def _sample(df: pd.DataFrame, n: int | None, seed: int) -> pd.DataFrame:
    out = df.copy()
    out["source_row_id"] = np.arange(len(out))
    if n is not None and len(out) > n:
        out = out.sample(n=n, random_state=seed)
    return out.reset_index(drop=True)


def _species_lookup(frame: pd.DataFrame) -> dict[int, str]:
    cols = frame[["species_idx", "species"]].drop_duplicates("species_idx")
    return {int(row.species_idx): str(row.species) for row in cols.itertuples(index=False)}


def _uncontrolled_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["species_idx_original"] = out["species_idx"].astype(int)
    out["species_for_model"] = out["species_idx"].astype(int)
    out["input_species"] = out["species"].astype(str)
    out["species_control_type"] = "none"
    out["is_shuffled"] = False
    out["is_zero_species"] = False
    out["is_dummy_species"] = False
    return out


def _controlled_frame(
    df: pd.DataFrame,
    *,
    mode: str,
    seed: int,
    n_species: int,
    species_lookup: dict[int, str],
) -> pd.DataFrame:
    if mode == "none":
        return _uncontrolled_frame(df)
    return apply_species_control(
        df,
        mode=mode,
        seed=seed,
        n_species=n_species,
        species_lookup=species_lookup,
    ).frame


def run_rdkit_lgbm(cfg: RDKitLGBMConfig) -> dict:
    if cfg.baseline not in LIGHTGBM_BASELINES:
        raise ValueError(f"Unknown RDKit LightGBM baseline: {cfg.baseline!r}")
    baseline_spec = LIGHTGBM_BASELINES[cfg.baseline]
    include_species = bool(baseline_spec["include_species"])
    species_mode = str(baseline_spec["species_mode"])
    species_control = str(baseline_spec["species_control"])
    species_repr = str(baseline_spec.get("species_repr", "species_idx"))
    data_dir = Path(cfg.data_dir)
    out_root = Path(cfg.out_root)
    (out_root / "runs").mkdir(parents=True, exist_ok=True)
    (out_root / "predictions").mkdir(parents=True, exist_ok=True)

    tr_full_raw = pd.read_csv(data_dir / f"{cfg.split}_train.csv")
    te_full_raw = pd.read_csv(data_dir / f"{cfg.split}_test.csv")
    tr_full = _sample(tr_full_raw, cfg.limit_train, cfg.seed)
    te = _sample(te_full_raw, cfg.limit_test, cfg.seed + 17)
    full = pd.concat([tr_full_raw, te_full_raw], ignore_index=True)
    n_species = int(full["species_idx"].max()) + 1
    species_lookup = _species_lookup(full)

    rng = np.random.default_rng(cfg.seed)
    perm = rng.permutation(len(tr_full))
    n_val = max(1, int(len(tr_full) * cfg.val_frac))
    n_val = min(n_val, len(tr_full) - 1)
    vi, ti = perm[:n_val], perm[n_val:]

    # SPEC: with a fixed stopping rule the val carve is never used for stopping, yet a
    # seed-dependent 10% holdout changed the TRAINING SET each seed -- injecting variance
    # of the same magnitude as the species effect (shuffled beat no_species in 3/10 seeds).
    # Train on the FULL training set, matching the OOF base (which uses 100% and is stable).
    # val_raw is retained as an unused monitoring set only.
    train_raw = tr_full.reset_index(drop=True)
    val_raw = tr_full.iloc[vi].reset_index(drop=True)
    train_frame = _controlled_frame(
        train_raw,
        mode=species_control,
        seed=cfg.seed + 101,
        n_species=n_species,
        species_lookup=species_lookup,
    )
    val_frame = _controlled_frame(
        val_raw,
        mode=species_control,
        seed=cfg.seed + 202,
        n_species=n_species,
        species_lookup=species_lookup,
    )
    test_frame = _controlled_frame(
        te,
        mode=species_control,
        seed=cfg.seed + 303,
        n_species=n_species,
        species_lookup=species_lookup,
    )

    # SPEC 4-0b(A): remove the train-estimated additive endpoint/duration main effect from
    # the FITTING targets only. Test targets stay in original units; predictions are
    # restored below. Identical operation for every tier/backbone (jcim_v3.stratum).
    stratum_eff = fit_stratum_effect(tr_full, tr_full["target_log10"].to_numpy(np.float64))
    train_frame = train_frame.copy()
    val_frame = val_frame.copy()
    train_frame["target_log10"] = stratum_remove(
        train_frame, train_frame["target_log10"].to_numpy(np.float64), stratum_eff)
    val_frame["target_log10"] = stratum_remove(
        val_frame, val_frame["target_log10"].to_numpy(np.float64), stratum_eff)

    # GAP item 7: taxonomy tier — code each rank on the species the model sees
    # (species_for_model), so the shuffled control permutes taxonomy automatically.
    tax_cols = None
    if include_species and species_repr != "species_idx":
        ranks = TAX_RANKS[species_repr]
        sp = full.drop_duplicates("species_idx").set_index("species_idx")
        code_maps = {}
        for r in ranks:
            vals = sp[r].astype("string").fillna("__unknown__")
            vals = vals.mask(vals.str.strip() == "", "__unknown__")
            cats = pd.Categorical(vals)
            code_maps[r] = {int(k): int(v) for k, v in zip(sp.index.astype(int), cats.codes)}
        tax_cols = [f"{r}__code" for r in ranks]
        for frame in (train_frame, val_frame, test_frame):
            sfm = frame["species_for_model"].astype(int)
            for r, col in zip(ranks, tax_cols):
                frame[col] = sfm.map(code_maps[r]).fillna(-1).astype("int32")

    X_train, y_train = _features(train_frame, include_species, species_repr, tax_cols)
    X_val, y_val = _features(val_frame, include_species, species_repr, tax_cols)
    Xte, yte = _features(test_frame, include_species, species_repr, tax_cols)
    categorical = ((["species_idx"] if species_repr == "species_idx" else tax_cols)
                   if include_species else [])

    dtrain = lgb.Dataset(
        X_train,
        label=y_train,
        categorical_feature=categorical or "auto",
        free_raw_data=False,
    )
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": cfg.learning_rate,
        "num_leaves": cfg.num_leaves,
        "min_child_samples": cfg.min_child_samples,
        "seed": cfg.seed,
        "deterministic": True,
        "force_row_wise": True,
        "verbose": -1,
    }

    t0 = time.time()
    # SPEC: FIXED stopping rule for every lgbm tier. Early stopping made the stop point
    # seed-dependent, which injected noise of the same magnitude as the species effect
    # (shuffled beat no_species in 3/10 seeds). All tiers now train exactly
    # cfg.n_estimators rounds -- matching the OOF base, whose folds all hit the 300 cap.
    booster = lgb.train(
        params,
        dtrain,
        num_boost_round=cfg.n_estimators,
        valid_sets=[dval],
        callbacks=[lgb.log_evaluation(0)],
    )
    train_sec = time.time() - t0
    pred = booster.predict(Xte, num_iteration=cfg.n_estimators)
    # restore the stratum main effect: predictions were made in adjusted space
    pred = stratum_restore(test_frame, np.asarray(pred, dtype=np.float64), stratum_eff)
    train_counts = tr_full["species_idx"].value_counts().to_dict()
    run_id = f"{cfg.baseline}_{cfg.split}_s{cfg.seed}"

    pred_frame = test_frame.copy()
    if "species_idx_original" not in pred_frame.columns:
        pred_frame["species_idx_original"] = pred_frame["species_idx"].astype(int)
    pred_frame["scaffold"] = pred_frame["smiles"].map(bemis_murcko_scaffold)
    pred_frame["compound_key"] = pred_frame["smiles"].astype(str)
    pred_frame["scaffold_key"] = pred_frame["scaffold"].astype(str)
    pred_frame["pred_log10"] = pred
    pred_frame["true_log10"] = yte
    pred_frame["error_log10"] = pred - yte
    pred_frame["model_name"] = cfg.baseline
    pred_frame["backbone"] = "lightgbm_rdkit"
    pred_frame["species_mode"] = species_mode
    pred_frame["injection_location"] = "categorical_feature" if include_species else "not_applicable"
    pred_frame["species_control"] = species_control
    pred_frame["species_control_type"] = species_control
    pred_frame["baseline"] = cfg.baseline
    pred_frame["split"] = cfg.split
    pred_frame["seed"] = cfg.seed
    pred_path = out_root / "predictions" / f"{run_id}.csv"
    json_path = out_root / "runs" / f"{run_id}.json"
    pred_frame.to_csv(pred_path, index=False, encoding="utf-8")

    result = {
        "config": {
            **asdict(cfg),
            "run_id": run_id,
            "descriptor_set": "RDKit_6",
            "descriptor_names": list(DESCRIPTOR_NAMES),
            "species_input": "categorical" if include_species else "none",
            "species_mode": species_mode,
            "species_control": species_control,
            "prediction_file": str(pred_path),
        },
        "A": perf_metrics(pred, yte),
        "B": species_binned_rmse(pred, yte, te["species_idx"].to_numpy(), train_counts),
        "C": None,
        "D": {
            "n_trees": int(booster.best_iteration or cfg.n_estimators),
            "n_features": int(X_train.shape[1]),
            "train_sec": round(train_sec, 2),
            "device": "cpu",
            "reference_mordred_baseline": "preserved in CC-MPNN/data/runs as supplementary/reference only",
        },
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return result
