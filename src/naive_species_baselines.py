from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from src.featurizer import bemis_murcko_scaffold
from src.paths import CC_MPNN_DATA, RESULTS_ROOT
from src.rdkit_lgbm import DESCRIPTOR_NAMES, TAX_RANKS, _features
from src.stats import prediction_metrics
from src.stratum import fit_stratum_effect
from src.stratum import remove as stratum_remove
from src.stratum import restore as stratum_restore


NAIVE_BASELINES = (
    "Naive_global_mean",
    "Naive_species_mean",
    "LightGBM_RDKit_no_species_oof_base",
    "LightGBM_RDKit_species_residual_calibration",
)


@dataclass(frozen=True)
class NaiveSpeciesBaselineConfig:
    split: str = "compound_random"
    seed: int = 0
    n_estimators: int = 300
    learning_rate: float = 0.05
    num_leaves: int = 31
    min_child_samples: int = 20
    n_folds: int = 5
    limit_train: int | None = None
    limit_test: int | None = None
    data_dir: str = str(CC_MPNN_DATA)
    out_root: str = str(RESULTS_ROOT / "naive_species_baselines")


def _sample(df: pd.DataFrame, n: int | None, seed: int) -> pd.DataFrame:
    out = df.copy()
    out["source_row_id"] = np.arange(len(out))
    if n is not None and len(out) > n:
        out = out.sample(n=n, random_state=seed)
    return out.reset_index(drop=True)


def _params(cfg: NaiveSpeciesBaselineConfig) -> dict:
    return {
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


def _train_booster(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    *,
    cfg: NaiveSpeciesBaselineConfig,
    valid: tuple[pd.DataFrame, np.ndarray] | None = None,
    num_boost_round: int | None = None,
) -> lgb.Booster:
    dtrain = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
    callbacks = [lgb.log_evaluation(0)]
    valid_sets = None
    if valid is not None:
        X_valid, y_valid = valid
        valid_sets = [lgb.Dataset(X_valid, label=y_valid, reference=dtrain, free_raw_data=False)]
        # SPEC: fixed stopping rule -- no early stopping anywhere in the lgbm family, so the
        # stop point cannot vary by seed. (Folds already hit the 300 cap in every case.)
    return lgb.train(
        _params(cfg),
        dtrain,
        num_boost_round=int(num_boost_round or cfg.n_estimators),
        valid_sets=valid_sets,
        callbacks=callbacks,
    )


def _oof_lightgbm_predictions(
    train: pd.DataFrame,
    test: pd.DataFrame,
    cfg: NaiveSpeciesBaselineConfig,
) -> tuple[np.ndarray, np.ndarray, dict]:
    X, y = _features(train, include_species=False)
    X_test, _ = _features(test, include_species=False)
    n_splits = min(int(cfg.n_folds), len(train))
    if n_splits < 2:
        raise ValueError("OOF residual calibration requires at least two training rows")

    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=cfg.seed)
    oof = np.empty(len(train), dtype=np.float64)
    fold_rounds = []
    t0 = time.time()
    for fold, (tr_idx, va_idx) in enumerate(kfold.split(X), start=1):
        booster = _train_booster(
            X.iloc[tr_idx].reset_index(drop=True),
            y[tr_idx],
            cfg=cfg,
            valid=(X.iloc[va_idx].reset_index(drop=True), y[va_idx]),
        )
        best_round = int(booster.best_iteration or cfg.n_estimators)
        fold_rounds.append(best_round)
        oof[va_idx] = booster.predict(X.iloc[va_idx], num_iteration=best_round)

    final_rounds = int(np.median(fold_rounds)) if fold_rounds else int(cfg.n_estimators)
    final_rounds = max(1, final_rounds)
    final_booster = _train_booster(X, y, cfg=cfg, num_boost_round=final_rounds)
    test_pred = final_booster.predict(X_test, num_iteration=final_rounds)
    meta = {
        "n_folds": int(n_splits),
        "fold_best_iterations": [int(x) for x in fold_rounds],
        "final_num_boost_round": int(final_rounds),
        "n_features": int(X.shape[1]),
        "descriptor_set": "RDKit_6",
        "descriptor_names": list(DESCRIPTOR_NAMES),
        "train_sec": round(time.time() - t0, 2),
    }
    return oof, np.asarray(test_pred, dtype=np.float64), meta


def _stratum_key(df: pd.DataFrame) -> np.ndarray:
    """endpoint x duration stratum label. Constant (identity) on single-stratum data."""
    n = len(df)
    ep = df["endpoint"].astype(str).to_numpy() if "endpoint" in df.columns else np.full(n, "NA")
    du = df["duration"].astype(str).to_numpy() if "duration" in df.columns else np.full(n, "NA")
    return np.array([f"{a}@{b}" for a, b in zip(ep, du)], dtype=object)


def _species_offsets(
    train: pd.DataFrame, residual: np.ndarray
) -> tuple[dict[int, float], dict[int, int], dict[str, float]]:
    """SPEC 4-0b Tier 1' definition: strip endpoint/duration MAIN EFFECTS from the
    residual first, then estimate the per-species scalar offset from the purged
    residual.

    The species term remains a per-species scalar (ladder semantics unchanged);
    the stratum main effect is carried as a separate additive term at prediction
    time. On single-stratum data (discovery) the stratum effect is a constant, so
    base + const + (offset - const) is algebraically identical to the old
    definition -- discovery is unchanged by construction.
    """
    frame = pd.DataFrame(
        {
            "species_idx": train["species_idx"].astype(int).to_numpy(),
            "stratum": _stratum_key(train),
            "residual": np.asarray(residual, dtype=np.float64),
        }
    )
    stratum_eff = frame.groupby("stratum")["residual"].mean()
    frame["purged"] = frame["residual"] - frame["stratum"].map(stratum_eff).to_numpy()
    offsets = frame.groupby("species_idx")["purged"].mean().to_dict()
    counts = frame["species_idx"].value_counts().to_dict()
    return (
        {int(k): float(v) for k, v in offsets.items()},
        {int(k): int(v) for k, v in counts.items()},
        {str(k): float(v) for k, v in stratum_eff.items()},
    )


def _base_prediction_frame(
    test: pd.DataFrame,
    *,
    pred: np.ndarray,
    cfg: NaiveSpeciesBaselineConfig,
    model_name: str,
    backbone: str,
    species_mode: str,
    injection_location: str,
    species_control_type: str,
    extra_columns: dict[str, np.ndarray | list | str | float | int | bool] | None = None,
) -> pd.DataFrame:
    y_true = test["target_log10"].to_numpy(np.float64)
    out = test.copy()
    out["species_idx_original"] = out["species_idx"].astype(int)
    out["species_for_model"] = out["species_idx"].astype(int)
    out["input_species"] = out["species"].astype(str)
    out["scaffold"] = out["smiles"].map(bemis_murcko_scaffold)
    out["compound_key"] = out["smiles"].astype(str)
    out["scaffold_key"] = out["scaffold"].astype(str)
    out["pred_log10"] = np.asarray(pred, dtype=np.float64)
    out["true_log10"] = y_true
    out["error_log10"] = out["pred_log10"] - y_true
    out["model_name"] = model_name
    out["backbone"] = backbone
    out["species_mode"] = species_mode
    out["injection_location"] = injection_location
    out["species_control"] = species_control_type
    out["species_control_type"] = species_control_type
    out["is_shuffled"] = False
    out["is_zero_species"] = False
    out["is_dummy_species"] = False
    out["baseline"] = model_name
    out["split"] = cfg.split
    out["seed"] = int(cfg.seed)
    if extra_columns:
        for key, value in extra_columns.items():
            out[key] = value
    return out


def _write_prediction_and_run(
    pred_frame: pd.DataFrame,
    *,
    cfg: NaiveSpeciesBaselineConfig,
    run_id: str,
    run_type: str,
    run_summary: dict,
) -> dict:
    out_root = Path(cfg.out_root)
    for subdir in ["predictions", "runs", "metrics"]:
        (out_root / subdir).mkdir(parents=True, exist_ok=True)

    pred_path = out_root / "predictions" / f"{run_id}.csv"
    run_path = out_root / "runs" / f"{run_id}.json"
    metric_path = out_root / "metrics" / f"{run_id}.json"
    pred_frame.to_csv(pred_path, index=False, encoding="utf-8")

    metrics = prediction_metrics(pred_frame)
    payload = {
        "config": {
            **asdict(cfg),
            "run_id": run_id,
            "prediction_file": str(pred_path),
            **run_summary,
        },
        "A": metrics,
        "B": None,
        "C": None,
        "D": {
            "trainable_params": 0,
            "species_trainable_params": 0,
            **run_summary,
        },
    }
    with open(run_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    metric_payload = {
        "run_id": run_id,
        "run_type": run_type,
        "config": payload["config"],
        "metrics": metrics,
        "species_bin_metrics": None,
        "training_or_model_summary": payload["D"],
    }
    with open(metric_path, "w", encoding="utf-8") as f:
        json.dump(metric_payload, f, ensure_ascii=False, indent=2)
    return {
        "run_id": run_id,
        "run_type": run_type,
        "split": cfg.split,
        "seed": int(cfg.seed),
        "model_name": str(pred_frame["model_name"].iloc[0]),
        "backbone": str(pred_frame["backbone"].iloc[0]),
        "species_mode": str(pred_frame["species_mode"].iloc[0]),
        "prediction_file": str(pred_path),
        "run_file": str(run_path),
        "metrics_file": str(metric_path),
        **metrics,
        **run_summary,
    }


def run_naive_species_baselines(cfg: NaiveSpeciesBaselineConfig) -> list[dict]:
    data_dir = Path(cfg.data_dir)
    out_root = Path(cfg.out_root)
    for subdir in ["predictions", "runs", "metrics", "parameter_counts", "summary_tables"]:
        (out_root / subdir).mkdir(parents=True, exist_ok=True)

    train_raw = pd.read_csv(data_dir / f"{cfg.split}_train.csv")
    test_raw = pd.read_csv(data_dir / f"{cfg.split}_test.csv")
    train = _sample(train_raw, cfg.limit_train, cfg.seed)
    test = _sample(test_raw, cfg.limit_test, cfg.seed + 17)
    # SPEC 4-0b(A): fit the additive endpoint/duration main effect on train (original
    # units), then fit every baseline in adjusted space. Test targets stay original;
    # each prediction is restored before metrics.
    stratum_eff = fit_stratum_effect(train, train["target_log10"].to_numpy(np.float64))
    train = train.copy()
    train["target_log10"] = stratum_remove(
        train, train["target_log10"].to_numpy(np.float64), stratum_eff
    )
    y_train = train["target_log10"].to_numpy(np.float64)
    global_mean = float(np.mean(y_train))
    species_mean = train.groupby(train["species_idx"].astype(int))["target_log10"].mean().to_dict()
    species_count = train["species_idx"].astype(int).value_counts().to_dict()
    test_species = test["species_idx"].astype(int)
    is_cold = ~test_species.isin(species_mean)

    outputs: list[dict] = []
    global_pred = np.full(len(test), global_mean, dtype=np.float64)
    global_frame = _base_prediction_frame(
        test,
        pred=stratum_restore(test, global_pred, stratum_eff),
        cfg=cfg,
        model_name="Naive_global_mean",
        backbone="naive_species",
        species_mode="global_mean",
        injection_location="not_applicable",
        species_control_type="none",
        extra_columns={
            "species_train_count": test_species.map(species_count).fillna(0).astype(int).to_numpy(),
            "is_cold_species": is_cold.to_numpy(),
        },
    )
    outputs.append(
        _write_prediction_and_run(
            global_frame,
            cfg=cfg,
            run_id=f"Naive_global_mean_{cfg.split}_s{cfg.seed}",
            run_type="naive_species",
            run_summary={
                "baseline_family": "naive_species_mean",
                "global_train_mean": global_mean,
                "n_features": 0,
                "n_trees": 0,
            },
        )
    )

    species_pred = test_species.map(species_mean).fillna(global_mean).to_numpy(np.float64)
    species_frame = _base_prediction_frame(
        test,
        pred=stratum_restore(test, species_pred, stratum_eff),
        cfg=cfg,
        model_name="Naive_species_mean",
        backbone="naive_species",
        species_mode="species_mean",
        injection_location="species_mean_lookup",
        species_control_type="true",
        extra_columns={
            "species_train_count": test_species.map(species_count).fillna(0).astype(int).to_numpy(),
            "is_cold_species": is_cold.to_numpy(),
            "cold_species_fallback": "global_mean",
        },
    )
    outputs.append(
        _write_prediction_and_run(
            species_frame,
            cfg=cfg,
            run_id=f"Naive_species_mean_{cfg.split}_s{cfg.seed}",
            run_type="naive_species",
            run_summary={
                "baseline_family": "naive_species_mean",
                "global_train_mean": global_mean,
                "n_species_offsets": int(len(species_mean)),
                "n_cold_test_rows": int(is_cold.sum()),
                "n_features": 0,
                "n_trees": 0,
            },
        )
    )

    oof_pred, test_base_pred, lgbm_meta = _oof_lightgbm_predictions(train, test, cfg)
    residual = y_train - oof_pred
    offsets, offset_counts, stratum_eff = _species_offsets(train, residual)
    test_stratum_eff = (
        pd.Series(_stratum_key(test)).map(stratum_eff).fillna(0.0).to_numpy(np.float64)
    )
    test_offsets = test_species.map(offsets).fillna(0.0).to_numpy(np.float64)
    residual_calibrated = test_base_pred + test_stratum_eff + test_offsets

    base_frame = _base_prediction_frame(
        test,
        pred=stratum_restore(test, test_base_pred, stratum_eff),
        cfg=cfg,
        model_name="LightGBM_RDKit_no_species_oof_base",
        backbone="lightgbm_rdkit",
        species_mode="no_species_oof_base",
        injection_location="not_applicable",
        species_control_type="none",
        extra_columns={
            "species_train_count": test_species.map(offset_counts).fillna(0).astype(int).to_numpy(),
            "is_cold_species": is_cold.to_numpy(),
            "residual_offset_log10": np.zeros(len(test), dtype=np.float64),
        },
    )
    outputs.append(
        _write_prediction_and_run(
            base_frame,
            cfg=cfg,
            run_id=f"LightGBM_RDKit_no_species_oof_base_{cfg.split}_s{cfg.seed}",
            run_type="naive_residual_calibration",
            run_summary={
                "baseline_family": "rdkit_lgbm_oof_base",
                **lgbm_meta,
                "n_trees": int(lgbm_meta["final_num_boost_round"]),
            },
        )
    )

    calibrated_frame = _base_prediction_frame(
        test,
        pred=stratum_restore(test, residual_calibrated, stratum_eff),
        cfg=cfg,
        model_name="LightGBM_RDKit_species_residual_calibration",
        backbone="lightgbm_rdkit",
        species_mode="species_residual_calibration",
        injection_location="posthoc_species_residual_mean",
        species_control_type="true",
        extra_columns={
            "species_train_count": test_species.map(offset_counts).fillna(0).astype(int).to_numpy(),
            "is_cold_species": is_cold.to_numpy(),
            "residual_offset_log10": test_offsets,
            "stratum_effect_log10": test_stratum_eff,
            "cold_species_fallback": "zero_residual_offset",
        },
    )
    outputs.append(
        _write_prediction_and_run(
            calibrated_frame,
            cfg=cfg,
            run_id=f"LightGBM_RDKit_species_residual_calibration_{cfg.split}_s{cfg.seed}",
            run_type="naive_residual_calibration",
            run_summary={
                "baseline_family": "rdkit_lgbm_species_residual_calibration",
                **lgbm_meta,
                "n_trees": int(lgbm_meta["final_num_boost_round"]),
                "n_species_offsets": int(len(offsets)),
                "n_cold_test_rows": int(is_cold.sum()),
                "n_strata": int(len(stratum_eff)),
                "offset_fit": "train_oof_residual_mean_by_species_after_endpoint_duration_purge",
            },
        )
    )
    return outputs


# ---------------------------------------------------------------------------
# GAP (Step 4 Phase 2): naive taxon-group mean baseline (Tier 3a/3b, naive backbone).
# Hierarchical back-off group mean over the SAME 5 core ranks as item 7 / Task A
# (kingdom/phylum/class/order/family; no superclass backfill). For a test row, use the
# mean of the finest taxonomic group present in train (family -> order -> ... -> kingdom
# -> global). Unresolved rank -> "__unknown__" bucket; rows are never dropped. `true` uses
# the real species->taxonomy map; `shuffled` control permutes species->taxonomy (seeded,
# consistent train+test), which should collapse the taxonomy signal.
# ---------------------------------------------------------------------------
def _species_taxonomy_map(full: pd.DataFrame, ranks: list[str]) -> dict[int, dict[str, str]]:
    sp = full.drop_duplicates("species_idx")
    out: dict[int, dict[str, str]] = {}
    for _, row in sp.iterrows():
        d = {}
        for r in ranks:
            v = row[r] if r in row else None
            d[r] = "__unknown__" if (pd.isna(v) or str(v).strip() == "") else str(v)
        out[int(row["species_idx"])] = d
    return out


def _taxon_cols_via_remap(df: pd.DataFrame, ranks: list[str],
                          sp2tax: dict[int, dict[str, str]], remap: dict[int, int]) -> dict[str, np.ndarray]:
    eff = df["species_idx"].astype(int).map(remap).to_numpy()
    unk = {r: "__unknown__" for r in ranks}
    return {r: np.array([sp2tax.get(int(s), unk)[r] for s in eff], dtype=object) for r in ranks}


def _level_key(cols: dict[str, np.ndarray], ranks: list[str], k: int) -> np.ndarray:
    key = pd.Series(cols[ranks[0]]).astype(str)
    for j in range(1, k):
        key = key.str.cat(pd.Series(cols[ranks[j]]).astype(str), sep="|")
    return key.to_numpy()


def _taxon_backoff_predict(train_cols, test_cols, ranks, y_train, global_mean):
    # level k (1..len) group means on train; predict test at the finest populated level
    levelmeans = []
    for k in range(1, len(ranks) + 1):
        keys = _level_key(train_cols, ranks, k)
        levelmeans.append(pd.Series(y_train).groupby(keys).mean().to_dict())
    n = len(test_cols[ranks[0]])
    pred = np.full(n, np.nan, dtype=np.float64)
    remaining = np.ones(n, dtype=bool)
    hit_level = np.zeros(n, dtype=int)
    for k in range(len(ranks), 0, -1):
        keys = _level_key(test_cols, ranks, k)
        lm = levelmeans[k - 1]
        for i in np.where(remaining)[0]:
            v = lm.get(keys[i])
            if v is not None:
                pred[i] = v; remaining[i] = False; hit_level[i] = k
    pred[remaining] = global_mean  # hit_level 0 => global fallback
    return pred, hit_level


def run_naive_taxonomy_baselines(cfg: NaiveSpeciesBaselineConfig) -> list[dict]:
    data_dir = Path(cfg.data_dir)
    out_root = Path(cfg.out_root)
    for subdir in ["predictions", "runs", "metrics", "parameter_counts", "summary_tables"]:
        (out_root / subdir).mkdir(parents=True, exist_ok=True)
    train_raw = pd.read_csv(data_dir / f"{cfg.split}_train.csv")
    test_raw = pd.read_csv(data_dir / f"{cfg.split}_test.csv")
    train = _sample(train_raw, cfg.limit_train, cfg.seed)
    test = _sample(test_raw, cfg.limit_test, cfg.seed + 17)
    stratum_eff = fit_stratum_effect(train, train["target_log10"].to_numpy(np.float64))
    train = train.copy()
    train["target_log10"] = stratum_remove(train, train["target_log10"].to_numpy(np.float64), stratum_eff)
    y_train = train["target_log10"].to_numpy(np.float64)
    global_mean = float(np.mean(y_train))
    full = pd.concat([train_raw, test_raw], ignore_index=True)

    outputs: list[dict] = []
    for tax, lab in [("taxonomy_original", "original"), ("taxonomy_ncbi", "ncbi")]:
        ranks = TAX_RANKS[tax]
        sp2tax = _species_taxonomy_map(full, ranks)
        for control in ["true", "shuffled"]:
            if control == "true":
                remap = {s: s for s in sp2tax}
            else:
                rng = np.random.RandomState(cfg.seed + 991)
                uniq = np.array(sorted(sp2tax.keys()))
                remap = {int(a): int(b) for a, b in zip(uniq, rng.permutation(uniq))}
            tr_cols = _taxon_cols_via_remap(train, ranks, sp2tax, remap)
            te_cols = _taxon_cols_via_remap(test, ranks, sp2tax, remap)
            pred, hit = _taxon_backoff_predict(tr_cols, te_cols, ranks, y_train, global_mean)
            model_name = f"Naive_{'shuffled_' if control == 'shuffled' else ''}taxon_mean_{lab}"
            frame = _base_prediction_frame(
                test, pred=stratum_restore(test, pred, stratum_eff), cfg=cfg,
                model_name=model_name, backbone="naive_species",
                species_mode=f"taxon_mean_{lab}", injection_location="taxon_group_backoff_mean",
                species_control_type=control,
                extra_columns={"taxon_hit_level": hit, "n_global_fallback": int((hit == 0).sum())},
            )
            frame["is_shuffled"] = (control == "shuffled")
            outputs.append(_write_prediction_and_run(
                frame, cfg=cfg, run_id=f"{model_name}_{cfg.split}_s{cfg.seed}", run_type="naive_species",
                run_summary={"baseline_family": "naive_taxon_mean", "taxonomy": lab, "control": control,
                             "n_ranks": len(ranks), "global_train_mean": global_mean,
                             "n_global_fallback": int((hit == 0).sum()), "n_features": 0, "n_trees": 0},
            ))
    return outputs
