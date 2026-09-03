"""Task B (item 8) — GNN-native Tier 1' (residual calibration) via 5-fold OOF.

For each (backbone, split, seed):
  * 5-fold KFold over the TRAIN rows (shuffle, random_state=seed, mirrors LightGBM oof_base).
  * Each fold: train a *no_species* GNN on 4/5 (standard matched budget: val_frac=0.1 +
    early stopping, per director decision), predict the held-out 1/5 -> OOF no-species preds.
  * Per-species additive offset  = mean_over_train_rows_of_species( true_log10 - oof_pred )
    keyed by species_idx_original (true species identity; leakage-free via OOF).
  * Tier 1' TEST prediction = existing no_species TEST pred  +  offset[test-row species]
    (species unseen in train -> offset 0, i.e. falls back to no_species).

Writes  {bb}_tier1prime_oof_{split}_s{seed}_e100_nfull.csv  in the GNN predictions dir,
same schema as the other prediction CSVs so bootstrap_q2_ladder.load_gnn can read it.
Resumable: skips a (bb,split,seed) whose output already exists.
"""
from __future__ import annotations
import sys, time, argparse
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import KFold

sys.path.insert(0, r".")
from src.runner import (
    V3RunConfig, _setup_reproducible, _carve_val, _train, _predict, _species_lookup,
    model_spec_from_variant, build_v3_model, GraphDataset, StandardScaler,
    apply_species_control, fit_stratum_effect, stratum_remove, stratum_restore, ATOM_FDIM,
)
from src.paths import CC_MPNN_DATA
from src.naive_species_baselines import _stratum_key  # identical stratum labelling as LightGBM rung
from src.tier_input_guard import TierInputDegenerate

DATA = Path(r".\results\q2_v4\data")
PRED = Path(r".\results\q2_v4\runs\gnn\predictions")
BASE_VARIANT = "no_species"
OOF_VARIANT = "tier1prime_oof"
N_FOLDS = 5


def _fold_holdout_pred(spec, cfg, n_species, species_lookup, fold_tr, fold_ho, device):
    """Mirror run_v3_smoke exactly; the held-out fold plays test's role. Returns preds
    aligned to fold_ho row order, restored to original log10 units."""
    cm = spec.species_control
    tr_raw, va_raw = _carve_val(fold_tr, cfg.seed, cfg.val_frac)
    tr = apply_species_control(tr_raw, mode=cm, seed=cfg.seed + 101, n_species=n_species, species_lookup=species_lookup)
    va = apply_species_control(va_raw, mode=cm, seed=cfg.seed + 202, n_species=n_species, species_lookup=species_lookup)
    ho = apply_species_control(fold_ho, mode=cm, seed=cfg.seed + 303, n_species=n_species, species_lookup=species_lookup)
    eff = fit_stratum_effect(tr.frame, tr.frame["target_log10"].to_numpy(np.float64))
    tr.frame["target_log10"] = stratum_remove(tr.frame, tr.frame["target_log10"].to_numpy(np.float64), eff)
    va.frame["target_log10"] = stratum_remove(va.frame, va.frame["target_log10"].to_numpy(np.float64), eff)
    base = GraphDataset(tr.frame)
    ts = StandardScaler().fit(base.target.reshape(-1, 1))
    ds_sc = None if base.raw_desc is None else StandardScaler().fit(base.raw_desc)
    tr_ds = GraphDataset(tr.frame, target_scaler=ts, desc_scaler=ds_sc)
    va_ds = GraphDataset(va.frame, target_scaler=ts, desc_scaler=ds_sc)
    ho_ds = GraphDataset(ho.frame, target_scaler=ts, desc_scaler=ds_sc)
    model = build_v3_model(spec=spec, atom_fdim=ATOM_FDIM, n_species=n_species, hidden=cfg.hidden,
                           depth=cfg.depth, dropout=cfg.dropout, species_emb_dim=cfg.species_emb_dim,
                           desc_fdim=tr_ds.desc_fdim)
    model, _ = _train(model, tr_ds, va_ds, cfg=cfg, device=device)
    pred = _predict(model, ho_ds, cfg.batch_size, device)
    return stratum_restore(ho.frame, np.asarray(pred, dtype=np.float64), eff)


def run_one(backbone: str, split: str, seed: int, control: str = "true") -> dict:
    ovar = OOF_VARIANT if control == "true" else f"shuffled_{OOF_VARIANT}"
    out = PRED / f"{backbone}_{ovar}_{split}_s{seed}_e100_nfull.csv"
    if out.exists():
        return {"skipped": True, "out": str(out)}
    cfg = V3RunConfig(backbone=backbone, variant=BASE_VARIANT, split=split, seed=seed, epochs=100,
                      batch_size=256, lr=5e-4, weight_decay=1e-5, hidden=300, depth=3, dropout=0.1,
                      species_emb_dim=16, val_frac=0.1, limit_train=None, limit_test=None)
    spec = model_spec_from_variant(backbone, BASE_VARIANT)
    device = _setup_reproducible(seed)
    tr_full = pd.read_csv(DATA / f"{split}_train.csv").reset_index(drop=True)
    te_full = pd.read_csv(DATA / f"{split}_test.csv")
    full = pd.concat([tr_full, te_full], ignore_index=True)
    n_species = int(full["species_idx"].max()) + 1
    species_lookup = _species_lookup(full)

    t0 = time.time()
    cache_dir = PRED.parent / "oof_cache"; cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"{backbone}_{split}_s{seed}_oof.npy"
    if cache.exists():
        oof = np.load(cache)
        if len(oof) != len(tr_full):
            oof = None
        else:
            print(f"    [oof-cache hit] {cache.name}", flush=True)
    else:
        oof = None
    if oof is None:
        oof = np.full(len(tr_full), np.nan, dtype=np.float64)
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        for k, (tr_idx, va_idx) in enumerate(kf.split(np.arange(len(tr_full))), start=1):
            fold_tr = tr_full.iloc[tr_idx].reset_index(drop=True)
            fold_ho = tr_full.iloc[va_idx].reset_index(drop=True)
            pred = _fold_holdout_pred(spec, cfg, n_species, species_lookup, fold_tr, fold_ho, device)
            assert len(pred) == len(va_idx), f"fold {k}: {len(pred)} preds vs {len(va_idx)} holdout rows"
            oof[va_idx] = pred
            print(f"    fold {k}/{N_FOLDS} holdout={len(va_idx)} done ({time.time()-t0:.0f}s)", flush=True)
        assert not np.isnan(oof).any(), "OOF has unfilled rows"
        np.save(cache, oof)

    # SPEC 4-0b Tier 1' definition, mirrored EXACTLY from the LightGBM rung
    # (naive_species_baselines._species_offsets + its test application), so the GNN and LightGBM
    # Tier 1' rungs compute the same quantity and stay comparable on MULTI-STRATUM (replication):
    #   1. strip the endpoint x duration residual main-effect (stratum_eff) from the OOF residual,
    #   2. per-species scalar offset = mean of the *purged* residual,
    #   3. at test: pred = no_species_base + stratum_eff[test-row stratum] + offset[species].
    # Identity on single-stratum discovery (stratum_eff is one constant that cancels the purge).
    # RAW train uses `species_idx` for the true species; the TEST CSV records the same true index as
    # `species_idx_original` (same integer space) -> species keys align on application below.
    resid = tr_full["target_log10"].to_numpy(np.float64) - oof
    strat_tr = _stratum_key(tr_full)
    sf = pd.DataFrame({"stratum": strat_tr, "species": tr_full["species_idx"].to_numpy(), "resid": resid})
    stratum_eff = sf.groupby("stratum")["resid"].mean()
    sf["purged"] = sf["resid"].to_numpy(np.float64) - sf["stratum"].map(stratum_eff).to_numpy(np.float64)
    off_map = {int(k): float(v) for k, v in sf.groupby("species")["purged"].mean().to_dict().items()}
    strat_map = {str(k): float(v) for k, v in stratum_eff.to_dict().items()}
    # tier-input non-degeneracy (t1'): the per-species offset must cover >=2 species AND vary.
    _n_off, _n_distinct = len(off_map), len({round(v, 10) for v in off_map.values()})
    if _n_off < 2 or _n_distinct < 2:
        raise TierInputDegenerate(
            f"tier1' offset degenerate {split}/s{seed}: n_species_offset={_n_off} distinct_values={_n_distinct}")
    # shuffled control: permute the species->offset assignment (each species receives another
    # species' offset; seeded, non-identity). Mirrors the shuffled species control at the
    # offset-lookup stage; base + stratum term unchanged, only the species scalar is misassigned.
    if control == "shuffled":
        rng = np.random.RandomState(seed + 4242)
        keys = np.array(sorted(off_map.keys()))
        vals = np.array([off_map[k] for k in keys], dtype=np.float64)
        perm = rng.permutation(len(keys))
        if len(keys) > 1 and np.array_equal(perm, np.arange(len(keys))):
            perm = np.roll(perm, 1)
        off_map_applied = {int(keys[i]): float(vals[perm[i]]) for i in range(len(keys))}
    else:
        off_map_applied = off_map

    # Tier 1' test = no_species base + stratum residual term (per test stratum) + species offset
    base_csv = PRED / f"{backbone}_{BASE_VARIANT}_{split}_s{seed}_e100_nfull.csv"
    df = pd.read_csv(base_csv)
    test_stratum_eff = pd.Series(_stratum_key(df)).map(strat_map).fillna(0.0).to_numpy(np.float64)
    test_offsets = df["species_idx_original"].map(off_map_applied).fillna(0.0).to_numpy(np.float64)
    n_unseen = int((~df["species_idx_original"].isin(off_map_applied)).sum())
    df["pred_log10"] = df["pred_log10"].to_numpy(np.float64) + test_stratum_eff + test_offsets
    df["error_log10"] = df["pred_log10"] - df["true_log10"]
    df["model_name"] = f"{backbone}_{ovar}"
    df["variant"] = ovar
    df.to_csv(out, index=False)
    rmse_base = float(np.sqrt(((pd.read_csv(base_csv)["pred_log10"] - df["true_log10"]) ** 2).mean()))
    rmse_t1p = float(np.sqrt(((df["pred_log10"] - df["true_log10"]) ** 2).mean()))
    return {"skipped": False, "out": str(out), "sec": time.time() - t0, "n_train": len(tr_full),
            "n_species_offset": len(off_map), "n_strata": len(strat_map), "n_test_unseen": n_unseen,
            "rmse_no_species": rmse_base, "rmse_tier1prime": rmse_t1p}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="discovery_group,replication_group")
    ap.add_argument("--backbones", default="dmpnn,graphconv")
    ap.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    ap.add_argument("--controls", default="true,shuffled")
    a = ap.parse_args()
    splits = a.splits.split(","); bbs = a.backbones.split(","); seeds = [int(x) for x in a.seeds.split(",")]
    controls = a.controls.split(",")
    done = skip = 0; t_all = time.time()
    for split in splits:
        for bb in bbs:
            for sd in seeds:
              for ctrl in controls:
                tag = f"{bb} {split} s{sd} {ctrl}"
                try:
                    r = run_one(bb, split, sd, ctrl)
                    if r["skipped"]:
                        skip += 1; print(f"[SKIP] {tag}", flush=True)
                    else:
                        done += 1
                        print(f"[DONE] {tag}  {r['sec']:.0f}s  n_tr={r['n_train']} "
                              f"off_species={r['n_species_offset']} n_strata={r['n_strata']} test_unseen={r['n_test_unseen']} "
                              f"RMSE no_sp={r['rmse_no_species']:.4f} -> T1'={r['rmse_tier1prime']:.4f}", flush=True)
                except TierInputDegenerate as deg:
                    print(f"\n### HALT — tier1' input degenerate ###\n{deg}", flush=True); sys.exit(2)
                except Exception as e:
                    print(f"[FAIL] {tag}: {type(e).__name__}: {e}", flush=True)
    print(f"\n=== OOF Tier1' summary: done={done} skipped={skip} total={ (time.time()-t_all)/60:.1f}m ===", flush=True)
