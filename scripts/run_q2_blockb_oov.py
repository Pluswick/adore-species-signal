"""Block B (species-cold) — eval-time OOV mapping (D-OOV). Train once, predict 3 ways.

oov=mean (primary): cold species / OOV rank categories -> mean of TRAIN-split species / categories.
oov=untrained: current (untrained own row). oov=collapse: species/tax contribution -> 0.
Training is UNCHANGED (warm/block A untouched; gate "same comparison" preserved).
Range = {t0, t2, t3a, t3b, t4} (tier 1/1' N/A). Env: conda run -n jcim_v3.
"""
from __future__ import annotations
import sys, json, time, argparse, copy
from pathlib import Path
import numpy as np, pandas as pd, torch

sys.path.insert(0, r".")
from jcim_v3.runner import (V3RunConfig, _setup_reproducible, _carve_val, _train, _predict,
    _species_lookup, model_spec_from_variant, build_v3_model, GraphDataset, StandardScaler,
    apply_species_control, fit_stratum_effect, stratum_remove, stratum_restore, ATOM_FDIM)
from jcim_v3.rdkit_lgbm import TAX_RANKS
from jcim_v3.tier_input_guard import check_blockb_oov_input, TierInputDegenerate

DATA = Path(r".\results\q2_v4\data")
PRED = Path(r".\results\q2_v4\runs\gnn\predictions")


def _species_emb_module(model, backbone, injection):
    if injection != "late_fusion":
        return None
    return model.species_emb if backbone == "graphconv" else model.context.species.emb


def _tax_parts(model, backbone):
    if backbone == "graphconv":
        return model.tax_rank_embs, model.tax_codes
    return model.context.tax_rank_embs, model.context.tax_codes


def _ffn_first_linear(model, backbone):
    ffn = model.ffn
    return ffn[0]  # nn.Linear(readout_dim, hidden)


def apply_oov(model, spec, mode, train_sp, hidden, n_species):
    """In-place OOV substitution for cold species / OOV categories.
    Returns (restore(), n_changed) — n_changed lets the post-OOV guard confirm the swap fired."""
    if mode == "untrained":
        return lambda: None, 0
    inj = spec.injection
    saved = []
    train_set = set(int(s) for s in train_sp)

    if inj == "late_fusion":                     # tier 4: species embedding
        emb = _species_emb_module(model, spec.backbone, inj)
        W = emb.weight.data
        cold = [s for s in range(n_species) if s not in train_set]
        mean_vec = W[list(train_set)].mean(0)
        saved.append((W, {c: W[c].clone() for c in cold}))
        for c in cold:
            W[c] = torch.zeros_like(mean_vec) if mode == "collapse" else mean_vec

    elif inj in ("taxonomy_original", "taxonomy_ncbi"):   # tier 3a/3b: per-rank OOV categories
        embs, codes = _tax_parts(model, spec.backbone)
        codes = codes.cpu().numpy()
        for r in range(len(embs)):
            train_cats = set(int(codes[s, r]) for s in train_set)
            W = embs[r].weight.data
            allc = range(W.shape[0]); oov = [c for c in allc if c not in train_cats]
            mean_vec = W[list(train_cats)].mean(0)
            saved.append((W, {c: W[c].clone() for c in oov}))
            for c in oov:
                W[c] = torch.zeros_like(mean_vec) if mode == "collapse" else mean_vec

    elif inj == "categorical":                   # tier 2: one-hot -> FFN column swap
        lin = _ffn_first_linear(model, spec.backbone)
        W = lin.weight.data                       # [hidden, readout_dim], one-hot cols at [hidden: hidden+n_species]
        off = hidden
        cold = [s for s in range(n_species) if s not in train_set]
        train_cols = W[:, off + np.array(sorted(train_set))].mean(1)
        saved.append((W, {("c", s): W[:, off + s].clone() for s in cold}))
        for s in cold:
            W[:, off + s] = 0.0 if mode == "collapse" else train_cols

    n_changed = sum(len(d) for _, d in saved)

    def restore():
        for W, d in saved:
            for key, v in d.items():
                if isinstance(key, tuple):     # ffn column
                    W[:, off + key[1]] = v
                else:
                    W[key] = v
    return restore, n_changed


def run_one(backbone, variant, split, seed, control, epochs=100):
    spec = model_spec_from_variant(backbone, variant)
    modes = ["untrained"] if not spec.uses_species else ["mean", "untrained", "collapse"]
    outs = {m: PRED / f"{backbone}_{variant}_{split}_s{seed}_e{epochs}_oov-{m}.csv" for m in modes}
    if all(p.exists() for p in outs.values()):
        return {"skipped": True}
    cfg = V3RunConfig(backbone=backbone, variant=variant, split=split, seed=seed, epochs=epochs,
                      batch_size=256, lr=5e-4, weight_decay=1e-5, hidden=300, depth=3, dropout=0.1,
                      species_emb_dim=16, val_frac=0.1, data_dir=str(DATA))
    device = _setup_reproducible(seed)
    tr_full = pd.read_csv(DATA / f"{split}_train.csv"); te = pd.read_csv(DATA / f"{split}_test.csv")
    full = pd.concat([tr_full, te], ignore_index=True)
    n_species = int(full["species_idx"].max()) + 1
    slook = _species_lookup(full)
    train_sp = tr_full["species_idx"].astype(int).unique()

    tr_raw, va_raw = _carve_val(tr_full, seed, cfg.val_frac)
    cm = spec.species_control
    tr = apply_species_control(tr_raw, mode=cm, seed=seed+101, n_species=n_species, species_lookup=slook)
    va = apply_species_control(va_raw, mode=cm, seed=seed+202, n_species=n_species, species_lookup=slook)
    ho = apply_species_control(te, mode=cm, seed=seed+303, n_species=n_species, species_lookup=slook)
    eff = fit_stratum_effect(tr.frame, tr.frame["target_log10"].to_numpy(np.float64))
    tr.frame["target_log10"] = stratum_remove(tr.frame, tr.frame["target_log10"].to_numpy(np.float64), eff)
    va.frame["target_log10"] = stratum_remove(va.frame, va.frame["target_log10"].to_numpy(np.float64), eff)
    base = GraphDataset(tr.frame); ts = StandardScaler().fit(base.target.reshape(-1,1))
    ds = None if base.raw_desc is None else StandardScaler().fit(base.raw_desc)
    tr_ds = GraphDataset(tr.frame, target_scaler=ts, desc_scaler=ds)
    va_ds = GraphDataset(va.frame, target_scaler=ts, desc_scaler=ds)
    ho_ds = GraphDataset(ho.frame, target_scaler=ts, desc_scaler=ds)
    # tax codes for taxonomy tiers
    tax_codes = tax_cards = None
    if spec.injection in ("taxonomy_original", "taxonomy_ncbi"):
        ranks = TAX_RANKS[spec.injection]
        sp = full.drop_duplicates("species_idx").set_index("species_idx")
        maps = []
        for rk in ranks:
            vals = sp[rk].astype("string").fillna("__unknown__"); vals = vals.mask(vals.str.strip()=="", "__unknown__")
            cats = pd.Categorical(vals); maps.append(({int(k): int(v) for k,v in zip(sp.index.astype(int), cats.codes)}, len(cats.categories)))
        tax_cards = [c for _, c in maps]
        tax_codes = np.zeros((n_species, len(ranks)), dtype=np.int64)
        for j,(mp,_) in enumerate(maps):
            for s in range(n_species): tax_codes[s, j] = mp.get(s, 0)
    model = build_v3_model(spec=spec, atom_fdim=ATOM_FDIM, n_species=n_species, hidden=cfg.hidden,
                           depth=cfg.depth, dropout=cfg.dropout, species_emb_dim=cfg.species_emb_dim,
                           desc_fdim=tr_ds.desc_fdim, tax_codes=tax_codes, tax_cards=tax_cards)
    model, _ = _train(model, tr_ds, va_ds, cfg=cfg, device=device)

    yte = te["target_log10"].to_numpy(np.float64); res = {}
    cold_sp = set(te["species_idx"].astype(int).unique()) - set(int(s) for s in train_sp)
    guard_log = str(PRED / "blockb_oov_guard.jsonl")
    for m in modes:
        restore, n_changed = apply_oov(model, spec, m, train_sp, cfg.hidden, n_species)
        # post-OOV non-degeneracy: cold species identified + (for mean/collapse) swap fired
        g = check_blockb_oov_input(variant, split, train_sp, cold_sp, m, swapped_ok=(n_changed > 0))
        Path(guard_log).parent.mkdir(parents=True, exist_ok=True)
        with open(guard_log, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(g, ensure_ascii=False) + "\n")
        if g["degenerate"]:
            restore()
            raise TierInputDegenerate(f"block B post-OOV degenerate {variant}/{split}/{m}: {g['reasons']}")
        pred = _predict(model, ho_ds, cfg.batch_size, device)
        restore()
        pred = stratum_restore(ho.frame, np.asarray(pred, np.float64), eff)
        rmse = float(np.sqrt(((pred - yte)**2).mean()))
        pf = te.copy(); pf["pred_log10"]=pred; pf["true_log10"]=yte; pf["error_log10"]=pred-yte
        pf["backbone"]=backbone; pf["variant"]=variant; pf["oov_mode"]=m
        pf["species_idx_original"]=te["species_idx"].astype(int)
        pf.to_csv(outs[m], index=False, encoding="utf-8"); res[m]=round(rmse,4)
    return {"skipped": False, "rmse_by_oov": res}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="discovery_species_cold,replication_species_cold")
    ap.add_argument("--backbones", default="dmpnn,graphconv")
    ap.add_argument("--variants", default="no_species,true_species_categorical,true_species_taxonomy_original,true_species_taxonomy_ncbi,true_species_late_fusion")
    ap.add_argument("--controls", default="true,shuffled")
    ap.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    ap.add_argument("--epochs", type=int, default=100)
    a = ap.parse_args()
    for split in a.splits.split(","):
        for bb in a.backbones.split(","):
            for va in a.variants.split(","):
                for sd in [int(x) for x in a.seeds.split(",")]:
                    try:
                        r = run_one(bb, va, split, sd, "true", a.epochs)
                        print(json.dumps({"cell": f"{bb}/{va}/{split}/s{sd}", **r}), flush=True)
                    except Exception as e:
                        print(json.dumps({"cell": f"{bb}/{va}/{split}/s{sd}", "FAIL": f"{type(e).__name__}: {e}"}), flush=True)
