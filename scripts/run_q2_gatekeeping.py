"""ADORE Q2 comparison / TOST / hierarchical gatekeeping / FDR pipeline (§4C·§4C-Explore·§4Δ·§4G·§4G-7).

⚠ EXECUTION = UNBLINDING. Does NOT run without --execute (director-approved). Default REFUSES.
--dry-check enumerates the FULL comparison set + verifies frozen files/cells resolve, WITHOUT any Δ.

[C#] = director 16-item checklist. δ/δ′/δ_det READ from frozen files, never recomputed [C14].
Env: conda run -n src.
"""
from __future__ import annotations
import sys, json, argparse, hashlib
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, r".")
from src.prediction_io import load_prediction_csv                          # [C12] SSOT whitelist loader
from src.gatekeeping import (paired_dd_bootstrap, ensemble_dd_bootstrap, decide,
                                 is_fourth_cell, bh_fdr, stage2_reached)

R = Path(r".\results\q2_v4")
GNN = R / "runs" / "gnn" / "predictions"
LGB = R / "runs" / "replication" / "lgbm" / "predictions"
LGB_NAIVE = R / "runs" / "replication" / "naive" / "predictions"
LGB_REP = R / "runs" / "replication" / "predictions"
DATA = R / "data"
MORT = Path(r"<ECOTOX_DATA_DIR>\processed\ecotox_mortality_processed.csv")
FROZEN = {"delta": R / "audit" / "delta_primary_frozen.json",                   # per-seed δ [C14]
          "delta_prime": R / "audit" / "delta_prime_frozen.json",              # ensemble δ′ [C16]
          "delta_det": R / "audit" / "delta_det_frozen.json"}                   # det δ_det [C15]
KEY = ["smiles", "species", "endpoint", "duration"]                            # [C13] strata key
N_BOOT = 2000                                                                   # [C3]
GNN_BB = ["dmpnn", "graphconv"]
SEEDS = list(range(10))

# tier -> GNN variant name
GVAR = {"t0": "no_species", "t1": "species_bias_only", "t1p": "tier1prime_oof", "t2": "true_species_categorical",
        "t3a": "true_species_taxonomy_original", "t3b": "true_species_taxonomy_ncbi", "t4": "true_species_late_fusion",
        "t3a_g": "true_species_taxonomy_genus", "t3a_gf": "true_species_taxonomy_genusfamily"}
GSHUF = {"t1": "shuffled_species_bias_only", "t1p": "shuffled_tier1prime_oof", "t2": "shuffled_species_categorical",
         "t3a": "shuffled_species_taxonomy_original", "t3b": "shuffled_species_taxonomy_ncbi",
         "t4": "shuffled_species_late_fusion"}
# tier -> (dir, LightGBM stem)
LSTEM = {"t0": (LGB, "LightGBM_RDKit_no_species"), "t1p": (LGB_NAIVE, "LightGBM_RDKit_species_residual_calibration"),
         "t2": (LGB, "LightGBM_RDKit_species_categorical"), "t3a": (LGB, "LightGBM_RDKit_taxonomy_original"),
         "t3b": (LGB, "LightGBM_RDKit_taxonomy_ncbi"), "t4": (LGB_REP, "LightGBM_RDKit_species_svd_factor"),
         "t3a_g": (LGB, "LightGBM_RDKit_taxonomy_genus"), "t3a_gf": (LGB, "LightGBM_RDKit_taxonomy_genusfamily")}
SUP_TIERS = ["t1", "t1p", "t2", "t3a", "t3b", "t4"]     # superiority: species tiers
TOST_TIERS = ["t3a", "t3b", "t4"]                       # equivalence TOST vs t2 (§4C)


def load_frozen(key):                                                           # [C14] fail if absent
    p = FROZEN[key]
    if not p.exists():
        raise FileNotFoundError(f"frozen {key} absent: {p} — pipeline must NOT recompute. Freeze first.")
    return json.loads(p.read_text(encoding="utf-8"))


def file_sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16] if Path(p).exists() else None


# ---- species -> {train count bin, tax_group}  (derived from DATASET / mortality source; SSOT)
def _count_bins(split):
    tr = pd.read_csv(DATA / f"{split}_train.csv", usecols=["species"])
    cnt = tr.groupby("species").size()
    def b(c): return "1-5" if c <= 5 else "6-20" if c <= 20 else "21-100" if c <= 100 else "100+"
    return {sp: b(c) for sp, c in cnt.items()}


def _tax_group():
    m = pd.read_csv(MORT, usecols=["tax_gs", "tax_group"], low_memory=False).drop_duplicates("tax_gs")
    m["species"] = m["tax_gs"].astype(str).str.strip().str.lower()
    return dict(zip(m["species"], m["tax_group"]))


# ---- unified arm loader (backbone in {dmpnn,graphconv}=GNN 10-seed; 'lightgbm'=1-seed) ----
def load_arm(backbone, name, split):
    """name = GNN variant string (GNN) or tier key (lightgbm). Returns {key, P[rows, nseed]} or None."""
    if backbone == "lightgbm":
        d, stem = LSTEM[name]
        f = d / f"{stem}_{split}_s0.csv"
        if not f.exists():
            return None
        x = load_prediction_csv(f, columns=KEY + ["pred_log10", "true_log10"])
        return {"key": x[KEY + ["true_log10"]].reset_index(drop=True), "P": x["pred_log10"].to_numpy(np.float64)[:, None]}
    ref = None; cols = []
    for s in SEEDS:
        f = GNN / f"{backbone}_{name}_{split}_s{s}_e100_nfull.csv"
        if not f.exists():
            return None
        dd = load_prediction_csv(f, columns=KEY + ["pred_log10", "true_log10"]).set_index(KEY)
        if ref is None:
            ref = dd[["true_log10"]].copy(); order = ref.index
        cols.append(dd.loc[order, "pred_log10"].to_numpy(np.float64))
    return {"key": ref.reset_index(), "P": np.stack(cols, axis=1)}


def align(arms):
    base = arms[0]["key"][KEY + ["true_log10"]].copy(); base["_r0"] = np.arange(len(base))
    merged = base
    for i, a in enumerate(arms[1:], 1):
        k = a["key"][KEY].copy(); k[f"_r{i}"] = np.arange(len(k))
        merged = merged.merge(k, on=KEY, how="inner")
    true = merged["true_log10"].to_numpy(np.float64)
    blk = merged["smiles"].to_numpy()                                          # [§4δ_det] block = smiles
    Ps = [arms[i]["P"][merged[f"_r{i}"].to_numpy(np.int64)] for i in range(4)]
    # cross-backbone: a deterministic 1-seed arm (LightGBM) mixed with 10-seed GNN arms. Its RMSE is a
    # constant on the seed axis (0 seed variance, block variance kept). Tile 1->nmax so the common sset
    # is well-defined; identical columns => the det arm contributes its fixed value under any sset.
    nmax = max(P.shape[1] for P in Ps)
    if nmax > 1:
        Ps = [np.repeat(P, nmax, axis=1) if P.shape[1] == 1 else P for P in Ps]
    return true, blk, Ps, merged[KEY]


def run_comparison(arms, delta, species_set=None, ensemble=False):
    if any(a is None for a in arms):
        return {"status": "missing"}
    true, blk, Ps, kf = align(arms)
    if species_set is not None:
        mask = kf["species"].isin(species_set).to_numpy()
        true = true[mask]; blk = blk[mask]; Ps = [P[mask] for P in Ps]; kf = kf[mask]
    has_nan = bool(np.isnan(true).any() or any(np.isnan(P).any() for P in Ps))
    n_blocks = int(pd.unique(blk).size) if len(blk) else 0
    n_strata = int((kf["endpoint"].astype(str) + "@" + kf["duration"].astype(str)).nunique()) if len(kf) else 0
    if len(true) == 0 or n_blocks < 2 or has_nan:
        return {"status": "empty" if not has_nan else "nan", "n_rows": int(len(true)),
                "n_strata": n_strata, "n_blocks": n_blocks, "has_nan": has_nan}
    fn = ensemble_dd_bootstrap if ensemble else paired_dd_bootstrap
    bs = fn(true, blk, Ps, n_boot=N_BOOT)                                       # [C3][C4][C5][C6]
    return {"status": "ok", **bs, "category": decide(bs["ci_lo"], bs["ci_hi"], delta),   # [C7]
            "fourth_cell": is_fourth_cell(bs["ci_lo"], bs["ci_hi"], delta),
            "n_rows": int(len(true)), "n_strata": n_strata, "n_blocks": n_blocks, "has_nan": has_nan}


# ---------------------------------------------------------------- comparison-set builders
def _primary_like(family, split, gated_stage2=None):
    """superiority (species vs t0 & vs shuffled) + TOST (t3a/t3b/t4 vs t2), GNN within-backbone. [§4C]"""
    out = []
    for bb in GNN_BB:
        for t in SUP_TIERS:                                                     # superiority (2-arm; base cancels)
            out.append(dict(family=family, backbone=bb, test="superiority", split=split, delta="delta",
                            cand=(bb, GVAR[t]), candbase=(bb, GVAR["t0"]), ref=(bb, GVAR["t0"]), refbase=(bb, GVAR["t0"]),
                            label=f"{bb}/{t}>t0"))
            if t in GSHUF:
                out.append(dict(family=family, backbone=bb, test="superiority", split=split, delta="delta",
                                cand=(bb, GVAR[t]), candbase=(bb, GSHUF[t]), ref=(bb, GSHUF[t]), refbase=(bb, GSHUF[t]),
                                label=f"{bb}/{t}>shuf"))
        for t in TOST_TIERS:                                                    # equivalence TOST vs t2
            out.append(dict(family=family, backbone=bb, test="TOST", split=split, delta="delta",
                            cand=(bb, GVAR[t]), candbase=(bb, GVAR["t0"]), ref=(bb, GVAR["t2"]), refbase=(bb, GVAR["t0"]),
                            label=f"{bb}/{t}~t2", gated_stage2=gated_stage2))
    return out


def comparison_set():
    comps = _primary_like("primary", "discovery_group", gated_stage2="discovery_species_cold")
    comps += _primary_like("confirmatory", "replication_group", gated_stage2="replication_species_cold")
    # [C15] deterministic (LightGBM within-'backbone') TOST: t3a/t3b/t4 vs t2 ; decide with δ_det
    for t in TOST_TIERS:
        comps.append(dict(family="deterministic", backbone="lightgbm", test="TOST", split="discovery_group",
                          delta="delta_det", cand=("lightgbm", t), candbase=("lightgbm", "t0"),
                          ref=("lightgbm", "t2"), refbase=("lightgbm", "t0"), label=f"lgbm/{t}~t2"))
    # ---- exploratory (§4C-Explore) — own BH-FDR, NOT gated ----
    E = "exploratory"
    # rank truncation: genus/genusfamily vs {t3a full, t2}, GNN + LightGBM
    for bb in GNN_BB + ["lightgbm"]:
        cv = (lambda k: GVAR[k]) if bb != "lightgbm" else (lambda k: k)
        for trunc in ["t3a_g", "t3a_gf"]:
            for refk in ["t3a", "t2"]:
                comps.append(dict(family=E, subtype="rank", backbone=bb, test="TOST", split="discovery_group",
                                  delta="delta", cand=(bb, cv(trunc)), candbase=(bb, cv("t0")),
                                  ref=(bb, cv(refk)), refbase=(bb, cv("t0")), label=f"{bb}/{trunc}~{refk}"))
    # support-bin: primary TOST pairs decomposed by species train-count bin (row filter)
    for bb in GNN_BB:
        for t in TOST_TIERS:
            for b in ["1-5", "6-20", "21-100", "100+"]:
                comps.append(dict(family=E, subtype="support-bin", backbone=bb, test="TOST", split="discovery_group",
                                  delta="delta", cand=(bb, GVAR[t]), candbase=(bb, GVAR["t0"]), ref=(bb, GVAR["t2"]),
                                  refbase=(bb, GVAR["t0"]), filt=("bin", b), label=f"{bb}/{t}~t2/bin{b}"))
    # tax_group: primary TOST pairs decomposed by fish/crusta/algae (row filter)
    for bb in GNN_BB:
        for t in TOST_TIERS:
            for g in ["fish", "crusta", "algae"]:
                comps.append(dict(family=E, subtype="tax_group", backbone=bb, test="TOST", split="discovery_group",
                                  delta="delta", cand=(bb, GVAR[t]), candbase=(bb, GVAR["t0"]), ref=(bb, GVAR["t2"]),
                                  refbase=(bb, GVAR["t0"]), filt=("tax", g), label=f"{bb}/{t}~t2/{g}"))
    # scaffold + designed-leaky: primary set on other splits
    for sp in ["discovery_scaffold", "discovery_scaffold_generic", "discovery_designed_leaky"]:
        for c in _primary_like(E, sp):
            c["subtype"] = "scaffold" if "scaffold" in sp else "designed-leaky"; comps.append(c)
    # cross-backbone (a): cand={t3a,t3b,t4}@{dmpnn,graphconv} vs t2@lightgbm, each own t0 (true 4-arm DD)
    for bb in GNN_BB:
        for t in TOST_TIERS:
            comps.append(dict(family=E, subtype="cross-backbone", backbone=f"{bb}x lgbm", test="TOST",
                              split="discovery_group", delta="delta", cand=(bb, GVAR[t]), candbase=(bb, GVAR["t0"]),
                              ref=("lightgbm", "t2"), refbase=("lightgbm", "t0"), label=f"{bb}/{t} ~ lgbm/t2"))
    # [C16] ensemble sensitivity: primary+confirmatory TOST re-decided with δ′ (ensemble Δ′), own FDR, NOT gated
    for c in [c for c in comps if c["test"] == "TOST" and c["family"] in ("primary", "confirmatory")]:
        base = {k: v for k, v in c.items() if k not in ("gated_stage2", "family", "delta", "label", "ensemble")}
        comps.append(dict(base, family="sensitivity_ensemble", delta="delta_prime", ensemble=True, label=c["label"] + "/ens"))
    return comps


def hkey(c):
    return (c["backbone"], c["cand"][1], c["ref"][1], c["test"], c["family"] in ("primary",), c["split"])


def run(execute, dry_check):
    deltas = {k: load_frozen(k)[{"delta": "delta", "delta_prime": "delta_prime", "delta_det": "delta_det"}[k]]
              for k in FROZEN}
    comps = comparison_set()
    bins = _count_bins("discovery_group"); taxg = _tax_group()
    bin_species = {b: {s for s, bb in bins.items() if bb == b} for b in ["1-5", "6-20", "21-100", "100+"]}
    tax_species = {g: {s for s, gg in taxg.items() if gg == g} for g in ["fish", "crusta", "algae"]}

    # ---- enumerate FULL comparison set (for dry-check log / report 2-0) ----
    enum = [f"{c['family']}|{c.get('subtype','-')}|{c['backbone']}|{c['cand'][1]} vs {c['ref'][1]}|{c['split']}|"
            f"{c['test']}|filt={c.get('filt','-')}" for c in comps]
    if dry_check:
        (R / "runs" / "bootstrap").mkdir(parents=True, exist_ok=True)
        (R / "runs" / "bootstrap" / "comparison_set_enumeration.txt").write_text("\n".join(enum), encoding="utf-8")
        miss = 0
        for c in comps:
            arms = [load_arm(c[a][0], c[a][1], c["split"]) for a in ("cand", "candbase", "ref", "refbase")]
            if any(a is None for a in arms):
                miss += 1; print(f"  [MISSING] {c['label']} @ {c['split']}")
        print(f"[dry-check] deltas={ {k: round(v,6) for k,v in deltas.items()} } | comparisons={len(comps)} missing={miss}")
        print(f"  enumeration -> runs/bootstrap/comparison_set_enumeration.txt ({len(enum)} rows)")
        return
    if not execute:
        print("REFUSED: --execute computes Δ/TOST = UNBLINDING (director approval required). Use --dry-check.")
        sys.exit(3)

    # ===== EXECUTION (director-approved) =====
    prim = {hkey(c) for c in comps if c["family"] == "primary"}
    conf = {(b, ca, rf, t, True, sp.replace("replication", "discovery")) for (b, ca, rf, t, _p, sp) in
            {hkey(c) for c in comps if c["family"] == "confirmatory"}}
    if not conf.issubset(prim):                                                # [C10]
        print("ABORT [C10]: confirmatory ⊄ primary"); sys.exit(4)
    results = []
    for c in comps:
        arms = [load_arm(c[a][0], c[a][1], c["split"]) for a in ("cand", "candbase", "ref", "refbase")]
        ss = None
        if c.get("filt"):
            kind, val = c["filt"]; ss = bin_species[val] if kind == "bin" else tax_species[val]
        r = run_comparison(arms, deltas[c["delta"]], species_set=ss, ensemble=c.get("ensemble", False))
        rec = {k: c[k] for k in ("family", "subtype", "backbone", "test", "split", "delta", "label") if k in c}
        rec.update({"cand": c["cand"][1], "ref": c["ref"][1], **r})
        if c["family"] == "deterministic" and r.get("status") == "ok":         # [C15] GNN δ borrowed = sensitivity
            rec["category_gnn_delta_sensitivity"] = decide(r["ci_lo"], r["ci_hi"], deltas["delta"])
        results.append(rec)
    # [C9] per-family BH-FDR + global ; [C8] gate (primary/confirmatory TOST only)
    for fam in ("primary", "confirmatory", "exploratory", "deterministic", "sensitivity_ensemble"):
        ok = [r for r in results if r.get("family") == fam and r.get("status") == "ok"]
        for r, q in zip(ok, bh_fdr([r["p"] for r in ok])):
            r["q_family"] = float(q)
    allok = [r for r in results if r.get("status") == "ok"]
    for r, q in zip(allok, bh_fdr([r["p"] for r in allok])):
        r["q_global"] = float(q)
    for r in results:
        if r.get("test") == "TOST" and r.get("status") == "ok" and r.get("family") in ("primary", "confirmatory"):
            r["reaches_stage2"] = stage2_reached(r["category"])                # [C8]
    provenance = {"script_sha16": file_sha(__file__), "gatekeeping_module_sha16": file_sha(R.parents[1] / "src" / "gatekeeping.py"),
                  "frozen": {k: {"value": deltas[k], "sha16": file_sha(v)} for k, v in FROZEN.items()},
                  "n_comparisons": len(comps), "param_hash": hashlib.sha256(json.dumps(sorted(map(str, prim))).encode()).hexdigest()[:16],
                  "confirmatory_subset_of_primary": bool(conf.issubset(prim))}
    out = R / "runs" / "bootstrap" / "gatekeeping_results.json"
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"provenance": provenance, "enumeration": enum, "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    import os
    os.replace(tmp, out)                                                       # atomic write
    print(f"[EXECUTED] {len(results)} comparisons -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--dry-check", action="store_true")
    a = ap.parse_args()
    run(execute=a.execute, dry_check=a.dry_check)
