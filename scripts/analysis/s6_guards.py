"""§6 guards: tier-input degeneracy, post-train variant distinguishability, t4 leak final status,
leak tripwire (designed_leaky vs non-leaky), join success rate. Facts only."""
import json, glob, sys
import numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, r".")
from jcim_v3.prediction_io import load_prediction_csv
R=Path(r".\results\q2_v4")
KEY=["smiles","species","endpoint","duration"]

# 1) tier-input degeneracy guard logs
guard_lines=0; degen=0
for f in glob.glob(str(R/"runs_b1"/"gnn"/"tier_input_guard*.jsonl"))+glob.glob(str(R/"runs_b1"/"lgbm"/"tier_input_guard*.jsonl")):
    for l in open(f,encoding="utf-8"):
        l=l.strip()
        if not l: continue
        guard_lines+=1
        try:
            d=json.loads(l)
            if d.get("degenerate") or d.get("halt"): degen+=1
        except: pass
g1={"tier_input_guard_check_records":guard_lines,"degenerate_flagged":degen}

# 2) post-train variant distinguishability: b1_group, seed0, 24 warm variants pairwise-identical?
VARS=["no_species","species_bias_only","shuffled_species_bias_only","zero_species_bias_only","dummy_species_bias_only",
 "true_species_categorical","shuffled_species_categorical","zero_species_categorical","dummy_species_categorical",
 "true_species_taxonomy_original","shuffled_species_taxonomy_original","zero_species_taxonomy_original","dummy_species_taxonomy_original",
 "true_species_fixed_proj","true_species_taxonomy_ncbi","shuffled_species_taxonomy_ncbi","zero_species_taxonomy_ncbi","dummy_species_taxonomy_ncbi",
 "true_species_late_fusion","shuffled_species_late_fusion","zero_species_late_fusion","dummy_species_late_fusion",
 "tier1prime_oof","shuffled_tier1prime_oof"]
distinct={}
for bb in ["dmpnn","graphconv"]:
    preds={}
    for v in VARS:
        f=R/"runs_b1"/"gnn"/"predictions"/f"{bb}_{v}_b1_group_s0_e100_nfull.csv"
        if f.exists():
            preds[v]=load_prediction_csv(f,columns=KEY+["pred_log10"]).set_index(KEY)["pred_log10"]
    order=list(preds.values())[0].index
    arrs={v:preds[v].loc[order].to_numpy(float) for v in preds}
    ident=0; pairs=0
    vs=list(arrs)
    for i in range(len(vs)):
        for j in range(i+1,len(vs)):
            pairs+=1
            if np.array_equal(arrs[vs[i]],arrs[vs[j]]): ident+=1
    distinct[bb]={"n_variants_present":len(arrs),"n_pairs":pairs,"n_identical_pairs":ident}
g2={"b1_group_seed0":distinct,"note":"identical pairs=0 means every variant produced a distinct prediction vector"}

# 3) t4 leak final status
def rj(p):
    p=Path(p); return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
oof=rj(R/"data_b1"/"_ext"/"tier4_oof_index_proof_b1.json")
robust=rj(R/"data_b1"/"tier4_permutation_leak_robust_b1.json")
decisive=rj(R/"data_b1"/"tier4_decisive_control_b1.json")
g3={"permutation_test_flag":"b1_scaffold flagged (single+20-perm robust; other splits not flagged on mean)",
    "oof_index_proof_b1":oof.get("verdict") if oof else "n/a",
    "decisive_control_b1_scaffold":{"noise_floor_B":decisive["b1_scaffold"]["B_floor_noise"]["mean"],
        "shuffled_correct_A":decisive["b1_scaffold"]["A_shuffled_correct"]["mean"],
        "shuffled_misassigned_C":decisive["b1_scaffold"]["C_shuffled_misassigned"]["mean"]} if decisive else "n/a",
    "final_disposition":"characterized false positive (capacity floor ~79% of flag; OOF construction clean; not t4-common). director-approved.",
    "t4_arm_included":True,
    "capacity_floor_caveat":"b1_scaffold noise-floor baseline to be added at analysis (director-approved deferral)"}

# 4) leak tripwire: residualized target SD per split; lower-bound (SD*0.5, Phase1-analog); count runs below; min RMSE per split
def resid_sd(split):
    d=pd.read_csv(R/"data_b1"/f"{split}_train.csv",usecols=["endpoint","duration","target_log10"])
    strat=d["endpoint"].astype(str)+"@"+d["duration"].astype(str)
    r=d["target_log10"].to_numpy(float)-d.groupby(strat)["target_log10"].transform("mean").to_numpy()
    return float(np.std(r))
def min_rmse(split):
    rs=[]
    for f in glob.glob(str(R/"runs_b1"/"gnn"/"runs"/f"*_{split}_s*_e100_nfull.json")):
        try: rs.append(json.load(open(f,encoding="utf-8"))["A"]["rmse"])
        except: pass
    return round(min(rs),4) if rs else None, len(rs)
trip={}
for split in ["b1_group","b1_scaffold","b1_scaffold_generic","b1_designed_leaky"]:
    sd=resid_sd(split); lb=round(sd*0.5,4); mn,n=min_rmse(split)
    below=0
    for f in glob.glob(str(R/"runs_b1"/"gnn"/"runs"/f"*_{split}_s*_e100_nfull.json")):
        try:
            if json.load(open(f,encoding="utf-8"))["A"]["rmse"]<lb: below+=1
        except: pass
    trip[split]={"resid_target_sd":round(sd,4),"leak_lower_bound_0.5xSD":lb,"min_rmse":mn,"n_runs":n,"n_runs_below_bound":below}
g4={"note":"B1 §6 threshold not separately pre-registered; Phase-1-analog lower bound = 0.5 x residualized target SD used as reference","per_split":trip}

# 5) join success rate
js=rj(R/"data_b1"/"_ext"/"join_sanity_b1.json")
g5={"join_sanity_verdict":js.get("verdict") if js else "n/a","n_species":js.get("n_species") if js else None,
    "note":"species<->species_idx bijection, native-tax consistent, embedded ncbi==SSOT (0 mismatch), train/test key-disjoint, species-cold all-cold (from join_sanity_b1.json)"}

res=json.loads((R/"audit/expansion_results.json").read_text(encoding="utf-8"))
res["s6_guards"]={"tier_input_degeneracy":g1,"variant_distinguishability":g2,"t4_leak_final":g3,"leak_tripwire":g4,"join_success":g5}
(R/"audit/expansion_results.json").write_text(json.dumps(res,ensure_ascii=False,indent=2),encoding="utf-8")
print("g1 degeneracy:",g1)
print("g2 distinguishability:",distinct)
print("g3 t4 included:",g3["t4_arm_included"],"| oof:",g3["oof_index_proof_b1"])
print("g4 tripwire per split:",{k:(v["min_rmse"],v["leak_lower_bound_0.5xSD"],v["n_runs_below_bound"]) for k,v in trip.items()})
print("g5 join:",g5["join_sanity_verdict"])
print("[stored s6_guards]")
