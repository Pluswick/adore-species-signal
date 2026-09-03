"""§4-4 absolute RMSE: per-seed mean+-sd and ensemble RMSE, per (tier,backbone,split), both experiments.
Computed directly from prediction CSVs (same source as frozen delta). Facts only."""
import json, numpy as np
from pathlib import Path
import sys
sys.path.insert(0, r".")
from src.prediction_io import load_prediction_csv
R=Path(r".\results\q2_v4")
KEY=["smiles","species","endpoint","duration"]
GVAR={"t0":"no_species","t1":"species_bias_only","t1p":"tier1prime_oof","t2":"true_species_categorical",
      "t3a":"true_species_taxonomy_original","t3b":"true_species_taxonomy_ncbi","t4":"true_species_late_fusion"}
TIERS=list(GVAR); BB=["dmpnn","graphconv"]
EXP={"phase1":{"pred":R/"runs"/"gnn"/"predictions","splits":["discovery_group","discovery_scaffold","discovery_scaffold_generic","discovery_designed_leaky"]},
     "b1":{"pred":R/"runs_b1"/"gnn"/"predictions","splits":["b1_group","b1_scaffold","b1_scaffold_generic","b1_designed_leaky"]}}

def cond(pred, bb, tier, split):
    var=GVAR[tier]
    frames=[]
    for s in range(10):
        f=pred/f"{bb}_{var}_{split}_s{s}_e100_nfull.csv"
        if not f.exists(): return None
        frames.append(load_prediction_csv(f, columns=KEY+["pred_log10","true_log10"]).set_index(KEY))
    order=frames[0].index; true=frames[0].loc[order,"true_log10"].to_numpy(float)
    ps=[f.loc[order,"pred_log10"].to_numpy(float) for f in frames]
    perseed=[float(np.sqrt(np.mean((p-true)**2))) for p in ps]
    ens=float(np.sqrt(np.mean((np.mean(ps,axis=0)-true)**2)))
    return {"per_seed_mean":round(float(np.mean(perseed)),4),"per_seed_sd":round(float(np.std(perseed,ddof=1)),4),
            "ensemble":round(ens,4),"n_test":int(len(true))}

out={}
for exp,cfg in EXP.items():
    out[exp]={}
    for split in cfg["splits"]:
        sk=split.replace("discovery_","").replace("b1_","")
        out[exp][sk]={}
        for bb in BB:
            for tier in TIERS:
                c=cond(cfg["pred"],bb,tier,split)
                out[exp][sk][f"{bb}/{tier}"]=c
    print(f"[{exp}] done")
# store
res=json.loads((R/"audit/expansion_results.json").read_text(encoding="utf-8"))
res.setdefault("s4_results",{})["s4_4_absolute_rmse"]=out
(R/"audit/expansion_results.json").write_text(json.dumps(res,ensure_ascii=False,indent=2),encoding="utf-8")
# print group split as sanity
for exp in ["phase1","b1"]:
    print(f"\n[{exp}] group split (per-seed mean+-sd | ensemble):")
    for bb in BB:
        row=" ".join(f"{t}={out[exp]['group'][f'{bb}/{t}']['per_seed_mean']}+-{out[exp]['group'][f'{bb}/{t}']['per_seed_sd']}/{out[exp]['group'][f'{bb}/{t}']['ensemble']}" for t in TIERS)
        print(f"  {bb}: {row}")
print("\n[stored s4_4_absolute_rmse]")
