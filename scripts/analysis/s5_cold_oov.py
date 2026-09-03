"""species-cold ungated TOST (t3a/t3b/t4 vs t2, oov=mean) B1+Phase1; §5-3 Phase1 warm OOV per split +
OOV-removed re-aggregated dd (b1_group, reference only). Facts only."""
import json, sys
import numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, r".")
sys.path.insert(0, r".\scripts")
import run_q2_gatekeeping_b1 as GKB
import run_q2_gatekeeping as GKP
from src.prediction_io import load_prediction_csv
from src.gatekeeping import paired_dd_bootstrap, decide
R=Path(r".\results\q2_v4")
KEY=["smiles","species","endpoint","duration"]
GVAR=GKB.GVAR; TOST=["t3a","t3b","t4"]

def load_cold_arm(pred, bb, tier, split, mode):
    var=GVAR[tier]; frames=[]
    for s in range(10):
        f=pred/f"{bb}_{var}_{split}_s{s}_e100_oov-{mode}.csv"
        if not f.exists(): return None
        frames.append(load_prediction_csv(f, columns=KEY+["pred_log10","true_log10"]).set_index(KEY))
    ref=frames[0]; order=ref.index
    return {"key":ref.reset_index()[KEY+["true_log10"]],"P":np.stack([f.loc[order,"pred_log10"].to_numpy(float) for f in frames],axis=1)}

def cold_tost(pred, split, delta):
    out={}
    for bb in ["dmpnn","graphconv"]:
        # base t0 = no_species (oov untrained; cancels in dd). cand/ref use oov=mean.
        base=load_cold_arm(pred,bb,"t0",split,"untrained")
        t2=load_cold_arm(pred,bb,"t2",split,"mean")
        for t in TOST:
            cand=load_cold_arm(pred,bb,t,split,"mean")
            arms=[cand,base,t2,base]
            r=GKB.run_comparison(arms, delta)  # module-agnostic (pure fn)
            k=f"{bb}/{t}~t2"
            out[k]={"dd":round(r["dd"],4),"ci":[round(r["ci_lo"],4),round(r["ci_hi"],4)],"cat":r["category"],"n_rows":r["n_rows"]} if r.get("status")=="ok" else {"status":r.get("status")}
    return out

dB=GKB.load_frozen("delta")["delta"]; dP=GKP.load_frozen("delta")["delta"]
cold={"note":"ungated (Stage-1 passed 0, so this is descriptive, NOT a gate result); oov=mean; base t0 cancels in dd",
      "b1_species_cold":cold_tost(R/"runs_b1"/"gnn"/"predictions","b1_species_cold",dB),
      "phase1_discovery_species_cold":cold_tost(R/"runs"/"gnn"/"predictions","discovery_species_cold",dP)}

# §5-3 Phase1 warm OOV per split
def warm_oov(gk, splits):
    o={}
    for sp in splits:
        tr=pd.read_csv(gk.DATA/f"{sp}_train.csv",usecols=["species_idx"]); te=pd.read_csv(gk.DATA/f"{sp}_test.csv",usecols=["species_idx"])
        s_tr=set(tr["species_idx"].astype(int)); s_te=set(te["species_idx"].astype(int)); oov=s_te-s_tr
        o[sp.replace('discovery_','')]={"n_test_OOV_species":len(oov),"n_test_OOV_rows":int(te["species_idx"].astype(int).isin(list(oov)).sum()),"n_test_rows":len(te)}
    return o
p1_oov=warm_oov(GKP,["discovery_group","discovery_scaffold","discovery_scaffold_generic","discovery_designed_leaky"])

# §5-3 OOV-removed re-aggregated dd (b1_group, non-OOV species only) — reference only
tr=pd.read_csv(GKB.DATA/"b1_group_train.csv",usecols=["species"]); trsp=set(tr["species"].astype(str))
oov_removed={}
for bb in ["dmpnn","graphconv"]:
    arms_t2=[GKB.load_arm(bb,GVAR["t2"],"b1_group"),GKB.load_arm(bb,GVAR["t0"],"b1_group")]
    for t in TOST:
        arms=[GKB.load_arm(bb,GVAR[t],"b1_group"),GKB.load_arm(bb,GVAR["t0"],"b1_group"),GKB.load_arm(bb,GVAR["t2"],"b1_group"),GKB.load_arm(bb,GVAR["t0"],"b1_group")]
        r=GKB.run_comparison(arms, dB, species_set=trsp)  # include only train (non-OOV) species
        oov_removed[f"{bb}/{t}~t2"]={"dd":round(r["dd"],4),"ci":[round(r["ci_lo"],4),round(r["ci_hi"],4)],"cat":r["category"],"n_rows":r["n_rows"]} if r.get("status")=="ok" else {"status":r.get("status")}

res=json.loads((R/"audit/expansion_results.json").read_text(encoding="utf-8"))
res["s4_results"]["species_cold_ungated_TOST"]=cold
res["s5_species_oov"]["phase1_warm_oov"]=p1_oov
res["s5_species_oov"]["b1_group_OOV_removed_reagg_dd"]={"note":"non-OOV(train) species only; reference, NOT used for judgment","dd":oov_removed}
(R/"audit/expansion_results.json").write_text(json.dumps(res,ensure_ascii=False,indent=2),encoding="utf-8")
print("cold B1:", {k:v.get("dd") for k,v in cold["b1_species_cold"].items()})
print("cold P1:", {k:v.get("dd") for k,v in cold["phase1_discovery_species_cold"].items()})
print("Phase1 warm OOV:", p1_oov)
print("B1 group OOV-removed dd:", {k:v.get("dd") for k,v in oov_removed.items()})
print("[stored cold, §5-3]")
