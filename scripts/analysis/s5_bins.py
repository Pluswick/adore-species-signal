"""§5-1 support-bin (director bins 1-4/5-9/10-19/20-49/50+) dd+CI for t2->t3a/t3b/t4, per bin, both
experiments. §5-2 species counts + one-hot dim (Phase 1 re-read from source). Facts only."""
import json, sys
import numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, r".")
sys.path.insert(0, r".\scripts")
import run_q2_gatekeeping_b1 as GKB
import run_q2_gatekeeping as GKP
R=Path(r".\results\q2_v4")
BINS=["1-4","5-9","10-19","20-49","50+"]
def dbin(c): return "1-4" if c<=4 else "5-9" if c<=9 else "10-19" if c<=19 else "20-49" if c<=49 else "50+"
TOST=["t3a","t3b","t4"]

def support_bins(gk, split, delta):
    tr=pd.read_csv(gk.DATA/f"{split}_train.csv", usecols=["species"])
    cnt=tr.groupby("species").size()
    binsp={b:{s for s,c in cnt.items() if dbin(c)==b} for b in BINS}
    te=pd.read_csv(gk.DATA/f"{split}_test.csv", usecols=["species"])
    te_bin=te["species"].map(lambda s: dbin(cnt.get(s,0)))
    sizes={b:{"n_species":len(binsp[b]),"n_test_rows":int((te_bin==b).sum())} for b in BINS}
    res={}
    for bb in ["dmpnn","graphconv"]:
        for t in TOST:
            arms=[gk.load_arm(bb,gk.GVAR[t],split),gk.load_arm(bb,gk.GVAR["t0"],split),
                  gk.load_arm(bb,gk.GVAR["t2"],split),gk.load_arm(bb,gk.GVAR["t0"],split)]
            for b in BINS:
                r=gk.run_comparison(arms, delta, species_set=binsp[b])
                key=f"{bb}/{t}~t2/{b}"
                if r.get("status")=="ok":
                    res[key]={"dd":round(r["dd"],4),"ci":[round(r["ci_lo"],4),round(r["ci_hi"],4)],
                              "cat":r["category"],"n_rows":r["n_rows"]}
                else:
                    res[key]={"status":r.get("status"),"n_rows":r.get("n_rows",0)}
    return {"bin_sizes":sizes,"dd":res}

dB=GKB.load_frozen("delta")["delta"]; dP=GKP.load_frozen("delta")["delta"]
s5_1={"bins":BINS,"b1":support_bins(GKB,"b1_group",dB),"phase1":support_bins(GKP,"discovery_group",dP)}

# §5-2 species counts + one-hot
def sp_counts(gk, splits):
    o={}
    allidx=set()
    for sp in splits:
        tr=pd.read_csv(gk.DATA/f"{sp}_train.csv",usecols=["species_idx"]); te=pd.read_csv(gk.DATA/f"{sp}_test.csv",usecols=["species_idx"])
        o[sp.replace('discovery_','').replace('replication_','').replace('b1_','')]={"n_train_species":int(tr['species_idx'].nunique())}
        allidx|=set(tr["species_idx"].astype(int))|set(te["species_idx"].astype(int))
    return o, max(allidx)+1
p1_disc,_=sp_counts(GKP,["discovery_group","discovery_scaffold","discovery_scaffold_generic","discovery_designed_leaky"])
# Phase 1 one-hot per partition (discovery vs replication) — read both
import glob
def onehot(gk, split):
    tr=pd.read_csv(gk.DATA/f"{split}_train.csv",usecols=["species_idx"]);te=pd.read_csv(gk.DATA/f"{split}_test.csv",usecols=["species_idx"])
    return int(max(set(tr['species_idx'].astype(int))|set(te['species_idx'].astype(int)))+1)
s5_2={"phase1_one_hot_discovery":onehot(GKP,"discovery_group"),
      "phase1_one_hot_replication":onehot(GKP,"replication_group"),
      "b1_one_hot":1975,
      "phase1_discovery_n_train_species":p1_disc,
      "note":"one-hot dim = max species_idx+1 per partition (Phase 1) / global (B1)."}
res=json.loads((R/"audit/expansion_results.json").read_text(encoding="utf-8"))
res["s5_1_support_bins"]=s5_1
res["s5_species_oov"]["phase1_species"]=s5_2
(R/"audit/expansion_results.json").write_text(json.dumps(res,ensure_ascii=False,indent=2),encoding="utf-8")
print("§5-2 one-hot: Phase1 disc=%d repl=%d | B1=1975"%(s5_2["phase1_one_hot_discovery"],s5_2["phase1_one_hot_replication"]))
print("§5-2 Phase1 disc n_train_species:", p1_disc)
print("\n§5-1 bin sizes B1:", {b:s5_1["b1"]["bin_sizes"][b] for b in BINS})
print("§5-1 B1 t4~t2 dd by bin (dmpnn):", {b:s5_1["b1"]["dd"].get(f"dmpnn/t4~t2/{b}") for b in BINS})
print("[stored s5_1, s5_2]")
