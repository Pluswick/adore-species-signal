"""Device-component diagnostic for the margins (NO retraining; re-aggregate stored per-seed RMSEs
from the frozen margin JSONs; NO tier comparisons). Facts only.

seed->device (from ledger shard tags + Phase1 BLOCK_A_LAUNCH_MANIFEST.md):
  B1 warm:    seeds 0-4 -> RTX 4090 ; seeds 5-9 -> RTX 5060 Ti
  Phase1 warm: seeds 0-6 -> RTX 5060 Ti ; seeds 7-9 -> RTX 4090   (7 vs 3, unequal)
  B1 dprime:  seeds 10-54 -> 4090 ; 55-99 -> 5060 Ti  => delta' ensembles(10-seed): j1-4=4090, j6-9=5060Ti, j0,j5 mixed
  Phase1 dprime: within each decade ...0-6 -> 5060Ti, ...7-9 -> 4090  => EVERY delta' ensemble is device-mixed
"""
import json, numpy as np
from pathlib import Path
A = Path(r".\results\q2_v4\audit")
def rd(f): return json.loads((A/f).read_text(encoding="utf-8"))

def decomp(vals, idxA, idxB):
    """one-way (device) variance decomposition on a list of per-seed/ensemble RMSEs."""
    x = np.array(vals, float); a = x[idxA]; b = x[idxB]
    grand = x[np.r_[idxA, idxB]].mean()
    ssb = len(a)*(a.mean()-grand)**2 + len(b)*(b.mean()-grand)**2
    sst = ((x[np.r_[idxA, idxB]]-grand)**2).sum()
    s_c_full = np.std(x[np.r_[idxA, idxB]], ddof=1)  # SD used in delta (over the pooled group)
    return {"meanA":a.mean(),"sdA":np.std(a,ddof=1),"meanB":b.mean(),"sdB":np.std(b,ddof=1),
            "mean_diff":a.mean()-b.mean(),"abs_diff_over_sc":abs(a.mean()-b.mean())/s_c_full if s_c_full>0 else float('nan'),
            "frac_between":ssb/sst if sst>0 else float('nan'),"ssb":ssb,"sst":sst,"s_c_full":s_c_full}

def subset_delta(perc, idx):
    """recompute delta over a device-homogeneous seed/ensemble subset: sqrt(mean_c Var_subset)."""
    s2=[]
    for cond,v in perc.items():
        arr=np.array(v, float)[idx]
        s2.append(np.var(arr, ddof=1))
    return float(np.sqrt(np.mean(s2))), len(idx)-1, len(perc)

def run_delta(frozen, idxA, idxB, key="rmses"):
    d=rd(frozen); per=d["per_condition"]
    rows={}; fb=[]; ssb_tot=sst_tot=0.0
    percvals={c:per[c][key] for c in per}
    for c in per:
        r=decomp(per[c][key], idxA, idxB); rows[c]=r; fb.append(r["frac_between"])
        ssb_tot+=r["ssb"]; sst_tot+=r["sst"]
    dA,dfA,C=subset_delta(percvals, idxA); dB,dfB,_=subset_delta(percvals, idxB)
    return rows, {"mean_frac_between":float(np.mean(fb)),"median_frac_between":float(np.median(fb)),
                  "pooled_frac_between":ssb_tot/sst_tot,
                  "delta_subsetA":dA,"df_subsetA":dfA*C,"delta_subsetB":dB,"df_subsetB":dfB*C,
                  "frozen_full_delta":d.get("delta") or d.get("delta_prime")}

out={}
# ---- delta primary ----
print("="*70); print("delta (per-seed RMSE)  — device between-component"); print("="*70)
for tag,frozen,idxA,idxB,nameA,nameB in [
    ("B1", "delta_primary_frozen_b1.json", list(range(0,5)), list(range(5,10)), "4090(0-4)","5060Ti(5-9)"),
    ("Phase1","delta_primary_frozen.json", list(range(0,7)), list(range(7,10)), "5060Ti(0-6)","4090(7-9)")]:
    rows,agg=run_delta(frozen, idxA, idxB)
    out[f"delta_{tag}"]={"map":f"{nameA} vs {nameB}","agg":agg,"per_condition":{c:{k:round(v,6) for k,v in r.items()} for c,r in rows.items()}}
    print(f"\n[{tag}] {nameA} vs {nameB}   frozen delta={agg['frozen_full_delta']:.6f}")
    print(f"  frac_between(device): mean={agg['mean_frac_between']*100:.1f}%  median={agg['median_frac_between']*100:.1f}%  pooled={agg['pooled_frac_between']*100:.1f}%")
    print(f"  delta[{nameA} only]={agg['delta_subsetA']:.6f}(df{agg['df_subsetA']})  delta[{nameB} only]={agg['delta_subsetB']:.6f}(df{agg['df_subsetB']})")
    print(f"  per-condition |meanDiff|/s_c : " + ", ".join(f"{c.split('/')[0][:2]}{c.split('/')[1]}={r['abs_diff_over_sc']:.2f}" for c,r in rows.items()))
# ---- delta' ensemble (B1 only has device-homogeneous ensembles) ----
print("\n"+"="*70); print("delta' (ensemble RMSE) — device between-component"); print("="*70)
# B1: ensembles j1-4 = 4090, j6-9 = 5060Ti ; j0,j5 mixed (excluded from homogeneous decomposition)
rows,agg=run_delta("delta_prime_frozen_b1.json", [1,2,3,4], [6,7,8,9], key="ensemble_rmses")
out["deltaprime_B1"]={"map":"4090 ens(1-4) vs 5060Ti ens(6-9); ens0,5 device-mixed (excluded)","agg":agg,
                      "per_condition":{c:{k:round(v,6) for k,v in r.items()} for c,r in rows.items()}}
print(f"\n[B1 delta'] 4090 ens(1-4) vs 5060Ti ens(6-9)  frozen delta'={agg['frozen_full_delta']:.6f}  (ens0,5 mixed, excluded)")
print(f"  frac_between(device): mean={agg['mean_frac_between']*100:.1f}%  median={agg['median_frac_between']*100:.1f}%  pooled={agg['pooled_frac_between']*100:.1f}%")
print(f"  delta'[4090 ens1-4]={agg['delta_subsetA']:.6f}(df{agg['df_subsetA']})  delta'[5060Ti ens6-9]={agg['delta_subsetB']:.6f}(df{agg['df_subsetB']})")
print("\n[Phase1 delta'] every ensemble spans both GPUs (card rule ...0-6 5060Ti / ...7-9 4090 within each decade) -> NO device-homogeneous ensemble subset exists; between-device decomposition N/A.")
out["deltaprime_Phase1"]={"note":"every ensemble device-mixed (card rule splits each decade across both GPUs); no homogeneous subset; decomposition N/A"}

(A/"expansion_device_diagnostic.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print("\n[written] audit/expansion_device_diagnostic.json")
PY
