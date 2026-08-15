# -*- coding: utf-8 -*-
"""Fix capture_rate as an artifact (rough calc; denominator includes measurement noise -> not bounded by 1).
capture_rate(tier,bb) = (RMSE_t0^2 - RMSE_tier^2) / (species-axis pairwise SD)^2
RMSE = per-seed mean, group, warm, main (Phase1 discovery_group / B1 b1_group).
species-axis SD = pairwise |Δ target_log10| SD over 'fix compound, vary species' (Phase1 discovery=0.960; B1 computed).
"""
import json, itertools
import numpy as np, pandas as pd
from pathlib import Path
R=Path(r".\results\q2_v4")

def species_axis_sd(df):
    """fix compound (smiles), pairwise |Δ| of per-(smiles,species) mean target across species. Returns (sd,median,n_units,n_pairs)."""
    agg=df.groupby(["smiles","species"])["target_log10"].mean().reset_index()
    diffs=[]; n_units=0; n_pairs=0
    for sm,g in agg.groupby("smiles"):
        v=g["target_log10"].to_numpy(float)
        if len(v)<2: continue
        n_units+=1
        dif=np.abs(v[:,None]-v[None,:]); iu=np.triu_indices(len(v),1)
        d=dif[iu]; diffs.append(d); n_pairs+=len(d)
    allp=np.concatenate(diffs)
    return float(np.std(allp)), float(np.median(allp)), n_units, n_pairs

# ---- validate method on discovery (expect SD~0.960, n_units 1146, n_pairs 157441) ----
disc=pd.concat([pd.read_csv(R/"data"/f"discovery_group_{s}.csv",usecols=["smiles","species","target_log10"]) for s in ["train","test"]],ignore_index=True)
sd_d,med_d,nu_d,np_d=species_axis_sd(disc)
print(f"[VALIDATE discovery] SD={sd_d:.4f} median={med_d:.4f} n_units={nu_d} n_pairs={np_d}  (PART6: SD=0.960 units=1146 pairs=157441)")
match = abs(sd_d-0.960)<0.01 and nu_d==1146 and np_d==157441

# ---- B1 species-axis SD (same method) ----
b1=pd.concat([pd.read_csv(R/"data_b1"/f"b1_group_{s}.csv",usecols=["smiles","species","target_log10"]) for s in ["train","test"]],ignore_index=True)
sd_b,med_b,nu_b,np_b=species_axis_sd(b1)
print(f"[B1] species-axis SD={sd_b:.4f} median={med_b:.4f} n_units={nu_b} n_pairs={np_b}")

# ---- RMSE (per-seed mean) from s4_4 (== PART 3) ----
res=json.loads((R/"audit/expansion_results.json").read_text(encoding="utf-8"))
rm=res["s4_results"]["s4_4_absolute_rmse"]
TIERS=["t0","t1","t1p","t2","t3a","t3b","t4"]; BB=["dmpnn","graphconv"]
SD={"phase1":0.960,"b1":sd_b}
def cr(exp,bb):
    r0=rm[exp]["group"][f"{bb}/t0"]["per_seed_mean"]
    return {t: round((r0**2 - rm[exp]["group"][f"{bb}/{t}"]["per_seed_mean"]**2)/SD[exp]**2,4) for t in TIERS}
out={"definition":"capture_rate(tier,bb) = (RMSE_t0^2 - RMSE_tier^2) / (species-axis pairwise SD)^2",
     "inputs":{"RMSE":"per-seed mean, group split, warm, main condition (Phase1 discovery_group / B1 b1_group; PART3 table = expansion_results.json s4_4)",
               "species_axis_SD":{"phase1_discovery":0.960,"b1":round(sd_b,4)},
               "species_axis_SD_source":"pairwise |Δ target_log10| SD over 'fix compound (smiles), vary species (per-(smiles,species) mean)'; Phase1 from PART6 (mechanism_facts_compute.txt), B1 computed here (same method, validated on discovery)"},
     "caveat":"ROUGH calculation ('거친 계산'). Denominator (species-axis pairwise SD)^2 includes measurement/replicate noise, so capture_rate is NOT bounded above by 1.",
     "method_validation_discovery":{"recomputed_SD":round(sd_d,4),"n_units":nu_d,"n_pairs":np_d,"matches_PART6_0.960/1146/157441":bool(match)},
     "capture_rate":{"phase1":{bb:cr("phase1",bb) for bb in BB},"b1":{bb:cr("b1",bb) for bb in BB}}}
# director's 4 hand-values check
chk={"t2_dmpnn (~19%)":out["capture_rate"]["phase1"]["dmpnn"]["t2"],
     "t2_graphconv (~17%)":out["capture_rate"]["phase1"]["graphconv"]["t2"],
     "t3b_dmpnn (~23%)":out["capture_rate"]["phase1"]["dmpnn"]["t3b"],
     "t3b_graphconv (~27%)":out["capture_rate"]["phase1"]["graphconv"]["t3b"]}
out["director_hand_values_check"]=chk
A=R/"audit"
(A/"capture_rate.json").write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding="utf-8")
# md
L=["# capture_rate (거친 계산 — 산출물 고정)\n",
 "> 정의: capture_rate(tier,backbone) = (RMSE_t0² − RMSE_tier²) / (species-axis pairwise SD)²",
 "> ⚠ 거친 계산: 분모(종축 pairwise SD)²는 측정/복제 잡음을 포함하므로 상한이 1이 아님.\n",
 "## 입력·출처",
 "- RMSE = per-seed 평균, group·warm·main (Phase1 discovery_group / B1 b1_group; PART3 = `expansion_results.json s4_4`).",
 f"- species-axis SD: Phase1 discovery = **0.960** (PART6 `mechanism_facts_compute.txt`); B1 = **{sd_b:.4f}** (동일 방법, 본 파일에서 산출).",
 f"- 방법 검증(discovery 재현): SD={sd_d:.4f}, n_units={nu_d}, n_pairs={np_d} → PART6(0.960/1146/157441) 일치={match}.\n",
 "## capture_rate (%) — 전 tier × 양 backbone × 양 실험",
 "| tier | P1 dmpnn | P1 graphconv | B1 dmpnn | B1 graphconv |","|---|---|---|---|---|"]
for t in TIERS:
    L.append(f"| {t} | {out['capture_rate']['phase1']['dmpnn'][t]*100:.1f}% | {out['capture_rate']['phase1']['graphconv'][t]*100:.1f}% | {out['capture_rate']['b1']['dmpnn'][t]*100:.1f}% | {out['capture_rate']['b1']['graphconv'][t]*100:.1f}% |")
L+=["","## director 손계산 4값 대조",
 f"- t2 D-MPNN: 기재 ~19% vs 산출 **{chk['t2_dmpnn (~19%)']*100:.1f}%**",
 f"- t2 GraphConv: 기재 ~17% vs 산출 **{chk['t2_graphconv (~17%)']*100:.1f}%**",
 f"- t3b D-MPNN: 기재 ~23% vs 산출 **{chk['t3b_dmpnn (~23%)']*100:.1f}%**",
 f"- t3b GraphConv: 기재 ~27% vs 산출 **{chk['t3b_graphconv (~27%)']*100:.1f}%**"]
(A/"capture_rate.md").write_text("\n".join(L),encoding="utf-8")
print("\n=== director 4값 대조 ==="); [print(f"  {k}: {v*100:.1f}%") for k,v in chk.items()]
print("\n[written] audit/capture_rate.md, capture_rate.json")
