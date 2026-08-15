"""§7 full Phase-1 difference list (facts). §8 computed direction/>=2x differences. NO explanation."""
import json
from pathlib import Path
R=Path(r".\results\q2_v4")
res=json.loads((R/"audit/expansion_results.json").read_text(encoding="utf-8"))

# ---- §7 differences (enumerated facts; "none" stated explicitly where none) ----
s7=[
 {"item":"confirmatory family","phase1":"present (replication_group, 30 comparisons)","b1":"ABSENT — no replication/confirmatory partition (structural exception; no substitute fabricated)"},
 {"item":"GPU seed->device map","phase1":"dual-GPU: seed 0-6 -> 5060Ti, 7-9 -> 4090 (BLOCK_A_LAUNCH_MANIFEST)","b1":"dual-GPU: seed 0-4 -> 4090, 5-9 -> 5060Ti (director-approved; both experiments dual-GPU, maps differ)"},
 {"item":"three margins","phase1":"delta 0.019777 / delta_det 0.087189 / delta' 0.005717","b1":"RE-DERIVED (no reuse): delta 0.035641 / delta_det 0.074082 / delta' 0.015957"},
 {"item":"NCBI taxonomy resolver","phase1":"resolved once for Phase-1 species","b1":"SAME resolver re-run on B1's 1975 species (validated: 100% agreement reproducing Phase-1 committed ncbi_taxonomy output)"},
 {"item":"t4 leak handling","phase1":"permutation test clean (all splits)","b1":"b1_scaffold flagged -> diagnosed (OOF-proof PASS, Phase1 20-perm clean, decisive control: ~79% capacity floor) -> characterized false positive -> t4 arm INCLUDED; capacity-floor baseline deferred to analysis (director-approved)"},
 {"item":"tax_group source coverage","phase1":"mortality map covers Phase-1 species","b1":"same MORT map covers only 453/1788 b1_group train species (B1's new species absent from Phase-1 mortality file) -> tax_group exploratory sub-family reduced coverage"},
 {"item":"one-hot dimension","phase1":"779 (discovery) / 1006 (replication)","b1":"1975 (single global vocab)"},
 {"item":"comparison count","phase1":"225 (30 primary + 30 confirmatory + 3 det + 150 exploratory + 12 sensitivity)","b1":"189 (30 primary + 0 confirmatory + 3 det + 150 exploratory + 6 sensitivity)"},
 {"item":"prereg freeze integrity","phase1":"n/a","b1":"PREREG_EXPANSION.md amended after initial creation incl. GPU deviation post-dating first run (15:00); all amendments predate margin derivation/performance"},
 {"item":"support-bin edges (§5-1 director-specified)","phase1":"gatekeeping default 1-5/6-20/21-100/100+","b1":"director §5-1 bins 1-4/5-9/10-19/20-49/50+ computed separately (both retained)"},
 {"item":"exploratory replication","phase1":"full exploratory set","b1":"exploratory computed (rank/support-bin/tax_group/scaffold/designed-leaky/cross-backbone); tax_group reduced coverage as above; otherwise same structure"},
 {"item":"dataset rows","phase1":"discovery+replication ECOTOX (Phase-1 curation)","b1":"B1_final (2026 ECOTOX pull, P-derived filters, disjoint from Phase-1 training P by result_id + precise dup)"},
]

# ---- §8 large diffs (sign flip OR >=2x magnitude). facts only, no 'why'. ----
s8=[]
def ratio(a,b):
    if a==0 or b==0: return None
    return abs(b)/abs(a)
# margins
for k,p,b in [("delta",res["s3_margins"]["delta_primary"]["phase1"],res["s3_margins"]["delta_primary"]["b1"]),
              ("delta_det",res["s3_margins"]["delta_det"]["phase1"],res["s3_margins"]["delta_det"]["b1"]),
              ("delta_prime",res["s3_margins"]["delta_prime"]["phase1"],res["s3_margins"]["delta_prime"]["b1"])]:
    r=ratio(p,b)
    if r and (r>=2 or r<=0.5): s8.append({"item":f"margin {k}","phase1":round(p,6),"b1":round(b,6),"B1/P1":round(r,2)})
# one-hot
s8.append({"item":"one-hot dim","phase1":"779/1006","b1":1975,"B1/P1":">=2x vs discovery(779)"})
# warm OOV rows (group)
p1o=res["s5_species_oov"]["phase1_warm_oov"]["group"]; b1o=res["s5_species_oov"]["warm_splits"]["b1_group"]
s8.append({"item":"warm OOV test rows (group)","phase1":p1o["n_test_OOV_rows"],"b1":b1o["n_test_OOV_rows"],"B1/P1":round(b1o["n_test_OOV_rows"]/p1o["n_test_OOV_rows"],2)})
s8.append({"item":"warm OOV test species (group)","phase1":p1o["n_test_OOV_species"],"b1":b1o["n_test_OOV_species"],"B1/P1":round(b1o["n_test_OOV_species"]/p1o["n_test_OOV_species"],2)})
# effect-size dd: sign flip or >=2x, per (split,comp,bb)
ef=res["s4_results"]["s4_2_effect_sizes"]
for split,comps in ef.items():
    for comp,row in comps.items():
        for bb in ["dmpnn","graphconv"]:
            p=row.get(f"P1_{bb}"); b=row.get(f"B1_{bb}")
            if not p or not b: continue
            pd_,bd_=p["dd"],b["dd"]
            flip = (pd_>0)!=(bd_>0) and abs(pd_)>0.005 and abs(bd_)>0.005
            r=ratio(pd_,bd_)
            big = r is not None and (r>=2 or r<=0.5) and (abs(pd_)>0.01 or abs(bd_)>0.01)
            if flip or big:
                s8.append({"item":f"dd {split}/{comp}/{bb}","phase1":pd_,"b1":bd_,"sign_flip":flip,"B1/P1":round(r,2) if r else None})

res["s7_phase1_differences"]=s7
res["s8_large_diffs"]={"criterion":"sign differs OR magnitude ratio >=2x (facts only; no explanation)","items":s8}
(R/"audit/expansion_results.json").write_text(json.dumps(res,ensure_ascii=False,indent=2),encoding="utf-8")
print(f"§7: {len(s7)} difference items")
print(f"§8: {len(s8)} large-diff items:")
for x in s8: print("  ",x.get("item"),"P1=",x.get("phase1"),"B1=",x.get("b1"),"flip" if x.get("sign_flip") else "",("x"+str(x.get("B1/P1"))) if x.get("B1/P1") else "")
print("[stored s7,s8]")
