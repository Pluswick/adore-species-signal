"""Parse B1 + Phase1 gatekeeping_results.json into the director's §4 tables. Facts only, re-read
from artifacts. Phase1 <-> B1 split map: discovery_group<->b1_group, discovery_scaffold<->b1_scaffold,
discovery_scaffold_generic<->b1_scaffold_generic, discovery_designed_leaky<->b1_designed_leaky."""
import json
from pathlib import Path
R=Path(r".\results\q2_v4")
P1=json.loads((R/"runs"/"bootstrap"/"gatekeeping_results.json").read_text(encoding="utf-8"))["results"]
B1=json.loads((R/"runs_b1"/"bootstrap"/"gatekeeping_results.json").read_text(encoding="utf-8"))["results"]

def cat_census(res, family, test=None, subtype=None):
    c={"equivalent":0,"significant":0,"indeterminate":0,"missing/empty":0}
    for r in res:
        if r.get("family")!=family: continue
        if test and r.get("test")!=test: continue
        if subtype and r.get("subtype")!=subtype: continue
        st=r.get("status");
        if st!="ok": c["missing/empty"]+=1; continue
        c[r["category"]]=c.get(r["category"],0)+1
    return c

def find(res, family, backbone, cand, ref, split, test):
    for r in res:
        if (r.get("family")==family and r.get("backbone")==backbone and r.get("cand")==cand
            and r.get("ref")==ref and r.get("split")==split and r.get("test")==test):
            return r
    return None

GVAR={"t0":"no_species","t2":"true_species_categorical","t3a":"true_species_taxonomy_original",
      "t3b":"true_species_taxonomy_ncbi","t4":"true_species_late_fusion"}
def dd_ci(res, backbone, cand_t, ref_t, split, test, family):
    r=find(res, family, backbone, GVAR[cand_t], GVAR[ref_t], split, test)
    if r is None or r.get("status")!="ok": return None
    return {"dd":round(r["dd"],4),"ci":[round(r["ci_lo"],4),round(r["ci_hi"],4)],"cat":r["category"],
            "q_family":round(r.get("q_family",float('nan')),4),"n_rows":r.get("n_rows")}

# ---- §4-1 census ----
print("="*72); print("§4-1 DECISION CENSUS  [equivalent / significant / indeterminate]"); print("="*72)
def show(lbl,c): print(f"  {lbl:<34} eq={c['equivalent']} sig={c['significant']} inc={c['indeterminate']} (miss/empty={c['missing/empty']})")
for exp,res in [("Phase1",P1),("B1",B1)]:
    print(f"[{exp}]")
    show("primary superiority", cat_census(res,"primary","superiority"))
    show("primary TOST",        cat_census(res,"primary","TOST"))
    show("confirmatory superiority", cat_census(res,"confirmatory","superiority"))
    show("confirmatory TOST",   cat_census(res,"confirmatory","TOST"))
    show("deterministic TOST",  cat_census(res,"deterministic","TOST"))
    for sub in ["rank","support-bin","tax_group","scaffold","designed-leaky","cross-backbone"]:
        show(f"exploratory {sub}", cat_census(res,"exploratory",subtype=sub))
    show("sensitivity_ensemble", cat_census(res,"sensitivity_ensemble"))
    s2=sum(1 for r in res if r.get("reaches_stage2") is True)
    print(f"  reaches_stage2 (primary+confirmatory TOST equivalent): {s2}")

# ---- §4-2 effect sizes: t0->t2 (superiority), t2->t3a/t3b/t4 (TOST) per bb per split ----
print("\n"+"="*72); print("§4-2 EFFECT SIZES dd [90% CI]  (block=smiles, N_BOOT 2000)"); print("="*72)
SPLITMAP=[("group","discovery_group","b1_group","primary"),
          ("scaffold","discovery_scaffold","b1_scaffold","exploratory"),
          ("scaffold_generic","discovery_scaffold_generic","b1_scaffold_generic","exploratory"),
          ("designed_leaky","discovery_designed_leaky","b1_designed_leaky","exploratory")]
COMPS=[("t0->t2","t2","t0","superiority"),("t2->t3a","t3a","t2","TOST"),
       ("t2->t3b","t3b","t2","TOST"),("t2->t4","t4","t2","TOST")]
s4_2={}
for sname,psplit,bsplit,fam in SPLITMAP:
    s4_2[sname]={}
    print(f"\n-- split {sname} (Phase1 {fam if sname!='group' else 'primary'} / B1 {fam}) --")
    for clbl,ct,rt,test in COMPS:
        row={}
        for exp,res,split in [("P1",P1,psplit),("B1",B1,bsplit)]:
            for bb in ["dmpnn","graphconv"]:
                row[f"{exp}_{bb}"]=dd_ci(res,bb,ct,rt,split,test, fam if exp=="B1" else ("primary" if sname=="group" else "exploratory"))
        s4_2[sname][clbl]=row
        def fmt(x): return f"{x['dd']:+.4f}[{x['ci'][0]:+.3f},{x['ci'][1]:+.3f}]({x['cat'][:3]})" if x else "NA"
        print(f"  {clbl:<8} P1 dm={fmt(row['P1_dmpnn'])} gc={fmt(row['P1_graphconv'])}")
        print(f"  {'':8} B1 dm={fmt(row['B1_dmpnn'])} gc={fmt(row['B1_graphconv'])}")

# ---- §4-3 primary TOST q-values ----
print("\n"+"="*72); print("§4-3 PRIMARY TOST q_family"); print("="*72)
for exp,res in [("Phase1",P1),("B1",B1)]:
    qs=[(r["label"],round(r["q_family"],4),r["category"]) for r in res if r.get("family")=="primary" and r.get("test")=="TOST" and r.get("status")=="ok"]
    passed=sum(1 for _,q,_ in qs if q<0.05)
    minq=min(q for _,q,_ in qs) if qs else None
    print(f"[{exp}] primary TOST n={len(qs)} min_q={minq} q<0.05 count={passed}")
    for l,q,c in sorted(qs,key=lambda x:x[1]): print(f"    {l:<22} q={q:.4f} {c}")

out={"s4_1_note":"printed above; stored structured below","s4_2_effect_sizes":s4_2}
OUT_PATH = Path(__file__).resolve().parents[2] / "results" / "q2_v4" / "audit" / "s4_parsed.json"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
OUT_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\n[written {OUT_PATH}]")
