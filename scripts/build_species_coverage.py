#!/usr/bin/env python3
"""Per-species resource-coverage table (12 columns; key = ECOTOX species_number).
Rows = union of E-full (raw ECOTOX June-2026) and P (the ADORE processed mortality corpus).
tax_group is the ECOTOX ecotox_group field; in_phylo / in_deb are the authoritative ADORE
flags for the modelled species and resource name-membership for the remaining field species.
See species_coverage_README.md.

External data (obtain separately -- see the top-level README, \"Data\"):
  ECOTOX_ASCII_DIR : ECOTOX June-2026 ASCII release directory
  ADORE_DIR        : the ADORE distribution (Schur et al. 2023)
"""
import csv, re, os
DUMP=os.environ.get("ECOTOX_ASCII_DIR","ecotox_ascii_06_11_2026")
ADORE=os.environ.get("ADORE_DIR","adore_dataset")
TAX=os.path.join(ADORE,"taxonomy")
OUT=os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","results","q2_v4","audit","species_coverage.csv")
EP={"LC50","EC50","LC50*","LC50/","EC50*","EC50/","LD50/","LC50*/","EC50*/","IC50","EC0"}
EFF={"MOR","ITX","POP","GRO","PHY","~MOR","MOR/","~ITX","PHY/","POP/","~MOR/","ITX/"}
MED={"FW","SW","FW/","SW/","CUL/"}
def norm(s): return re.sub(r"[\s_]+","_",str(s).strip().lower())

# resources
with open(os.path.join(TAX,"FCA_pdm_species.csv"),encoding="utf-8",errors="ignore") as f:
    phylo={norm(x) for x in next(csv.reader(f))[1:] if x.strip()}
deb=set()
with open(os.path.join(TAX,"FC_amp_lifehistory.csv"),encoding="utf-8",errors="ignore") as f:
    for r in csv.DictReader(f):
        v=(r.get("latin_name") or "").strip()
        if v: deb.add(norm(v))
RES=phylo|deb

# synonyms: species_number -> list of original names
syn={}
with open(os.path.join(DUMP,"validation","species_synonyms.txt"),encoding="utf-8",errors="ignore") as f:
    f.readline()
    for line in f:
        p=line.rstrip("\n").split("|")
        if len(p)>=2 and p[0].strip(): syn.setdefault(p[0].strip(),[]).append(p[1].strip())

# E-full (v2 recipe) -> species_number -> n_records_field
res_q={}
with open(os.path.join(DUMP,"results.txt"),encoding="utf-8",errors="ignore") as f:
    f.readline()
    for line in f:
        p=line.rstrip("\n").split("|")
        if len(p)<22: continue
        if p[18].strip() in EP and p[21].strip() in EFF:
            res_q[p[1].strip()]=res_q.get(p[1].strip(),0)+1
field={}
with open(os.path.join(DUMP,"tests.txt"),encoding="utf-8",errors="ignore") as f:
    f.readline()
    for line in f:
        p=line.rstrip("\n").split("|")
        if len(p)<78: continue
        if p[18].strip()!="Water" or p[77].strip() not in MED: continue
        q=res_q.get(p[0].strip())
        if q and p[17].strip(): field[p[17].strip()]=field.get(p[17].strip(),0)+q

# P (ADORE processed). 26 ADORE species use COMPOSITE species_number keys (":"-joined ECOTOX
# IDs that ADORE consolidated; each maps to several raw E-full species). Attribute each ADORE
# species to ONE primary E-full component = the component carrying the most E-full records.
def primary_key(k):
    if k.isdigit(): return k
    comps=k.split(":")
    inef=[c for c in comps if c in field]
    return max(inef, key=lambda c: field.get(c,0)) if inef else comps[0]
praw={}
with open(os.path.join(ADORE,"processed","ecotox_mortality_processed.csv"),encoding="utf-8",errors="ignore") as f:
    for row in csv.DictReader(f):
        n=(row.get("species_number") or "").strip()
        if not n: continue
        d=praw.setdefault(n,{"nrec":0,"pdm":0,"amp":0,"gs":""})
        d["nrec"]+=1
        if not d["gs"]:
            d["gs"]=(row.get("tax_gs") or row.get("tax_species") or "").strip().replace("_"," ")
            d["pdm"]=1 if str(row.get("tax_pdm_available","")).strip().lower()=="true" else 0
            d["amp"]=1 if (row.get("tax_ps_ampv") or "").strip() else 0
padore={}; pname={}; auth_pdm={}; auth_amp={}
akey={}   # every ECOTOX id (composite component or integer) -> its ADORE species_number
for k,d in praw.items():
    pk=primary_key(k)
    padore[pk]=padore.get(pk,0)+d["nrec"]
    auth_pdm[pk]=max(auth_pdm.get(pk,0),d["pdm"]); auth_amp[pk]=max(auth_amp.get(pk,0),d["amp"])
    if not pname.get(pk): pname[pk]=d["gs"]
    for c in (k.split(":") if not k.isdigit() else [k]): akey[c]=k
print(f"[P composite check] ADORE mortality keys={len(praw)} -> distinct primaries={len(padore)}  (expect 1267)")
# A203 (subset of P) -> map through same primary
a203=set()
with open(os.path.join(ADORE,"processed","a-FCA2FCA_mortality.csv"),encoding="utf-8",errors="ignore") as f:
    for row in csv.DictReader(f):
        n=(row.get("species_number") or "").strip()
        if n: a203.add(primary_key(n))

# species.txt taxonomy for the union
union=set(field)|set(padore)
info={}
with open(os.path.join(DUMP,"validation","species.txt"),encoding="utf-8",errors="ignore") as f:
    f.readline()
    for line in f:
        p=line.rstrip("\n").split("|")
        if len(p)<15: continue
        if p[0].strip() in union:
            info[p[0].strip()]={"latin":p[2].strip(),"phylum":p[4].strip(),"subphylum":p[5].strip(),
                                "class":p[7].strip(),"eg":p[14].strip()}

# provisional tax_group (phylum/class rule; algae NOT final)
FISH={"Actinopterygii","Actinopteri","Chondrichthyes","Sarcopterygii","Cephalaspidomorphi","Myxini","Cladistii","Holocephali","Elasmobranchii","Coelacanthi","Dipneusti","Petromyzonti","Hyperoartia"}
CRU={"Malacostraca","Branchiopoda","Maxillopoda","Ostracoda","Copepoda","Cephalocarida","Remipedia","Hexanauplia","Ichthyostraca","Cirripedia"}
ALGP={"Chlorophyta","Bacillariophyta","Cyanophycota","Pyrrophycophyta","Charophyta","Cryptophycophyta","Phaeophyta","Haptophyta","Chrysophyta","Rhodophycota","Prasinophyta","Ochrophyta","Euglenophycota","Rhodophyta"}
ALGC={"Chlorophyceae","Bacillariophyceae","Cyanophyceae","Trebouxiophyceae","Zygnematophyceae","Coscinodiscophyceae","Fragilariophyceae","Ulvophyceae","Chrysophyceae","Xanthophyceae","Dinophyceae","Cryptophyceae","Prymnesiophyceae","Rhodophyceae","Charophyceae","Phaeophyceae","Chlorodendrophyceae","Conjugatophyceae","Prasinophyceae"}
def taxgroup(i):  # OPTION A (director-confirmed): ECOTOX ecotox_group field
    eg=(i.get("eg","") or "").split(";")[0].strip().lower()
    return {"fish":"fish","crustaceans":"crusta","algae":"algae"}.get(eg,"other")

def resolve(num):
    acc=info.get(num,{}).get("latin","") or pname.get(num,"")
    names=[acc]+syn.get(num,[])
    # AUTHORITATIVE flag for modelled (P) species; else resource name-membership
    if num in padore:
        inp=auth_pdm.get(num,0); ind=auth_amp.get(num,0)
    else:
        inp=1 if any(norm(x) in phylo for x in names if x) else 0
        ind=1 if any(norm(x) in deb for x in names if x) else 0
    rn=acc
    for x in names:
        if x and norm(x) in RES: rn=x; break
    return acc,rn,int(inp),int(ind)

rows=[]
for num in union:
    i=info.get(num,{})
    acc,rn,inp,ind=resolve(num)
    rows.append({"species_number":num,"adore_species_number":akey.get(num,""),
                 "latin_name":acc,"resolved_name":rn,
                 "ecotox_group":i.get("eg",""),"tax_group":taxgroup(i),
                 "n_records_field":field.get(num,0),"n_records_adore":padore.get(num,0),
                 "in_adore_processed":int(num in padore),"in_adore_public":int(num in a203),
                 "in_phylo":inp,"in_deb":ind})
rows.sort(key=lambda r:(-r["n_records_field"],-r["n_records_adore"],r["latin_name"]))
with open(OUT,"w",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

# ---- VERIFY algae-independent aggregates ----
ef=[r for r in rows if r["n_records_field"]>0]
P=[r for r in rows if r["in_adore_processed"]]
A=[r for r in rows if r["in_adore_public"]]
D=[r for r in rows if r["in_adore_processed"] and not r["in_adore_public"]]
def pct(sub,key,basis=None):
    n=len(sub); hit=[r for r in sub if r[key]]
    tot=sum(r[basis] for r in sub) if basis else 0
    rw=100*sum(r[basis] for r in hit)/tot if basis and tot else 0
    return len(hit),100*len(hit)/n if n else 0,rw
print(f"[written] {OUT}")
print(f"TOTAL rows (E-full U P) = {len(rows)}")
print(f"E-full species = {len(ef)}  records = {sum(r['n_records_field'] for r in ef)}   (target 3268 / 160869)")
print(f"  P subset-of-Efull? P not in Efull = {sum(1 for r in P if r['n_records_field']==0)}")
ph=pct(ef,'in_phylo','n_records_field'); db=pct(ef,'in_deb','n_records_field')
print(f"  E-full phylo: {ph[0]} sp {ph[1]:.1f}% raw / {ph[2]:.1f}% recwt   (target 596 18.2 / 56.0)")
print(f"  E-full DEB:   {db[0]} sp {db[1]:.1f}% raw / {db[2]:.1f}% recwt   (target 247  7.6 / 50.6)")
from collections import Counter
print(f"P species = {len(P)}  records = {sum(r['n_records_adore'] for r in P)}   (target 1267 / 70670)")
php=pct(P,'in_phylo','n_records_adore'); pdb=pct(P,'in_deb','n_records_adore')
print(f"  P phylo: {php[0]} sp {php[1]:.1f}% raw / {php[2]:.1f}% recwt   (target 526 41.5 / 75.3)")
print(f"  P DEB:   {pdb[0]} sp {pdb[1]:.1f}% raw / {pdb[2]:.1f}% recwt   (target 238 18.8 / 68.3)")
print(f"A203 species = {len(A)}  records = {sum(r['n_records_adore'] for r in A)}   (target 203 / 44964)")
ap=pct(A,'in_phylo','n_records_adore'); ad=pct(A,'in_deb','n_records_adore')
print(f"  A203 phylo: {ap[0]} {ap[1]:.1f}%   DEB: {ad[0]} {ad[1]:.1f}%   (target 203 100 / 157 77.3)")
print(f"dropped species = {len(D)}  records = {sum(r['n_records_adore'] for r in D)}   (target 1064 / 25706)")

# ==== FINAL: recomputed coverage tables for the manuscript (A = ecotox_group) ====
print("\n" + "="*72)
print("RECOMPUTED COVERAGE (director-confirmed A rule; authoritative-hybrid phylo/DEB)")
print("="*72)
print("\n-- E-full tax_group breakdown (Fig 3 / coverage_by_taxgroup) --")
print("group     n_sp   phylo_raw   DEB_raw     [published within_taxon]")
pub={'fish':(640,'','','' ),'crusta':(653,'',''),'algae':(385,16.62,0.0),'other':(1590,0.57,0.0)}
for g in ["fish","crusta","algae","other"]:
    sub=[r for r in ef if r["tax_group"]==g]; n=len(sub)
    ph=100*sum(r["in_phylo"] for r in sub)/n; db=100*sum(r["in_deb"] for r in sub)/n
    pubn=pub[g][0]
    print(f"{g:7s} {n:5d}   {ph:5.1f}%      {db:4.1f}%       (published n={pubn})")
print("\n-- Table 7 (coverage_ladder) rows, NEW vs PUBLISHED --")
def row(name,sub,basis):
    n=len(sub); tot=sum(r[basis] for r in sub) or 1
    pr=[r for r in sub if r["in_phylo"]]; dbb=[r for r in sub if r["in_deb"]]
    return (n,sum(r[basis] for r in sub),100*len(pr)/n,100*sum(r[basis] for r in pr)/tot,
            100*len(dbb)/n,100*sum(r[basis] for r in dbb)/tot)
algae_ef=[r for r in ef if r["tax_group"]=="algae"]
specs=[("E_full",ef,"n_records_field","3268/160869 phylo18.2/56.0 DEB7.6/50.6"),
       ("P",P,"n_records_adore","1267/70670 phylo41.5/75.3 DEB18.8/68.3"),
       ("A203",A,"n_records_adore","203/44964 phylo100/100 DEB77.3/97.7"),
       ("dropped",D,"n_records_adore","1064/25706 phylo30.4/32.2 DEB7.6/16.9"),
       ("algae(field)",algae_ef,"n_records_field","385 phylo16.62 DEB0")]
print(f"{'set':13s} {'n_sp':>5} {'n_rec':>7} {'phylo_raw':>9} {'phylo_rw':>8} {'DEB_raw':>7} {'DEB_rw':>7}   published")
for nm,sub,basis,pubs in specs:
    a=row(nm,sub,basis)
    print(f"{nm:13s} {a[0]:5d} {a[1]:7d} {a[2]:8.1f}% {a[3]:7.1f}% {a[4]:6.1f}% {a[5]:6.1f}%   [{pubs}]")
print("\nKEY CHANGES vs manuscript:  algae 385->%d (ecotox_group) | E-full phylo 18.2%%->%.1f%%"
      % (len(algae_ef), row('E',ef,'n_records_field')[2]))

