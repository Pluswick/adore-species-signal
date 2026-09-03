"""POST-HOC (⚠ 언블라인딩 이후 지정 — 사후 분석) t2(one-hot) vs t1(additive-bias) within-backbone DD, 4건.

director 요청 2026-08-05, `gatekeeping_results.json` 언블라인딩 **이후**. 사전등록 판정과 **동일 지위 아님**.
- 절차는 사전등록과 동일: per-seed paired Δ · block bootstrap 2000 · block=smiles · seed×block 공통 sset ·
  90% CI · 동결 δ=0.019777(파일 read) · 3범주+네 번째 칸 · SSOT 로더 · strata 키 동일.
- FDR = **이 4건 내부에서만** BH. 기존 패밀리와 미혼합.
- 기존 산출물(`gatekeeping_results.json`·`REPORT_2-0_to_2-9.txt`) **불변**. 신규 파일에만 기록.
동일 절차 보장을 위해 동결 파이프라인(run_q2_gatekeeping)의 load_arm/run_comparison/GVAR/δ 로더를 그대로 import.
Env: conda run -n src.
"""
from __future__ import annotations
import sys, os, json
from pathlib import Path
sys.path.insert(0, r".")
sys.path.insert(0, r".\scripts")
from run_q2_gatekeeping import load_arm, run_comparison, load_frozen, GVAR, R   # 동일 절차 재사용
from src.gatekeeping import bh_fdr

DELTA = load_frozen("delta")["delta"]                                           # 파일에서 읽음 (재계산 금지)
# cand=t2, ref=t1, within-backbone; base = 각 backbone 자기 t0 (공유 → 4항 DD가 직접차로 축약)
COMBOS = [("dmpnn", "discovery_group"), ("graphconv", "discovery_group"),
          ("dmpnn", "replication_group"), ("graphconv", "replication_group")]

# 기존 동결 산출물 READ-ONLY: 우월성 dd로부터 유도한 t2−t1 간격과 대조
EXIST = json.loads((R / "runs" / "bootstrap" / "gatekeeping_results.json").read_text(encoding="utf-8"))["results"]


def sup_dd(backbone, split, tierlabel):
    fam = "primary" if split == "discovery_group" else "confirmatory"
    for r in EXIST:
        if (r.get("family") == fam and r.get("test") == "superiority"
                and r.get("backbone") == backbone and r.get("label") == f"{backbone}/{tierlabel}"):
            return r["dd"]
    return None


results = []
for bb, split in COMBOS:
    arms = [load_arm(bb, GVAR["t2"], split),        # cand    = t2 (one-hot)
            load_arm(bb, GVAR["t0"], split),        # candbase= t0
            load_arm(bb, GVAR["t1"], split),        # ref     = t1 (additive bias)
            load_arm(bb, GVAR["t0"], split)]        # refbase = t0
    r = run_comparison(arms, DELTA)                 # per-seed paired, block=smiles, 2000 boot, 90% CI, decide
    d_t2 = sup_dd(bb, split, "t2>t0")
    d_t1 = sup_dd(bb, split, "t1>t0")
    derived = (d_t2 - d_t1) if (d_t2 is not None and d_t1 is not None) else None
    r.update({"backbone": bb, "split": split, "data": ("discovery" if "discovery" in split else "replication"),
              "cand": "t2", "ref": "t1",
              "sup_dd_t2_gt_t0": d_t2, "sup_dd_t1_gt_t0": d_t1, "derived_t2_minus_t1": derived,
              "direct_dd": r.get("dd"),
              "direct_minus_derived": (r["dd"] - derived) if (derived is not None and r.get("dd") is not None) else None})
    results.append(r)

ok = [r for r in results if r.get("status") == "ok"]                            # BH within these 4 only
for r, q in zip(ok, bh_fdr([r["p"] for r in ok])):
    r["q_posthoc"] = float(q)

out = R / "runs" / "bootstrap" / "posthoc_t2_vs_t1_results.json"
payload = {"phase": "POST-HOC (post-unblinding, director 2026-08-05) — NOT prereg status",
           "comparison": "cand=t2(one-hot) vs ref=t1(additive_bias), within-backbone, base=own t0",
           "procedure": "per-seed paired Δ · block=smiles · 2000 boot · seed×block common sset · 90% CI · "
                        "frozen δ read from delta_primary_frozen.json · 3-cat + 4th cell",
           "delta": DELTA, "fdr": "BH within these 4 only (NOT mixed with prereg families)",
           "frozen_delta_source": str((R / "audit" / "delta_primary_frozen.json")),
           "existing_files_untouched": ["gatekeeping_results.json", "REPORT_2-0_to_2-9.txt"],
           "results": results}
tmp = out.with_suffix(".json.tmp")
tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
os.replace(tmp, out)                                                            # atomic
print(f"[POSTHOC] {len(results)} comparisons -> {out}")
for r in results:
    print(f"  {r['backbone']:<9} {r['data']:<11} dd={r['dd']:+.6f} derived={r['derived_t2_minus_t1']:+.6f} "
          f"diff={r['direct_minus_derived']:+.2e} n={r['n_rows']} blk={r['n_blocks']} cat={r['category']} "
          f"p={r['p']:.4f} q={r.get('q_posthoc'):.4f}")
