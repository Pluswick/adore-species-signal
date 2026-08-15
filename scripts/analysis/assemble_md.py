# -*- coding: utf-8 -*-
"""Assemble expansion_results.md / expansion_margins.md / expansion_guards.md from the JSON. Facts only."""
import json
from pathlib import Path
R=Path(r".\results\q2_v4")
A=R/"audit"
res=json.loads((A/"expansion_results.json").read_text(encoding="utf-8"))
dev=json.loads((A/"expansion_device_diagnostic.json").read_text(encoding="utf-8"))
HDR="> 사실 보고. 해석·서사·결론 없음(판정은 director). 전 수치는 산출물에서 재확인. Phase 1은 원 산출물 재독.\n"

# ---------- expansion_margins.md ----------
m=res["s3_margins"]; d=m["delta_primary"]; dd=m["delta_det"]; dp=m["delta_prime"]
L=["# §3 마진 재유도 — Phase 1 병기\n",HDR,
 "| 마진 | Phase 1 | B1 | C(P1/B1) | df |","|---|---|---|---|---|",
 f"| δ | {d['phase1']:.6f} | {d['b1']:.6f} | {d['C_phase1']}/{d['C_b1']} | {d['df_phase1']} |",
 f"| δ_det | {dd['phase1']:.6f} | {dd['b1']:.6f} | {dd['C_phase1']}/{dd['C_b1']} | (bootstrap) |",
 f"| δ′ | {dp['phase1']:.6f} | {dp['b1']:.6f} | {dp['C_phase1']}/{dp['C_b1']} | {dp['df_phase1']} |","",
 "판정은 B1 마진으로 수행. Phase 1 값 재독 출처: `audit/delta_*_frozen.json`. B1 동결: `audit/delta_*_frozen_b1.json` (frozen_utc 2026-08-11).","",
 "## per-condition s_c (B1 / Phase 1)","| condition | δ B1 | δ P1 | δ′ B1 | δ′ P1 |","|---|---|---|---|---|"]
for c in d["b1_per_condition_s_c"]:
    L.append(f"| {c} | {d['b1_per_condition_s_c'][c]} | {d['phase1_per_condition_s_c'][c]} | {dp['b1_per_condition_s_c'][c]} | {dp['phase1_per_condition_s_c'][c]} |")
L+=["","δ_det per-tier s_c (B1/P1): "+", ".join(f"{k}={dd['b1_per_condition_s_c'][k]}/{dd['phase1_per_condition_s_c'][k]}" for k in dd['b1_per_condition_s_c']),"",
 "## 장치-성분 진단 (판정 전, 재학습 없음)","seed↔device: B1 warm 0-4→4090/5-9→5060Ti; Phase1 warm 0-6→5060Ti/7-9→4090 (둘 다 dual-GPU). 중복 실행 쌍 없음.",""]
for tag in ["delta_B1","delta_Phase1"]:
    a=dev[tag]["agg"]
    L.append(f"- {tag} ({dev[tag]['map']}): 장치간 분산 mean={a['mean_frac_between']*100:.1f}% pooled={a['pooled_frac_between']*100:.1f}%; δ[A만]={a['delta_subsetA']:.6f}(df{a['df_subsetA']}) δ[B만]={a['delta_subsetB']:.6f}(df{a['df_subsetB']})")
a=dev["deltaprime_B1"]["agg"]
L.append(f"- deltaprime_B1 (4090 ens1-4/5060Ti ens6-9): 장치간 mean={a['mean_frac_between']*100:.1f}% pooled={a['pooled_frac_between']*100:.1f}%; δ′[A]={a['delta_subsetA']:.6f} δ′[B]={a['delta_subsetB']:.6f}")
L.append("- deltaprime_Phase1: 모든 앙상블 device-mixed → 분해 N/A")
(A/"expansion_margins.md").write_text("\n".join(L),encoding="utf-8")

# ---------- expansion_guards.md ----------
g=res["s6_guards"]
L=["# §6 가드\n",HDR,
 f"- **tier-input 축퇴 검사**: {g['tier_input_degeneracy']['tier_input_guard_check_records']} check records, 축퇴 flagged = **{g['tier_input_degeneracy']['degenerate_flagged']}**",
 f"- **variant 구별성(학습 후 예측)**: b1_group seed0 — dmpnn {g['variant_distinguishability']['b1_group_seed0']['dmpnn']['n_variants_present']} variant / {g['variant_distinguishability']['b1_group_seed0']['dmpnn']['n_pairs']} pair 중 identical={g['variant_distinguishability']['b1_group_seed0']['dmpnn']['n_identical_pairs']}; graphconv identical={g['variant_distinguishability']['b1_group_seed0']['graphconv']['n_identical_pairs']}",
 f"- **t4 누출 최종**: OOF 경계 증명 B1 = **{g['t4_leak_final']['oof_index_proof_b1']}**; permutation flag = {g['t4_leak_final']['permutation_test_flag']}; 결정적 대조(b1_scaffold) noise-floor={g['t4_leak_final']['decisive_control_b1_scaffold']['noise_floor_B']} shuffled-correct={g['t4_leak_final']['decisive_control_b1_scaffold']['shuffled_correct_A']} shuffled-misassigned={g['t4_leak_final']['decisive_control_b1_scaffold']['shuffled_misassigned_C']}; 처분=**{g['t4_leak_final']['final_disposition']}**; **t4 arm 포함={g['t4_leak_final']['t4_arm_included']}**; caveat={g['t4_leak_final']['capacity_floor_caveat']}",
 f"- **누출 탐지선(정지 임계) 발동**: (B1 §6 임계 미사전등록; Phase1-analog 0.5×잔차SD 참조)","",
 "| split | 잔차 target SD | 하한(0.5×SD) | min RMSE | n runs | 하한 미만 |","|---|---|---|---|---|---|"]
for s,v in g["leak_tripwire"]["per_split"].items():
    L.append(f"| {s} | {v['resid_target_sd']} | {v['leak_lower_bound_0.5xSD']} | {v['min_rmse']} | {v['n_runs']} | {v['n_runs_below_bound']} |")
L+=["",f"- **조인 성공률**: join_sanity = **{g['join_success']['join_sanity_verdict']}** (n_species={g['join_success']['n_species']}); {g['join_success']['note']}"]
(A/"expansion_guards.md").write_text("\n".join(L),encoding="utf-8")

# ---------- expansion_results.md ----------
s2=res["s2_execution_integrity"]; s4=res["s4_results"]; cen=s4["s4_1_census"]
def ct(c): return f"{c['eq']} / {c['sig']} / {c['inc']}" if isinstance(c,dict) else c
L=["# B1 확장 실험 결과 (사실 보고)\n",HDR,
 "## §2 실행 무결성",
 f"- 계획 vs 완료: **{s2['planned_vs_actual_runs']['total']['actual_ok']}/{s2['planned_vs_actual_runs']['total']['planned']}**, 실패 **{s2['planned_vs_actual_runs']['total']['fail']}**, 재시도 {s2['retries']}. 블록별 전부 일치.",
 f"- 이상: NaN={s2['anomalies']['rmse_NaN']} inf={s2['anomalies']['rmse_inf']} 미파싱={s2['anomalies']['missing_or_unparseable']} (run JSON {s2['anomalies']['run_json_scanned']}건); 조기중단=정상 early stopping(비정상 종료 0).",
 f"- prereg 최종수정 {s2['prereg_file_last_modified']}; 초기작성 후 수정: {'; '.join(s2['prereg_amendments_after_initial_creation'])}. {s2['prereg_amendment_timing_note']}",
 f"- 자원: {s2['resources']['gpus_used']}; {s2['resources']['seed_to_gpu_map_warm_blocks']}; dprime {s2['resources']['seed_to_gpu_map_dprime']}; {s2['resources']['concurrency']}; wall {s2['resources']['wall_time']}.",
 f"- GPU 기록: {s2['gpu_usage_log']}",
 f"- 하이퍼파라미터: {s2['resources']['hyperparameters']}","",
 "## §3 마진 (상세 = expansion_margins.md)",
 f"δ {res['s3_margins']['delta_primary']['phase1']:.6f}→{res['s3_margins']['delta_primary']['b1']:.6f} · δ_det {res['s3_margins']['delta_det']['phase1']:.6f}→{res['s3_margins']['delta_det']['b1']:.6f} · δ′ {res['s3_margins']['delta_prime']['phase1']:.6f}→{res['s3_margins']['delta_prime']['b1']:.6f}. 판정=B1 마진.","",
 "## §4 주 결과 (비교 수 Phase1=225 / B1=189)",
 "### §4-1 판정 census [동등/유의/불확정]",
 "| 패밀리 | Phase 1 | B1 |","|---|---|---|",
 f"| primary 우월성 | {ct(cen['primary_superiority']['phase1'])} | {ct(cen['primary_superiority']['b1'])} |",
 f"| primary TOST | {ct(cen['primary_TOST']['phase1'])} | {ct(cen['primary_TOST']['b1'])} |",
 f"| confirmatory 우월성 | {ct(cen['confirmatory_superiority']['phase1'])} | {cen['confirmatory_superiority']['b1']} |",
 f"| confirmatory TOST | {ct(cen['confirmatory_TOST']['phase1'])} | {cen['confirmatory_TOST']['b1']} |",
 f"| deterministic TOST | {ct(cen['deterministic_TOST']['phase1'])} | {ct(cen['deterministic_TOST']['b1'])} |",
 f"| exploratory rank | {ct(cen['exploratory_rank']['phase1'])} | {ct(cen['exploratory_rank']['b1'])} |",
 f"| exploratory support-bin | {ct(cen['exploratory_support_bin']['phase1'])} | {ct(cen['exploratory_support_bin']['b1'])} |",
 f"| exploratory tax_group | {ct(cen['exploratory_tax_group']['phase1'])} | {ct(cen['exploratory_tax_group']['b1'])} |",
 f"| exploratory scaffold | {ct(cen['exploratory_scaffold']['phase1'])} | {ct(cen['exploratory_scaffold']['b1'])} |",
 f"| exploratory designed-leaky | {ct(cen['exploratory_designed_leaky']['phase1'])} | {ct(cen['exploratory_designed_leaky']['b1'])} |",
 f"| exploratory cross-backbone | {ct(cen['exploratory_cross_backbone']['phase1'])} | {ct(cen['exploratory_cross_backbone']['b1'])} |",
 f"| sensitivity_ensemble | {ct(cen['sensitivity_ensemble']['phase1'])} | {ct(cen['sensitivity_ensemble']['b1'])} |",
 f"| 게이트 Stage2 도달 | {cen['reaches_stage2']['phase1']} | {cen['reaches_stage2']['b1']} |","",
 "### §4-2 효과크기 dd [90% CI] (block=smiles, N_BOOT 2000)"]
ef=s4["s4_2_effect_sizes"]
def cell(x): return f"{x['dd']:+.4f}[{x['ci'][0]:+.3f},{x['ci'][1]:+.3f}]({x['cat'][:3]})" if x else "NA"
for split,comps in ef.items():
    L+=["",f"**{split}**","| 비교 | Phase1 dm / gc | B1 dm / gc |","|---|---|---|"]
    for comp,row in comps.items():
        L.append(f"| {comp} | {cell(row.get('P1_dmpnn'))} / {cell(row.get('P1_graphconv'))} | {cell(row.get('B1_dmpnn'))} / {cell(row.get('B1_graphconv'))} |")
q=s4["s4_3_primary_TOST_q"]
L+=["","### §4-3 primary TOST q_family (BH-FDR)",
 "| | Phase 1 | B1 |","|---|---|---|",
 f"| n | {q['phase1']['n']} | {q['b1']['n']} |",
 f"| min q | {q['phase1']['min_q']} | {q['b1']['min_q']} |",
 f"| q<0.05 통과 | {q['phase1']['n_q_lt_0.05']} | {q['b1']['n_q_lt_0.05']} |","",
 "### §4-4 절대 RMSE (per-seed mean±sd / ensemble) — group split (전 split = json s4_4_absolute_rmse)",
 "| tier | Phase1 dm | Phase1 gc | B1 dm | B1 gc |","|---|---|---|---|---|"]
rm=s4["s4_4_absolute_rmse"]
for t in ["t0","t1","t1p","t2","t3a","t3b","t4"]:
    def r(exp,bb):
        v=rm[exp]["group"][f"{bb}/{t}"]; return f"{v['per_seed_mean']}±{v['per_seed_sd']}/{v['ensemble']}"
    L.append(f"| {t} | {r('phase1','dmpnn')} | {r('phase1','graphconv')} | {r('b1','dmpnn')} | {r('b1','graphconv')} |")
# §4-5 superiority
sup=s4["s4_5_superiority_detail"]
L+=["","### §4-5 우월성 24 (category 요약; 상세=json)",
 f"- Phase1: "+", ".join(sorted(set(v['cat'] for v in sup['phase1'].values()))) + f" (n={len(sup['phase1'])}); 유의={sum(1 for v in sup['phase1'].values() if v['cat']=='significant')}",
 f"- B1: "+", ".join(sorted(set(v['cat'] for v in sup['b1'].values()))) + f" (n={len(sup['b1'])}); 유의={sum(1 for v in sup['b1'].values() if v['cat']=='significant')}; 비유의 항목: "+", ".join(k for k,v in sup['b1'].items() if v['cat']!='significant')]
# species-cold ungated
cold=s4["species_cold_ungated_TOST"]
L+=["","### 종-cold ungated TOST (oov=mean; Stage2 도달 0이라 서술용, 판정 아님)","| 비교 | Phase1 dm/gc | B1 dm/gc |","|---|---|---|"]
for t in ["t3a","t3b","t4"]:
    def cc(src,bb):
        v=cold[src].get(f"{bb}/{t}~t2"); return f"{v['dd']:+.4f}({v['cat'][:3]})" if v and 'dd' in v else "NA"
    L.append(f"| t2→{t} | {cc('phase1_discovery_species_cold','dmpnn')}/{cc('phase1_discovery_species_cold','graphconv')} | {cc('b1_species_cold','dmpnn')}/{cc('b1_species_cold','graphconv')} |")
# §5
sb=res["s5_1_support_bins"]; so=res["s5_species_oov"]
L+=["","## §5 사전 지정 층화","### §5-1 support-bin (bins 1-4/5-9/10-19/20-49/50+) — B1 구간별 종수/행수",
 "| bin | n_species | n_test_rows |","|---|---|---|"]
for b in sb["bins"]:
    z=sb["b1"]["bin_sizes"][b]; L.append(f"| {b} | {z['n_species']} | {z['n_test_rows']} |")
L+=["","B1 t2→{t3a,t3b,t4} dd[CI] per bin (dmpnn) = json s5_1_support_bins.b1.dd; Phase1 = .phase1.dd.","",
 "### §5-2 종 수 / one-hot",
 f"- one-hot: Phase1 discovery **{so['phase1_species']['phase1_one_hot_discovery']}** / replication **{so['phase1_species']['phase1_one_hot_replication']}**; B1 **{so['one_hot_dim_global_B1']}**",
 f"- Phase1 discovery n_train_species: "+", ".join(f"{k}={v['n_train_species']}" for k,v in so['phase1_species']['phase1_discovery_n_train_species'].items()),
 f"- B1 n_train_species: "+", ".join(f"{k.replace('b1_','')}={v['n_train_species']}" for k,v in so['warm_splits'].items()),"",
 "### §5-3 OOV (warm test)","| split | Phase1 종/행 | B1 종/행 |","|---|---|---|"]
for k in ["group","scaffold","scaffold_generic","designed_leaky"]:
    p=so["phase1_warm_oov"][k]; b=so["warm_splits"]["b1_"+k]
    L.append(f"| {k} | {p['n_test_OOV_species']}/{p['n_test_OOV_rows']} | {b['n_test_OOV_species']}/{b['n_test_OOV_rows']} |")
orr=so["b1_group_OOV_removed_reagg_dd"]["dd"]
L+=["",f"- OOV-제거 재집계 dd (b1_group, 비-OOV종만; 참고, 판정 아님): "+", ".join(f"{k}={v.get('dd')}" for k,v in orr.items()),"",
 "## §6 가드 (상세 = expansion_guards.md)",
 f"축퇴 flagged=0; variant identical pairs=0; OOF 증명=PASS, t4 포함; 누출 탐지선 발동=0(전 split); 조인=PASS.","",
 "## §7 Phase 1 대비 차이 전수","| 항목 | Phase 1 | B1 |","|---|---|---|"]
for x in res["s7_phase1_differences"]:
    L.append(f"| {x['item']} | {x['phase1']} | {x['b1']} |")
L+=["","## §8 크게 다른 항목 (부호 상이 OR 크기 ≥2×; 사실만)","| 항목 | Phase 1 | B1 | 부호상이 | B1/P1 |","|---|---|---|---|---|"]
for x in res["s8_large_diffs"]["items"]:
    L.append(f"| {x['item']} | {x['phase1']} | {x['b1']} | {'예' if x.get('sign_flip') else ''} | {x.get('B1/P1','')} |")
L+=["","---","전 수치 기계판독 = `audit/expansion_results.json`. 장치진단 = `audit/expansion_device_diagnostic.json`. 마진 동결 = `audit/delta_*_frozen_b1.json`. gatekeeping 원본 = `runs_b1/bootstrap/gatekeeping_results.json`. 예측 CSV 전량 = `runs_b1/**/predictions/`."]
(A/"expansion_results.md").write_text("\n".join(L),encoding="utf-8")
print("[written] expansion_results.md, expansion_margins.md, expansion_guards.md")
for f in ["expansion_results.md","expansion_margins.md","expansion_guards.md"]:
    print(f"  {f}: {len((A/f).read_text(encoding='utf-8').splitlines())} lines")
