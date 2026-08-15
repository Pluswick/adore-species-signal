# ECOTOX 확장 — STEP 0 접근성 보고 (측정 전 게이트). 사실만. STEP A/B 미착수.

## 1. ECOTOX 원본 접근 = 가능 (로컬 전체 릴리스 존재)
- 경로: `<DATA_ROOT>\ecotox_ascii_06_11_2026\` (pipe `|` 구분 ASCII)
- 파일·레코드 수 (release_notes 기재 = 실측 일치):
  - `results.txt` **1,242,356행 · 137열** — result_id, test_id, endpoint, effect, obs_duration_mean/unit/op, conc1_mean/unit/op 전부 존재
  - `tests.txt` **724,501행 · 131열** — test_id, reference_number, test_cas, species_number, media_type, exposure_type, organism_habitat 존재
  - `validation/species.txt` **29,630종** (species_number, latin_name, kingdom..genus..species)
  - species_synonyms.txt 21,207 · chemicals.txt 18,534 · references.txt 131,732 · doses.txt 734,905 · chemical_carriers.txt 187,403 · media_characteristics.txt
- 별도 pull 스크립트 불필요(원본 dump 로컬 보유). ADORE 저장소 pull 스크립트는 로컬에 **없음**.
- ADORE `processed/before_aggregation/`(ecotox_results.csv 47MB, ecotox_species.csv=**1,684종**)은 **ADORE 스코프 부분집합**(전체 아님) — E-full은 위 ASCII dump에서 산출해야 함.

## 2. 릴리스 버전 = June 2026 (⚠ ADORE 기준과 불일치)
- 로컬 dump = "June 2026 - scheduled data update" (release_notes_06_11_26.txt).
- ADORE 파생 기준 = ECOTOX **2022-09** 릴리스 (Schür 2023).
- ⚠ 함의: E-full을 이 dump로 재면 **2022-09 이후 신규 등재분이 포함**. B1 = E-full − P 는 (a) ADORE가 스코프/필터로 제외한 레코드 + (b) 2022-09 이후 추가 레코드가 **혼재**. 분리하려면 (i) 2022-09 릴리스 별도 확보 또는 (ii) results/tests 등재일 필드로 컷. 2022-09 이후 신규분 카운트 = STEP A에서 산출(등재일 필드 확인 필요). **분리 정책 = director 결정 대기.**

## 3. 계통거리·DEB 소스 경로 + 종 키 형식 (⚠ underscore vs space)
- 계통거리: `adore_dataset/taxonomy/FCA_pdm_species.csv` (853×854 종×종 행렬) + `FCA_species.nwk` + `FCA_replaced_species.csv`(종명 치환맵). 키 = **UNDERSCORE** (`Abramis_brama`).
- DEB/형질: `adore_dataset/taxonomy/FC_amp_{pseudodata,lifehistory,ecology}.csv`. 키 = `latin_name` **SPACE** (`Pimephales promelas`).
- ECOTOX 원본 `species.txt` latin_name = **SPACE** (`Pimephales promelas`, underscore 없음).
- 우리 mortality `tax_gs` = **UNDERSCORE** (`Cyprinus_carpio`).
- ⚠ **키 형식 2계열**: {ADORE mortality, 계통거리 행렬} = underscore / {ECOTOX 원본, DEB} = space. ECOTOX 원본 ↔ 계통거리/우리 taxonomy 조인 시 space↔underscore 정규화 필수(과거 tier 축퇴 전례). 동의어 해소용 `species_synonyms.txt`(21,207) 병용 가능.

## 상태: STEP A/B 미착수. director 진행 승인 + 버전 분리 정책 대기.
