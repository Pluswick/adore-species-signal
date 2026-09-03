# Mechanism fact-finding (director; desk/compute, GPU 0). Facts only — no interpretation.

> Parts 1/2/7 = documentation + architecture + provenance (verbatim). Parts 3–6 = computations in
> `mechanism_facts_compute.{json,txt}`. Not tier judgments; used in no decision. Existing files untouched.

## PART 1 — FiLM / fusion-locus records

**Injection loci actually experimented** (CC-MPNN study; dataset = **tox-learn**, Yuan et al. bioRxiv 2025.11.24.690199, repo mgtools/tox-learn — CC-MPNN/HANDOFF.md:63; **NOT ADORE**):
- "계층 1 (fusion locus): GBDT / plain MPNN(맥락 없음) / late-fusion / FiLM / CC-MPNN(결합 조건화)" (CC-MPNN/codex.md:8; also project_decisions.md:174).
- FiLM = "readout 채널 변조" (project_decisions_v2.md:28); "FiLM (채널 단위 변조) | 조건화 방식 비교 | 필수" (project_decisions.md:157).
- The current `src/models.py` additionally implements `early_injection` and `message_level` loci (lines 181–195), plus `film` (lines 189–195, 231–233).

**Result numbers** (CC-MPNN `data/sweep_summary.csv`, column `A.rmse`, mean±sd over 5 seeds):
- random split: gbdt 0.8307±0.0033 · late-embed 0.8992±0.0078 · **film-embed 0.9196±0.0112** · none(plain) 0.9283±0.0066.
- scaffold split: late-embed 1.1023±0.0282 · gbdt 1.1134±0.0125 · **film-embed 1.1322±0.0231** · none(plain) 1.1832±0.0340.
- (16-variant bond-conditioning grid also present; e.g. random best `ccmpnn-bilinear-gating-onehot` 0.9273±0.0093.)
- Per-seed run files: `film-embed_{random,scaffold}_s0..s4.json` (5 seeds × 2 splits).

**FiLM "abandonment" reason** — no explicit performance-based abandonment statement found. FiLM was a **mandatory comparison / fusion-locus ablation arm** ("필수", project_decisions.md:157; "**FiLM 행**... 메인표 6행×2split 완성", HANDOFF.md:47), kept in the main table; the **proposed** method was CC-MPNN bond-level conditioning, and "GNN 비교군 4종(없음/late/FiLM/결합조건화)은 같은 D-MPNN 토대·readout·FFN을 공유하고 맥락 합류 위치(fusion locus)만 다른 ablation 변형" (project_decisions.md:162).
- **ADORE exclusion** (namespace/scope, not performance), verbatim `results/q2_v4/audit/adore_tier_namespace.md:24`: "**film = 구 fusion-locus tier 5 → Phase 1에서 사용 안 함(ADORE t5=phylo는 성능 실험 제외). 학습 variant 목록에서 제외.**"

## PART 2 — current architecture interaction (src/models.py)

**Where species enters** (ADORE main tiers t2/t3a/t3b/t4 = readout/global concat; t1 = post-readout scalar):
- GraphConv: t4 late_fusion `H = torch.cat([H, species_emb(species_idx)], dim=1)` (models.py:235); t2 categorical `H = torch.cat([H, one_hot(species_idx)], dim=1)` (:237); t3a/t3b `H = torch.cat([H, tax_vec], dim=1)` (:239). Readout FFN `_build_ffn` (:214, def :83): ffn_layers=2 → `Linear(readout_dim→hidden) → ReLU → Dropout → Linear(hidden→1)` (:86–89).
- D-MPNN (ccmpnn): t4 `VariantConfig(fusion="late", species_repr="embed")` (:471); t2 `fusion="late", species_repr="onehot"` (:477); t3a/t3b `DMPNNTaxonomyContext.apply_global` → `torch.cat([H, vec], dim=1)` (:382).

**Combination method** = **concat (연결) then a nonlinear readout FFN** (Linear→ReLU→Linear), NOT plain addition. Exception (not used in ADORE): FiLM = multiplicative `H = gamma * H + beta` (models.py:233).

**Interaction structure** (fact): for t2/t3a/t3b/t4 the molecular vector H is computed **without species** (species does not enter message passing for these tiers — only early_injection/message_level/film touch message passing, none of which are ADORE main tiers). Species is concatenated to the fixed molecular vector; the two therefore mix **only inside the readout FFN's single hidden ReLU layer**. So species can interact with molecular features, but only at the readout FFN — the molecular embedding itself is species-independent for these tiers.

**tier1 vs tier2** (representational relation, facts):
- tier1 `SpeciesBiasOnlyModel` (models.py:397–411): `base_model(bmg) + species_bias(species_idx)`, `species_bias = nn.Embedding(n_species, 1)` → a per-species **scalar** added to the final scalar output (init zeros).
- tier2 categorical: per-species **one-hot** (dim n_species) concatenated to H, then the nonlinear readout FFN.
- Both index by the **same input** (species identity). tier2 gives each species a learned contribution into the readout hidden layer (first FFN layer: n_species × hidden params); tier1 gives each species one learned scalar added post-readout (n_species params) → tier2 has strictly more per-species degrees of freedom.
- Exact architectural nesting: tier2 does **not** reduce to tier1 as an exact weight-setting subcase — tier1's scalar is added **after** the nonlinear readout (bypasses the FFN) whereas tier2's species term passes **through** the shared ReLU readout. (With sufficient FFN width tier2 can *approximate* a molecule-head-plus-additive-species-bias function; that is approximation, not exact architectural containment.)

## PART 7 — ADORE provenance & ECOTOX relationship

**Primary source (verified verbatim, PMC10584858):** Schür, C., Gasser, L., Perez-Cruz, F., Schirmer, K., & Baity-Jesi, M. (2023). "A benchmark dataset for machine learning in ecotoxicology." *Scientific Data* 10, 718. **DOI 10.1038/s41597-023-02612-2.** Data repo (paper-cited): Eawag ERIC DOI 10.25678/0008C9; code/data: renkulab.io/gitlab/mltox/adore.

- Data source (verbatim): "The main source of our dataset is the ECOTOX database from the United States Environmental Protection Agency (US EPA)."
- ECOTOX release (verbatim): "Our dataset is based on the release from September 2022, which contains over 1.1 million entries of more than 12,000 chemicals and close to 14,000 species."
- Record key (verbatim): "Each data point is uniquely identified by the result_id."
- Filters (paper, agent-extracted): files species/tests/results/media joined via keys (e.g. species_number); endpoints filtered to **LC50 and EC50**; effect = mortality-comparable; durations **24/48/72/96 h**; exposure types **S/F/R/NR**; units kept = mass/molar → unified to **mg/L**; extreme QC (EC50/LC50 ≥1e5 or ≤1e-5 mg/L removed when orders-of-magnitude disagreement); min–max entries averaged if within one order of magnitude; "**These filtering steps remove around 50% of the processed data**."
- Final released scope (Table 4): **33,448 data points, 2,408 chemicals, 203 species**; taxa fish/crustaceans/algae (fish 26,114 / crusta 6,630 / algae 704). Challenge prefixes a-/t-/s- (whole/within-group/single-species); 5-fold CV splits (random, by-compound, by-occurrence, scaffold murcko/generic, leave-one/last-out).

**Separability (facts):** `result_id` = ECOTOX's native per-result key = primary separation key against a fresh ECOTOX pull; raw files shipped in the repo; fallback composite key = CAS + species + endpoint + duration + effect-concentration(mg/L); temporal cutoff = ECOTOX **September 2022** release (post-release records cannot be in ADORE). UNVERIFIABLE (not checked): whether `result_id` is physically retained as a column in the released tables (repo files not fetched); no finer extraction date than "September 2022 release."

**⚠ Factual discrepancy with our working file:** our pipeline's `ADORE\processed\ecotox_mortality_processed.csv` = **70,670 records / 1,267 species / 3,295 compounds** (HANDOFF §C, measured; `_filtered.csv` 66,896), versus the paper's **final** 33,448 / 203 species / 2,408 chemicals. Our discovery/replication partitions carry **779 / 1,006** species. The paper's "~50% attrition of the processed data" (70,670 → ~33k) reconciles the record counts: our file is the **pre-final-filter processed** ADORE artifact; the 33,448/203-species figure is **post-final-filter**. Our experiment applies its own censoring (§1/D16) to the processed file, so its species coverage exceeds the paper's headline set.
