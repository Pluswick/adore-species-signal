# External reference scales — literature (director request; numbers + sources only)

> Descriptive reference material for placing "effect of species representation" next to independent
> yardsticks. NOT a decision input, NOT applied to any tier judgment. Assembled 2026-08-05 via web
> research + independent primary-source verification. Anti-selection-bias: full ranges reported; nothing
> chosen by our observed effect size. Unverifiable items are flagged, not asserted.

## Verification status (independently re-fetched by Claude Code, not just agent-reported)
- ✓ US EPA SAF table — Cornell LII 40 CFR 132 App. A
- ✓ EU TGD 2003 source identity + AF=1000 clause verbatim — JRC PDF (pdftotext)
- ✓ Kikuchi 2017 CV 13/15/62% — PubMed 27984774
- ✓ Hrovat 2009 "several orders of magnitude", 44 cpd/4654 rec — PubMed 19467285
- ✓ OECD GD69 §221 — OECD PDF (pdftotext)
- ✓ Sheffield & Judson 2019 RMSE 0.83/0.98, 81/76% — PMC7047609
- ✓ Reuschenbach 2008 69/64/60% factor-10 — PubMed 18262586
- (read directly from primary OECD PDFs by research agent, not re-fetched here) OECD TG 201/202/203
  validity criteria; Busquet 2014; OECD Validation Report No. 431 algae CVs; TGD Table 16 rows 100/50/10.

---

## (a) Inter-laboratory / within-laboratory variability of acute aquatic toxicity tests

"Ring test / round-robin" = designed inter-lab exercise (between-lab reproducibility, and within-lab
repeatability where multiple runs/lab). "Database/retrospective" = variability mined from historical
records (mixes biological + technical + between-lab; usually not cleanly separable). Every row states
factor-vs-CV and within-vs-between.

| Taxon / test | within/between | value | exact meaning | source (DOI) |
|---|---|---|---|---|
| Fish acute (ECOTOX, 96h LC50) | pooled/total (not separated) | "several orders of magnitude" (verbal; no single factor/CV) | retrospective DB, 44 cpd (≥10 rec), 4654 records; residual variability after controlling species/stage/temp/pH/hardness attributed to technical/measurement | Hrovat, Segner, Jeram 2009, Reg Tox Pharmacol 54(3):294-300; 10.1016/j.yrtph.2009.05.013 |
| Fish EMBRYO (ZFET, OECD TG 236 — NOT TG 203; adjacent assay) | both intra & inter | CV <30% (most); CV >30% (very toxic/volatile/near-solubility) | ring test, 48/96h LC50 zebrafish embryo; 20 chem×5 conc×3 runs×≥3 labs | Busquet et al. 2014, Reg Tox Pharmacol 69(3):496-511; 10.1016/j.yrtph.2014.05.018 |
| Daphnia (TG 202) | between-lab | 0.6–2.1 mg/L = factor ≈3.5 | reference substance K2Cr2O7 **24h** EC50 range, international ring tests + ISO 6341 Tech. Corrigendum (quoted in TG 202 footnote 1) | OECD 2004, Test No. 202 |
| Daphnia magna (TG 202) | within-lab (repeatability) | CV 13% | 48h EC50 K2Cr2O7, same dilution-water sample | Kikuchi et al. 2017, Chemosphere 170:113-117; 10.1016/j.chemosphere.2016.11.158 |
| Daphnia magna (TG 202) | between-lab (reproducibility) | CV 15% | 48h EC50 K2Cr2O7, six contract labs, standardized M4 medium | Kikuchi et al. 2017 (same) |
| Daphnia magna (TG 202) | across dilution waters (upper bound) | CV 62% | 48h EC50 K2Cr2O7 across seven dilution-water samples | Kikuchi et al. 2017 (same) |
| Algae R. subcapitata (TG 201) | between-lab | CV 23% (K2Cr2O7); 38% (3,5-DCP) | 72h EC50; ISO 8692 ring test 1980-81 | OECD Validation Report No. 431 (Yamagishi & Yamamoto 2025), Table 4-1 |
| Algae D. subspicatus (TG 201) | between-lab | CV 14% (K2Cr2O7); 37% (3,5-DCP) | 72h EC50; ISO 8692 ring test 1980-81 | same report, Table 4-1 |
| Algae M. permitis (TG 201) | between-lab | CV 33.3% (Cr); 21.2% (3,5-DCP); max factor 2.28/1.87 ("≈twofold") | 72h ErC50 diatom; new 5-lab ring test | same report, Table 3-1/3-2/4-1 |
| Algae (TG 201) control endpoint | within-lab | avg growth-rate CV 0.93–6.00%; section-by-section 13.3–41.6% | per-lab control-culture growth-rate CV (the quantity the validity criteria bound; NOT EC50) | same report |

OECD validity-criteria clauses (variability-related), verbatim-source:
- TG 203 (2025) §8: control mortality ≤10%; dissolved O2 ≥60% saturation; analytical measurement compulsory. **No reference-substance LC50 range / no CV limit.**
- TG 202 (2004) §6/footnote 1: control immobilisation ≤10%; O2 ≥3 mg/L; K2Cr2O7 24h EC50 "within the range 0.6 mg/l to 2.1 mg/l".
- TG 201 (2026 rev.) §12: control biomass ↑ ≥ factor 16 in 72h (≈0.92 day⁻¹); mean CV of section-by-section growth rate ≤35%; CV of whole-test average growth rate ≤7% (R. subcapitata/D. subspicatus), ≤10% (other species). (These bound CONTROL growth-rate precision, not EC50 reproducibility.)

Span across opened sources: CV ≈13% (within-lab Daphnia) → ≈62% (Daphnia across waters); fish DB total ≈ orders of magnitude.

## (b) Assessment factors (PNEC derivation)

**EU — TGD 2003 (EUR 20418 EN/2), Table 16 §3.3.1.1** (predecessor carried into ECHA R.10):

| AF | data-availability condition (verbatim) |
|---|---|
| 1000 | at least one short-term L(E)C50 from each of three trophic levels (fish, Daphnia, algae) |
| 100 | one long-term NOEC (fish or Daphnia) |
| 50 | two long-term NOECs from species representing two trophic levels |
| 10 | long-term NOECs from at least three species representing three trophic levels |
| 5–1 (case by case) | species sensitivity distribution (SSD) method |
| case by case | field data or model ecosystems |

Verbatim clause confirmed: "When only short-term toxicity data are available, an assessment factor of 1000 will be applied on the lowest L(E)C50…". "A factor of 10 cannot be decreased on the basis of laboratory studies."

**US EPA — 40 CFR 132 App. A (Great Lakes WQI), Table A-1 Secondary Acute Factors:**

| # minimum data requirements satisfied | SAF |
|---|---|
| 1 | 21.9 |
| 2 | 13.0 |
| 3 | 8.0 |
| 4 | 7.0 |
| 5 | 6.1 |
| 6 | 5.2 |
| 7 | 4.3 |

(8 satisfied ⇒ full Tier I Final Acute Value, no SAF.) Assumed acute-to-chronic ratio when <3 available: **18** (§XIII, verbatim). SCV = FAV/SACR (§XIV).

UNVERIFIABLE: ECHA R.10 (Table R.10-4) — Azure WAF blocked WebFetch/curl/browser; not opened. Anchored to TGD 2003 predecessor instead.

## (c) QSAR / predictive-model acceptable error

**Answer: No established REGULATORY numeric acceptance criterion.** OECD validation frameworks are qualitative/context-dependent. A de facto LITERATURE convention exists: prediction within a factor of 10 (≈ one order of magnitude ≈ 1 log unit).

- OECD GD69 (No.69, ENV/JM/MONO(2007)2) §221 verbatim: "It is not the aim of this document to define acceptability criteria for the regulatory use of QSAR models, since the use of data in decision-making is highly context-dependent."
- Reuschenbach et al. 2008, Chemosphere 71(10):1986-95 (10.1016/j.chemosphere.2007.12.006): fish/Daphnia/algae correct 69%/64%/60% "when a tolerance factor of 10 was allowed."
- Sheffield & Judson 2019, ES&T 53(21):12793-802 (10.1021/acs.est.9b03957): LC50/NOEC "within one order of magnitude 81% and 76% of the time … RMSEs of roughly 0.83 and 0.98 log10(mg/L)."
- ECHA Practical Guide 5 (2016): model-level rule of thumb R² ≥ ~0.7 (not a per-prediction error limit).
- Range coexisting: factor-of-10 / ~1 log unit most common; good-model RMSE ≈0.8–1.0 log; ECHA R²≥0.7; stricter per-study factors 2–5 (~0.3–0.7 log) appear but none standardized.

## Excluded / unverifiable (full list)
- ECHA R.10 PDF — Azure WAF 403 (all routes). UNVERIFIABLE.
- Persoone et al. 2009 KMAE Daphnia review (10.1051/kmae/2009012) — 403 all routes. Snippet numbers (intra CV ~23.7%, inter 43%/21%) NOT asserted. UNVERIFIABLE.
- ECETOC TR-134 — 404. UNVERIFIABLE.
- eCFR canonical 40 CFR 132 — captcha redirect; used Cornell LII verbatim instead.
- ScienceDirect abstracts (Hrovat, Busquet) 403 → substituted PubMed (opened).
- US 1985 National AWQC guidelines, US ACR=10 acute/chronic — not a QSAR criterion; not conflated.
- Secondary/derivative pages (chemsafetypro, chemradar, vendor/CRO) — not primary; not cited.
