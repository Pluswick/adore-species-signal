# ADORE stage verification — B-1 / B-3 provenance facts (verbatim). Facts only.

> Companion to compute files `mechanism_supplement2.{json,txt}` (A, B-2, B-4) and `mechanism_b5_solubility.json` (B-5).
> Source = Schür et al. 2023, Scientific Data 10:718, DOI 10.1038/s41597-023-02612-2 (PMC10584858; cross-verified
> against Europe PMC REST full-text XML — identical wording). Not a decision input.

## B-1. Can the ADORE final dataset / 203 species be specified?
- YES. The ADORE package ships the final per-challenge modeling files in `adore_dataset/processed/`:
  whole-dataset `a-FCA2FCA_mortality.csv` (+ `a-CA2F-same/diff`), within-group `t-{A2A,C2C,F2F}`, single-species `s-*`.
- **Final species universe = 203** (measured: union of `tax_gs` across the a-/t- challenge files = 203; matches paper Table 4). The 203-species list is directly extractable from `a-FCA2FCA_mortality.csv`.
- `a-FCA2FCA_mortality.csv` = 66,896 rows (pre-aggregation), 203 species; paper's 33,448 "points" = post-aggregation.
- **ECOTOX keys retained** (README, verbatim): "For the sake of retraceability, we retain the ECOTOX test and result id for each entry, which allows users to check entries against ECOTOX." (`test_id`, `result_id` present as columns — confirmed.)

## B-3. Final-filter criteria (verbatim) + species-reduction rule

**Complete filter-step list (verbatim, paper Methods):**
- Taxonomic + endpoint gate: "we filtered the data to only contain acute mortality experiments for the three taxonomic groups fish, crustaceans and algae. … The endpoints were filtered to only include LC50 and EC50."
- Duration: "We converted them to hours and retained 24, 48, 72, and 96 hours experimental periods … Only experimental periods up to 96 hours were included."
- Exposure type: "We only retain the exposure types that are most common and ensure consistent experimental designs (Static (S…), flow through (F…), renewal (R…), and not reported aquatic experiments (NR)."
- Medium: "This feature was reduced to only contain fresh water and salt water experiments, also with the intention to remove tests conducted in environmental water samples that are basically non-reproducible and could have an effect by themselves."
- Units: "We only kept the units related to a mass or molar concentration and unified them to mg/L, from which we calculated the molar concentration in mol/L."
- Min–max averaging: "Other entries do not contain a mean effect value but a minimum and a maximum value … We averaged these two values if they were within one order of magnitude."
- Repeated measurements: NOT deduplicated — "Thus, we included all such data points. If needed, one can filter to the latest data point for an experiment that is uniquely identified by the reference number."
- Extreme-value QC: "Some toxicity values were found to be either very high or very low, i.e., EC50/LC50 ≥10^5 mg/L and ≤10^−5 mg/L." … "We considered experimental settings with at least 25 repetitions and entries if their toxicity value was outside 3 times the interquartile range of the first and third quartile (based on the 'outlier' definition of boxplots)." … "If that was also not available or the values differed by several orders of magnitude, the data point in question was removed."
- Net: "These filtering steps remove around 50% of the processed data and leaves us with a dataset comprised of predominantly organic chemicals."

**Species-reduction rule (1,267 → 203):** **NO EXPLICIT PER-SPECIES MINIMUM-COUNT RULE STATED** (cross-verified, PMC + Europe PMC REST XML; also confirmed by direct WebFetch: "no explicit minimum number of records per species is stated"). The species reduction is a **side-effect of the record-level filters** (taxonomic restriction, LC50/EC50-only, ≤96 h, exposure types, fresh/salt water, mg/L units, extreme-value QC). The only counting threshold anywhere is the QC outlier rule ("at least 25 repetitions … outside 3× IQR"), which is not a per-species keep/drop rule. Authors' stated data-retention philosophy: "the bottleneck for environmental risk assessment models is usually the scarcity of data, we prefer a bigger dataset to a larger feature space."

**Per-filter rationale (verbatim where present):**
- LC50/EC50: "The use of the lethal concentration 50 (LC50) … is very common and allows for comparison of toxicity across chemicals and species." / EC50 "analogously describes the induction of a 50% effect level." (no separate 'why exclude others' argument).
- Durations: grounded in OECD 203 (fish 96h)/202 (crustacea 48h)/algae 72h.
- Exposure types: "most common … consistent experimental designs."
- Fresh/salt water: remove non-reproducible environmental-water tests.
- mg/L units: NOT STATED (conversion described, no reason given).
- Extreme-value cut: no standalone motivational sentence beyond values being "either very high or very low"; justified operationally by the cross-check-then-remove logic.

**Challenge subsetting:** a-/t-/s- = whole-dataset / within-group / single-species. a-FCA2FCA = all 203 species. t- restrict to one group (fish≈140, crustacea≈17, algae≈46 species per helper-read Table 4 — treat per-group split as helper-reported). Per-challenge species/point counts in Table 3 body = NOT retrievable (PMC table HTML behind bot wall).
