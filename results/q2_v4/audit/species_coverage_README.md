# Species coverage table

`species_coverage.csv` is the per-species resource-coverage table underlying the
availability analysis (Table 7 and Figure 3). Each row is one species and records
whether a phylogenetic-distance vector and energy-budget (DEB) traits exist for it,
together with its record counts at field scale (raw ECOTOX, June 2026) and at
modelling scale (the ADORE processed mortality table).

## Row count (read this first)

The file has **3,274 rows** = the union of the field pool (E-full) and the modelling
corpus (P).

- **Filter `n_records_field > 0` to obtain the 3,268-species field pool (E-full)** —
  species with at least one qualifying LC50/EC50 record in a Water-habitat test in
  raw ECOTOX June-2026.
- The remaining **6 rows are P-only**: ADORE modelling species with no qualifying
  record in the raw 2026 field pool (`n_records_field = 0`, `in_adore_processed = 1`).

Applying this filter reproduces the manuscript's 3,268; the raw row count of 3,274 is
the union and is intentionally larger.

## Columns

| column | meaning |
|---|---|
| `species_number` | ECOTOX species id (stable join key) |
| `adore_species_number` | the ADORE species id this row belongs to; blank if the species is not in the ADORE corpus. For 26 taxa ADORE consolidated several ECOTOX ids into one `:`-joined key (see below) |
| `latin_name` | ECOTOX accepted Latin name |
| `resolved_name` | name used for resource matching (a synonym form where an ECOTOX synonym matched a resource list) |
| `ecotox_group` | raw ECOTOX `ecotox_group` field |
| `tax_group` | fish / crusta / algae / other, derived from `ecotox_group` |
| `n_records_field` | qualifying record count at field scale (E-full basis; column sum = 160,869) |
| `n_records_adore` | record count in the ADORE processed mortality table (P basis; column sum = 70,670) |
| `in_adore_processed` | 1 if in the 1,267-species ADORE processed corpus (P) |
| `in_adore_public` | 1 if in the 203-species ADORE public benchmark (A203) |
| `in_phylo` | 1 if a phylogenetic-distance vector exists (FCA_pdm matrix; for modelled species this is ADORE's `tax_pdm_available` flag) |
| `in_deb` | 1 if energy-budget (DEB) traits exist (AmP database; for modelled species this is ADORE's `tax_ps_ampv` flag) |

## Method

- **E-full (field pool).** Raw ECOTOX June-2026 records whose endpoint, effect,
  habitat and media values fall in the sets occupied by the ADORE mortality corpus
  (the P-derived definition recorded in `ecotox_expansion_v2.json`), obtained by
  joining `results` -> `tests` -> `validation/species` in the ECOTOX ASCII release.
- **`in_phylo` / `in_deb`.** Authoritative for the 1,267 modelled species (ADORE's
  precomputed `tax_pdm_available` / `tax_ps_ampv` flags); resource name-membership
  (FCA_pdm matrix, 853 species / AmP species list) for the remaining field species.
- **`tax_group`.** ECOTOX `ecotox_group` field (Fish / Crustaceans / Algae ->
  fish / crusta / algae; everything else -> other).

## Composite ADORE species (26)

ECOTOX June-2026 lists the following 26 taxa under several species ids
(subspecies / variety / synonym level) that the ADORE 2022 build had consolidated
into one species. Each such taxon therefore appears as several field rows (one per
component id, with differing subspecies names). `in_adore_processed = 1`,
`in_adore_public`, `n_records_adore` and the authoritative `in_phylo` / `in_deb`
flags are placed on the **primary component** (the component carrying the most field
records); every component row shares the same `adore_species_number`, so filtering on
that column returns all rows for the taxon. This is a property of the data: ECOTOX
2026 resolves these taxa more finely than the ADORE 2022 build did.

| adore_species_number | species |
|---|---|
| 16:2298 | Gambusia affinis |
| 21:1520:3020 | Cyprinus carpio |
| 34:11611:11783:32408:32409 | Oncorhynchus clarkii |
| 49:2385:4757 | Salmo trutta |
| 59:4632 | Morone saxatilis |
| 90:5780 | Oncorhynchus nerka |
| 98:11487 | Navicula seminulum var. hustedtii |
| 130:2120 | Platichthys flesus |
| 146:4752 | Leuciscus idus ssp. melanotus |
| 317:32059:32060 | Fragilaria capucina ssp. rumpens |
| 406:6060 | Procambarus acutus ssp. acutus |
| 437:17481 | Aulacoseira granulata var. angustissima |
| 479:17520 | Chlorella vulgaris |
| 497:1320 | Ankistrodesmus falcatus |
| 521:6061 | Procambarus simulans ssp. simulans |
| 643:1780 | Balanus amphitrite |
| 696:2350 | Austropotamobius pallipes ssp. pallipes |
| 727:5700 | Rasbora daniconius neilgeriensis |
| 851:1070 | Idotea balthica |
| 962:4594:11172 | Chlorella fusca var. vacuolata |
| 1260:1280 | Daphnia galeata |
| 1518:6064 | Rhodeus sericeus ssp. amarus |
| 2547:11660 | Oncorhynchus gilae ssp. apache |
| 2981:2982 | Melanotaenia splendida ssp. inornata |
| 2999:17854 | Scenedesmus acutus var. acutus |
| 20372:34553 | Panulirus homarus |

## Reproduce

`python scripts/build_species_coverage.py` regenerates
`results/q2_v4/audit/species_coverage.csv`. Set the environment variables `ECOTOX_ASCII_DIR`
(ECOTOX June-2026 ASCII release) and `ADORE_DIR` (the ADORE distribution) to your local
copies first; see the top of the script.

## Aggregate coverage (recomputed from this table)

| set | n_species | n_records | phylo (species) | phylo (record-wt) | DEB (species) | DEB (record-wt) |
|---|---|---|---|---|---|---|
| E-full | 3,268 | 160,869 | 17.7% | 55.7% | 7.6% | 51.2% |
| P | 1,267 | 70,670 | 41.5% | 75.3% | 18.8% | 68.3% |
| A203 | 203 | 44,964 | 100% | 100% | 77.3% | 97.7% |
| dropped | 1,064 | 25,706 | 30.4% | 32.2% | 7.6% | 16.9% |
| algae (field) | 460 | 15,167 | 15.2% | 20.7% | 0% | 0% |

Per-group (E-full) phylogenetic-distance / DEB coverage, species scale:
fish 63.5% / 31.8%, crustaceans 15.2% / 7.0%, algae 15.2% / 0%, other 0.1% / 0%.
