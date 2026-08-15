# Readout structure — GraphConv vs D-MPNN (code read, GPU 0). Facts + code locations only. No verdict.

> Not a decision input. Existing outputs/prereg untouched.

## 1. Readout path construction (side by side)

| | GraphConv | D-MPNN (ccmpnn) |
|---|---|---|
| forward | `GraphConvModel.forward` `models.py:216-245`: pool atoms→H[n_mols,300]; concat species; `self.ffn(H)` | `DMPNNModel.forward` `CC-MPNN/ccmpnn/model.py:41-44`: `H=encoder(bmg)`[n_mols,300]; `H=context.apply_global(H)`; `self.ffn(H)` |
| FFN builder | `_build_ffn` `models.py:83-90` | `_build_ffn` `CC-MPNN/ccmpnn/model.py:17-24` — **byte-identical code** |
| FFN layers (ffn_num_layers=2) | `Linear(in_dim→300) → ReLU → Dropout(0.1) → Linear(300→1)` | `Linear(in_dim→300) → ReLU → Dropout(0.1) → Linear(300→1)` |
| hidden width (ffn_hidden) | 300 (= hidden) | 300 (config.ffn_hidden=None→hidden=300, `config.py:25,39-40`) |
| activation / dropout / norm | ReLU / Dropout 0.1 / **no normalization** | ReLU / Dropout 0.1 / **no normalization** |
| nonlinear layers AFTER species concat | **1** (the single ReLU hidden layer) | **1** (same) |
| species concat point | readout, `H=torch.cat([H, vec],1)` `models.py:234-239` | readout, `apply_global` `variants.py:153-155` (late/onehot) / `models.py:375-382` (taxonomy) |
| `in_dim` (readout input) | `readout_dim` `models.py:209-213` | `context.out_dim(hidden)` `model.py:34` |

Encoder differs by design (GraphConv `GraphConvLayer` `models.py:93-118` vs D-MPNN `MPNEncoder`); the READOUT FFN code and config are identical.

## 2. Species–molecule mixing / degeneracy

| tier | species path (both backbones identical) | linear-only? |
|---|---|---|
| t1 (species_bias_only) | `SpeciesBiasOnlyModel` `models.py:397-411`: `base_model(bmg) + species_bias(idx)`, `species_bias=Embedding(n_species,1)`; added **AFTER** the FFN (post-readout scalar) | **YES — additive scalar offset** (species bypasses the FFN nonlinearity, both backbones) |
| t2 (categorical/onehot) | one-hot(dim n_species) concat → FFN | NO — passes through the shared ReLU hidden layer |
| t3a/t3b (taxonomy) | per-rank embeddings summed (dim 16) concat → FFN | NO — through shared ReLU |
| t4 (late embed) | embedding(dim 16) concat → FFN | NO — through shared ReLU |

- For t2/t3a/t3b/t4, `H_molecule` is computed WITHOUT species (late/onehot/taxonomy do not enter message passing); species is concatenated to the fixed molecule vector, and molecule+species mix in the first FFN Linear then the shared ReLU — **1 nonlinear layer**, both backbones.
- t1's species term is added after the FFN → **degenerates to a per-species additive offset**, both backbones.
- (FiLM `H=γ(e_s)⊙H+β(e_s)` `variants.py:170-173` / `models.py:231-233` is the one multiplicative path — NOT in the ADORE tier set.)

## 3. Species-path parameters per tier (identical for both backbones)

emb_dim=16, ffn_hidden=300. species-path = (embedding table) + (first-FFN species-slice = repr_dim × 300).

| tier | discovery (n_species=779) | replication (n_species=1006) |
|---|---|---|
| t1 | 779 (bias table 779×1; post-FFN, no FFN slice) | 1,006 |
| t2 | 233,700 (onehot 0 params + 779×300) | 301,800 (1006×300) |
| t3a | 15,536 (671×16 + 16×300) | 19,216 (901×16 + 16×300) |
| t3b | 16,272 (717×16 + 16×300) | 20,480 (980×16 + 16×300) |
| t4 | 17,264 (779×16 + 16×300) | 20,896 (1006×16 + 16×300) |

tax cardinalities (tier_input_reference.json): disc native class14/order51/family182/genus424 (Σ671); disc ncbi 18/62/200/437 (Σ717); repl native 28/91/252/530 (Σ901); repl ncbi 37/112/285/546 (Σ980). Backbone-invariant because readout + `SpeciesEncoder`/taxonomy embeddings are identical code.

## 4. Hyperparameter symmetry

- Set in `run_q2_gnn_ladder.py:86` (and header :3-5), applied by the SAME build call regardless of backbone (no per-backbone branch): epochs 100, patience 15, batch 256, lr 5e-4, wd 1e-5, **dropout 0.1, hidden 300, depth 3, species_emb_dim 16, val_frac 0.1**. `ffn_num_layers=2` (default both — not overridden).
- **readout hidden width (300) and dropout (0.1): SAME for both backbones.**
- Per-backbone tuning record: **없음 (no ADORE per-backbone hyperparameter tuning/sweep record found).** The values are a single fixed config applied to both backbones. Related prereg text `PREREGISTRATION.md:126`: "하이퍼파라미터·패킹·결정론 완전 동일" (in the δ′-run context, asserting run-to-run identity). Launch manifest `BLOCK_A_LAUNCH_MANIFEST.md:3` records the same fixed values. (CC-MPNN predecessor used hidden 300/depth 3 but dropout 0.0; ADORE uses dropout 0.1 — no stated ADORE-specific tuning rationale.)

## 5. Species vector dimension across backbones

| tier | GraphConv | D-MPNN | same? |
|---|---|---|---|
| t4 (late embed) | species_emb_dim=16 | species_emb_dim=16 | yes |
| t3a/t3b (taxonomy) | per-rank emb dim=16 (summed) | per-rank emb dim=16 (summed) | yes |
| t2 (onehot) | n_species (779/1006) | n_species (779/1006) | yes |
| t1 (bias) | scalar (1) | scalar (1) | yes |

`SpeciesEncoder` (ccmpnn `variants.py:111-129`): embed→out_dim=emb_dim(16); onehot→out_dim=n_species. GraphConv mirror `models.py:172-176,255-259`. species_emb_dim=16 passed identically to both (`models.py` build_v3_model → ModelConfig for D-MPNN / GraphConvModel arg).

## Not verifiable
- None for the readout path — the ccmpnn readout (`model.py`, `variants.py`, `config.py`, `context.py`) was traced end to end (source is present/read-only, fully readable). Encoder internals differ by design and were not the subject of this comparison.
