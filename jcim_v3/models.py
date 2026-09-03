from __future__ import annotations

from dataclasses import dataclass, replace as dataclass_replace

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from jcim_v3.paths import add_ccmpnn_to_path

add_ccmpnn_to_path()

from ccmpnn import ModelConfig, VariantConfig, build_model  # noqa: E402
from ccmpnn.context import ContextBase  # noqa: E402
from ccmpnn.model import DMPNNModel  # noqa: E402

from jcim_v3.featurizer import BOND_FDIM, DESC_FDIM


@dataclass(frozen=True)
class V3ModelSpec:
    backbone: str
    variant: str
    uses_species: bool
    species_control: str
    injection: str


_SPECIES_CONDITIONS = {
    "true_species": "true",
    "zero_species": "zero",
    "shuffled_species": "shuffled",
    "dummy_species": "dummy",
}

_BIAS_ONLY_CONTROLS = {
    "species_bias_only": "bias_only",           # Tier 1 (true labels)
    "shuffled_species_bias_only": "shuffled",   # control: offset attached to the WRONG species
    "zero_species_bias_only": "zero",           # control: species information removed
    "dummy_species_bias_only": "dummy",         # control: random meaningless labels
}

_INJECTION_SUFFIXES = {
    "late_fusion": "late_fusion",
    "early_injection": "early_injection",
    "message_level": "message_level",
    "film": "film",
    # GAP item 6 — GNN "Tier 2" analogs at the readout:
    "categorical": "categorical",   # raw species one-hot concat (dim n_species); capacity NOT matched
    "fixed_proj": "fixed_proj",     # frozen random emb_dim projection; readout capacity matched to Tier 4
    # GAP item 7 — GNN taxonomy on both backbones (per-rank embeddings summed):
    "taxonomy_original": "taxonomy_original",
    "taxonomy_ncbi": "taxonomy_ncbi",
    # rank-truncation study (native ranks, tier-3a lineage): genus-only and genus+family.
    "taxonomy_genus": "taxonomy_genus",
    "taxonomy_genusfamily": "taxonomy_genusfamily",
}

# All taxonomy injections share one code path (per-rank embeddings summed via TAX_RANKS[injection]);
# the rank-truncation variants differ only in how many ranks TAX_RANKS supplies. Behavior for the
# original two is unchanged (they remain members), so any block-A resume stays bit-consistent.
TAXONOMY_INJECTIONS = frozenset({"taxonomy_original", "taxonomy_ncbi",
                                 "taxonomy_genus", "taxonomy_genusfamily"})


def model_spec_from_variant(backbone: str, variant: str) -> V3ModelSpec:
    if backbone not in {"dmpnn", "graphconv"}:
        raise ValueError(f"Unsupported backbone: {backbone!r}")
    if variant == "no_species":
        return V3ModelSpec(backbone, variant, False, "none", "none")
    # Tier 1 (additive species bias) + its control suite. The controls are DATA-side
    # (apply_species_control permutes/zeroes species_idx); the architecture is identical
    # to species_bias_only, so the family is parameter-matched by construction.
    if variant in _BIAS_ONLY_CONTROLS:
        return V3ModelSpec(backbone, variant, True, _BIAS_ONLY_CONTROLS[variant], "output_bias")
    for prefix, control in _SPECIES_CONDITIONS.items():
        for suffix, injection in _INJECTION_SUFFIXES.items():
            if variant == f"{prefix}_{suffix}":
                return V3ModelSpec(backbone, variant, True, control, injection)
    raise ValueError(f"Unsupported smoke variant: {variant!r}")


def _build_ffn(in_dim: int, hidden: int, out_dim: int, n_layers: int, dropout: float) -> nn.Module:
    if n_layers == 1:
        return nn.Sequential(nn.Linear(in_dim, out_dim))
    layers: list[nn.Module] = [nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout)]
    for _ in range(n_layers - 2):
        layers += [nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout)]
    layers.append(nn.Linear(hidden, out_dim))
    return nn.Sequential(*layers)


class GraphConvLayer(nn.Module):
    """Bond-aware GraphConv layer over the existing BatchMolGraph contract."""

    def __init__(self, hidden: int, bond_fdim: int, dropout: float, agg: str = "mean"):
        super().__init__()
        if agg not in {"mean", "sum"}:
            raise ValueError("agg must be 'mean' or 'sum'")
        self.agg = agg
        self.msg = nn.Linear(hidden + bond_fdim, hidden, bias=False)
        self.self_proj = nn.Linear(hidden, hidden, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: Tensor, bmg) -> Tensor:
        bond_idx = bmg.a2b
        src_atom = bmg.b2a[bond_idx]
        nei_h = h[src_atom]
        bond_f = bmg.f_bonds[bond_idx]
        msg = self.msg(torch.cat([nei_h, bond_f], dim=-1))
        mask = (bond_idx != 0).unsqueeze(-1).to(msg.dtype)
        msg = msg * mask
        agg_msg = msg.sum(dim=1)
        if self.agg == "mean":
            denom = mask.sum(dim=1).clamp_min(1.0)
            agg_msg = agg_msg / denom
        h_next = F.relu(self.self_proj(h) + agg_msg)
        return self.dropout(h_next)


def _species_embedding(
    *,
    n_species: int,
    species_emb_dim: int,
    enabled: bool,
) -> nn.Embedding | None:
    if not enabled:
        return None
    if n_species <= 0:
        raise ValueError("n_species > 0 required for species injection")
    return nn.Embedding(n_species, species_emb_dim)


class GraphConvModel(nn.Module):
    """Second backbone for v3: GraphConv-based GNN with configurable species injection."""

    def __init__(
        self,
        *,
        atom_fdim: int,
        bond_fdim: int,
        hidden: int,
        depth: int,
        dropout: float,
        n_species: int,
        species_emb_dim: int,
        injection: str,
        zero_species: bool = False,
        agg: str = "mean",
        ffn_layers: int = 2,
        desc_fdim: int = 0,
        tax_codes=None,
        tax_cards=None,
    ):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")
        if injection not in ({"none", "late_fusion", "early_injection", "message_level", "film",
                              "categorical", "fixed_proj"} | TAXONOMY_INJECTIONS):
            raise ValueError(f"Unsupported GraphConv injection: {injection!r}")
        self.injection = injection
        self.desc_fdim = int(desc_fdim)
        self.zero_species = zero_species
        self.hidden = hidden
        self.agg = agg
        self.atom_proj = nn.Linear(atom_fdim, hidden, bias=False)
        self.layers = nn.ModuleList(
            GraphConvLayer(hidden, bond_fdim, dropout, agg=agg) for _ in range(depth)
        )
        self.n_species = n_species
        uses_embedding = injection in {"late_fusion", "early_injection", "message_level", "film", "fixed_proj"}
        self.species_emb = _species_embedding(
            n_species=n_species,
            species_emb_dim=species_emb_dim,
            enabled=uses_embedding,
        )
        if injection == "fixed_proj" and self.species_emb is not None:
            # capacity-matched control (GAP item 6): a FROZEN random emb_dim projection of the
            # species one-hot — same readout capacity as Tier 4 but not learned from toxicity.
            self.species_emb.weight.requires_grad_(False)
        if injection == "early_injection":
            self.species_early_proj = nn.Linear(species_emb_dim, hidden, bias=False)
        else:
            self.species_early_proj = None
        if injection == "message_level":
            self.species_message_proj = nn.Linear(species_emb_dim, hidden, bias=False)
        else:
            self.species_message_proj = None
        if injection == "film":
            self.species_film_gamma = nn.Linear(species_emb_dim, hidden)
            self.species_film_beta = nn.Linear(species_emb_dim, hidden)
            nn.init.zeros_(self.species_film_gamma.weight)
            nn.init.ones_(self.species_film_gamma.bias)
            nn.init.zeros_(self.species_film_beta.weight)
            nn.init.zeros_(self.species_film_beta.bias)
        else:
            self.species_film_gamma = None
            self.species_film_beta = None
        # GAP item 7 (GraphConv taxonomy): per-rank embeddings summed to a species_emb_dim vector,
        # concatenated at readout. Rank codes are looked up by species_idx (so the shuffled control,
        # which permutes species_idx, automatically attaches the wrong species' taxonomy).
        self.tax_n_ranks = 0
        if injection in TAXONOMY_INJECTIONS:
            if tax_codes is None or tax_cards is None:
                raise ValueError(f"{injection} requires tax_codes and tax_cards")
            self.register_buffer("tax_codes", torch.as_tensor(tax_codes, dtype=torch.long))
            self.tax_rank_embs = nn.ModuleList(nn.Embedding(int(c), species_emb_dim) for c in tax_cards)
            self.tax_n_ranks = len(tax_cards)
        readout_dim = (hidden
                       + (species_emb_dim if (injection in {"late_fusion", "fixed_proj"}
                                              or injection in TAXONOMY_INJECTIONS) else 0)
                       + (n_species if injection == "categorical" else 0)
                       + self.desc_fdim)
        self.ffn = _build_ffn(readout_dim, hidden, 1, ffn_layers, dropout)

    def forward(self, bmg) -> Tensor:
        h = F.relu(self.atom_proj(bmg.f_atoms))
        if self.injection == "early_injection":
            h = F.relu(h + self._atom_species_term(bmg, self.species_early_proj))
        for layer in self.layers:
            h = layer(h, bmg)
            if self.injection == "message_level":
                h = F.relu(h + self._atom_species_term(bmg, self.species_message_proj))

        mols = []
        for start, size in bmg.a_scope:
            chunk = h[start : start + size]
            mols.append(chunk.mean(dim=0) if self.agg == "mean" else chunk.sum(dim=0))
        H = torch.stack(mols, dim=0)

        if self.injection == "film":
            gamma, beta = self.film_parameters_for_model(bmg.species_idx)
            H = gamma * H + beta
        if self.injection in {"late_fusion", "fixed_proj"}:
            H = torch.cat([H, self.species_vector_for_model(bmg.species_idx)], dim=1)
        if self.injection == "categorical":
            H = torch.cat([H, self._species_onehot(bmg.species_idx)], dim=1)
        if self.injection in TAXONOMY_INJECTIONS:
            H = torch.cat([H, self._species_taxonomy(bmg.species_idx)], dim=1)
        if self.desc_fdim:
            # SPEC 4-0b: endpoint/duration exposed identically to every tier.
            if bmg.f_descriptors is None:
                raise ValueError("desc_fdim>0 requires bmg.f_descriptors (endpoint/duration)")
            H = torch.cat([H, bmg.f_descriptors], dim=1)
        return self.ffn(H)

    def species_vector_for_model(self, species_idx: Tensor) -> Tensor:
        if self.species_emb is None:
            raise ValueError("species embedding is not enabled")
        if species_idx is None:
            raise ValueError("species_idx is required for species injection")
        emb = self.species_emb(species_idx)
        return emb * 0.0 if self.zero_species else emb

    def _species_onehot(self, species_idx: Tensor) -> Tensor:
        if species_idx is None:
            raise ValueError("species_idx is required for categorical injection")
        oh = F.one_hot(species_idx.long(), num_classes=self.n_species).to(torch.float32)
        return oh * 0.0 if self.zero_species else oh

    def _species_taxonomy(self, species_idx: Tensor) -> Tensor:
        if species_idx is None:
            raise ValueError("species_idx is required for taxonomy injection")
        codes = self.tax_codes[species_idx.long()]          # [batch, n_ranks]
        vec = sum(self.tax_rank_embs[r](codes[:, r]) for r in range(self.tax_n_ranks))
        return vec * 0.0 if self.zero_species else vec

    def _atom_species_term(self, bmg, proj: nn.Module | None) -> Tensor:
        if proj is None:
            raise ValueError("species projection is not enabled")
        per_mol = proj(self.species_vector_for_model(bmg.species_idx))
        term = per_mol[bmg.a2mol]
        term = term.clone()
        term[0] = 0.0
        return term

    def message_species_term_for_model(self, species_idx: Tensor) -> Tensor:
        if self.species_message_proj is None:
            raise ValueError("message-level species projection is not enabled")
        return self.species_message_proj(self.species_vector_for_model(species_idx))

    def film_parameters_for_model(self, species_idx: Tensor) -> tuple[Tensor, Tensor]:
        if self.species_film_gamma is None or self.species_film_beta is None:
            raise ValueError("FiLM species projection is not enabled")
        vec = self.species_vector_for_model(species_idx)
        return self.species_film_gamma(vec), self.species_film_beta(vec)


class DMPNNEarlyInjectionModel(nn.Module):
    """Inject species context into D-MPNN atom features before message passing."""

    def __init__(
        self,
        *,
        base_model: DMPNNModel,
        atom_fdim: int,
        n_species: int,
        species_emb_dim: int,
        zero_species: bool,
    ):
        super().__init__()
        self.base_model = base_model
        self.zero_species = zero_species
        self.species_emb = nn.Embedding(n_species, species_emb_dim)
        self.species_atom_proj = nn.Linear(species_emb_dim, atom_fdim, bias=False)

    def species_vector_for_model(self, species_idx: Tensor) -> Tensor:
        emb = self.species_emb(species_idx)
        return emb * 0.0 if self.zero_species else emb

    def early_species_term_for_model(self, species_idx: Tensor) -> Tensor:
        return self.species_atom_proj(self.species_vector_for_model(species_idx))

    def forward(self, bmg) -> Tensor:
        if bmg.species_idx is None:
            raise ValueError("species_idx is required for true_species_early_injection")
        per_mol = self.early_species_term_for_model(bmg.species_idx)
        atom_term = per_mol[bmg.a2mol].clone()
        atom_term[0] = 0.0
        bmg2 = dataclass_replace(bmg, f_atoms=bmg.f_atoms + atom_term)
        return self.base_model(bmg2)


class DMPNNMessageLevelContext(ContextBase):
    """Inject species context into each D-MPNN message update."""

    def __init__(
        self,
        *,
        n_species: int,
        species_emb_dim: int,
        hidden: int,
        zero_species: bool,
    ):
        super().__init__()
        self.zero_species = zero_species
        self.species_emb = nn.Embedding(n_species, species_emb_dim)
        self.species_message_proj = nn.Linear(species_emb_dim, hidden, bias=False)

    def species_vector_for_model(self, species_idx: Tensor) -> Tensor:
        emb = self.species_emb(species_idx)
        return emb * 0.0 if self.zero_species else emb

    def message_species_term_for_model(self, species_idx: Tensor) -> Tensor:
        return self.species_message_proj(self.species_vector_for_model(species_idx))

    def bond_term(self, h: Tensor, f_bonds: Tensor, *, bmg):
        if bmg.species_idx is None:
            raise ValueError("species_idx is required for true_species_message_level")
        term = self.message_species_term_for_model(bmg.species_idx)[bmg.b2mol]
        term = term.clone()
        term[0] = 0.0
        return term


class DMPNNTaxonomyContext(ContextBase):
    """D-MPNN taxonomy (adore_t3a/t3b): per-rank LEARNED nn.Embedding summed, concat at readout.

    Mirrors GraphConvModel._species_taxonomy (same species_emb_dim, same default init,
    same per-rank-sum shrinkage) so tier 3 is backbone-consistent between GraphConv and
    D-MPNN. Uses only the public ccmpnn ContextBase seam -- ccmpnn source untouched.
    """

    def __init__(self, *, tax_codes, tax_cards, species_emb_dim: int, zero_species: bool):
        super().__init__()
        self.zero_species = zero_species
        self.species_emb_dim = int(species_emb_dim)
        self.register_buffer("tax_codes", torch.as_tensor(tax_codes, dtype=torch.long))
        self.tax_rank_embs = nn.ModuleList(nn.Embedding(int(c), species_emb_dim) for c in tax_cards)
        self.n_ranks = len(tax_cards)

    def out_dim(self, hidden: int) -> int:
        return hidden + self.species_emb_dim

    def apply_global(self, H: Tensor, *, bmg) -> Tensor:
        if bmg.species_idx is None:
            raise ValueError("species_idx is required for D-MPNN taxonomy")
        codes = self.tax_codes[bmg.species_idx.long()]          # [n_mols, n_ranks]
        vec = sum(self.tax_rank_embs[r](codes[:, r]) for r in range(self.n_ranks))
        if self.zero_species:
            vec = vec * 0.0
        return torch.cat([H, vec], dim=1)


class ZeroedSpeciesEncoder(nn.Module):
    """Keep species encoder parameters while forwarding an all-zero vector."""

    def __init__(self, base: nn.Module):
        super().__init__()
        self.base = base
        self.out_dim = base.out_dim

    def forward(self, species_idx: Tensor) -> Tensor:
        return self.base(species_idx) * 0.0


class SpeciesBiasOnlyModel(nn.Module):
    """No species fusion inside the encoder; add only a species-specific scalar bias."""

    def __init__(self, base_model: nn.Module, n_species: int):
        super().__init__()
        if n_species <= 0:
            raise ValueError("species_bias_only requires n_species > 0")
        self.base_model = base_model
        self.species_bias = nn.Embedding(n_species, 1)
        nn.init.zeros_(self.species_bias.weight)

    def forward(self, bmg) -> Tensor:
        if bmg.species_idx is None:
            raise ValueError("species_idx is required for species_bias_only")
        return self.base_model(bmg) + self.species_bias(bmg.species_idx)


## Endpoint/duration exposure for the GNN backbones. Disabled until the uniform-exposure
## route is chosen (ccmpnn's mol_feat axis is unusable for our fusions; see build_v3_model).
GNN_STRATUM_EXPOSURE = False


def build_v3_model(
    *,
    spec: V3ModelSpec,
    atom_fdim: int,
    n_species: int,
    hidden: int,
    depth: int,
    dropout: float,
    species_emb_dim: int,
    desc_fdim: int = 0,
    tax_codes=None,
    tax_cards=None,
):
    # SPEC 4-0b wants endpoint/duration exposed identically to every tier. The ccmpnn
    # descriptor route (VariantConfig.mol_feat) is UNAVAILABLE here: mol_feat is a
    # CC-MPNN-only axis and is rejected for fusion in {none, late, film} -- i.e. every
    # fusion our ladder uses -- and ccmpnn is read-only. Until the uniform-exposure
    # design is settled, GNN stratum exposure stays DISABLED so all tiers and both
    # backbones remain identical (see results/q2_v4/runs/replication/TASK_D_STATUS.md).
    desc_fdim = int(desc_fdim) if GNN_STRATUM_EXPOSURE else 0
    use_late_species = spec.injection == "late_fusion"
    use_species_embedding = spec.injection in {
        "late_fusion",
        "early_injection",
        "message_level",
        "film",
        "fixed_proj",
    }
    # categorical (one-hot) uses no learned embedding but still consumes species, so zero
    # must apply to any species-using injection, not only embedding-based ones.
    zero_species = spec.species_control == "zero" and spec.injection != "none"

    if spec.backbone == "dmpnn":
        cfg = ModelConfig(
            atom_fdim=atom_fdim,
            bond_fdim=BOND_FDIM,
            desc_fdim=desc_fdim,
            hidden=hidden,
            depth=depth,
            dropout=dropout,
            n_species=n_species,
            species_emb_dim=species_emb_dim,
        )
        if spec.injection in TAXONOMY_INJECTIONS:
            # adore_t3a/t3b on D-MPNN: learned (not frozen) per-rank embedding context,
            # mirroring GraphConv taxonomy via the public ccmpnn ContextBase seam.
            if tax_codes is None or tax_cards is None:
                raise ValueError(f"{spec.injection} requires tax_codes and tax_cards")
            context = DMPNNTaxonomyContext(tax_codes=tax_codes, tax_cards=tax_cards,
                                           species_emb_dim=species_emb_dim, zero_species=zero_species)
            return DMPNNModel(cfg, context=context)
        if spec.injection == "late_fusion":
            model = build_model(cfg, VariantConfig(fusion="late", species_repr="embed"))
            if zero_species:
                model.context.species = ZeroedSpeciesEncoder(model.context.species)
            return model
        if spec.injection == "categorical":
            # GAP item 6: species ONE-HOT at readout (ccmpnn late + onehot). Capacity NOT matched.
            model = build_model(cfg, VariantConfig(fusion="late", species_repr="onehot"))
            if zero_species:
                model.context.species = ZeroedSpeciesEncoder(model.context.species)
            return model
        if spec.injection == "fixed_proj":
            # GAP item 6: FROZEN random emb projection (capacity-matched control to Tier 4).
            model = build_model(cfg, VariantConfig(fusion="late", species_repr="embed"))
            for p in model.context.species.parameters():
                p.requires_grad_(False)
            if zero_species:
                model.context.species = ZeroedSpeciesEncoder(model.context.species)
            return model
        if spec.injection == "film":
            model = build_model(cfg, VariantConfig(fusion="film", species_repr="embed"))
            if zero_species:
                model.context.species = ZeroedSpeciesEncoder(model.context.species)
            return model
        if spec.injection == "early_injection":
            # NOTE: ccmpnn forbids mol_feat when fusion='none' (read-only dependency),
            # so plain-fusion variants cannot take the readout-concat descriptor path.
            base = build_model(cfg, VariantConfig(fusion="none"))
            return DMPNNEarlyInjectionModel(
                base_model=base,
                atom_fdim=atom_fdim,
                n_species=n_species,
                species_emb_dim=species_emb_dim,
                zero_species=zero_species,
            )
        if spec.injection == "message_level":
            context = DMPNNMessageLevelContext(
                n_species=n_species,
                species_emb_dim=species_emb_dim,
                hidden=hidden,
                zero_species=zero_species,
            )
            return DMPNNModel(cfg, context=context)
        # NOTE: ccmpnn forbids mol_feat with fusion='none' -> Tier 0 / Tier 1 (output_bias)
        # cannot receive endpoint via the readout-concat path. UNRESOLVED (see TASK_D_STATUS).
        model = build_model(cfg, VariantConfig(fusion="none"))
        if spec.injection == "output_bias":
            return SpeciesBiasOnlyModel(model, n_species=n_species)
        return model

    graphconv = GraphConvModel(
        atom_fdim=atom_fdim,
        bond_fdim=BOND_FDIM,
        hidden=hidden,
        depth=depth,
        dropout=dropout,
        n_species=n_species,
        species_emb_dim=species_emb_dim,
        injection=spec.injection if spec.injection != "output_bias" else "none",
        zero_species=zero_species,
        desc_fdim=desc_fdim,
        tax_codes=tax_codes,
        tax_cards=tax_cards,
    )
    if spec.injection == "output_bias":
        return SpeciesBiasOnlyModel(graphconv, n_species=n_species)
    return graphconv


def count_trainable_params(model: nn.Module) -> tuple[int, int]:
    total = 0
    species = 0
    species_keywords = (
        "species",
        "to_gamma",
        "to_beta",
        "gamma",
        "beta",
    )
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        n = param.numel()
        total += n
        if any(key in name for key in species_keywords):
            species += n
    return total, species
