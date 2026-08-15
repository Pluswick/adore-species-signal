"""Tier-input non-degeneracy guard (director standing rule, Session 22).

At each tier start, assert the tier's SPECIES REPRESENTATION input is not degenerate, log
the stats, and HALT that tier (raise TierInputDegenerate) if degenerate. Motivated by the
NCBI-join bug where ncbi_* went all-null and Tier 3b silently collapsed into Tier 0 — caught
only by an RMSE coincidence. This makes that failure mode structural, not lucky.

Degeneracy (any -> halt): a representation column is all-null (or below a non-null floor),
its cardinality is 1, or its cardinality differs from a stored reference by >=10x (order of
magnitude). Cost ~0; called once per (split, tier) before training its cells.
"""
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np
import pandas as pd

from jcim_v3.rdkit_lgbm import TAX_RANKS

NONNULL_FLOOR = 0.5        # any representation column below this non-null fraction = degenerate
OOM = 10.0                 # cardinality off from reference by >= this factor = degenerate


class TierInputDegenerate(RuntimeError):
    """Raised when a tier's species-representation input is degenerate. The tier must not start."""


def tier_and_injection(variant: str) -> tuple[str, str]:
    """Map a variant name to (tier_label, injection) without importing torch/models."""
    if variant == "no_species":
        return ("t0", "none")
    if variant.endswith("bias_only"):
        return ("t1", "output_bias")
    # longer suffixes first so 'taxonomy_genusfamily' is matched before any shorter suffix
    for suf, tier in (("taxonomy_genusfamily", "t3a_gf"), ("taxonomy_genus", "t3a_g"),
                      ("taxonomy_original", "t3a"), ("taxonomy_ncbi", "t3b"),
                      ("late_fusion", "t4"), ("categorical", "t2"), ("fixed_proj", "fixed_proj"),
                      ("early_injection", "t4"), ("message_level", "t4"), ("film", "t4")):
        if variant.endswith(suf):
            return (tier, suf)
    raise ValueError(f"tier_and_injection: unrecognized variant {variant!r}")


def _idx_stats(full: pd.DataFrame, train: pd.DataFrame) -> dict:
    col = full["species_idx"]
    n_species = int(col.max()) + 1 if col.notna().any() else 0
    uniq = set(int(x) for x in col.dropna().unique())
    return {
        "species_idx_nonnull": float(col.notna().mean()),
        "n_species": n_species,
        "n_unique_full": len(uniq),
        "n_unique_train": int(train["species_idx"].dropna().nunique()),
        "contiguous_0_to_n": bool(uniq == set(range(n_species))) if n_species else False,
    }


def _rank_stats(full: pd.DataFrame, ranks: list[str]) -> dict:
    out = {}
    for rk in ranks:
        if rk not in full.columns:
            out[rk] = {"present": False, "nonnull": 0.0, "cardinality": 0}
            continue
        s = full[rk].astype("string").str.strip()
        s = s.mask(s.eq("") | s.str.lower().eq("nan"), pd.NA)
        out[rk] = {"present": True, "nonnull": float(s.notna().mean()),
                   "cardinality": int(s.dropna().nunique())}
    return out


def check_tier_input(variant: str, split: str, train: pd.DataFrame, full: pd.DataFrame,
                     reference: dict | None = None) -> dict:
    """Compute tier-input stats and decide degeneracy. Pure (no I/O, no raise)."""
    tier, inj = tier_and_injection(variant)
    checks: list[dict] = []
    reasons: list[str] = []

    def add(name, value, ok, floor=None):
        checks.append({"name": name, "value": value, "floor": floor, "ok": bool(ok)})
        if not ok:
            reasons.append(f"{name}={value!r} (floor={floor!r})")

    if tier == "t0":
        # no species representation: nothing can degenerate. Record and pass.
        return {"tier": tier, "injection": inj, "variant": variant, "split": split,
                "checks": [{"name": "no_species_representation", "value": True, "floor": None, "ok": True}],
                "degenerate": False, "reasons": [], "stats": {}}

    idx = _idx_stats(full, train)
    # every species-using tier needs a real, varied species_idx
    add("species_idx_nonnull", idx["species_idx_nonnull"], idx["species_idx_nonnull"] >= 0.999, 0.999)
    add("n_unique_train", idx["n_unique_train"], idx["n_unique_train"] >= 2, 2)

    stats = {"idx": idx}
    if tier in ("t2", "t4", "fixed_proj"):
        # one-hot / learned embedding: table sized by n_species; require contiguity + real trained rows
        add("n_species", idx["n_species"], idx["n_species"] >= 2, 2)
        add("species_idx_contiguous_0_to_n", idx["contiguous_0_to_n"], idx["contiguous_0_to_n"], True)
        if reference and "n_species" in reference:
            add("n_species_vs_ref", idx["n_species"],
                _within_oom(idx["n_species"], reference["n_species"]), f"~{reference['n_species']}")

    if inj in TAX_RANKS:                       # all taxonomy injections (incl. rank-truncation)
        ranks = TAX_RANKS[inj]
        rs = _rank_stats(full, ranks)
        stats["ranks"] = rs
        for rk in ranks:
            r = rs[rk]
            add(f"{rk}.present", r["present"], r["present"], True)
            add(f"{rk}.nonnull", round(r["nonnull"], 4), r["nonnull"] >= NONNULL_FLOOR, NONNULL_FLOOR)
            add(f"{rk}.cardinality", r["cardinality"], r["cardinality"] >= 2, 2)
            if reference and rk in reference.get("ranks", {}):
                ref_card = reference["ranks"][rk]["cardinality"]
                add(f"{rk}.card_vs_ref", r["cardinality"],
                    _within_oom(r["cardinality"], ref_card), f"~{ref_card}")

    return {"tier": tier, "injection": inj, "variant": variant, "split": split,
            "checks": checks, "degenerate": len(reasons) > 0, "reasons": reasons, "stats": stats}


def _within_oom(v: float, ref: float) -> bool:
    if ref <= 0 or v <= 0:
        return v == ref
    return abs(math.log10(v / ref)) < 1.0


def assert_tier_input(variant: str, split: str, data_dir: str, log_path: str,
                      reference_path: str | None = None) -> dict:
    """Load the split, run the check, append to the jsonl guard log, and raise if degenerate."""
    d = Path(data_dir)
    train = pd.read_csv(d / f"{split}_train.csv")
    test = pd.read_csv(d / f"{split}_test.csv")
    full = pd.concat([train, test], ignore_index=True)
    reference = None
    if reference_path and Path(reference_path).exists():
        ref_all = json.loads(Path(reference_path).read_text(encoding="utf-8"))
        reference = ref_all.get(split)
    rec = check_tier_input(variant, split, train, full, reference)
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    if rec["degenerate"]:
        raise TierInputDegenerate(
            f"TIER INPUT DEGENERATE — {variant} @ {split} [{rec['tier']}]: " + "; ".join(rec["reasons"]))
    return rec


def check_blockb_oov_input(variant: str, split: str, train_sp, cold_sp, applied_mode: str,
                           swapped_ok: bool) -> dict:
    """Post-OOV non-degeneracy for block B (director: check the input AFTER OOV mapping).
    Degenerate if no cold species were identified, or (for mean/collapse) the swap did nothing."""
    tier, inj = tier_and_injection(variant)
    reasons = []
    n_cold = len(set(int(s) for s in cold_sp))
    n_train = len(set(int(s) for s in train_sp))
    if tier != "t0":
        if n_train < 2:
            reasons.append(f"n_train_species={n_train}")
        if applied_mode in ("mean", "collapse") and not swapped_ok:
            reasons.append(f"oov_swap_noop mode={applied_mode}")
    return {"tier": tier, "variant": variant, "split": split, "oov_mode": applied_mode,
            "n_cold_species": n_cold, "n_train_species": n_train,
            "degenerate": len(reasons) > 0, "reasons": reasons}
