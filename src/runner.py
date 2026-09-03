from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.models import build_v3_model, count_trainable_params, model_spec_from_variant
from src.paths import CC_MPNN_DATA, RESULTS_ROOT, add_ccmpnn_to_path
from src.species_controls import apply_species_control
from src.dataset import GraphDataset, StandardScaler, iterate_batches
from src.stratum import fit_stratum_effect
from src.stratum import remove as stratum_remove
from src.stratum import restore as stratum_restore
from src.featurizer import ATOM_FDIM, bemis_murcko_scaffold

add_ccmpnn_to_path()

from ccmpnn.metrics import perf_metrics, species_binned_rmse  # noqa: E402


@dataclass(frozen=True)
class V3RunConfig:
    backbone: str
    variant: str
    split: str = "scaffold"
    seed: int = 0
    epochs: int = 1
    batch_size: int = 64
    lr: float = 5e-4
    weight_decay: float = 1e-5
    hidden: int = 300
    depth: int = 3
    dropout: float = 0.1
    species_emb_dim: int = 16
    val_frac: float = 0.1
    limit_train: int | None = 512
    limit_test: int | None = 128
    data_dir: str = str(CC_MPNN_DATA)
    out_root: str = str(RESULTS_ROOT / "smoke")
    save_species_artifacts: bool = False


def _sample_frame(df: pd.DataFrame, n: int | None, seed: int) -> pd.DataFrame:
    out = df.copy()
    out["source_row_id"] = np.arange(len(out))
    if n is not None and len(out) > n:
        out = out.sample(n=n, random_state=seed)
    return out.reset_index(drop=True)


def _carve_val(df: pd.DataFrame, seed: int, val_frac: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(df))
    n_val = max(1, int(len(df) * val_frac))
    n_val = min(n_val, len(df) - 1)
    val = df.iloc[perm[:n_val]].reset_index(drop=True)
    train = df.iloc[perm[n_val:]].reset_index(drop=True)
    return train, val


def _setup_reproducible(seed: int) -> torch.device:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def _predict(model: nn.Module, ds: GraphDataset, batch_size: int, device: torch.device) -> np.ndarray:
    model.eval()
    pred = []
    for idx in iterate_batches(len(ds), batch_size, shuffle=False, seed=0):
        bmg, _ = ds.batch(idx)
        pred.append(model(bmg.to(device)).detach().cpu().numpy())
    out = np.concatenate(pred, axis=0)
    if ds.target_scaler is not None:
        out = ds.target_scaler.inverse(out)
    return out.ravel()


def _rmse(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - true) ** 2)))


def _species_lookup(frame: pd.DataFrame) -> dict[int, str]:
    cols = frame[["species_idx", "species"]].drop_duplicates("species_idx")
    return {int(row.species_idx): str(row.species) for row in cols.itertuples(index=False)}


def _control_sanity_rows(controlled: dict[str, object], *, backbone: str, variant: str) -> list[dict]:
    rows = []
    for partition, result in controlled.items():
        row = {
            "backbone": backbone,
            "species_mode": variant,
            "partition": partition,
            **result.sanity,
        }
        rows.append(row)
    return rows


def _species_vector_all_zero(model: nn.Module, frame: pd.DataFrame, device: torch.device) -> bool | None:
    if len(frame) == 0:
        return None
    target = (
        model.base_model
        if hasattr(model, "species_bias") and hasattr(model, "base_model")
        else model
    )
    idx = torch.as_tensor(
        frame["species_idx"].head(min(32, len(frame))).to_numpy(dtype=np.int64).copy(),
        dtype=torch.long,
        device=device,
    )
    with torch.no_grad():
        try:
            if hasattr(target, "species_vector_for_model"):
                vec = target.species_vector_for_model(idx)
            elif hasattr(target, "context") and hasattr(target.context, "species_vector_for_model"):
                vec = target.context.species_vector_for_model(idx)
            elif hasattr(target, "context") and hasattr(target.context, "species"):
                vec = target.context.species(idx)
            else:
                return None
        except ValueError:
            # bias_only variants (injection='none') expose no species embedding vector;
            # the embedding-zeroing sanity check does not apply to them.
            return None
    return bool(torch.allclose(vec, torch.zeros_like(vec)))


def _injection_shape_sanity(
    model: nn.Module,
    *,
    spec,
    frame: pd.DataFrame,
    hidden: int,
    device: torch.device,
) -> dict:
    if len(frame) == 0:
        return {
            "injection_location": spec.injection,
            "shape_check_applicable": False,
        }
    target = (
        model.base_model
        if hasattr(model, "species_bias") and hasattr(model, "base_model")
        else model
    )
    idx = torch.as_tensor(
        frame["species_idx"].head(min(8, len(frame))).to_numpy(dtype=np.int64).copy(),
        dtype=torch.long,
        device=device,
    )
    out = {
        "injection_location": spec.injection,
        "shape_check_applicable": spec.injection
        in {"early_injection", "message_level", "film"},
        "film_gamma_beta_shape_ok": None,
        "message_level_hidden_shape_ok": None,
        "early_injection_shape_ok": None,
    }
    if not out["shape_check_applicable"]:
        return out
    target.to(device)
    with torch.no_grad():
        if spec.injection == "film":
            if hasattr(target, "film_parameters_for_model"):
                gamma, beta = target.film_parameters_for_model(idx)
            elif hasattr(target, "context") and hasattr(target.context, "to_gamma"):
                emb = target.context.species(idx)
                gamma, beta = target.context.to_gamma(emb), target.context.to_beta(emb)
            else:
                gamma = beta = None
            out["film_gamma_beta_shape_ok"] = bool(
                gamma is not None
                and beta is not None
                and tuple(gamma.shape) == (len(idx), hidden)
                and tuple(beta.shape) == (len(idx), hidden)
            )
        if spec.injection == "message_level":
            if hasattr(target, "message_species_term_for_model"):
                term = target.message_species_term_for_model(idx)
            elif hasattr(target, "context") and hasattr(target.context, "message_species_term_for_model"):
                term = target.context.message_species_term_for_model(idx)
            else:
                term = None
            out["message_level_hidden_shape_ok"] = bool(
                term is not None and tuple(term.shape) == (len(idx), hidden)
            )
        if spec.injection == "early_injection":
            if hasattr(target, "early_species_term_for_model"):
                term = target.early_species_term_for_model(idx)
                expected = target.base_model.config.atom_fdim
            elif hasattr(target, "species_early_proj"):
                term = target.species_early_proj(target.species_vector_for_model(idx))
                expected = hidden
            else:
                term = None
                expected = None
            out["early_injection_shape_ok"] = bool(
                term is not None and expected is not None and tuple(term.shape) == (len(idx), expected)
            )
    return out


def _save_species_bias_if_available(
    model: nn.Module,
    *,
    out_root: Path,
    run_id: str,
    n_species: int,
    target_scaler: StandardScaler,
) -> str | None:
    if not hasattr(model, "species_bias"):
        return None
    bias = model.species_bias.weight.detach().cpu().numpy().reshape(-1)
    scale = float(np.asarray(target_scaler.std_).reshape(-1)[0])
    frame = pd.DataFrame(
        {
            "species_idx": np.arange(n_species, dtype=int),
            "bias_model_output_space": bias,
            "bias_log10_space": bias * scale,
        }
    )
    bias_dir = out_root / "species_bias"
    bias_dir.mkdir(parents=True, exist_ok=True)
    path = bias_dir / f"{run_id}_species_bias.csv"
    frame.to_csv(path, index=False, encoding="utf-8")
    return str(path)


def _species_index_frame(data_dir: Path, train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    species_index_path = data_dir / "species_index.csv"
    if species_index_path.exists():
        species_index = pd.read_csv(species_index_path)
    else:
        species_index = (
            pd.concat([train[["species", "species_idx"]], test[["species", "species_idx"]]], ignore_index=True)
            .drop_duplicates()
            .copy()
        )
        species_index["species_idx"] = species_index["species_idx"].astype(int)
        duplicated_idx = species_index[species_index["species_idx"].duplicated(keep=False)]
        if not duplicated_idx.empty:
            raise ValueError(
                "species_idx maps to multiple species names and species_index.csv is missing: "
                + repr(duplicated_idx.sort_values("species_idx").head(10).to_dict(orient="records"))
            )
    train_counts = train["species_idx"].astype(int).value_counts().to_dict()
    test_counts = test["species_idx"].astype(int).value_counts().to_dict()
    out = species_index.copy()
    out["species_idx"] = out["species_idx"].astype(int)
    out["train_count"] = out["species_idx"].map(train_counts).fillna(0).astype(int)
    out["test_count"] = out["species_idx"].map(test_counts).fillna(0).astype(int)
    return out.sort_values("species_idx").reset_index(drop=True)


def _embedding_module(model: nn.Module) -> nn.Module | None:
    target = (
        model.base_model
        if hasattr(model, "species_bias") and hasattr(model, "base_model")
        else model
    )
    if hasattr(target, "species_emb") and target.species_emb is not None:
        return target.species_emb
    if hasattr(target, "context") and hasattr(target.context, "species_emb"):
        return target.context.species_emb
    if hasattr(target, "context") and hasattr(target.context, "species"):
        species = target.context.species
        if hasattr(species, "base"):
            species = species.base
        if hasattr(species, "emb"):
            return species.emb
    return None


def _species_related_parameter_metadata(model: nn.Module) -> list[dict]:
    rows = []
    keywords = ("species", "gamma", "beta", "to_gamma", "to_beta")
    for name, param in model.named_parameters():
        if not any(key in name for key in keywords):
            continue
        arr = param.detach().cpu().numpy()
        rows.append(
            {
                "name": name,
                "shape": list(arr.shape),
                "trainable": bool(param.requires_grad),
                "mean": float(arr.mean()) if arr.size else 0.0,
                "std": float(arr.std()) if arr.size else 0.0,
                "l2_norm": float(np.linalg.norm(arr.ravel())) if arr.size else 0.0,
            }
        )
    return rows


def _save_species_artifacts(
    model: nn.Module,
    *,
    spec,
    cfg: V3RunConfig,
    data_dir: Path,
    out_root: Path,
    train_full: pd.DataFrame,
    test_full: pd.DataFrame,
    target_scaler: StandardScaler,
) -> dict:
    embeddings_dir = out_root / "embeddings"
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{cfg.backbone}__{cfg.variant}__{cfg.split}__seed{cfg.seed}"
    species_index = _species_index_frame(data_dir, train_full, test_full)

    files: dict[str, str | None] = {
        "species_embeddings": None,
        "species_embedding_metadata": None,
        "species_bias": None,
    }
    emb = _embedding_module(model)
    if emb is not None:
        weight = emb.weight.detach().cpu().numpy()
        frame = species_index.copy()
        for i in range(weight.shape[1]):
            frame[f"embedding_dim_{i}"] = weight[:, i]
        emb_path = embeddings_dir / f"{prefix}__species_embeddings.csv"
        frame.to_csv(emb_path, index=False, encoding="utf-8")
        files["species_embeddings"] = str(emb_path)

    if hasattr(model, "species_bias"):
        bias = model.species_bias.weight.detach().cpu().numpy().reshape(-1)
        scale = float(np.asarray(target_scaler.std_).reshape(-1)[0])
        frame = species_index.copy()
        frame["species_bias_model_output_space"] = bias
        frame["species_bias_log10_space"] = bias * scale
        bias_path = embeddings_dir / f"{prefix}__species_bias.csv"
        frame.to_csv(bias_path, index=False, encoding="utf-8")
        files["species_bias"] = str(bias_path)

    metadata = {
        "backbone": cfg.backbone,
        "species_mode": cfg.variant,
        "seed": cfg.seed,
        "split": cfg.split,
        "injection_location": spec.injection,
        "species_control": spec.species_control,
        "n_species": int(len(species_index)),
        "species_emb_dim": cfg.species_emb_dim,
        "has_species_embedding": emb is not None,
        "has_species_bias": hasattr(model, "species_bias"),
        "species_embedding_file": files["species_embeddings"],
        "species_bias_file": files["species_bias"],
        "species_related_parameters": _species_related_parameter_metadata(model),
        "taxonomy_input_policy": "Taxonomy columns are not model inputs; they may be used only for post-hoc coloring or interpretation references.",
    }
    meta_path = embeddings_dir / f"{prefix}__species_embedding_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    files["species_embedding_metadata"] = str(meta_path)
    return files


def _train(
    model: nn.Module,
    train_ds: GraphDataset,
    val_ds: GraphDataset,
    *,
    cfg: V3RunConfig,
    device: torch.device,
) -> tuple[nn.Module, list[float]]:
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.MSELoss()
    history: list[float] = []
    best = float("inf")
    best_state = None
    patience = 3 if cfg.epochs <= 3 else 15
    wait = 0
    for ep in range(cfg.epochs):
        model.train()
        for idx in iterate_batches(len(train_ds), cfg.batch_size, shuffle=True, seed=cfg.seed * 1000 + ep):
            bmg, y = train_ds.batch(idx)
            bmg, y = bmg.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(bmg), y)
            loss.backward()
            opt.step()
        pred = _predict(model, val_ds, cfg.batch_size, device)
        val_rmse = _rmse(pred, val_ds.target)
        history.append(val_rmse)
        if val_rmse < best - 1e-4:
            best = val_rmse
            wait = 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def run_v3_smoke(cfg: V3RunConfig) -> dict:
    spec = model_spec_from_variant(cfg.backbone, cfg.variant)
    device = _setup_reproducible(cfg.seed)
    data_dir = Path(cfg.data_dir)
    out_root = Path(cfg.out_root)
    (out_root / "runs").mkdir(parents=True, exist_ok=True)
    (out_root / "predictions").mkdir(parents=True, exist_ok=True)

    tr_full = pd.read_csv(data_dir / f"{cfg.split}_train.csv")
    te_full = pd.read_csv(data_dir / f"{cfg.split}_test.csv")
    full = pd.concat([tr_full, te_full], ignore_index=True)
    n_species = int(full["species_idx"].max()) + 1
    species_lookup = _species_lookup(full)

    tr_limited = _sample_frame(tr_full, cfg.limit_train, cfg.seed)
    te_limited = _sample_frame(te_full, cfg.limit_test, cfg.seed + 17)
    train_raw, val_raw = _carve_val(tr_limited, cfg.seed, cfg.val_frac)

    control_mode = spec.species_control
    train_ctrl = apply_species_control(
        train_raw,
        mode=control_mode,
        seed=cfg.seed + 101,
        n_species=n_species,
        species_lookup=species_lookup,
    )
    val_ctrl = apply_species_control(
        val_raw,
        mode=control_mode,
        seed=cfg.seed + 202,
        n_species=n_species,
        species_lookup=species_lookup,
    )
    test_ctrl = apply_species_control(
        te_limited,
        mode=control_mode,
        seed=cfg.seed + 303,
        n_species=n_species,
        species_lookup=species_lookup,
    )
    control_sanity = _control_sanity_rows(
        {"train": train_ctrl, "val": val_ctrl, "test": test_ctrl},
        backbone=cfg.backbone,
        variant=cfg.variant,
    )

    # SPEC 4-0b(A): remove the train-estimated additive endpoint/duration main effect from
    # the FITTING targets (train+val). Test targets stay in original units; predictions are
    # restored below. Identical operation to the lgbm/naive tiers. Identity on discovery
    # (single stratum -> effect == 0).
    stratum_eff = fit_stratum_effect(
        train_ctrl.frame, train_ctrl.frame["target_log10"].to_numpy(np.float64)
    )
    train_ctrl.frame["target_log10"] = stratum_remove(
        train_ctrl.frame, train_ctrl.frame["target_log10"].to_numpy(np.float64), stratum_eff
    )
    val_ctrl.frame["target_log10"] = stratum_remove(
        val_ctrl.frame, val_ctrl.frame["target_log10"].to_numpy(np.float64), stratum_eff
    )

    base = GraphDataset(train_ctrl.frame)
    target_scaler = StandardScaler().fit(base.target.reshape(-1, 1))
    # SPEC 4-0b: endpoint/duration stratum vector, standardized on TRAIN only.
    desc_scaler = None if base.raw_desc is None else StandardScaler().fit(base.raw_desc)
    train_ds = GraphDataset(train_ctrl.frame, target_scaler=target_scaler, desc_scaler=desc_scaler)
    val_ds = GraphDataset(val_ctrl.frame, target_scaler=target_scaler, desc_scaler=desc_scaler)
    test_ds = GraphDataset(test_ctrl.frame, target_scaler=target_scaler, desc_scaler=desc_scaler)

    # GAP item 7 (GraphConv taxonomy): build a species_idx -> per-rank integer-code lookup from the
    # RAW data (true species->taxonomy), same ranks/source as the LightGBM taxonomy tier. The model
    # indexes it by (possibly control-permuted) species_idx, so the shuffled control auto-applies.
    tax_codes = tax_cards = None
    from src.models import TAXONOMY_INJECTIONS
    if spec.injection in TAXONOMY_INJECTIONS:
        from src.rdkit_lgbm import TAX_RANKS
        ranks = TAX_RANKS[spec.injection]
        sp = full.drop_duplicates("species_idx").set_index("species_idx")
        tax_codes = np.zeros((n_species, len(ranks)), dtype=np.int64)
        tax_cards = []
        for j, r in enumerate(ranks):
            vals = sp[r].astype("string").fillna("__unknown__")
            vals = vals.mask(vals.str.strip() == "", "__unknown__")
            cats = pd.Categorical(vals)
            code_map = {int(k): int(v) for k, v in zip(sp.index.astype(int), cats.codes)}
            unk = len(cats.categories)  # extra bucket for any species_idx missing from the lookup
            tax_codes[:, j] = [code_map.get(si, unk) for si in range(n_species)]
            tax_cards.append(unk + 1)

    model = build_v3_model(
        spec=spec,
        atom_fdim=ATOM_FDIM,
        n_species=n_species,
        hidden=cfg.hidden,
        depth=cfg.depth,
        dropout=cfg.dropout,
        species_emb_dim=cfg.species_emb_dim,
        desc_fdim=train_ds.desc_fdim,
        tax_codes=tax_codes,
        tax_cards=tax_cards,
    )
    trainable_params, species_params = count_trainable_params(model)
    injection_shape_sanity = _injection_shape_sanity(
        model,
        spec=spec,
        frame=test_ctrl.frame,
        hidden=cfg.hidden,
        device=device,
    )

    t0 = time.time()
    model, history = _train(model, train_ds, val_ds, cfg=cfg, device=device)
    train_sec = time.time() - t0

    pred = _predict(model, test_ds, cfg.batch_size, device)
    # restore the stratum main effect: the model predicted in adjusted space
    pred = stratum_restore(test_ctrl.frame, np.asarray(pred, dtype=np.float64), stratum_eff)
    true = test_ds.target
    zero_species_vector_all_zero = (
        _species_vector_all_zero(model, test_ctrl.frame, device)
        if control_mode == "zero"
        else None
    )
    train_counts = tr_full["species_idx"].value_counts().to_dict()

    metrics = perf_metrics(pred, true)
    binned = species_binned_rmse(pred, true, test_ctrl.frame["species_idx_original"].to_numpy(), train_counts)
    run_id = (
        f"{cfg.backbone}_{cfg.variant}_{cfg.split}_s{cfg.seed}"
        f"_e{cfg.epochs}_n{cfg.limit_train or 'full'}"
    )
    pred_frame = test_ctrl.frame.copy()
    pred_frame["scaffold"] = pred_frame["smiles"].map(bemis_murcko_scaffold)
    pred_frame["compound_key"] = pred_frame["smiles"].astype(str)
    pred_frame["scaffold_key"] = pred_frame["scaffold"].astype(str)
    pred_frame["pred_log10"] = pred
    pred_frame["true_log10"] = true
    pred_frame["error_log10"] = pred - true
    pred_frame["model_name"] = f"{cfg.backbone}_{cfg.variant}"
    pred_frame["backbone"] = cfg.backbone
    pred_frame["variant"] = cfg.variant
    pred_frame["species_mode"] = cfg.variant
    pred_frame["injection_location"] = spec.injection
    pred_frame["species_control"] = control_mode
    pred_frame["split"] = cfg.split
    pred_frame["seed"] = cfg.seed

    pred_path = out_root / "predictions" / f"{run_id}.csv"
    json_path = out_root / "runs" / f"{run_id}.json"
    pred_frame.to_csv(pred_path, index=False, encoding="utf-8")
    species_bias_file = _save_species_bias_if_available(
        model,
        out_root=out_root,
        run_id=run_id,
        n_species=n_species,
        target_scaler=target_scaler,
    )
    species_artifact_files = (
        _save_species_artifacts(
            model,
            spec=spec,
            cfg=cfg,
            data_dir=data_dir,
            out_root=out_root,
            train_full=tr_full,
            test_full=te_full,
            target_scaler=target_scaler,
        )
        if cfg.save_species_artifacts
        else None
    )

    result = {
        "config": {
            **asdict(cfg),
            "run_id": run_id,
            "model_spec": asdict(spec),
            "data_source": str(data_dir),
            "prediction_file": str(pred_path),
            "species_bias_file": species_bias_file,
            "species_artifact_files": species_artifact_files,
        },
        "A": metrics,
        "B": binned,
        "C": None,
        "D": {
            "trainable_params": trainable_params,
            "species_trainable_params": species_params,
            "parameter_match_tolerance": 0.05,
            "parameter_match_status": "not_applicable_smoke",
            "train_sec": round(train_sec, 2),
            "epochs": len(history),
            "best_val_rmse": round(min(history), 4) if history else None,
            "device": device.type,
            "optimizer": "AdamW",
            "lr": cfg.lr,
            "weight_decay": cfg.weight_decay,
            "zero_species_vector_all_zero": zero_species_vector_all_zero,
            "injection_shape_sanity": injection_shape_sanity,
        },
        "E": {
            "control_sanity": control_sanity,
        },
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return result
