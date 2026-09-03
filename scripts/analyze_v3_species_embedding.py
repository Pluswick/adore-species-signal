from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parents[1] / "results" / "src" / ".matplotlib_cache"),
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.paths import RESULTS_ROOT


def _load_config(path: str | None) -> dict:
    if not path:
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _safe_corr(x: np.ndarray, y: np.ndarray) -> dict:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 3 or np.nanstd(x) < 1e-12 or np.nanstd(y) < 1e-12:
        return {
            "n": int(len(x)),
            "spearman_r": np.nan,
            "spearman_p": np.nan,
            "pearson_r": np.nan,
            "pearson_p": np.nan,
        }
    pearson_r = _pearson_r(x, y)
    spearman_r = _pearson_r(_rank_values(x), _rank_values(y))
    return {
        "n": int(len(x)),
        "spearman_r": spearman_r,
        "spearman_p": _approx_corr_p(spearman_r, len(x)),
        "pearson_r": pearson_r,
        "pearson_p": _approx_corr_p(pearson_r, len(x)),
    }


def _rank_values(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method="average").to_numpy(np.float64)


def _pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    x0 = x.astype(np.float64) - float(np.mean(x))
    y0 = y.astype(np.float64) - float(np.mean(y))
    denom = math.sqrt(float(np.sum(x0 * x0) * np.sum(y0 * y0)))
    if denom < 1e-12:
        return float("nan")
    return float(np.sum(x0 * y0) / denom)


def _approx_corr_p(r: float, n: int) -> float:
    if not np.isfinite(r) or n < 4:
        return float("nan")
    r = max(min(float(r), 0.999999), -0.999999)
    z = 0.5 * math.log((1.0 + r) / (1.0 - r)) * math.sqrt(n - 3)
    return float(math.erfc(abs(z) / math.sqrt(2.0)))


def _euclidean_distance_matrix(x: np.ndarray) -> np.ndarray:
    out = np.empty((len(x), len(x)), dtype=np.float64)
    chunk = 128
    for start in range(0, len(x), chunk):
        end = min(start + chunk, len(x))
        diff = x[start:end, None, :] - x[None, :, :]
        sq_dist = np.sum(diff * diff, axis=2)
        np.maximum(sq_dist, 0.0, out=sq_dist)
        out[start:end] = np.sqrt(sq_dist)
    return out


def _cosine_distance_matrix(x: np.ndarray) -> np.ndarray:
    norm = np.sqrt(np.sum(x * x, axis=1, keepdims=True))
    norm[norm < 1e-12] = 1.0
    xn = x / norm
    out = np.empty((len(x), len(x)), dtype=np.float64)
    chunk = 128
    for start in range(0, len(x), chunk):
        end = min(start + chunk, len(x))
        sim = np.sum(xn[start:end, None, :] * xn[None, :, :], axis=2)
        dist = 1.0 - sim
        np.clip(dist, 0.0, 2.0, out=dist)
        out[start:end] = dist
    return out


def _pca_2d(x: np.ndarray) -> np.ndarray:
    mean = np.mean(x, axis=0, keepdims=True)
    std = np.std(x, axis=0, keepdims=True)
    std[std < 1e-12] = 1.0
    xs = (x - mean) / std
    n_dim = int(xs.shape[1])
    cov = [[0.0 for _ in range(n_dim)] for _ in range(n_dim)]
    for row in xs:
        vals = [float(v) for v in row]
        for i in range(n_dim):
            vi = vals[i]
            for j in range(i, n_dim):
                cov[i][j] += vi * vals[j]
    denom = float(max(len(xs) - 1, 1))
    for i in range(n_dim):
        for j in range(i, n_dim):
            cov[i][j] /= denom
            cov[j][i] = cov[i][j]
    v1, lambda1 = _power_iteration(cov, seed_offset=0)
    cov2 = [
        [cov[i][j] - lambda1 * v1[i] * v1[j] for j in range(n_dim)]
        for i in range(n_dim)
    ]
    v2, _ = _power_iteration(cov2, seed_offset=1)
    scores = np.zeros((len(xs), 2), dtype=np.float64)
    for i, row in enumerate(xs):
        vals = [float(v) for v in row]
        scores[i, 0] = sum(vals[j] * v1[j] for j in range(n_dim))
        scores[i, 1] = sum(vals[j] * v2[j] for j in range(n_dim))
    return scores


def _power_iteration(matrix: list[list[float]], *, seed_offset: int) -> tuple[list[float], float]:
    n_dim = len(matrix)
    vec = [1.0 + ((i + seed_offset) % 3) * 0.1 for i in range(n_dim)]
    vec = _normalize_vector(vec)
    for _ in range(100):
        nxt = [sum(matrix[i][j] * vec[j] for j in range(n_dim)) for i in range(n_dim)]
        nxt = _normalize_vector(nxt)
        if sum(abs(nxt[i] - vec[i]) for i in range(n_dim)) < 1e-10:
            vec = nxt
            break
        vec = nxt
    mv = [sum(matrix[i][j] * vec[j] for j in range(n_dim)) for i in range(n_dim)]
    eigenvalue = sum(vec[i] * mv[i] for i in range(n_dim))
    return vec, eigenvalue


def _normalize_vector(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    if norm < 1e-12:
        return [0.0 for _ in vec]
    return [v / norm for v in vec]


def _metadata_for_embedding(path: Path) -> dict:
    meta_path = path.with_name(path.name.replace("__species_embeddings.csv", "__species_embedding_metadata.json"))
    with open(meta_path, encoding="utf-8") as f:
        return json.load(f)


def _embedding_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if col.startswith("embedding_dim_")]


def _pairwise_correlation_rows(
    emb: pd.DataFrame,
    summary: pd.DataFrame,
    *,
    metadata: dict,
    bias: pd.DataFrame | None,
    warnings: list[dict],
) -> list[dict]:
    cols = _embedding_columns(emb)
    merged = emb.merge(
        summary,
        on=["species", "species_idx"],
        how="inner",
        suffixes=("", "_summary"),
    )
    merged = merged[merged["n_test_rows_for_summary"] > 0].copy()
    if bias is not None:
        merged = merged.merge(
            bias[["species_idx", "species_bias_log10_space"]],
            on="species_idx",
            how="left",
        )
    if len(merged) < 3:
        warnings.append(
            {
                "analysis": "embedding_sensitivity_correlation",
                "backbone": metadata["backbone"],
                "species_mode": metadata["species_mode"],
                "split": metadata.get("split"),
                "seed": metadata.get("seed"),
                "warning": "too few species with prediction summary",
            }
        )
        return []

    X = merged[cols].to_numpy(np.float64)
    dist_specs = {
        "euclidean": _euclidean_distance_matrix(X),
        "cosine": _cosine_distance_matrix(X),
    }
    variables = {
        "mean_true_log10_test": merged["mean_true_log10_test"].to_numpy(np.float64),
        "mean_residual": merged["mean_error_log10"].to_numpy(np.float64),
    }
    if "species_bias_log10_space" in merged:
        variables["species_bias"] = merged["species_bias_log10_space"].to_numpy(np.float64)

    tri = np.triu_indices(len(merged), k=1)
    rows = []
    for dist_name, dist_mat in dist_specs.items():
        dist_values = dist_mat[tri]
        for variable, values in variables.items():
            sensitivity_dist = np.abs(values[:, None] - values[None, :])[tri]
            corr = _safe_corr(dist_values, sensitivity_dist)
            rows.append(
                {
                    "backbone": metadata["backbone"],
                    "species_mode": metadata["species_mode"],
                    "split": metadata.get("split"),
                    "seed": metadata.get("seed"),
                    "embedding_distance_type": dist_name,
                    "sensitivity_variable": variable,
                    "n_species": int(len(merged)),
                    "n_pairs": int(corr["n"]),
                    "spearman_r": corr["spearman_r"],
                    "spearman_p": corr["spearman_p"],
                    "pearson_r": corr["pearson_r"],
                    "pearson_p": corr["pearson_p"],
                }
            )
    return rows


def _pca_coordinates(
    emb: pd.DataFrame,
    summary: pd.DataFrame,
    *,
    metadata: dict,
) -> pd.DataFrame:
    cols = _embedding_columns(emb)
    X = emb[cols].to_numpy(np.float64)
    pc = _pca_2d(X)
    frame = emb[["species", "species_idx", "train_count", "test_count"]].copy()
    frame["backbone"] = metadata["backbone"]
    frame["species_mode"] = metadata["species_mode"]
    frame["split"] = metadata.get("split")
    frame["seed"] = metadata.get("seed")
    frame["pc1"] = pc[:, 0]
    frame["pc2"] = pc[:, 1]
    keep = summary[
        [
            "species",
            "species_idx",
            "species_count_bin",
            "mean_true_log10_test",
            "mean_error_log10",
            "rmse_by_species",
            "n_test_rows_for_summary",
        ]
    ].copy()
    keep = keep.rename(columns={"mean_true_log10_test": "mean_true_log10"})
    frame = frame.merge(keep, on=["species", "species_idx"], how="left")
    frame["species_count_bin"] = frame["species_count_bin"].fillna(
        frame["train_count"].map(lambda c: "cold" if c == 0 else "few" if c <= 4 else "mid" if c <= 20 else "rich")
    )
    return frame


def _plot_pca(coords: pd.DataFrame, path: Path) -> None:
    colors = {"cold": "#777777", "few": "#1f77b4", "mid": "#ff7f0e", "rich": "#2ca02c"}
    fig, ax = plt.subplots(figsize=(8, 6))
    for label, group in coords.groupby("species_count_bin"):
        ax.scatter(
            group["pc1"],
            group["pc2"],
            s=8,
            alpha=0.45,
            label=str(label),
            c=colors.get(str(label), "#444444"),
        )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Species embedding PCA smoke projection")
    ax.legend(markerscale=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _embedding_vs_bias_rows(coords: pd.DataFrame, bias_files: dict[tuple[str, str, int], pd.DataFrame]) -> list[dict]:
    rows = []
    for (backbone, species_mode, split, seed), group in coords.groupby(
        ["backbone", "species_mode", "split", "seed"]
    ):
        bias = bias_files.get((str(backbone), str(split), int(seed)))
        if bias is None:
            continue
        merged = group.merge(
            bias[["species_idx", "species_bias_log10_space"]],
            on="species_idx",
            how="inner",
        )
        emb_files = [col for col in group.columns if col.startswith("embedding_dim_")]
        if emb_files:
            values = merged[emb_files].to_numpy(np.float64)
            merged["embedding_norm"] = np.sqrt(np.sum(values * values, axis=1))
        else:
            merged["embedding_norm"] = np.sqrt(merged["pc1"] ** 2 + merged["pc2"] ** 2)
        variables = ["embedding_norm", "pc1", "pc2"]
        bias_variables = {
            "species_bias": merged["species_bias_log10_space"].to_numpy(np.float64),
            "mean_residual": merged["mean_error_log10"].to_numpy(np.float64),
        }
        for var in variables:
            x = merged[var].to_numpy(np.float64)
            for bias_name, y in bias_variables.items():
                corr = _safe_corr(x, y)
                rows.append(
                    {
                        "backbone": backbone,
                        "species_mode": species_mode,
                        "split": split,
                        "seed": int(seed),
                        "embedding_summary_variable": var,
                        "bias_variable": bias_name,
                        "n_species": corr["n"],
                        "spearman_r": corr["spearman_r"],
                        "spearman_p": corr["spearman_p"],
                        "pearson_r": corr["pearson_r"],
                        "pearson_p": corr["pearson_p"],
                    }
                )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v3_embedding_smoke.json")
    parser.add_argument(
        "--out-root",
        default=str(RESULTS_ROOT / "smoke" / "src_env" / "embedding_smoke"),
    )
    args = parser.parse_args()
    config = _load_config(args.config)
    out_root = Path(config.get("out_root", args.out_root))
    embeddings_dir = Path(config.get("embedding_root", out_root / "embeddings"))
    species_summary_path = out_root / "species_summary.csv"
    warnings_path = out_root / "embedding_warnings.json"
    warnings: list[dict] = []
    species_summary = pd.read_csv(species_summary_path)

    bias_files = {}
    for path in sorted(embeddings_dir.glob("*__species_bias.csv")):
        meta_path = path.with_name(path.name.replace("__species_bias.csv", "__species_embedding_metadata.json"))
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        bias_files[(str(meta["backbone"]), str(meta["split"]), int(meta["seed"]))] = pd.read_csv(path)

    corr_rows = []
    pca_frames = []
    enriched_pca_frames = []
    for path in sorted(embeddings_dir.glob("*__species_embeddings.csv")):
        meta = _metadata_for_embedding(path)
        emb = pd.read_csv(path)
        summary_mask = (
            (species_summary["backbone"] == meta["backbone"])
            & (species_summary["species_mode"] == meta["species_mode"])
        )
        if "split" in species_summary.columns:
            summary_mask &= species_summary["split"] == meta["split"]
        if "seed" in species_summary.columns:
            summary_mask &= species_summary["seed"] == int(meta["seed"])
        summary = species_summary[summary_mask].copy()
        bias = bias_files.get((str(meta["backbone"]), str(meta["split"]), int(meta["seed"])))
        corr_rows.extend(
            _pairwise_correlation_rows(
                emb,
                summary,
                metadata=meta,
                bias=bias,
                warnings=warnings,
            )
        )
        coords = _pca_coordinates(emb, summary, metadata=meta)
        for col in _embedding_columns(emb):
            coords[col] = emb[col].to_numpy()
        pca_frames.append(coords.drop(columns=[col for col in coords.columns if col.startswith("embedding_dim_")]))
        enriched_pca_frames.append(coords)

    corr = pd.DataFrame(corr_rows)
    corr.to_csv(out_root / "embedding_sensitivity_correlation.csv", index=False, encoding="utf-8")
    pca = pd.concat(pca_frames, ignore_index=True) if pca_frames else pd.DataFrame()
    pca.to_csv(out_root / "embedding_pca_coordinates.csv", index=False, encoding="utf-8")
    if len(pca):
        _plot_pca(pca, out_root / "embedding_pca_plot.png")

    enriched = pd.concat(enriched_pca_frames, ignore_index=True) if enriched_pca_frames else pd.DataFrame()
    bias_rows = _embedding_vs_bias_rows(enriched, bias_files) if len(enriched) else []
    pd.DataFrame(bias_rows).to_csv(out_root / "embedding_vs_species_bias.csv", index=False, encoding="utf-8")

    if not config.get("enable_umap", False):
        warnings.append(
            {
                "analysis": "umap",
                "warning": "UMAP disabled for smoke because import/runtime can hang in this environment; PCA output was generated.",
            }
        )

    with open(warnings_path, "w", encoding="utf-8") as f:
        json.dump({"warnings": warnings}, f, ensure_ascii=False, indent=2)
    print(
        json.dumps(
            {
                "embedding_sensitivity_correlation": str(out_root / "embedding_sensitivity_correlation.csv"),
                "embedding_pca_coordinates": str(out_root / "embedding_pca_coordinates.csv"),
                "embedding_pca_plot": str(out_root / "embedding_pca_plot.png"),
                "embedding_vs_species_bias": str(out_root / "embedding_vs_species_bias.csv"),
                "warnings": str(warnings_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
