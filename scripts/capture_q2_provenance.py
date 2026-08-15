"""Q2 v4 Phase 1/2 — source provenance capture, copy verification, reproducibility.

Fixes the provenance of the tox-learn raw CSVs so the Methods data-source
sentence ("...from the dataset link at commit <HASH>, SHA-256 <...>") is
reproducible. This script does NOT open replication endpoint values: it reads
raw files as bytes (SHA-256) and counts CSV records using only column 0. No EDA.

NOTE on row counting: raw newline counts overcount records because some fields
contain embedded newlines inside quotes. We therefore count records with a real
CSV parser (pandas, usecols=[0]) to match the audit record (50,604).

Stages:
  source  (Phase 1)  -> hash/size/mtime of raw CSVs + git + dataset link, write provenance.
  verify  (Phase 2)  -> hash vendor copies, compare to source, count records, assert total.
  repro   (Task R)   -> hash off-git archive backup, record reproducibility limitations.

Usage:
  python scripts/capture_q2_provenance.py --stage source
  python scripts/capture_q2_provenance.py --stage verify
  python scripts/capture_q2_provenance.py --stage repro
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

SOURCE_DIR = Path(r"<DATA_ROOT>\tox-learn")
SOURCE_FILES = ["groupsplit_train.csv", "groupsplit_test.csv"]
VENDOR_DIR = Path("results/q2_v4/vendor/toxlearn")
ARCHIVE_DIR = Path(r"<DATA_ROOT>\_q2v4_vendor_archive")  # off-git backup
OUT_DIR = Path("results/q2_v4/provenance")

# From tox-learn/README.md, set by repo commit 8eb48b6 ("Update dataset link").
DATASET_LINK = "https://drive.google.com/drive/folders/1D-paglmLlnHGQOCLe94TT2F7JtjYhmaP?usp=sharing"
DATASET_LINK_PREVIOUS = "https://drive.google.com/file/d/FILE_ID/view?usp=sharing"  # placeholder, pre-8eb48b6
DATASET_README_DESC = (
    "Sample dataset with mordred fingerprint, using default taxonomy for "
    "species representation and split by CAS_ID group"
)
EXPECTED_TOTAL_ROWS = 50604  # audit record: CSV records (train 40484 + test 10120), excl. header

# --- Task T / R findings ---------------------------------------------------
DATASET_VARIANT = (
    "ORIGINAL ('default') taxonomy variant, NOT NCBI. splitting.py reads "
    "integrated_dataset_log10detect_filled.csv, which fill_taxo.py builds as df_orig_filled "
    "(original taxonomy + fill_taxonomy_hierarchy gap-fill) -- NOT replace_taxonomy_with_ncbi. "
    "fingerprint_benchmark.py explicitly labels this file 'origin'. The NCBI variant "
    "(integrated_dataset_ncbi_filled.csv) is a separate pipeline output, absent from this "
    "download (the Drive folder 'tox_data' holds only the two groupsplit CSVs)."
)
REPRO_LIMITATION = (
    "The dataset link target (Google Drive folder 'tox_data') is NOT version-controlled; "
    "commit 8eb48b6 pins only the README link text, not the file bytes. As of capture the "
    "folder holds the same two files (Drive-modified 2025-10-27) matching our sizes, but it "
    "can change. SHA-256 proves what we used, not that a future reader receives identical "
    "bytes. At submission, deposit the derived splits (row indices + SHA-256, and/or the raw "
    "CSVs where license permits) to a versioned archive (Zenodo/figshare) and cite the DOI."
)
ETE3_PLAN = (
    "NCBI ('ncbi') taxonomy self-generated via Yuan's own fill_taxo.py get_taxonomy_from_ncbi, "
    "run in the 'mordred_env' conda env (ete3==3.1.3). NCBITaxa DB BUILT 2026-07-16 07:23Z "
    "(taxa.sqlite, 701 MB, at C:\\.etetoolkit\\taxa.sqlite -- under conda-run HOME resolved to "
    "the drive root, not ~/.etetoolkit). Coverage: 1646/2285 species resolved (72.0%); 639 "
    "unresolved (287 non-binomial names + 352 NCBI-untranslatable). Output: "
    "vendor/toxlearn/derived/taxonomy_ncbi.csv (species -> 6 ranks kingdom..family; superclass "
    "NOT produced by NCBI). Caveat: NCBI DB is time-versioned -> methodologically equivalent to "
    "Yuan's ncbi variant but not byte-identical. Raw CSVs unmodified."
)


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).isoformat()


def hash_bytes(path: Path, chunk: int = 1 << 20) -> tuple[str, int]:
    """Return (sha256_hexdigest, byte_size) in a single streaming pass."""
    h = hashlib.sha256()
    size = 0
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
            size += len(block)
    return h.hexdigest(), size


def csv_record_count(path: Path) -> int:
    """True CSV record count (excludes header, respects quoted embedded newlines).

    Reads only column 0 to stay minimal; this is a row-count only (seal-allowed)."""
    col0 = pd.read_csv(path, usecols=[0], dtype=str)
    return int(len(col0))


def file_record(path: Path) -> dict:
    st = path.stat()
    digest, size = hash_bytes(path)
    return {
        "path": str(path),
        "bytes": size,
        "sha256": digest,
        "csv_record_count": csv_record_count(path),
        "mtime_utc": _iso(st.st_mtime),
        "ctime_utc": _iso(st.st_ctime),
    }


def git_info(root: Path) -> dict:
    def run(args: list[str]) -> str:
        try:
            r = subprocess.run(
                ["git", "-C", str(root), *args],
                capture_output=True, text=True, check=True,
            )
            return r.stdout.strip()
        except Exception as exc:  # noqa: BLE001
            return f"<error: {exc}>"

    tracked = run(["ls-files", "--", *SOURCE_FILES])
    return {
        "commit": run(["rev-parse", "HEAD"]),
        "branch": run(["rev-parse", "--abbrev-ref", "HEAD"]),
        "describe": run(["describe", "--always", "--tags"]),
        "remote_origin": run(["remote", "get-url", "origin"]),
        "csv_files_tracked_by_git": bool(tracked.strip()),
        "note": "CSVs are untracked (downloaded via dataset link); integrity anchored by SHA-256 below.",
    }


def stage_source() -> dict:
    files = {name: file_record(SOURCE_DIR / name) for name in SOURCE_FILES}
    folder_stat = SOURCE_DIR.stat()
    return {
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "source_dir": str(SOURCE_DIR),
        "source_folder_ctime_utc": _iso(folder_stat.st_ctime),
        "source_folder_mtime_utc": _iso(folder_stat.st_mtime),
        "download_time_estimate_note": (
            "No explicit download timestamp; earliest file ctime is the best proxy. "
            "Folder and CSV ctimes cluster on 2026-06-26 (local)."
        ),
        "dataset_link": DATASET_LINK,
        "dataset_link_previous_placeholder": DATASET_LINK_PREVIOUS,
        "dataset_link_commit": "8eb48b6598a20b1232889bd4d24c288dd236ef2c",
        "dataset_readme_description": DATASET_README_DESC,
        "git": git_info(SOURCE_DIR),
        "source_files": files,
        "expected_total_records": EXPECTED_TOTAL_ROWS,
        "verification": {"status": "pending_copy"},
    }


def stage_verify(prov: dict) -> dict:
    results = {}
    total_rows = 0
    all_ok = True
    for name in SOURCE_FILES:
        src = prov["source_files"][name]
        vpath = VENDOR_DIR / name
        if not vpath.exists():
            results[name] = {"status": "missing_copy", "vendor_path": str(vpath)}
            all_ok = False
            continue
        digest, size = hash_bytes(vpath)
        records = csv_record_count(vpath)
        total_rows += records
        match = (digest == src["sha256"]) and (size == src["bytes"])
        all_ok = all_ok and match
        results[name] = {
            "vendor_path": str(vpath),
            "sha256": digest,
            "bytes": size,
            "csv_record_count": records,
            "sha256_matches_source": digest == src["sha256"],
            "bytes_match_source": size == src["bytes"],
            "records_match_source": records == src["csv_record_count"],
            "status": "ok" if match else "MISMATCH",
        }
    prov["verification"] = {
        "status": "ok" if all_ok else "FAILED",
        "verified_utc": datetime.now(timezone.utc).isoformat(),
        "per_file": results,
        "total_records": total_rows,
        "total_matches_expected": total_rows == EXPECTED_TOTAL_ROWS,
        "expected_total_records": EXPECTED_TOTAL_ROWS,
    }
    return prov


def stage_repro(prov: dict) -> dict:
    files = {}
    all_ok = True
    for name in SOURCE_FILES:
        src = prov["source_files"][name]
        apath = ARCHIVE_DIR / name
        if not apath.exists():
            files[name] = {"status": "missing", "archive_path": str(apath)}
            all_ok = False
            continue
        digest, size = hash_bytes(apath)
        ok = (digest == src["sha256"]) and (size == src["bytes"])
        all_ok = all_ok and ok
        files[name] = {
            "archive_path": str(apath),
            "sha256": digest,
            "bytes": size,
            "sha256_matches_source": digest == src["sha256"],
            "status": "ok" if ok else "MISMATCH",
        }
    prov["reproducibility"] = {
        "recorded_utc": datetime.now(timezone.utc).isoformat(),
        "backup_archive_dir": str(ARCHIVE_DIR),
        "backup_status": "ok" if all_ok else "FAILED",
        "backup_files": files,
        "dataset_variant": DATASET_VARIANT,
        "dataset_link_not_versioned_limitation": REPRO_LIMITATION,
        "ncbi_taxonomy_plan": ETE3_PLAN,
    }
    return prov


def write_markdown(prov: dict, path: Path) -> None:
    g = prov["git"]
    lines = [
        "# Q2 v4 — Source Provenance (tox-learn raw)",
        "",
        f"- Captured (UTC): `{prov['captured_utc']}`",
        f"- Source dir: `{prov['source_dir']}`",
        "",
        "## Dataset origin",
        "",
        f"- Repository: `{g['remote_origin']}`",
        f"- Commit: `{g['commit']}` (branch `{g['branch']}`, describe `{g['describe']}`)",
        f"- Dataset link (commit 8eb48b6): {prov['dataset_link']}",
        f"- Previous link (placeholder, pre-8eb48b6): `{prov['dataset_link_previous_placeholder']}`",
        f"- README description: \"{prov['dataset_readme_description']}\"",
        f"- CSVs tracked by git: `{g['csv_files_tracked_by_git']}` — {g['note']}",
        f"- Download-time estimate: {prov['download_time_estimate_note']}",
        f"  - folder ctime `{prov['source_folder_ctime_utc']}`",
        "",
        "## Source files (SHA-256)",
        "",
        "| file | bytes | records | mtime (UTC) | SHA-256 |",
        "|---|---:|---:|---|---|",
    ]
    for name, rec in prov["source_files"].items():
        lines.append(
            f"| `{name}` | {rec['bytes']:,} | {rec['csv_record_count']:,} | "
            f"{rec['mtime_utc']} | `{rec['sha256']}` |"
        )
    v = prov["verification"]
    lines += ["", "## Copy verification (vendor)", "", f"- Status: `{v['status']}`"]
    if v["status"] != "pending_copy":
        lines.append(
            f"- Total records in vendor copies: `{v.get('total_records')}` "
            f"(expected `{v.get('expected_total_records')}`, match=`{v.get('total_matches_expected')}`)"
        )
        for name, r in v.get("per_file", {}).items():
            lines.append(
                f"  - `{name}`: {r.get('status')} "
                f"(sha match=`{r.get('sha256_matches_source')}`, "
                f"bytes match=`{r.get('bytes_match_source')}`, "
                f"records match=`{r.get('records_match_source')}`)"
            )
    if "reproducibility" in prov:
        rp = prov["reproducibility"]
        lines += [
            "",
            "## Reproducibility",
            "",
            "### Dataset variant (Task T)",
            f"- {rp['dataset_variant']}",
            "",
            "### Off-git backup (R-1)",
            f"- Archive dir: `{rp['backup_archive_dir']}`  (status `{rp['backup_status']}`)",
        ]
        for name, r in rp.get("backup_files", {}).items():
            lines.append(f"  - `{name}`: {r.get('status')} (sha match=`{r.get('sha256_matches_source')}`)")
        lines += [
            "",
            "### Dataset-link durability limitation (R-2)",
            f"- {rp['dataset_link_not_versioned_limitation']}",
            "",
            "### NCBI taxonomy generation plan (T-3)",
            f"- {rp['ncbi_taxonomy_plan']}",
        ]
    lines += [
        "",
        "## Methods sentence (fill-in ready)",
        "",
        "> We used the curated tox-learn aquatic toxicity dataset of Yuan et al. "
        "(bioRxiv 2025.11.24.690199; repository mgtools/tox-learn, commit "
        f"`{g['commit'][:7]}`), obtained from the dataset link recorded in that commit "
        f"({prov['dataset_link']}). The two source files (the **original**-taxonomy variant, "
        "split by CAS group) were fixed by SHA-256 (`groupsplit_train.csv`, "
        "`groupsplit_test.csv`; see table above). The original group-split partitions were "
        "preserved throughout; each partition was filtered and aggregated independently, "
        "without merging.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["source", "verify", "repro"], required=True)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUT_DIR / "source_provenance.json"
    md_path = OUT_DIR / "source_provenance.md"

    if args.stage == "source":
        prov = stage_source()
    else:
        if not json_path.exists():
            raise SystemExit("Run --stage source first (source_provenance.json missing).")
        prov = json.loads(json_path.read_text(encoding="utf-8"))
        prov = stage_verify(prov) if args.stage == "verify" else stage_repro(prov)

    json_path.write_text(json.dumps(prov, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(prov, md_path)
    out_key = "reproducibility" if args.stage == "repro" else "verification"
    print(json.dumps(prov.get(out_key, {}), indent=2, ensure_ascii=False))
    print(f"\nwrote {json_path}\nwrote {md_path}")


if __name__ == "__main__":
    main()
