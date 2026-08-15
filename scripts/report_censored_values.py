from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jcim_v3.censored_report import generate_censored_report


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir", default=None)
    p.add_argument("--data-dir", default=None)
    p.add_argument("--out-dir", default=None)
    args = p.parse_args()

    kwargs = {}
    if args.raw_dir:
        kwargs["raw_dir"] = args.raw_dir
    if args.data_dir:
        kwargs["data_dir"] = args.data_dir
    if args.out_dir:
        kwargs["out_dir"] = args.out_dir
    summary = generate_censored_report(**kwargs)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

