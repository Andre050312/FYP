r"""Launch paired LEADv2 training for one preprocessed frequency variant.

First create the data with 03_preprocess_mdd_frequency.py. This wrapper checks
that preprocessing finished and then invokes 02_lead_mdd_train_from_scratch.py
with its random-initialization supervised training path. It does not load a
pretrained checkpoint or evaluate TEST.

Example from FYP with the E:\LEAD environment activated:
    python .\working\04_train_mdd_frequency.py --band alpha --seeds 41:43
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


BANDS = ("delta", "theta", "alpha", "beta", "gamma")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--band", required=True, choices=BANDS)
    parser.add_argument("--mode", choices=("omit", "only"), default="omit")
    parser.add_argument("--seeds", default="41", help="One seed, FIRST:LAST, or comma-separated seeds")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--loss", choices=("subject", "sample"), default="subject")
    parser.add_argument("--lead-root", type=Path, default=Path(r"C:\Users\ROG\Desktop\LEAD"))
    parser.add_argument("--frequency-root", type=Path,
                        help="Defaults to FYP/working/frequency_data")
    args = parser.parse_args()

    fyp_root = Path(__file__).resolve().parent.parent
    frequency_root = (args.frequency_root or fyp_root / "working" / "frequency_data").resolve()
    data_root = frequency_root / f"{args.mode}_{args.band}"
    dataset = data_root / "MDD-Mumtaz"
    meta_path = dataset / "meta.json"
    with meta_path.open(encoding="utf-8") as stream:
        meta = json.load(stream)
    if meta.get("frequency_variant") != f"{args.mode}_{args.band}" or not meta.get("window_alignment_verified"):
        raise RuntimeError(f"Incomplete or mismatched frequency dataset: {dataset}")
    for name in ("X.dat", "y.dat", "qc_report.csv"):
        if not (dataset / name).is_file():
            raise FileNotFoundError(dataset / name)
    output_root = fyp_root / "output_Test" / "from_scratch_frequency" / f"{args.mode}_{args.band}"
    command = [
        sys.executable, str(fyp_root / "working" / "02_lead_mdd_train_from_scratch.py"),
        "--lead-root", str(args.lead_root), "--data-root", str(data_root),
        "--output-root", str(output_root), "--seeds", args.seeds,
        "--epochs", str(args.epochs), "--patience", str(args.patience),
        "--batch-size", str(args.batch_size),
        "--learning-rate", str(args.learning_rate), "--loss", args.loss,
    ]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
