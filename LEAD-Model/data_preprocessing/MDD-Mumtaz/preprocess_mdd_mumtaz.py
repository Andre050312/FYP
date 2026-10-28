"""Convert Mumtaz EC EDF recordings to LEAD L400 memmaps.

Only eyes-closed recordings with unambiguous subject IDs are used. The
original EDF files are never modified. Run this once from the LEAD environment.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

import mne
import numpy as np


CHANNELS = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T7", "C3", "Cz",
    "C4", "T8", "P7", "P3", "Pz", "P4", "P8", "O1", "O2",
]
LEGACY = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}
NAME_RE = re.compile(r"(H|MDD)\s*S(\d+)\s+EC\.edf", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--sampling-rate", type=int, default=200)
    p.add_argument("--segment-length", type=int, default=400)
    p.add_argument("--step", type=int, default=200)
    p.add_argument("--edge-seconds", type=float, default=2.0)
    p.add_argument("--low-cut", type=float, default=0.5)
    p.add_argument("--high-cut", type=float, default=45.0)
    p.add_argument("--max-abs-uv", type=float, default=500.0)
    p.add_argument("--max-p2p-uv", type=float, default=500.0)
    p.add_argument("--min-std-uv", type=float, default=0.1)
    return p.parse_args()


def discover(root: Path) -> list[tuple[Path, str, int, int]]:
    if not root.is_dir():
        raise FileNotFoundError(root)
    records = []
    seen = set()
    for path in root.glob("*.edf"):
        match = NAME_RE.fullmatch(path.name)
        if not match:
            continue  # EO, TASK, and ambiguously named files are not in scope.
        group, number = match.group(1).upper(), int(match.group(2))
        if (group, number) in seen:
            raise ValueError(f"Duplicate EC recording for {group} S{number}")
        seen.add((group, number))
        label = 0 if group == "H" else 1
        subject_id = number if group == "H" else 100 + number
        records.append((path, group, label, subject_id))
    counts = Counter(group for _, group, _, _ in records)
    if counts != {"H": 28, "MDD": 30}:
        raise ValueError(f"Unexpected EC subject counts: {counts}")
    return sorted(records, key=lambda record: record[3])


def load_signal(path: Path, args: argparse.Namespace) -> tuple[np.ndarray, float]:
    raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
    original_rate = float(raw.info["sfreq"])
    if not np.isclose(original_rate, 256.0):
        raise ValueError(f"{path.name}: expected 256 Hz, got {original_rate}")
    rename = {}
    for original in raw.ch_names:
        if original.startswith("EEG ") and original.endswith("-LE"):
            name = original[4:-3]
            rename[original] = LEGACY.get(name, name)
    raw.rename_channels(rename)
    missing = sorted(set(CHANNELS) - set(raw.ch_names))
    if missing:
        raise ValueError(f"{path.name}: missing channels {missing}")
    raw.pick(CHANNELS)
    raw.reorder_channels(CHANNELS)
    raw.set_montage("standard_1005", on_missing="raise", verbose="ERROR")
    # EDF uses linked-ear reference. One common average reference is applied
    # to all subjects and both downstream models.
    raw.set_eeg_reference("average", projection=False, verbose="ERROR")
    raw.filter(args.low_cut, args.high_cut, verbose="ERROR")
    raw.resample(args.sampling_rate, verbose="ERROR")
    edge = int(round(args.edge_seconds * args.sampling_rate))
    if raw.n_times <= 2 * edge + args.segment_length:
        raise ValueError(f"{path.name}: too short after filtering and edge crop")
    data = raw.get_data()[:, edge: raw.n_times - edge].T.astype(np.float32, copy=False)
    return data, original_rate


def clean(window: np.ndarray, args: argparse.Namespace) -> bool:
    if not np.isfinite(window).all():
        return False
    uv = window * 1e6
    return (
        np.max(np.abs(uv)) <= args.max_abs_uv
        and np.max(np.ptp(uv, axis=0)) <= args.max_p2p_uv
        and np.min(np.std(uv, axis=0)) >= args.min_std_uv
    )


def main() -> None:
    args = parse_args()
    if args.sampling_rate != 200 or args.segment_length != 400:
        raise ValueError("This L400 protocol requires 200 Hz and 400 samples")
    if not 0 < args.step <= args.segment_length:
        raise ValueError("Require 0 < step <= segment-length")
    if not 0 <= args.low_cut < args.high_cut < args.sampling_rate / 2:
        raise ValueError("Invalid frequency range")
    if args.edge_seconds < 0:
        raise ValueError("edge-seconds must be nonnegative")

    input_root = args.input_root.resolve(strict=True)
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_root}")
    records = discover(input_root)
    output_root.mkdir(parents=True, exist_ok=False)
    x_part = output_root / "X.dat.part"
    y_part = output_root / "y.dat.part"
    qc_rows = []
    kept_total = rejected_total = 0
    with x_part.open("wb") as x_file, y_part.open("wb") as y_file:
        for index, (path, group, label, subject_id) in enumerate(records, 1):
            data, original_rate = load_signal(path, args)
            kept = rejected = 0
            for start in range(0, len(data) - args.segment_length + 1, args.step):
                window = data[start:start + args.segment_length]
                if not clean(window, args):
                    rejected += 1
                    continue
                np.ascontiguousarray(window).tofile(x_file)
                np.asarray((label, subject_id, args.sampling_rate), dtype=np.float32).tofile(y_file)
                kept += 1
            if kept == 0:
                raise ValueError(f"{path.name}: no clean windows retained")
            kept_total += kept
            rejected_total += rejected
            qc_rows.append({
                "file": path.name, "group": group, "label": label,
                "subject_id": subject_id, "original_rate": original_rate,
                "usable_seconds": round(len(data) / args.sampling_rate, 3),
                "retained_windows": kept, "rejected_windows": rejected,
            })
            print(f"[{index:02d}/{len(records)}] {path.name}: kept={kept}, rejected={rejected}", flush=True)

    expected_x = kept_total * args.segment_length * len(CHANNELS) * 4
    expected_y = kept_total * 3 * 4
    if x_part.stat().st_size != expected_x or y_part.stat().st_size != expected_y:
        raise RuntimeError("Output file size does not match retained window count")
    x_part.rename(output_root / "X.dat")
    y_part.rename(output_root / "y.dat")
    meta = {
        "N": kept_total, "T": args.segment_length, "C": len(CHANNELS),
        "SAMPLE_RATE_LIST": [args.sampling_rate],
        "OVERLAP": args.segment_length - args.step, "STEP": args.step,
        "channels": CHANNELS, "label_map": {"HC": 0, "MDD": 1},
        "source": str(input_root), "source_variant": "Mumtaz EC EDF",
        "reference": "average of 19 selected channels",
        "filter_hz": [args.low_cut, args.high_cut],
        "edge_seconds": args.edge_seconds,
        "artifact_limits_uv": {
            "max_abs": args.max_abs_uv, "max_p2p": args.max_p2p_uv,
            "min_std": args.min_std_uv,
        },
        "subject_counts": {"HC": 28, "MDD": 30},
        "X_path": "X.dat", "y_path": "y.dat",
    }
    with (output_root / "meta.json").open("w", encoding="utf-8") as file:
        json.dump(meta, file, ensure_ascii=False, indent=2)
    with (output_root / "qc_report.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(qc_rows[0]))
        writer.writeheader()
        writer.writerows(qc_rows)
    print(f"Complete: {len(records)} subjects, {kept_total} windows kept, {rejected_total} rejected")
    print(f"Output: {output_root}")


if __name__ == "__main__":
    main()
