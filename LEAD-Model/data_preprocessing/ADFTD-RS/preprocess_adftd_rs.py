"""Convert OpenNeuro ds004504 derivatives to LEAD memmap format.

This script deliberately uses the dataset-provided denoised derivatives and does
not modify the downloaded BIDS dataset.
"""

from __future__ import annotations

import argparse
import json
from math import gcd
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from scipy.signal import resample_poly


TARGET_CHANNELS = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T7", "C3", "Cz",
    "C4", "T8", "P7", "P3", "Pz", "P4", "P8", "O1", "O2",
]
LEGACY_RENAME = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}
LABEL_MAP = {"C": 0, "A": 1, "F": 2}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--segment-length", type=int, default=400)
    parser.add_argument("--step", type=int, default=200)
    parser.add_argument("--sampling-rates", type=int, nargs="+", default=[200, 100, 50])
    return parser.parse_args()


def load_subjects(input_root: Path) -> list[tuple[str, int, int, Path]]:
    table = pd.read_csv(input_root / "participants.tsv", sep="\t")
    required = {"participant_id", "Group"}
    if not required.issubset(table.columns):
        raise ValueError(f"participants.tsv lacks columns: {sorted(required - set(table.columns))}")

    subjects = []
    for row in table.itertuples(index=False):
        name = str(row.participant_id)
        group = str(row.Group).strip()
        if group not in LABEL_MAP:
            raise ValueError(f"Unknown group {group!r} for {name}")
        sid = int(name.split("-")[-1])
        eeg_path = input_root / "derivatives" / name / "eeg" / f"{name}_task-eyesclosed_eeg.set"
        if not eeg_path.is_file():
            raise FileNotFoundError(eeg_path)
        subjects.append((name, sid, LABEL_MAP[group], eeg_path))
    return subjects


def prepare_raw(path: Path, preload: bool):
    raw = mne.io.read_raw_eeglab(path, preload=preload, verbose="ERROR")
    rename = {old: new for old, new in LEGACY_RENAME.items() if old in raw.ch_names}
    if rename:
        raw.rename_channels(rename)
    missing = [ch for ch in TARGET_CHANNELS if ch not in raw.ch_names]
    if missing:
        raise ValueError(f"{path}: missing channels {missing}; found {raw.ch_names}")
    raw.pick(TARGET_CHANNELS)
    raw.reorder_channels(TARGET_CHANNELS)
    return raw


def output_length(n_raw: int, original_fs: int, target_fs: int) -> int:
    factor = gcd(original_fs, target_fs)
    up, down = target_fs // factor, original_fs // factor
    return (n_raw * up + down - 1) // down


def count_windows(n_samples: int, segment_length: int, step: int) -> int:
    return 0 if n_samples < segment_length else 1 + (n_samples - segment_length) // step


def main() -> None:
    args = parse_args()
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    if args.segment_length <= 0 or args.step <= 0 or args.step > args.segment_length:
        raise ValueError("Require 0 < step <= segment-length")

    subjects = load_subjects(input_root)
    if len(subjects) != 88:
        raise ValueError(f"Expected 88 subjects, found {len(subjects)}")

    label_counts = {label: sum(s[2] == label for s in subjects) for label in (0, 1, 2)}
    if label_counts != {0: 29, 1: 36, 2: 23}:
        raise ValueError(f"Unexpected subject label counts: {label_counts}")

    per_subject_counts: dict[str, dict[int, int]] = {}
    total = 0
    for name, _, _, path in subjects:
        raw = prepare_raw(path, preload=False)
        original_fs = int(round(raw.info["sfreq"]))
        counts = {}
        for target_fs in args.sampling_rates:
            n_resampled = output_length(raw.n_times, original_fs, target_fs)
            counts[target_fs] = count_windows(n_resampled, args.segment_length, args.step)
            total += counts[target_fs]
        per_subject_counts[name] = counts

    output_root.mkdir(parents=True, exist_ok=False)
    x_path = output_root / "X.dat"
    y_path = output_root / "y.dat"
    x_mem = np.memmap(x_path, dtype=np.float32, mode="w+", shape=(total, args.segment_length, len(TARGET_CHANNELS)))
    y_mem = np.memmap(y_path, dtype=np.float32, mode="w+", shape=(total, 3))

    cursor = 0
    for index, (name, sid, label, path) in enumerate(subjects, start=1):
        raw = prepare_raw(path, preload=True)
        original_fs = int(round(raw.info["sfreq"]))
        data = raw.get_data().astype(np.float32, copy=False)
        for target_fs in args.sampling_rates:
            factor = gcd(original_fs, target_fs)
            up, down = target_fs // factor, original_fs // factor
            scaled = resample_poly(data, up, down, axis=1).astype(np.float32, copy=False).T
            starts = range(0, scaled.shape[0] - args.segment_length + 1, args.step)
            written = 0
            for start in starts:
                x_mem[cursor] = scaled[start:start + args.segment_length]
                y_mem[cursor] = (label, sid, target_fs)
                cursor += 1
                written += 1
            expected = per_subject_counts[name][target_fs]
            if written != expected:
                raise RuntimeError(f"{name} at {target_fs} Hz: expected {expected}, wrote {written}")
        print(f"[{index:02d}/88] {name} complete", flush=True)

    if cursor != total:
        raise RuntimeError(f"Expected {total} windows, wrote {cursor}")
    x_mem.flush()
    y_mem.flush()
    del x_mem, y_mem

    meta = {
        "N": int(total),
        "T": args.segment_length,
        "C": len(TARGET_CHANNELS),
        "SAMPLE_RATE_LIST": [int(value) for value in args.sampling_rates],
        "OVERLAP": args.segment_length - args.step,
        "STEP": args.step,
        "channels": TARGET_CHANNELS,
        "label_map": {"CN": 0, "AD": 1, "FTD": 2},
        "source": str(input_root),
        "source_variant": "dataset-provided derivatives",
        "X_path": "X.dat",
        "y_path": "y.dat",
    }
    with (output_root / "meta.json").open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, ensure_ascii=False, indent=2)

    print("Preprocessing complete")
    print(f"Subjects: {len(subjects)}; labels CN/AD/FTD: 29/36/23")
    print(f"Windows: {total}; shape: ({total}, {args.segment_length}, {len(TARGET_CHANNELS)})")
    print(f"Output: {output_root}")


if __name__ == "__main__":
    main()
