"""Preprocess the RepOD schizophrenia EEG dataset for LEADv2.

This script reads the 28 EDF recordings, standardizes channel names/order,
filters and re-references the signals, resamples them to multiple sampling
rates, rejects artifact-contaminated windows, and writes LEAD-compatible
X.dat, y.dat and meta.json files.

Nothing runs when this file is imported. Execute it explicitly from a shell.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import mne
import numpy as np


OLD_TO_NEW = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}
TARGET_CHANNELS = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T7", "C3", "Cz",
    "C4", "T8", "P7", "P3", "Pz", "P4", "P8", "O1", "O2",
]
FILE_PATTERN = re.compile(r"^(HC|SZ)_(\d{2})\.edf$", re.IGNORECASE)


@dataclass(frozen=True)
class Recording:
    path: Path
    group: str
    label: int
    subject_id: int


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Convert RepOD EDF recordings to LEADv2 memmap files."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=repo_root / "raw_data" / "RepOD",
        help="Folder containing HC/HC_01.edf... and SZ/SZ_01.edf...",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=repo_root / "dataset" / "L400" / "RepOD",
        help="Destination folder for X.dat, y.dat, meta.json and QC report.",
    )
    parser.add_argument("--segment-length", type=int, default=400)
    parser.add_argument(
        "--step", type=int, default=200,
        help="Sliding-window step in samples (200 gives 50%% overlap for L400).",
    )
    parser.add_argument(
        "--sampling-rates", type=int, nargs="+", default=[200, 100, 50]
    )
    parser.add_argument("--low-cut", type=float, default=0.5)
    parser.add_argument("--high-cut", type=float, default=45.0)
    parser.add_argument("--notch", type=float, default=50.0)
    parser.add_argument(
        "--edge-seconds", type=float, default=2.0,
        help="Discard this duration from both ends after filtering.",
    )
    parser.add_argument(
        "--max-abs-uv", type=float, default=500.0,
        help="Reject a window if any absolute sample exceeds this value.",
    )
    parser.add_argument(
        "--max-p2p-uv", type=float, default=500.0,
        help="Reject a window if any channel peak-to-peak exceeds this value.",
    )
    parser.add_argument(
        "--min-std-uv", type=float, default=0.1,
        help="Reject a window if any channel is flatter than this value.",
    )
    parser.add_argument(
        "--keep-artifacts", action="store_true",
        help="Disable amplitude/flat-channel window rejection (not recommended).",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Allow replacement of existing final output files.",
    )
    return parser.parse_args()


def discover_recordings(input_root: Path) -> list[Recording]:
    if not input_root.is_dir():
        raise FileNotFoundError(f"Input folder does not exist: {input_root}")

    paths = sorted(input_root.rglob("*.edf"), key=lambda p: p.name.lower())
    recordings: list[Recording] = []
    seen: set[tuple[str, int]] = set()

    for path in paths:
        match = FILE_PATTERN.fullmatch(path.name)
        if not match:
            raise ValueError(
                f"Unexpected EDF filename: {path.name}. "
                "Expected HC_01.edf...HC_14.edf or SZ_01.edf...SZ_14.edf."
            )
        group = match.group(1).upper()
        number = int(match.group(2))
        key = (group, number)
        if key in seen:
            raise ValueError(f"Duplicate recording: {group}_{number:02d}")
        seen.add(key)
        label = 0 if group == "HC" else 1
        subject_id = number if group == "HC" else 14 + number
        recordings.append(Recording(path, group, label, subject_id))

    expected = {(group, number) for group in ("HC", "SZ") for number in range(1, 15)}
    missing = sorted(expected - seen)
    extra = sorted(seen - expected)
    if missing or extra or len(recordings) != 28:
        raise ValueError(
            f"Expected exactly 28 recordings. Missing={missing}, extra={extra}, "
            f"found={len(recordings)}"
        )
    return sorted(recordings, key=lambda r: r.subject_id)


def load_and_standardize(recording: Recording, args: argparse.Namespace) -> mne.io.BaseRaw:
    raw = mne.io.read_raw_edf(recording.path, preload=True, verbose="ERROR")
    if not np.isclose(raw.info["sfreq"], 250.0):
        raise ValueError(f"{recording.path.name}: expected 250 Hz, got {raw.info['sfreq']}")

    rename = {old: new for old, new in OLD_TO_NEW.items() if old in raw.ch_names}
    raw.rename_channels(rename)
    missing = [channel for channel in TARGET_CHANNELS if channel not in raw.ch_names]
    unexpected = [channel for channel in raw.ch_names if channel not in TARGET_CHANNELS]
    if missing or unexpected or len(raw.ch_names) != len(TARGET_CHANNELS):
        raise ValueError(
            f"{recording.path.name}: channel mismatch; missing={missing}, "
            f"unexpected={unexpected}"
        )

    # Raw.pick() in some supported MNE versions has no ``ordered`` argument.
    # Pick first, then explicitly enforce the LEAD channel order.
    raw.pick(TARGET_CHANNELS)
    raw.reorder_channels(TARGET_CHANNELS)
    raw.set_montage("standard_1005", on_missing="raise", verbose="ERROR")

    # RepOD was recorded in Poland, where 50 Hz mains interference is expected.
    if args.notch > 0 and args.notch < raw.info["sfreq"] / 2:
        raw.notch_filter([args.notch], verbose="ERROR")
    raw.filter(args.low_cut, args.high_cut, verbose="ERROR")
    raw.set_eeg_reference("average", projection=False, verbose="ERROR")

    edge_samples = int(round(args.edge_seconds * raw.info["sfreq"]))
    if edge_samples > 0:
        if raw.n_times <= 2 * edge_samples:
            raise ValueError(f"{recording.path.name}: recording is too short after edge crop")
        raw.crop(
            tmin=edge_samples / raw.info["sfreq"],
            tmax=(raw.n_times - edge_samples - 1) / raw.info["sfreq"],
            include_tmax=True,
        )
    return raw


def window_is_clean(window_v: np.ndarray, args: argparse.Namespace) -> bool:
    if not np.isfinite(window_v).all():
        return False
    if args.keep_artifacts:
        return True

    window_uv = window_v * 1e6
    if np.max(np.abs(window_uv)) > args.max_abs_uv:
        return False
    if np.max(np.ptp(window_uv, axis=0)) > args.max_p2p_uv:
        return False
    if np.min(np.std(window_uv, axis=0)) < args.min_std_uv:
        return False
    return True


def validate_args(args: argparse.Namespace) -> None:
    if args.segment_length <= 0 or args.step <= 0:
        raise ValueError("segment-length and step must be positive")
    if not args.sampling_rates or any(fs <= 0 for fs in args.sampling_rates):
        raise ValueError("sampling-rates must contain positive integers")
    if len(set(args.sampling_rates)) != len(args.sampling_rates):
        raise ValueError("sampling-rates must not contain duplicates")
    if not 0 <= args.low_cut < args.high_cut:
        raise ValueError("Require 0 <= low-cut < high-cut")
    # MNE applies anti-alias filtering while resampling, so the 50 Hz copy can
    # safely be produced after the original 250 Hz recording is filtered to
    # 0.5--45 Hz. Its effective upper frequency is limited by its Nyquist rate.


def prepare_outputs(output_root: Path, overwrite: bool) -> tuple[Path, Path, Path, Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    final_paths = (
        output_root / "X.dat",
        output_root / "y.dat",
        output_root / "meta.json",
        output_root / "qc_report.csv",
    )
    existing = [str(path) for path in final_paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Output files already exist. Remove them or rerun with --overwrite:\n"
            + "\n".join(existing)
        )
    return final_paths


def preprocess(args: argparse.Namespace) -> None:
    validate_args(args)
    recordings = discover_recordings(args.input_root.resolve())
    x_path, y_path, meta_path, qc_path = prepare_outputs(
        args.output_root.resolve(), args.overwrite
    )
    x_tmp = x_path.with_suffix(".dat.part")
    y_tmp = y_path.with_suffix(".dat.part")

    for temp_path in (x_tmp, y_tmp):
        if temp_path.exists():
            temp_path.unlink()

    total_kept = 0
    total_rejected = 0
    qc_rows: list[dict[str, object]] = []

    try:
        with x_tmp.open("wb") as x_file, y_tmp.open("wb") as y_file:
            for recording in recordings:
                raw = load_and_standardize(recording, args)
                original_duration = raw.n_times / raw.info["sfreq"]

                for fs in args.sampling_rates:
                    scaled = raw.copy().resample(fs, verbose="ERROR")
                    data = scaled.get_data().T.astype(np.float32, copy=False)
                    possible = max(
                        0, 1 + (len(data) - args.segment_length) // args.step
                    )
                    kept = 0
                    rejected = 0

                    for start in range(
                        0, len(data) - args.segment_length + 1, args.step
                    ):
                        window = data[start : start + args.segment_length]
                        if not window_is_clean(window, args):
                            rejected += 1
                            continue
                        np.ascontiguousarray(window, dtype=np.float32).tofile(x_file)
                        np.asarray(
                            [recording.label, recording.subject_id, fs],
                            dtype=np.float32,
                        ).tofile(y_file)
                        kept += 1

                    total_kept += kept
                    total_rejected += rejected
                    qc_rows.append(
                        {
                            "file": recording.path.name,
                            "group": recording.group,
                            "label": recording.label,
                            "subject_id": recording.subject_id,
                            "original_sfreq": 250,
                            "target_sfreq": fs,
                            "duration_seconds_after_crop": round(original_duration, 3),
                            "possible_windows": possible,
                            "kept_windows": kept,
                            "rejected_windows": rejected,
                            "rejection_rate": round(rejected / possible, 6) if possible else 0,
                        }
                    )
                    print(
                        f"{recording.path.name} | fs={fs:3d} | "
                        f"kept={kept:5d} | rejected={rejected:5d}"
                    )

        if total_kept == 0:
            raise RuntimeError("No clean windows were retained; review rejection thresholds")

        expected_x_bytes = total_kept * args.segment_length * len(TARGET_CHANNELS) * 4
        expected_y_bytes = total_kept * 3 * 4
        if x_tmp.stat().st_size != expected_x_bytes:
            raise RuntimeError("X.dat.part size validation failed")
        if y_tmp.stat().st_size != expected_y_bytes:
            raise RuntimeError("y.dat.part size validation failed")

        x_tmp.replace(x_path)
        y_tmp.replace(y_path)

        with qc_path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(qc_rows[0]))
            writer.writeheader()
            writer.writerows(qc_rows)

        meta = {
            "N": total_kept,
            "T": args.segment_length,
            "C": len(TARGET_CHANNELS),
            "dtype": "float32",
            "label_mapping": {"HC": 0, "SZ": 1},
            "subject_id_mapping": {"HC": "1-14", "SZ": "15-28"},
            "sampling_rate_list": args.sampling_rates,
            "step": args.step,
            "channels": TARGET_CHANNELS,
            "preprocessing": {
                "rename_channels": OLD_TO_NEW,
                "notch_hz": args.notch,
                "bandpass_hz": [args.low_cut, args.high_cut],
                "reference": "average",
                "edge_crop_seconds": args.edge_seconds,
                "artifact_rejection_enabled": not args.keep_artifacts,
                "max_abs_uv": args.max_abs_uv,
                "max_p2p_uv": args.max_p2p_uv,
                "min_std_uv": args.min_std_uv,
            },
            "rejected_windows": total_rejected,
            "X_path": str(x_path),
            "y_path": str(y_path),
        }
        with meta_path.open("w", encoding="utf-8") as handle:
            json.dump(meta, handle, indent=2, ensure_ascii=False)

        # Final read-only memmap check verifies that LEAD can open the exact shapes.
        x_check = np.memmap(
            x_path, dtype=np.float32, mode="r",
            shape=(total_kept, args.segment_length, len(TARGET_CHANNELS)),
        )
        y_check = np.memmap(y_path, dtype=np.float32, mode="r", shape=(total_kept, 3))
        if not np.isfinite(x_check[: min(100, total_kept)]).all():
            raise RuntimeError("Non-finite values found during final X.dat validation")
        if set(np.unique(y_check[:, 0]).astype(int)) != {0, 1}:
            raise RuntimeError("Final label validation failed")
        if len(np.unique(y_check[:, 1]).astype(int)) != 28:
            raise RuntimeError("Final subject count validation failed")
        del x_check, y_check

    except Exception:
        for temp_path in (x_tmp, y_tmp):
            if temp_path.exists():
                temp_path.unlink()
        raise

    print("\nPreprocessing complete")
    print(f"Retained windows: {total_kept}")
    print(f"Rejected windows: {total_rejected}")
    print(f"Output folder: {args.output_root.resolve()}")


def main() -> int:
    try:
        preprocess(parse_args())
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
