r"""Create MDD-Mumtaz leave-one-frequency-band-out datasets for LEADv2.

Run only after reviewing the paths/options. This script reads the original EC
EDF files and the original LEAD L400 baseline; it never changes either.
Filtering is performed on each continuous recording before its 2-second edge
crop and 400-sample windows. Every retained window is checked against the
original broadband X.dat/y.dat so variants use exactly the same examples.

Example from FYP, in the E:\LEAD environment:
    python .\working\03_preprocess_mdd_frequency.py --bands alpha

Output: working/frequency_data/omit_alpha/MDD-Mumtaz/{X.dat,y.dat,meta.json,qc_report.csv}
The output folder must not exist; this script never overwrites an experiment.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from contextlib import ExitStack
from pathlib import Path

import mne
import numpy as np
from scipy.signal import butter, sosfiltfilt


LEAD_ROOT = Path(r"C:\Users\ROG\Desktop\LEAD")
BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lead-root", type=Path, default=LEAD_ROOT)
    parser.add_argument("--input-root", type=Path,
                        help="Original EC EDF folder; defaults to LEAD/raw_data/MDD-Mumtaz")
    parser.add_argument("--baseline-root", type=Path,
                        help="Original broadband folder; defaults to LEAD/dataset/L400/MDD-Mumtaz")
    parser.add_argument("--output-root", type=Path,
                        help="Defaults to FYP/working/frequency_data")
    parser.add_argument("--bands", nargs="+", choices=tuple(BANDS), required=True)
    parser.add_argument("--mode", choices=("omit", "only"), default="omit",
                        help="Remove a band (default) or retain only that band")
    parser.add_argument("--filter-order", type=int, default=4)
    return parser.parse_args()


def load_reference_preprocessor(lead_root: Path):
    source = lead_root / "data_preprocessing" / "MDD-Mumtaz" / "preprocess_mdd_mumtaz.py"
    spec = importlib.util.spec_from_file_location("mdd_preprocess_reference", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import original preprocessing protocol: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_continuous(path: Path, reference, baseline_meta: dict) -> np.ndarray:
    """Repeat the original broadband preprocessing, but retain the edge crop for later."""
    raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
    if not np.isclose(float(raw.info["sfreq"]), 256.0):
        raise ValueError(f"{path.name}: expected original sampling rate 256 Hz")
    rename = {}
    for original in raw.ch_names:
        if original.startswith("EEG ") and original.endswith("-LE"):
            name = original[4:-3]
            rename[original] = reference.LEGACY.get(name, name)
    raw.rename_channels(rename)
    missing = sorted(set(reference.CHANNELS) - set(raw.ch_names))
    if missing:
        raise ValueError(f"{path.name}: missing channels {missing}")
    raw.pick(reference.CHANNELS)
    raw.reorder_channels(reference.CHANNELS)
    raw.set_montage("standard_1005", on_missing="raise", verbose="ERROR")
    raw.set_eeg_reference("average", projection=False, verbose="ERROR")
    low, high = map(float, baseline_meta["filter_hz"])
    raw.filter(low, high, verbose="ERROR")
    rate = int(baseline_meta["SAMPLE_RATE_LIST"][0])
    raw.resample(rate, verbose="ERROR")
    return raw.get_data().T  # (time, channels), float64; crop after band removal.


def remove_band(continuous: np.ndarray, band: str, rate: int, order: int) -> np.ndarray:
    low, high = BANDS[band]
    # The broadband baseline already suppresses frequencies outside 0.5-45 Hz.
    if band == "delta":
        sos = butter(order, high, btype="highpass", fs=rate, output="sos")
    elif band == "gamma":
        sos = butter(order, low, btype="lowpass", fs=rate, output="sos")
    else:
        sos = butter(order, (low, high), btype="bandstop", fs=rate, output="sos")
    return sosfiltfilt(sos, continuous, axis=0)


def retain_band(continuous: np.ndarray, band: str, rate: int, order: int) -> np.ndarray:
    low, high = BANDS[band]
    sos = butter(order, (low, high), btype="bandpass", fs=rate, output="sos")
    return sosfiltfilt(sos, continuous, axis=0)


def band_power_ratio(broad: np.ndarray, variant: np.ndarray, band: str, rate: int) -> float:
    """Window-level QC only; not a classifier feature or statistical test."""
    frequency = np.fft.rfftfreq(broad.shape[0], d=1.0 / rate)
    low, high = BANDS[band]
    bins = (frequency >= low) & (frequency < high)
    original_power = np.abs(np.fft.rfft(broad, axis=0)) ** 2
    variant_power = np.abs(np.fft.rfft(variant, axis=0)) ** 2
    denominator = float(original_power[bins].sum())
    return float(variant_power[bins].sum() / denominator) if denominator > 0 else float("nan")


def main() -> None:
    args = arguments()
    if len(args.bands) != len(set(args.bands)):
        raise ValueError("Duplicate name in --bands")
    if args.filter_order < 1:
        raise ValueError("--filter-order must be positive")
    lead_root = args.lead_root.resolve(strict=True)
    input_root = (args.input_root or lead_root / "raw_data" / "MDD-Mumtaz").resolve(strict=True)
    baseline_root = (args.baseline_root or lead_root / "dataset" / "L400" / "MDD-Mumtaz").resolve(strict=True)
    output_root = (args.output_root or Path(__file__).resolve().parent / "frequency_data").resolve()
    target_dirs = {band: output_root / f"{args.mode}_{band}" / "MDD-Mumtaz" for band in args.bands}
    for target in target_dirs.values():
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite existing dataset: {target}")

    reference = load_reference_preprocessor(lead_root)
    with (baseline_root / "meta.json").open(encoding="utf-8") as stream:
        baseline_meta = json.load(stream)
    if (int(baseline_meta["T"]), int(baseline_meta["C"])) != (400, 19):
        raise ValueError("Expected 400 samples and 19 channels in broadband baseline")
    if list(baseline_meta["SAMPLE_RATE_LIST"]) != [200]:
        raise ValueError("Expected 200-Hz broadband baseline")
    if tuple(baseline_meta["channels"]) != tuple(reference.CHANNELS):
        raise ValueError("Channel order does not match original preprocessing")
    if (int(baseline_meta["STEP"]), int(baseline_meta["OVERLAP"])) != (200, 200):
        raise ValueError("Expected original 400-sample windows with 200-sample step")
    if list(map(float, baseline_meta["filter_hz"])) != [0.5, 45.0]:
        raise ValueError("Expected original 0.5-45 Hz broadband filtering")

    count, length, channels, rate = int(baseline_meta["N"]), 400, 19, 200
    original_x = np.memmap(baseline_root / "X.dat", dtype=np.float32, mode="r",
                           shape=(count, length, channels))
    original_y = np.memmap(baseline_root / "y.dat", dtype=np.float32, mode="r",
                           shape=(count, 3))
    protocol = argparse.Namespace(
        max_abs_uv=float(baseline_meta["artifact_limits_uv"]["max_abs"]),
        max_p2p_uv=float(baseline_meta["artifact_limits_uv"]["max_p2p"]),
        min_std_uv=float(baseline_meta["artifact_limits_uv"]["min_std"]),
    )
    records = reference.discover(input_root)
    edge = int(round(float(baseline_meta["edge_seconds"]) * rate))
    qc = {band: [] for band in args.bands}
    index = 0
    with ExitStack() as stack:
        streams = {}
        for band, target in target_dirs.items():
            target.mkdir(parents=True, exist_ok=False)
            streams[band] = (
                stack.enter_context((target / "X.dat.part").open("wb")),
                stack.enter_context((target / "y.dat.part").open("wb")),
            )
        for position, (path, group, label, subject_id) in enumerate(records, 1):
            continuous = load_continuous(path, reference, baseline_meta)
            if continuous.shape[0] <= 2 * edge + length:
                raise ValueError(f"{path.name}: recording too short after edge crop")
            broad = continuous[edge:-edge].astype(np.float32, copy=False)
            accepted = []
            for start in range(0, len(broad) - length + 1, 200):
                window = broad[start:start + length]
                if not reference.clean(window, protocol):
                    continue
                if index >= count:
                    raise RuntimeError("More accepted windows than original baseline")
                expected_label = np.asarray((label, subject_id, rate), dtype=np.float32)
                if not np.array_equal(original_y[index], expected_label):
                    raise RuntimeError(f"Baseline label/order mismatch at window {index}")
                if not np.allclose(window, original_x[index], rtol=1e-4, atol=1e-8):
                    raise RuntimeError(f"Baseline EEG mismatch at window {index}; do not compare bands")
                accepted.append(start)
                index += 1
            if not accepted:
                raise RuntimeError(f"{path.name}: no accepted windows")

            for band in args.bands:
                variant = (remove_band if args.mode == "omit" else retain_band)(
                    continuous, band, rate, args.filter_order)
                variant = variant[edge:-edge].astype(np.float32, copy=False)
                x_file, y_file = streams[band]
                sample_ratios = []
                for start in accepted:
                    window = np.ascontiguousarray(variant[start:start + length])
                    if not np.isfinite(window).all():
                        raise RuntimeError(f"Non-finite filtered window: {path.name}, {band}")
                    window.tofile(x_file)
                    np.asarray((label, subject_id, rate), dtype=np.float32).tofile(y_file)
                    if len(sample_ratios) < 20:
                        sample_ratios.append(band_power_ratio(
                            broad[start:start + length], window, band, rate))
                qc[band].append({
                    "file": path.name, "label": label, "subject_id": subject_id,
                    "retained_windows": len(accepted),
                    "mean_target_band_power_ratio": float(np.nanmean(sample_ratios)),
                })
            print(f"[{position:02d}/{len(records)}] {path.name}: {len(accepted)} matched windows", flush=True)

    if index != count:
        raise RuntimeError(f"Expected {count} original windows, matched only {index}")
    for band, target in target_dirs.items():
        x_part, y_part = target / "X.dat.part", target / "y.dat.part"
        if x_part.stat().st_size != count * length * channels * 4:
            raise RuntimeError(f"Unexpected X.dat size for {band}")
        if y_part.stat().st_size != count * 3 * 4:
            raise RuntimeError(f"Unexpected y.dat size for {band}")
        x_part.rename(target / "X.dat")
        y_part.rename(target / "y.dat")
        meta = dict(baseline_meta)
        meta.update({
            "X_path": "X.dat", "y_path": "y.dat",
            "frequency_variant": f"{args.mode}_{band}",
            f"frequency_{'removal' if args.mode == 'omit' else 'retained'}_hz": list(BANDS[band]),
            "frequency_filter": f"Butterworth order {args.filter_order}, SOS zero-phase on continuous EEG",
            "frequency_input": str(input_root),
            "matched_broadband_baseline": str(baseline_root),
            "window_alignment_verified": True,
        })
        (target / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        with (target / "qc_report.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(qc[band][0]))
            writer.writeheader()
            writer.writerows(qc[band])
        print(f"Saved {args.mode}_{band}: {count} matched windows at {target}")


if __name__ == "__main__":
    main()
