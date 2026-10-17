r"""First FYP experiment: LEADv2 MDD validation-set electrode ablation.

This script performs inference only. It does not train, modify LEAD, or write
to the EEG dataset. A channel is set to zero *after* LEAD's per-window
normalization, so zero represents that channel's normalized mean.

Example (from the FYP directory, with E:\LEAD activated):
    python .\working\01_lead_mdd_electrode_ablation.py --seed 41

Each run writes a new CSV to FYP/output_Test. Existing results are preserved.

Use the validation split to develop the ranking. Do not choose electrodes by
looking at the test split; reserve test data for the final, fixed method.
"""

from __future__ import annotations

import argparse
import csv
import random
import re
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader


DEFAULT_LEAD_ROOT = Path(r"C:\Users\ROG\Desktop\LEAD")
DEFAULT_CHANNELS = (
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T7", "C3", "Cz",
    "C4", "T8", "P7", "P3", "Pz", "P4", "P8", "O1", "O2",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lead-root", type=Path, default=DEFAULT_LEAD_ROOT)
    parser.add_argument("--seed", type=int, default=41, choices=range(41, 46))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--channels", default="all",
        help="Comma-separated subset for a quick pilot, or 'all' (default).",
    )
    parser.add_argument(
        "--output", type=Path,
        help="Optional CSV filename; it is saved inside FYP/output_Test.",
    )
    return parser.parse_args()


def model_config(meta: dict) -> SimpleNamespace:
    channels = tuple(meta["channels"])
    if channels != DEFAULT_CHANNELS or int(meta["C"]) != len(channels):
        raise ValueError("MDD channel order differs from the checkpoint configuration")
    if int(meta["T"]) != 400 or list(meta["SAMPLE_RATE_LIST"]) != [200]:
        raise ValueError("Expected the existing 400-sample, 200-Hz MDD data")
    return SimpleNamespace(
        task_name="finetune", output_attention=False, patch_len=50, stride=50,
        enc_in=len(channels), seq_len=int(meta["T"]), d_model=128, n_heads=8,
        e_layers=12, d_ff=256, dropout=0.1, activation="gelu",
        channel_names=",".join(channels), montage_name="standard_1005",
        augmentations="patch", num_class=2,
    )


def load_model(lead_root: Path, seed: int, meta: dict, device: torch.device):
    from models.LEADv2 import Model

    checkpoint = (
        lead_root / "checkpoints" / "LEADv2" / "finetune" / "LEADv2"
        / "P-Base-F-MDD-Mumtaz-200Hz-MCCV5"
        / f"nh8_el12_dm128_df256_seed{seed}" / "checkpoint.pth"
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    raw_state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = {}
    for key, value in raw_state.items():
        if key == "n_averaged":
            continue  # SWA wrapper bookkeeping, not a model parameter.
        while key.startswith("module."):
            key = key[len("module."):]
        if key in state:
            raise ValueError(f"Duplicate checkpoint key after unwrapping: {key}")
        state[key] = value
    model = Model(model_config(meta))
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model, checkpoint


def load_validation_data(lead_root: Path, seed: int):
    from data_provider.data_loader import MultiDatasetsLoader
    from data_provider.uea import collate_fn

    data_root = str(lead_root / "dataset" / "L400")
    args = SimpleNamespace(
        root_path=data_root,
        no_normalize=False, pretraining_datasets="TDBRAIN",
        training_datasets="MDD-Mumtaz", testing_datasets="MDD-Mumtaz",
        classify_choice="mdd_vs_hc", cross_val="mccv", seed=seed,
        ratio_a=0.8, ratio_b=0.9, sampling_rate_list="200",
    )
    data = MultiDatasetsLoader(
        args=args, root_path=data_root, flag="VAL"
    )
    return data, collate_fn


def score(model, data, collate_fn, batch_size: int, device, channel_index):
    loader = DataLoader(
        data, batch_size=batch_size, shuffle=False, num_workers=0,
        collate_fn=lambda items: collate_fn(items, max_len=data.max_seq_len),
    )
    probabilities, labels, subjects = [], [], []
    with torch.inference_mode():
        for signals, label_id, padding_mask in loader:
            signals = signals.to(device, dtype=torch.float32)
            if channel_index is not None:
                signals[:, :, channel_index] = 0.0
            logits = model(
                signals, padding_mask.to(device, dtype=torch.float32),
                None, None, label_id[:, 2], None,
            )
            probabilities.extend(torch.softmax(logits, dim=1)[:, 1].cpu().tolist())
            labels.extend(label_id[:, 0].long().tolist())
            subjects.extend(label_id[:, 1].long().tolist())

    probabilities = np.asarray(probabilities)
    labels = np.asarray(labels)
    subjects = np.asarray(subjects)
    predicted = (probabilities >= 0.5).astype(int)
    subject_true, subject_predicted = [], []
    for sid in np.unique(subjects):
        chosen = subjects == sid
        true_values = np.unique(labels[chosen])
        if len(true_values) != 1:
            raise ValueError(f"Inconsistent labels for subject {sid}")
        subject_true.append(int(true_values[0]))
        # Match LEAD's Counter(...).most_common(1) voting rule, including ties.
        subject_predicted.append(Counter(predicted[chosen]).most_common(1)[0][0])
    if len(np.unique(labels)) != 2:
        raise ValueError("Validation split must contain both classes")
    return {
        "windows": len(labels), "subjects": len(subject_true),
        "window_accuracy": float(accuracy_score(labels, predicted)),
        "window_auroc": float(roc_auc_score(labels, probabilities)),
        "subject_accuracy": float(accuracy_score(subject_true, subject_predicted)),
    }


def check_baseline(lead_root: Path, seed: int, observed: dict) -> None:
    """Stop if our unmasked inference disagrees with LEAD's recorded validation."""
    results_path = (
        lead_root / "results" / "LEADv2" / "finetune" / "LEADv2"
        / "P-Base-F-MDD-Mumtaz-200Hz-MCCV5" / "results.txt"
    )
    contents = results_path.read_text(encoding="utf-8")
    marker = f"Model Setting: nh8_el12_dm128_df256_seed{seed}"
    blocks = contents.split(marker)
    if len(blocks) != 2:
        raise ValueError(f"Expected exactly one {marker} in {results_path}")
    block = blocks[1].split("Model Setting:", 1)[0]
    patterns = {
        "window_accuracy": r"Sample-level results:\s*Validation results --- Accuracy:\s*([0-9.]+)",
        "subject_accuracy": r"Subject-level results after majority voting:\s*Validation results --- Accuracy:\s*([0-9.]+)",
    }
    for name, pattern in patterns.items():
        found = re.search(pattern, block)
        if not found:
            raise ValueError(f"Missing recorded {name} for seed {seed}")
        expected = float(found.group(1))
        if abs(observed[name] - expected) > 0.005:
            raise RuntimeError(
                f"Unmasked {name}={observed[name]:.5f} differs from LEAD "
                f"record={expected:.5f}; do not interpret masked results"
            )


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    output_dir = Path(__file__).resolve().parent.parent / "output_Test"
    lead_root = args.lead_root.resolve()
    sys.path.insert(0, str(lead_root))
    meta_path = lead_root / "dataset" / "L400" / "MDD-Mumtaz" / "meta.json"
    import json
    with meta_path.open(encoding="utf-8") as stream:
        meta = json.load(stream)
    config = model_config(meta)
    channel_names = config.channel_names.split(",")
    if args.channels == "all":
        selected = channel_names
    else:
        selected = [name.strip() for name in args.channels.split(",")]
        if not selected or any(name not in channel_names for name in selected):
            raise ValueError(f"Choose channels from: {', '.join(channel_names)}")
        if len(selected) != len(set(selected)):
            raise ValueError("Duplicate channel in --channels")

    if args.output:
        filename = args.output.name
        if not filename.lower().endswith(".csv"):
            raise ValueError("--output must name a .csv file")
    else:
        channel_label = "all" if args.channels == "all" else "_".join(selected)
        filename = f"lead_mdd_val_seed{args.seed}_{channel_label}.csv"

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data, collate_fn = load_validation_data(lead_root, args.seed)
    model, checkpoint = load_model(lead_root, args.seed, meta, device)
    print(f"Checkpoint: {checkpoint}")
    print(f"Validation subject IDs: {data.datasets[0].val_ids}")
    print("Scoring unmasked baseline...")
    baseline = score(model, data, collate_fn, args.batch_size, device, None)
    check_baseline(lead_root, args.seed, baseline)
    rows = [{"seed": args.seed, "mask": "none", **baseline,
             "delta_window_accuracy": 0.0, "delta_window_auroc": 0.0,
             "delta_subject_accuracy": 0.0}]
    for name in selected:
        print(f"Scoring channel mask: {name}")
        result = score(
            model, data, collate_fn, args.batch_size,
            device, channel_names.index(name),
        )
        rows.append({
            "seed": args.seed, "mask": name, **result,
            "delta_window_accuracy": baseline["window_accuracy"] - result["window_accuracy"],
            "delta_window_auroc": baseline["window_auroc"] - result["window_auroc"],
            "delta_subject_accuracy": baseline["subject_accuracy"] - result["subject_accuracy"],
        })

    output_dir.mkdir(parents=True, exist_ok=True)
    requested = Path(filename)
    for run_number in range(1, 10000):
        suffix = "" if run_number == 1 else f"_run{run_number:02d}"
        output = output_dir / f"{requested.stem}{suffix}{requested.suffix}"
        try:
            # Exclusive creation prevents even concurrent runs from overwriting each other.
            with output.open("x", newline="", encoding="utf-8-sig") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            break
        except FileExistsError:
            continue
    else:
        raise RuntimeError(f"No unused output filename available in {output_dir}")
    print(f"Baseline: {baseline}")
    print(f"Saved: {output}")
    print("Positive delta means masking reduced performance; validate across seeds before ranking electrodes.")


if __name__ == "__main__":
    main()
