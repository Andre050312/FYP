r"""Train LEADv2 from random initialization on MDD-Mumtaz.

No published/pretrained/fine-tuned checkpoint is loaded. For a masked run,
the chosen electrode is set to zero after per-window normalization in TRAIN,
VAL, and TEST. Splits are subject-disjoint and stratified by diagnosis.

Run from the FYP directory with the E:\LEAD environment activated, e.g.:
    python .\working\02_lead_mdd_train_from_scratch.py --seeds 41:55 --mask none
    python .\working\02_lead_mdd_train_from_scratch.py --seeds 41:55 --mask P3

Each invocation creates a separate run directory under output_Test/from_scratch.
By default TEST is not evaluated, so electrode selection can use VAL only.
Add --evaluate-test only for a final, pre-specified comparison.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader


CHANNELS = (
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T7", "C3", "Cz",
    "C4", "T8", "P7", "P3", "Pz", "P4", "P8", "O1", "O2",
)
DEFAULT_LEAD_ROOT = Path(r"C:\Users\ROG\Desktop\LEAD")


def parse_seeds(value: str) -> list[int]:
    if ":" in value:
        parts = value.split(":")
        if len(parts) != 2:
            raise argparse.ArgumentTypeError("Use FIRST:LAST, inclusive")
        first, last = map(int, parts)
        if first > last:
            raise argparse.ArgumentTypeError("FIRST must not exceed LAST")
        seeds = list(range(first, last + 1))
    else:
        seeds = [int(part.strip()) for part in value.split(",")]
    if not seeds or len(seeds) != len(set(seeds)) or any(s < 0 for s in seeds):
        raise argparse.ArgumentTypeError("Seeds must be unique non-negative integers")
    return seeds


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lead-root", type=Path, default=DEFAULT_LEAD_ROOT)
    parser.add_argument("--data-root", type=Path,
                        help="Parent directory containing MDD-Mumtaz; defaults to LEAD/dataset/L400")
    parser.add_argument("--output-root", type=Path,
                        help="Run directory parent; defaults to FYP/output_Test/from_scratch")
    parser.add_argument("--seeds", type=parse_seeds, default=[41], help="E.g. 41:55 or 41,43,45")
    parser.add_argument("--mask", choices=("none", *CHANNELS), default="none")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--loss", choices=("subject", "sample"), default="subject")
    parser.add_argument("--evaluate-test", action="store_true",
                        help="Evaluate held-out TEST after training; omit while selecting electrodes")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_datasets(data_root: Path, seed: int):
    from data_provider.data_loader import MultiDatasetsLoader
    from data_provider.uea import collate_fn

    config = SimpleNamespace(
        root_path=str(data_root), no_normalize=False,
        pretraining_datasets="TDBRAIN", training_datasets="MDD-Mumtaz",
        testing_datasets="MDD-Mumtaz", classify_choice="mdd_vs_hc",
        cross_val="mccv", seed=seed, ratio_a=0.8, ratio_b=0.9,
        sampling_rate_list="200",
    )
    datasets = {
        name: MultiDatasetsLoader(config, str(data_root), flag=name)
        for name in ("TRAIN", "VAL", "TEST")
    }
    splits = {
        name: set(getattr(datasets[name].datasets[0], f"{name.lower()}_ids"))
        for name in datasets
    }
    if any(splits[a] & splits[b] for a, b in (("TRAIN", "VAL"), ("TRAIN", "TEST"), ("VAL", "TEST"))):
        raise RuntimeError("Subject leakage detected between data splits")
    if any(len(datasets[name]) == 0 for name in datasets):
        raise RuntimeError("An empty data split was produced")
    if any(datasets[name].enc_in != 19 or datasets[name].max_seq_len != 400 for name in datasets):
        raise RuntimeError("Expected MDD data with 19 channels and 400 samples")
    return datasets, splits, collate_fn


def make_loader(dataset, collate_fn, batch_size: int, seed: int, train: bool):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=train, drop_last=train,
        num_workers=0, pin_memory=torch.cuda.is_available(), generator=generator,
        collate_fn=lambda items: collate_fn(items, max_len=dataset.max_seq_len),
    )


def build_model(meta: dict):
    from models.LEADv2 import Model

    if tuple(meta["channels"]) != CHANNELS or int(meta["T"]) != 400:
        raise RuntimeError("Dataset channel order/length does not match this script")
    if list(meta["SAMPLE_RATE_LIST"]) != [200]:
        raise RuntimeError("Expected 200-Hz MDD data")
    config = SimpleNamespace(
        task_name="supervised", output_attention=False, patch_len=50, stride=50,
        enc_in=19, seq_len=400, d_model=128, n_heads=8, e_layers=12, d_ff=256,
        dropout=0.1, activation="gelu", channel_names=",".join(CHANNELS),
        montage_name="standard_1005", augmentations="patch", num_class=2,
    )
    return Model(config)


def prepare_batch(batch, device: torch.device, mask_index: int | None):
    signals, label_id, padding_mask = batch
    signals = signals.to(device, dtype=torch.float32)
    if mask_index is not None:
        signals[:, :, mask_index] = 0.0
    return signals, label_id, padding_mask.to(device, dtype=torch.float32)


def metrics(labels: list[int], probabilities: list[float], subjects: list[int]) -> tuple[dict, list[dict]]:
    y = np.asarray(labels)
    p = np.asarray(probabilities)
    sid = np.asarray(subjects)
    predicted = (p >= 0.5).astype(int)
    subject_rows = []
    for subject_id in np.unique(sid):
        chosen = sid == subject_id
        true_labels = np.unique(y[chosen])
        if len(true_labels) != 1:
            raise RuntimeError(f"Inconsistent labels for subject {subject_id}")
        subject_rows.append({
            "subject_id": int(subject_id), "true_label": int(true_labels[0]),
            "mean_probability": float(p[chosen].mean()),
            "majority_vote": int(Counter(predicted[chosen]).most_common(1)[0][0]),
            "window_count": int(chosen.sum()),
        })
    sy = np.asarray([row["true_label"] for row in subject_rows])
    sp = np.asarray([row["majority_vote"] for row in subject_rows])
    sm = np.asarray([row["mean_probability"] for row in subject_rows])
    if len(np.unique(y)) != 2 or len(np.unique(sy)) != 2:
        raise RuntimeError("Evaluation split must contain both classes")
    result = {
        "window_count": len(y), "subject_count": len(sy),
        "window_accuracy": float(accuracy_score(y, predicted)),
        "window_macro_f1": float(f1_score(y, predicted, average="macro", zero_division=0)),
        "window_auroc": float(roc_auc_score(y, p)),
        "subject_accuracy": float(accuracy_score(sy, sp)),
        "subject_balanced_accuracy": float(balanced_accuracy_score(sy, sp)),
        "subject_macro_f1": float(f1_score(sy, sp, average="macro", zero_division=0)),
        "subject_auroc_mean_probability": float(roc_auc_score(sy, sm)),
    }
    return result, subject_rows


@torch.no_grad()
def evaluate(model, loader, device: torch.device, mask_index: int | None):
    model.eval()
    labels, probabilities, subjects = [], [], []
    for batch in loader:
        signals, label_id, padding_mask = prepare_batch(batch, device, mask_index)
        logits = model(signals, padding_mask, None, None, label_id[:, 2], None)
        probabilities.extend(torch.softmax(logits, dim=1)[:, 1].cpu().tolist())
        labels.extend(label_id[:, 0].long().tolist())
        subjects.extend(label_id[:, 1].long().tolist())
    return metrics(labels, probabilities, subjects)


def train_one(seed: int, args: argparse.Namespace, data_root: Path, meta: dict, output_root: Path) -> Path:
    from utils.losses import subject_ce_loss

    set_seed(seed)
    datasets, splits, collate_fn = load_datasets(data_root, seed)
    if len(datasets["TRAIN"]) < args.batch_size:
        raise RuntimeError("Training split has fewer windows than --batch-size")
    loaders = {
        name: make_loader(datasets[name], collate_fn, args.batch_size, seed, name == "TRAIN")
        for name in datasets
    }
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = build_model(meta).to(device)  # Random initialization; no load_state_dict.
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    mask_index = None if args.mask == "none" else CHANNELS.index(args.mask)
    best_f1, best_state, best_epoch, stale = -1.0, None, 0, 0
    history = []

    print(f"Seed {seed}, mask={args.mask}, device={device}, split subjects="
          f"{ {name: len(ids) for name, ids in splits.items()} }")
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in loaders["TRAIN"]:
            signals, label_id, padding_mask = prepare_batch(batch, device, mask_index)
            logits = model(signals, padding_mask, None, None, label_id[:, 2], None)
            labels = label_id[:, 0].long().to(device)
            if args.loss == "subject":
                loss = subject_ce_loss(logits, labels, label_id[:, 1])
            else:
                loss = torch.nn.functional.cross_entropy(logits, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=4.0)
            optimizer.step()
            losses.append(float(loss.item()))

        val_metrics, _ = evaluate(model, loaders["VAL"], device, mask_index)
        val_f1 = val_metrics["window_macro_f1"]
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), **{
            f"val_{key}": value for key, value in val_metrics.items()
        }})
        print(f"  Epoch {epoch:02d}: train_loss={np.mean(losses):.4f}, "
              f"val_macro_f1={val_f1:.4f}, val_subject_acc={val_metrics['subject_accuracy']:.4f}")
        if val_f1 > best_f1 + 1e-5:
            best_f1, best_epoch, stale = val_f1, epoch, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
        scheduler.step()
        if stale >= args.patience:
            break

    if best_state is None:
        raise RuntimeError("No valid model checkpoint selected")
    model.load_state_dict(best_state)
    val_metrics, val_subjects = evaluate(model, loaders["VAL"], device, mask_index)
    test_metrics, test_subjects = (evaluate(model, loaders["TEST"], device, mask_index)
                                   if args.evaluate_test else (None, []))

    # A unique directory keeps even repeated/concurrent runs separate.
    run_dir = output_root / f"seed{seed}_{args.mask}_{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:8]}"
    run_dir.mkdir(parents=True, exist_ok=False)
    torch.save(best_state, run_dir / "checkpoint.pth")
    with (run_dir / "history.csv").open("x", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    with (run_dir / "subject_predictions.csv").open("x", newline="", encoding="utf-8-sig") as stream:
        rows = [{"split": split, **row} for split, group in (("VAL", val_subjects), ("TEST", test_subjects)) for row in group]
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "method": "LEADv2 random-initialization supervised training",
        "seed": seed, "mask": args.mask, "mask_stage": "post-normalization in TRAIN/VAL/TEST",
        "dataset": "MDD-Mumtaz", "split": "stratified subject-disjoint MCCV 80/10/10",
        "data_root": str(data_root),
        "frequency_variant": meta.get("frequency_variant", "broadband"),
        "subject_ids": {name: sorted(map(int, ids)) for name, ids in splits.items()},
        "best_epoch": best_epoch, "epochs_completed": len(history),
        "hyperparameters": {"epochs": args.epochs, "patience": args.patience,
                            "batch_size": args.batch_size, "learning_rate": args.learning_rate,
                            "loss": args.loss},
        "validation": val_metrics, "test": test_metrics,
    }
    (run_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved {run_dir}; validation subject accuracy={val_metrics['subject_accuracy']:.4f}")
    if test_metrics is not None:
        print(f"  Final test subject accuracy={test_metrics['subject_accuracy']:.4f}")
    return run_dir


def main() -> None:
    pause_file = Path(__file__).resolve().parent / "PAUSE_LEAD_SCRATCH"
    if pause_file.exists():
        raise SystemExit(
            f"Training paused before starting a new run. Remove {pause_file} to resume."
        )
    args = arguments()
    if args.epochs < 1 or args.patience < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        raise ValueError("Epochs, patience, batch size, and learning rate must be positive")
    lead_root = args.lead_root.resolve()
    data_root = (args.data_root or (lead_root / "dataset" / "L400")).resolve()
    meta_path = data_root / "MDD-Mumtaz" / "meta.json"
    with meta_path.open(encoding="utf-8") as stream:
        meta = json.load(stream)
    sys.path.insert(0, str(lead_root))
    output_root = (args.output_root or
                   (Path(__file__).resolve().parent.parent / "output_Test" / "from_scratch")).resolve()
    for seed in args.seeds:
        train_one(seed, args, data_root, meta, output_root)


if __name__ == "__main__":
    main()
