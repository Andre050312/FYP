r"""LEADv2 disease separation and healthy-control source audit.

Experiments:
  disease3  - ADFTD-RS: HC / AD / FTD (one 3-class head)
  source_hc - HC only: ADFTD-RS / MDD-Mumtaz / RepOD (dataset source)

Both use subject-disjoint 80/10/10 splits, 200 Hz, 19 channels, 400 samples.
The source experiment is a confounding audit, not a disease classifier.
No pretrained checkpoint is loaded and TEST data is never evaluated here.

Manual examples from FYP in the E:\LEAD environment:
  python .\working\06_lead_disease_separation.py --experiment disease3 --seed 41 --check-data
  python .\working\06_lead_disease_separation.py --experiment disease3 --seed 41 --epochs 1 --max-steps 1
  python .\working\06_lead_disease_separation.py --experiment source_hc --seed 41 --epochs 30
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset


FYP_ROOT = Path(__file__).resolve().parent.parent
DATASETS = ("ADFTD-RS", "MDD-Mumtaz", "RepOD")
CHANNELS = (
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T7", "C3", "Cz",
    "C4", "T8", "P7", "P3", "Pz", "P4", "P8", "O1", "O2",
)
LABELS = {"disease3": ("HC", "AD", "FTD"), "source_hc": DATASETS}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=tuple(LABELS), required=True)
    parser.add_argument("--lead-root", type=Path, default=FYP_ROOT / "LEAD-Model")
    parser.add_argument("--data-root", type=Path,
                        default=Path(r"C:\Users\ROG\Desktop\LEAD\dataset\L400"))
    parser.add_argument("--output-root", type=Path,
                        default=FYP_ROOT / "output_Test" / "disease_separation")
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--cross-val", choices=("mccv", "fixed"), default="mccv")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-steps", type=int,
                        help="Limit optimizer steps per epoch for a manual smoke run")
    parser.add_argument("--check-data", action="store_true",
                        help="Inspect metadata, labels and subject splits without training")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def metadata(data_root: Path, experiment: str) -> dict:
    result = {}
    names = ("ADFTD-RS",) if experiment == "disease3" else DATASETS
    for name in names:
        folder = data_root / name
        with (folder / "meta.json").open(encoding="utf-8") as stream:
            meta = json.load(stream)
        rates = meta.get("SAMPLE_RATE_LIST", meta.get("sampling_rate_list", []))
        if int(meta["T"]) != 400 or int(meta["C"]) != 19 or tuple(meta["channels"]) != CHANNELS:
            raise ValueError(f"{name}: expected 19 common channels and 400 samples")
        if 200 not in rates:
            raise ValueError(f"{name}: 200 Hz is unavailable")
        if not all((folder / file).is_file() for file in ("X.dat", "y.dat")):
            raise FileNotFoundError(f"{name}: X.dat or y.dat is missing")
        result[name] = {"windows_all_rates": int(meta["N"]), "T": 400, "C": 19,
                        "selected_sampling_rate": 200}
    return result


def loader_config(data_root: Path, name: str, choice: str, args: argparse.Namespace):
    return SimpleNamespace(
        root_path=str(data_root), no_normalize=False,
        pretraining_datasets="TDBRAIN", training_datasets=name,
        testing_datasets=name, classify_choice=choice,
        cross_val=args.cross_val, seed=args.seed,
        ratio_a=0.8, ratio_b=0.9, sampling_rate_list="200",
    )


class HealthySourceDataset(Dataset):
    """Select HC from one LEAD loader and replace disease label with source ID."""

    def __init__(self, base, source_id: int):
        self.base = base
        self.source_id = source_id
        self.indices = np.flatnonzero(base.y[:, 0].astype(int) == 0).tolist()
        if not self.indices:
            raise RuntimeError(f"No HC windows for source {source_id}")
        self.max_seq_len = base.max_seq_len

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        x, y = self.base[self.indices[index]]
        y = y.clone()
        y[0] = self.source_id
        # [source label, original subject ID, fs, source ID + 1]
        y = torch.cat((y, y.new_tensor([self.source_id + 1])))
        return x, y


def build_data(args: argparse.Namespace, data_root: Path):
    from data_provider.data_loader import MultiDatasetsLoader, data_folder_dict
    from data_provider.uea import collate_fn

    split_ids: dict[str, dict[str, list[int]]] = {}
    if args.experiment == "disease3":
        config = loader_config(data_root, "ADFTD-RS", "multi_class", args)
        datasets = {}
        for split in ("TRAIN", "VAL"):
            ds = MultiDatasetsLoader(config, str(data_root), flag=split)
            if ds.num_class != 3:
                raise RuntimeError(f"{split}: expected HC, AD, FTD")
            datasets[split] = ds
            raw = ds.datasets[0]
            split_ids[split] = {
                name: sorted(map(int, getattr(raw, name.lower() + "_ids")))
                for name in ("TRAIN", "VAL", "TEST")
            }
        splits = split_ids["TRAIN"]
        if splits != split_ids["VAL"]:
            raise RuntimeError("Split IDs changed between TRAIN and VAL loading")
        split_ids = {"ADFTD-RS": splits}
        actual = {
            split: set(datasets[split].datasets[0].y[:, 0].astype(int))
            for split in ("TRAIN", "VAL")
        }
        if any(labels != {0, 1, 2} for labels in actual.values()):
            raise RuntimeError(f"Unexpected disease labels: {actual}")
    else:
        datasets_by_split = {"TRAIN": [], "VAL": []}
        for source_id, name in enumerate(DATASETS):
            # MDD's loader only accepts mdd_vs_hc. The other two loaders
            # accept these choices and the wrapper retains HC windows only.
            choice = {"ADFTD-RS": "ad_vs_hc", "MDD-Mumtaz": "mdd_vs_hc",
                      "RepOD": "ad_vs_hc"}[name]
            config = loader_config(data_root, name, choice, args)
            raw_splits = {}
            for split in ("TRAIN", "VAL"):
                raw = data_folder_dict[name](config, str(data_root / name), flag=split)
                if raw.max_seq_len != 400 or raw.X.shape[2] != 19:
                    raise RuntimeError(f"{name}/{split}: incompatible shape")
                datasets_by_split[split].append(HealthySourceDataset(raw, source_id))
                raw_splits[split] = {
                    key: sorted(map(int, getattr(raw, key.lower() + "_ids")))
                    for key in ("TRAIN", "VAL", "TEST")
                }
            if raw_splits["TRAIN"] != raw_splits["VAL"]:
                raise RuntimeError(f"{name}: split IDs changed between loaders")
            # Record the subjects actually included in this HC-only audit.
            with (data_root / name / "meta.json").open(encoding="utf-8") as stream:
                count = int(json.load(stream)["N"])
            all_targets = np.memmap(data_root / name / "y.dat", dtype=np.float32,
                                    mode="r", shape=(count, 3))
            hc_ids = set(map(int, all_targets[all_targets[:, 0] == 0, 1]))
            split_ids[name] = {
                key: sorted(hc_ids & set(values))
                for key, values in raw_splits["TRAIN"].items()
            }
        datasets = {split: ConcatDataset(parts) for split, parts in datasets_by_split.items()}

    for name, splits in split_ids.items():
        sets = {key: set(values) for key, values in splits.items()}
        for left, right in (("TRAIN", "VAL"), ("TRAIN", "TEST"), ("VAL", "TEST")):
            if sets[left] & sets[right]:
                raise RuntimeError(f"{name}: {left}/{right} subject leakage")
        print(f"{name}: subject counts TRAIN={len(sets['TRAIN'])}, "
              f"VAL={len(sets['VAL'])}, TEST={len(sets['TEST'])}")
    print(f"Selected windows: TRAIN={len(datasets['TRAIN'])}, VAL={len(datasets['VAL'])}")
    return datasets, split_ids, collate_fn


def make_loader(ds, collate_fn, batch_size: int, seed: int, train: bool):
    return DataLoader(
        ds, batch_size=batch_size, shuffle=train, drop_last=train,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0, pin_memory=torch.cuda.is_available(),
        collate_fn=lambda items: collate_fn(items, max_len=400),
    )


def make_model(lead_root: Path):
    from models.LEADv2 import Model
    config = SimpleNamespace(
        task_name="supervised", output_attention=False,
        patch_len=50, stride=50, enc_in=19, seq_len=400,
        d_model=128, n_heads=8, e_layers=12, d_ff=256,
        dropout=0.1, activation="gelu", channel_names=",".join(CHANNELS),
        montage_name="standard_1005", augmentations="patch", num_class=3,
    )
    return Model(config)


def summarize_predictions(y, probs, keys, class_names):
    y = np.asarray(y, dtype=int)
    probs = np.asarray(probs, dtype=float)
    pred = probs.argmax(axis=1)
    groups = {}
    for index, key in enumerate(keys):
        groups.setdefault(key, []).append(index)
    subject_rows = []
    for key, indices in groups.items():
        labels = np.unique(y[indices])
        if len(labels) != 1:
            raise RuntimeError(f"Conflicting labels for {key}")
        mean_prob = probs[indices].mean(axis=0)
        subject_rows.append({
            "subject_key": key, "true_label": int(labels[0]),
            "predicted_label": int(mean_prob.argmax()),
            "window_count": len(indices),
            **{f"p_{name}": float(mean_prob[i]) for i, name in enumerate(class_names)},
        })
    subject_y = np.asarray([row["true_label"] for row in subject_rows])
    subject_pred = np.asarray([row["predicted_label"] for row in subject_rows])
    subject_prob = np.asarray([[row[f"p_{name}"] for name in class_names]
                               for row in subject_rows])
    if set(np.unique(y)) != set(range(3)) or set(np.unique(subject_y)) != set(range(3)):
        raise RuntimeError("Validation needs all three classes for macro metrics")
    metrics = {
        "window_count": len(y), "subject_count": len(subject_y),
        "window_accuracy": float(accuracy_score(y, pred)),
        "window_macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "window_auroc_ovr": float(roc_auc_score(y, probs, multi_class="ovr", average="macro")),
        "subject_accuracy": float(accuracy_score(subject_y, subject_pred)),
        "subject_macro_f1": float(f1_score(subject_y, subject_pred, average="macro", zero_division=0)),
        "subject_auroc_ovr": float(roc_auc_score(subject_y, subject_prob,
                                                 multi_class="ovr", average="macro")),
        "subject_confusion_matrix": confusion_matrix(subject_y, subject_pred, labels=[0, 1, 2]).tolist(),
    }
    return metrics, subject_rows


@torch.no_grad()
def evaluate(model, loader, device, class_names):
    model.eval()
    ys, probs, keys = [], [], []
    for signals, target, padding_mask in loader:
        logits = model(signals.to(device, dtype=torch.float32), padding_mask,
                       None, None, target[:, 2], None)
        ys.extend(target[:, 0].long().tolist())
        probs.extend(torch.softmax(logits, dim=1).cpu().tolist())
        # Include source ID: different datasets may reuse numerical subject IDs.
        keys.extend(f"{int(source)}:{int(subject)}"
                    for subject, source in zip(target[:, 1], target[:, 3]))
    return summarize_predictions(ys, probs, keys, class_names)


def main() -> None:
    args = arguments()
    if args.seed < 0 or min(args.epochs, args.patience, args.batch_size) < 1:
        raise ValueError("Seed must be nonnegative; epochs, patience and batch size positive")
    if args.learning_rate <= 0 or (args.max_steps is not None and args.max_steps < 1):
        raise ValueError("Learning rate and max steps must be positive")
    lead_root = args.lead_root.resolve()
    if not (lead_root / "models" / "LEADv2.py").is_file():
        raise FileNotFoundError(f"LEADv2 code not found under {lead_root}")
    sys.path.insert(0, str(lead_root))
    data_root = args.data_root.resolve()
    meta = metadata(data_root, args.experiment)
    set_seed(args.seed)
    datasets, split_ids, collate_fn = build_data(args, data_root)
    if args.check_data:
        print("Data checks passed. No model built or training started.")
        return

    loaders = {split: make_loader(ds, collate_fn, args.batch_size, args.seed,
                                  train=(split == "TRAIN"))
               for split, ds in datasets.items()}
    if len(loaders["TRAIN"]) == 0:
        raise RuntimeError("Training split has fewer windows than batch size")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = make_model(lead_root).to(device)  # Random initialization.
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    best_f1, best_epoch, best_state, stale = -1.0, 0, None, 0
    history = []
    class_names = LABELS[args.experiment]
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for step, (signals, target, padding_mask) in enumerate(loaders["TRAIN"], 1):
            logits = model(signals.to(device, dtype=torch.float32), padding_mask,
                           None, None, target[:, 2], None)
            loss = nn.functional.cross_entropy(logits, target[:, 0].long().to(device))
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 4.0)
            optimizer.step()
            losses.append(float(loss.item()))
            if args.max_steps is not None and step >= args.max_steps:
                break
        val, _ = evaluate(model, loaders["VAL"], device, class_names)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)),
                        **{f"val_{name}": value for name, value in val.items()
                           if name != "subject_confusion_matrix"}})
        print(f"Epoch {epoch}: loss={np.mean(losses):.4f}, "
              f"VAL window macro-F1={val['window_macro_f1']:.4f}, "
              f"VAL subject accuracy={val['subject_accuracy']:.4f}")
        if val["window_macro_f1"] > best_f1 + 1e-5:
            best_f1, best_epoch, stale = val["window_macro_f1"], epoch, 0
            best_state = {key: value.detach().cpu().clone()
                          for key, value in model.state_dict().items()}
        else:
            stale += 1
        scheduler.step()
        if stale >= args.patience:
            break

    if best_state is None:
        raise RuntimeError("No checkpoint selected")
    model.load_state_dict(best_state)
    val, subject_rows = evaluate(model, loaders["VAL"], device, class_names)
    run_dir = args.output_root.resolve() / args.experiment / (
        f"seed{args.seed}_{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:8]}")
    run_dir.mkdir(parents=True, exist_ok=False)
    torch.save(best_state, run_dir / "checkpoint.pth")
    with (run_dir / "history.csv").open("x", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    with (run_dir / "val_subject_predictions.csv").open("x", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(subject_rows[0]))
        writer.writeheader()
        writer.writerows(subject_rows)
    summary = {
        "experiment": args.experiment, "class_names": class_names,
        "seed": args.seed, "cross_val": args.cross_val,
        "model": "LEADv2 random-initialization 3-class supervised",
        "data_root": str(data_root), "metadata": meta, "subject_ids": split_ids,
        "best_epoch": best_epoch, "epochs_completed": len(history),
        "hyperparameters": {"epochs": args.epochs, "patience": args.patience,
                            "batch_size": args.batch_size, "learning_rate": args.learning_rate,
                            "max_steps": args.max_steps},
        "validation": val, "test": None,
    }
    (run_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved {run_dir}")


if __name__ == "__main__":
    main()
