r"""Train four binary tasks with a shared LEADv2 encoder and separate heads.

AD and FTD: ADFTD-RS; MDD: MDD-Mumtaz; SZ: RepOD.
This script uses subject-disjoint splits and checks the shared ADFTD-RS
subjects across both tasks. It trains from random initialization, selects
the checkpoint on validation macro-F1, and does not evaluate TEST.

From FYP with E:\LEAD activated:
  python .\working\05_lead_four_task.py --check-data --seed 41
  python .\working\05_lead_four_task.py --seed 41 --epochs 1 --max-steps 1
  python .\working\05_lead_four_task.py --seed 41 --epochs 30
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
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader


FYP_ROOT = Path(__file__).resolve().parent.parent
CHANNELS = (
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T7", "C3", "Cz",
    "C4", "T8", "P7", "P3", "Pz", "P4", "P8", "O1", "O2",
)
TASKS = {
    "AD": ("ADFTD-RS", "ad_vs_hc"),
    "FTD": ("ADFTD-RS", "ftd_vs_hc"),
    "MDD": ("MDD-Mumtaz", "mdd_vs_hc"),
    "SZ": ("RepOD", "ad_vs_hc"),  # Legacy loader choice; RepOD label 1 means SZ.
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lead-root", type=Path, default=FYP_ROOT / "LEAD-Model")
    parser.add_argument("--data-root", type=Path,
                        default=Path(r"C:\Users\ROG\Desktop\LEAD\dataset\L400"))
    parser.add_argument("--output-root", type=Path,
                        default=FYP_ROOT / "output_Test" / "four_task")
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--cross-val", choices=("mccv", "fixed"), default="mccv")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-steps", type=int,
                        help="Cap joint optimizer steps per epoch for a smoke run")
    parser.add_argument("--check-data", action="store_true",
                        help="Check metadata, labels, and subject splits without training")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def inspect_metadata(data_root: Path) -> dict:
    metadata = {}
    for name in sorted({dataset for dataset, _ in TASKS.values()}):
        folder = data_root / name
        with (folder / "meta.json").open(encoding="utf-8") as stream:
            meta = json.load(stream)
        if int(meta["C"]) != 19 or int(meta["T"]) != 400 or tuple(meta["channels"]) != CHANNELS:
            raise ValueError(f"{name}: expected 19 channels in the common order and 400 samples")
        rates = meta.get("SAMPLE_RATE_LIST", meta.get("sampling_rate_list", []))
        if 200 not in rates:
            raise ValueError(f"{name}: no 200 Hz data declared")
        if not (folder / "X.dat").is_file() or not (folder / "y.dat").is_file():
            raise FileNotFoundError(f"{name}: X.dat or y.dat is missing")
        metadata[name] = {"N": int(meta["N"]), "C": 19, "T": 400, "selected_rate": 200}
    return metadata


def load_tasks(data_root: Path, seed: int, cross_val: str):
    from data_provider.data_loader import MultiDatasetsLoader
    from data_provider.uea import collate_fn

    datasets, subject_ids = {}, {}
    for task, (name, choice) in TASKS.items():
        config = SimpleNamespace(
            root_path=str(data_root), no_normalize=False,
            pretraining_datasets="TDBRAIN", training_datasets=name,
            testing_datasets=name, classify_choice=choice,
            cross_val=cross_val, seed=seed, ratio_a=0.8, ratio_b=0.9,
            sampling_rate_list="200",
        )
        datasets[task], subject_ids[task] = {}, {}
        for split in ("TRAIN", "VAL"):
            ds = MultiDatasetsLoader(config, str(data_root), flag=split)
            if ds.enc_in != 19 or ds.max_seq_len != 400 or ds.num_class != 2:
                raise ValueError(f"{task}/{split}: unexpected input shape or class count")
            raw = ds.datasets[0]
            if set(raw.y[:, 0].astype(int)) != {0, 1}:
                raise ValueError(f"{task}/{split}: expected binary labels 0 and 1")
            datasets[task][split] = ds
            subject_ids[task][split] = set(map(int, getattr(raw, split.lower() + "_ids")))
            subject_ids[task]["TEST"] = set(map(int, raw.test_ids))
        for a, b in (("TRAIN", "VAL"), ("TRAIN", "TEST"), ("VAL", "TEST")):
            if subject_ids[task][a] & subject_ids[task][b]:
                raise RuntimeError(f"{task}: {a}/{b} subject leakage")

    # AD and FTD are drawn from the same source. No source subject may cross
    # splits merely because its diagnosis is used in a different binary task.
    for a in ("TRAIN", "VAL", "TEST"):
        for b in ("TRAIN", "VAL", "TEST"):
            if a != b and subject_ids["AD"][a] & subject_ids["FTD"][b]:
                raise RuntimeError(f"AD/FTD cross-task {a}/{b} subject leakage")
    for task in TASKS:
        print(f"{task}: TRAIN {len(subject_ids[task]['TRAIN'])} subjects / "
              f"{len(datasets[task]['TRAIN'])} windows; VAL "
              f"{len(subject_ids[task]['VAL'])} subjects / {len(datasets[task]['VAL'])} windows")
    return datasets, subject_ids, collate_fn


def make_loader(ds, collate_fn, batch_size: int, seed: int, train: bool):
    return DataLoader(
        ds, batch_size=batch_size, shuffle=train, drop_last=train,
        num_workers=0, pin_memory=torch.cuda.is_available(),
        generator=torch.Generator().manual_seed(seed),
        collate_fn=lambda rows: collate_fn(rows, max_len=ds.max_seq_len),
    )


class FourTaskLEAD(nn.Module):
    """Original LEADv2 features, followed by one linear head per task."""

    def __init__(self):
        super().__init__()
        from models.LEADv2 import Model
        config = SimpleNamespace(
            task_name="supervised", output_attention=False,
            patch_len=50, stride=50, enc_in=19, seq_len=400,
            d_model=128, n_heads=8, e_layers=12, d_ff=256,
            dropout=0.1, activation="gelu", channel_names=",".join(CHANNELS),
            montage_name="standard_1005", augmentations="patch", num_class=2,
        )
        base = Model(config)  # No checkpoint is loaded.
        self.embedding = base.enc_embedding
        self.encoder = base.encoder
        self.dropout = base.dropout
        self.heads = nn.ModuleDict({task: nn.Linear(base.input_dim, 2) for task in TASKS})

    def forward(self, signals: torch.Tensor, rates: torch.Tensor, task: str) -> torch.Tensor:
        tokens = self.embedding(signals, rates)
        tokens, _, _ = self.encoder(tokens, attn_mask=None)
        batch, channel_patches, dim = tokens.shape
        if channel_patches % 19:
            raise RuntimeError(f"Unexpected token count {channel_patches}")
        cls = tokens.reshape(batch, 19, channel_patches // 19, dim)[:, :, -1, :]
        features = self.dropout(F.gelu(cls)).reshape(batch, -1)
        return self.heads[task](features)


@torch.no_grad()
def evaluate(model, loader, task: str, device: torch.device) -> dict:
    model.eval()
    labels, probabilities, subjects = [], [], []
    for signals, target, _ in loader:
        logits = model(signals.to(device, dtype=torch.float32), target[:, 2], task)
        probabilities.extend(torch.softmax(logits, dim=1)[:, 1].cpu().tolist())
        labels.extend(target[:, 0].long().tolist())
        subjects.extend(target[:, 1].long().tolist())
    y, p, sid = np.asarray(labels), np.asarray(probabilities), np.asarray(subjects)
    subject_y, subject_p = [], []
    for subject in np.unique(sid):
        chosen = sid == subject
        unique_labels = np.unique(y[chosen])
        if len(unique_labels) != 1:
            raise RuntimeError(f"{task}: inconsistent labels for subject {subject}")
        subject_y.append(int(unique_labels[0]))
        subject_p.append(float(p[chosen].mean()))
    subject_y, subject_p = np.asarray(subject_y), np.asarray(subject_p)
    if len(np.unique(y)) != 2 or len(np.unique(subject_y)) != 2:
        raise RuntimeError(f"{task}: validation split must contain both classes")
    return {
        "windows": int(len(y)), "subjects": int(len(subject_y)),
        "window_macro_f1": float(f1_score(y, p >= 0.5, average="macro", zero_division=0)),
        "window_auroc": float(roc_auc_score(y, p)),
        "subject_accuracy_mean_probability": float(accuracy_score(subject_y, subject_p >= 0.5)),
        "subject_auroc_mean_probability": float(roc_auc_score(subject_y, subject_p)),
    }


def main() -> None:
    args = parse_args()
    if args.seed < 0 or min(args.epochs, args.patience, args.batch_size) < 1:
        raise ValueError("Seed must be nonnegative; epochs, patience, batch size must be positive")
    if args.learning_rate <= 0 or (args.max_steps is not None and args.max_steps < 1):
        raise ValueError("Learning rate and max steps must be positive")
    lead_root, data_root = args.lead_root.resolve(), args.data_root.resolve()
    if not (lead_root / "models" / "LEADv2.py").is_file():
        raise FileNotFoundError(f"LEADv2 source missing under {lead_root}")
    sys.path.insert(0, str(lead_root))
    metadata = inspect_metadata(data_root)
    set_seed(args.seed)
    datasets, subject_ids, collate_fn = load_tasks(data_root, args.seed, args.cross_val)
    if args.check_data:
        print("Data checks passed. No model built or training started.")
        return

    loaders = {
        task: {split: make_loader(ds, collate_fn, args.batch_size, args.seed, split == "TRAIN")
               for split, ds in splits.items()}
        for task, splits in datasets.items()
    }
    if any(len(loaders[task]["TRAIN"]) == 0 for task in TASKS):
        raise RuntimeError("A training task has fewer windows than batch size")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = FourTaskLEAD().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    best_score, best_epoch, best_state, stale = -1.0, 0, None, 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        iters = {task: iter(loaders[task]["TRAIN"]) for task in TASKS}
        steps = max(len(loaders[task]["TRAIN"]) for task in TASKS)
        if args.max_steps is not None:
            steps = min(steps, args.max_steps)
        losses = []
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            task_losses = []
            for task in TASKS:
                try:
                    signals, target, _ = next(iters[task])
                except StopIteration:
                    iters[task] = iter(loaders[task]["TRAIN"])
                    signals, target, _ = next(iters[task])
                logits = model(signals.to(device, dtype=torch.float32), target[:, 2], task)
                task_loss = F.cross_entropy(logits, target[:, 0].long().to(device))
                if not torch.isfinite(task_loss):
                    raise RuntimeError(f"Non-finite loss in {task}")
                task_losses.append(float(task_loss.item()))
                (task_loss / len(TASKS)).backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=4.0)
            optimizer.step()
            losses.append(float(np.mean(task_losses)))

        val = {task: evaluate(model, loaders[task]["VAL"], task, device) for task in TASKS}
        score = float(np.mean([val[task]["window_macro_f1"] for task in TASKS]))
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)),
                        "mean_val_window_macro_f1": score,
                        **{f"{task}_val_window_macro_f1": val[task]["window_macro_f1"] for task in TASKS}})
        print(f"Epoch {epoch}: mean VAL macro-F1={score:.4f}; " +
              ", ".join(f"{task}={val[task]['window_macro_f1']:.4f}" for task in TASKS))
        if score > best_score + 1e-5:
            best_score, best_epoch, stale = score, epoch, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
        scheduler.step()
        if stale >= args.patience:
            break

    if best_state is None:
        raise RuntimeError("No checkpoint selected")
    model.load_state_dict(best_state)
    val = {task: evaluate(model, loaders[task]["VAL"], task, device) for task in TASKS}
    run_dir = args.output_root.resolve() / f"seed{args.seed}_{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:8]}"
    run_dir.mkdir(parents=True, exist_ok=False)
    torch.save(best_state, run_dir / "checkpoint.pth")
    with (run_dir / "history.csv").open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    summary = {
        "model": "LEADv2 shared encoder and four binary heads, random initialization",
        "seed": args.seed, "cross_val": args.cross_val,
        "tasks": {task: {"dataset": name, "choice": choice} for task, (name, choice) in TASKS.items()},
        "metadata": metadata,
        "subject_ids": {task: {split: sorted(ids) for split, ids in splits.items()}
                        for task, splits in subject_ids.items()},
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
