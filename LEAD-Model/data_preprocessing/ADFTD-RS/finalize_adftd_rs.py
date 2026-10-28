"""Validate completed ADFTD-RS memmaps and repair/write meta.json."""

import json
from pathlib import Path

import numpy as np


ROOT = Path(r"C:\Users\ROG\Desktop\LEAD\dataset\L400\ADFTD-RS")
T = 400
CHANNELS = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T7", "C3", "Cz",
    "C4", "T8", "P7", "P3", "Pz", "P4", "P8", "O1", "O2",
]

y_path = ROOT / "y.dat"
x_path = ROOT / "X.dat"
if not x_path.is_file() or not y_path.is_file():
    raise FileNotFoundError("X.dat or y.dat is missing")
if y_path.stat().st_size % (3 * 4):
    raise ValueError("Invalid y.dat size")

n = y_path.stat().st_size // (3 * 4)
expected_x_bytes = n * T * len(CHANNELS) * 4
if x_path.stat().st_size != expected_x_bytes:
    raise ValueError(f"X.dat size mismatch: {x_path.stat().st_size} != {expected_x_bytes}")

y = np.memmap(y_path, dtype=np.float32, mode="r", shape=(n, 3))
subjects = np.unique(y[:, 1]).astype(int)
rates = np.unique(y[:, 2]).astype(int)
labels = np.unique(y[:, 0]).astype(int)
if not np.array_equal(subjects, np.arange(1, 89)):
    raise ValueError(f"Unexpected subjects: {subjects}")
if not np.array_equal(rates, np.array([50, 100, 200])):
    raise ValueError(f"Unexpected sampling rates: {rates}")
if not np.array_equal(labels, np.array([0, 1, 2])):
    raise ValueError(f"Unexpected labels: {labels}")

subject_labels = {}
for sid in subjects:
    labs = np.unique(y[y[:, 1] == sid, 0]).astype(int)
    if len(labs) != 1:
        raise ValueError(f"Subject {sid} has inconsistent labels: {labs}")
    subject_labels[int(sid)] = int(labs[0])
counts = {label: sum(v == label for v in subject_labels.values()) for label in (0, 1, 2)}
if counts != {0: 29, 1: 36, 2: 23}:
    raise ValueError(f"Unexpected subject label counts: {counts}")

x = np.memmap(x_path, dtype=np.float32, mode="r", shape=(n, T, len(CHANNELS)))
for start in range(0, n, 4096):
    block = x[start:min(start + 4096, n)]
    if not np.isfinite(block).all():
        raise ValueError(f"Non-finite EEG values in block starting {start}")
del x, y

meta = {
    "N": int(n), "T": T, "C": len(CHANNELS),
    "SAMPLE_RATE_LIST": [200, 100, 50], "OVERLAP": 200, "STEP": 200,
    "channels": CHANNELS, "label_map": {"CN": 0, "AD": 1, "FTD": 2},
    "source": r"C:\Users\ROG\Desktop\OpenN",
    "source_variant": "dataset-provided derivatives",
    "X_path": "X.dat", "y_path": "y.dat",
}
with (ROOT / "meta.json").open("w", encoding="utf-8") as handle:
    json.dump(meta, handle, ensure_ascii=False, indent=2)

print("Validation complete")
print(f"N={n}, T={T}, C={len(CHANNELS)}")
print(f"Subjects={len(subjects)}, CN/AD/FTD={counts[0]}/{counts[1]}/{counts[2]}")
print(f"Sampling rates={rates.tolist()}")
print(f"Output={ROOT}")
