# LEADv2 code for the FYP

This folder is a code-only copy from the local `C:\Users\ROG\Desktop\LEAD` checkout, made for developing LEADv2 downstream models. The original checkout was not modified. Source Python files, `run.py`, `requirements.txt`, and `LICENSE` were copied without edits.

## Contents

- `models/`, `layers/`: LEADv2 and the original model/layer import dependencies.
- `data_provider/`: dataset loaders, including `ADFTD-RS`, `APAVA`, `RepOD`, and `MDD-Mumtaz`.
- `exp/`, `utils/`, `run.py`: original experiment entry points and utilities.
- `data_preprocessing/`: Python preprocessing scripts used for ADFTD-RS, RepOD, and MDD-Mumtaz.

The original `run.py` imports all registered baselines at startup, so the small `models/`, `layers/`, and `data_provider/` Python packages are kept intact rather than copying only `LEADv2.py`.

## Not copied

Raw EEG, generated datasets, checkpoints, downloads, results, figures, notebooks, shell scripts, and virtual environments are **not** in this folder. FYP scripts `../working/01`–`04` now default to this folder for **code**. Their default data, original checkpoint, and reference-result paths still point to the old LEAD checkout. These paths can be overridden with each script's command-line options. Migrating code alone does not make old checkpoint-based runs self-contained.

For a new experiment, run from this folder with the existing `E:\LEAD` environment, and pass an explicit `--root_path` pointing to the existing preprocessed dataset directory (for example `C:\Users\ROG\Desktop\LEAD\dataset\L400`). A finetuning run also needs an explicit `--checkpoints_path` pointing to its original weight file. Do not copy the large datasets or model weights into this code folder unless the experiment actually needs a portable copy.

Before changing a copied upstream file, keep a clear record of the FYP-specific change. The upstream license is retained in `LICENSE`; check its non-commercial/share-alike terms before redistribution.
