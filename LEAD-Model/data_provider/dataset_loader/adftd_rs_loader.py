"""Dataset loader for the OpenNeuro ds004504 resting-state EEG data."""

import random

import numpy as np

from data_provider.dataset_loader.base_loader import BaseLoader


SUPPORTED_TASKS = {"ad_vs_hc", "ftd_vs_hc", "ad_vs_ftd", "multi_class"}


def _classes_for_task(classify_choice):
    if classify_choice == "ad_vs_hc":
        return (0, 1)
    if classify_choice == "ftd_vs_hc":
        return (0, 2)
    if classify_choice == "ad_vs_ftd":
        return (1, 2)
    if classify_choice == "multi_class":
        return (0, 1, 2)
    raise ValueError(
        f"ADFTD-RS does not support classify_choice={classify_choice!r}. "
        f"Choose one of {sorted(SUPPORTED_TASKS)}."
    )


def get_id_list_adftd_rs(args, data_list, a=0.8, b=0.9):
    """Make a stratified subject-level split; segments are never split."""
    classes = _classes_for_task(args.classify_choice)
    ids_by_class = {
        label: [int(v) for v in data_list[data_list[:, 0] == label, 1]]
        for label in classes
    }
    all_ids = sorted(sid for ids in ids_by_class.values() for sid in ids)

    if args.cross_val in ("fixed", "mccv"):
        seed = 42 if args.cross_val == "fixed" else int(args.seed)
        rng = random.Random(seed)
        train_ids, val_ids, test_ids = [], [], []
        for label in classes:
            ids = ids_by_class[label][:]
            rng.shuffle(ids)
            cut_a = int(a * len(ids))
            cut_b = int(b * len(ids))
            train_ids.extend(ids[:cut_a])
            val_ids.extend(ids[cut_a:cut_b])
            test_ids.extend(ids[cut_b:])
        return all_ids, sorted(train_ids), sorted(val_ids), sorted(test_ids)

    if args.cross_val == "loso":
        test_ids = [all_ids[(int(args.seed) - 41) % len(all_ids)]]
        train_ids = [sid for sid in all_ids if sid not in test_ids]
        rng = random.Random(int(args.seed))
        rng.shuffle(train_ids)
        # Preserve the behavior used by the existing LEAD loaders.
        val_ids = train_ids[:]
        return all_ids, sorted(train_ids), sorted(val_ids), sorted(test_ids)

    raise ValueError("ADFTD-RS supports cross_val: fixed, mccv, or loso")


class ADFTDRSLoader(BaseLoader):
    def _get_id_lists(self, args, data_list, a, b):
        return get_id_list_adftd_rs(args, data_list, a, b)

    def _postprocess_labels(self, args, flag):
        if flag == "PRETRAIN":
            return

        classes = _classes_for_task(args.classify_choice)
        keep = np.isin(self.y[:, 0].astype(int), classes)
        self.indices = self.indices[keep]
        self.y = self.y[keep]

        if args.classify_choice == "ftd_vs_hc":
            # CN=0, FTD=1
            self.y[:, 0] = np.where(self.y[:, 0] == 2, 1, self.y[:, 0])
        elif args.classify_choice == "ad_vs_ftd":
            # FTD=0, AD=1
            self.y[:, 0] = np.where(self.y[:, 0] == 2, 0, self.y[:, 0])

