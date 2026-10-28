"""LEAD loader for the Mumtaz eyes-closed MDD vs healthy-control dataset."""

import random

import numpy as np

from data_provider.dataset_loader.base_loader import BaseLoader


def get_id_list_mdd_mumtaz(args, data_list: np.ndarray, a=0.8, b=0.9):
    """Stratified, subject-disjoint fixed or Monte Carlo split.

    The same sorted-ID, single-RNG procedure is used by the LaBraM adapter.
    """
    if args.classify_choice != "mdd_vs_hc":
        raise ValueError("MDD-Mumtaz requires --classify_choice mdd_vs_hc")
    if args.cross_val not in ("fixed", "mccv"):
        raise ValueError("MDD-Mumtaz supports cross_val fixed or mccv")
    labels = set(data_list[:, 0].astype(int))
    if labels != {0, 1}:
        raise ValueError(f"Expected HC=0 and MDD=1, got labels {labels}")

    rng = random.Random(42 if args.cross_val == "fixed" else int(args.seed))
    train, val, test = [], [], []
    for label in (0, 1):
        ids = sorted(int(v) for v in data_list[data_list[:, 0] == label, 1])
        rng.shuffle(ids)
        cut_a, cut_b = int(a * len(ids)), int(b * len(ids))
        if not 0 < cut_a < cut_b < len(ids):
            raise ValueError(f"Empty MDD-Mumtaz split for label {label}; adjust ratios")
        train.extend(ids[:cut_a])
        val.extend(ids[cut_a:cut_b])
        test.extend(ids[cut_b:])
    all_ids = sorted(int(v) for v in data_list[:, 1])
    return all_ids, sorted(train), sorted(val), sorted(test)


class MDDMumtazLoader(BaseLoader):
    def _get_id_lists(self, args, data_list, a, b):
        return get_id_list_mdd_mumtaz(args, data_list, a, b)

    def _postprocess_labels(self, args, flag):
        if flag != "PRETRAIN" and args.classify_choice != "mdd_vs_hc":
            raise ValueError("MDD-Mumtaz requires --classify_choice mdd_vs_hc")
