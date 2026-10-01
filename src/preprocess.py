"""Read original MATLAB files and audit every cell before evaluation."""
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
from .features import extract_features

ROOT = Path(__file__).resolve().parents[1]
BATCH_FILES = {1: "2017-05-12", 2: "2018-02-20", 3: "2018-04-12"}
BATCH1_EXCLUSIONS = {0, 1, 2, 3, 4, 8, 10, 12, 13, 22}
HOLDOUT_POLICIES = {"4.8C(80%)-4.8C", "6C(30%)-3.6C", "6C(50%)-3C", "6C(60%)-3C"}


def read_early_cell(mat, batch, i):
    group = mat[batch["summary"][i, 0]]
    summary = {k: group[k][()].ravel() for k in ["cycle", "QDischarge", "chargetime", "IR", "Tavg"]}
    curves = mat[batch["cycles"][i, 0]]
    qs = []
    for k in [10, 100]:
        positions = np.flatnonzero(summary["cycle"] == k)
        if len(positions) != 1:
            raise ValueError(f"Missing/duplicate cycle {k}")
        ds = mat[curves["Qdlin"][int(positions[0]), 0]]
        if ds.attrs.get("MATLAB_empty", 0):
            raise ValueError(f"Empty Qdlin at cycle {k}")
        qs.append(ds[()].ravel())
    voltage = mat[batch["Vdlin"][i, 0]][()].ravel()
    return summary, qs[0], qs[1], voltage


def load_batch(batch_number):
    path = ROOT / "archive" / (BATCH_FILES[batch_number] + "_batchdata_updated_struct_errorcorrect.mat")
    rows, audits = [], []
    with h5py.File(path, "r") as mat:
        batch = mat["batch"]
        for i in range(batch["summary"].shape[0]):
            life = float(mat[batch["cycle_life"][i, 0]][()].ravel()[0])
            policy = "".join(chr(int(v)) for v in mat[batch["policy_readable"][i, 0]][()].ravel())
            reason = "included"
            if batch_number == 1 and i in BATCH1_EXCLUSIONS:
                reason = "continued_in_later_run" if i < 5 else "early_stopped_incomplete_target"
            elif not np.isfinite(life) or life <= 100:
                reason = "missing_or_invalid_cycle_life"
            feature_values = None
            if reason == "included":
                try:
                    feature_values = extract_features(*read_early_cell(mat, batch, i))
                except ValueError as error:
                    reason = "invalid_early_features: " + str(error)
            audits.append({"batch": batch_number, "cell_id": i, "policy": policy,
                           "cycle_life": life, "included": reason == "included", "reason": reason})
            if reason == "included":
                rows.append({"batch": batch_number, "cell_id": i, "policy": policy,
                             "cycle_life": life, **feature_values})
    return pd.DataFrame(rows), pd.DataFrame(audits)


def development_split(batch1):
    # Preserve the DAY 1 split inspected at task start without a deleted CSV dependency.
    merged = batch1.copy()
    merged["split"] = np.where(merged.policy.isin(HOLDOUT_POLICIES), "validation", "development")
    assert len(merged) == len(batch1) == 36
    dev = merged[merged["split"] == "development"].reset_index(drop=True)
    valid = merged[merged["split"] == "validation"].reset_index(drop=True)
    assert (len(dev), len(valid)) == (29, 7)
    assert not set(dev.policy) & set(valid.policy)
    return dev, valid
