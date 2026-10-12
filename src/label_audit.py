"""Audit stored Batch 2 labels against original cycle numbers; no label changes."""
from pathlib import Path
import json
import argparse
import h5py
import numpy as np
import pandas as pd
from .preprocess import ROOT
OUT = ROOT / "DAY2/results"
THRESHOLD = 0.88

def audit_batch2_labels(archive_dir=None):
    """Check exact first threshold crossing using recorded cycle numbers."""
    source = Path(archive_dir or ROOT / 'archive') / '2018-02-20_batchdata_updated_struct_errorcorrect.mat'
    csv_path, summary_path = OUT / 'batch2_cycle_life_audit.csv', OUT / 'batch2_cycle_life_audit_summary.json'
    if not source.exists():
        if archive_dir is not None:
            raise FileNotFoundError(source)
        if csv_path.exists() and summary_path.exists():
            return pd.read_csv(csv_path), json.loads(summary_path.read_text())
        raise FileNotFoundError('Place the original MAT in archive/ or run with --archive-dir first')
    predictions = pd.read_csv(OUT / 'predictions_all.csv')
    evaluated = predictions[predictions.batch == 2].sort_values('cell_id')
    assert len(evaluated) == evaluated.cell_id.nunique() == 39
    rows = []
    with h5py.File(source, 'r') as mat:
        batch = mat['batch']
        for cell in evaluated.itertuples():
            cid = int(cell.cell_id)
            saved = float(mat[batch['cycle_life'][cid, 0]][()].ravel()[0])
            assert saved == cell.cycle_life
            summary = mat[batch['summary'][cid, 0]]
            cycles, q = summary['cycle'][()].ravel(), summary['QDischarge'][()].ravel()
            assert cycles.shape == q.shape
            finite_cycles = cycles[np.isfinite(cycles)]
            assert (np.diff(finite_cycles) > 0).all(), f'Nonmonotonic/duplicate cycle number in cell {cid}'
            valid = np.isfinite(cycles) & np.isfinite(q)
            hits = np.flatnonzero(valid & (q <= THRESHOLD))
            reached = bool(len(hits))
            position = int(hits[0]) if reached else None
            first = float(cycles[position]) if reached else np.nan
            matched = reached and saved == first  # Exact equality; no tolerance.
            previous = np.flatnonzero(valid & (np.arange(len(q)) < position)) if reached else []
            last = int(np.flatnonzero(valid)[-1])
            recovered = bool((q[position+1:][valid[position+1:]] > THRESHOLD).any()) if reached else False
            rows.append({'batch': 2, 'cell_id': cid, 'stored_cycle_life': saved,
                         'first_qd_le_088_cycle': first, 'crossing_minus_stored_cycles': first-saved,
                         'not_reached': not reached, 'exact_match': matched,
                         'status': 'exact_match' if matched else 'mismatch' if reached else 'not_reached',
                         'first_crossing_qd_ah': float(q[position]) if reached else np.nan,
                         'previous_valid_cycle': float(cycles[previous[-1]]) if len(previous) else np.nan,
                         'previous_valid_qd_ah': float(q[previous[-1]]) if len(previous) else np.nan,
                         'last_recorded_cycle': float(cycles[last]), 'last_qd_ah': float(q[last]),
                         'recovered_above_088_after_first_crossing': recovered,
                         'n_nonfinite_pairs': int((~valid).sum()), 'threshold_ah': THRESHOLD,
                         'cycle_number_source': 'summary.cycle; never array position'})
    frame = pd.DataFrame(rows)
    result = {'n_evaluated': len(frame), 'n_exact_match': int(frame.exact_match.sum()),
              'n_mismatch': int((frame.status == 'mismatch').sum()), 'n_not_reached': int(frame.not_reached.sum()),
              'threshold_ah': THRESHOLD, 'comparison': 'exact equality; no tolerance',
              'source_file': str(source.resolve()), 'cycle_number_source': 'summary.cycle',
              'labels_changed': False, 'cells_excluded_by_this_audit': 0,
              'n_recovered_after_crossing': int(frame.recovered_above_088_after_first_crossing.sum())}
    frame.to_csv(csv_path, index=False, encoding='utf-8-sig')
    summary_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    return frame, result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-dir")
    print(audit_batch2_labels(parser.parse_args().archive_dir)[1])
