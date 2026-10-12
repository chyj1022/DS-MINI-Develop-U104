"""Check the allowed feature scope, group scores, Target gaps and report links."""
import hashlib
import json
import re
import nbformat
import numpy as np
import pandas as pd
from .preprocess import ROOT
from .features import ALL_FEATURES
from .train import candidates
from .paper_comparison import life_labels, classification_metrics, PARTITIONS

OUT = ROOT / 'DAY2/results'


def main():
    frozen = json.loads((OUT/'frozen_config.json').read_text())
    assert set(frozen['features']) <= set(ALL_FEATURES)
    assert len(ALL_FEATURES) == 5 and len(candidates()) == 212
    p = pd.read_csv(OUT/'predictions_all.csv')
    groups = pd.read_csv(OUT/'life_class_predictions.csv')
    scores = pd.read_csv(OUT/'life_class_metrics.csv')
    for scheme in ['binary_550','three_500_1000']:
        for part, subset in p.groupby('partition',sort=False):
            actual = groups[(groups.scheme==scheme)&(groups.partition==part)].set_index(['batch','cell_id'])
            aligned = actual.loc[list(zip(subset.batch,subset.cell_id))]
            np.testing.assert_allclose(aligned.predicted_cycle_life,subset.predicted_cycle_life)
            np.testing.assert_array_equal(aligned.observed_class,life_labels(subset.cycle_life,scheme))
            np.testing.assert_array_equal(aligned.predicted_class,life_labels(subset.predicted_cycle_life,scheme))
            pieces = [g for _,g in subset.groupby('fold')] if part==PARTITIONS[0] else [subset]
            metrics = [classification_metrics(life_labels(g.cycle_life,scheme),life_labels(g.predicted_cycle_life,scheme),scheme) for g in pieces]
            row = scores[(scores.scheme==scheme)&(scores.partition==part)].iloc[0]
            for key in ['macro_f1','accuracy_pct']:
                np.testing.assert_allclose(row[key],np.mean([m[key] for m in metrics]))
        perf = pd.read_csv(OUT/(scheme+'_performance.csv'))
        assert len(perf)==6 and np.isnan(perf.iloc[5]['F1-Score (macro)'])
        lookup = scores[scores.scheme==scheme].set_index('partition')
        for column,key in [('F1-Score (macro)','macro_f1'),('Accuracy (%)','accuracy_pct')]:
            tr,va,te=[lookup.loc[x,key] for x in PARTITIONS[:3]]
            np.testing.assert_allclose(perf[column].iloc[:5],[tr,va,te,tr-va,va-te])
        if scheme=='binary_550':
            np.testing.assert_allclose(perf.iloc[5]['Accuracy (%)'],95.1-lookup.loc[PARTITIONS[2],'accuracy_pct'])
        else:
            assert np.isnan(perf.iloc[5]['Accuracy (%)'])
        if 3 in p.batch.unique():
            extended = pd.read_csv(OUT/(scheme+'_with_batch3.csv'))
            assert len(extended)==9 and extended.iloc[-1]['구분']=='Gap (Target-Test, Batch 3)'
            for column,key in [('F1-Score (macro)','macro_f1'),('Accuracy (%)','accuracy_pct')]:
                m2,m3=[lookup.loc[x,key] for x in PARTITIONS[2:]]
                np.testing.assert_allclose(extended[column].iloc[6:8],[m3,m2-m3])
            assert np.isnan(extended.iloc[8]['F1-Score (macro)'])
            if scheme=='binary_550':
                np.testing.assert_allclose(extended.iloc[8]['Accuracy (%)'],95.1-lookup.loc[PARTITIONS[3],'accuracy_pct'])
    audit = pd.read_csv(OUT/'batch2_cycle_life_audit.csv')
    assert set(audit.cell_id)==set(p[p.batch==2].cell_id)
    np.testing.assert_array_equal(audit.exact_match,audit.stored_cycle_life==audit.first_qd_le_088_cycle)
    provenance=json.loads((OUT/'paper_comparison_provenance.json').read_text())
    for name,expected in provenance['immutable_inputs_sha256'].items():
        assert hashlib.sha256((OUT/name).read_bytes()).hexdigest()==expected
    text=(ROOT/'README.md').read_text()
    for link in re.findall(r'\]\(([^)]+)\)',text):
        if not link.startswith(('https://','http://','#')):
            assert (ROOT/link).exists(), f'Missing README link: {link}'
    for removed in ['paper_models.py','paper_selected_regression_performance','최종 Discharge','L1 Logistic Regression']:
        assert removed not in text
    nb=nbformat.read(ROOT/'DAY2/03_modeling.ipynb',as_version=4)
    assert not any('from src.paper_models' in c.source for c in nb.cells)
    assert not (ROOT/'src/paper_models.py').exists()
    print('PASS: five planned features; 212 candidates; unchanged regression predictions; 2/3-group metrics and gaps; label audit; report links; no paper-feature training path')


if __name__ == '__main__':
    main()
