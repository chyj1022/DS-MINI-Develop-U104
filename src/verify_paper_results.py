"""Check source early windows, fold isolation, actual model outputs and tables."""
import copy
import json

import h5py
import joblib
import numpy as np
import pandas as pd

from .preprocess import ROOT
from .paper_models import columns, extract_window, fit, evaluate
from .paper_comparison import digest, life_labels, PARTITIONS

OUT=ROOT/'DAY2/results'


def manual_classification(actual,predicted,n_classes):
    cm=np.zeros((n_classes,n_classes),dtype=int)
    np.add.at(cm,(np.asarray(actual,dtype=int),np.asarray(predicted,dtype=int)),1)
    denominator=cm.sum(axis=0)+cm.sum(axis=1)
    per_class=np.divide(2*np.diag(cm),denominator,out=np.zeros(n_classes,dtype=float),where=denominator>0)
    return {'accuracy_pct':float(100*np.trace(cm)/cm.sum()),'macro_f1':float(per_class.mean())}


def main():
    for name in ['paper_models_provenance','paper_comparison_provenance']:
        for path,checksum in json.loads((OUT/f'{name}.json').read_text())['immutable_inputs_sha256'].items():
            assert digest(OUT/path)==checksum
    np.testing.assert_array_equal(life_labels([499,500,550,1000,1001],'binary_550'),[0,0,0,1,1])
    np.testing.assert_array_equal(life_labels([499,500,550,1000,1001],'three_500_1000'),[0,1,1,1,2])
    features=pd.read_csv(OUT/'paper_model_features.csv')
    assert len(columns(100,'Discharge'))==13 and len(columns(100,'Full'))==20 and len(columns(5,'Full'))==18
    # Independent source extraction for both windows and a future-data mutation test.
    with h5py.File(ROOT/'archive/2017-05-12_batchdata_updated_struct_errorcorrect.mat') as mat:
        b=mat['batch'];summary=mat[b['summary'][5,0]];curves=mat[b['cycles'][5,0]]
        s={k:summary[k][()].ravel() for k in ['cycle','QDischarge','chargetime','IR','Tmax','Tmin']}
        axis=mat[b['Vdlin'][5,0]][()].ravel()
        expected=features[(features.batch==1)&(features.cell_id==5)].iloc[0]
        for window,start in [(5,4),(100,10)]:
            pos={k:int(np.flatnonzero(s['cycle']==k)[0]) for k in [start,window]}
            a,z=[mat[curves['Qdlin'][pos[k],0]][()].ravel() for k in [start,window]]
            integral=0.
            for i in np.flatnonzero((s['cycle']>=2)&(s['cycle']<=window)):
                t=mat[curves['t'][i,0]][()].ravel();temp=mat[curves['T'][i,0]][()].ravel()
                keep=np.isfinite(t)&np.isfinite(temp)
                integral+=np.trapezoid(temp[keep],t[keep])
            source=extract_window(s,a,z,axis,integral,window)
            np.testing.assert_allclose(source[f'c{window}_dq_variance_log'],np.log10(np.var(z-a,ddof=1)))
            for key,value in source.items():
                np.testing.assert_allclose(value,expected[key],equal_nan=True,atol=1e-12)
            changed=copy.deepcopy(s)
            for key in changed:
                if key!='cycle':changed[key][changed['cycle']>window]=999999
            after=extract_window(changed,a,z,axis,integral,window)
            for key,value in source.items():np.testing.assert_allclose(value,after[key],equal_nan=True)
        # Direct parity of all 35 high-life early-5 curves, including the group IQR.
        population=features[(features.batch==1)&(features.cycle_life>550)]
        deltas=[]
        for cid in population.cell_id:
            ss=mat[b['summary'][int(cid),0]];cc=mat[b['cycles'][int(cid),0]];cy=ss['cycle'][()].ravel()
            pos4,pos5=[int(np.flatnonzero(cy==k)[0]) for k in [4,5]]
            deltas.append(mat[cc['Qdlin'][pos5,0]][()].ravel()-mat[cc['Qdlin'][pos4,0]][()].ravel())
        stored=pd.read_csv(OUT/'paper_early5_curves_batch1.csv');stored=stored[stored.observed_class==1]
        np.testing.assert_allclose(stored[['dq_q25','dq_median','dq_q75']].to_numpy().T,np.percentile(deltas,[25,50,75],axis=0),atol=1e-14)
    manifest=pd.read_csv(OUT/'paper_model_fold_manifest.csv')
    for _,g in manifest.groupby(['task','family','fold']):
        tr,va=g[g.role=='train'],g[g.role=='valid']
        assert not set(tr.cell_id)&set(va.cell_id) and not set(tr.policy)&set(va.policy)
    scores=pd.read_csv(OUT/'paper_model_metrics.csv');p=pd.read_csv(OUT/'paper_model_predictions.csv')
    configs=json.loads((OUT/'paper_models_config.json').read_text())
    models=joblib.load(OUT/'models/paper_models.joblib')
    b1=features[features.batch==1]
    development_ids=set(pd.read_csv(OUT/'split_manifest.csv').query("split=='development'").cell_id)
    dev=b1[b1.cell_id.isin(development_ids)]
    for cfg in configs:
        task,family=cfg['task'],cfg['family'];model,c=models[(task,family)]
        assert c==cfg['features'] and len(c)==(18 if task=='binary_550' and family=='Full' else len(c))
        np.testing.assert_allclose(model.named_steps['imputer'].statistics_,b1[c].median(),equal_nan=True)
        expected_mean=model.named_steps['imputer'].transform(b1[c]).mean(axis=0)
        np.testing.assert_allclose(model.named_steps['scaler'].mean_,expected_mean)
        for number,g in features[features.batch>1].groupby('batch'):
            pred,_=evaluate(model,c,g,task)
            stored=p[(p.task==task)&(p.family==family)&(p.batch==number)].set_index('cell_id').loc[g.cell_id]
            np.testing.assert_allclose(pred,stored.prediction)
        holdout_model,hc,_=fit(dev,task,family,cfg['selected_config'])
        holdout=features[(features.batch==1)&(~features.cell_id.isin(development_ids))]
        pred,_=evaluate(holdout_model,hc,holdout,task)
        expected=p[(p.task==task)&(p.family==family)&(p.partition==PARTITIONS[1])].set_index('cell_id').loc[holdout.cell_id]
        np.testing.assert_allclose(pred,expected.prediction)
    for row in scores.itertuples():
        subset=p[(p.task==row.task)&(p.family==row.family)&(p.partition==row.partition)]
        assert len(subset)==row.n_cells
        groups=[g for _,g in subset.groupby('fold')] if row.partition==PARTITIONS[0] else [subset]
        computed=[]
        for g in groups:
            if row.task=='regression':
                residual=g.prediction-g.cycle_life
                computed.append({'mape_pct':float((100*residual.abs()/g.cycle_life).mean()),'mae_cycles':float(residual.abs().mean()),'rmse_cycles':float(np.sqrt(np.mean(residual**2)))})
            else:
                computed.append(manual_classification(g.observed_class,g.prediction,2 if row.task=='binary_550' else 3))
        for key in computed[0]:np.testing.assert_allclose(getattr(row,key),np.mean([v[key] for v in computed]))
    selected=scores[(scores.task=='regression')&(scores.family=='Selected')].set_index('partition')
    tr,va,te=[selected.loc[n].mape_pct for n in PARTITIONS[:3]]
    perf=pd.read_csv(OUT/'paper_selected_regression_performance.csv')
    np.testing.assert_allclose(perf['MAPE (%)'],[tr,va,te,va-tr,te-va,te-9.1])
    ref=pd.read_csv(OUT/'paper_classification_metrics.csv')
    target=float(ref[(ref.model=='Full classifier')&(ref.partition=='Primary + Secondary test')].macro_f1_reconstructed.iloc[0])
    assert round(float(ref[(ref.model=='Full classifier')&(ref.partition=='Primary + Secondary test')].accuracy_pct.iloc[0]),1)==95.1
    for task in ['binary_550','three_500_1000']:
        table=pd.read_csv(OUT/f'paper_{task}_performance.csv')
        values=scores[(scores.task==task)&(scores.family=='Full')].set_index('partition')
        for key,col in [('macro_f1','F1-Score (macro)'),('accuracy_pct','Accuracy (%)')]:
            a,b,c=[values.loc[n,key] for n in PARTITIONS[:3]]
            np.testing.assert_allclose(table[col].iloc[:5],[a,b,c,a-b,b-c])
            expected=(target-c if key=='macro_f1' else 95.1-c) if task=='binary_550' else np.nan
            np.testing.assert_allclose(table[col].iloc[5],expected,equal_nan=True)
            if PARTITIONS[3] in values.index:
                extended=pd.read_csv(OUT/f'paper_{task}_with_batch3.csv')
                d=values.loc[PARTITIONS[3],key]
                tg=(target-d if key=='macro_f1' else 95.1-d) if task=='binary_550' else np.nan
                np.testing.assert_allclose(extended[col].iloc[-3:],[d,c-d,tg],equal_nan=True)
                formatted=pd.read_csv(OUT/f'paper_{task}_with_batch3_formatted.csv').fillna('')
                assert formatted.shape==(9,5)
                assert list(formatted.columns)==['구분','비교 항목','F1-Score (macro)','Accuracy (%)','비고']
                assert list(formatted['비교 항목'].iloc[[5,8]])==['Gap (Target-Test)','Gap (Target-Test)']
    if PARTITIONS[3] in selected.index:
        m3=selected.loc[PARTITIONS[3]].mape_pct
        extended=pd.read_csv(OUT/'paper_selected_regression_with_batch3.csv')
        np.testing.assert_allclose(extended['MAPE (%)'].iloc[-3:],[m3,m3-te,m3-9.1])
        sensitivity=pd.read_csv(OUT/'paper_selected_batch3_quality.csv')
        g=p[(p.task=='regression')&(p.family=='Selected')&(p.batch==3)&(~p.cell_id.isin([2,37,42,43]))]
        assert len(g)==sensitivity.iloc[1].n_cells==40
        np.testing.assert_allclose(sensitivity.iloc[1].mape_pct,(100*(g.prediction-g.cycle_life).abs()/g.cycle_life).mean())
    print('PASS: paper early-5/100 source parity; future-window invariance; raw early-5 curves; 13/20/18 features; disjoint policies; train-only preprocessing; actual saved/refitted predictions; nested means; regression/classification Gaps; reconstructed paper F1; original model unchanged')


if __name__=='__main__':main()
