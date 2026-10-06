"""Integration checks: source parity, future invariance, groups and score math."""
import json
import copy
import contextlib
import io
import warnings
import matplotlib.pyplot as plt
import h5py
import joblib
import nbformat
import numpy as np
import pandas as pd
from .features import extract_features, ALL_FEATURES
from .preprocess import ROOT, read_early_cell
from .train import OUT, metrics, predict


def main():
    plt.switch_backend("Agg")
    frame = pd.read_csv(OUT / "cell_features.csv")
    eligible = frame[frame.batch==1].set_index("cell_id")
    with h5py.File(ROOT / "archive/2017-05-12_batchdata_updated_struct_errorcorrect.mat") as f:
        args=read_early_cell(f,f["batch"],5)
        calculated=extract_features(*args)
        for feature in ALL_FEATURES:
            np.testing.assert_allclose(eligible.loc[5,feature],calculated[feature],atol=1e-12)
        changed=copy.deepcopy(args[0])
        for key in changed:
            if key != "cycle": changed[key][changed["cycle"] > 100] = 99999
        a,b=extract_features(*args),extract_features(changed,*args[1:])
        for key in a: np.testing.assert_allclose(a[key],b[key],equal_nan=True)
    split=pd.read_csv(OUT / "split_manifest.csv")
    assert not set(split[split.split=="development"].policy) & set(split[split.split=="validation"].policy)
    fold=pd.read_csv(OUT / "cv_fold_manifest.csv")
    for _,g in fold.groupby("fold"):
        assert not set(g[g.role=="train"].policy) & set(g[g.role=="valid"].policy)
    pred=pd.read_csv(OUT / "predictions.csv")
    assert len(pred)==75 and len(pred.drop_duplicates(["batch","cell_id"]))==75
    scores=pd.read_csv(OUT / "metrics.csv")
    for _,row in scores.iterrows():
        p=pred[pred.partition==row["index"]]
        if row["index"]=="Train (Batch 1 CV)":
            calculated={key:np.mean([metrics(g.cycle_life,g.predicted_cycle_life)[key] for _,g in p.groupby("fold")]) for key in ["mape_pct","mae_cycles","rmse_cycles"]}
        else: calculated=metrics(p.cycle_life,p.predicted_cycle_life)
        for key in calculated: np.testing.assert_allclose(row[key],calculated[key],rtol=1e-10)
    perf=pd.read_csv(OUT / "model_performance.csv")
    tr,va,te=scores.mape_pct
    np.testing.assert_allclose(perf["MAPE (%)"],[tr,va,te,va-tr,te-va,te-9.1])
    saved=joblib.load(OUT / "models/final_model.joblib")
    assert set(saved["features"]) <= set(ALL_FEATURES)
    b2=frame[frame.batch==2]
    p=pred[pred.partition=="Test (Batch 2)"].set_index("cell_id").loc[b2.cell_id]
    np.testing.assert_allclose(p.predicted_cycle_life,predict(saved["model"],b2,saved["config"]))
    # Recompute the constant baseline and bias directly from labels/predictions.
    baseline=pd.read_csv(OUT / "batch2_baseline_comparison.csv")
    median=float(frame[frame.batch==1].cycle_life.median())
    assert baseline.iloc[0].constant_prediction_cycles==median
    for row,values in [(baseline.iloc[0],np.full(len(p),median)),(baseline.iloc[1],p.predicted_cycle_life)]:
        assert row.n_cells==len(p)
        for key,value in metrics(p.cycle_life,values).items():
            np.testing.assert_allclose(row[key],value)
    np.testing.assert_allclose(baseline.iloc[1].mape_reduction_vs_baseline_pct,
                              100*(1-baseline.iloc[1].mape_pct/baseline.iloc[0].mape_pct))
    bias=pd.read_csv(OUT / "prediction_bias.csv")
    below=p[p.cycle_life<frame[frame.batch==1].cycle_life.min()]
    for row,group in [(bias.iloc[0],p),(bias.iloc[1],below)]:
        residual=group.predicted_cycle_life-group.cycle_life
        assert row.n_cells==len(group) and row.overpredicted_cells==int((residual>0).sum())
        assert row.underpredicted_cells==int((residual<0).sum())
        np.testing.assert_allclose(row.mean_signed_error_cycles,residual.mean())
        np.testing.assert_allclose(row.mae_cycles,residual.abs().mean())
        np.testing.assert_allclose(row.median_ape_pct,(100*residual.abs()/group.cycle_life).median())
    standard=set(p.loc[~p.policy.str.contains("newstructure")].index)
    overlap=pd.read_csv(OUT / "subgroup_overlap.csv").iloc[0]
    assert overlap.n_group_a==len(standard) and overlap.n_group_b==len(below)
    assert overlap.n_intersection==len(standard & set(below.index))
    assert bool(overlap.identical_cell_sets)==(standard==set(below.index))
    train_values=frame[frame.batch==1][saved["features"]]
    fitted_imputer=saved["model"].named_steps["imputer"]
    np.testing.assert_allclose(fitted_imputer.statistics_,train_values.median().to_numpy())
    np.testing.assert_allclose(saved["model"].named_steps["scaler"].mean_,fitted_imputer.transform(train_values).mean(axis=0))
    holdout=joblib.load(OUT / "models/holdout_model.joblib")
    dev_ids=set(split[split.split=="development"].cell_id)
    dev_values=frame[(frame.batch==1)&frame.cell_id.isin(dev_ids)][holdout["features"]]
    np.testing.assert_allclose(holdout["model"].named_steps["imputer"].statistics_,dev_values.median().to_numpy())
    frozen=json.loads((OUT / "frozen_config.json").read_text())
    assert frozen["n_final_training"]==36
    from .diagnostics import sha256
    summary=json.loads((OUT / "diagnostics_summary.json").read_text())
    provenance=json.loads((OUT / "provenance.json").read_text())
    assert sha256(OUT / "models/final_model.joblib")==summary["model_sha256"]==provenance["model_sha256"]
    assert provenance["config"]==saved["config"]
    for source in provenance["sources"]:
        assert (ROOT / "archive" / source["filename"]).stat().st_size==source["bytes"]
    all_pred=pd.read_csv(OUT / "predictions_all.csv")
    all_frame=pd.read_csv(OUT / "cell_features_all.csv")
    assert len(all_pred.drop_duplicates(["batch","cell_id"]))==len(all_pred)==len(all_frame)
    all_scores=pd.read_csv(OUT / "metrics_all.csv")
    for number in [2,3] if summary["include_batch3"] else [2]:
        partition=all_pred[all_pred.batch==number]
        batch_frame=all_frame[all_frame.batch==number]
        np.testing.assert_allclose(partition.predicted_cycle_life,predict(saved["model"],batch_frame,saved["config"]))
        calculated=metrics(partition.cycle_life,partition.predicted_cycle_life)
        row=all_scores[all_scores["index"]==f"Test (Batch {number})"].iloc[0]
        for key,value in calculated.items(): np.testing.assert_allclose(row[key],value)
    if summary["include_batch3"]:
        assert len(all_frame[all_frame.batch==3])==44
        extended=pd.read_csv(OUT / "model_performance_with_batch3.csv")
        m3=all_scores[all_scores["index"]=="Test (Batch 3)"].mape_pct.iloc[0]
        np.testing.assert_allclose(extended["MAPE (%)"].iloc[-3:],[m3,m3-te,m3-9.1])
        sensitivity=pd.read_csv(OUT / "batch3_quality_sensitivity.csv")
        p3=all_pred[all_pred.batch==3]
        author_filtered=p3[~p3.cell_id.isin([2,37,42,43])]
        assert len(author_filtered)==40
        np.testing.assert_allclose(sensitivity.iloc[1].mape_pct,metrics(author_filtered.cycle_life,author_filtered.predicted_cycle_life)["mape_pct"])
    eda=pd.read_csv(OUT / "eda_cell_features.csv")
    assert eda.delta_variance_origin_invariance_error.max()<1e-12
    # Audit the requested EDA groups against saved cell records, retaining n=0.
    counts=pd.read_csv(OUT / 'life_group_proportions.csv')
    group_stats=pd.read_csv(OUT / 'life_group_statistics.csv')
    degradation=pd.read_csv(OUT / 'degradation_evidence.csv')
    knee=pd.read_csv(OUT / 'knee_timing_summary.csv')
    representatives=pd.read_csv(OUT / 'life_group_representatives.csv')
    masks={'short_lt500':lambda y:y<500,'middle_500_1000':lambda y:(y>=500)&(y<=1000),'long_gt1000':lambda y:y>1000}
    for number,g in eda.groupby('batch'):
        assert counts[counts.batch==number].n_cells.sum()==len(g)
        for name,mask in masks.items():
            chosen=g[mask(g.cycle_life)]
            count=counts[(counts.batch==number)&(counts.life_group==name)].iloc[0]
            assert count.n_cells==len(chosen) and count.batch_n_cells==len(g)
            np.testing.assert_allclose(count.proportion_pct,100*len(chosen)/len(g))
            row=group_stats[(group_stats.batch==number)&(group_stats.life_group==name)].iloc[0]
            for feature in ['cycle_life','log10_dq_variance','dq_mean','dq_min']:
                np.testing.assert_allclose(row[feature+'_median'],chosen[feature].median(),equal_nan=True)
    for name,mask in masks.items():
        ds=degradation[mask(degradation.cycle_life)]
        accepted=ds[ds.shape_ok & ~ds.search_boundary]
        row=knee[(knee.scope=='Pooled (descriptive)')&(knee.life_group==name)].iloc[0]
        assert row.n_cells==len(ds) and row.n_search_boundary==int(ds.search_boundary.sum())
        assert row.n_shape_ok==int(ds.shape_ok.sum()) and row.n_interior_shape_ok==len(accepted)
        np.testing.assert_allclose(row.interior_candidate_median_cycle,accepted.knee_candidate_cycle.median())
        ordered=eda[mask(eda.cycle_life)].sort_values(['cycle_life','batch','cell_id'])
        expected=ordered.iloc[(len(ordered)-1)//2]
        rep=representatives[representatives.life_group==name].iloc[0]
        assert (rep.batch,rep.cell_id)==(expected.batch,expected.cell_id)
    # Direct source parity for the Batch 2 short-life group's 1000-point ΔQ curve.
    chosen=eda[(eda.batch==2)&(eda.cycle_life<500)]
    curves=[]
    with h5py.File(ROOT / 'archive/2018-02-20_batchdata_updated_struct_errorcorrect.mat') as f:
        for cid in chosen.cell_id:
            _,q10,q100,axis=read_early_cell(f,f['batch'],int(cid))
            curves.append(q100-q10)
    expected=np.percentile(np.array(curves),[25,50,75],axis=0)
    stored_curves=pd.read_csv(OUT / 'dq_life_group_curves.csv')
    stored_curves=stored_curves[(stored_curves.batch==2)&(stored_curves.life_group=='short_lt500')]
    np.testing.assert_allclose(stored_curves.voltage,axis)
    np.testing.assert_allclose(stored_curves[['dq_q25','dq_median','dq_q75']].to_numpy().T,expected,atol=1e-14)
    policies=pd.read_csv(OUT / 'policy_summary.csv')
    for r in policies.itertuples():
        chosen=eda[(eda.batch==r.batch)&(eda.policy==r.policy)]
        assert r.n_cells==len(chosen)
        np.testing.assert_allclose(r.mean_life,chosen.cycle_life.mean())
    uncertainty=pd.read_csv(OUT / "metric_uncertainty.csv")
    assert (uncertainty.ci_low_pct<=uncertainty.mape_pct).all()
    assert (uncertainty.ci_high_pct>=uncertainty.mape_pct).all()
    domains=pd.read_csv(OUT / "input_domain_audit.csv")
    bounds=frame[frame.batch==1][saved["features"]].agg(["min","max"])
    for key in saved["features"]:
        expected=(domains[key]<bounds.loc["min",key])|(domains[key]>bounds.loc["max",key])
        assert np.array_equal(expected,domains[key+"_outside_train_range"])
    from .verify_paper_results import main as verify_paper_results
    verify_paper_results()
    nb=nbformat.read(ROOT / "DAY2/03_modeling.ipynb",as_version=4)
    nbformat.validate(nb)
    environment={}
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.filterwarnings("ignore",message="FigureCanvasAgg is non-interactive.*",category=UserWarning)
        for cell in nb.cells:
            if cell.cell_type=="code":
                exec(cell.source,environment)
                plt.close("all")
    print("PASS: source features; future-cycle invariance; disjoint protocols; all predictions and Gaps; Batch 2 baseline; signed bias and subgroup overlap; training-only preprocessing; frozen-model hash; Batch 3 sensitivity; origin invariance; input domain; notebook execution")


if __name__=="__main__": main()
