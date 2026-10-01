"""Reproducible EDA and uncertainty for an already frozen model.

This module never chooses a model or changes its coefficients. All Batch 2/3
diagnostics are post-evaluation analyses; they are not selection criteria.
"""
import argparse
import hashlib
import importlib.metadata
import json
import platform
import re
from pathlib import Path

import h5py
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .features import ALL_FEATURES
from .preprocess import ROOT, BATCH_FILES, load_batch, read_early_cell
from .train import OUT, metrics, predict, prediction_rows

EDA_FEATURES = [*ALL_FEATURES, "dq_mean", "dq_min", "qd_change_100_2", "tmax_max_2_100", "ideal_equivalent_c"]
LABELS = {
    "log10_dq_variance":"Delta-Q log variance", "qd_slope_2_100":"Capacity slope",
    "charge_time_mean_2_6":"Charge time", "ir_change_100_2":"IR change",
    "tavg_mean_2_100":"Mean temperature", "dq_mean":"Delta-Q mean", "dq_min":"Delta-Q minimum",
    "qd_change_100_2":"Capacity change", "tmax_max_2_100":"Max temperature", "ideal_equivalent_c":"Equivalent C-rate",
}
COLORS = {1:"#2563eb",2:"#e87c26",3:"#269576"}


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda:handle.read(8*1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()


def save_csv(name, frame):
    frame.to_csv(OUT / (name+".csv"),index=False,encoding="utf-8-sig")


def external_evaluation_evidence(features, predictions):
    """Compare a training-only constant baseline and audit signed test errors."""
    train = features[features.batch == 1]
    test = predictions[predictions.batch == 2].copy()
    assert not test.cell_id.duplicated().any()
    median = float(train.cycle_life.median())
    baseline = metrics(test.cycle_life, np.full(len(test), median))
    model = metrics(test.cycle_life, test.predicted_cycle_life)
    comparison = pd.DataFrame([
        {"model": "Batch 1 median baseline", "n_cells": len(test),
         "constant_prediction_cycles": median, **baseline, "mape_reduction_vs_baseline_pct": 0.0},
        {"model": "Frozen selected model", "n_cells": len(test),
         "constant_prediction_cycles": np.nan, **model,
         "mape_reduction_vs_baseline_pct": 100 * (1 - model["mape_pct"] / baseline["mape_pct"])},
    ])
    below = test[test.cycle_life < train.cycle_life.min()]
    bias = []
    for name, group in [("Batch 2 all", test), ("below_batch1_life_min", below)]:
        residual = group.predicted_cycle_life - group.cycle_life
        bias.append({"group": name, "n_cells": len(group),
                     "overpredicted_cells": int((residual > 0).sum()),
                     "underpredicted_cells": int((residual < 0).sum()),
                     "mean_signed_error_cycles": float(residual.mean()),
                     "mae_cycles": float(residual.abs().mean()),
                     "median_ape_pct": float((100 * residual.abs() / group.cycle_life).median())})
    standard_ids = set(test.loc[~test.policy.str.contains("newstructure"), "cell_id"])
    below_ids = set(below.cell_id)
    overlap = pd.DataFrame([{"group_a": "standard_structure", "group_b": "below_batch1_life_min",
                             "n_group_a": len(standard_ids), "n_group_b": len(below_ids),
                             "n_intersection": len(standard_ids & below_ids),
                             "identical_cell_sets": standard_ids == below_ids}])
    bias = pd.DataFrame(bias)
    for name, frame in [("batch2_baseline_comparison", comparison), ("prediction_bias", bias),
                        ("subgroup_overlap", overlap)]:
        save_csv(name, frame)
    return comparison, bias, overlap


def grouped_interval(predictions, samples=2000, seed=42):
    """Percentile cluster bootstrap: resample policies and retain all their cells."""
    rng=np.random.default_rng(seed)
    grouped=[g.ape_pct.to_numpy() for _,g in predictions.groupby("policy")]
    estimates=np.empty(samples)
    for i in range(samples):
        ids=rng.integers(0,len(grouped),size=len(grouped))
        estimates[i]=np.concatenate([grouped[j] for j in ids]).mean()
    low,high=np.quantile(estimates,[.025,.975])
    return {"mape_pct":float(predictions.ape_pct.mean()), "ci_low_pct":float(low),"ci_high_pct":float(high),
            "n_cells":len(predictions),"n_policies":len(grouped),"bootstrap_samples":samples,"seed":seed}


def vif(frame, keys):
    x=frame[keys].dropna().to_numpy(float)
    sd=x.std(axis=0)
    if np.any(sd==0):
        return [{"feature":k,"vif":np.nan,"n_cells":len(x),"note":"constant column in set"} for k in keys]
    z=(x-x.mean(axis=0))/sd
    result=[]
    for j,key in enumerate(keys):
        design=np.column_stack([np.ones(len(z)),np.delete(z,j,axis=1)])
        coef=np.linalg.lstsq(design,z[:,j],rcond=None)[0]
        residual=z[:,j]-np.sum(design*coef,axis=1)
        sse=np.sum(residual**2)
        result.append({"feature":key,"vif":float(np.sum(z[:,j]**2)/sse) if sse>1e-12 else np.inf,"n_cells":len(x),"note":"complete cases in this feature set"})
    return result


def degradation_summary(cycle, capacity, life):
    """Future observations are descriptive evidence, never model inputs."""
    mask=(cycle>=2)&np.isfinite(cycle)&np.isfinite(capacity)&(capacity>0)&(capacity<=1.2)
    x,y=cycle[mask],capacity[mask]
    progress=(x-2)/(cycle.max()-2)
    slopes=[]
    for lo,hi in [(0,1/3),(1/3,2/3),(2/3,1.000001)]:
        chosen=(progress>=lo)&(progress<hi)
        slopes.append(float(np.polyfit(x[chosen],y[chosen],1)[0]*100) if chosen.sum()>=20 else np.nan)
    use=x>=100;xx,yy=x[use],y[use]
    scale=xx.max()-xx.min();z=(xx-xx.min())/scale
    base=np.column_stack([np.ones(len(z)),z]);beta=np.linalg.lstsq(base,yy,rcond=None)[0]
    base_sse=np.sum((yy-np.sum(base*beta,axis=1))**2)
    grid=np.linspace(max(100,2+.2*(cycle.max()-2)),2+.8*(cycle.max()-2),80)
    best=None
    for k in grid:
        if (xx<=k).sum()<50 or (xx>k).sum()<50:continue
        design=np.column_stack([np.ones(len(z)),z,np.maximum(0,(xx-k)/scale)])
        coef=np.linalg.lstsq(design,yy,rcond=None)[0]
        sse=np.sum((yy-np.sum(design*coef,axis=1))**2)
        if best is None or sse<best["sse"]:
            before,after=coef[1]/scale,(coef[1]+coef[2])/scale
            best={"knee_candidate_cycle":float(k),"knee_fraction_of_life":float(k/life),"sse":float(sse),
                  "sse_improvement":float(1-sse/base_sse),"shape_ok":bool(after<0 and after<before and abs(after)>=2*max(abs(before),1e-9)),
                  "search_boundary":bool(np.isclose(k,grid[0]) or np.isclose(k,grid[-1]))}
    return {"early_slope_ah_per100":slopes[0],"middle_slope_ah_per100":slopes[1],"late_slope_ah_per100":slopes[2],
            "late_faster_than_middle":bool(slopes[2]<slopes[1]), **(best or {})}


def raw_evidence(features):
    rows,degradation,curves={},[],{}
    for number,subset in features.groupby("batch"):
        path=ROOT/"archive"/(BATCH_FILES[int(number)]+"_batchdata_updated_struct_errorcorrect.mat")
        with h5py.File(path,"r") as mat:
            batch=mat["batch"]
            for r in subset.itertuples():
                summary,q10,q100,axis=read_early_cell(mat,batch,int(r.cell_id))
                early=(summary["cycle"]>=2)&(summary["cycle"]<=100)
                q=summary["QDischarge"];c=summary["cycle"]
                group=mat[batch["summary"][int(r.cell_id),0]]
                tmax=group["Tmax"][()].ravel()[early]
                parsed=re.match(r"([\d.]+)C\(([\d.]+)%\)-([\d.]+)C",r.policy)
                equivalent=np.nan
                if parsed:
                    c1,soc,c2=map(float,parsed.groups());equivalent=.8/((soc/100)/c1+(.8-soc/100)/c2)
                row=r._asdict()
                shifted=(q100-q100[0])-(q10-q10[0])
                row.update({"dq_mean":float((q100-q10).mean()),"dq_min":float((q100-q10).min()),
                            "qd_change_100_2":float(q[c==100][0]-q[c==2][0]),
                            "tmax_max_2_100":float(np.nanmax(tmax)),"ideal_equivalent_c":equivalent,
                            "q10_high_voltage_origin":float(q10[0]),"q100_high_voltage_origin":float(q100[0]),
                            "delta_variance_origin_invariance_error":float(abs(np.var(shifted)-np.var(q100-q10)))})
                rows[(int(number),int(r.cell_id))]=row
                description={"batch":int(number),"cell_id":int(r.cell_id),"cycle_life":r.cycle_life,
                             **degradation_summary(c,q,r.cycle_life)}
                degradation.append(description)
                curves[(int(number),int(r.cell_id))]={"cycle":c,"capacity":q,"axis":axis,"delta":q100-q10}
    return pd.DataFrame(rows.values()),pd.DataFrame(degradation),curves


def plot_evidence(eda,degradation,curves,intervals,domains):
    fig,axes=plt.subplots(1,2,figsize=(12,4.8))
    for number,g in eda.groupby("batch"):
        dq=np.array([curves[(int(number),int(cid))]["delta"] for cid in g.cell_id])
        axis=next(v["axis"] for k,v in curves.items() if k[0]==number)
        low,median,high=np.percentile(dq,[25,50,75],axis=0)
        axes[0].plot(axis,median,color=COLORS[number],label=f"Batch {number} (n={len(g)})")
        axes[0].fill_between(axis,low,high,color=COLORS[number],alpha=.15)
        d=degradation[degradation.batch==number]
        vals=[d[k].median() for k in ["early_slope_ah_per100","middle_slope_ah_per100","late_slope_ah_per100"]]
        axes[1].plot([0,1,2],vals,marker="o",color=COLORS[number],label=f"Batch {number}")
    axes[0].set(title="Delta-Q shape · median and IQR",xlabel="Common voltage (V)",ylabel="Q100(V) - Q10(V) (Ah)");axes[0].legend(fontsize=8)
    axes[1].set(title="Descriptive degradation speed",ylabel="Median capacity slope (Ah / 100 cycles)",xticks=[0,1,2],xticklabels=["Early third","Middle third","Late third"]);axes[1].legend(fontsize=8)
    axes[1].text(.02,.04,"Later cycles are excluded from model features",transform=axes[1].transAxes,fontsize=8)
    fig.tight_layout();fig.savefig(OUT/"04_eda_curve_evidence.png",dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,6))
    matrix=eda[EDA_FEATURES].corr()
    im=axes[0].imshow(matrix,vmin=-1,vmax=1,cmap="RdBu_r")
    names=[LABELS[k] for k in EDA_FEATURES]
    axes[0].set(xticks=range(10),yticks=range(10),xticklabels=names,yticklabels=names,title="Pooled exploratory Pearson correlation")
    plt.setp(axes[0].get_xticklabels(),rotation=65,ha="right",fontsize=8);plt.setp(axes[0].get_yticklabels(),fontsize=8)
    fig.colorbar(im,ax=axes[0],fraction=.04)
    for number,g in eda.groupby("batch"):
        values=[g[k].corr(g.cycle_life,method="spearman") for k in ALL_FEATURES]
        axes[1].plot(range(5),values,marker="o",color=COLORS[number],label=f"Batch {number}")
    axes[1].axhline(0,color="#64748b",ls="--")
    axes[1].set(xticks=range(5),xticklabels=[LABELS[k] for k in ALL_FEATURES],ylabel="Spearman with cycle life",ylim=(-1,1),title="Transfer stability of five representative features")
    plt.setp(axes[1].get_xticklabels(),rotation=40,ha="right",fontsize=9);axes[1].legend()
    fig.tight_layout();fig.savefig(OUT/"05_feature_stability.png",dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    x=np.arange(len(intervals)); y=intervals.mape_pct.to_numpy()
    axes[0].errorbar(x,y,yerr=[y-intervals.ci_low_pct.to_numpy(),intervals.ci_high_pct.to_numpy()-y],fmt="o",capsize=5,color="#2563eb")
    axes[0].axhline(9.1,color="#d14e43",ls="--",label="Assignment target 9.1%")
    axes[0].set(xticks=x,xticklabels=[r.partition.replace(" (Batch ","\n(Batch ") for r in intervals.itertuples()],ylabel="MAPE (%)",title="Policy-cluster bootstrap · fixed-model 95% interval");axes[0].legend(fontsize=8)
    for number,g in domains.groupby("batch"):
        axes[1].scatter(g.cycle_life,g.ape_pct,color=COLORS[number],marker="o" if number==2 else "^",label=f"Batch {number}",alpha=.75)
    axes[1].axvline(534,color="#64748b",ls="--",lw=1,label="Batch 1 min life")
    axes[1].axvline(1074,color="#64748b",ls=":",lw=1,label="Batch 1 max life")
    axes[1].set(xlabel="Observed cycle life",ylabel="Absolute percentage error (%)",title="Error across lifetime ranges");axes[1].legend(fontsize=8)
    fig.tight_layout();fig.savefig(OUT/"06_uncertainty_and_domain.png",dpi=160);plt.close(fig)


def main(include_batch3=False, refresh_report=True):
    plt.switch_backend("Agg")
    saved=joblib.load(OUT/"models/final_model.joblib")
    frozen=json.loads((OUT/"frozen_config.json").read_text())
    assert saved["config"]==frozen["selected_config"]
    model_hash=sha256(OUT/"models/final_model.joblib")
    features=pd.read_csv(OUT/"cell_features.csv")
    audit=pd.read_csv(OUT/"cell_audit.csv")
    predictions=pd.read_csv(OUT/"predictions.csv")
    all_metrics=pd.read_csv(OUT/"metrics.csv")
    if include_batch3:
        b3,audit3=load_batch(3)
        b3["split"]="additional_test"
        p3=predict(saved["model"],b3,saved["config"])
        extra_predictions=prediction_rows(b3,p3,"Test (Batch 3)")
        extra_predictions["fold"]=np.nan
        predictions=pd.concat([predictions,extra_predictions],ignore_index=True)
        features=pd.concat([features,b3],ignore_index=True);audit=pd.concat([audit,audit3],ignore_index=True)
        all_metrics=pd.concat([all_metrics,pd.DataFrame([{"index":"Test (Batch 3)","n_cells":len(b3),**metrics(b3.cycle_life,p3)}])],ignore_index=True)
    for name,frame in [("cell_features_all",features),("cell_audit_all",audit),("predictions_all",predictions),("metrics_all",all_metrics)]:save_csv(name,frame)
    external_evaluation_evidence(features, predictions)
    extended=pd.read_csv(OUT/"model_performance.csv")
    if include_batch3:
        m2=all_metrics.loc[all_metrics["index"]=="Test (Batch 2)","mape_pct"].iloc[0]
        m3=all_metrics.loc[all_metrics["index"]=="Test (Batch 3)","mape_pct"].iloc[0]
        extra=pd.DataFrame([
            {"구분":"Test (Batch 3)","MAPE (%)":m3,"비고":"추가 44셀; 동일 Batch 1 고정 모델"},
            {"구분":"Gap (Batch2-Batch3)","MAPE (%)":m3-m2,"비고":"Batch 3 − Batch 2; (+): Batch 3 오차 증가"},
            {"구분":"Gap (Target-Test, Batch 3)","MAPE (%)":m3-9.1,"비고":"Batch 3 − 9.1%; 동일 조건 재현 아님"}])
        extended=pd.concat([extended,extra],ignore_index=True)
    save_csv("model_performance_with_batch3",extended)
    if include_batch3:
        # Author LoadData.m: remove original 38th channel, then endcap > .885,
        # then positions [3,40,41] in that reduced list (MATLAB 1-based).
        # For this source that maps to original zero-based 2,23,32,37,42,43.
        flagged={2,37,42,43}  # 23,32 already have missing targets.
        p3=predictions[predictions.batch==3]
        sensitivity=[]
        for name,subset in [("all_finite_targets_44",p3),("author_quality_filter_40",p3[~p3.cell_id.isin(flagged)])]:
            sensitivity.append({"scope":name,"n_cells":len(subset),**metrics(subset.cycle_life,subset.predicted_cycle_life)})
        save_csv("batch3_quality_sensitivity",pd.DataFrame(sensitivity))
    eda,degradation,curves=raw_evidence(features)
    save_csv("eda_cell_features",eda);save_csv("degradation_evidence",degradation)
    distributions=[];corr=[];vifs=[]
    for number,g in eda.groupby("batch"):
        y=g.cycle_life
        distributions.append({"batch":int(number),"n_cells":len(g),"min":y.min(),"q25":y.quantile(.25),"median":y.median(),"mean":y.mean(),"std":y.std(),"q75":y.quantile(.75),"max":y.max(),"skewness":y.skew(),"short_lt500":int((y<500).sum()),"long_gt1000":int((y>1000).sum())})
        for key in EDA_FEATURES:
            corr.append({"batch":int(number),"feature":key,"spearman_life":g[key].corr(y,method="spearman"),"pearson_loglife":g[key].corr(np.log10(y)),"n_valid":int(g[key].notna().sum())})
    for scope,g in [("Batch 1",eda[eda.batch==1]),("Pooled (descriptive)",eda)]:
        for group,keys in [("all10",EDA_FEATURES),("representative5",ALL_FEATURES)]:
            vifs.extend({"scope":scope,"feature_set":group,**r} for r in vif(g,keys))
    save_csv("eda_distribution",pd.DataFrame(distributions));save_csv("eda_correlations",pd.DataFrame(corr));save_csv("eda_vif",pd.DataFrame(vifs))
    policy=eda.groupby(["batch","policy"]).agg(n_cells=("cell_id","size"),mean_life=("cycle_life","mean"),median_life=("cycle_life","median"),equivalent_c=("ideal_equivalent_c","mean")).reset_index()
    save_csv("policy_summary",policy)
    from .eda_reporting import generate as generate_eda_reporting
    generate_eda_reporting(eda,degradation,curves,policy)
    intervals=pd.DataFrame([{"partition":name,**grouped_interval(g)} for name,g in predictions.groupby("partition",sort=False) if name!="Train (Batch 1 CV)"])
    save_csv("metric_uncertainty",intervals)
    bounds=features[features.batch==1][saved["features"]].agg(["min","max"])
    tests=features[features.batch!=1].merge(predictions[predictions.batch!=1][["batch","cell_id","ape_pct","predicted_cycle_life","signed_error_cycles"]],on=["batch","cell_id"],validate="one_to_one")
    out_count=np.zeros(len(tests),int)
    for key in saved["features"]:
        flag=(tests[key]<bounds.loc["min",key])|(tests[key]>bounds.loc["max",key])
        tests[key+"_outside_train_range"]=flag;out_count+=flag.to_numpy(int)
    tests["n_features_outside_train_range"]=out_count
    train_y=features[features.batch==1].cycle_life
    tests["life_outside_train_range"]=(tests.cycle_life<train_y.min())|(tests.cycle_life>train_y.max())
    save_csv("input_domain_audit",tests)
    domain_scores=[]
    for batch,g in tests.groupby("batch"):
        for name,chosen in [("all",g),("inputs_inside_train_range",g[g.n_features_outside_train_range==0]),("inputs_outside_train_range",g[g.n_features_outside_train_range>0]),("life_below_train_min",g[g.cycle_life<train_y.min()]),("life_above_train_max",g[g.cycle_life>train_y.max()])]:
            if len(chosen):domain_scores.append({"batch":int(batch),"group":name,"n_cells":len(chosen),**metrics(chosen.cycle_life,chosen.predicted_cycle_life),"overprediction_fraction":float((chosen.signed_error_cycles>0).mean())})
    save_csv("domain_metrics",pd.DataFrame(domain_scores))
    # Compare auxiliary inclusion within each fixed alpha/ratio, without retuning.
    cv=pd.read_csv(OUT/"cv_results.csv")
    from .features import FEATURE_SETS
    pairs=[]
    for key in ALL_FEATURES[1:]:
        for _,base in cv[(cv.model=="Ridge")&cv.target_scale.isna()].iterrows():
            fs=FEATURE_SETS[base.feature_set]
            if key in fs:continue
            wanted=set(fs+[key]);name=next((name for name,vals in FEATURE_SETS.items() if set(vals)==wanted),None)
            match=cv[(cv.model=="Ridge")&(cv.feature_set==name)&(cv.alpha==base.alpha)&cv.target_scale.isna()]
            if len(match):
                add=match.iloc[0];diff=np.array([add[f"fold_{k}_mape_pct"]-base[f"fold_{k}_mape_pct"] for k in range(1,5)])
                pairs.append({"added_feature":key,"base_set":base.feature_set,"expanded_set":name,"alpha":base.alpha,"delta_mape_pp":float(diff.mean()),"folds_improved":int((diff<0).sum()),"n_folds":4})
    save_csv("paired_ablation",pd.DataFrame(pairs))
    packages={p:importlib.metadata.version(p) for p in ["numpy","pandas","scikit-learn","h5py","matplotlib","nbformat","joblib","scipy"]}
    provenance={"model_sha256":model_hash,"config":frozen["selected_config"],"diagnostic_python":platform.python_version(),"packages":packages,"sources":[]}
    for n in sorted(features.batch.unique()):
        path=ROOT/"archive"/(BATCH_FILES[int(n)]+"_batchdata_updated_struct_errorcorrect.mat")
        provenance["sources"].append({"batch":int(n),"filename":path.name,"bytes":path.stat().st_size,"sha256":sha256(path)})
    assert sha256(OUT/"models/final_model.joblib")==model_hash
    (OUT/"provenance.json").write_text(json.dumps(provenance,ensure_ascii=False,indent=2))
    plot_evidence(eda,degradation,curves,intervals,tests)
    summary={"include_batch3":include_batch3,"model_sha256":model_hash,"n_cells":len(features),
             "selection_unchanged":True,"bootstrap_method":"policy-cluster percentile; fixed model; 2000 samples; seed 42",
             "batch3_exclusions":audit.loc[(audit.batch==3)&(~audit.included),"cell_id"].tolist(),
             "n_late_faster_than_middle":int(degradation.late_faster_than_middle.sum()),
             "n_knee_search_boundary":int(degradation.search_boundary.sum())}
    if include_batch3:
        summary["batch3_quality_flag_ids"]=[2,37,42,43]
    (OUT/"diagnostics_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    if refresh_report:
        from .report import main as report_main
        report_main()
    print(all_metrics.to_string(index=False));print("Diagnostics generated; frozen model unchanged",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch3",action="store_true",help="Evaluate the existing frozen model on Batch 3")
    args=parser.parse_args();main(include_batch3=args.batch3)
