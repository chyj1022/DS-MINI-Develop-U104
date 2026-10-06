"""Additional paper-inspired experiments: early-100 regression, early-5 binary.

Feature families follow Supplementary Table 1 / Note 1. Grouped nested CV
selects parameters using Batch 1 only. Three-class logistic is an extension.
"""
import argparse
import json
import warnings

import h5py
import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LinearRegression, LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .preprocess import ROOT, BATCH_FILES, load_batch, development_split
from .paper_comparison import life_labels, classification_metrics, PARTITIONS, digest
from .train import metrics

OUT = ROOT / "DAY2/results"
DQ = ["dq_min_log", "dq_mean_log", "dq_variance_log", "dq_skew_log", "dq_kurtosis_log", "dq_2v_log"]
CAPACITY = ["capacity_slope", "capacity_intercept", "late_slope", "late_intercept", "capacity_2", "capacity_max_minus_2", "capacity_end"]
OTHER = ["charge_time", "temperature_max", "temperature_min", "temperature_integral", "ir_2", "ir_min", "ir_change"]


def columns(window, family):
    names = ["dq_variance_log"] if family == "Variance" else DQ + CAPACITY
    if window == 5:
        names = [n for n in names if n not in ["late_slope", "late_intercept"]]
    if family == "Full":
        names += OTHER
    return [f"c{window}_{n}" for n in names]


def extract_window(summary, q_early, q_end, voltage, temperature_integral, window):
    """Pure early-window calculation; later summary values are never used."""
    cycle = summary["cycle"]
    early = (cycle >= 2) & (cycle <= window)
    if len(q_early) != 1000 or len(q_end) != 1000 or len(voltage) != 1000:
        raise ValueError("Expected 1000 aligned samples")
    if not all(np.isfinite(a).all() for a in [q_early, q_end, voltage]):
        raise ValueError("Nonfinite curve")
    if not (np.all(np.diff(voltage)>0) or np.all(np.diff(voltage)<0)):
        raise ValueError("Nonmonotonic voltage")
    np.testing.assert_allclose(np.sort(voltage), np.linspace(2, 3.5, 1000), atol=1e-6)
    delta = q_end-q_early
    centered = delta-delta.mean()
    moment2 = np.mean(centered**2)
    def log_abs(x):
        return float(np.log10(abs(x))) if np.isfinite(x) and x != 0 else np.nan
    # Note 1 gives a sum in the skewness denominator; all curves have p=1000.
    result = dict(zip(DQ, [log_abs(delta.min()), log_abs(delta.mean()), log_abs(np.var(delta, ddof=1)),
        log_abs(np.mean(centered**3)/(np.sum(centered**2)**1.5)),
        log_abs(np.mean(centered**4)/(moment2**2)), log_abs(delta[np.argmin(voltage)])]))
    q = summary["QDischarge"]
    good = early & np.isfinite(q) & (q>0) & (q<=1.2)
    result["capacity_slope"], result["capacity_intercept"] = np.polyfit(cycle[good], q[good], 1) if good.sum()>=3 else [np.nan]*2
    def at(name, k, positive=False):
        a = summary[name][cycle==k]
        if len(a)!=1 or not np.isfinite(a[0]) or (positive and a[0]<=0):
            return np.nan
        if name == "QDischarge" and not 0<a[0]<=1.2:
            return np.nan
        return float(a[0])
    result["capacity_2"] = at("QDischarge", 2)
    result["capacity_max_minus_2"] = float(q[good].max()-result["capacity_2"]) if good.any() else np.nan
    result["capacity_end"] = at("QDischarge", window)
    if window == 100:
        late = good & (cycle>=91)
        result["late_slope"], result["late_intercept"] = np.polyfit(cycle[late], q[late], 1) if late.sum()>=3 else [np.nan]*2
    # Early-5 adaptation stops at 5; early-100 uses Note 1's cycles 2..6.
    ct = summary["chargetime"][(cycle>=2)&(cycle<=min(6, window))]
    ct = ct[np.isfinite(ct)&(ct>0)]
    result["charge_time"] = float(ct.mean()) if len(ct) else np.nan
    for key, array_name, fn in [("temperature_max", "Tmax", np.max), ("temperature_min", "Tmin", np.min), ("ir_min", "IR", np.min)]:
        a = summary[array_name][early]
        a = a[np.isfinite(a)&((a>0) if array_name=="IR" else True)]
        result[key] = float(fn(a)) if len(a) else np.nan
    result["temperature_integral"] = temperature_integral
    result["ir_2"] = at("IR", 2, True)
    result["ir_change"] = at("IR", window, True)-result["ir_2"]
    return {f"c{window}_{key}": float(value) for key, value in result.items()}


def read_features(number, eligible):
    rows, audit, early_curves = [], [], []
    with h5py.File(ROOT / "archive" / f"{BATCH_FILES[number]}_batchdata_updated_struct_errorcorrect.mat") as mat:
        b = mat["batch"]
        for row in eligible.itertuples():
            cid = int(row.cell_id)
            s = mat[b["summary"][cid, 0]]
            summary = {k: s[k][()].ravel() for k in ["cycle", "QDischarge", "chargetime", "IR", "Tmax", "Tmin"]}
            curves = mat[b["cycles"][cid, 0]]
            voltage = mat[b["Vdlin"][cid, 0]][()].ravel()
            qs = {}
            positions = {}
            for k in [4, 5, 10, 100]:
                pos = np.flatnonzero(summary["cycle"] == k)
                if len(pos) != 1:
                    raise ValueError(f"Batch {number} cell {cid}: cycle {k} missing/duplicate")
                positions[k] = int(pos[0])
                qs[k] = mat[curves["Qdlin"][pos[0], 0]][()].ravel()
            early_curves.append((row.cycle_life,qs[5]-qs[4]))
            integrals = {}
            for idx in np.flatnonzero((summary["cycle"]>=2)&(summary["cycle"]<=100)):
                t = mat[curves["t"][idx, 0]][()].ravel()
                temp = mat[curves["T"][idx, 0]][()].ravel()
                if len(t) != len(temp):
                    raise ValueError("Unaligned time and temperature")
                keep = np.isfinite(t)&np.isfinite(temp)
                time, temperature = t[keep], temp[keep]
                valid = len(time)>=2 and np.all(np.diff(time)>=0)
                integrals[int(summary["cycle"][idx])] = float(np.trapezoid(temperature,time)) if valid else np.nan
            values = {"batch": number, "cell_id": cid, "policy": row.policy, "cycle_life": row.cycle_life}
            for window, start in [(5, 4), (100, 10)]:
                parts = [integrals.get(k, np.nan) for k in range(2,window+1)]
                integral = float(sum(parts)) if np.isfinite(parts).all() else np.nan
                values.update(extract_window(summary, qs[start], qs[window], voltage, integral, window))
                audit.append({"batch": number, "cell_id": cid, "window": window, "temperature_expected_cycles": window-1,
                              "temperature_valid_cycles": int(np.isfinite(parts).sum()),
                              "temperature_integral_missing": not np.isfinite(integral)})
            rows.append(values)
    curve_rows=[]
    for label in range(2):
        curves_in_group=[curve for life,curve in early_curves if int(life_labels([life],"binary_550")[0])==label]
        if curves_in_group:
            q25,median,q75=np.percentile(curves_in_group,[25,50,75],axis=0)
            curve_rows.extend({"batch":number,"observed_class":label,"n_cells":len(curves_in_group),"voltage":float(v),
                               "dq_q25":float(lo),"dq_median":float(mid),"dq_q75":float(hi)}
                              for v,lo,mid,hi in zip(voltage,q25,median,q75))
    pd.DataFrame(curve_rows).to_csv(OUT/f"paper_early5_curves_batch{number}.csv",index=False,encoding="utf-8-sig")
    return pd.DataFrame(rows), pd.DataFrame(audit)


def candidates(task, family):
    if task == "regression" and family == "Selected":
        return [{**config, "family": f} for f in ["Variance", "Discharge", "Full"] for config in candidates(task,f)]
    if task == "regression":
        return [{"estimator":"Linear"}] if family == "Variance" else [
            {"estimator":"ElasticNet", "alpha":a, "l1_ratio":r} for a in [.001,.01,.1,1.] for r in [.2,.5,.8]]
    return [{"estimator":"Logistic", "C":c, "penalty":"l1" if task=="binary_550" and family=="Full" else "l2",
             "class_weight":weight} for weight in [None,"balanced"] for c in [.1,1.,10.]]


def fit(frame, task, family, config):
    if family == "Selected":
        family = config["family"]
    window = 5 if task == "binary_550" else 100
    selected = columns(window, family)
    y = np.log10(frame.cycle_life) if task == "regression" else life_labels(frame.cycle_life, task)
    fallback = task != "regression" and len(np.unique(y)) == 1
    if fallback:
        estimator = DummyClassifier(strategy="constant", constant=int(y[0]))
    elif config["estimator"] == "Linear":
        estimator = LinearRegression()
    elif config["estimator"] == "ElasticNet":
        estimator = ElasticNet(alpha=config["alpha"],l1_ratio=config["l1_ratio"],max_iter=100000,tol=1e-7)
    else:
        estimator = LogisticRegression(C=config["C"],penalty=config["penalty"],class_weight=config["class_weight"],
                                       solver="liblinear" if task=="binary_550" else "lbfgs",max_iter=10000,random_state=42)
    model = Pipeline([("imputer",SimpleImputer(strategy="median",keep_empty_features=True)),
                      ("scaler",StandardScaler()),("estimator",estimator)])
    model.fit(frame[selected],y)
    return model, selected, fallback


def evaluate(model, selected, frame, task):
    p = model.predict(frame[selected])
    if task == "regression":
        p = 10**p
        return p, metrics(frame.cycle_life, p)
    return p, classification_metrics(life_labels(frame.cycle_life,task),p,task)


def choose(frame, task, family):
    records = []
    splits = list(GroupKFold(4).split(frame,groups=frame.policy))
    for idx, config in enumerate(candidates(task,family)):
        values, constant_folds = [], 0
        for tr, va in splits:
            train, valid = frame.iloc[tr], frame.iloc[va]
            assert not set(train.policy)&set(valid.policy)
            model, selected, fallback = fit(train,task,family,config)
            _, score = evaluate(model,selected,valid,task)
            values.append(score["mape_pct"] if task=="regression" else score["macro_f1"])
            constant_folds += int(fallback)
        records.append({"candidate_id":idx,"config":json.dumps(config),"cv_score":float(np.mean(values)),
                        "cv_score_sd":float(np.std(values)),"one_class_training_folds":constant_folds})
    comparison = pd.DataFrame(records).sort_values("cv_score",ascending=task=="regression",kind="stable")
    return candidates(task,family)[int(comparison.iloc[0].candidate_id)],comparison


def run(include_batch3=True):
    snapshots = {name:digest(OUT/name) for name in ["predictions_all.csv","frozen_config.json","models/final_model.joblib"]}
    eligible1,_ = load_batch(1)
    b1,audit1 = read_features(1,eligible1)
    dev, valid = development_split(b1)
    systems = [("regression",f) for f in ["Variance","Discharge","Full","Selected"]]+[("binary_550",f) for f in ["Variance","Full"]]+[("three_500_1000","Full")]
    predictions, scores, comparisons, folds, manifest, configs, models = [], [], [], [], [], [], {}
    def add_prediction(task,family,frame,pred,partition,fold=None):
        table = frame[["batch","cell_id","policy","cycle_life"]].copy()
        table["task"],table["family"],table["partition"],table["fold"] = task,family,partition,fold
        table["prediction"] = pred
        if task != "regression":
            table["observed_class"] = life_labels(frame.cycle_life,task)
        predictions.append(table)
    for task,family in systems:
        print(f"Batch 1 nested CV: {task} / {family}",flush=True)
        outer = []
        for k,(tr,va) in enumerate(GroupKFold(4).split(dev,groups=dev.policy),1):
            train,validation = dev.iloc[tr],dev.iloc[va]
            assert not set(train.policy)&set(validation.policy)
            config,_ = choose(train,task,family)
            m,c,fallback = fit(train,task,family,config)
            pred,values = evaluate(m,c,validation,task)
            add_prediction(task,family,validation,pred,PARTITIONS[0],k)
            outer.append(values)
            folds.append({"task":task,"family":family,"fold":k,"n_train":len(train),"n_valid":len(validation),
                          "constant_fallback":fallback,"config":json.dumps(config),**values})
            manifest.extend({"task":task,"family":family,"fold":k,"role":role,"cell_id":int(r.cell_id),"policy":r.policy}
                            for role,subset in [("train",train),("valid",validation)] for r in subset.itertuples())
        cv = {key:float(np.mean([r[key] for r in outer])) for key in outer[0]}
        scores.append({"task":task,"family":family,"partition":PARTITIONS[0],"n_cells":len(dev),**cv})
        config,comparison = choose(dev,task,family)
        comparisons.append(comparison.assign(task=task,family=family))
        m,c,fallback = fit(dev,task,family,config)
        pred,values = evaluate(m,c,valid,task)
        add_prediction(task,family,valid,pred,PARTITIONS[1])
        scores.append({"task":task,"family":family,"partition":PARTITIONS[1],"n_cells":len(valid),**values})
        final,c,fallback = fit(b1,task,family,config)
        models[(task,family)] = (final,c)
        configs.append({"task":task,"family":family,"window":5 if task=="binary_550" else 100,"features":c,"selected_config":config,
                        "n_train":len(b1),"n_development":len(dev),"n_holdout":len(valid),"constant_fallback":fallback})
    # All model choices and coefficients are frozen before external raw features are read.
    (OUT/"paper_models_config.json").write_text(json.dumps(configs,ensure_ascii=False,indent=2))
    (OUT/"models").mkdir(exist_ok=True)
    joblib.dump(models,OUT/"models/paper_models.joblib")
    batches,audits = [b1],[audit1]
    for number in ([2,3] if include_batch3 else [2]):
        print(f"Frozen models: Batch {number} evaluation",flush=True)
        eligible,_ = load_batch(number)
        frame,audit = read_features(number,eligible)
        batches.append(frame);audits.append(audit)
        for task,family in systems:
            m,c = models[(task,family)]
            pred,values = evaluate(m,c,frame,task)
            add_prediction(task,family,frame,pred,f"Test (Batch {number})")
            scores.append({"task":task,"family":family,"partition":f"Test (Batch {number})","n_cells":len(frame),**values})
    for name,table in {"paper_model_features":pd.concat(batches,ignore_index=True),"paper_feature_audit":pd.concat(audits,ignore_index=True),
                       "paper_model_predictions":pd.concat(predictions,ignore_index=True),"paper_model_metrics":pd.DataFrame(scores),
                       "paper_model_candidates":pd.concat(comparisons,ignore_index=True),"paper_model_cv_folds":pd.DataFrame(folds),
                       "paper_model_fold_manifest":pd.DataFrame(manifest)}.items():
        table.to_csv(OUT/f"{name}.csv",index=False,encoding="utf-8-sig")
    coefficients=[]
    for (task,family),(model,c) in models.items():
        est = model.named_steps["estimator"]
        if hasattr(est,"coef_"):
            coefs = np.atleast_2d(est.coef_)
            for k,coef in enumerate(coefs):
                coefficients.extend({"task":task,"family":family,"coefficient_row":k,"feature":feature,"coefficient":float(value),"nonzero":abs(value)>1e-12}
                                    for feature,value in zip(c,coef))
    pd.DataFrame(coefficients).to_csv(OUT/"paper_model_coefficients.csv",index=False,encoding="utf-8-sig")
    for name,checksum in snapshots.items():
        assert digest(OUT/name)==checksum
    (OUT/"paper_models_provenance.json").write_text(json.dumps({"existing_model_unchanged":True,"immutable_inputs_sha256":snapshots,
        "feature_source":"Severson et al. 2019, Supplementary Table 1, Notes 1 and 4",
        "max_cycle":{"regression":100,"binary_550":5,"three_500_1000":100},
        "regression_variance_ddof":1,"temperature_integral":"sum of within-cycle trapezoidal integrals; cycles 2..window; degree C * min",
        "early_5_charge_time":"finite positive cycles 2..5; cycle 6 excluded",
        "model_selection":"Batch 1 development nested policy GroupKFold; minimize MAPE / maximize fixed-label macro F1",
        "one_class_fold_rule":"constant observed training class; separately flagged; no validation labels used",
        "external_evaluation":"additional analysis after the original external evaluation; no external scores used for configuration selection",
        "classification_extensions":"class_weight None/balanced chosen in development; three-group Logistic L2 is project extension"},ensure_ascii=False,indent=2))
    print(pd.DataFrame(scores).to_string(index=False),flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch3",action="store_true")
    run(parser.parse_args().batch3)
