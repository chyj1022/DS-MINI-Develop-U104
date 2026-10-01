"""Batch 1 selection/refit, then a frozen Batch 2 evaluation.

Run from repository root: python -m src.train
"""
import json
import platform
import argparse
import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from .features import FEATURE_SETS
from .preprocess import ROOT, load_batch, development_split

OUT = ROOT / "DAY2/results"
TARGET_MAPE = 9.1


def candidates():
    grid = [{"model": "Median", "feature_set": "variance"},
            {"model": "Linear", "feature_set": "variance"}]
    for fs in FEATURE_SETS:
        grid.extend({"model": "Ridge", "feature_set": fs, "alpha": a} for a in [.1, 1., 10.])
        grid.extend({"model": "ElasticNet", "feature_set": fs, "alpha": a, "l1_ratio": r}
                    for a in [.001, .01, .1] for r in [.2, .5, .8])
    for fs in ["variance", "full"]:
        grid.extend({"model":"SVR", "feature_set":fs,"C":c,"gamma":g,"epsilon":.02} for c in [1.,10.,100.] for g in ["scale",.1])
    grid.extend({"model":"RandomForest", "feature_set":"full", "max_depth":d,"min_samples_leaf":s} for d in [2,4] for s in [3,5])
    grid.extend([{"model":"Linear", "feature_set":"variance", "target_scale":"raw"},
                 {"model":"Ridge", "feature_set":"full", "alpha":1., "target_scale":"raw"}])
    return grid


def build_model(config):
    kind = config["model"]
    estimator = (DummyRegressor(strategy="median") if kind == "Median" else
                 LinearRegression() if kind == "Linear" else
                 Ridge(alpha=config["alpha"]) if kind == "Ridge" else
                 SVR(C=config["C"],gamma=config["gamma"],epsilon=config["epsilon"]) if kind=="SVR" else
                 RandomForestRegressor(n_estimators=100,max_depth=config["max_depth"],min_samples_leaf=config["min_samples_leaf"],random_state=42,n_jobs=1) if kind=="RandomForest" else
                 ElasticNet(alpha=config["alpha"], l1_ratio=config["l1_ratio"], max_iter=100000, tol=1e-7))
    return Pipeline([("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                     ("scaler", StandardScaler()), ("regressor", estimator)])


def metrics(y, prediction):
    assert np.isfinite(prediction).all() and (prediction > 0).all()
    return {"mape_pct": float(mean_absolute_percentage_error(y, prediction) * 100),
            "mae_cycles": float(mean_absolute_error(y, prediction)),
            "rmse_cycles": float(np.sqrt(mean_squared_error(y, prediction)))}


def fit_model(frame, config):
    model = build_model(config)
    target = frame.cycle_life if config.get("target_scale")=="raw" else np.log10(frame.cycle_life)
    model.fit(frame[FEATURE_SETS[config["feature_set"]]], target)
    return model


def predict(model, frame, config):
    pred = model.predict(frame[FEATURE_SETS[config["feature_set"]]])
    return np.maximum(pred, 1.) if config.get("target_scale")=="raw" else 10 ** pred


def folds(frame):
    return list(GroupKFold(n_splits=4).split(frame, groups=frame.policy))


def select(frame):
    records = []
    splits = folds(frame)
    for index, config in enumerate(candidates()):
        scores = []
        for tr, va in splits:
            train, valid = frame.iloc[tr], frame.iloc[va]
            assert not set(train.policy) & set(valid.policy)
            model = fit_model(train, config)
            scores.append(metrics(valid.cycle_life, predict(model, valid, config))["mape_pct"])
        records.append({"candidate_id": index, **config, "cv_mean_mape_pct": float(np.mean(scores)),
                        "cv_std_mape_pct": float(np.std(scores)), **{f"fold_{i+1}_mape_pct": s for i, s in enumerate(scores)}})
    table = pd.DataFrame(records).sort_values(["cv_mean_mape_pct", "candidate_id"]).reset_index(drop=True)
    best = candidates()[int(table.iloc[0].candidate_id)]
    return best, table


def prediction_rows(frame, pred, partition, fold=None):
    result = frame[["batch", "cell_id", "policy", "cycle_life"]].copy()
    result["partition"] = partition
    result["fold"] = fold
    result["predicted_cycle_life"] = pred
    result["signed_error_cycles"] = pred - result.cycle_life
    result["ape_pct"] = np.abs(pred - result.cycle_life) / result.cycle_life * 100
    return result


def plot_results(predictions, comparison, batch1, batch2):
    fig, ax = plt.subplots(figsize=(8, 4))
    top = comparison.groupby("model",sort=False).head(1).iloc[::-1]
    labels = [f"{r.model} / {r.feature_set}" for r in top.itertuples()]
    ax.barh(labels, top.cv_mean_mape_pct, xerr=top.cv_std_mape_pct, color="#2563eb")
    ax.set(xlabel="Development grouped CV MAPE (%) · error bar: fold SD", title="Candidate selection using Batch 1 development only")
    fig.tight_layout(); fig.savefig(OUT / "01_model_comparison.png", dpi=160); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    colors = {"Train (Batch 1 CV)": "#64748b", "Valid (Batch 1 Hold-out)": "#2563eb", "Test (Batch 2)": "#ea580c"}
    for name, group in predictions.groupby("partition", sort=False):
        axes[0].scatter(group.cycle_life, group.predicted_cycle_life, color=colors[name], label=name, alpha=.75)
    axes[0].plot([300, 1250], [300, 1250], "k--", lw=1)
    axes[0].set(xlabel="Observed cycle life", ylabel="Predicted cycle life", title="Predictions in original cycle units")
    axes[0].legend(fontsize=8)
    test = predictions[predictions.partition == "Test (Batch 2)"]
    sc = axes[1].scatter(test.cycle_life, test.signed_error_cycles, c=test.ape_pct, cmap="Reds", edgecolors="#64748b")
    fig.colorbar(sc, ax=axes[1], label="Absolute percentage error (%)")
    axes[1].axhline(0, color="black", ls="--", lw=1)
    axes[1].set(xlabel="Observed cycle life", ylabel="Prediction minus observed (cycles)", title="Batch 2 residuals")
    fig.tight_layout(); fig.savefig(OUT / "02_predictions_residuals.png", dpi=160); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(batch1.cycle_life, bins=np.arange(300,1301,100), alpha=.65, label="Batch 1 (36)")
    axes[0].hist(batch2.cycle_life, bins=np.arange(300,1301,100), alpha=.65, label=f"Batch 2 ({len(batch2)})")
    axes[0].set(xlabel="Cycle life", ylabel="Cells", title="Post-freeze distribution comparison"); axes[0].legend()
    axes[1].scatter(batch1.log10_dq_variance, batch1.cycle_life, label="Batch 1")
    axes[1].scatter(batch2.log10_dq_variance, batch2.cycle_life, label="Batch 2")
    axes[1].set(xlabel="log10 variance of Q100(V) - Q10(V)", ylabel="Cycle life", title="Delta-Q feature transfer"); axes[1].legend()
    fig.tight_layout(); fig.savefig(OUT / "03_batch_shift.png", dpi=160); plt.close(fig)


def main(include_batch3=False):
    plt.switch_backend("Agg")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "models").mkdir(exist_ok=True)
    batch1, audit1 = load_batch(1)
    dev, valid = development_split(batch1)
    # Nested selection gives an out-of-fold estimate without selecting on outer folds.
    outer_scores, outer_predictions, fold_manifest = [], [], []
    for k, (tr, va) in enumerate(folds(dev), 1):
        outer_train, outer_valid = dev.iloc[tr], dev.iloc[va]
        config, _ = select(outer_train)
        model = fit_model(outer_train, config)
        pred = predict(model, outer_valid, config)
        outer_scores.append({"fold": k, "n_train": len(tr), "n_valid": len(va),
                             "selected_config": json.dumps(config), **metrics(outer_valid.cycle_life, pred)})
        outer_predictions.append(prediction_rows(outer_valid, pred, "Train (Batch 1 CV)", k))
        for role, subset in [("train",outer_train), ("valid",outer_valid)]:
            fold_manifest.extend({"fold": k, "role": role, "cell_id": int(r.cell_id), "policy": r.policy} for r in subset.itertuples())
        print(f"Nested fold {k}: {outer_scores[-1]['mape_pct']:.3f}% / {config}", flush=True)
    config, comparison = select(dev)
    model_dev = fit_model(dev, config)
    valid_pred = predict(model_dev, valid, config)
    valid_metrics = metrics(valid.cycle_life, valid_pred)
    baseline_rows = []
    for c in candidates()[:2]:
        m = fit_model(dev, c)
        row = comparison[comparison.candidate_id == candidates().index(c)].iloc[0]
        baseline_rows.append({"model": c["model"], "development_cv_mape_pct": float(row.cv_mean_mape_pct),
                              "holdout_mape_pct": metrics(valid.cycle_life, predict(m, valid, c))["mape_pct"]})
    # Freeze choice and fit on ALL eligible Batch 1 cells before reading Batch 2.
    final_model = fit_model(batch1, config)
    frozen = {"selected_config": config, "features": FEATURE_SETS[config["feature_set"]],
              "selection_rule": "minimum mean 4-fold protocol-grouped development CV MAPE",
              "target_transform": "log10, inverse 10**prediction", "target_mape_pct": TARGET_MAPE,
              "n_final_training": len(batch1), "n_development": len(dev), "n_holdout": len(valid),
              "python": platform.python_version(), "sklearn": sklearn.__version__,
              "pandas": pd.__version__, "numpy": np.__version__}
    frozen["target_transform"] = "raw; positive clipping at 1 cycle" if config.get("target_scale")=="raw" else "log10, inverse 10**prediction"
    (OUT / "frozen_config.json").write_text(json.dumps(frozen, ensure_ascii=False, indent=2))
    joblib.dump({"model": final_model, "config": config, "features": frozen["features"]}, OUT / "models/final_model.joblib")
    joblib.dump({"model": model_dev, "config": config, "features": frozen["features"]}, OUT / "models/holdout_model.joblib")
    batch2, audit2 = load_batch(2)
    test_pred = predict(final_model, batch2, config)
    test_metrics = metrics(batch2.cycle_life, test_pred)
    train_metrics = {name: float(np.mean([r[name] for r in outer_scores])) for name in ["mape_pct", "mae_cycles", "rmse_cycles"]}
    scores = pd.DataFrame([{ "index": name, "n_cells": n, **m} for name,n,m in [
        ("Train (Batch 1 CV)",len(dev),train_metrics), ("Valid (Batch 1 Hold-out)",len(valid),valid_metrics), ("Test (Batch 2)",len(batch2),test_metrics)]])
    perf = [{"구분": r["index"], "MAPE (%)": r["mape_pct"], "비고": note} for r,note in zip(scores.to_dict("records"),
            ["개발 Nested-CV (29셀); 외부 fold 검증오차 평균", "정책 분리 7셀; 개발 29셀로 학습", "Batch 1 36셀로 재학습한 고정 모델; 39셀 평가"])]
    perf.extend([
        {"구분":"Gap (Train-Valid)", "MAPE (%)": valid_metrics["mape_pct"]-train_metrics["mape_pct"], "비고":"Valid − Train; (+): 오차 증가, 과적합 가능성"},
        {"구분":"Gap (Valid-Test)", "MAPE (%)": test_metrics["mape_pct"]-valid_metrics["mape_pct"], "비고":"Test − Valid; (+): 배치 간 일반화 저하 가능성"},
        {"구분":"Gap (Target-Test)", "MAPE (%)": test_metrics["mape_pct"]-TARGET_MAPE, "비고":"Test − 9.1%; 과제 Target 대비 참고 비교"}])
    predictions = pd.concat(outer_predictions + [prediction_rows(valid,valid_pred,"Valid (Batch 1 Hold-out)"), prediction_rows(batch2,test_pred,"Test (Batch 2)")], ignore_index=True)
    combined = pd.concat([batch1.assign(split="batch1_refit"), batch2.assign(split="test")],ignore_index=True)
    combined.loc[combined.batch == 1,"split"] = combined.loc[combined.batch == 1,"cell_id"].map(pd.concat([dev,valid]).set_index("cell_id").split)
    errors = predictions[predictions.partition == "Test (Batch 2)"].sort_values("ape_pct",ascending=False)
    subgroup = []
    for name, subset in [("standard_structure",errors[~errors.policy.str.contains("newstructure")]), ("newstructure",errors[errors.policy.str.contains("newstructure")]), ("below_batch1_life_min",errors[errors.cycle_life < batch1.cycle_life.min()])]:
        if len(subset): subgroup.append({"group":name,"n_cells":len(subset), **metrics(subset.cycle_life,subset.predicted_cycle_life), "mean_signed_error_cycles": float(subset.signed_error_cycles.mean())})
    for name,subset in [("life_lt500",errors[errors.cycle_life<500]),("life_500_to1000",errors[(errors.cycle_life>=500)&(errors.cycle_life<=1000)]),("life_gt1000",errors[errors.cycle_life>1000])]:
        if len(subset): subgroup.append({"group":name,"n_cells":len(subset), **metrics(subset.cycle_life,subset.predicted_cycle_life),"mean_signed_error_cycles":float(subset.signed_error_cycles.mean())})
    for name, table in {"cell_features":combined,"cell_audit":pd.concat([audit1,audit2]),"split_manifest":pd.concat([dev,valid])[["batch","cell_id","policy","split"]],"cv_fold_manifest":pd.DataFrame(fold_manifest),"cv_results":comparison,"nested_cv_folds":pd.DataFrame(outer_scores),"baseline_comparison":pd.DataFrame(baseline_rows),"metrics":scores,"model_performance":pd.DataFrame(perf),"predictions":predictions,"error_analysis":errors,"subgroup_metrics":pd.DataFrame(subgroup)}.items():
        table.to_csv(OUT / (name+".csv"),index=False,encoding="utf-8-sig")
    estimator = final_model.named_steps["regressor"]
    if hasattr(estimator,"coef_"):
        pd.DataFrame({"feature":frozen["features"], "standardized_log10_life_coefficient":np.atleast_1d(estimator.coef_)}).to_csv(OUT/"model_coefficients.csv",index=False)
    plot_results(predictions,comparison,batch1,batch2)
    from .report import build_report
    from .diagnostics import main as build_diagnostics
    build_diagnostics(include_batch3=include_batch3, refresh_report=False)
    build_report(frozen, scores, pd.DataFrame(perf), comparison, pd.DataFrame(baseline_rows), errors, combined, pd.DataFrame(subgroup))
    print(scores.to_string(index=False)); print(json.dumps(config))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch3",action="store_true",help="Also evaluate the frozen model on Batch 3")
    args=parser.parse_args()
    main(include_batch3=args.batch3)
