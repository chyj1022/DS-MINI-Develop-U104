"""Compare frozen regression and derived life groups with Severson (2019).

No fitting, feature selection, threshold search or test-cell exclusion occurs here.
The derived labels are supplemental evaluations, not trained early-5 classifiers.
"""
import hashlib
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from .preprocess import ROOT

OUT = ROOT / "DAY2/results"
PAPER = "https://www.nature.com/articles/s41560-019-0356-8"
SCHEMES = {
    "binary_550": ("short_le550", "long_gt550"),
    "three_500_1000": ("short_lt500", "middle_500_1000", "long_gt1000"),
}
NAMES = {"short_le550": "단수명 ≤550", "long_gt550": "장수명 >550",
         "short_lt500": "단수명 <500", "middle_500_1000": "중간 500–1,000", "long_gt1000": "장수명 >1,000"}
PARTITIONS = ["Train (Batch 1 CV)", "Valid (Batch 1 Hold-out)", "Test (Batch 2)", "Test (Batch 3)"]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def life_labels(values, scheme):
    """Fixed boundaries; use the same mapping for observed and predicted life."""
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Life labels require finite values")
    if scheme == "binary_550":
        return np.where(values <= 550, 0, 1)
    if scheme == "three_500_1000":
        return np.where(values < 500, 0, np.where(values <= 1000, 1, 2))
    raise ValueError(scheme)


def classification_metrics(observed, predicted, scheme):
    labels = list(range(len(SCHEMES[scheme])))
    return {"macro_f1": float(f1_score(observed, predicted, labels=labels, average="macro", zero_division=0)),
            "accuracy_pct": float(100 * accuracy_score(observed, predicted))}


def reporting_format(scores, scheme, target_f1=None, trained=False):
    lookup = scores[scores.scheme == scheme].set_index("partition")
    rows = []
    for partition in PARTITIONS[:3]:
        r = lookup.loc[partition]
        rows.append({"구분": partition, "F1-Score (macro)": r.macro_f1,
                     "Accuracy (%)": r.accuracy_pct, "비고": f"{int(r.n_cells)}셀; " + ("실제 학습 분류기" if trained else "고정 회귀 예측 구간화")})
    for label, a, b in [("Gap (Train-Valid)", PARTITIONS[0], PARTITIONS[1]),
                        ("Gap (Valid-Test)", PARTITIONS[1], PARTITIONS[2])]:
        rows.append({"구분": label, "F1-Score (macro)": lookup.loc[a].macro_f1 - lookup.loc[b].macro_f1,
                     "Accuracy (%)": lookup.loc[a].accuracy_pct - lookup.loc[b].accuracy_pct,
                     "비고": "앞 단계 점수 − 뒤 단계 점수; (+): 점수 하락"})
    def target_gap(partition):
        return {"구분": "Gap (Target-Test)", "F1-Score (macro)": target_f1-lookup.loc[partition].macro_f1 if target_f1 is not None and scheme=="binary_550" else np.nan,
                "Accuracy (%)": 95.1 - lookup.loc[partition].accuracy_pct if scheme == "binary_550" else np.nan,
                "비고": "Accuracy: 95.1 − Test; F1: 논문 혼동행렬 재계산값 − Test" if target_f1 is not None and scheme=="binary_550" else "Accuracy 95.1 − Test; F1 N/A" if scheme == "binary_550"
                else "3분류 Target·F1은 원논문에 없음; N/A"}
    rows.append(target_gap(PARTITIONS[2]))
    if PARTITIONS[3] in lookup.index:
        r = lookup.loc[PARTITIONS[3]]
        rows.extend([
            {"구분": PARTITIONS[3], "F1-Score (macro)": r.macro_f1, "Accuracy (%)": r.accuracy_pct,
             "비고": f"{int(r.n_cells)}셀; 동일 고정 모델"},
            {"구분": "Gap (Batch2-Batch3)", "F1-Score (macro)": lookup.loc[PARTITIONS[2]].macro_f1-r.macro_f1,
             "Accuracy (%)": lookup.loc[PARTITIONS[2]].accuracy_pct-r.accuracy_pct, "비고": "Batch 2 점수 − Batch 3 점수"},
            target_gap(PARTITIONS[3]),
        ])
    return pd.DataFrame(rows)


def generate():
    """Only consume frozen saved predictions; retain absent groups with n=0."""
    model_path = OUT / "models/final_model.joblib"
    before = {name: digest(OUT / name) for name in ["predictions_all.csv", "frozen_config.json"]}
    if model_path.exists():
        before["models/final_model.joblib"] = digest(model_path)
    p = pd.read_csv(OUT / "predictions_all.csv")
    features = pd.read_csv(OUT / "cell_features_all.csv")
    b1 = features[features.batch == 1]
    split = pd.read_csv(OUT / "split_manifest.csv")
    development = b1[b1.cell_id.isin(split[split.split == "development"].cell_id)]
    labels_rows, counts, scores, fold_scores, confusions, by_class, subgroups, baselines = [], [], [], [], [], [], [], []
    for scheme, names in SCHEMES.items():
        for batch, group in features.groupby("batch", sort=True):
            actual = life_labels(group.cycle_life, scheme)
            for i, name in enumerate(names):
                counts.append({"scheme": scheme, "batch": batch, "life_group": name, "n_cells": int((actual == i).sum()),
                               "batch_n_cells": len(group), "proportion_pct": float(100 * (actual == i).mean())})
        for partition, group in p.groupby("partition", sort=False):
            actual = life_labels(group.cycle_life, scheme)
            predicted = life_labels(group.predicted_cycle_life, scheme)
            label_frame = group[["batch", "cell_id", "partition", "fold", "cycle_life", "predicted_cycle_life"]].copy()
            label_frame["scheme"] = scheme
            label_frame["observed_class"] = actual
            label_frame["predicted_class"] = predicted
            labels_rows.append(label_frame)
            pooled = classification_metrics(actual, predicted, scheme)
            if partition == PARTITIONS[0]:
                folds = []
                for fold, fg in label_frame.groupby("fold", sort=True):
                    values = classification_metrics(fg.observed_class, fg.predicted_class, scheme)
                    folds.append(values)
                    fold_scores.append({"scheme": scheme, "fold": int(fold), "n_cells": len(fg), **values,
                                        "observed_classes": json.dumps(sorted(fg.observed_class.unique().tolist()))})
                values = {key: float(np.mean([v[key] for v in folds])) for key in pooled}
            else:
                values = pooled
            scores.append({"scheme": scheme, "partition": partition, "n_cells": len(group), **values,
                           "aggregation": "mean of 4 outer-fold validation scores" if partition == PARTITIONS[0] else "all cells",
                           "pooled_macro_f1": pooled["macro_f1"], "pooled_accuracy_pct": pooled["accuracy_pct"]})
            cm = confusion_matrix(actual, predicted, labels=list(range(len(names))))
            for i, name in enumerate(names):
                for j, other in enumerate(names):
                    confusions.append({"scheme": scheme, "partition": partition, "observed_group": name,
                                       "predicted_group": other, "n_cells": int(cm[i, j])})
                support, predicted_support, tp = int(cm[i].sum()), int(cm[:, i].sum()), int(cm[i, i])
                denom = support + predicted_support
                by_class.append({"scheme": scheme, "partition": partition, "life_group": name, "n_actual": support,
                                 "n_predicted": predicted_support, "n_correct": tp,
                                 "recall_pct": 100 * tp / support if support else np.nan,
                                 "f1": 2 * tp / denom if denom else np.nan})
                sub = group.iloc[np.flatnonzero(actual == i)]
                residual = sub.predicted_cycle_life-sub.cycle_life
                subgroups.append({"scheme": scheme, "partition": partition, "life_group": name, "n_cells": len(sub),
                                  "mape_pct": float((100 * residual.abs()/sub.cycle_life).mean()),
                                  "mae_cycles": float(residual.abs().mean()), "mean_signed_error_cycles": float(residual.mean()),
                                  "overpredicted_cells": int((residual > 0).sum()),
                                  "aggregation": "pooled outer-fold predictions within observed-life group" if partition == PARTITIONS[0]
                                  else "all cells within observed-life group"})
            # Majority labels come only from the appropriate training cells, including CV folds.
            if partition == PARTITIONS[0]:
                majority_prediction = np.empty(len(group), dtype=int)
                fold_baselines = []
                for fold, fg in group.groupby("fold", sort=True):
                    training = development[~development.cell_id.isin(fg.cell_id)]
                    assert not set(training.policy) & set(fg.policy)
                    majority = int(np.bincount(life_labels(training.cycle_life, scheme), minlength=len(names)).argmax())
                    positions = np.flatnonzero(group.fold.to_numpy() == fold)
                    majority_prediction[positions] = majority
                    fold_baselines.append(classification_metrics(actual[positions], majority_prediction[positions], scheme))
                baseline = {key: float(np.mean([v[key] for v in fold_baselines])) for key in pooled}
            else:
                training = development if partition == PARTITIONS[1] else b1
                majority = int(np.bincount(life_labels(training.cycle_life, scheme), minlength=len(names)).argmax())
                majority_prediction = np.full(len(group), majority)
                baseline = classification_metrics(actual, majority_prediction, scheme)
            baselines.extend([{"scheme": scheme, "partition": partition, "model": "training-majority constant", "n_cells": len(group), **baseline},
                              {"scheme": scheme, "partition": partition, "model": "frozen regression discretized", "n_cells": len(group), **values}])
    tables = {"life_class_predictions": pd.concat(labels_rows, ignore_index=True), "life_class_distribution": pd.DataFrame(counts),
              "life_class_metrics": pd.DataFrame(scores), "life_class_cv_folds": pd.DataFrame(fold_scores),
              "life_class_confusion": pd.DataFrame(confusions), "life_class_per_class": pd.DataFrame(by_class),
              "life_group_prediction_errors": pd.DataFrame(subgroups), "life_class_baselines": pd.DataFrame(baselines)}
    for scheme in SCHEMES:
        tables[f"{scheme}_performance"] = reporting_format(tables["life_class_metrics"], scheme)
    # Original paper, p. 4, Table 1. Parentheses exclude the anomalous primary-test cell.
    tables["paper_regression_metrics"] = pd.DataFrame([
        {"model": model, "train_mape_pct": tr, "primary_mape_pct": pt, "primary_excluded_mape_pct": pe,
         "secondary_mape_pct": st, "train_rmse_cycles": rt, "primary_rmse_cycles": rp,
         "primary_excluded_rmse_cycles": re, "secondary_rmse_cycles": rs, "source": PAPER, "locator": "Table 1 (PDF p.4)"}
        for model, tr, pt, pe, st, rt, rp, re, rs in [
            ("Variance", 14.1, 14.7, 13.2, 11.4, 103, 138, 138, 196),
            ("Discharge", 9.8, 13.0, 10.1, 8.6, 76, 91, 86, 173),
            ("Full", 5.6, 14.1, 7.5, 10.7, 51, 118, 100, 214)]])
    tables.update(trained_evidence())
    for name, table in tables.items():
        table.to_csv(OUT / f"{name}.csv", index=False, encoding="utf-8-sig")
    for name, checksum in before.items():
        assert digest(OUT / name) == checksum, f"Frozen input modified: {name}"
    (OUT / "paper_comparison_provenance.json").write_text(json.dumps({
        "source": PAPER, "source_pdf_name": "severson2019.pdf", "paper_tables_pdf_page": 4,
        "classification_description_pdf_page": 5, "regression_target_mape_pct": 9.1,
        "binary_target_accuracy_pct": 95.1, "three_class_target": None,
        "method": "Discretize existing early-100 regression predictions; no classifier fitting or test-based tuning",
        "binary_boundary": "short <=550, long >550; no observed/predicted value equals 550",
        "three_class_boundary": "short <500, middle 500<=life<=1000, long >1000",
        "macro_f1": "fixed 2/3 labels, zero_division=0; an absent and unpredicted label contributes 0",
        "train_aggregation": "unweighted mean of 4 existing nested outer-fold validation scores",
        "immutable_inputs_sha256": before,
    }, ensure_ascii=False, indent=2))
    plot_confusions(tables["life_class_confusion"])
    plot_trained_evidence(tables)
    return tables


def paper_classification_reference():
    """Reconstruct Macro-F1 from Supplementary Tables 4 and 6, low class first."""
    matrices = {
        "Variance classifier": [np.array([[16,4],[3,15]]),np.array([[14,5],[3,20]]),np.array([[0,1],[0,39]])],
        "Full classifier": [np.array([[19,1],[0,18]]),np.array([[19,0],[3,19]]),np.array([[0,1],[0,39]])],
    }
    metrics_rows, confusion_rows = [], []
    for model, cms in matrices.items():
        for partition, cm in zip(["Train","Primary test","Secondary test","Primary + Secondary test"],cms+[cms[1]+cms[2]]):
            denom = cm.sum(axis=0)+cm.sum(axis=1)
            f1 = np.divide(2*np.diag(cm),denom,out=np.zeros(2,dtype=float),where=denom>0)
            metrics_rows.append({"model":model,"partition":partition,"n_cells":int(cm.sum()),"accuracy_pct":100*np.trace(cm)/cm.sum(),
                                 "macro_f1_reconstructed":float(f1.mean()),"short_support":int(cm[0].sum()),
                                 "short_recall_pct":100*cm[0,0]/cm[0].sum(),"source":PAPER,
                                 "locator":"Supplementary Table 4" if model.startswith("Variance") else "Supplementary Table 6"})
            confusion_rows.extend({"model":model,"partition":partition,"observed_class":i,"predicted_class":j,"n_cells":int(cm[i,j])}
                                  for i in range(2) for j in range(2))
    return pd.DataFrame(metrics_rows),pd.DataFrame(confusion_rows)


def regression_reporting(scores):
    lookup=scores[(scores.task=="regression")&(scores.family=="Selected")].set_index("partition")
    rows=[]
    for p in PARTITIONS[:3]:
        rows.append({"구분":p,"MAPE (%)":lookup.loc[p].mape_pct,"비고":f"{int(lookup.loc[p].n_cells)}셀; 논문 피처군·파라미터를 Batch 1 내부 선택"})
    tr,va,te=[lookup.loc[p].mape_pct for p in PARTITIONS[:3]]
    rows.extend([{"구분":"Gap (Train-Valid)","MAPE (%)":va-tr,"비고":"Valid − Train; (+): 검증 오차 증가"},
                 {"구분":"Gap (Valid-Test)","MAPE (%)":te-va,"비고":"Test − Valid; (+): 외부 배치 오차 증가"},
                 {"구분":"Gap (Target-Test)","MAPE (%)":te-9.1,"비고":"Test − 9.1; 원논문 Target"}])
    mandatory=pd.DataFrame(rows)
    if PARTITIONS[3] in lookup.index:
        m3=lookup.loc[PARTITIONS[3]].mape_pct
        rows.extend([{"구분":PARTITIONS[3],"MAPE (%)":m3,"비고":"동일 Batch 1 고정 모델"},
                     {"구분":"Gap (Batch2-Batch3)","MAPE (%)":m3-te,"비고":"Batch 3 − Batch 2"},
                     {"구분":"Gap (Target-Test, Batch 3)","MAPE (%)":m3-9.1,"비고":"Batch 3 − 9.1"}])
    return mandatory,pd.DataFrame(rows)


def trained_evidence():
    scores=pd.read_csv(OUT/"paper_model_metrics.csv")
    predictions=pd.read_csv(OUT/"paper_model_predictions.csv")
    reference,reference_cm=paper_classification_reference()
    target_f1=float(reference[(reference.model=="Full classifier")&(reference.partition=="Primary + Secondary test")].macro_f1_reconstructed.iloc[0])
    tables={"paper_classification_metrics":reference,"paper_classification_confusion":reference_cm,
            "paper_classification_table2":pd.DataFrame([
                {"model":model,"train_accuracy_pct":tr,"primary_accuracy_pct":pt,"secondary_accuracy_pct":st}
                for model,tr,pt,st in [("Variance classifier",82.1,78.6,97.5),("Full classifier",97.4,92.7,97.5)]])}
    audit=[]
    for r in reference[reference.partition!='Primary + Secondary test'].itertuples():
        paper_row=tables['paper_classification_table2'].set_index('model').loc[r.model]
        expected_n={'Train':39,'Primary test':41,'Secondary test':40}[r.partition]
        reported=paper_row[{'Train':'train_accuracy_pct','Primary test':'primary_accuracy_pct','Secondary test':'secondary_accuracy_pct'}[r.partition]]
        audit.append({'model':r.model,'partition':r.partition,'caption_n_cells':expected_n,'matrix_n_cells':r.n_cells,
                      'table2_reported_accuracy_pct':reported,'matrix_computed_accuracy_pct':r.accuracy_pct,
                      'count_agrees':expected_n==r.n_cells,'rounded_accuracy_agrees':round(r.accuracy_pct,1)==reported})
    tables['paper_classification_source_audit']=pd.DataFrame(audit)
    trained=scores[(scores.task!="regression")&(scores.family=="Full")].rename(columns={"task":"scheme"})
    for scheme in SCHEMES:
        extended=reporting_format(trained,scheme,target_f1,trained=True)
        tables[f"paper_{scheme}_performance"]=extended.iloc[:6].copy()
        tables[f"paper_{scheme}_with_batch3"]=extended
        if len(extended)>6:
            formatted=extended.copy()
            formatted.insert(1,'비교 항목','')
            gap=formatted['구분'].str.startswith('Gap (')
            formatted.loc[gap,'비교 항목']=formatted.loc[gap,'구분']
            formatted.loc[gap,'구분']=''
            tables[f"paper_{scheme}_with_batch3_formatted"]=formatted
    mandatory,extended=regression_reporting(scores)
    tables["paper_selected_regression_performance"]=mandatory
    tables["paper_selected_regression_with_batch3"]=extended
    cm_rows,per_rows=[],[]
    for (task,family,partition),g in predictions[predictions.task!="regression"].groupby(["task","family","partition"],sort=False):
        names=SCHEMES[task]
        cm=confusion_matrix(g.observed_class,g.prediction,labels=range(len(names)))
        for i,name in enumerate(names):
            cm_rows.extend({"task":task,"family":family,"partition":partition,"observed_group":name,"predicted_group":names[j],"n_cells":int(cm[i,j])} for j in range(len(names)))
            per_rows.append({"task":task,"family":family,"partition":partition,"life_group":name,"n_actual":int(cm[i].sum()),
                             "n_correct":int(cm[i,i]),"recall_pct":100*cm[i,i]/cm[i].sum() if cm[i].sum() else np.nan})
    tables["paper_model_confusion"]=pd.DataFrame(cm_rows)
    tables["paper_model_class_recalls"]=pd.DataFrame(per_rows)
    selected=predictions[(predictions.task=="regression")&(predictions.family=="Selected")]
    errors=[]
    for scheme,names in SCHEMES.items():
        for partition,g in selected.groupby("partition",sort=False):
            actual=life_labels(g.cycle_life,scheme)
            for i,name in enumerate(names):
                sub=g.iloc[np.flatnonzero(actual==i)]
                residual=sub.prediction-sub.cycle_life
                errors.append({"scheme":scheme,"partition":partition,"life_group":name,"n_cells":len(sub),
                               "mape_pct":float((100*residual.abs()/sub.cycle_life).mean()),"mae_cycles":float(residual.abs().mean()),
                               "mean_signed_error_cycles":float(residual.mean()),"overpredicted_cells":int((residual>0).sum())})
    tables["paper_selected_group_errors"]=pd.DataFrame(errors)
    bias=[]
    for partition,g in selected.groupby("partition",sort=False):
        residual=g.prediction-g.cycle_life
        bias.append({"partition":partition,"n_cells":len(g),"overpredicted_cells":int((residual>0).sum()),
                     "mean_signed_error_cycles":float(residual.mean()),"mae_cycles":float(residual.abs().mean()),
                     "median_ape_pct":float((100*residual.abs()/g.cycle_life).median())})
    tables["paper_selected_prediction_bias"]=pd.DataFrame(bias)
    b1=pd.read_csv(OUT/"paper_model_features.csv");median=float(b1[b1.batch==1].cycle_life.median())
    baseline=[]
    for partition,g in selected[selected.batch>1].groupby("partition",sort=False):
        for name,pred in [("Batch 1 median",np.full(len(g),median)),("Selected Discharge ElasticNet",g.prediction.to_numpy())]:
            baseline.append({"model":name,"partition":partition,"n_cells":len(g),"mape_pct":float((100*np.abs(pred-g.cycle_life)/g.cycle_life).mean())})
    tables["paper_selected_baseline"]=pd.DataFrame(baseline)
    if PARTITIONS[3] in scores.partition.unique():
        sensitivity=[]
        third=selected[selected.batch==3]
        for scope,g in [('Batch 3 all eligible',third),('Batch 3 author quality rule',third[~third.cell_id.isin([2,37,42,43])])]:
            residual=g.prediction-g.cycle_life
            sensitivity.append({'scope':scope,'n_cells':len(g),'mape_pct':float((100*residual.abs()/g.cycle_life).mean()),
                                'mae_cycles':float(residual.abs().mean()),'rmse_cycles':float(np.sqrt(np.mean(residual**2)))})
        tables['paper_selected_batch3_quality']=pd.DataFrame(sensitivity)
    return tables


def plot_trained_evidence(tables):
    table=tables["paper_model_confusion"]
    partitions=[p for p in PARTITIONS[2:] if p in table.partition.unique()]
    fig,axes=plt.subplots(2,len(partitions),figsize=(6*len(partitions),9),squeeze=False)
    for row,(task,names) in enumerate(SCHEMES.items()):
        for col,partition in enumerate(partitions):
            sub=table[(table.task==task)&(table.family=="Full")&(table.partition==partition)]
            cm=sub.pivot(index="observed_group",columns="predicted_group",values="n_cells").loc[list(names),list(names)].to_numpy()
            ax=axes[row,col];ax.imshow(cm,cmap="Blues",vmin=0,vmax=max(1,cm.max()))
            for (i,j),v in np.ndenumerate(cm):
                ax.text(j,i,str(v),ha="center",va="center",fontsize=17,color="white" if v>cm.max()/2 else "black")
            labels=["Short <=550","Long >550"] if row==0 else ["Short <500","Middle 500-1000","Long >1000"]
            ax.set(xticks=range(len(names)),yticks=range(len(names)),xticklabels=labels,yticklabels=labels,xlabel="Predicted class",ylabel="Observed class",
                   title=f"{partition} | {'Early-5 binary' if row==0 else 'Early-100 three-class'} | n={cm.sum()}")
            ax.tick_params(axis="x",labelrotation=15)
    fig.suptitle("Actually trained Logistic models: external-batch confusion matrices",fontsize=14)
    fig.tight_layout(rect=(0,0,1,.95));fig.savefig(OUT/"11_paper_classifiers_confusion.png",dpi=160);plt.close(fig)
    features=pd.read_csv(OUT/"paper_model_features.csv")
    batches=sorted(features.batch.unique())
    fig,axes=plt.subplots(2,len(batches),figsize=(5*len(batches),8),squeeze=False)
    for col,number in enumerate(batches):
        curves=pd.read_csv(OUT/f"paper_early5_curves_batch{number}.csv")
        for label,group in curves.groupby("observed_class"):
            color="#ea580c" if label==0 else "#2563eb"
            axes[0,col].plot(group.voltage,group.dq_median,color=color,label=f"{'Short <=550' if label==0 else 'Long >550'} (n={int(group.n_cells.iloc[0])})")
            axes[0,col].fill_between(group.voltage,group.dq_q25,group.dq_q75,color=color,alpha=.15)
        axes[0,col].set(title=f"Batch {number}: Q5(V) - Q4(V)",xlabel="Voltage (V)",ylabel="Delta-Q (Ah)");axes[0,col].legend(fontsize=8)
        group=features[features.batch==number]
        color=np.where(group.cycle_life<=550,"#ea580c","#2563eb")
        axes[1,col].scatter(group.c100_dq_variance_log,group.c5_dq_variance_log,c=color)
        axes[1,col].set(xlabel="log10 Var(Q100-Q10)",ylabel="log10 Var(Q5-Q4)",title="Early-100 vs early-5 signal")
    fig.suptitle("Paper-aligned early-5 features: median/IQR and per-cell comparison",fontsize=14)
    fig.tight_layout(rect=(0,0,1,.94));fig.savefig(OUT/"12_paper_early5_signal.png",dpi=160);plt.close(fig)
    pred=pd.read_csv(OUT/'paper_model_predictions.csv')
    pred=pred[(pred.task=='regression')&(pred.family=='Selected')]
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    for partition,g in pred.groupby('partition',sort=False):
        axes[0].scatter(g.cycle_life,g.prediction,label=partition,alpha=.7)
    upper=max(pred.cycle_life.max(),pred.prediction.max())*1.05
    axes[0].plot([300,upper],[300,upper],'k--',lw=1)
    axes[0].set(xlabel='Observed cycle life',ylabel='Predicted cycle life',title='Selected paper-inspired regression');axes[0].legend(fontsize=8)
    old=pd.read_csv(OUT/'metrics_all.csv').set_index('index')
    new=pd.read_csv(OUT/'paper_model_metrics.csv');new=new[(new.task=='regression')&(new.family=='Selected')].set_index('partition')
    partitions=[p for p in PARTITIONS if p in new.index];x=np.arange(len(partitions))
    a=axes[1].bar(x-.18,[old.loc[p].mape_pct for p in partitions],width=.36,label='Original 2-feature Ridge',color='#94a3b8')
    b=axes[1].bar(x+.18,[new.loc[p].mape_pct for p in partitions],width=.36,label='Selected Discharge Elastic Net',color='#2563eb')
    axes[1].bar_label(a,fmt='%.2f',fontsize=8,padding=2);axes[1].bar_label(b,fmt='%.2f',fontsize=8,padding=2)
    axes[1].axhline(9.1,color='#ea580c',ls='--',label='Paper test Target 9.1%')
    axes[1].set(xticks=x,xticklabels=['Nested CV','Hold-out','Batch 2','Batch 3'][:len(partitions)],ylabel='MAPE (%)',title='Same cells: regression comparison');axes[1].legend(fontsize=8)
    fig.tight_layout();fig.savefig(OUT/'13_paper_regression_comparison.png',dpi=160);plt.close(fig)


def plot_confusions(table):
    partitions = [p for p in PARTITIONS[2:] if p in table.partition.unique()]
    fig, axes = plt.subplots(2, len(partitions), figsize=(6*len(partitions), 9), squeeze=False)
    for row, (scheme, names) in enumerate(SCHEMES.items()):
        for col, partition in enumerate(partitions):
            subset = table[(table.scheme == scheme) & (table.partition == partition)]
            cm = subset.pivot(index="observed_group", columns="predicted_group", values="n_cells").loc[list(names), list(names)].to_numpy()
            ax = axes[row, col]
            ax.imshow(cm, cmap="Blues", vmin=0, vmax=max(1, cm.max()))
            for (i, j), count in np.ndenumerate(cm):
                ax.text(j, i, str(count), ha="center", va="center", fontsize=17,
                        color="white" if count > cm.max()/2 else "black")
            ticks = ["Short <=550", "Long >550"] if row == 0 else ["Short <500", "Middle 500-1000", "Long >1000"]
            ax.set(xticks=range(len(names)), yticks=range(len(names)), xticklabels=ticks, yticklabels=ticks,
                   xlabel="Predicted life group", ylabel="Observed life group",
                   title=f"{partition} | {'Binary 550' if row == 0 else 'Three groups'} | n={cm.sum()}")
            ax.tick_params(axis="x", labelrotation=15)
    fig.suptitle("Supplemental grouping of frozen early-100 regression predictions\nCounts; not separately trained early-5 classifiers", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, .93))
    fig.savefig(OUT / "10_life_class_confusion.png", dpi=160)
    plt.close(fig)


def readme_section(tables):
    from .report import markdown_table, batch3_reporting_format
    mt=markdown_table
    scores=pd.read_csv(OUT/'paper_model_metrics.csv')
    configs=json.loads((OUT/'paper_models_config.json').read_text())
    config=next(r for r in configs if r['task']=='regression' and r['family']=='Selected')
    selected=scores[(scores.task=='regression')&(scores.family=='Selected')].set_index('partition')
    has3=PARTITIONS[3] in selected.index
    original=tables['paper_regression_metrics']
    paper_reg=pd.DataFrame({'원논문 모델':original.model,'Train MAPE (%)':original.train_mape_pct,
        'Primary MAPE 전체 (제외 후)':[f'{r.primary_mape_pct:.1f} ({r.primary_excluded_mape_pct:.1f})' for r in original.itertuples()],
        'Secondary MAPE (%)':original.secondary_mape_pct,
        'Primary RMSE 전체 (제외 후)':[f'{r.primary_rmse_cycles} ({r.primary_excluded_rmse_cycles})' for r in original.itertuples()],
        'Secondary RMSE (회)':original.secondary_rmse_cycles})
    reg=scores[scores.task=='regression'].pivot(index='family',columns='partition',values='mape_pct').reindex(['Variance','Discharge','Full','Selected'])
    reg=reg.rename(columns={PARTITIONS[0]:'Nested-CV MAPE (%)',PARTITIONS[1]:'Hold-out MAPE (%)',PARTITIONS[2]:'Batch 2 MAPE (%)',PARTITIONS[3]:'Batch 3 MAPE (%)'}).reset_index().rename(columns={'family':'이번 모델'})
    coeff=pd.read_csv(OUT/'paper_model_coefficients.csv')
    coeff=coeff[(coeff.task=='regression')&(coeff.family=='Selected')&coeff.nonzero][['feature','coefficient']].rename(columns={'feature':'최종 비영 피처','coefficient':'표준화 log 수명 계수'})
    c=pd.read_csv(OUT/'paper_model_candidates.csv');best=c[(c.task=='regression')&(c.family=='Selected')].iloc[0]
    ref=tables['paper_classification_metrics']
    rf=ref[(ref.model=='Full classifier')&(ref.partition=='Primary + Secondary test')].iloc[0]
    paper_c=ref[ref.model=='Full classifier'][['model','partition','n_cells','accuracy_pct','macro_f1_reconstructed','short_support','short_recall_pct']].rename(columns={
        'model':'원논문 분류기','partition':'분할','n_cells':'셀 수','accuracy_pct':'Accuracy (%)','macro_f1_reconstructed':'Macro-F1 재계산',
        'short_support':'단수명 셀','short_recall_pct':'단수명 Recall (%)'})
    counts=tables['life_class_distribution']
    cnt=counts.pivot(index=['scheme','batch'],columns='life_group',values='n_cells').reset_index()
    count_rows=[]
    for scheme,names in SCHEMES.items():
        for number,g in counts[counts.scheme==scheme].groupby('batch'):
            lookup=g.set_index('life_group');count_rows.append({'기준':'논문 장·단' if scheme=='binary_550' else 'DAY 1 단·중·장','Batch':int(number),
                '단수명':int(lookup.loc[names[0]].n_cells),'중간':int(lookup.loc[names[1]].n_cells) if len(names)==3 else '해당 없음',
                '장수명':int(lookup.loc[names[-1]].n_cells)})
    cl=scores[scores.task!='regression'][['task','family','partition','macro_f1','accuracy_pct']].rename(columns={
        'task':'태스크','family':'피처군','partition':'평가 구분','macro_f1':'Macro-F1','accuracy_pct':'Accuracy (%)'})
    cl['태스크']=cl['태스크'].map({'binary_550':'초기 5회 이진','three_500_1000':'초기 100회 3분류'})
    rec=tables['paper_model_class_recalls'];r2=rec[(rec.task=='binary_550')&(rec.family=='Full')&(rec.partition==PARTITIONS[2])&(rec.life_group=='short_le550')].iloc[0]
    three2=rec[(rec.task=='three_500_1000')&(rec.partition==PARTITIONS[2])&(rec.life_group=='short_lt500')].iloc[0]
    group=tables['paper_selected_group_errors'];group=group[group.partition.isin(PARTITIONS[2:])].copy()
    group['scheme']=group.scheme.map({'binary_550':'550회 이진 기준','three_500_1000':'500/1,000회 세 구간'})
    group['life_group']=group.life_group.map(NAMES)
    group=group.rename(columns={'scheme':'그룹 기준','partition':'평가 배치','life_group':'실제 그룹','n_cells':'셀 수',
        'mape_pct':'MAPE (%)','mae_cycles':'MAE (회)','mean_signed_error_cycles':'평균 예측−실제 (회)','overpredicted_cells':'과대예측 셀'})
    bias=tables['paper_selected_prediction_bias'];bias2=bias[bias.partition==PARTITIONS[2]].iloc[0]
    base=tables['paper_selected_baseline'];base2=base[base.partition==PARTITIONS[2]]
    median_mape=float(base2.iloc[0].mape_pct);m2=float(selected.loc[PARTITIONS[2]].mape_pct)
    binary_notes=pd.read_csv(OUT/'paper_model_cv_folds.csv');nconstant=int(binary_notes[(binary_notes.task=='binary_550')&(binary_notes.family=='Full')].constant_fallback.sum())
    additional=('\n\n### 추가 회귀 평가: Batch 3\n\n'+mt(batch3_reporting_format(tables['paper_selected_regression_with_batch3']))+
        '\n\nGap 단위는 %p입니다. 회귀 Gap은 Valid−Train, Test−Valid, Test−9.1, Batch 3−Batch 2입니다.') if has3 else ''
    b3_text=''
    if has3:
        r3=rec[(rec.task=='binary_550')&(rec.family=='Full')&(rec.partition==PARTITIONS[3])&(rec.life_group=='short_le550')].iloc[0]
        b3_text=f'Batch 3 이진 단수명 {int(r3.n_actual)}셀의 Recall은 {r3.recall_pct:.1f}%입니다. 높은 전체 Accuracy와 단수명 식별 성능을 구분해 해석합니다.'
    return f'''## 원논문 기반 추가 개발·비교

### 1. 원논문 설계를 실제 구현한 추가 실험

[Severson et al. (2019) 원논문]({PAPER})의 본문 Table 1/2와 같은 논문의 Supplementary Table 1·4·5·6, Notes 1·4를 확인하고 **회귀 피처 확장, 초기 5회 이진 분류, 단·중·장 3분류, 그룹별 회귀 오차**를 실제 구현·학습·평가했습니다. 참고문헌은 원논문 한 편이며 보충자료는 그 논문의 일부입니다.

| 실험 | 관측 데이터 | 구현 | 평가 기준 |
| --- | --- | --- | --- |
| Variance 회귀 | 초기 100회 | log Var(Q100−Q10) 단일 피처, 선형 log 수명 회귀 | MAPE·MAE·RMSE |
| Discharge 회귀 | 초기 100회 | ΔQ 6개 + 용량 7개 = 13개 후보, Elastic Net | Batch 1 CV 선택, Batch 2/3 평가 |
| Full 회귀 | 초기 100회 | 위 13개 + 충전 시간·온도·IR 7개 = 20개 후보, Elastic Net | 같은 분할·전처리·선택 규칙 |
| Variance 이진 분류 | **초기 5회** | log Var(Q5−Q4), Logistic Regression | 수명 550회 경계, Macro-F1·Accuracy |
| Full 이진 분류 | **초기 5회** | 18개 후보, **L1 Logistic Regression** | 수명 550회 경계, Macro-F1·Accuracy·혼동행렬 |
| 단·중·장 분류 | 초기 100회 | 20개 후보, L2 Logistic Regression | <500 / 500–1,000 / >1,000회; 프로젝트 확장 |

새 분석은 회귀값을 구간으로 나누는 것에 그치지 않고 **분류기를 따로 학습**했습니다. 초기 5회 피처에는 6회 이후 자료가 들어가지 않습니다. 기존 회귀 예측의 구간화 결과도 `life_class_*.csv`로 보존해 실제 분류기와 구분합니다.

#### 피처 계산과 논문 대응

- ΔQ: 2.0–3.5V의 공통 1,000점에서 회귀는 Q100−Q10, 이진 분류는 Q5−Q4를 계산합니다. 최솟값·평균·표본 분산·왜도·첨도·2V 값을 보충자료의 log 절댓값 형태로 변환합니다. 분산은 논문 식의 **ddof=1**을 적용합니다.
- 용량: 2회부터 관측 끝까지의 선형 기울기·절편, Q2, 초기 최대 Q−Q2, 관측 끝 Q를 구현합니다. 91–100회의 기울기·절편은 100회 회귀에만 추가합니다.
- 충전·온도·IR: 충전 시간 평균, 최고·최저 온도, **원시 T(t)와 t의 사다리꼴 온도 적분**, IR2·최소 IR·끝 회차−2회 IR을 구현합니다. 온도 적분은 사이클 내부 시간을 이용해 적분한 뒤 관측 구간에서 합하며 단위는 °C·min입니다.
- 충전 시간은 회귀에서 논문 Note 1 식의 2–6회, 초기 5회 분류에서는 2–5회 유한 양수 값만 평균냅니다. 초기 5회 정의를 지키기 위해 6회 값은 제외합니다. 0 이하 IR은 결측, 결측 대치·표준화는 각 학습 fold 안에서 적합합니다.

원본 셀별 38개 계산 피처는 `paper_model_features.csv`, 온도 적분의 유효 회차 수는 `paper_feature_audit.csv`에 있습니다. 피처 추출은 `src/paper_models.py`의 `extract_window`·`read_features`로 재현합니다.

![논문 초기 5회 ΔQ 그룹 곡선과 초기 100회 신호 비교](DAY2/results/12_paper_early5_signal.png)

상단은 실제 550회 수명 그룹별 Q5−Q4 중앙값과 IQR, 하단은 셀별 초기 100회와 초기 5회 log 분산을 비교합니다. 서로 다른 관측 시점의 신호를 같은 피처로 간주하지 않고 각각 학습·평가했습니다. Batch 1 단수명 곡선은 1셀로 IQR이 0이며 집단 변동을 추정하지 않습니다.

### 2. 논문 피처군별 회귀 모델 비교와 최종 선택

{mt(reg)}

Selected 행은 **Variance·Discharge·Full의 피처군과 파라미터를 내부 CV에서 함께 선택한 Nested-CV**입니다. 각 외부 검증 fold는 피처군 선택에도 사용하지 않습니다. 개발 29셀의 최종 선택 CV MAPE **{best.cv_score:.3f}%**로 **{config['selected_config']['family']} / {config['selected_config']['estimator']}**, `alpha={config['selected_config'].get('alpha')}`, `l1_ratio={config['selected_config'].get('l1_ratio')}`가 선택됐습니다. 최종 계수는 Batch 1 전체 36셀로 재학습하고 고정했습니다. Batch 2/3 점수로 모델을 바꾸지 않습니다.

기존 2피처 Ridge의 개발 선택 CV는 9.024%, Nested-CV는 10.842%였습니다. 새 설계는 논문의 후보 피처와 Elastic Net 선택을 반영했으며, 최종 Discharge 모델의 13개 후보 중 {len(coeff)}개가 비영 계수로 남았습니다.

{mt(coeff)}

계수는 표준화 입력에 대한 log10 수명 계수입니다. 상관된 피처들의 조건부 계수이므로 물리적 인과 효과로 해석하지 않습니다. 후보·외부 fold별 선택·전처리 및 정책 분리 근거는 `paper_model_candidates.csv`, `paper_model_cv_folds.csv`, `paper_model_fold_manifest.csv`, `paper_models_config.json`에 있습니다.

### 3. 원논문 회귀 Table 1과 비교

{mt(paper_reg)}

논문 Methods의 Mean percent error는 이번 MAPE와 같은 `100 × mean(|y−ŷ|/y)`입니다. 괄호는 논문이 특이한 Primary-test 셀 1개를 제외한 결과입니다. 과제의 **Target 9.1%는 논문 Abstract의 대표 성능**으로, Table 1의 Full Primary 전체값 14.1%와 구분합니다. 논문의 Train은 학습셋 성능이고 이번 Train은 Nested-CV 검증 평균입니다.

추가 최종 Discharge의 Batch 2 MAPE는 **{m2:.3f}%**, Target Gap은 **{m2-9.1:+.3f}%p**입니다. 기존 Ridge 28.609%에서 **{28.60918728493949-m2:.3f}%p 감소**했지만 Target 9.1%에는 도달하지 못했습니다. Batch 2 MAE/RMSE는 **{selected.loc[PARTITIONS[2]].mae_cycles:.3f} / {selected.loc[PARTITIONS[2]].rmse_cycles:.3f}회**입니다. 기존 Ridge 141.752 / 158.676회와 비교하면 **MAPE 개선이 MAE·RMSE 개선까지 뜻하지는 않습니다.** 짧은 수명을 더 크게 반영하는 MAPE로 선택한 결과와 큰 절대 오차를 함께 평가합니다.

원논문의 Primary test는 2017-05-12·2017-06-30 혼합 분할이고 과제 Batch 2는 2018-02-20입니다. 따라서 같은 정의의 지표·Target을 비교하되 동일 데이터 분할의 재현 성능으로 표시하지 않습니다. 이번 추가 실험은 기존 외부 결과를 확인한 이후의 분석이며 새 미사용 테스트셋을 확보한 실험은 아닙니다. 모델 파라미터 선택에는 Batch 1만 사용했습니다.

### 4. 장·단과 단·중·장 그룹 구성

{mt(pd.DataFrame(count_rows))}

논문 이진 기준을 **단수명 ≤550 / 장수명 >550회**로 구현했습니다. 현재 실제값·예측값에 정확히 550회가 없어 등호 방향은 이번 결과에 영향을 주지 않습니다. 세 구간은 DAY 1 기준을 유지합니다. 동일한 '단수명'이라는 이름도 경계가 달라 두 그룹의 셀 수가 다릅니다.

### 5. 실제 학습한 초기 5회 분류: 논문과의 비교

{mt(tables['paper_classification_table2'].rename(columns={'model':'원논문 분류기','train_accuracy_pct':'Train Accuracy (%)','primary_accuracy_pct':'Primary Accuracy (%)','secondary_accuracy_pct':'Secondary Accuracy (%)'}))}

위는 본문 Table 2의 보고값을 그대로 옮긴 표입니다. 아래는 Full classifier의 보충자료 혼동행렬 셀 개수에서 별도로 재계산한 결과입니다.

{mt(paper_c)}

논문 Table 2의 Accuracy에 더해 **보충자료 Table 4·6의 혼동행렬에서 Macro-F1과 단수명 Recall을 재계산**했습니다. 이는 논문이 F1을 직접 보고했다는 뜻이 아닙니다. Full의 Primary+Secondary 81셀에서 Accuracy {rf.accuracy_pct:.3f}%가 한 자리 반올림으로 **95.1%**, 재계산 Macro-F1은 **{rf.macro_f1_reconstructed:.6f}**입니다. 이 두 값을 이진 Target 비교의 근거로 사용합니다. 보충자료 캡션은 Train 39셀로 적었지만 Full 혼동행렬의 합은 38셀입니다. Variance 혼동행렬도 캡션·본문 Accuracy와 일부 불일치합니다. 보고값을 임의로 수정하지 않고 `paper_classification_source_audit.csv`에 대조 결과를 저장했으며, F1은 명시된 혼동행렬 셀 개수로 계산했습니다.

{mt(cl)}

단수명 재현율을 포함한 세부 결과는 `paper_model_class_recalls.csv`에 있습니다. 초기 5회 Full 모델은 L1 정규화 강도와 `class_weight=None/balanced`를 Batch 1 내부 Macro-F1로 선택합니다. class_weight 비교는 표본 불균형을 반영한 프로젝트 확장입니다. 이진 분류의 외부 CV 학습셋이 한 클래스뿐인 {nconstant}개 fold는 그 학습 클래스만 예측하는 상수 모델을 적용하고 `constant_fallback`으로 기록했습니다. 검증 라벨로 학습 클래스를 보완하지 않습니다. 따라서 CV 평균은 이 fallback을 포함한 파이프라인의 결과입니다.

### 6. 장·단 이진 분류 성능표: 초기 5회 Full

{mt(tables['paper_binary_550_performance'])}

{('#### 초기 5회 이진 분류: Batch 3 추가 양식' + chr(10)*2 + mt(tables['paper_binary_550_with_batch3_formatted'])) if has3 else ''}

F1은 고정된 2개 클래스의 **Macro-F1(0–1)**, Accuracy는 **%(0–100)**입니다. Train은 4개 Nested-CV 외부 fold 점수의 단순 평균입니다. 클래스가 실제·예측 모두 없으면 해당 클래스 F1을 0으로 포함(`zero_division=0`)합니다. Hold-out은 단수명 정답이 없어 Accuracy 100%가 단수명 식별의 검증을 뜻하지 않습니다.

점수는 높을수록 좋으므로 Gap (Train-Valid)=Train−Valid, Gap (Valid-Test)=Valid−Test, Gap (Target-Test)=Target−Test, Gap (Batch2-Batch3)=Batch 2−Batch 3입니다. F1 Gap은 점수 차이, Accuracy Gap은 %p입니다. Target F1은 위 논문 혼동행렬에서 재계산한 값이고, Target Accuracy는 과제 지정 95.1%입니다.

### 7. 단·중·장 분류 성능표: 초기 100회 Full

{mt(tables['paper_three_500_1000_performance'])}

{('#### 단·중·장 분류: Batch 3 추가 양식' + chr(10)*2 + mt(tables['paper_three_500_1000_with_batch3_formatted'])) if has3 else ''}

고정된 3개 클래스의 Macro-F1을 사용합니다. 3분류 Target은 원논문에 없어 `—`로 둡니다. Batch 1의 관측 클래스는 중간·장수명 두 개이며, 학습된 Logistic 모델은 이 관측 클래스에 대해 계수를 적합합니다. <500회 클래스는 학습셋에 없으므로 외부 셀의 단수명 Recall도 같이 제시합니다. 3분류를 이진 논문 Accuracy 95.1%와 동등한 실험으로 비교하지 않습니다.

![논문 기반 초기 5회 이진 및 초기 100회 3분류의 혼동행렬](DAY2/results/11_paper_classifiers_confusion.png)

행은 실제, 열은 예측 클래스입니다. Batch 2의 이진 단수명 **{int(r2.n_actual)}셀 중 {int(r2.n_correct)}셀**을 맞혀 Recall **{r2.recall_pct:.1f}%**입니다. 3분류의 <500회 **{int(three2.n_actual)}셀 중 {int(three2.n_correct)}셀**을 맞혔습니다. {b3_text} 원논문에서도 Secondary-test 단수명은 1셀로, 혼동행렬과 재현율을 함께 보면 Accuracy의 구성 차이를 확인할 수 있습니다.

### 8. 최종 회귀를 같은 수명 기준으로 나눈 오류 분석

{mt(group)}

이는 실제 수명으로 나눈 사후 오류 분석입니다. 구간별 MAPE·MAE·과대예측을 함께 확인하고 0셀은 `—`로 둡니다. 논문은 이 세 구간별 회귀 오차를 보고하지 않아 위 구간에 별도의 논문 Target을 만들지 않습니다. 원논문과의 전체 MAPE 비교는 주 성능표에서 수행합니다.

추가 최종 모델은 Batch 2 **{int(bias2.n_cells)}셀 중 {int(bias2.overpredicted_cells)}셀**을 과대예측했습니다. 평균 예측−실제는 **{bias2.mean_signed_error_cycles:+.1f}회**, 오차율 중앙값은 **{bias2.median_ape_pct:.2f}%**입니다. Batch 1 중앙값 772.5회 기준선 MAPE **{median_mape:.2f}%** 대비 새 모델 **{m2:.2f}%**로 상대 오차가 **{100*(1-m2/median_mape):.2f}% 감소**했습니다. 낮은 수명 셀의 교체 시점 판단에는 그룹별 편향도 함께 반영해야 합니다.

{('#### 최종 회귀 Batch 3의 품질 제외 민감도' + chr(10)*2 + mt(tables['paper_selected_batch3_quality']) + chr(10)*2 + '같은 저자 품질 규칙(원본 ID 2·37·42·43)을 추가 최종 모델에도 적용했습니다. 전체 44셀 평가를 유지하고 40셀 민감도를 나란히 보고하며, 오차 크기로 제외 대상을 정하지 않습니다.') if has3 else ''}

### 추가 분석의 실행·계산 근거

```bash
python -m src.paper_models --batch3   # 원본 피처 추출, 실제 추가 학습과 평가
python -m src.report                  # 논문 비교표·그림·README·노트북 갱신
python -m src.verify_results          # 기존 모델 및 추가 실험의 계산·누수 검증
```

전체 `python -m src.train --batch3`에도 추가 실험을 연결했습니다. 기존 Ridge 모델·예측·설정은 비교 기준으로 보존하며 해시 일치를 검사합니다. 논문 수치는 `paper_regression_metrics.csv`, `paper_classification_metrics.csv`, `paper_classification_confusion.csv`, 추가 실험은 `paper_model_*.csv`, 주 성능표는 `paper_selected_regression_performance.csv`, 이진·3분류 양식은 `paper_binary_550_performance.csv`·`paper_three_500_1000_performance.csv`에 있습니다.
'''


if __name__ == "__main__":
    plt.switch_backend("Agg")
    results = generate()
    print(results["life_class_metrics"].to_string(index=False))
