"""Compare frozen regression and derived life groups with Severson (2019).

No fitting, feature selection, threshold search or test-cell exclusion occurs here.
The derived labels are supplemental evaluations, not trained early-5 classifiers.
"""
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
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


def reporting_format(scores, scheme):
    lookup = scores[scores.scheme == scheme].set_index('partition')
    rows = []
    gap_note = '단수명 Hold-out 0셀; 클래스 구성 차이를 포함한 참고 차이'
    for part in PARTITIONS[:3]:
        r = lookup.loc[part]
        rows.append({'구분': part, 'F1-Score (macro)': r.macro_f1,
                     'Accuracy (%)': r.accuracy_pct, '비고': f'{int(r.n_cells)}셀; 회귀 예측 구간화'})
    for label, a, b in [('Gap (Train-Valid)', PARTITIONS[0], PARTITIONS[1]),
                         ('Gap (Valid-Test)', PARTITIONS[1], PARTITIONS[2])]:
        rows.append({'구분': label, 'F1-Score (macro)': lookup.loc[a].macro_f1-lookup.loc[b].macro_f1,
                     'Accuracy (%)': lookup.loc[a].accuracy_pct-lookup.loc[b].accuracy_pct, '비고': gap_note})
    def target(part, label):
        return {'구분': label, 'F1-Score (macro)': np.nan,
                'Accuracy (%)': 95.1-lookup.loc[part].accuracy_pct if scheme=='binary_550' else np.nan,
                '비고': ('Accuracy Target 95.1%; 100회 회귀 구간화와 논문 5회 분류의 참고 비교; F1 Target 없음'
                       if scheme=='binary_550' else '논문 3구간 Target 없음; N/A')}
    rows.append(target(PARTITIONS[2], 'Gap (Target-Test)'))
    if PARTITIONS[3] in lookup.index:
        r = lookup.loc[PARTITIONS[3]]
        rows.extend([{'구분': PARTITIONS[3], 'F1-Score (macro)': r.macro_f1,
                      'Accuracy (%)': r.accuracy_pct, '비고': f'{int(r.n_cells)}셀; 동일 회귀 예측 구간화'},
                     {'구분': 'Gap (Batch2-Batch3)', 'F1-Score (macro)': lookup.loc[PARTITIONS[2]].macro_f1-r.macro_f1,
                      'Accuracy (%)': lookup.loc[PARTITIONS[2]].accuracy_pct-r.accuracy_pct, '비고': 'Batch 2 점수 − Batch 3 점수'},
                     target(PARTITIONS[3], 'Gap (Target-Test, Batch 3)')])
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
    reference, reference_cm = paper_classification_reference()
    tables.update({'paper_classification_metrics': reference, 'paper_classification_confusion': reference_cm,
                   'paper_classification_table2': pd.DataFrame([
                       {'model': model, 'train_accuracy_pct': tr, 'primary_accuracy_pct': pt, 'secondary_accuracy_pct': st}
                       for model,tr,pt,st in [('Variance classifier',82.1,78.6,97.5),('Full classifier',97.4,92.7,97.5)]])})
    for name, table in tables.items():
        output = table.iloc[:6] if name in {f"{s}_performance" for s in SCHEMES} else table
        output.to_csv(OUT / f"{name}.csv", index=False, encoding="utf-8-sig")
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


if __name__ == '__main__':
    generate()
