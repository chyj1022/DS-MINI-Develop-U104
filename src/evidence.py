"""Attach computed rubric evidence and direct reproduction to the report."""
import ast
import base64
import contextlib
import io
import json
import re

import nbformat
import numpy as np
import pandas as pd

from .preprocess import ROOT
from .train import OUT


def enhance_report():
    from .report import markdown_table, batch3_reporting_format
    from .diagnostics import sha256, external_evaluation_evidence
    from .eda_reporting import readme_section
    read=lambda name:pd.read_csv(OUT/(name+".csv"))
    summary=json.loads((OUT/"diagnostics_summary.json").read_text())
    provenance=json.loads((OUT/"provenance.json").read_text())
    frozen=json.loads((OUT/"frozen_config.json").read_text())
    assert summary["model_sha256"]==sha256(OUT/"models/final_model.joblib")
    distribution=read("eda_distribution")
    vifs=read("eda_vif")
    vif_summary=vifs.groupby(["scope","feature_set"]).agg(n_cells=("n_cells","first"),max_vif=("vif","max")).reset_index()
    domain=read("domain_metrics")
    intervals=read("metric_uncertainty")
    inside=domain[(domain.batch==2)&(domain.group=="inputs_inside_train_range")].iloc[0]
    outside=domain[(domain.batch==2)&(domain.group=="inputs_outside_train_range")].iloc[0]
    all_scores=read("metrics_all")
    extra_performance=read("model_performance_with_batch3")
    formatted_performance=batch3_reporting_format(extra_performance)
    if summary['include_batch3']:
        formatted_performance.to_csv(OUT/'model_performance_with_batch3_formatted.csv',index=False,encoding='utf-8-sig')
    cfg=frozen["selected_config"]
    scores=read("cv_results")
    best_core=scores[(scores.feature_set=="variance")&scores.model.isin(["Ridge","ElasticNet"])].iloc[0]
    effect=float(best_core.cv_mean_mape_pct-scores.iloc[0].cv_mean_mape_pct)
    has3=summary["include_batch3"]
    quality=read("batch3_quality_sensitivity") if has3 else None
    baseline, bias, overlap = external_evaluation_evidence(read("cell_features"), read("predictions"))
    baseline_mape = float(baseline.iloc[0].mape_pct)
    model_mape = float(baseline.iloc[1].mape_pct)
    reduction = float(baseline.iloc[1].mape_reduction_vs_baseline_pct)
    median_prediction = float(baseline.iloc[0].constant_prediction_cycles)
    all_bias, below_bias = bias.iloc[0], bias.iloc[1]
    assert bool(overlap.iloc[0].identical_cell_sets)
    nested = read("nested_cv_folds")
    cv_range = f"{nested.mape_pct.min():.2f}~{nested.mape_pct.max():.2f}"
    assessment=pd.DataFrame([
        {"평가항목":"EDA","배점":50,"직접 근거":"분포·열화·ΔQ·정책·상관·VIF를 원본 재계산","산출물":"eda_distribution / degradation_evidence / eda_correlations / eda_vif"},
        {"평가항목":"EDA → 전략 연결성","배점":30,"직접 근거":"관측→시사점→설계→코드→검증 연결표","산출물":"노트북 1~3절"},
        {"평가항목":"모델링 전략 수립","배점":20,"직접 근거":"회귀·정규화·212후보·기준선·개선 불확실성","산출물":"cv_results / feature_ablation / paired_ablation"},
        {"평가항목":"전략 → 구현 반영","배점":20,"직접 근거":"100회 이내 피처·타깃 변환·실제 선택 모델","산출물":"features.py / model_coefficients / 노트북 직접 추출"},
        {"평가항목":"Pipeline 개발","배점":40,"직접 근거":"원본→정책 분리→fold 내부 전처리→선택→재학습→평가","산출물":"train.py / 직접 재현 셀 / provenance / verify_results.py"},
        {"평가항목":"성능 리포팅 및 해석","배점":20,"직접 근거":"지정 형식·Gap 방향·동일 Batch 2 기준선·불확실성·조건 차이","산출물":"model_performance / batch2_baseline_comparison / metric_uncertainty"},
        {"평가항목":"분석 결과 해석","배점":20,"직접 근거":"과대예측 편향·동일 하위집단 확인·입력 범위 반례·ESS 해석","산출물":"prediction_bias / subgroup_overlap / domain_metrics / ESS 의사결정표"},
    ])
    assessment.to_csv(OUT/"assessment_evidence.csv",index=False,encoding="utf-8-sig")
    core_vif=float(vif_summary[(vif_summary.scope=="Batch 1")&(vif_summary.feature_set=="representative5")].max_vif.iloc[0])
    full_vif=float(vif_summary[(vif_summary.scope=="Batch 1")&(vif_summary.feature_set=="all10")].max_vif.iloc[0])
    eda_text=f"""### 원본에서 다시 확인한 통계·해석

분포는 평균·SD·IQR·장단수명 수를 함께 확인합니다. Batch 1의 36셀에는 <500회 셀이 없고 Batch 2에는 28셀이 있어 회귀를 선택했습니다. 전체를 합친 통계만으로 배치 이동을 숨기지 않습니다.

{markdown_table(distribution)}

전체 {summary['n_cells']}셀 중 {summary['n_late_faster_than_middle']}셀은 관측 말기 1/3의 용량 기울기가 중기보다 더 음수입니다. 구간선형 knee 탐색의 {summary['n_knee_search_boundary']}셀은 탐색 경계에 걸립니다. 후반 가속과 정확한 knee 시점의 확정은 구분해야 합니다. 진행률·knee·말기 기울기는 **설명용 미래 관측이며 모델 입력이 아닙니다**.

학습 배치만 계산한 전체 10개 피처 최대 VIF는 {full_vif:.2f}, 대표 5개는 {core_vif:.2f}입니다. 통합 VIF와 학습 VIF는 다르며 각 세트에서 결측을 제외한 n을 기록합니다. 0 이하 IR을 결측 처리한 이번 QC에서는 통합 VIF 계산 표본이 선행 DAY 1과 달라질 수 있습니다. 정책별 수명·등가 C-rate는 `policy_summary.csv`, 상관은 `eda_correlations.csv`에 있습니다.

![ΔQ와 열화 근거](DAY2/results/04_eda_curve_evidence.png)
![피처 중복과 배치 안정성](DAY2/results/05_feature_stability.png)
"""
    baseline_text=f"""### 같은 Batch 2에서의 기준선 비교

Batch 1 전체 36셀의 **산술 중앙값 {median_prediction:.1f}회**를 Batch 2 모든 셀에 동일하게 예측하는 기준선입니다. Batch 2 정답을 이용해 중앙값을 정하지 않았으며, 최종 모델과 같은 39셀에서 비교했습니다. 기존 `baseline_comparison.csv`의 개발 CV·Hold-out 기준선과 구분합니다.

{markdown_table(baseline[['model','n_cells','mape_pct','mae_cycles','rmse_cycles','mape_reduction_vs_baseline_pct']])}

모델 MAPE {model_mape:.2f}%는 기준선 {baseline_mape:.2f}%보다 **상대적으로 {reduction:.2f}% 감소**했습니다. 이는 예측 신호가 있음을 보여주지만 원논문 Target이나 교체 의사결정에 필요한 정확도를 충족했다는 뜻은 아닙니다. 비교 대상은 단순 상수 기준선으로 한정됩니다.
"""
    bias_text=f"""### 일부 이상치보다 넓게 나타나는 과대예측 편향

Batch 2 **{int(all_bias.n_cells)}셀 중 {int(all_bias.overpredicted_cells)}셀**을 과대예측했고 평균 부호 오차(예측−실제)는 **{all_bias.mean_signed_error_cycles:+.1f}회**, 셀별 절대 오차율의 중앙값은 **{all_bias.median_ape_pct:.2f}%**입니다. 일부 큰 오차만으로 평균 MAPE가 높아진 상황으로 설명하기 어렵습니다.

{markdown_table(bias)}

Batch 1 최소 수명보다 짧은 **{int(below_bias.n_cells)}셀 모두 과대예측**됐습니다. 이 집단에서는 MAE와 평균 부호 오차가 모두 {below_bias.mae_cycles:.1f}회입니다. 단수명 셀의 과대예측은 교체 지연 위험으로 이어질 수 있어 현재 모델을 단독 교체 시점 결정에 사용하기 어렵습니다. 표본 수만으로 원인을 설명하기보다 학습 수명 범위·배치 분포·피처와 수명의 관계를 먼저 점검해야 합니다.

`standard_structure`와 `below_batch1_life_min`은 **정확히 같은 {int(overlap.iloc[0].n_intersection)}셀**입니다. 두 그룹 표는 동일 집합의 재표현이며 구조와 수명 범위의 독립적인 효과를 뒷받침하지 않습니다. 비교 근거는 `subgroup_overlap.csv`에 저장했습니다.
"""
    uncertainty_text=f"""### 오차 불확실성

과제의 Train은 학습 오차가 아니라 **개발 Nested-CV** 검증오차입니다. 외부 fold별 MAPE는 **{cv_range}%**이고 Hold-out은 7셀뿐입니다. 평균과 Hold-out이 가깝고 검증에서 오차가 급증하지 않았지만, −0.407%p 차이를 안정성의 강한 증거로 해석할 수 없습니다.

충전 정책을 복원추출하고 선택된 정책의 모든 셀을 함께 포함하는 **정책 단위 bootstrap**(seed=42, 2,000회)으로 고정 모델 MAPE의 95% percentile 구간을 계산했습니다. 이 구간은 평가 표본의 불확실성만 나타내며 재학습·미래 배치의 불확실성이나 셀별 예측 구간은 아닙니다. Hold-out은 4정책뿐이라 구간 안정성도 제한됩니다.
"""+"\n"+markdown_table(intervals)
    domain_text=f"""### 오류 원인 가설을 반례와 함께 점검

입력 피처가 Batch 1 최솟값~최댓값 안인 **{int(inside.n_cells)}셀의 MAPE {inside.mape_pct:.3f}%**가 범위 밖 **{int(outside.n_cells)}셀의 {outside.mape_pct:.3f}%**보다 큽니다. 따라서 **입력 범위 이탈만으로 Batch 2 오차를 설명할 수 없습니다.** 같은 입력 범위에서도 수명과의 관계가 배치마다 달라졌을 가능성, 정책·실험 조건의 교란을 함께 고려해야 합니다.

실제 수명이 학습 범위 밖이라는 구분은 정답을 본 뒤의 오류 분석용입니다. 배포 시에는 실제 수명을 모르므로 입력 분포·신규 정책·현장 라벨 수집으로 감시해야 합니다. 각 피처가 범위 안이라는 사실만으로 다변량 분포까지 같다고 보장되지 않습니다. `standard_structure` 30셀과 학습 최소 수명 미만 30셀은 동일 집합이므로 구조와 수명 범위의 효과를 분리할 수 없습니다.

{markdown_table(domain)}

![오차 불확실성과 수명 범위](DAY2/results/06_uncertainty_and_domain.png)
"""
    selection_text=f"""### 작은 CV 개선을 해석하는 기준

단일 ΔQ 피처 CV **{best_core.cv_mean_mape_pct:.5f}%**와 충전 시간을 추가한 선택 후보 **{scores.iloc[0].cv_mean_mape_pct:.5f}%**의 차이는 **{effect:.5f}%p**로 매우 작습니다. 충전 시간 피처의 실질적 개선이 입증됐다고 주장할 수 없습니다. '최적 모델'은 현재 후보군·고정 분할·선택 지표에서의 최저 후보를 의미합니다. 충전 시간의 선택 근거는 재검토 대상으로 명시하고 다음 개발 실험에서 단일 피처의 단순 모델을 우선 비교합니다. 현재 시험 결과로 모델을 바꾸지 않으며 개선 모델은 후속 미사용 배치에서 검증해야 합니다.

`paired_ablation.csv`는 같은 Ridge alpha·같은 CV fold에서 피처 하나의 추가 효과를 비교합니다. delta가 음수일 때 개선입니다. 최저 파라미터를 각각 고른 조합 비교와 구분해 해석합니다. 이번 고정 모델은 Batch 2/3 결과로 재선택하지 않았습니다.
"""
    b3_text=""
    if has3:
        m3=float(all_scores[all_scores['index']=='Test (Batch 3)'].mape_pct.iloc[0])
        m2=float(all_scores[all_scores['index']=='Test (Batch 2)'].mape_pct.iloc[0])
        b3_text=f"""### Batch 3 추가 검증·품질 민감도

동일 Batch 1 고정 모델로 44셀을 평가한 MAPE는 **{m3:.3f}%**, Batch 3−Batch 2는 **{m3-m2:+.3f}%p**입니다. 이번 모델은 Batch 3보다 Batch 2의 짧은 수명에서 더 취약합니다. 배치별 난이도는 수명 분포·조건·모델에 따라 달라집니다.

{markdown_table(formatted_performance)}

Gap (Train-Valid)은 Valid − Train, Gap (Valid-Test)은 Test − Valid로 계산합니다. Target Gap은 해당 배치 Test − 9.1, Batch2-Batch3 Gap은 Batch 3 − Batch 2이며 단위는 %p입니다.

수명 라벨이 있는 44셀 결과를 유지하고, [저자 LoadData.m](https://github.com/rdbraatz/data-driven-prediction-of-battery-cycle-life-before-capacity-degradation/blob/master/LoadData.m)의 알려진 품질 제외 규칙을 적용한 40셀 결과도 민감도로 보고합니다. 원본 ID 23·32는 라벨 결측이고, 추가 점검 ID **2·37·42·43**은 원논문 중간 삭제 후 인덱스를 원본 0-based로 환산한 값입니다. 오차 크기를 보고 정한 제외 규칙이 아닙니다.

{markdown_table(quality)}

공통 전압축 2.0~3.5V·1,000점을 확인했고 ΔQ 분산은 곡선의 상수 원점 이동에 불변임을 검사했습니다. 이는 배치별 충전·방전 절차나 곡선 형태의 차이가 모두 교정됐다는 뜻은 아닙니다. 평균·최솟값에 동일한 불변성을 가정하지 않습니다.
"""
    ess_table=pd.DataFrame([
        {"의사결정":"셀 선별·점검 우선순위","활용":"초기 신호로 상대 수명 비교","추가 조건":"같은 화학계·현장 운전 조건 검증"},
        {"의사결정":"교체 예산·정비 계획","활용":"총수명과 과대예측 위험 참고","추가 조건":"셀별 예측 구간·보수적 기준·실제 SOH"},
        {"의사결정":"충전 운영 검토","활용":"정책별 오차·신규 정책 감시","추가 조건":"관측 상관을 최적 C-rate의 인과 근거로 쓰지 않음"},
    ])
    reproducibility_text=f"""### 재현 근거

원본 파일 SHA-256·크기·고정 모델 해시·실행 패키지는 `provenance.json`에 있습니다. 학습 환경 Python {frozen['python']}, 추가 검증 환경 Python {provenance['diagnostic_python']}에서 같은 scikit-learn {frozen['sklearn']} 모델의 예측을 검산했습니다.

```bash
python -m src.train --batch3       # 원본부터 전체 파이프라인
python -m src.diagnostics --batch3 # 기존 고정 모델의 추가 검증
python -m src.verify_results      # 결과·누수·코드 셀 검증
```

`--batch3`를 생략하면 필수 Batch 1/2만 수행합니다. 추가 모델 선택 없이 고정 모델을 평가합니다. 원본·모델 해시 일치도 검사합니다.
"""
    path=ROOT/"README.md"
    # Show key evidence in README; retain full calculations in the notebook.
    table=read("model_performance")
    additional_reporting=("\n\n### 추가 평가: Batch 3 (과제 확장 양식)\n\n"+markdown_table(formatted_performance)+
        "\n\nGap (Train-Valid)은 Valid − Train으로 계산합니다. Batch2-Batch3 Gap은 Batch 3 − Batch 2, 마지막 Target Gap은 Batch 3 − 9.1입니다. 단위는 %p이며 표의 두 Target Gap은 각각 Batch 2·Batch 3 결과에 해당합니다.\n\n"+
        f"Batch 3 MAPE {all_scores[all_scores['index']=='Test (Batch 3)'].mape_pct.iloc[0]:.3f}%는 Batch 2보다 {abs(all_scores[all_scores['index']=='Test (Batch 3)'].mape_pct.iloc[0]-model_mape):.3f}%p 낮습니다. 이번 결과는 모든 외부 배치가 동일하게 어렵다는 가정과 맞지 않으며, 짧은 수명에 집중된 Batch 2에서 모델이 더 취약함을 보여줍니다. 구조·정책·수명 분포가 함께 달라 원인을 하나로 단정하지 않습니다. 추가 양식의 계산 근거는 `model_performance_with_batch3_formatted.csv`에 저장합니다.") if has3 else ''
    ci2=intervals[intervals.partition=="Test (Batch 2)"].iloc[0]
    chosen_features=", ".join(frozen["features"])
    feature_table=read('feature_definitions').rename(columns={'feature':'피처','window':'관측 구간','unit':'단위','selected':'최종 선택'})
    feature_table['최종 선택']=feature_table['최종 선택'].map({True:'선택',False:'후보 비교'})
    baseline_table=baseline[['model','n_cells','mape_pct','mae_cycles']].rename(columns={'model':'모델','n_cells':'평가 셀 수','mape_pct':'MAPE (%)','mae_cycles':'MAE (회)'})
    baseline_table['모델']=['Batch 1 중앙값 고정 예측','고정 Ridge 모델']
    top_error_table=read('error_analysis').head(5)[['cell_id','cycle_life','predicted_cycle_life','signed_error_cycles','ape_pct']].rename(columns={'cell_id':'셀 ID','cycle_life':'실제 수명','predicted_cycle_life':'예측 수명','signed_error_cycles':'예측−실제 (회)','ape_pct':'오차율 (%)'})
    detailed_eda=readme_section(distribution, core_vif, full_vif)
    concise=f"""# ESS 배터리 수명 예측

초기 100사이클 데이터로 총 Cycle Life를 예측하고 **Batch 1 학습 → Batch 2 평가**로 배치 간 일반화를 확인합니다. EDA 근거와 모델 구현, 성능·오류·ESS 해석을 연결합니다.

## 프로젝트 개요

- 데이터셋: MIT–Stanford Battery Dataset 계열 (Severson et al., 2019)
- 태스크: **Regression**; `log10(cycle_life)` 학습 후 사이클 단위 복원
- 학습: Batch 1 (`2017-05-12`), 46 → 36셀; 개발 29 / 정책 Hold-out 7
- 평가: Batch 2 (`2018-02-20`), 47 → 39셀{'; 추가 Batch 3 (`2018-04-12`), 46 → 44셀' if has3 else ''}
- 최종 모델: **{cfg['model']}**, 피처 `{chosen_features}`, `alpha={cfg.get('alpha','—')}`
- [상세 분석·직접 실행 노트북](DAY2/03_modeling.ipynb)

## 파일 구조

```text
├── archive/                    # 원본 MAT (Git 제외)
├── data/README.md              # 데이터·제외 기준
├── DAY1/                       # 기존 모델 전략 보고서
├── DAY2/
│   ├── 03_modeling.ipynb
│   └── results/                # 성능·CV·피처·audit·오류·그래프·모델
├── src/
│   ├── preprocess.py
│   ├── features.py
│   ├── train.py
│   ├── diagnostics.py
│   ├── eda_reporting.py        # 수명 그룹·knee·ΔQ·정책 EDA
│   ├── report.py / evidence.py
│   └── verify_results.py
├── requirements.txt
└── README.md
```

## 환경 설정

Python 3.11 환경에서 프로젝트 루트 기준으로 실행합니다. 원본 MAT를 [데이터 안내](data/README.md)의 경로에 둡니다.

```bash
git clone https://github.com/chyj1022/DS-MINI-Develop-U104.git
cd DS-MINI-Develop-U104
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.train --batch3
python -m src.verify_results
```

필수 Batch 1/2만 실행하려면 `--batch3`를 생략합니다. 저장 모델의 추가 진단만 실행하려면 `python -m src.diagnostics --batch3`를 사용합니다. 노트북 커널은 패키지를 설치한 환경을 선택합니다. 실행 Python {provenance['diagnostic_python']}, scikit-learn {frozen['sklearn']}; 원본 SHA-256·파일 크기·환경·고정 모델 해시는 `provenance.json`에 기록합니다.

대용량 원본 `archive/`, 가상환경 `.venv/`, 저장 모델 `*.joblib`는 Git 업로드에서 제외합니다. 공개된 CSV·그림·실행 결과는 바로 열람할 수 있고, 원본부터 재학습하거나 저장 모델을 재생성하려면 위의 MAT 파일과 환경이 필요합니다. 상세 분석의 재현 기준은 DAY 2 노트북이며 기존 통합 EDA는 선행 탐색 기록입니다.

{detailed_eda}
## Modeling

### 피처 엔지니어링 전략

ΔQ log 분산을 고정하고 초기 용량 기울기·충전 시간·IR 변화·평균 온도를 포함/제외한 **16개 부분집합**을 비교했습니다. 용량 기울기는 `0<Qd≤1.2Ah` 값으로 적합하며 결측 대치·표준화는 학습 fold 안에서만 수행합니다. 셀 ID·수명·기록 길이·최종 용량·후반 knee는 입력에서 제외합니다.

{markdown_table(feature_table)}

핵심 피처는 `log10 Var(Q100(V) − Q10(V))`입니다. 충전 시간은 2~6회의 유한한 양수 값 평균, 용량 기울기는 2~100회 유효 용량에 대한 선형 적합, IR 변화는 양수인 100회 값−2회 값, 온도는 2~100회 유한 값 평균으로 구현했습니다. 전압축의 단조성·2.0~3.5V 범위·1,000개 공통 점을 확인합니다. 최종 입력과 품질 진단용 플래그는 구분합니다.

### 개발 파이프라인과 누수 방지

| 단계 | 수행 내용 | 확인할 파일 |
| --- | --- | --- |
| 원본·품질 검사 | 불완전·결측 타깃 제외 및 셀별 이유 기록 | `src/preprocess.py`, `cell_audit_all.csv` |
| 초기 피처 추출 | 100사이클 이내 정보만 사용 | `src/features.py`, `feature_definitions.csv` |
| 정책 Hold-out | Batch 1 개발 29셀·16정책 / 검증 7셀·4정책 분리 | `split_manifest.csv` |
| Nested-CV | 외부 4-fold 검증, 각 외부 학습 fold 안에서 내부 4-fold 후보 선택 | `nested_cv_folds.csv`, `cv_fold_manifest.csv` |
| 최종 후보 선택 | 개발 29셀의 정책별 CV MAPE로 설정 선택 | `cv_results.csv`, `frozen_config.json` |
| Hold-out 평가 | 개발 29셀에만 적합한 모델로 검증 7셀 평가 | `metrics.csv` |
| 외부 배치 평가 | 설정 고정 후 Batch 1 전체 36셀 재학습, Batch 2·3 평가 | `predictions_all.csv`, `metrics_all.csv` |

셀 단위 분리만으로 동일 정책의 중복이 자동으로 차단되지는 않으므로, **동일 충전 정책이 개발·Hold-out 및 CV train·valid를 가로지르지 않도록** 분리합니다. 결측 중앙값과 표준화 통계도 각 학습 fold에서만 적합합니다. Hold-out 및 외부 배치 점수는 후보 선택에 사용하지 않았습니다. 선행 EDA에서 평가 배치를 확인했다는 한계는 별도로 보고합니다.

### 모델 선택 및 근거

중앙값·단일 피처 Linear·Ridge·Elastic Net·제한된 RBF SVR/Random Forest, 원 단위 타깃 대안을 포함한 **212조합**을 개발 29셀의 정책별 4-fold CV에서 비교했습니다. {cfg['model']}가 최저 CV({scores.iloc[0].cv_mean_mape_pct:.5f}%)로 선택됐지만 단일 ΔQ 피처 후보({best_core.cv_mean_mape_pct:.5f}%)와 차이는 **{effect:.5f}%p**로 작습니다. 충전 시간의 실질적 개선을 주장하지 않으며 다음 개발 실험에서 선택 근거를 재검토하고 단순 모델을 우선 비교합니다. 현재 시험 결과로 모델을 바꾸지 않았습니다.

과제의 Train은 **개발 Nested-CV**(nested 4×4 GroupKFold)의 외부 fold 검증오차 평균이며 학습 오차가 아닙니다. Valid는 별도 7셀 Hold-out입니다. 설정 고정 후 전체 Batch 1 **36셀**로 재학습해 Batch 2{'·3' if has3 else ''}를 평가했습니다. 같은 정책이 개발·검증에 걸치지 않습니다. 피처 추가 효과는 `paired_ablation.csv`에서 같은 alpha·같은 fold로 비교합니다.

### 후보 모델 비교

![Batch 1 개발 정책별 CV의 모델군별 최저 후보 비교](DAY2/results/01_model_comparison.png)

각 막대는 해당 모델군에서 선택된 최저 후보의 개발 CV MAPE이며 오차 막대는 **fold 간 표준편차**입니다. 성능 리포팅의 Nested-CV와는 별도입니다. Ridge·Elastic Net·Linear의 성능이 비슷해 작은 점수 차이를 우월성의 강한 근거로 보지 않습니다. `plus_2`는 ΔQ log 분산과 초기 충전 시간 조합입니다.

## 성능 결과

### 필수 평가: Regression (Batch 1 → Batch 2)

{markdown_table(table)}

행 이름을 유지하되 **Gap (Train-Valid)은 Valid − Train으로 계산**합니다. Gap (Valid-Test)은 Test − Valid, Gap (Target-Test)은 Test − 9.1, Gap (Batch2-Batch3)은 Batch 3 − Batch 2이며 단위는 **%p**입니다.

MAPE=`100 × mean(|실제−예측|/실제)`. 필수 6행은 `model_performance.csv`, 추가 배치 계산값은 `model_performance_with_batch3.csv`에 있습니다. 개발 Nested-CV 외부 fold는 **{cv_range}%**로 변하고 Hold-out은 7셀뿐이므로, −0.407%p 차이를 안정성의 강한 증거로 삼기는 어렵습니다.

{additional_reporting}

같은 Batch 2 39셀에서 **Batch 1 중앙값 {median_prediction:.1f}회 기준선 MAPE {baseline_mape:.2f}% → 모델 {model_mape:.2f}%**로 상대 오차가 **{reduction:.2f}% 감소**했습니다. 예측 신호는 있지만 정확도에는 한계가 있습니다. [기준선 비교](DAY2/results/batch2_baseline_comparison.csv)는 학습 Batch 1로만 중앙값을 정하며 모델 선택에는 사용하지 않습니다.

{markdown_table(baseline_table)}

상대 감소율은 `100 × (1 − 모델 MAPE / 기준선 MAPE)`입니다. 논문과의 비교에 더해, 같은 외부 셀에서의 실질적 개선을 보여주는 단순 기준선 비교입니다.

Batch 2는 Target에 미달했습니다. 정책 단위 bootstrap(2,000회·seed=42)의 고정 모델 MAPE 95% 구간은 **{ci2.ci_low_pct:.2f}~{ci2.ci_high_pct:.2f}%**입니다. 재학습·미래 배치의 성능 보증이나 셀별 예측 구간은 아닙니다.

원논문은 `2017-05-12`·`2017-06-30` 혼합 분할이고 과제는 `2018-02-20` 외부 배치 평가여서 **Target Gap은 참고 비교**입니다. 선행 EDA가 세 배치를 봤다는 한계가 있으며, 모델 선택·계수 추정은 Batch 1에 제한했습니다. Valid 모델 29셀·Test 모델 36셀의 학습 규모 차이도 Gap에 포함됩니다. [저자 분할 코드](https://github.com/rdbraatz/data-driven-prediction-of-battery-cycle-life-before-capacity-degradation/blob/master/LoadData.m).

## 오류 분석

### 실제값·예측값과 Batch 2 잔차

![실제 수명 대비 예측 수명 및 Batch 2의 부호 오차](DAY2/results/02_predictions_residuals.png)

왼쪽의 대각선은 실제값과 예측값이 같은 기준선입니다. Train 점은 개발 Nested-CV의 외부 fold 예측으로 학습 적합값이 아닙니다. 오른쪽에서 잔차(예측−실제)가 0보다 크면 과대예측입니다. 짧은 수명의 Batch 2 셀에서 양의 잔차가 반복되어 **일부 큰 오차보다 넓게 나타나는 과대예측 편향**을 확인할 수 있습니다.

- Batch 2 **{int(all_bias.n_cells)}셀 중 {int(all_bias.overpredicted_cells)}셀**을 과대예측했습니다. 평균 부호 오차(예측−실제)는 **{all_bias.mean_signed_error_cycles:+.1f}회**, 오차율 중앙값은 **{all_bias.median_ape_pct:.2f}%**로 일부 이상치만의 문제가 아닙니다.
- Batch 1 최소 수명보다 짧은 **{int(below_bias.n_cells)}셀 모두 과대예측**됐고 MAE와 평균 부호 오차는 {below_bias.mae_cycles:.1f}회입니다. `standard_structure`와 `below_batch1_life_min`은 정확히 같은 30셀로, 두 결과는 독립적인 원인 증거가 아닙니다. 구조와 수명 범위의 효과를 분리해 주장하지 않습니다.
- 입력 범위 안 {int(inside.n_cells)}셀의 MAPE {inside.mape_pct:.2f}%가 범위 밖 {int(outside.n_cells)}셀 {outside.mape_pct:.2f}%보다 높습니다. **입력 범위 이탈만으로 성능 저하를 설명할 수 없으며** 피처–수명 관계·정책·배치 효과를 함께 봐야 합니다.
- {f'Batch 3 전체 44셀 {quality.iloc[0].mape_pct:.2f}%, 저자 품질 규칙을 적용한 40셀 {quality.iloc[1].mape_pct:.2f}%를 함께 보고했습니다. ' if has3 else ''}개선은 단수명 개발 표본 확보·현장 조건 검증을 우선하며 변경 모델은 새 미사용 배치로 평가해야 합니다.

셀별 근거: `error_analysis.csv`, `prediction_bias.csv`, `subgroup_overlap.csv`, `input_domain_audit.csv`, `domain_metrics.csv`. 표본 수만으로 원인을 설명하기보다 학습 수명 범위·배치 분포·피처–수명 관계를 우선 점검합니다. 추가 Batch 3의 곡선 정렬·원점 불변성·품질 민감도와 반례 해석은 [노트북](DAY2/03_modeling.ipynb)에 있습니다.

### 오차율 상위 5셀

{markdown_table(top_error_table)}

다섯 셀은 모두 실제 수명 500회 미만이며 과대예측됐습니다. 전체 39셀의 부호 오차와 함께 보아야 하며, 이 셀들을 제거해서 성능을 개선하지 않았습니다. 다음 실험은 단수명 개발 표본 확보, 단일 ΔQ 피처와 충전 시간 조합 재비교, 신규 배치 검증을 우선합니다.

## ESS 도메인 해석

셀 선별·점검 우선순위·교체 예산의 보조 지표로 활용할 수 있습니다. 짧은 수명 과대예측은 교체 지연 위험으로 이어집니다. 실험실 단일 셀·36셀 학습·특정 충전 조건에 한정돼 현장 달력 열화·온도·부분 충방전·셀 불균형, 제조 배치와 셀별 예측 구간의 추가 검증이 필요합니다.

동일 배치에서는 약 10% 검증 오차를 얻었지만, 단수명 셀이 많은 외부 Batch 2에서 과대예측 편향과 일반화 한계를 확인했습니다. 현재 모델은 교체 시점의 단독 결정 근거로 사용하기 어렵습니다.

{markdown_table(ess_table)}

예측 대상은 **총 Cycle Life**로 현재 잔여 수명(RUL)이나 안전 사고 확률이 아닙니다. 실 배포를 위해 현장 라벨 확보와 제조 배치·운전 조건별 검증, 셀별 예측 구간, 신규 정책 및 입력 분포의 감시 기준이 필요합니다. 관측된 C-rate 상관을 최적 충전 정책의 인과 근거로 사용하지 않습니다.

## 평가항목별 확인 위치

{markdown_table(assessment[['평가항목','배점','직접 근거','산출물']])}

배점은 과제 기준을 옮긴 것으로 자체 채점 점수가 아닙니다. 모델 전략 수립 100점·모델 개발 및 평가 100점의 각 근거를 README와 노트북, 결과 CSV에서 확인할 수 있습니다.

## 참고문헌

- Severson et al. (2019). Data-driven prediction of battery cycle life before capacity degradation. *Nature Energy*, 4, 383–391. [원논문](https://www.nature.com/articles/s41560-019-0356-8).

## 팀 구성

- 최유정(울산 3반 U104), **개인 수행**: EDA, 모델 전략 수립, 피처 엔지니어링, 파이프라인·모델 개발, Batch 2·3 성능 평가, 오류 분석 및 보고서 작성 전 과정.

평가항목별 근거는 노트북과 `assessment_evidence.csv`에서 확인할 수 있습니다. 자동 검증은 원본 피처·미래 정보 불변성·정책 분리·학습 전용 전처리·수치·모델·노트북 실행을 확인합니다.
"""
    path.write_text(concise)

    nb=nbformat.read(ROOT/"DAY2/03_modeling.ipynb",as_version=4)
    environment={}
    setup="from pathlib import Path\nimport sys, json\nimport numpy as np\nimport pandas as pd\nROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / 'src/train.py').exists() and (p / 'archive').exists())\nif str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))\nOUT = ROOT / 'DAY2/results'\nprint('결과 경로:', OUT)"
    with contextlib.redirect_stdout(io.StringIO()):
        exec(setup,environment)
    def code(source):
        tree=ast.parse(source);last=tree.body[-1];stream=io.StringIO();result=None
        with contextlib.redirect_stdout(stream):
            if isinstance(last,ast.Expr):
                exec(compile(ast.Module(body=tree.body[:-1],type_ignores=[]),"<notebook>","exec"),environment)
                result=eval(compile(ast.Expression(last.value),"<notebook>","eval"),environment)
            else:exec(source,environment)
        cell=nbformat.v4.new_code_cell(source)
        if stream.getvalue():cell.outputs.append(nbformat.v4.new_output("stream",name="stdout",text=stream.getvalue()))
        if result is not None:
            data={"text/plain":str(result)}
            if isinstance(result,pd.DataFrame):data["text/html"]=result.to_html(index=False,float_format=lambda v:f"{v:.3f}")
            cell.outputs.append(nbformat.v4.new_output("execute_result",execution_count=1,data=data,metadata={}))
        return cell
    def md(text):return nbformat.v4.new_markdown_cell(text)
    def picture(filename):
        source=f"import matplotlib.pyplot as plt\nplt.figure(figsize=(14,6))\nplt.imshow(plt.imread(OUT / '{filename}'))\nplt.axis('off')\nplt.show()"
        cell=nbformat.v4.new_code_cell(source)
        cell.outputs=[nbformat.v4.new_output("display_data",data={"image/png":base64.b64encode((OUT/filename).read_bytes()).decode()},metadata={})]
        return cell
    def insert_before(title,added):
        index=next(i for i,c in enumerate(nb.cells) if c.cell_type=="markdown" and c.source.startswith(title))
        nb.cells[index:index]=added
    nb.cells[1]=code(setup)
    eda_parts=re.split(r"(?=### [1-5]\. )", detailed_eda.replace("## EDA\n", "", 1))
    def eda_markdown(part):
        cleaned=re.sub(r"^!\[.*?\]\([^)]+\)\n", "", part, flags=re.MULTILINE)
        cleaned=cleaned.replace("](DAY2/results/", "](results/").replace("](DAY2/03_modeling.ipynb)", "](03_modeling.ipynb)").replace("](30-ESSHealth-scratch.ipynb)", "](../30-ESSHealth-scratch.ipynb)")
        return md(cleaned)
    eda_cells=[eda_markdown(eda_parts[0]), eda_markdown(eda_parts[1]),
        code("pd.read_csv(OUT / 'eda_distribution.csv')"),
        code("pd.read_csv(OUT / 'life_group_proportions.csv')"),picture("03_batch_shift.png"),
        eda_markdown(eda_parts[2]),picture("07_life_groups_degradation.png"),
        code("pd.read_csv(OUT / 'life_group_degradation.csv')"),
        code("knee = pd.read_csv(OUT / 'knee_timing_summary.csv')\nknee[knee.scope=='Pooled (descriptive)']"),
        eda_markdown(eda_parts[3]),picture("04_eda_curve_evidence.png"),picture("08_dq_life_groups.png"),
        code("pd.read_csv(OUT / 'life_group_statistics.csv')"),
        eda_markdown(eda_parts[4]),picture("09_charging_policy_life.png"),
        code("policy = pd.read_csv(OUT / 'policy_summary.csv')\npolicy[policy.batch==1].sort_values('mean_life',ascending=False)"),
        code("pd.read_csv(OUT / 'charging_rate_correlations.csv')"),
        eda_markdown(eda_parts[5]),picture("05_feature_stability.png"),
        code("correlations = pd.read_csv(OUT / 'eda_correlations.csv')\ncorrelations.pivot(index='feature',columns='batch',values='spearman_life').reset_index()"),
        code("vifs = pd.read_csv(OUT / 'eda_vif.csv')\nvifs.groupby(['scope','feature_set']).agg(n_cells=('n_cells','first'),max_vif=('vif','max')).reset_index()")]
    insert_before("## 2.",eda_cells)
    source_cell=code("from src.preprocess import load_batch, development_split\nfrom src.features import ALL_FEATURES\nfrom src.train import fit_model, predict, metrics\nfrozen = json.loads((OUT / 'frozen_config.json').read_text())\nconfig = frozen['selected_config']\nbatch1, audit1 = load_batch(1)\ndev, valid = development_split(batch1)\nassert not set(dev.policy) & set(valid.policy)\nstored = pd.read_csv(OUT / 'cell_features.csv')\nexpected = stored[stored.batch==1].set_index('cell_id').loc[batch1.cell_id]\nnp.testing.assert_allclose(batch1[ALL_FEATURES],expected[ALL_FEATURES],equal_nan=True)\nprint('원본','추출:',len(batch1),'셀, 개발',len(dev),'검증',len(valid))\nbatch1[['cell_id','policy',*frozen['features']]].head()")
    insert_before("## 3.",[md("### 원본에서 피처·분할을 직접 재현\n\n이 셀은 저장 표를 읽는 것에 더해 원본 MAT에서 초기 피처를 다시 추출하고 정책 분리와 계산값 일치를 확인합니다."),source_cell])
    pipeline_cell=code("# 고정 설정으로 직접 재학습: 개발 29셀 → Hold-out, 전체 36셀 → Batch 2\ndevelopment_model = fit_model(dev,config)\nvalid_prediction = predict(development_model,valid,config)\nfinal_model = fit_model(batch1,config)\nbatch2, audit2 = load_batch(2)\ntest_prediction = predict(final_model,batch2,config)\nsaved = pd.read_csv(OUT / 'predictions.csv')\nexpected_test = saved[saved.batch==2].set_index('cell_id').loc[batch2.cell_id]\nnp.testing.assert_allclose(test_prediction,expected_test.predicted_cycle_life)\npd.DataFrame([{'구분':'Valid',**metrics(valid.cycle_life,valid_prediction)},{'구분':'Test Batch 2',**metrics(batch2.cycle_life,test_prediction)}])")
    insert_before("## 4.",[md(selection_text),code("paired = pd.read_csv(OUT / 'paired_ablation.csv')\npaired.loc[(paired.base_set=='variance') & (paired.alpha==0.1)]"),md("### 고정 모델 학습·평가를 직접 재현\n\n아래 코드는 개발 29셀 모델과 Batch 1 전체 36셀 모델을 구분해 적합하고 그 뒤 Batch 2를 읽습니다. 설정을 새로 선택하지 않습니다. 전체 후보·nested CV 재현은 `python -m src.train --batch3`로 실행합니다."),pipeline_cell])
    insert_before("## 5.",[md(baseline_text),code("pd.read_csv(OUT / 'batch2_baseline_comparison.csv')"),md(uncertainty_text),code("pd.read_csv(OUT / 'metric_uncertainty.csv')"),picture("06_uncertainty_and_domain.png")])
    extra=[md(bias_text),code("pd.read_csv(OUT / 'prediction_bias.csv')"),code("pd.read_csv(OUT / 'subgroup_overlap.csv')"),md(domain_text.replace("![오차 불확실성과 수명 범위](DAY2/results/06_uncertainty_and_domain.png)","")),code("pd.read_csv(OUT / 'domain_metrics.csv')")]
    if has3:
        extra.extend([md(b3_text),code("batch3, audit3 = load_batch(3)\nprediction3 = predict(final_model,batch3,config)\nexpected3 = pd.read_csv(OUT / 'predictions_all.csv')\nexpected3 = expected3[expected3.batch==3].set_index('cell_id').loc[batch3.cell_id]\nnp.testing.assert_allclose(prediction3,expected3.predicted_cycle_life)\npd.DataFrame([{'n_cells':len(batch3),**metrics(batch3.cycle_life,prediction3)}])"),code("pd.read_csv(OUT / 'model_performance_with_batch3_formatted.csv').fillna('')"),code("eda = pd.read_csv(OUT / 'eda_cell_features.csv')\nprint('곡선 원점 이동 후 분산 차이 최대:',eda.delta_variance_origin_invariance_error.max())\nassert eda.delta_variance_origin_invariance_error.max() < 1e-12\npd.read_csv(OUT / 'batch3_quality_sensitivity.csv')")])
    insert_before("## 6.",extra)
    insert_before("## 7.",[md(markdown_table(ess_table)+"\n\n개선 우선순위: 단수명 개발 데이터 확보 → 새로운 배치·정책 검증 → 단순 기준선과 보조 피처 재비교 → 셀별 예측 구간·감시 기준. 현재 평가 데이터를 본 후의 개선 모델은 새로운 미사용 배치로 검증해야 합니다.")])
    nb.cells.extend([md("## 계산 근거·평가항목·재현성\n\n"+markdown_table(assessment)+"\n\n이 표는 자체 점수가 아니라 평가자가 확인할 계산 근거의 위치입니다.\n\n"+reproducibility_text),code("provenance = json.loads((OUT / 'provenance.json').read_text())\nprint('고정 모델 SHA-256:',provenance['model_sha256'])\nprint('실행 패키지:',provenance['packages'])\npd.DataFrame(provenance['sources'])")])
    count=0
    for cell in nb.cells:
        if cell.cell_type=="code":
            count+=1;cell.execution_count=count
            for output in cell.outputs:
                if output.output_type=="execute_result":output.execution_count=count
    nb.metadata.language_info.version=provenance["diagnostic_python"]
    nbformat.validate(nb);nbformat.write(nb,ROOT/"DAY2/03_modeling.ipynb")
