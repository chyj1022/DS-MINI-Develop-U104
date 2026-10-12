"""Report the DAY 1 feature selection and frozen predictions, before/after grouping."""
import base64
import contextlib
import io
import json
import re
import nbformat
import numpy as np
import pandas as pd
from .preprocess import ROOT
from .report import markdown_table as mt, batch3_reporting_format
from .paper_comparison import generate, PARTITIONS, life_labels, classification_metrics
from .diagnostics import vif
from .label_audit import audit_batch2_labels

OUT = ROOT / 'DAY2/results'


def read(name):
    return pd.read_csv(OUT / (name + '.csv'))


def build():
    tables = generate()
    _, audit = audit_batch2_labels()
    frozen = json.loads((OUT / 'frozen_config.json').read_text())
    features = read('cell_features_all')
    predictions = read('predictions_all')
    scores = read('metrics_all')
    cv = read('cv_results')
    has3 = 3 in features.batch.unique()
    b1 = features[features.batch == 1]
    selected_vif = pd.DataFrame(vif(b1, frozen['features']))
    selected_vif.to_csv(OUT / 'selected_feature_vif.csv', index=False, encoding='utf-8-sig')
    summary = scores[['index', 'n_cells', 'mape_pct', 'mae_cycles', 'rmse_cycles']].rename(columns={
        'index': '평가', 'n_cells': '셀 수', 'mape_pct': 'MAPE (%)', 'mae_cycles': 'MAE (회)', 'rmse_cycles': 'RMSE (회)'})
    family = cv.groupby('model', sort=False).head(1)[['model','feature_set','cv_mean_mape_pct','cv_std_mape_pct']].rename(columns={
        'model':'후보 모델','feature_set':'선택 피처 세트','cv_mean_mape_pct':'개발 선택 CV MAPE (%)','cv_std_mape_pct':'fold 표준편차 (%p)'})
    ablation = read('feature_ablation')
    best = float(cv.iloc[0].cv_mean_mape_pct)
    single = float(ablation[ablation.feature_set == 'variance'].cv_mean_mape_pct.iloc[0])
    vifs = read('eda_vif')
    full_vif = vifs[(vifs.scope == 'Batch 1') & (vifs.feature_set == 'all10')].vif.max()
    core_vif = vifs[(vifs.scope == 'Batch 1') & (vifs.feature_set == 'representative5')].vif.max()
    from .eda_reporting import readme_section
    eda = readme_section(read('eda_distribution'), core_vif, full_vif)
    eda = eda.replace('추가 3분류 라벨','예측 수명의 3구간 평가').replace('추가 개발·비교 절','구간화 후 비교 절')
    eda = eda.replace('### 5. 추가 확인: 피처 중복·전이 안정성과 품질','### 5. 초기 피처와 수명의 상관·다중공선성')
    mandatory = read('model_performance')
    extended = read('model_performance_with_batch3')
    if has3:
        extended.loc[extended.index[-1], '구분'] = 'Gap (Target-Test, Batch 3)'
        extended.to_csv(OUT / 'model_performance_with_batch3.csv', index=False, encoding='utf-8-sig')
        batch3_reporting_format(extended).to_csv(OUT / 'model_performance_with_batch3_formatted.csv', index=False, encoding='utf-8-sig')
    paper = tables['paper_regression_metrics'][['model','primary_mape_pct','primary_excluded_mape_pct','secondary_mape_pct']].rename(columns={
        'model':'논문 모델','primary_mape_pct':'Primary MAPE (%)','primary_excluded_mape_pct':'Primary 이상 셀 제외 후 (%)','secondary_mape_pct':'Secondary MAPE (%)'})
    grouped = tables['life_class_metrics']
    overview = grouped[['scheme','partition','macro_f1','accuracy_pct']].copy()
    overview.scheme = overview.scheme.map({'binary_550':'2구간 (550회)','three_500_1000':'3구간 (500·1,000회)'})
    overview.columns = ['구간','평가','Macro-F1','Accuracy (%)']
    group_sections = []
    for scheme, title in [('binary_550','2구간: 단수명 ≤550회 / 장수명 >550회'),
                          ('three_500_1000','3구간: <500회 / 500–1,000회 / >1,000회')]:
        perf = tables[scheme + '_performance']
        perf.iloc[:6].to_csv(OUT / (scheme + '_performance.csv'), index=False, encoding='utf-8-sig')
        extra = ''
        if has3:
            perf.to_csv(OUT / (scheme + '_with_batch3.csv'), index=False, encoding='utf-8-sig')
            formatted = perf.copy()
            formatted.insert(1, '비교 항목', np.where(formatted['구분'].str.startswith('Gap'), formatted['구분'], ''))
            formatted.loc[formatted['구분'].str.startswith('Gap'), '구분'] = ''
            formatted.to_csv(OUT / (scheme + '_with_batch3_formatted.csv'), index=False, encoding='utf-8-sig')
            extra = '\n\n<details>\n<summary>Batch 3 추가 평가 — 지정 9행</summary>\n\n' + mt(formatted) + '\n\n</details>'
        group_sections.append('### ' + title + '\n\n' + mt(perf.iloc[:6]) + extra)
    recalls = tables['life_class_per_class']
    short = recalls[recalls.partition.isin(PARTITIONS[2:]) & recalls.life_group.str.startswith('short')][
        ['scheme','partition','n_actual','n_correct','recall_pct']].rename(columns={
            'scheme':'구간','partition':'평가','n_actual':'실제 단수명','n_correct':'맞힌 단수명','recall_pct':'단수명 Recall (%)'})
    short['구간'] = short['구간'].map({'binary_550':'2구간 ≤550','three_500_1000':'3구간 <500'})
    secondary_rows = []
    if has3:
        reference = tables['paper_classification_metrics']
        r = reference[(reference.model == 'Full classifier') & (reference.partition == 'Secondary test')].iloc[0]
        secondary_rows.append({'평가':'논문 Full Secondary (초기 5회 분류)','셀 수':int(r.n_cells),'Accuracy (%)':r.accuracy_pct,
                               'Macro-F1':r.macro_f1_reconstructed,'단수명 Recall (%)':r.short_recall_pct})
        p3 = predictions[predictions.batch == 3]
        for label, p in [('이번 Batch 3 전체 (100회 회귀 구간화)',p3), ('이번 Batch 3 품질 규칙 (100회 회귀 구간화)',p3[~p3.cell_id.isin([2,37,42,43])])]:
            y, yp = life_labels(p.cycle_life,'binary_550'), life_labels(p.predicted_cycle_life,'binary_550')
            m = classification_metrics(y,yp,'binary_550')
            secondary_rows.append({'평가':label,'셀 수':len(p),'Accuracy (%)':m['accuracy_pct'],'Macro-F1':m['macro_f1'],
                                   '단수명 Recall (%)':100*float(((y==0)&(yp==0)).sum())/int((y==0).sum())})
    secondary = pd.DataFrame(secondary_rows)
    secondary.to_csv(OUT / 'binary_secondary_reference.csv', index=False, encoding='utf-8-sig')
    top = read('error_analysis').sort_values('ape_pct',ascending=False).head(5)[['cell_id','policy','cycle_life','predicted_cycle_life','ape_pct']]
    assessment = pd.DataFrame([
        ['EDA',50,'분포·열화·ΔQ·충전 정책·상관·VIF','EDA 절 / eda_* / DAY1 보고서'],
        ['EDA → 전략 연결성',30,'관측→피처·후보 모델→검증 연결','EDA→전략 표 / feature_ablation.csv'],
        ['모델링 전략 수립',20,'회귀 타깃·5개 후보·선택 규칙','features.py / cv_results.csv'],
        ['전략 → 구현 반영',20,'계획서 피처에서 선택한 Ridge','train.py / frozen_config.json / model_coefficients.csv'],
        ['Pipeline 개발',40,'정책 분할·fold 내부 전처리·고정 외부 평가','preprocess.py / train.py / provenance.json'],
        ['성능 리포팅 및 해석',20,'회귀 지정 양식·Target Gap·2/3구간 참고 비교','model_performance*.csv / life_class_*.csv'],
        ['분석 결과 해석',20,'오류 셀·배치 차이·ESS 활용과 한계','error_analysis.csv / subgroup_metrics.csv / 오류 분석 절'],
    ],columns=['평가항목','배점','직접 근거','산출물'])
    assessment.to_csv(OUT / 'assessment_evidence.csv',index=False,encoding='utf-8-sig')
    intro = f'''# ESS 배터리 수명 예측

초기 100사이클에서 **계획서의 피처 후보를 계산·선택한 회귀 모델**로 총 Cycle Life를 예측합니다. 같은 예측 수명을 2구간·3구간으로 나누어 결과를 각각 평가하고 논문 보고값과 비교합니다. 논문은 성능 비교의 기준으로 사용합니다.

## 프로젝트 개요

- 주 태스크: Regression — log10(cycle_life) 학습 후 사이클 단위 복원
- 데이터: MIT–Stanford Battery Dataset 계열; 학습 Batch 1 (2017-05-12), 필수 평가 Batch 2 (2018-02-20), 추가 평가 Batch 3 (2018-04-12)
- 학습 36셀: 개발 29셀·16정책 / Hold-out 7셀·4정책
- 최종 모델: **Ridge, alpha=0.1**; 선택 피처 **ΔQ log 분산 + 초기 충전 시간**
- 추가 평가: 같은 회귀 예측의 2구간(550회)·3구간(500·1,000회) 구간화. 별도 분류기 학습 결과와 구분합니다.
- [DAY 1 계획서](DAY1/DS-MINI-Design-울산_3반-최유정.pdf) · [실행 결과와 직접 재현 노트북](DAY2/03_modeling.ipynb)

### 핵심 결과 — 구간화 전 회귀

{mt(summary)}

Batch 2 MAPE는 **28.609%**, 과제 Target 9.1%와의 차이는 **+19.509%p**입니다. 단수명 범위에서 과대예측이 반복됐으며, 구간화 후의 점수도 아래에서 별도로 보고합니다.

## 파일 구조

```text
├── archive/                 # 원본 MAT; Git 제외
├── data/README.md           # 데이터·라벨·제외 기준
├── DAY1/                    # 모델 전략 보고서
├── DAY2/03_modeling.ipynb
├── DAY2/results/            # 피처·CV·예측·지표·그림·검증 근거
├── src/
│   ├── preprocess.py / features.py / train.py
│   ├── diagnostics.py / eda_reporting.py
│   ├── paper_comparison.py  # 저장된 회귀 예측 구간화·논문 수치 비교
│   ├── label_audit.py       # Batch 2 라벨 검증
│   ├── report.py / plan_report.py
│   └── verify_results.py / verify_plan_results.py
└── requirements.txt
```

## 환경 설정

Python 3.11, scikit-learn 1.6.1 기준입니다. 원본 MAT는 [데이터 안내](data/README.md)에 따라 archive/에 둡니다.

```bash
git clone https://github.com/chyj1022/DS-MINI-Develop-U104.git
cd DS-MINI-Develop-U104
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.train --batch3
python -m src.verify_results
```

필수 Batch 1·2만 재학습하려면 `--batch3`를 생략합니다. DAY 1의 세 배치 EDA는 [계획서](DAY1/DS-MINI-Design-울산_3반-최유정.pdf)와 [기존 노트북](30-ESSHealth-scratch.ipynb)에서 확인합니다. 저장된 결과로 문서·구간화 표만 갱신하려면 `python -m src.report`, 라벨 재검사는 `python -m src.label_audit --archive-dir archive`입니다. 원본 MAT와 모델 joblib는 Git 제외이며, 재학습하면 모델이 생성됩니다. 원본 해시·크기·환경은 `provenance.json`에 기록합니다.

### 타깃과 데이터 품질

논문의 총 Cycle Life는 명목 1.1Ah의 80%, **0.88Ah** 도달까지의 사이클 수입니다. 현재 학습 라벨은 MAT에 저장된 cycle_life입니다. Batch 1의 후속 측정 5셀과 조기 종료 5셀을 제외해 36셀, Batch 2는 결측 라벨 8셀을 제외해 39셀을 사용합니다. 학습 수명 범위는 534–1,074회이며, 장수명 후속 기록 제외로 범위가 좁아진 조건을 고려합니다. [제외 내역](DAY2/results/cell_audit_all.csv)

Batch 2 유효 {audit['n_evaluated']}셀은 원본 summary.cycle의 최초 Qd≤0.88Ah 도달 사이클과 저장 라벨을 대조했습니다. **정확 일치 {audit['n_exact_match']}셀 / 불일치 {audit['n_mismatch']}셀 / 미도달 {audit['n_not_reached']}셀**입니다. [셀별 검사](DAY2/results/batch2_cycle_life_audit.csv). 입력 관측 창은 100회까지이며 전체 수명 종료 용량·knee·기록 길이는 설명용으로만 사용합니다.

'''
    modeling = f'''## Modeling

### 피처 엔지니어링과 선택 근거

계획서의 대표 5개 후보에서 ΔQ log 분산을 고정하고 나머지 4개의 포함·제외를 조합한 **16개 부분집합**을 비교합니다.

| 후보 피처 | 의미·구간 | EDA 근거 | 최종 선택 |
| --- | --- | --- | --- |
| log10_dq_variance | log10 Var(Q100−Q10), ddof=0 | 배치별 수명과 일관된 음의 상관 | 선택 |
| qd_slope_2_100 | 유효 Qd의 2–100회 선형 기울기 | 초기 열화 속도, 배치별 관계 변화 | 제외 |
| charge_time_mean_2_6 | 2–6회 유한 양수 충전 시간 평균 | 충전 조건의 보조 신호; 전이 위험 점검 | 선택 |
| ir_change_100_2 | 양수 IR100−IR2 | 초기 저항 변화의 추가 설명력 검토 | 제외 |
| tavg_mean_2_100 | 2–100회 평균 온도 | 초기 열 환경의 추가 설명력 검토 | 제외 |

0<Qd≤1.2Ah 범위로 기울기를 계산하고 0 이하 IR은 결측 처리합니다. 결측 중앙값 대치·표준화는 학습 fold 안에서만 적합합니다. 상관된 ΔQ 요약값은 대표 변수로 줄였고, 최종 선택은 예측 성능으로 결정합니다. 최종 두 피처의 Batch 1 사후 VIF는 아래와 같습니다. 낮은 VIF는 통계적 독립성의 증명이 아니며 Ridge도 피처 상관을 제거하지 않습니다.

{mt(selected_vif[['feature','vif','n_cells']])}

상관계수는 방향과 단조 관계의 강도를 설명하며 인과 효과를 의미하지 않습니다. 상관 피처 사이에서 계수의 크기를 독립적인 영향으로 읽지 않습니다. [배치별 상관](DAY2/results/feature_batch_correlations.csv) · [부분집합 비교](DAY2/results/feature_ablation.csv)

### 후보 모델과 EDA 연결

| 후보 | 선정 이유 |
| --- | --- |
| 중앙값·단일 ΔQ Linear | 예측 신호와 복잡도 증가의 이득을 확인할 기준선 |
| Ridge·Elastic Net | 소표본·상관 피처에서 정규화로 계수 추정의 변동을 제어 |
| 제한된 RBF SVR | 초기 신호와 수명의 비선형 관계 가능성 비교 |
| 얕은 Random Forest | 피처 상호작용 가능성 비교; 깊이·최소 잎 표본 수 제한 |

36셀의 정형 피처 데이터이므로 정규화 모델의 검증·해석을 우선해 딥러닝은 후보에서 제외했습니다. 연속적인 교체·수명 계획 정보를 유지하고 학습 배치의 단수명 표본 부족을 고려해 **회귀를 주 태스크로 선택**했습니다.

### 모델 선택 기준과 검증 순서

1. 고정한 정책 Hold-out 7셀을 분리하고 개발 29셀만 후보 선택에 사용합니다.
2. **212개 후보 조합**을 정책별 4-fold CV의 원 사이클 단위 MAPE 평균으로 비교합니다. 회귀 적합은 주로 log 수명의 제곱오차와 정규화 목적함수를 사용하고, **후보 선택 지표는 MAPE**입니다.
3. 최저 개발 CV {best:.5f}%의 Ridge·두 피처·alpha=0.1을 선택합니다. 단일 ΔQ Ridge {single:.5f}%와 차이는 **{single-best:.5f}%p**여서 충전 시간의 추가 효과를 강하게 주장하지 않습니다. 외부 평가 후 후보를 재선택하지 않습니다.
4. Nested 4×4 GroupKFold에서 피처·모델·파라미터 선택 과정 전체를 평가합니다. 외부 fold마다 선택 모델이 달라질 수 있으므로 Train 10.842%는 최종 Ridge만의 고정 CV 점수가 아닙니다.
5. 개발 29셀로 Hold-out을 평가한 뒤 설정을 고정하고 Batch 1 전체 36셀로 재학습해 Batch 2·3를 평가합니다.

{mt(family)}

![후보 모델의 개발 CV 비교](DAY2/results/01_model_comparison.png)

위 표는 각 모델군의 최저 **선택용 CV**이며 아래 성능표의 Nested-CV와 다릅니다. [CV 후보](DAY2/results/cv_results.csv) · [외부 fold별 선택](DAY2/results/nested_cv_folds.csv) · [최종 계수](DAY2/results/model_coefficients.csv)

## 성능 결과 — 구간화 전 회귀

### 필수 평가: Batch 1 학습 → Batch 2

{mt(mandatory)}

**Gap 공통 규칙:** 과제 행 이름을 유지하며 (+)는 오차 증가 또는 점수 하락입니다. 회귀는 Valid−Train / Test−Valid / Test−9.1 / Batch3−Batch2, 구간화의 F1·Accuracy는 Train−Valid / Valid−Test / Target−Test / Batch2−Batch3입니다. MAPE·Accuracy 차이는 %p, F1 차이는 0–1 점수 차이입니다. 과제 F1 Target과 논문 3구간 Target은 없어 N/A로 표시합니다.

Train은 개발 29셀의 외부 CV fold 평균입니다. Valid 7셀과 외부 fold MAPE 5.44–16.62%의 변동을 고려하면 작은 Train–Valid Gap만으로 과적합 부재를 확정할 수 없습니다. Valid와 Test는 학습 셀 수도 29·36셀로 다릅니다. 선행 EDA에서 세 배치를 확인했으므로 완전히 미관측한 독립 시험으로 주장하지 않습니다.

'''
    if has3:
        modeling += '### Batch 3 추가 평가 — 지정 9행\n\n'+mt(batch3_reporting_format(extended))+'''

Batch 3 오차는 Batch 2보다 16.453%p 낮습니다. 단수명 중심 Batch 2와 장수명 비중이 높은 Batch 3의 구성 차이를 함께 해석하며, Gap만으로 특정 피처의 과적합이나 구조의 인과 효과를 확정하지 않습니다. [품질 규칙 민감도: 44셀·40셀](DAY2/results/batch3_quality_sensitivity.csv)

'''
    modeling += f'''### 논문 회귀 결과와 비교

{mt(paper)}

과제 Target **9.1%**는 논문 Abstract의 대표 보고값입니다. Table 1은 모델·평가셋별 값이며, Primary 제외 후는 이상 셀 1개를 제외한 결과입니다. 이번 Ridge의 Batch 2 28.609%는 Target보다 **19.509%p** 높습니다. 논문 학습·Primary는 2017-05-12·2017-06-30 혼합 분할이고, 이번 학습은 2017-05-12만, 필수 테스트는 2018-02-20이므로 동일 조건 재현이 아닙니다. 논문 Train은 학습 적합 성능, 이번 Train은 Nested-CV입니다.

Batch 3의 논문 품질 규칙 적용 40셀에서 Ridge MAPE는 **11.430%**이며, 논문 Secondary Variance 11.4%, Discharge 8.6%, Full 10.7%와 참고 비교합니다. 이번 두 피처 Ridge는 논문의 모델과 다르므로 숫자가 가깝다고 재현 성공을 주장하지 않습니다.

## 구간화 후 결과 — 같은 회귀 예측의 2구간·3구간 평가

회귀에서 얻은 예측 수명과 실제 수명에 같은 고정 경계를 적용합니다. **모델·입력 피처·회귀 예측값은 동일하며, 구간화는 평가 방식의 추가입니다.** 구간화 전 MAPE와 구간화 후 Accuracy·F1은 다른 질문에 대한 지표여서 서로의 개선율로 비교하지 않습니다. 경계는 기존 EDA·논문 기준이며 테스트 점수로 조정하지 않았습니다.

{mt(overview)}

F1은 고정 2개 또는 3개 클래스의 Macro-F1입니다. 실제·예측 모두 없는 클래스도 F1=0으로 평균에 포함합니다. Hold-out에는 단수명 정답이 없어 관련 Gap은 **클래스 구성 차이를 포함한 참고 차이**이며, 단수명 식별력이나 동일 클래스 구성의 일반화를 판단하는 근거로 사용하지 않습니다.

{chr(10).join(group_sections)}

### 단수명 식별과 2구간·3구간 해석

{mt(short)}

Batch 2는 2구간 Accuracy **41.026%**, Macro-F1 **0.409**로 3구간 **25.641%**, **0.352**보다 높습니다. 2구간 단수명은 30셀 중 7셀(23.3%), 3구간 단수명은 28셀 중 2셀(7.1%)을 맞혔습니다. 단수명 경계와 클래스 수가 달라 구간 수 자체의 우월성으로 일반화하지 않습니다. Batch 1에 <500회 정답이 없어도 회귀값은 500회 미만으로 외삽할 수 있으므로, 단수명 출력을 못 하는 별도 분류기와 다릅니다.

Batch 3에서는 2구간 Accuracy **97.727%**가 높지만 전부 장수명으로 예측해 단수명 1셀을 놓쳤습니다. 3구간 Accuracy **81.818%**, Macro-F1 **0.545**이고 중간·장수명을 각각 구분했습니다. 2구간 Macro-F1 **0.494**보다 높지만 Batch 3에는 <500회 정답이 없어 그 클래스의 식별력을 검증한 것은 아닙니다. [혼동행렬](DAY2/results/life_class_confusion.csv) · [클래스별 Recall](DAY2/results/life_class_per_class.csv) · [학습 다수 클래스 기준선](DAY2/results/life_class_baselines.csv)

![회귀 예측 구간화의 혼동행렬](DAY2/results/10_life_class_confusion.png)

### 구간화 결과와 논문 분류 결과 비교

| 비교 항목 | 이번 프로젝트 | 논문 | 해석 |
| --- | --- | --- | --- |
| 2구간 입력·방법 | 초기 100회, Ridge 수명 예측을 550회로 구간화 | 초기 5회, 별도 Logistic 분류기 | 관측 기간·모델·학습 분할이 다른 참고 비교 |
| Batch 2 Accuracy | 41.026% | 과제 Target 95.1% | Target−Test = +54.074%p |
| Macro-F1 | Batch 2 0.409 | 과제 공식 Target 없음 | Target Gap N/A; 재계산 논문 F1은 참고값 |
| 3구간 | 500·1,000회 경계 | 대응하는 3구간 보고값 없음 | 자체 추가 평가; 논문 Target·Gap N/A |

논문 Table 2 Full classifier는 Primary **92.7%**, Secondary **97.5%**이고, 본문 통합 테스트 Accuracy는 **95.1%**입니다. 2구간 참고 비교는 이 보고값을 그대로 사용합니다.

{mt(secondary) if has3 else ''}

논문 Secondary Macro-F1은 보충자료 Full 혼동행렬에서 재계산한 값으로 논문 직접 보고값과 구분합니다. 해당 혼동행렬의 단수명 1셀도 놓쳤습니다. 높은 Accuracy가 단수명 식별을 보장하지 않는다는 점을 보여주며, 모델의 동등한 성능을 입증하지는 않습니다.

'''
    errors = f'''## 오류 분석

![실제·예측 수명과 Batch 2 잔차](DAY2/results/02_predictions_residuals.png)

{mt(top)}

절대 오차율 상위 5셀은 모두 실제 수명 500회 미만이며 과대예측했습니다. Batch 2 전체 **36/39셀 과대예측**, 평균 예측−실제 **+133.1회**입니다. 학습 최소 수명 534회보다 짧은 30셀 모두 과대예측됐습니다.

{mt(read('subgroup_metrics'))}

standard_structure와 below_batch1_life_min은 같은 30셀입니다. 구조·수명 범위·충전 정책이 함께 달라 독립적인 원인 증거로 중복 해석하지 않습니다. ΔQ log 분산의 수명 Spearman은 Batch 1 −0.826, Batch 2 −0.709인 반면 충전 시간은 +0.432에서 −0.802로 달라져 보조 피처 전이 위험을 시사합니다. 이는 원인 가설이며 인과 검증 결과는 아닙니다.

같은 Batch 2에서 Batch 1 중앙값 772.5회 고정 예측의 MAPE **58.728%** 대비 모델 **28.609%**로 상대 오차가 **51.29% 감소**했습니다. 입력 범위 안 22셀의 MAPE 34.30%가 밖 17셀 21.25%보다 높아 입력 범위 이탈만으로 오차를 설명하기도 어렵습니다. 단수명 개발 표본을 확보하고 개선 모델은 새로운 미사용 배치에서 검증해야 합니다.

## ESS 도메인 해석

배터리 기반 ESS(BESS)에서 초기 신호로 셀 선별·점검 우선순위·교체 예산을 검토하는 보조 지표로 활용할 수 있습니다. 목표는 **총 Cycle Life**이며 현재 시점의 RUL이나 화재 확률이 아닙니다. 단수명 과대예측은 교체 지연으로 이어질 수 있어 현재 성능으로 교체 시점을 단독 결정하기 어렵습니다.

| 의사결정 | 활용 가능성 | 추가 검증 |
| --- | --- | --- |
| 셀 선별·점검 우선순위 | 상대 수명과 단수명 누락 위험 참고 | 단수명 표본·제조 배치별 Recall·예측 구간 |
| 교체 예산·정비 계획 | 총수명 예측과 과대예측 편향 참고 | 실제 SOH·운전 이력·현장 데이터 |
| 측정 도입 | 100사이클 검사 시간·전력·설비 비용 평가 | 줄어든 점검 비용과 미탐·조기 교체 비용 비교 |

실험실 LFP/graphite 단일 셀·특정 급속 충전·36셀 학습 조건입니다. 현장의 달력 열화·온도·부분 충방전·셀 불균형과 분포 변화를 추가 검증해야 합니다. 현재 자료에 운영 비용이 없어 절감 금액이나 운영 임계값을 임의로 제시하지 않습니다.

## 평가항목과 산출물

{mt(assessment)}

배점은 과제의 DAY 1·DAY 2 평가 기준이며 자체 채점 점수가 아닙니다. 주 태스크 회귀의 지정 성능표와 Batch 2 Target Gap을 중심으로 평가 근거를 연결했습니다.

## 참고문헌

- Severson et al. (2019). Data-driven prediction of battery cycle life before capacity degradation. *Nature Energy*, 4, 383–391. [원논문](https://www.nature.com/articles/s41560-019-0356-8) · [보충자료](https://static-content.springer-cdn.com/esm/art%3A10.1038%2Fs41560-019-0356-8/MediaObjects/41560_2019_356_MOESM1_ESM.pdf)
- [저자 전처리·분할 코드](https://github.com/rdbraatz/data-driven-prediction-of-battery-cycle-life-before-capacity-degradation/blob/master/LoadData.m)
- [과제 지침·Reporting format](https://app.notion.com/p/DS-Mini-Project-3ecb7788ac2a808c972af1dc0cdce6e3)

## 수행자

최유정(울산 3반 U104): EDA, 모델 전략, 피처·모델 선택, 파이프라인, Batch 2·3 평가, 오류 분석 및 보고서 작성.
'''
    if not has3:
        modeling = re.sub(r'Batch 3의 논문 품질 규칙.*?\n\n', '', modeling, flags=re.S)
        modeling = re.sub(r'Batch 3에서는 2구간 Accuracy.*?\n\n', '', modeling, flags=re.S)
    text = intro + eda + modeling + errors
    (ROOT / 'README.md').write_text(text)
    write_notebook(text, frozen)
    print('README and executed notebook refreshed: DAY 1 feature scope; frozen regression and 2/3 groups.')


def write_notebook(text, frozen):
    """Reuse the report plus executable source/fit/prediction checks and inline figures."""
    cells = [nbformat.v4.new_markdown_cell('# DAY 2 · 계획서 기반 회귀와 구간화 비교\n\nREADME와 같은 결과를 직접 확인합니다. 전체 후보 탐색은 `python -m src.train --batch3`입니다.')]
    env = {}
    def code(source):
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            exec(source, env)
        cell = nbformat.v4.new_code_cell(source, execution_count=1+sum(c.cell_type=='code' for c in cells))
        cell.outputs = [nbformat.v4.new_output('stream',name='stdout',text=stream.getvalue())]
        cells.append(cell)
    code("from pathlib import Path\nimport sys, json\nimport numpy as np\nimport pandas as pd\nROOT=Path.cwd()\nif ROOT.name=='DAY2': ROOT=ROOT.parent\nsys.path.insert(0,str(ROOT))\nOUT=ROOT/'DAY2/results'\nassert (OUT/'frozen_config.json').exists()\nprint('계획서 피처 모델:', json.loads((OUT/'frozen_config.json').read_text())['selected_config'])")
    for section in re.split(r'(?=^## )', text, flags=re.M):
        section = section.replace('(DAY2/results/', '(results/').replace('(DAY1/', '(../DAY1/').replace('(data/', '(../data/').replace('(30-ESSHealth-scratch.ipynb)', '(../30-ESSHealth-scratch.ipynb)').replace('(DAY2/03_modeling.ipynb)', '(03_modeling.ipynb)')
        cell = nbformat.v4.new_markdown_cell(section)
        attachments = {}
        def image(match):
            filename = match.group(2)
            attachments[filename] = {'image/png':base64.b64encode((OUT/filename).read_bytes()).decode()}
            return '!['+match.group(1)+'](attachment:'+filename+')'
        cell.source = re.sub(r'!\[([^\]]*)\]\(results/([^)]*\.png)\)',image,cell.source)
        if attachments: cell['attachments']=attachments
        cells.append(cell)
    cells.append(nbformat.v4.new_markdown_cell('## 직접 재현: 고정 설정 재학습·구간화 확인\n\n아래는 저장된 계획서 피처에서 최종 설정을 재학습해 외부 예측을 대조합니다. 원본 피처와 미래 정보 배제는 `python -m src.verify_results`에서 검증합니다.'))
    code("from src.train import fit_model, predict\nfrom src.preprocess import development_split\nfrom src.paper_comparison import life_labels, classification_metrics\nfrozen=json.loads((OUT/'frozen_config.json').read_text())\nfeatures=pd.read_csv(OUT/'cell_features_all.csv')\npredictions=pd.read_csv(OUT/'predictions_all.csv')\nb1=features[features.batch==1]\nmodel=fit_model(b1,frozen['selected_config'])\nfor batch in sorted(set(features.batch)-{1}):\n    data=features[features.batch==batch]\n    actual=predictions[predictions.batch==batch].set_index('cell_id').loc[data.cell_id]\n    pred=predict(model,data,frozen['selected_config'])\n    np.testing.assert_allclose(pred,actual.predicted_cycle_life,rtol=1e-10)\n    for scheme in ['binary_550','three_500_1000']:\n        print(batch,scheme,classification_metrics(life_labels(data.cycle_life,scheme),life_labels(pred,scheme),scheme))\nprint('PASS: 고정 회귀 예측 재현; 같은 예측의 2구간·3구간 평가')")
    code("print(pd.read_csv(OUT/'model_performance.csv').to_string(index=False))\nprint(pd.read_csv(OUT/'life_class_metrics.csv').to_string(index=False))\nprint(pd.read_csv(OUT/'selected_feature_vif.csv').to_string(index=False))")
    nb = nbformat.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':frozen['python']}})
    nbformat.validate(nb)
    nbformat.write(nb, ROOT / 'DAY2/03_modeling.ipynb')


if __name__ == '__main__':
    build()
