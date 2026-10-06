"""Evidence-backed submission refinements without changing fitted models."""
import json
import re

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression

from .paper_comparison import PARTITIONS, trained_evidence
from .preprocess import ROOT
from .report import markdown_table

OUT = ROOT / 'DAY2/results'
PAPER = 'https://www.nature.com/articles/s41560-019-0356-8'
AUTHOR = 'https://github.com/rdbraatz/data-driven-prediction-of-battery-cycle-life-before-capacity-degradation/blob/master/LoadData.m'


def generate_evidence():
    features = pd.read_csv(OUT / 'paper_model_features.csv')
    coefficients = pd.read_csv(OUT / 'paper_model_coefficients.csv')
    selected = coefficients[(coefficients.task == 'regression') & (coefficients.family == 'Selected')]
    split = pd.read_csv(OUT / 'split_manifest.csv')
    development = set(split[split.split == 'development'].cell_id)
    vif_rows = []
    for scope, frame in [('Batch 1 full fit', features[features.batch == 1]),
                         ('Batch 1 development', features[(features.batch == 1) & features.cell_id.isin(development)])]:
        for name, columns in [('Discharge 13 candidates', selected.feature.tolist()),
                              ('Selected 6 nonzero', selected[selected.nonzero].feature.tolist())]:
            x = SimpleImputer(strategy='median', keep_empty_features=True).fit_transform(frame[columns])
            for i, column in enumerate(columns):
                other = np.delete(x, i, axis=1)
                r2 = LinearRegression().fit(other, x[:, i]).score(other, x[:, i])
                vif_rows.append({'scope': scope, 'feature_set': name, 'feature': column, 'n_cells': len(frame),
                                 'r_squared': r2, 'vif': 1 / (1-r2) if r2 < 1 else np.inf,
                                 'imputation': 'median fitted on this Batch 1 scope; diagnostic only'})
    tables = {'paper_selected_vif': pd.DataFrame(vif_rows)}
    audits = pd.read_csv(OUT / 'cell_audit_all.csv')
    tables['batch1_exclusions_verified'] = audits[(audits.batch == 1) & ~audits.included].copy()
    folds = pd.read_csv(OUT / 'paper_model_cv_folds.csv')
    rows = []
    for fold in sorted(folds.fold.unique()):
        a = folds[(folds.task == 'regression') & (folds.family == 'Selected') & (folds.fold == fold)].iloc[0]
        b = folds[(folds.task == 'regression') & (folds.family == 'Discharge') & (folds.fold == fold)].iloc[0]
        selected_config, discharge_config = json.loads(a.config), json.loads(b.config)
        family = selected_config.pop('family')
        assert family == 'Discharge' and selected_config == discharge_config
        assert np.isclose(a.mape_pct, b.mape_pct, rtol=0, atol=1e-12)
        rows.append({'fold': fold, 'selected_family': family, **selected_config,
                     'selected_mape_pct': a.mape_pct, 'discharge_mape_pct': b.mape_pct,
                     'same_config_and_score': True})
    tables['paper_selected_fold_choices'] = pd.DataFrame(rows)
    predictions = pd.read_csv(OUT / 'paper_model_predictions.csv')
    for fold in sorted(folds.fold.unique()):
        frame = predictions[(predictions.task == 'regression') & (predictions.partition == PARTITIONS[0]) &
                            (predictions.fold == fold)]
        a = frame[frame.family == 'Selected'].sort_values('cell_id')
        b = frame[frame.family == 'Discharge'].sort_values('cell_id')
        np.testing.assert_array_equal(a.cell_id, b.cell_id)
        np.testing.assert_allclose(a.prediction, b.prediction, rtol=0, atol=1e-12)
    reference = pd.read_csv(OUT / 'paper_regression_metrics.csv').set_index('model')
    comparisons = []
    for family in ['Variance', 'Discharge', 'Full']:
        group = predictions[(predictions.task == 'regression') & (predictions.family == family) &
                            (predictions.partition == PARTITIONS[3]) & ~predictions.cell_id.isin([2, 37, 42, 43])]
        if group.empty:
            continue
        residual = group.prediction - group.cycle_life
        mape = float(100 * (residual.abs()/group.cycle_life).mean())
        target = reference.loc[family].secondary_mape_pct
        comparisons.append({'model': family, 'n_cells': len(group), 'mape_pct': mape,
                            'paper_secondary_mape_pct': target, 'gap_pp': mape-target,
                            'comparison': 'same date and author quality rule; different training split', 'source': PAPER})
    tables['paper_secondary_comparison'] = pd.DataFrame(comparisons, columns=[
        'model', 'n_cells', 'mape_pct', 'paper_secondary_mape_pct', 'gap_pp', 'comparison', 'source'])
    tables['paper_target_91_audit'] = pd.DataFrame([{
        'model': 'Full', 'primary_excluded_n': 42, 'primary_excluded_reported_mape_pct': 7.5,
        'secondary_n': 40, 'secondary_reported_mape_pct': 10.7,
        'weighted_reported_mape_pct': (42*7.5+40*10.7)/82,
        'abstract_target_mape_pct': 9.1,
        'status': 'inference from rounded Table 1 values; pooling not explicitly stated', 'source': PAPER}])
    tables.update(trained_evidence())
    for name, frame in tables.items():
        frame.to_csv(OUT / f'{name}.csv', index=False, encoding='utf-8-sig')
    return tables


def replace_table(text, heading, table):
    start = text.index(heading) + len(heading)
    match = re.search(r'(?:^\|.*\n)+', text[start:], re.MULTILINE)
    assert match, heading
    left, right = start+match.start(), start+match.end()
    return text[:left] + markdown_table(table) + '\n' + text[right:]


def collapse_section(text, heading, end_heading, summary):
    start, end = text.index(heading), text.index(end_heading, text.index(heading)+len(heading))
    body = text[start+len(heading):end].strip()
    folded = f'<details>\n<summary>{summary}</summary>\n\n{body}\n\n</details>\n\n'
    return text[:start] + folded + text[end:]


def review_readme(text):
    """Apply verified feedback while preserving assignment tables and figures."""
    if '### 핵심 결과와 논문 비교 기준' in text:
        return text
    tables = generate_evidence()
    scores = pd.read_csv(OUT / 'paper_model_metrics.csv')
    regression = scores[(scores.task == 'regression') & (scores.family == 'Selected')].set_index('partition')
    quality = tables['paper_secondary_comparison'].set_index('model')
    has3 = not quality.empty
    rows = []
    for partition in PARTITIONS:
        if partition not in regression.index:
            continue
        r = regression.loc[partition]
        rows.append({'평가': partition, '셀 수': int(r.n_cells), '최종 회귀 MAPE (%)': r.mape_pct,
                     '비교·판단': '정책별 Nested-CV 검증 평균' if partition == PARTITIONS[0] else
                     '모델 선택에 사용하지 않은 정책 Hold-out' if partition == PARTITIONS[1] else
                     f'과제 Target 9.1 대비 +{r.mape_pct-9.1:.3f}%p; 논문 밖 배치 일반화' if partition == PARTITIONS[2] else
                     '전체 유효 셀의 추가 평가'})
    if 'Discharge' in quality.index:
        r = quality.loc['Discharge']
        rows.append({'평가': 'Batch 3 — 논문 품질 규칙', '셀 수': int(r.n_cells), '최종 회귀 MAPE (%)': r.mape_pct,
                     '비교·판단': f'논문 Discharge Secondary 8.6 대비 +{r.gap_pp:.3f}%p'})
    summary = markdown_table(pd.DataFrame(rows))
    core = f'''### 핵심 결과와 논문 비교 기준

{summary}

**과제 필수 평가는 Batch 1 학습 → Batch 2(2018-02-20) 테스트**입니다. 이 Batch 2는 논문 Methods의 세 배치에 포함되지 않으므로, Target Gap은 과제 목표 대비 차이이자 논문 밖 배치의 일반화 결과입니다. **원논문의 Secondary-test에 가장 가까운 비교는 Batch 3(2018-04-12)의 저자 품질 규칙 적용 40셀**입니다. 같은 날짜·품질 규칙으로 맞춘 평가이며, 학습 데이터 구성과 분할은 달라 완전한 재현 실험으로 부르지 않습니다. 논문의 9.1%는 과제 Target으로 유지하고, 배치가 대응하는 비교에는 Table 1의 모델별 Secondary 지표를 사용합니다.

Batch 2 MAPE는 Ridge **28.609% → 25.563%**로 개선됐지만 MAE·RMSE는 악화했고, 최종 모델은 **37/39셀을 과대예측**했습니다. 분류는 Batch 1 단수명 학습 표본이 **550회 기준 1셀, <500회 기준 0셀**이므로 활용에 제약이 있습니다. 초기 5회 이진 단수명 Recall은 **Batch 2 33.3%(10/30), Batch 3 0%(0/1)**입니다. 자세한 분류 표보다 이 학습 구성과 클래스별 Recall을 먼저 해석합니다.
'''
    text = text.replace('## 파일 구조', core.strip() + '\n\n## 파일 구조', 1)
    text = text.replace('- 주 태스크:', '- 타깃: 명목 용량 1.1Ah의 80%(0.88Ah)까지의 총 Cycle Life; 현재 코드는 원본 MAT의 저장된 `cycle_life`를 사용합니다.\n- 주 태스크:', 1)
    target = ('**9.1% Target의 출처:** Abstract의 보고값은 9.1%입니다. 본문은 이 값의 셀 집합을 명시하지 않습니다. '
              '회귀 Primary 43셀 중 이상 셀 1개 제외 후 42셀의 Full MAPE 7.5%와 Secondary 40셀의 10.7%를 '
              '가중하면 `(42×7.5 + 40×10.7)/82 = 9.060976%`로 한 자리 반올림 시 9.1%와 일치합니다. '
              '따라서 **Full의 Primary 제외 후 + Secondary 통합값으로 추정**할 수 있으나, 반올림된 표 값에서의 '
              '역산이며 논문이 통합 계산을 명시한 것은 아닙니다. [계산 근거](DAY2/results/paper_target_91_audit.csv).')
    marker = '### 원논문 회귀 Table 1과 비교'
    text = text.replace(marker, marker + '\n\n' + target, 1)
    text = text.replace('관측 끝 Q를', '입력 관측 창의 마지막 용량(Q100 또는 Q5)을')
    text = text.replace('관측 끝 Q', '입력 관측 창의 마지막 용량(Q100 또는 Q5)')

    exclusions = tables['batch1_exclusions_verified'][['cell_id', 'cycle_life', 'reason']].rename(columns={
        'cell_id': '원본 ID (0-based)', 'cycle_life': '파일의 cycle_life', 'reason': '제외 사유'})
    exclusions['제외 사유'] = exclusions['제외 사유'].map({
        'continued_in_later_run': '2017-06-30에서 후속 측정; 현재 원본만으로 완결 수명 확인 불가',
        'early_stopped_incomplete_target': '조기 종료; 종료 시 용량이 0.88Ah보다 높음'})
    label_note = f'''### 타깃 정의와 Batch 1 제외 내역

원논문 Cycle Life는 명목 용량 **1.1Ah의 80%=0.88Ah**에 도달할 때까지의 사이클 수입니다. 현재 코드는 MAT의 **저장된 `cycle_life`를 라벨로 읽고**, 유한값·100회 초과 조건과 완결성 제외 규칙을 적용합니다. 용량 곡선에서 0.88Ah 도달 사이클을 새로 계산한 라벨이 아닙니다. 입력의 마지막 용량은 관측 창의 **Q100 또는 Q5**로, 전체 수명 종료 용량을 사용하지 않습니다.

{markdown_table(exclusions)}

[저자 LoadData.m]({AUTHOR})은 첫 5셀의 기록을 2017-06-30 파일의 후속 측정과 연결하고 라벨을 계산합니다. 이 프로젝트의 원본 목록에는 해당 날짜 파일이 없어, 셀 0–4의 양수 `cycle_life`를 완결 수명으로 취급하지 않고 제외했습니다. 보충자료 Table 9에서 이 5셀의 최종 수명은 **1,852·2,160·2,237·1,434·1,709회**입니다. 나머지 5셀은 실제 기록 종료 용량이 0.88Ah보다 높습니다. 따라서 **46→36셀 처리로 학습 최대 수명이 1,074회까지 좁아진 것**이며, 원논문 Batch 1 전체가 원래 이 범위였다는 뜻은 아닙니다. Batch 3 >1,000회 그룹의 평균 −85.5회 과소예측과 함께 검토할 데이터 처리·학습 범위 조건으로 기록합니다. [제외 내역](DAY2/results/batch1_exclusions_verified.csv).
'''
    text = text.replace('## EDA\n\n', '## EDA\n\n' + label_note.strip() + '\n\n', 1)
    vif = tables['paper_selected_vif']
    six = vif[(vif.scope == 'Batch 1 full fit') & (vif.feature_set == 'Selected 6 nonzero')]
    diagnostics = f'''### 최종 피처의 다중공선성 진단

기존 **290.43→2.74**는 초기 10개·대표 5개 피처의 VIF이며 최종 Discharge 피처의 결과가 아닙니다. 최종 비영 피처 6개를 **Batch 1 전체 36셀의 중앙값으로 대치**하고, 각 피처를 나머지 피처에 절편 포함 OLS로 회귀해 `VIF=1/(1−R²)`를 계산했습니다. 이는 사후 학습 입력 진단이며 모델 선택·테스트 전처리에 사용하지 않습니다.

{markdown_table(six[['feature', 'vif']].rename(columns={'feature': '최종 비영 피처', 'vif': 'VIF'}))}

최대 VIF는 **{six.vif.max():.3f}**입니다. **상관된 피처를 유지하면서 Elastic Net으로 계수를 정규화했으며, 피처 독립성이나 계수 안정성을 보장하지 않습니다.** 높은 VIF가 정규화 모델의 예측 성능을 곧바로 무효화하지는 않지만, 개별 계수를 독립적인 영향으로 읽기는 어렵습니다. 정의와 해석은 [NIST VIF 설명](https://www.itl.nist.gov/div898/software/dataplot/refman2/auxillar/vif.htm)을 참고했습니다. 13개 후보·6개 비영 피처 및 개발 29셀의 진단도 [paper_selected_vif.csv](DAY2/results/paper_selected_vif.csv)에 저장합니다.
'''
    text = text.replace('### 개발 파이프라인과 데이터 분할', diagnostics.strip() + '\n\n### 개발 파이프라인과 데이터 분할', 1)
    text = text.replace('ΔQ 요약값 사이의 중복이 크고 보조 피처는 배치별 상관 부호가 달라질 수 있습니다.',
                        '다음 VIF 값은 **초기 후보 피처 진단**입니다. 최종 Discharge 피처 진단은 Modeling 절에서 별도로 제시합니다. ΔQ 요약값 사이의 중복이 크고 보조 피처는 배치별 상관 부호가 달라질 수 있습니다.', 1)
    choice = tables['paper_selected_fold_choices'][['fold', 'selected_family', 'alpha', 'l1_ratio', 'selected_mape_pct']].rename(columns={
        'fold': '외부 fold', 'selected_family': '선택 피처군', 'selected_mape_pct': 'MAPE (%)'})
    fold_note = ('#### Selected와 Discharge의 Nested-CV가 같은 이유\n\n' + markdown_table(choice) +
                 '\n\n외부 4개 fold 모두에서 Selected는 Discharge를 선택했고, **각 fold에서 Discharge 단독 탐색과 같은 파라미터·예측·점수**를 얻었습니다. '
                 '따라서 외부 평균 6.203%가 같습니다. **파라미터가 네 fold 전체에서 동일한 것은 아닙니다.** '
                 '검증 근거는 [paper_selected_fold_choices.csv](DAY2/results/paper_selected_fold_choices.csv)입니다. '
                 '논문의 Discharge도 13개 후보 중 6개 피처를 사용했습니다. 선택된 개수는 같지만 피처 집합과 학습 분할이 동일하다는 뜻은 아닙니다.')
    text = text.replace('### 최종 피처의 다중공선성 진단', fold_note + '\n\n### 최종 피처의 다중공선성 진단', 1)
    text = text.replace('### 모델 선택 및 근거', '### 모델 선택 및 근거\n\n학습 규모가 36셀인 표 형식 피처 데이터에서는 정규화 선형 모델의 검증·해석을 우선해 딥러닝을 후보에서 제외했습니다.', 1)

    comparison = tables['paper_secondary_comparison'][['model', 'n_cells', 'mape_pct', 'paper_secondary_mape_pct', 'gap_pp']].rename(columns={
        'model': '논문 대응 피처군', 'n_cells': 'Batch 3 셀 수', 'mape_pct': '이번 MAPE (%)',
        'paper_secondary_mape_pct': '논문 Secondary MAPE (%)', 'gap_pp': '이번−논문 (%p)'})
    secondary = ('### 논문 Secondary-test와 대응하는 Batch 3 40셀 비교\n\n' + markdown_table(comparison) +
                 '\n\n저자 코드의 채널·종료 용량·잡음 제외를 현재 원본 ID로 추적하면 전체 46셀에서 **2·23·32·37·42·43을 제외한 40셀**입니다. '
                 '그중 23·32는 기존 유효 라벨 검사에서 빠지고, 44셀에 2·37·42·43을 추가 제외합니다. '
                 '보충자료 Table 9의 Secondary 40셀과 저장된 수명값의 다중집합도 일치합니다. '
                 '전체 44셀 성능을 유지하면서 논문과 대응하는 40셀 비교를 함께 보고하며, 오차로 제외 대상을 정하지 않았습니다. '
                 'Full의 40셀 점수가 더 낮아도 테스트 결과로 모델을 바꾸지 않고 Batch 1 CV에서 선택한 **Discharge**를 최종 모델로 유지합니다. '
                 '초기 2피처 Ridge의 같은 40셀 MAPE는 11.430%로 논문 Variance 11.4%와 가깝지만, '
                 'Ridge는 충전 시간까지 사용하는 다른 모델이므로 직접 재현 근거로 사용하지 않습니다. '
                 '[모델별 비교 CSV](DAY2/results/paper_secondary_comparison.csv).')
    if has3:
        text = text.replace('### 원논문 회귀 Table 1과 비교', secondary + '\n\n### 원논문 회귀 Table 1과 비교', 1)
    rule = ('**Gap 공통 규칙: (+)는 성능 저하입니다.** 과제 행 이름을 유지하되, 낮을수록 좋은 회귀 MAPE는 '
            '`Valid−Train / Test−Valid / Test−Target / Batch3−Batch2`, 높을수록 좋은 분류 F1·Accuracy는 '
            '`Train−Valid / Valid−Test / Target−Test / Batch2−Batch3` 순서로 계산합니다. '
            'MAPE·Accuracy Gap은 %p, F1 Gap은 0–1 점수 차이입니다. Gap 부호만으로 과적합을 확정하지 않습니다.')
    text = text.replace('## 성능 결과\n\n', '## 성능 결과\n\n' + rule + '\n\n', 1)
    binary_caution = ('**분류 해석 전제:** Batch 1 단수명 ≤550회는 1셀뿐이고 개발셋에 포함됩니다. '
                      'OOF 예측은 이 단수명 1셀을 맞히지 못했으며, 해당 셀이 검증으로 빠진 fold의 학습셋에는 단수명 클래스가 없습니다. '
                      'Hold-out은 단수명 정답이 0셀이므로 **단수명 Recall은 N/A**입니다. Accuracy 100%와 Macro-F1 0.5는 '
                      '정의에 따라 계산한 값으로 유지하지만, 단수명 일반화 검증이나 Train–Valid 과적합 판단의 근거로 사용하지 않습니다. '
                      '이진 모델이 전혀 적합 불가능하다는 뜻은 아니며, 현재 표본 구성으로 안정적인 단수명 학습·검증을 하기 어렵다는 뜻입니다.')
    text = text.replace('### 실제 학습한 초기 5회 분류: 논문과의 비교',
                        '### 실제 학습한 초기 5회 분류: 논문과의 비교\n\n' + binary_caution, 1)
    three_caution = ('**학습 클래스 부재를 확인한 확장 실험:** Batch 1에 <500회 클래스가 없으므로 현재 Logistic 모델은 '
                     '중간·장수명만 학습하고 단수명 클래스를 출력하지 않습니다. Batch 2의 단수명 28셀은 이 모델 구조로 맞힐 수 없습니다. '
                     'Accuracy 17.949%는 정의된 평가값이지만, 학습에 없는 클래스를 포함한 외부 분포에서의 결과입니다. '
                     '일반적인 3클래스 식별력이나 분류 알고리즘 자체의 우열로 해석하지 않습니다.')
    text = text.replace('### 단·중·장 분류 성능표: 초기 100회 Full',
                        '### 단·중·장 분류 성능표: 초기 100회 Full\n\n' + three_caution, 1)
    for scheme, heading, extra in [
        ('binary_550', '### 장·단 이진 분류 성능표: 초기 5회 Full', '#### 초기 5회 이진 분류: Batch 3 추가 양식'),
        ('three_500_1000', '### 단·중·장 분류 성능표: 초기 100회 Full', '#### 단·중·장 분류: Batch 3 추가 양식')]:
        text = replace_table(text, heading, tables[f'paper_{scheme}_performance'])
        if extra in text:
            text = replace_table(text, extra, tables[f'paper_{scheme}_with_batch3_formatted'])

    text = collapse_section(text, '### 초기 구현의 비교 근거', '## 성능 결과', '초기 Ridge 피처·후보·파이프라인 상세 보기')
    text = collapse_section(text, '### 초기 Ridge 구현 성능: 같은 셀에서의 비교 기준',
                            '### 논문 Secondary-test와 대응하는 Batch 3 40셀 비교' if has3 else '### 원논문 회귀 Table 1과 비교',
                            '초기 Ridge 필수·추가 성능표와 기준선 상세 보기')
    text = collapse_section(text, '### 초기 Ridge 오류 분석: 비교 근거', '## ESS 도메인 해석', '초기 Ridge 잔차·오류 셀 상세 보기')
    old = pd.read_csv(OUT / 'metrics_all.csv').set_index('index')
    metric_comparison = pd.DataFrame([{'모델': label, 'MAPE (%)': r.mape_pct, 'MAE (회)': r.mae_cycles, 'RMSE (회)': r.rmse_cycles}
                                     for label, r in [('초기 Ridge', old.loc[PARTITIONS[2]]), ('최종 Discharge', regression.loc[PARTITIONS[2]])]])
    text = text.replace('### 최종 회귀 필수 평가: 논문 기반 Discharge Elastic Net',
                        '### 같은 Batch 2에서 초기·최종 모델 비교\n\n' + markdown_table(metric_comparison) +
                        '\n\n### 최종 회귀 필수 평가: 논문 기반 Discharge Elastic Net', 1)
    costs = '''### BESS 의사결정의 비용과 적용 기준

| 운영 판단 | 비용·평가 항목 | 적용 기준 |
| --- | --- | --- |
| 초기 100회 수명 평가 | 100사이클 측정 시간·전력·설비 점유·검사 지연 비용 | 예측으로 줄인 점검·교체 비용이 측정 비용을 상회하는지 검증 |
| 초기 5회 셀 선별 | 낮은 측정 부담과 단수명 미탐 비용 | 현재 Batch 2 단수명 Recall 33.3%로 단독 선별에 사용하지 않음 |
| 교체·점검 우선순위 | 과대예측으로 늦춘 교체 비용, 불필요한 제외·조기 교체 비용 | 실제 SOH·운전 이력과 함께 보수적 구간·점검 기준을 평가 |

현재 데이터에는 운영 비용이나 현장 RUL·안전 사고 라벨이 없어 금액·교체 임계값을 임의로 정하지 않습니다. 현장 비용표와 신규 배치 검증을 확보한 뒤 미탐·오탐 비용을 반영해 기준을 결정합니다.

### 해석 범위

정책·구조·수명·수집 조건이 함께 달라 관측 상관과 하위집단 차이만으로 인과 효과를 분리하지 않습니다. 계수는 상관된 입력의 조건부 값이며 개별 피처의 독립적 영향으로 해석하지 않습니다. 선행 탐색에서 외부 배치를 본 이력이 있어 이번 확장 분석은 새 미사용 테스트셋의 독립 검증과 구분합니다.'''
    text = text.replace('## 평가항목별 확인 위치', costs + '\n\n## 평가항목별 확인 위치', 1)
    text = text.replace('## 팀 구성', '## 수행자', 1)
    text = text.replace('│   └── results/                # 성능·CV·피처·audit·오류·그래프·모델',
                        '│   └── results/\n│       ├── provenance.json      # 원본·환경·초기 모델 해시\n│       ├── assessment_evidence.csv\n│       ├── paper_models_provenance.json\n│       ├── paper_selected_vif.csv\n│       └── ...                  # 성능·CV·피처·audit·오류·그래프')
    text = text.replace('│   └── verify_results.py',
                        '│   ├── verify_paper_results.py\n│   ├── readme_layout.py / submission_review.py\n│   └── verify_results.py')
    # Consolidate generic causal cautions; preserve all numerical and method details.
    for sentence in ['상관 변화 자체를 오차의 인과 원인으로 단정하지 않습니다. ',
                     '계수는 표준화 입력에 대한 log10 수명 계수입니다. 상관된 피처들의 조건부 계수이므로 물리적 인과 효과로 해석하지 않습니다.',
                     '상위 몇 셀의 공통점을 인과 원인으로 단정하거나 ',
                     '관측된 C-rate 상관을 최적 충전 정책의 인과 근거로 사용하지 않습니다.']:
        text = text.replace(sentence, '계수는 표준화 입력에 대한 log10 수명 계수입니다.' if sentence.startswith('계수는') else '')
    return '\n'.join(line.rstrip() for line in text.splitlines()) + '\n'
