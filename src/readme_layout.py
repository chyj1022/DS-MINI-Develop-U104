"""Arrange the detailed report in the assignment's README sample order."""
import re

import pandas as pd

from .preprocess import ROOT
from .report import markdown_table

OUT = ROOT / 'DAY2/results'


def sections(text, level):
    pattern = '^' + '#' * level + ' '
    matches = list(re.finditer(pattern + r'.+$', text, re.MULTILINE))
    preamble = text[:matches[0].start()] if matches else text
    parts = {}
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        parts[match.group()] = text[match.end():end].strip()
    return preamble.rstrip(), parts


def align_with_assignment(text):
    """Preserve detailed evidence while putting final-model evidence first."""
    if '## 원논문 기반 추가 개발·비교' not in text:
        return text
    preamble, main = sections(text, 2)
    _, paper = sections(main.pop('## 원논문 기반 추가 개발·비교'), 3)
    design = paper.pop('### 1. 원논문 설계를 실제 구현한 추가 실험')
    choice = paper.pop('### 2. 논문 피처군별 회귀 모델 비교와 최종 선택')
    groups = paper.pop('### 4. 장·단과 단·중·장 그룹 구성')
    error = paper.pop('### 8. 최종 회귀를 같은 수명 기준으로 나눈 오류 분석')
    commands = paper.pop('### 추가 분석의 실행·계산 근거')

    distribution = pd.read_csv(OUT / 'life_class_distribution.csv')
    binary = distribution[(distribution.scheme == 'binary_550') &
                          (distribution.life_group == 'short_le550')].sort_values('batch')
    proportions = ' / '.join(f'Batch {int(r.batch)}: {r.proportion_pct:.1f}%'
                             for r in binary.itertuples())
    group_text = ('#### 원논문 장·단수명 기준과 프로젝트 세 구간 비교\n\n' + groups +
                  '\n\n550회 기준의 단수명 비율은 **' + proportions + '**입니다. '
                  '두 기준 모두 배치별 구성 차이를 확인하는 데 사용하며, 실제 수명 그룹은 회귀 입력에서 제외합니다.')
    eda = main['## EDA'].replace('### 2. 열화 곡선 분석', group_text + '\n\n### 2. 열화 곡선 분석', 1)
    eda = eda.replace(
        '| ΔQ 요약값의 중복 | 대표 5개 피처의 16개 부분집합 비교 | `feature_ablation.csv`, 학습 배치 VIF |',
        '| ΔQ 요약값의 중복 | 초기 부분집합 비교 후 논문 13·20개 후보에 Elastic Net 적용 | 학습 배치 VIF·`paper_model_coefficients.csv` |')
    eda = eda.replace(
        '| 작은 표본과 정책별 실험 | 정규화 선형 모델·정책별 분할 | Nested-CV 및 정책 Hold-out |',
        '| 작은 표본과 정책별 실험 | 정규화 선형 모델·정책별 분할·fold 내부 전처리 | `paper_model_fold_manifest.csv`·Nested-CV·정책 Hold-out |')
    eda += ('\n\n논문의 장·단수명 기준에는 초기 5회 ΔQ 신호와 L1 Logistic Regression을 적용했습니다. '
            '프로젝트 세 구간 분류는 초기 100회 Full 피처를 사용합니다. 두 분류 모두 Batch 1의 클래스 구성을 '
            '확인하고 Macro-F1·단수명 Recall을 함께 평가합니다.')

    pipeline = '''### 개발 파이프라인과 데이터 분할

| 단계 | 최종 모델에서 수행한 내용 | 구현·저장 근거 |
| --- | --- | --- |
| 품질 검사 | 원본 Batch별 불완전·결측 타깃 제외와 셀별 사유 기록 | preprocess.py / cell_audit_all.csv |
| 피처 추출 | 회귀·3분류 초기 100회, 이진 분류 초기 5회만 사용 | paper_models.py / paper_model_features.csv |
| Hold-out 고정 | Batch 1 개발 29셀·16정책 / 검증 7셀·4정책 분리 | split_manifest.csv |
| Nested-CV | 정책별 외부 4-fold 검증; 각 학습 fold 안에서 내부 4-fold 선택 | paper_model_cv_folds.csv / paper_model_fold_manifest.csv |
| fold 내부 전처리 | 결측 중앙값·표준화 통계를 해당 학습 fold에만 적합 | paper_models.py의 fit |
| 최종 후보 선택 | 개발 29셀의 내부 CV로 피처군·파라미터 선택 | paper_model_candidates.csv / paper_models_config.json |
| Hold-out 평가 | 개발 29셀로 적합한 모델로 검증 7셀 평가 | paper_model_metrics.csv / paper_model_predictions.csv |
| 모델 고정·외부 평가 | 설정 고정 후 Batch 1 전체 36셀 재학습; 같은 모델로 Batch 2·3 평가 | paper_models_config.json / paper_model_predictions.csv |
| 수치·누수 검증 | 미래 정보 불변성·정책 분리·학습 전용 전처리·지표 재계산 | verify_paper_results.py / verify_results.py |

Hold-out은 최종 후보 선택에 쓰지 않는 별도 검증입니다. 셀을 나누는 것에 더해 **동일 충전 정책이 개발·Hold-out과 CV train·valid에 겹치지 않도록** 정책 단위로 분리했습니다. Train은 학습 오차가 아니라 Nested-CV 외부 검증 평균입니다. Batch 2·3의 성능으로 최종 후보를 선택하지 않았으며, 선행 탐색에서 외부 데이터를 확인한 이력은 성능 비교의 조건으로 함께 보고합니다.'''
    rationale = '''- **후보 모델:** 논문 Variance(단일 ΔQ 선형 회귀), Discharge(13개 피처 Elastic Net), Full(20개 피처 Elastic Net). 초기 구현에서 중앙값·Linear·Ridge·Elastic Net·SVR·Random Forest도 비교했습니다.
- **최종 모델:** Discharge Elastic Net, `alpha=0.01`, `l1_ratio=0.2`; `log10(cycle_life)` 학습 후 사이클 단위로 복원합니다.
- **선택 이유:** ΔQ의 초기 열화 신호와 용량 변화를 함께 반영하고, 상관된 후보 피처는 L1·L2 정규화로 제어합니다. Batch 1 개발 CV MAPE를 기준으로 피처군과 파라미터를 선택했고, 선택 과정을 포함한 Nested-CV와 별도 Hold-out으로 검증했습니다. 최종 13개 후보 중 비영 계수는 6개입니다. 여기서 최적은 사전에 정한 후보군·분할·선택 지표 안의 최저 후보를 의미합니다.
- **추가 분류 모델:** 논문 초기 5회 장·단수명 분류에는 Full L1 Logistic Regression, 프로젝트 확장 3분류에는 초기 100회 Full L2 Logistic Regression을 사용합니다. 분류의 선택 지표는 고정 클래스 Macro-F1입니다.'''
    initial = main.pop('## Modeling: 초기 구현과 논문 기반 확장')
    initial = re.sub(r'^### ', '#### ', initial, flags=re.MULTILINE)
    initial = initial.replace('| 최종 선택 |', '| 초기 Ridge 선택 |')
    modeling = ('### 피처 엔지니어링 전략\n\n' + design +
                '\n\n### 모델 선택 및 근거\n\n' + rationale +
                '\n\n#### 논문 피처군 비교와 Batch 1 최종 선택\n\n' + choice +
                '\n\n' + pipeline + '\n\n### 초기 구현의 비교 근거\n\n' + initial)

    performance = main['## 성능 결과']
    performance = performance.replace(
        '회귀는 MAPE, 분류는 Macro-F1·Accuracy를 보고합니다.',
        '회귀는 MAPE, 분류는 Macro-F1·Accuracy를 보고합니다. 원논문 Target은 회귀 MAPE **9.1%**, '
        '이진 분류 오류율 **4.9%(1−Accuracy)**, 즉 **Accuracy 95.1%**입니다.')
    for heading, content in paper.items():
        performance += '\n\n' + re.sub(r'^### \d+\. ', '### ', heading) + '\n\n' + content

    predictions = pd.read_csv(OUT / 'paper_model_predictions.csv')
    worst = predictions[(predictions.task == 'regression') & (predictions.family == 'Selected') &
                        (predictions.partition == 'Test (Batch 2)')].copy()
    worst['signed_error_cycles'] = worst.prediction - worst.cycle_life
    worst['ape_pct'] = 100 * worst.signed_error_cycles.abs() / worst.cycle_life
    worst = worst.sort_values(['ape_pct', 'cell_id'], ascending=[False, True]).head(5)
    worst[['batch', 'cell_id', 'policy', 'cycle_life', 'prediction', 'signed_error_cycles',
           'ape_pct']].to_csv(OUT / 'paper_selected_top_errors.csv', index=False, encoding='utf-8-sig')
    worst_table = worst[['cell_id', 'policy', 'cycle_life', 'prediction', 'signed_error_cycles', 'ape_pct']].rename(columns={
        'cell_id': '셀 ID', 'policy': '충전 정책', 'cycle_life': '실제 수명', 'prediction': '예측 수명',
        'signed_error_cycles': '예측−실제 (회)', 'ape_pct': '오차율 (%)'})
    nshort = int((worst.cycle_life < 500).sum())
    nmiddle = int(worst.cycle_life.between(500, 1000).sum())
    nover = int((worst.signed_error_cycles > 0).sum())
    top_errors = ('### 최종 모델이 가장 크게 틀린 Batch 2 셀\n\n' + markdown_table(worst_table) +
                  f'\n\nMAPE에 대응하는 셀별 절대 오차율 상위 5셀입니다. 이 중 단수명 <500회는 **{nshort}셀**, '
                  f'중간 500–1,000회는 **{nmiddle}셀**이며 **{nover}셀 모두 과대예측**했습니다. '
                  '중간 수명 셀에서도 큰 절대 오차가 발생해 최종 모델의 MAPE 개선을 모든 셀의 오차 감소로 해석하지 않습니다. '
                  '충전 정책과 구조 표시도 함께 제시하지만, 상위 몇 셀의 공통점을 인과 원인으로 단정하거나 오차가 큰 셀을 제거하지 않습니다. '
                  '셀별 근거는 [paper_selected_top_errors.csv](DAY2/results/paper_selected_top_errors.csv)에 저장합니다.\n\n'
                  '### 원인 가설 및 개선 방향\n\n'
                  '학습 수명 범위와 외부 배치의 수명·클래스 분포 차이, 충전 정책·수집 조건, 피처와 수명 관계의 변화를 함께 점검합니다. '
                  '최종 모델은 Batch 2 39셀 중 37셀을 과대예측하므로 일부 극단값만의 문제가 아닙니다. '
                  '후속 개발에서는 단수명 개발 표본을 확보하고, 신규 제조 배치·정책을 분리한 미사용 평가셋으로 검증하며, '
                  '과대예측 비용과 셀별 예측 구간을 반영한 의사결정 기준을 평가합니다. 현재 Batch 2·3에 맞춰 재조정한 성능을 새 검증 성능으로 보고하지 않습니다.')
    old_errors = main.pop('## 초기 Ridge 오류 분석: 추가 개발 전 비교 근거')
    old_errors = re.sub(r'^### ', '#### ', old_errors, flags=re.MULTILINE)
    errors = ('### 최종 모델의 수명 그룹별 오류와 편향\n\n' + error + '\n\n' + top_errors +
              '\n\n### 초기 Ridge 오류 분석: 비교 근거\n\n' + old_errors)
    main['## 환경 설정'] += '\n\n### 분석·보고서 재현 명령\n\n' + commands
    main['## EDA'] = eda
    main['## 성능 결과'] = performance
    main['## Modeling'] = modeling
    main['## 오류 분석'] = errors
    order = ['## 프로젝트 개요', '## 파일 구조', '## 환경 설정', '## EDA', '## Modeling',
             '## 성능 결과', '## 오류 분석', '## ESS 도메인 해석', '## 평가항목별 확인 위치',
             '## 참고문헌', '## 팀 구성']
    assert set(main) == set(order), 'Unexpected README sections; preserve them before rearranging'
    result = preamble + '\n\n' + '\n\n'.join(h + '\n\n' + main[h] for h in order) + '\n'
    result = result.replace('추가 개발·비교 절', 'Modeling·성능 결과 절')
    result = result.replace('원논문 기반 추가 개발·비교', 'Modeling·성능 결과')
    assert sorted(re.findall(r'!\[[^\]]*\]\(([^)]+)\)', result)) == sorted(re.findall(r'!\[[^\]]*\]\(([^)]+)\)', text))
    return result
