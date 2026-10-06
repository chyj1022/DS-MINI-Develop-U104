# 데이터

원본은 프로젝트 루트의 `archive/`에 둡니다. 기존 CSV 변환 파일은 탐색용이며 DAY 2 학습은 원본 MAT를 직접 읽습니다.

- Batch 1: `2017-05-12_batchdata_updated_struct_errorcorrect.mat`
- Batch 2: `2018-02-20_batchdata_updated_struct_errorcorrect.mat` (과제 지정)
- 추가 Batch 3: `2018-04-12_batchdata_updated_struct_errorcorrect.mat`

Batch 1은 후속 배치에서 이어진 셀 0~4와 조기 종료 셀 8,10,12,13,22를 제외해 36셀을 사용합니다. Batch 2는 `cycle_life`가 없는 셀 22,23,35,36,37,38,39,40을 제외해 39셀을 평가합니다. 셀 ID는 0부터 시작하며 제외 내역은 `DAY2/results/cell_audit.csv`에 기록합니다. 테스트 오차를 기준으로 셀을 제거하지 않습니다.

라벨은 MAT의 저장된 `cycle_life`를 읽으며, 용량 곡선의 0.88Ah 도달 사이클을 새로 계산하지 않습니다. 원논문 기준 0.88Ah는 명목 1.1Ah의 80%입니다. 셀 0–4는 2017-06-30의 후속 기록이 있어야 완결 수명을 확인할 수 있지만 현재 원본 목록에 이 파일이 없습니다. 양수인 저장 라벨을 완결 수명으로 취급하지 않고 제외한 결과 학습 수명 범위가 534–1,074회로 좁아졌습니다. 이 조건은 논문 학습 구성과의 차이이며 `DAY2/results/batch1_exclusions_verified.csv`에 별도로 기록합니다.

최종 Batch 2 유효 39셀은 `summary.cycle`의 원본 번호로 최초 `QDischarge ≤0.88Ah` 도달 시점과 저장 라벨을 대조했습니다. 정확 일치 39셀, 불일치·미도달 각 0셀입니다. 배열 위치나 허용 오차를 사용하지 않았고 라벨 수정·추가 제외도 적용하지 않았습니다. 셀별 결과는 `DAY2/results/batch2_cycle_life_audit.csv`, 요약은 `batch2_cycle_life_audit_summary.json`입니다. 재검사는 `python -m src.submission_finalize --archive-dir archive`로 실행할 수 있으며 원본 디렉터리를 지정해도 결과는 현재 프로젝트의 `DAY2/results`에 저장됩니다.

회귀는 초기 Cycle 10·100의 `Qdlin`과 `Vdlin`, 2–100회의 summary·원시 온도/시간을 피처 계산에 사용합니다. 논문 기반 추가 이진 분류는 Cycle 4·5의 `Qdlin`과 2–5회 summary·원시 온도/시간만 사용하며 6회 이후 자료를 넣지 않습니다. 원본의 수명·기록 길이·최종 용량은 입력에 포함하지 않습니다. 전압 축은 2.0–3.5V의 공통 1,000점을 검사합니다.

원논문 공개 코드의 두 번째 배치는 `2017-06-30`입니다. 날짜가 다른 과제 Batch 2에 논문의 제외 인덱스를 그대로 적용하지 않습니다. 대용량 데이터 파일은 Git 업로드에서 제외합니다.

출처: [Severson et al. (2019)](https://www.nature.com/articles/s41560-019-0356-8), [저자 전처리 코드](https://github.com/rdbraatz/data-driven-prediction-of-battery-cycle-life-before-capacity-degradation/blob/master/LoadData.m).

Batch 3는 결측 수명 셀 23·32를 제외한 44셀을 동일 고정 모델로 평가합니다. 저자 코드의 품질 제외 규칙에 따른 원본 ID 2·37·42·43을 추가 제외한 40셀 결과는 별도 민감도로 보고합니다. 성능을 본 뒤 정한 제외 규칙이 아닙니다. 전압 축을 확인하고 곡선의 상수 원점 이동에 대한 ΔQ 분산 불변성도 검사하며, 절차·곡선 모양 차이 자체가 교정됐다고 주장하지 않습니다.

이 40셀은 논문 Secondary-test의 2018-04-12 배치와 날짜·품질 규칙이 대응하며, 수명값의 다중집합도 보충자료 Table 9의 40셀과 일치합니다. 따라서 전체 44셀 평가와 함께 논문 모델별 Secondary MAPE를 비교합니다. 과제 Batch 2는 논문 Methods의 배치에 포함되지 않아 과제 Target 대비 일반화 평가로 해석합니다. 학습 구성과 분할이 달라 동일 모델의 완전한 재현으로 부르지 않습니다.

`python -m src.train --batch3`로 원본부터 실행하고 `python -m src.diagnostics --batch3`로 저장 모델의 추가 진단만 실행할 수 있습니다. 원본 해시·크기·패키지 버전은 `DAY2/results/provenance.json`, 전체 제외 내역은 `cell_audit_all.csv`에 기록합니다.

논문 기반 추가 회귀·초기 5회 이진·단중장 분류를 독립적으로 재현하려면 `python -m src.paper_models --batch3`를 실행합니다. `python -m src.train --batch3`에도 이 단계를 연결했습니다. 기존 Ridge 결과와 추가 최종 Discharge Elastic Net 결과는 각각 보존하며, 주 성능표는 `paper_selected_regression_performance.csv`입니다. 모델 선택은 Batch 1의 정책별 nested CV에서 진행하고 Batch 2/3는 고정된 모델로 평가합니다. 원논문의 분할과 과제 Batch 2의 수집 날짜가 다르므로 동일 분할 재현으로 해석하지 않습니다.
