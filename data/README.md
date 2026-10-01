# 데이터

원본은 프로젝트 루트의 `archive/`에 둡니다. 기존 CSV 변환 파일은 탐색용이며 DAY 2 학습은 원본 MAT를 직접 읽습니다.

- Batch 1: `2017-05-12_batchdata_updated_struct_errorcorrect.mat`
- Batch 2: `2018-02-20_batchdata_updated_struct_errorcorrect.mat` (과제 지정)
- 추가 Batch 3: `2018-04-12_batchdata_updated_struct_errorcorrect.mat`

Batch 1은 후속 배치에서 이어진 셀 0~4와 조기 종료 셀 8,10,12,13,22를 제외해 36셀을 사용합니다. Batch 2는 `cycle_life`가 없는 셀 22,23,35,36,37,38,39,40을 제외해 39셀을 평가합니다. 셀 ID는 0부터 시작하며 제외 내역은 `DAY2/results/cell_audit.csv`에 기록합니다. 테스트 오차를 기준으로 셀을 제거하지 않습니다.

초기 Cycle 10·100의 `Qdlin`과 `Vdlin`, 2~100회의 summary만 피처 계산에 사용합니다. 원본의 수명·기록 길이·최종 용량은 입력에 포함하지 않습니다. 전압 축은 2.0~3.5V의 공통 1,000점을 검사합니다.

원논문 공개 코드의 두 번째 배치는 `2017-06-30`입니다. 날짜가 다른 과제 Batch 2에 논문의 제외 인덱스를 그대로 적용하지 않습니다. 대용량 데이터 파일은 Git 업로드에서 제외합니다.

출처: [Severson et al. (2019)](https://www.nature.com/articles/s41560-019-0356-8), [저자 전처리 코드](https://github.com/rdbraatz/data-driven-prediction-of-battery-cycle-life-before-capacity-degradation/blob/master/LoadData.m).

Batch 3는 결측 수명 셀 23·32를 제외한 44셀을 동일 고정 모델로 평가합니다. 저자 코드의 품질 제외 규칙에 따른 원본 ID 2·37·42·43을 추가 제외한 40셀 결과는 별도 민감도로 보고합니다. 성능을 본 뒤 정한 제외 규칙이 아닙니다. 전압 축을 확인하고 곡선의 상수 원점 이동에 대한 ΔQ 분산 불변성도 검사하며, 절차·곡선 모양 차이 자체가 교정됐다고 주장하지 않습니다.

`python -m src.train --batch3`로 원본부터 실행하고 `python -m src.diagnostics --batch3`로 저장 모델의 추가 진단만 실행할 수 있습니다. 원본 해시·크기·패키지 버전은 `DAY2/results/provenance.json`, 전체 제외 내역은 `cell_audit_all.csv`에 기록합니다.
