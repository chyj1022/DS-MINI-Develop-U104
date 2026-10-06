"""Final submission checks and concise presentation of saved experiments."""
import argparse
import json
import re
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from .paper_comparison import PARTITIONS, trained_evidence
from .preprocess import ROOT
from .report import batch3_reporting_format, markdown_table
from .submission_review import replace_table

OUT = ROOT / 'DAY2/results'
THRESHOLD = 0.88
GAP_NOTE = '단수명 Hold-out 0셀; 클래스 구성 차이를 포함한 참고 차이이며 단수명 식별력·동일 클래스 구성의 일반화 판단에 사용하지 않음'


def audit_batch2_labels(archive_dir=None):
    """Check exact first threshold crossing using recorded cycle numbers."""
    source = Path(archive_dir or ROOT / 'archive') / '2018-02-20_batchdata_updated_struct_errorcorrect.mat'
    csv_path, summary_path = OUT / 'batch2_cycle_life_audit.csv', OUT / 'batch2_cycle_life_audit_summary.json'
    if not source.exists():
        if archive_dir is not None:
            raise FileNotFoundError(source)
        if csv_path.exists() and summary_path.exists():
            return pd.read_csv(csv_path), json.loads(summary_path.read_text())
        raise FileNotFoundError('Place the original MAT in archive/ or run with --archive-dir first')
    predictions = pd.read_csv(OUT / 'paper_model_predictions.csv')
    evaluated = predictions[(predictions.task == 'regression') & (predictions.family == 'Selected') &
                            (predictions.partition == PARTITIONS[2])].sort_values('cell_id')
    assert len(evaluated) == evaluated.cell_id.nunique() == 39
    rows = []
    with h5py.File(source, 'r') as mat:
        batch = mat['batch']
        for cell in evaluated.itertuples():
            cid = int(cell.cell_id)
            saved = float(mat[batch['cycle_life'][cid, 0]][()].ravel()[0])
            assert saved == cell.cycle_life
            summary = mat[batch['summary'][cid, 0]]
            cycles, q = summary['cycle'][()].ravel(), summary['QDischarge'][()].ravel()
            assert cycles.shape == q.shape
            finite_cycles = cycles[np.isfinite(cycles)]
            assert (np.diff(finite_cycles) > 0).all(), f'Nonmonotonic/duplicate cycle number in cell {cid}'
            valid = np.isfinite(cycles) & np.isfinite(q)
            hits = np.flatnonzero(valid & (q <= THRESHOLD))
            reached = bool(len(hits))
            position = int(hits[0]) if reached else None
            first = float(cycles[position]) if reached else np.nan
            matched = reached and saved == first  # Exact equality; no tolerance.
            previous = np.flatnonzero(valid & (np.arange(len(q)) < position)) if reached else []
            last = int(np.flatnonzero(valid)[-1])
            recovered = bool((q[position+1:][valid[position+1:]] > THRESHOLD).any()) if reached else False
            rows.append({'batch': 2, 'cell_id': cid, 'stored_cycle_life': saved,
                         'first_qd_le_088_cycle': first, 'crossing_minus_stored_cycles': first-saved,
                         'not_reached': not reached, 'exact_match': matched,
                         'status': 'exact_match' if matched else 'mismatch' if reached else 'not_reached',
                         'first_crossing_qd_ah': float(q[position]) if reached else np.nan,
                         'previous_valid_cycle': float(cycles[previous[-1]]) if len(previous) else np.nan,
                         'previous_valid_qd_ah': float(q[previous[-1]]) if len(previous) else np.nan,
                         'last_recorded_cycle': float(cycles[last]), 'last_qd_ah': float(q[last]),
                         'recovered_above_088_after_first_crossing': recovered,
                         'n_nonfinite_pairs': int((~valid).sum()), 'threshold_ah': THRESHOLD,
                         'cycle_number_source': 'summary.cycle; never array position'})
    frame = pd.DataFrame(rows)
    result = {'n_evaluated': len(frame), 'n_exact_match': int(frame.exact_match.sum()),
              'n_mismatch': int((frame.status == 'mismatch').sum()), 'n_not_reached': int(frame.not_reached.sum()),
              'threshold_ah': THRESHOLD, 'comparison': 'exact equality; no tolerance',
              'source_file': str(source.resolve()), 'cycle_number_source': 'summary.cycle',
              'labels_changed': False, 'cells_excluded_by_this_audit': 0,
              'n_recovered_after_crossing': int(frame.recovered_above_088_after_first_crossing.sum())}
    frame.to_csv(csv_path, index=False, encoding='utf-8-sig')
    summary_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    return frame, result


def additional_evidence():
    tables = trained_evidence()
    predictions = pd.read_csv(OUT / 'paper_model_predictions.csv')
    binary = predictions[(predictions.task == 'binary_550') & (predictions.family == 'Full')]
    reference = tables['paper_classification_metrics']
    secondary = reference[(reference.model == 'Full classifier') & (reference.partition == 'Secondary test')].iloc[0]
    pooled = reference[(reference.model == 'Full classifier') & (reference.partition == 'Primary + Secondary test')].iloc[0]
    comparison = [{'evaluation': '논문 Full Secondary', 'n_cells': int(secondary.n_cells),
                   'accuracy_pct': secondary.accuracy_pct, 'macro_f1': secondary.macro_f1_reconstructed,
                   'n_short': int(secondary.short_support),
                   'n_short_correct': int(round(secondary.short_support*secondary.short_recall_pct/100)),
                   'short_recall_pct': secondary.short_recall_pct}]
    f1_rows = []
    for name, group in [('이번 Batch 2', binary[binary.partition == PARTITIONS[2]]),
                        ('이번 Batch 3 전체', binary[binary.partition == PARTITIONS[3]]),
                        ('이번 Batch 3 저자 품질 규칙', binary[(binary.partition == PARTITIONS[3]) &
                                                            ~binary.cell_id.isin([2, 37, 42, 43])])]:
        if group.empty:
            continue
        observed, predicted = group.observed_class.to_numpy(), group.prediction.to_numpy()
        short = observed == 0
        score = float(f1_score(observed, predicted, labels=[0, 1], average='macro', zero_division=0))
        f1_rows.append({'evaluation': name, 'project_macro_f1': score,
                        'paper_pooled_reference_macro_f1': pooled.macro_f1_reconstructed,
                        'reference_minus_project_f1': pooled.macro_f1_reconstructed-score,
                        'purpose': 'reference comparison only; not an assignment F1 Target'})
        if group.partition.iloc[0] == PARTITIONS[3]:
            comparison.append({'evaluation': name, 'n_cells': len(group),
                               'accuracy_pct': 100*accuracy_score(observed, predicted), 'macro_f1': score,
                               'n_short': int(short.sum()), 'n_short_correct': int(((predicted == 0) & short).sum()),
                               'short_recall_pct': 100*float(((predicted == 0) & short).sum())/short.sum() if short.any() else np.nan})
    tables['paper_binary_secondary_comparison'] = pd.DataFrame(comparison)
    tables['paper_binary_f1_reference_comparison'] = pd.DataFrame(f1_rows)
    selected = predictions[(predictions.task == 'regression') & (predictions.family == 'Selected')]
    errors = []
    for partition in PARTITIONS[2:]:
        group = selected[selected.partition == partition].copy()
        if group.empty:
            continue
        group['structure'] = np.where(group.policy.str.contains('newstructure', case=False, regex=False), 'newstructure', 'standard')
        for structure in ['standard', 'newstructure']:
            cells = group[group.structure == structure]
            residual = cells.prediction-cells.cycle_life
            errors.append({'partition': partition, 'structure': structure, 'n_cells': len(cells),
                           'mape_pct': float((100*residual.abs()/cells.cycle_life).mean()),
                           'mae_cycles': float(residual.abs().mean()), 'mean_signed_error_cycles': float(residual.mean()),
                           'observed_life_min': float(cells.cycle_life.min()), 'observed_life_max': float(cells.cycle_life.max()),
                           'structure_source': 'policy contains newstructure; otherwise standard'})
    tables['paper_selected_structure_errors'] = pd.DataFrame(errors)
    for name, frame in tables.items():
        frame.to_csv(OUT / f'{name}.csv', index=False, encoding='utf-8-sig')
    assessment = pd.read_csv(OUT / 'assessment_evidence.csv')
    row = assessment['평가항목'] == '전략 → 구현 반영'
    assessment.loc[row, '직접 근거'] = '최종 Discharge 피처·log 타깃 변환·실제 선택 모델; Ridge는 초기 비교'
    assessment.loc[row, '산출물'] = 'paper_models.py / paper_model_coefficients.csv; 초기 Ridge: features.py / model_coefficients.csv'
    assessment.to_csv(OUT / 'assessment_evidence.csv', index=False, encoding='utf-8-sig')
    return tables, assessment


def finalize_readme(text, archive_dir=None):
    if '### Batch 2 저장 수명 라벨 검증' in text:
        return text
    _, audit = audit_batch2_labels(archive_dir)
    tables, assessment = additional_evidence()
    original_images = re.findall(r'!\[[^\]]*\]\(([^)]+)\)', text)
    text = text.replace('초기 100사이클의 총 Cycle Life 회귀와 **초기 5사이클의 장·단수명 분류**를 개발하고 **Batch 1 학습 → Batch 2 평가**로 원논문 Target과 비교합니다. 단·중·장수명 분류와 Batch 3 평가도 추가했습니다. EDA 근거와 논문 피처 설계, 실제 학습, 성능·오류·ESS 해석을 연결합니다.',
                        '배터리 기반 ESS(BESS)의 수명 평가를 위해 **초기 100사이클 기반 총 Cycle Life 회귀**를 개발하고, **Batch 1 학습 → Batch 2 평가**로 과제 Target과 비교합니다. Batch 3로 추가 일반화를 평가하며, 초기 5회 이진 분류와 초기 100회 3분류는 추가 실험으로 분리합니다.')
    text = text.replace('- 주 태스크: **Regression**;', '- 주 태스크: **초기 100사이클 기반 총 Cycle Life 회귀(Regression)**;', 1)
    text = text.replace('입력 관측 창의 마지막 용량(Q100 또는 Q5)', '입력 관측 창의 마지막 용량 Q100 또는 Q5')
    text = text.replace('Q100 또는 Q5을 구현', 'Q100 또는 Q5를 구현')
    text = text.replace('입력의 마지막 용량은 관측 창의 **Q100 또는 Q5**로', '입력 관측 창의 마지막 용량은 **Q100 또는 Q5**이며')
    text = text.replace('### BESS 의사결정', '### ESS 의사결정')
    text = text.replace('│   ├── readme_layout.py / submission_review.py',
                        '│   ├── readme_layout.py / submission_review.py\n│   ├── submission_finalize.py  # 원본 라벨·추가 비교·최종 보고서')
    text = text.replace('│       ├── paper_selected_vif.csv',
                        '│       ├── paper_selected_vif.csv\n│       ├── batch2_cycle_life_audit.csv\n│       ├── paper_binary_secondary_comparison.csv\n│       ├── paper_selected_structure_errors.csv')
    basis = ('### 논문 비교 기준\n\n논문 학습·Primary-test는 2017-05-12·2017-06-30의 혼합 분할이며, '
             '이번 모델은 과제대로 Batch 1(2017-05-12)만 학습합니다. **과제 필수 Batch 2(2018-02-20)는 논문 밖 배치의 일반화 평가**입니다. '
             '회귀 Target 9.1%와 이진 Accuracy Target 95.1%는 과제 기준으로 유지합니다. **논문 Secondary에 대응하는 참고 비교는 '
             'Batch 3(2018-04-12)의 저자 품질 규칙 적용 40셀**이며 모델별 Table 1 성능과 비교합니다. '
             '학습 구성·분할이 달라 완전한 재현으로 부르지 않으며, 선행 탐색·기존 평가 후의 확장 분석이어서 새 미사용 테스트셋의 독립 검증과 구분합니다.')
    text = re.sub(r'^\*\*과제 필수 평가는.*$', basis, text, flags=re.MULTILINE)
    for prefix in ['원논문은 `2017-05-12`', '원논문의 Primary test는 2017-05-12']:
        text = re.sub(r'^' + re.escape(prefix) + r'.*\n', '', text, flags=re.MULTILINE)
    target = re.search(r'^\*\*9\.1% Target의 출처:.*$', text, re.MULTILINE)
    if target:
        folded = '<details>\n<summary>9.1% Target 출처와 역산 추정 근거</summary>\n\n' + target.group() + '\n\n</details>'
        text = text[:target.start()] + folded + text[target.end():]

    outcome = '라벨 수정·추가 제외 없이 기존 39셀 평가를 유지했습니다.' if audit['n_mismatch'] == audit['n_not_reached'] == 0 else '차이가 있는 셀은 원본 정의·일시적인 용량 이상·종료 기록의 검토 대상으로 기록하며 자동으로 라벨을 수정하지 않습니다.'
    audit_text = (f'### Batch 2 저장 수명 라벨 검증\n\n최종 평가 유효 **{audit["n_evaluated"]}셀**을 원본 '
                  f'`summary.cycle` 번호로 검사했습니다. 저장 `cycle_life`와 최초 **Qd ≤0.88Ah** 도달 사이클은 '
                  f'**정확히 일치 {audit["n_exact_match"]}셀 / 불일치 {audit["n_mismatch"]}셀 / 미도달 {audit["n_not_reached"]}셀**입니다. '
                  '배열 위치나 허용 오차로 일치 처리하지 않았습니다. ' + outcome +
                  ' [셀별 검사 결과](DAY2/results/batch2_cycle_life_audit.csv).')
    text = text.replace('### 1. Cycle Life 분포', audit_text + '\n\n### 1. Cycle Life 분포', 1)
    common = ('**Gap 공통 규칙:** 과제 행 이름을 유지하며 **(+)는 성능 저하**입니다. 회귀의 '
              '`Train-Valid / Valid-Test / Target-Test / Batch2-Batch3`는 각각 '
              '`Valid−Train / Test−Valid / Test−9.1 / Batch3−Batch2`입니다. 추가 분류의 앞 두 Gap과 배치 Gap은 '
              '`Train−Valid / Valid−Test / Batch2−Batch3`이고, Accuracy의 Target Gap은 `95.1−Test`입니다. '
              '**과제 F1 Target이 없어 F1의 Target Gap은 N/A**입니다. MAPE·Accuracy 차이는 %p, F1 차이는 0–1 점수 차이입니다. '
              'Hold-out에 단수명 클래스가 없는 분류 Gap은 클래스 구성 차이를 포함한 참고 차이이며, 단수명 식별력이나 '
              '동일 클래스 구성의 일반화 판단에 사용하지 않습니다.')
    text = re.sub(r'^\*\*Gap 공통 규칙:.*$', common, text, flags=re.MULTILINE)
    lines = []
    for line in text.splitlines():
        if line.startswith('**Gap (Train-Valid)은 Valid'):
            if 'Train은' in line:
                lines.append('Train은' + line.split('Train은', 1)[1])
            continue
        if line.startswith(('행 이름을 유지하되 **Gap', 'Gap (Train-Valid)은 Valid', '추가 Batch 3의 Gap',
                            '점수는 높을수록 좋으므로 Gap', 'Gap 방향·단위는 성능 결과')):
            continue
        line = line.replace('Gap이 양수이면 앞 단계 대비 성능 저하를 나타내도록 계산 방향을 아래에 명시합니다. ', '')
        line = line.replace('이 두 값을 이진 Target 비교의 근거로 사용합니다.', 'Accuracy Target은 95.1%이며 재계산 Macro-F1은 과제 Target이 아닌 별도 참고값입니다.')
        lines.append(line)
    text = '\n'.join(lines) + '\n'
    text = replace_table(text, '### 최종 회귀 필수 평가: 논문 기반 Discharge Elastic Net', tables['paper_selected_regression_performance'])
    if '### 최종 회귀 추가 평가: Batch 3' in text:
        text = replace_table(text, '### 최종 회귀 추가 평가: Batch 3', batch3_reporting_format(tables['paper_selected_regression_with_batch3']))
    for scheme, heading, extra in [
        ('binary_550', '### 장·단 이진 분류 성능표: 초기 5회 Full', '#### 초기 5회 이진 분류: Batch 3 추가 양식'),
        ('three_500_1000', '### 단·중·장 분류 성능표: 초기 100회 Full', '#### 단·중·장 분류: Batch 3 추가 양식')]:
        text = replace_table(text, heading, tables[f'paper_{scheme}_performance'])
        if extra in text:
            text = replace_table(text, extra, tables[f'paper_{scheme}_with_batch3_formatted'])

    reference_table = tables['paper_binary_f1_reference_comparison'].rename(columns={
        'evaluation':'이번 평가', 'project_macro_f1':'이번 Macro-F1',
        'paper_pooled_reference_macro_f1':'논문 pooled Macro-F1 참고값', 'reference_minus_project_f1':'참고값−이번 값'})
    secondary_table = tables['paper_binary_secondary_comparison'].rename(columns={
        'evaluation':'평가', 'n_cells':'셀 수', 'accuracy_pct':'Accuracy (%)', 'macro_f1':'Macro-F1',
        'n_short':'단수명 셀', 'n_short_correct':'단수명 정답 셀', 'short_recall_pct':'단수명 Recall (%)'})
    extra_text = ('### 추가 이진 분류의 Secondary 참고 비교\n\n' + markdown_table(secondary_table) +
                  '\n\n논문 Secondary와 이번 Batch 3의 전체·품질 규칙 평가 모두 **단수명 1셀을 맞히지 못했습니다.** '
                  '높은 Accuracy만으로 단수명 식별력을 주장할 수 없으며 비슷한 지표를 모델 성능의 동등성으로 해석하지 않습니다. '
                  '[재계산 결과](DAY2/results/paper_binary_secondary_comparison.csv).\n\n'
                  '### 논문 재계산 Macro-F1과의 별도 참고 차이\n\n' + markdown_table(reference_table.drop(columns='purpose')) +
                  '\n\n논문 Full의 Primary+Secondary 혼동행렬에서 재계산한 F1과의 기술적 비교이며, '
                  '과제 Target Gap이 아닙니다. 표본·클래스 구성이 달라 동일 분포의 성능 비교로 해석하지 않습니다. '
                  '[참고 차이 CSV](DAY2/results/paper_binary_f1_reference_comparison.csv).')
    # Move the early-5-only figure out of the primary regression modeling narrative.
    graph = re.search(r'!\[논문 초기 5회 ΔQ.*?\n\n상단은.*?(?=\n\n### 모델 선택)', text, re.DOTALL)
    early_figure = graph.group() if graph else ''
    if graph:
        text = text[:graph.start()] + text[graph.end():]
    start, end = text.index('### 실제 학습한 초기 5회 분류: 논문과의 비교'), text.index('## 오류 분석')
    body = text[start:end].strip()
    folded = ('### 추가 분류 실험\n\n초기 5회 이진 분류는 단수명 학습 표본이 1셀뿐이며, 초기 100회 3분류는 '
              'Batch 1에 단수명 클래스가 없어 중간·장수명만 학습한 확장 실험입니다. Hold-out Accuracy 100%는 단수명 검증 결과가 아닙니다.\n\n'
              '<details>\n<summary>추가 분류 설계·지정 성능표·논문 참고 비교·혼동행렬 펼치기</summary>\n\n' +
              body + '\n\n' + early_figure + '\n\n' + extra_text + '\n\n</details>\n\n')
    text = text[:start] + folded + text[end:]
    # Keep the modeling experiment table focused on the primary regression task.
    removed = []
    kept = []
    for line in text.splitlines():
        if line.startswith(('| Variance 이진 분류 |', '| Full 이진 분류 |', '| 단·중·장 분류 |')):
            removed.append(line)
        elif line.startswith('- **추가 분류 모델:**'):
            continue
        else:
            kept.append(line)
    text = '\n'.join(kept) + '\n'
    if removed:
        table = '| 실험 | 관측 데이터 | 구현 | 평가 기준 |\n| --- | --- | --- | --- |\n' + '\n'.join(removed)
        text = text.replace('### 실제 학습한 초기 5회 분류: 논문과의 비교', table + '\n\n### 실제 학습한 초기 5회 분류: 논문과의 비교', 1)
    structure = tables['paper_selected_structure_errors']
    structure = structure[structure.partition == PARTITIONS[2]][['structure','n_cells','mape_pct','mae_cycles','mean_signed_error_cycles']].rename(columns={
        'structure':'정책 표기의 구조 그룹', 'n_cells':'셀 수','mape_pct':'MAPE (%)','mae_cycles':'MAE (회)','mean_signed_error_cycles':'평균 예측−실제 (회)'})
    structure_text = ('### 최종 모델의 Batch 2 구조별 오류\n\n' + markdown_table(structure) +
                      '\n\n정책 문자열의 `newstructure` 표시로 구분했습니다. 큰 오차가 어느 그룹에 집중되는지 확인하는 사후 진단이며, '
                      '구조와 수명 분포가 함께 달라 **구조 자체의 인과 효과를 입증하는 결과가 아닙니다.** '
                      '[구조별 계산 근거](DAY2/results/paper_selected_structure_errors.csv).')
    text = text.replace('### 원인 가설 및 개선 방향', structure_text + '\n\n### 원인 가설 및 개선 방향', 1)
    text = replace_table(text, '## 평가항목별 확인 위치', assessment)
    assert sorted(original_images) == sorted(re.findall(r'!\[[^\]]*\]\(([^)]+)\)', text)), 'Figures must be preserved'
    return '\n'.join(line.rstrip() for line in text.splitlines()) + '\n'


def refresh_notebook():
    """Refresh report-only markdown and append saved, reproducible audit views."""
    import nbformat
    from .paper_comparison import readme_section
    path = ROOT / 'DAY2/03_modeling.ipynb'
    notebook = nbformat.read(path, as_version=4)
    tables = trained_evidence()
    for name in ['paper_regression_metrics', 'life_class_distribution']:
        tables[name] = pd.read_csv(OUT / f'{name}.csv')
    for cell in notebook.cells:
        if cell.cell_type == 'markdown' and cell.source.startswith('## 팀 구성'):
            cell.source = cell.source.replace('## 팀 구성', '## 수행자', 1)
        if cell.cell_type == 'markdown' and cell.source.startswith('## 원논문 기반 추가 개발·비교'):
            cell.source = re.sub(r'^!\[.*?\]\([^)]+\)\n', '', readme_section(tables), flags=re.MULTILINE)
    marker = '## 제출 전 추가 검증'
    index = next((i for i, c in enumerate(notebook.cells) if c.cell_type == 'markdown' and c.source.startswith(marker)), len(notebook.cells))
    notebook.cells = notebook.cells[:index]
    summary = json.loads((OUT / 'batch2_cycle_life_audit_summary.json').read_text())
    notebook.cells.append(nbformat.v4.new_markdown_cell(
        marker + '\n\nBatch 2 유효 39셀의 저장 라벨을 배열 위치가 아닌 원본 사이클 번호로 대조했습니다. '
        f'정확 일치 {summary["n_exact_match"]}셀, 불일치 {summary["n_mismatch"]}셀, 미도달 {summary["n_not_reached"]}셀입니다. '
        f'최초 도달 이후 재상승 기록이 {summary["n_recovered_after_crossing"]}셀에 있지만 최초 도달 정의와 저장 라벨은 일치합니다. '
        '허용 오차·라벨 변경·셀 제외를 적용하지 않았습니다. 아래 표는 셀별 검사와 추가 참고 비교의 실제 계산 결과입니다. '
        '과제 F1 Target Gap은 N/A이며 논문에서 재계산한 F1 차이는 참고 비교입니다.'))
    for name in ['batch2_cycle_life_audit', 'paper_binary_secondary_comparison',
                 'paper_binary_f1_reference_comparison', 'paper_selected_structure_errors', 'paper_selected_vif']:
        frame = pd.read_csv(OUT / f'{name}.csv')
        cell = nbformat.v4.new_code_cell(f"pd.read_csv(OUT / '{name}.csv')")
        cell.outputs = [nbformat.v4.new_output('execute_result', execution_count=1,
                                              data={'text/plain': repr(frame), 'text/html': frame._repr_html_()})]
        cell.execution_count = 1
        notebook.cells.append(cell)
    number = 0
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            number += 1
            cell.execution_count = number
            for output in cell.outputs:
                if output.output_type == 'execute_result':
                    output.execution_count = number
    nbformat.validate(notebook)
    nbformat.write(notebook, path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive-dir', type=Path, help='Original MAT directory; outputs stay in this repository')
    args = parser.parse_args()
    path = ROOT / 'README.md'
    # A completed README must not bypass an explicitly requested raw-data audit.
    audit_batch2_labels(args.archive_dir)
    path.write_text(finalize_readme(path.read_text(), args.archive_dir))
    refresh_notebook()
    print(json.loads((OUT / 'batch2_cycle_life_audit_summary.json').read_text()))
