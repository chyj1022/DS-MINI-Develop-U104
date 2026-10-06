"""Generate the README in the supplied order and a result-filled notebook."""
import base64
import contextlib
import io
import json
from pathlib import Path
import nbformat
import numpy as np
import pandas as pd
from .preprocess import ROOT
from .features import FEATURE_SETS

OUT = ROOT / "DAY2/results"


def markdown_table(frame):
    columns = list(frame.columns)
    def fmt(value):
        if isinstance(value, (float, np.floating)):
            return f"{value:.3f}" if np.isfinite(value) else "—"
        return str(value).replace("|", "/").replace("\n", " ")
    return "| " + " | ".join(columns) + " |\n| " + " | ".join(["---"]*len(columns)) + " |\n" + "\n".join("| " + " | ".join(fmt(v) for v in row) + " |" for row in frame.itertuples(index=False,name=None))


def batch3_reporting_format(performance):
    """Match the assignment's four-column additional-batch layout."""
    rows=[]
    for _,row in performance.iterrows():
        label=row['구분']; is_gap=label.startswith('Gap (')
        if label=='Gap (Target-Test, Batch 3)': label='Gap (Target-Test)'
        rows.append({'구분': '' if is_gap else label,
                     '비교 항목': label if is_gap else '',
                     'MAPE (%)': row['MAPE (%)'], '비고': row['비고']})
    return pd.DataFrame(rows)


def build_report(frozen, scores, performance, comparison, baselines, errors, features, subgroups):
    # Preserve assignment row names while clarifying the meaning of Train.
    performance.loc[performance['구분'] == 'Train (Batch 1 CV)', '비고'] = '개발 Nested-CV (29셀); 외부 fold 검증오차 평균'
    performance.to_csv(OUT / 'model_performance.csv', index=False, encoding='utf-8-sig')
    extended_path = OUT / 'model_performance_with_batch3.csv'
    if extended_path.exists():
        extended = pd.read_csv(extended_path)
        extended.loc[extended['구분'] == 'Train (Batch 1 CV)', '비고'] = performance.iloc[0]['비고']
        extended.to_csv(extended_path, index=False, encoding='utf-8-sig')
    config = frozen["selected_config"]
    b1, b2 = features[features.batch==1], features[features.batch==2]
    stats = pd.DataFrame([{"Batch":name,"셀 수":len(f),"최소":f.cycle_life.min(),"중앙값":f.cycle_life.median(),"최대":f.cycle_life.max(),"<500":int((f.cycle_life<500).sum()),">1000":int((f.cycle_life>1000).sum())} for name,f in [("Batch 1",b1),("Batch 2",b2)]])
    winner = comparison.iloc[0]
    ablation = comparison[comparison.model.isin(["Ridge","ElasticNet"])].groupby("feature_set",sort=False).head(1).copy()
    ablation["features"] = ablation.feature_set.map(lambda key:", ".join(FEATURE_SETS[key]))
    ablation["n_features"] = ablation.feature_set.map(lambda key:len(FEATURE_SETS[key]))
    ablation = ablation[["feature_set","n_features","features","model","cv_mean_mape_pct"]].sort_values("cv_mean_mape_pct")
    ablation.to_csv(OUT / "feature_ablation.csv",index=False,encoding="utf-8-sig")
    feature_definitions = pd.DataFrame([
        {"feature":"log10_dq_variance","의미":"ΔQ(V) 변화 분산의 log10","window":"100회 − 10회","unit":"log10(Ah²)"},
        {"feature":"qd_slope_2_100","의미":"방전 용량 기울기","window":"2–100회","unit":"Ah/cycle"},
        {"feature":"charge_time_mean_2_6","의미":"초기 평균 충전 시간","window":"2–6회","unit":"min"},
        {"feature":"ir_change_100_2","의미":"내부저항 변화량","window":"100회 − 2회","unit":"Ω"},
        {"feature":"tavg_mean_2_100","의미":"초기 평균 온도","window":"2–100회","unit":"°C"},
    ])
    feature_definitions["selected"] = feature_definitions.feature.isin(frozen["features"])
    feature_definitions.to_csv(OUT / "feature_definitions.csv",index=False,encoding="utf-8-sig")
    correlations = pd.DataFrame([{"feature":f, "batch1_spearman":b1[f].corr(b1.cycle_life,method="spearman"), "batch2_spearman":b2[f].corr(b2.cycle_life,method="spearman")} for f in frozen["features"]])
    correlations.to_csv(OUT / "feature_batch_correlations.csv",index=False,encoding="utf-8-sig")
    top = comparison.groupby("model",sort=False).head(1)[["model","feature_set","alpha","l1_ratio","cv_mean_mape_pct","cv_std_mape_pct"]]
    top_errors = errors.head(5)[["cell_id","policy","cycle_life","predicted_cycle_life","ape_pct"]]
    strategy = """| EDA에서 확인한 사실 | 전략과 구현 | 검증 방법 |
| --- | --- | --- |
| ΔQ log 분산–log 수명 Pearson r=-0.844 | `log10 Var(Q100−Q10)`를 모든 후보에 포함 | 단일 피처 Linear 기준선과 비교 |
| ΔQ 분산·최솟값·평균의 상관 절댓값 ≥0.97 | 중복 피처를 동시에 넣지 않음 | 대표 5개 피처의 16개 부분집합 비교 |
| 용량 기울기 r=0.552, 초기 충전 시간 r=0.575 | 기울기(2–100회)와 평균 충전 시간(2–6회) 추가 | 개발 CV에서 추가 피처 효과 비교 |
| 온도·IR의 직접 관계가 약하고 센서 영향 가능 | 보조 피처로 포함·제외 | 대표 피처의 모든 부분집합 비교 |
| Batch 1 유효 표본 36셀, 정책 20개 | 소수 피처 + Ridge/Elastic Net 정규화 | 프로토콜 단위 nested GroupKFold |
| Batch 1에는 <500회 셀이 없음 | 이진 분류 대신 연속 수명 회귀 선택 | 원 단위 MAPE·MAE·RMSE |
| 조기 종료·후속 기록 10셀의 목표값 불완전 | 원본은 보존하고 10셀 제외 | 셀별 audit 및 DAY 1 분할표 대조 |"""
    below = int((b2.cycle_life < b1.cycle_life.min()).sum())
    test_mape = float(scores.iloc[2].mape_pct)
    cv_mape = float(scores.iloc[0].mape_pct)
    valid_mape = float(scores.iloc[1].mape_pct)
    summary_gap = f"Batch 2 MAPE는 **{test_mape:.3f}%**, 과제 Target 9.1%와의 차이는 **{test_mape-9.1:+.3f}%p**입니다. Valid−Train은 {valid_mape-cv_mape:+.3f}%p, Test−Valid는 {test_mape-valid_mape:+.3f}%p입니다. 작은 표본의 정책 분할과 CV fold 변동을 함께 고려해야 하며, Gap 한 값만으로 과적합을 확정할 수 없습니다."
    error_text = f"""가장 큰 오차 5셀은 아래와 같습니다. 예측값과 실제값의 차이를 원 사이클 단위로 복원해 분석했습니다. 상위 5셀은 모두 실제 수명 500회 미만이며, 모두 과대예측했습니다.

{markdown_table(top_errors)}

{markdown_table(subgroups)}

Batch 2에서 **{below}/{len(b2)}셀**의 수명이 Batch 1 최소값({b1.cycle_life.min():.0f}회)보다 짧습니다. Batch 1만 본 모델이 짧은 수명 범위로 외삽해야 한다는 점이 주요 원인 가설입니다. `newstructure`는 데이터의 정책 문자열에 표시된 그룹이며 표의 성능은 사후 진단입니다. `standard_structure`와 `below_batch1_life_min`은 정확히 같은 30셀로, 두 행은 독립적인 원인 증거가 아닙니다. 구조와 수명 범위의 효과를 분리하거나 인과관계로 단정할 수 없습니다. 잔차 부호가 양수이면 수명 과대예측이므로 교체 계획에서 특히 주의해야 합니다.

선택된 피처의 배치별 상관은 아래와 같습니다. 초기 충전 시간의 상관 부호가 배치에 따라 달라진다는 DAY 1 관찰과 일치합니다. 이는 보조 피처 전이의 위험을 설명하는 진단이며 오차의 인과 원인을 확정하지 않습니다.

{markdown_table(correlations)}

개선 방향: 새 개발 데이터에서 단수명 셀을 확보하고, 구조·온도·충전 조건의 영향을 독립적으로 검증합니다. 현재 Batch 2를 보고 피처나 파라미터를 재선택하지 않았으며, 개선 모델은 별도 미사용 배치로 평가해야 합니다."""
    domain = """초기 100사이클의 신호로 총수명을 추정해 셀 입고 선별, 점검 우선순위, 교체 예산 계획의 보조 지표로 활용할 수 있습니다. 이 모델의 목표는 총 Cycle Life이며 현재 시점의 RUL이나 화재 위험을 직접 예측하는 모델은 아닙니다. 과대예측은 교체 지연으로 이어질 수 있으므로 운영에서는 예측 불확실성과 보수적인 의사결정 기준이 필요합니다.

한계: 실험실 LFP/graphite 단일 셀과 특정 급속 충전 조건, 36셀 학습, Batch 1의 단수명 부재, 관측 종료 기준 차이, 세 배치 전체를 확인한 선행 EDA가 있습니다. 초기 용량 잡음은 범위 검사로 제외하고 센서 결측은 학습 fold 중앙값으로 대치하지만, 실 BESS의 달력 열화·온도 변화·부분 충방전·셀 불균형은 별도 검증이 필요합니다. 배포 전에는 현장 데이터와 다른 제조 배치, 프로토콜별 오차, 예측 구간, 입력 분포 변화 및 모니터링 기준을 검증해야 합니다."""
    evaluation = """| 평가항목 | 배점 | 확인할 구현·산출물 |
| --- | --- | --- |
| EDA | 50 | 30-ESSHealth-scratch.ipynb, DAY1 최종 보고서, 셀 audit·상관·분포·열화 |
| EDA → 전략 연결성 | 30 | 위 EDA→구현 연결표, 피처 세트별 CV 비교 |
| 모델링 전략 수립 | 20 | 회귀 선택 근거, 단일 피처 기준선, 소표본 정규화 |
| 전략 → 구현 반영 | 20 | src/features.py, CV 후보 비교 및 최종 계수 |
| Pipeline 개발 | 40 | src/preprocess.py·train.py, 정책 분할, fold 내부 대치·표준화, 고정 모델 저장 |
| 성능 리포팅 및 해석 | 20 | model_performance.csv, metrics.csv, Target Gap·조건 차이 설명 |
| 분석 결과 해석 | 20 | error_analysis.csv, subgroup_metrics.csv, 잔차·배치 비교 그래프, ESS 활용·한계 |"""
    nb = nbformat.v4.new_notebook()
    cells = [nbformat.v4.new_markdown_cell("# DAY 2 · 모델 개발 및 평가\n\nBatch 1 학습 → Batch 2 평가. 회귀 전략과 성능·오류·ESS 해석을 평가항목에 맞춰 연결했습니다. 이 노트북에는 실행으로 생성한 결과를 포함합니다.")]
    setup = "from pathlib import Path\nimport pandas as pd\nimport json\nROOT = Path.cwd()\nif ROOT.name == 'DAY2':\n    ROOT = ROOT.parent\nassert (ROOT / 'DAY2/results/frozen_config.json').exists(), '프로젝트 루트 또는 DAY2에서 실행하세요'\nOUT = ROOT / 'DAY2/results'\nprint('결과 경로:', OUT)"
    env={}
    def code(source):
        stream=io.StringIO()
        with contextlib.redirect_stdout(stream):
            exec(source,env)
        c=nbformat.v4.new_code_cell(source,execution_count=sum(x.cell_type=='code' for x in cells)+1)
        c.outputs=[nbformat.v4.new_output("stream",name="stdout",text=stream.getvalue())]
        cells.append(c)
    def md(text): cells.append(nbformat.v4.new_markdown_cell(text))
    code(setup)
    md("## 1. EDA → 모델 전략\n\n"+strategy+"\n\nBatch 1 최소 수명 534회, <500회 셀 0개이므로 회귀를 선택합니다. 후반 knee·최종 용량·관측 길이는 입력에서 제외합니다. DAY 1이 세 배치 전체를 본 탐색이라는 점은 내부 검증 해석의 한계입니다.")
    md("## 2. 피처·전처리·분할\n\n초기 100회까지의 방전곡선·용량 기울기·충전 시간·IR·온도만 사용합니다. 공통 전압축을 검사하고, fold 안에서 결측 대치·표준화합니다. 개발 29셀과 검증 7셀은 기존 정책 단위 분할을 유지합니다. Batch 1의 10셀은 불완전 목표값, Batch 2의 8셀은 결측 목표값 때문에 제외합니다.")
    code("audit = pd.read_csv(OUT / 'cell_audit.csv')\nprint(audit.groupby(['batch','included','reason']).size().to_string())\nmanifest = pd.read_csv(OUT / 'split_manifest.csv')\nprint(manifest.to_string(index=False))")
    md("## 3. 후보 비교 및 모델 선택\n\n후보 선택은 개발 29셀의 정책 단위 4-fold CV에서만 수행합니다. nested CV의 외부 fold는 모델·피처·파라미터 선택에 사용하지 않습니다. 과제의 Train 행은 **개발 Nested-CV** 외부 fold 검증오차 평균이며 학습 데이터에 적합한 오차가 아닙니다.")
    code("print(json.dumps(json.loads((OUT / 'frozen_config.json').read_text()), ensure_ascii=False, indent=2))\ncv = pd.read_csv(OUT / 'cv_results.csv')\nprint(cv.groupby('model',sort=False).head(1).to_string(index=False))\nprint(pd.read_csv(OUT / 'nested_cv_folds.csv').to_string(index=False))\nprint(pd.read_csv(OUT / 'baseline_comparison.csv').to_string(index=False))")
    code("print(pd.read_csv(OUT / 'feature_definitions.csv').to_string(index=False))\nprint(pd.read_csv(OUT / 'feature_ablation.csv').to_string(index=False))")
    def picture(filename, title):
        md("### "+title)
        c=nbformat.v4.new_code_cell("import matplotlib.pyplot as plt\nplt.figure(figsize=(12, 5))\nplt.imshow(plt.imread(OUT / '"+filename+"'))\nplt.axis('off')\nplt.show()",execution_count=sum(x.cell_type=='code' for x in cells)+1)
        c.outputs=[nbformat.v4.new_output("display_data",data={"image/png":base64.b64encode((OUT/filename).read_bytes()).decode()},metadata={})]
        cells.append(c)
    picture("01_model_comparison.png","CV 후보 비교")
    md("## 4. 성능 리포팅 및 Gap 해석\n\n"+summary_gap+"\n\nGap은 각각 Valid−Train, Test−Valid, Test−9.1이고 단위는 %p입니다. Valid는 29셀 모델, Test는 고정 설정을 36셀로 재학습한 모델입니다. 원논문 2017-06-30과 과제 지정 Batch 2 2018-02-20은 다르며 Target 비교는 동일 조건 재현이 아닙니다.")
    code("print(pd.read_csv(OUT / 'model_performance.csv').to_string(index=False))\nprint(pd.read_csv(OUT / 'metrics.csv').to_string(index=False))")
    md("과제 양식의 행 이름을 유지하되 **Gap (Train-Valid)은 Valid − Train으로 계산**합니다. Gap (Valid-Test)은 Test − Valid, Gap (Target-Test)은 Test − 9.1이며 모두 %p입니다. Train은 학습 오차가 아니라 **개발 Nested-CV 검증오차**입니다.")
    picture("02_predictions_residuals.png","원 단위 예측과 Batch 2 잔차")
    md("## 5. 오류 분석\n\n"+error_text)
    picture("03_batch_shift.png","고정 모델 평가 후의 배치 차이 분석")
    md("## 6. ESS 도메인 해석\n\n"+domain)
    md("## 7. 평가항목 대응\n\n"+evaluation)
    md("## 재학습 및 검증\n\n프로젝트 루트 터미널에서 `python -m src.train`, `python -m src.verify_results`를 실행하면 결과를 재생성하고 분할·성능표·모델 저장·미래 정보 배제를 확인합니다. 노트북에는 원본 피처 추출과 고정 설정의 재학습·평가 셀도 포함하며 전체 후보 탐색은 위 명령으로 수행합니다.")
    md("## 참고문헌\n\n- Severson et al. (2019). Data-driven prediction of battery cycle life before capacity degradation. *Nature Energy*, 4, 383–391. [원논문](https://www.nature.com/articles/s41560-019-0356-8).")
    md("## 수행자\n\n- 최유정(울산 3반 U104), 개인 수행: EDA, 모델 전략 수립, 피처 엔지니어링, 파이프라인·모델 개발, Batch 2·3 성능 평가, 오류 분석 및 보고서 작성 전 과정.")
    nb.cells=cells
    nb.metadata={"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python","version":frozen["python"]}}
    nbformat.validate(nb)
    nbformat.write(nb, ROOT / "DAY2/03_modeling.ipynb")
    from .evidence import enhance_report
    enhance_report()


def main():
    import matplotlib.pyplot as plt
    plt.switch_backend("Agg")
    """Refresh narrative/plots from frozen results without fitting or selecting."""
    from .train import plot_results
    frozen = json.loads((OUT / "frozen_config.json").read_text())
    read = lambda name:pd.read_csv(OUT / (name+".csv"))
    features=read("cell_features")
    comparison=read("cv_results")
    plot_results(read("predictions"), comparison, features[features.batch==1], features[features.batch==2])
    build_report(frozen,read("metrics"),read("model_performance"),comparison,read("baseline_comparison"),read("error_analysis"),features,read("subgroup_metrics"))
    print("README and notebook refreshed from frozen results")


if __name__=="__main__": main()
