"""Reproduce DAY 1 lifetime-group and charging-policy comparisons from MAT."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .train import OUT

GROUPS = ['short_lt500', 'middle_500_1000', 'long_gt1000']
LABELS = ['Short <500', 'Middle 500-1000', 'Long >1000']
COLORS = ['#d65b46', '#a78b36', '#2563eb']


def life_group(values):
    return np.where(values < 500, GROUPS[0], np.where(values <= 1000, GROUPS[1], GROUPS[2]))


def generate(eda, degradation, curves, policy):
    from .diagnostics import save_csv
    eda = eda.copy(); degradation = degradation.copy()
    eda['life_group'] = life_group(eda.cycle_life)
    degradation['life_group'] = life_group(degradation.cycle_life)
    proportions=[]; group_stats=[]; knee=[]; quantiles=[]; representatives=[]
    for batch, frame in eda.groupby('batch'):
        for group in GROUPS:
            subset=frame[frame.life_group==group]
            n=len(subset)
            proportions.append({'batch':int(batch),'life_group':group,'n_cells':n,
                                'batch_n_cells':len(frame),'proportion_pct':100*n/len(frame)})
            ds=degradation[(degradation.batch==batch)&(degradation.life_group==group)]
            group_stats.append({'batch':int(batch),'life_group':group,'n_cells':n,
                **{key+'_median':float(subset[key].median()) for key in ['cycle_life','log10_dq_variance','dq_mean','dq_min']},
                **{key+'_median':float(ds[key].median()) for key in ['early_slope_ah_per100','middle_slope_ah_per100','late_slope_ah_per100']}})
            if n:
                matrices=np.array([curves[(int(batch),int(cid))]['delta'] for cid in subset.cell_id])
                axis=curves[(int(batch),int(subset.cell_id.iloc[0]))]['axis']
                low,median,high=np.percentile(matrices,[25,50,75],axis=0)
                quantiles.extend({'batch':int(batch),'life_group':group,'n_cells':n,'voltage':float(v),
                                  'dq_q25':float(q1),'dq_median':float(q2),'dq_q75':float(q3)}
                                 for v,q1,q2,q3 in zip(axis,low,median,high))
    scopes=[('Pooled (descriptive)',degradation)]+[(f'Batch {int(b)}',g) for b,g in degradation.groupby('batch')]
    for scope,frame in scopes:
        for group in GROUPS:
            ds=frame[frame.life_group==group]
            accepted=ds[ds.shape_ok & ~ds.search_boundary]
            knee.append({'scope':scope,'life_group':group,'n_cells':len(ds),
                         'n_shape_ok':int(ds.shape_ok.sum()),'n_search_boundary':int(ds.search_boundary.sum()),
                         'n_interior_shape_ok':len(accepted),
                         'interior_candidate_min_cycle':float(accepted.knee_candidate_cycle.min()),
                         'interior_candidate_median_cycle':float(accepted.knee_candidate_cycle.median()),
                         'interior_candidate_max_cycle':float(accepted.knee_candidate_cycle.max()),
                         'interior_candidate_median_fraction_of_life':float(accepted.knee_fraction_of_life.median())})
    pooled=[]
    for group in GROUPS:
        subset=eda[eda.life_group==group]; ds=degradation[degradation.life_group==group]
        ordered=subset.sort_values(['cycle_life','batch','cell_id'])
        representative=ordered.iloc[(len(ordered)-1)//2]
        representatives.append({'life_group':group,'batch':int(representative.batch),
                                'cell_id':int(representative.cell_id),'cycle_life':float(representative.cycle_life),
                                'selection':'lower median of lifetime-sorted eligible group'})
        pooled.append({'life_group':group,'n_cells':len(subset),
                       **{key+'_median':float(ds[key].median()) for key in ['early_slope_ah_per100','middle_slope_ah_per100','late_slope_ah_per100']}})
    counts=pd.DataFrame(proportions); stats=pd.DataFrame(group_stats); knee=pd.DataFrame(knee)
    quantiles=pd.DataFrame(quantiles); pooled=pd.DataFrame(pooled); representatives=pd.DataFrame(representatives)
    for name,frame in [('life_group_proportions',counts),('life_group_statistics',stats),
                       ('knee_timing_summary',knee),('dq_life_group_curves',quantiles),
                       ('life_group_degradation',pooled),('life_group_representatives',representatives)]:
        save_csv(name,frame)
    fig,axes=plt.subplots(1,2,figsize=(12,4.8))
    for i,r in enumerate(representatives.itertuples()):
        curve=curves[(r.batch,r.cell_id)]; x,y=curve['cycle'],curve['capacity']
        mask=(x>=2)&np.isfinite(y)&(y>0)&(y<=1.2)
        axes[0].plot(x[mask],y[mask],color=COLORS[i],label=f'{LABELS[i]}: B{r.batch} cell {r.cell_id}, {r.cycle_life:g} cycles')
    axes[0].set(title='Representative cells: selected by median life',xlabel='Cycle',ylabel='Discharge capacity (Ah)')
    axes[0].legend(fontsize=8); axes[0].grid(alpha=.2)
    x=np.arange(3)
    for i,r in enumerate(pooled.itertuples()):
        values=[r.early_slope_ah_per100_median,r.middle_slope_ah_per100_median,r.late_slope_ah_per100_median]
        axes[1].plot(x,values,marker='o',color=COLORS[i],label=f'{LABELS[i]} (n={r.n_cells})')
    axes[1].set(title='Per-cell slope medians: descriptive only',ylabel='Capacity slope (Ah / 100 cycles)',
                xticks=x,xticklabels=['Early third','Middle third','Late third'])
    axes[1].legend(fontsize=8);axes[1].grid(alpha=.2)
    fig.tight_layout();fig.savefig(OUT/'07_life_groups_degradation.png',dpi=160);plt.close(fig)
    batches=sorted(eda.batch.unique())
    fig,axes=plt.subplots(1,len(batches),figsize=(5*len(batches),4.5),sharex=True,sharey=True,squeeze=False)
    for ax,batch in zip(axes.flat,batches):
        for i,group in enumerate(GROUPS):
            q=quantiles[(quantiles.batch==batch)&(quantiles.life_group==group)].sort_values('voltage')
            if q.empty: continue
            ax.plot(q.voltage,q.dq_median,color=COLORS[i],label=f'{LABELS[i]} (n={q.n_cells.iloc[0]})')
            ax.fill_between(q.voltage,q.dq_q25,q.dq_q75,color=COLORS[i],alpha=.15)
        ax.set(title=f'Batch {int(batch)}: median and IQR',xlabel='Common voltage (V)',ylabel='Q100(V) - Q10(V) (Ah)')
        ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(OUT/'08_dq_life_groups.png',dpi=160);plt.close(fig)
    b1=policy[policy.batch==1].sort_values('mean_life',ascending=True)
    fig,axes=plt.subplots(1,2,figsize=(14,7))
    labels=[f'{r.policy} (n={r.n_cells})' for r in b1.itertuples()]
    axes[0].barh(labels,b1.mean_life,color='#2563eb',alpha=.8)
    axes[0].set(title='Batch 1: mean life by charging policy',xlabel='Mean cycle life')
    for batch,g in eda.groupby('batch'):
        axes[1].scatter(g.ideal_equivalent_c,g.cycle_life,label=f'Batch {int(batch)}',alpha=.75)
    axes[1].set(title='Similar equivalent speed, different lifetimes',xlabel='Ideal equivalent C-rate (to 80% SOC)',ylabel='Cycle life')
    axes[1].legend();axes[1].grid(alpha=.2)
    fig.tight_layout();fig.savefig(OUT/'09_charging_policy_life.png',dpi=160);plt.close(fig)
    correlations=[]
    for batch,g in eda.groupby('batch'):
        correlations.append({'batch':int(batch),'n_valid':int(g.ideal_equivalent_c.notna().sum()),
                             'equivalent_c_min':float(g.ideal_equivalent_c.min()),
                             'equivalent_c_max':float(g.ideal_equivalent_c.max()),
                             'spearman_equivalent_c_life':g.ideal_equivalent_c.corr(g.cycle_life,method='spearman'),
                             'spearman_equivalent_c_life_rounded8':g.ideal_equivalent_c.round(8).corr(g.cycle_life,method='spearman')})
    save_csv('charging_rate_correlations',pd.DataFrame(correlations))
    print('EDA sample requirements generated: fixed DAY 1 groups; degradation/knee; delta-Q; policy means.')


def main():
    plt.switch_backend('Agg')
    from .diagnostics import raw_evidence
    features=pd.read_csv(OUT/'cell_features_all.csv')
    eda,degradation,curves=raw_evidence(features)
    policy=eda.groupby(['batch','policy']).agg(n_cells=('cell_id','size'),mean_life=('cycle_life','mean'),
                                              median_life=('cycle_life','median'),equivalent_c=('ideal_equivalent_c','mean')).reset_index()
    generate(eda,degradation,curves,policy)


def readme_section(distribution, core_vif, full_vif):
    """Write each requested EDA heading with computed evidence and a finding."""
    from .report import markdown_table
    read=lambda name:pd.read_csv(OUT/(name+'.csv'))
    names=dict(zip(GROUPS,['단수명 <500','중간 500–1,000','장수명 >1,000']))
    counts=read('life_group_proportions'); rates=read('life_group_degradation')
    statistics=read('life_group_statistics'); knee=read('knee_timing_summary')
    representatives=read('life_group_representatives')
    correlations=read('charging_rate_correlations'); policies=read('policy_summary')
    dist=distribution[['batch','n_cells','min','median','mean','std','max']].rename(columns={
        'batch':'Batch','n_cells':'유효 셀 수','min':'최소','median':'중앙값','mean':'평균','std':'표준편차','max':'최대'})
    ratio_rows=[]
    for batch,g in counts.groupby('batch'):
        row={'Batch':int(batch),'분모 (셀)':int(g.batch_n_cells.iloc[0])}
        for r in g.itertuples():row[names[r.life_group]]=f'{r.n_cells}셀 ({r.proportion_pct:.1f}%)'
        ratio_rows.append(row)
    slope_table=rates.rename(columns={'life_group':'수명 그룹','n_cells':'셀 수',
        'early_slope_ah_per100_median':'초기 기울기','middle_slope_ah_per100_median':'중기 기울기','late_slope_ah_per100_median':'말기 기울기'})
    slope_table['수명 그룹']=slope_table['수명 그룹'].map(names)
    slope_ratio=abs(rates.iloc[0].late_slope_ah_per100_median/rates.iloc[2].late_slope_ah_per100_median)
    candidate=knee[knee.scope=='Pooled (descriptive)'].copy()
    kt=pd.DataFrame({'수명 그룹':candidate.life_group.map(names),'셀 수':candidate.n_cells,
        '탐색 경계 후보':candidate.n_search_boundary,'내부·형태 조건 충족':candidate.n_interior_shape_ok,
        '후보 범위 (회)':[f'{a:.1f}–{b:.1f}' for a,b in zip(candidate.interior_candidate_min_cycle,candidate.interior_candidate_max_cycle)],
        '후보 중앙값 (회)':candidate.interior_candidate_median_cycle,
        '수명 대비 시점 중앙값':[f'{v*100:.1f}%' for v in candidate.interior_candidate_median_fraction_of_life]})
    dt=statistics[['batch','life_group','n_cells','log10_dq_variance_median','dq_min_median']].rename(columns={
        'batch':'Batch','life_group':'수명 그룹','n_cells':'셀 수','log10_dq_variance_median':'log10 ΔQ 분산 중앙값','dq_min_median':'ΔQ 최솟값 중앙값 (Ah)'})
    dt['수명 그룹']=dt['수명 그룹'].map(names)
    b2_short=statistics[(statistics.batch==2)&(statistics.life_group==GROUPS[0])].iloc[0]
    b2_long=statistics[(statistics.batch==2)&(statistics.life_group==GROUPS[2])].iloc[0]
    pt=policies[policies.batch==1][['policy','n_cells','equivalent_c','mean_life']].sort_values('mean_life',ascending=False).rename(columns={
        'policy':'Batch 1 충전 프로토콜','n_cells':'셀 수','equivalent_c':'등가 C-rate','mean_life':'평균 수명 (회)'})
    ct=correlations.rename(columns={'batch':'Batch','n_valid':'셀 수','equivalent_c_min':'최소 C-rate','equivalent_c_max':'최대 C-rate',
        'spearman_equivalent_c_life':'Spearman (원 계산)','spearman_equivalent_c_life_rounded8':'Spearman (동률 보정)'})
    representative_text=' / '.join(f"{names[r.life_group]}: Batch {r.batch} 셀 {r.cell_id}, {r.cycle_life:.0f}회" for r in representatives.itertuples())
    batch3_description = 'Batch 3은 더 넓은 범위로 분포하며 장수명 비중이 큽니다. ' if 3 in distribution.batch.values else ''
    return f"""## EDA

DAY 1과 같은 **단수명 <500회 / 중간 500–1,000회 / 장수명 >1,000회** 기준을 유지합니다. 분모는 모델링에 사용할 유효 라벨 셀이며 알려진 불완전·결측 라벨은 제외했습니다. 그룹이 없는 배치는 0셀로 표시하고 경계를 바꾸어 채우지 않습니다. 이 그룹은 설명·오류 분석용이며 회귀 입력이나 분류 모델의 학습 라벨이 아닙니다.

### 1. Cycle Life 분포

수명 통계의 단위는 사이클 수입니다.

{markdown_table(dist)}

{markdown_table(pd.DataFrame(ratio_rows))}

![Batch 1과 Batch 2의 수명 분포 및 ΔQ 피처 관계](DAY2/results/03_batch_shift.png)

Batch 1은 중간 수명에 집중되고 <500회 셀이 없습니다. Batch 2는 28/39셀(71.8%)이 <500회이고 소수의 긴 수명 셀 때문에 오른쪽 꼬리가 나타납니다. {batch3_description}위 Batch 2 그래프는 고정 모델 평가 후의 사후 진단으로 후보 선택에 사용하지 않았습니다.

**핵심 발견:** 학습 배치에 없는 단수명 구간이 외부 Batch 2의 대부분을 차지하므로 단수명 개발 데이터 확보와 배치 간 일반화 검증이 중요합니다.

### 2. 열화 곡선 분석

![수명 그룹별 대표 셀의 용량 곡선 및 초기·중기·말기 기울기](DAY2/results/07_life_groups_degradation.png)

왼쪽은 수명순으로 정렬한 각 그룹의 아래쪽 중앙 셀을 선택한 실제 곡선입니다. 곡선 모양을 보고 대표 셀을 고르지 않았습니다. {representative_text}. 오른쪽과 아래 표는 셀별 관측 기간을 3등분한 기울기의 그룹 중앙값으로 **Ah/100사이클** 단위입니다. 대표 셀 곡선과 전체 그룹 요약은 구분합니다.

{markdown_table(slope_table)}

단수명 말기 감소 속도의 절댓값은 장수명의 약 **{slope_ratio:.2f}배**입니다. 통합 그룹 비교에서 관측된 차이이며 단수명 28셀이 모두 Batch 2여서 수명·배치 효과를 분리할 수 없습니다. 초기·중기·말기는 같은 절대 사이클 구간이 아니라 각 셀의 기록 진행률 구간입니다.

#### Knee point 존재 여부와 시점

100회 이후 용량에 연속 구간선형 모델을 적합하고 관측 기간의 20–80% 범위에서 knee 후보를 탐색했습니다. 후반 기울기가 음수이고 이전보다 감소가 빠르며 크기가 2배 이상인 경우를 형태 조건으로 표시했습니다. 총 {int(candidate.n_cells.sum())}셀 중 {int(candidate.n_shape_ok.sum())}셀이 조건을 충족했지만 **{int(candidate.n_search_boundary.sum())}셀은 탐색 경계**에 걸렸습니다. 아래 시점은 경계에 걸리지 않고 형태 조건을 충족한 후보만의 요약입니다.

{markdown_table(kt)}

**핵심 발견:** 장·단수명 모두 후반 열화 가속이 관측되며 단수명 그룹의 말기 감소가 더 가파릅니다. Knee 후보 시점은 셀마다 달라지고 경계 후보가 있어 정확한 열화 시작점의 확정으로 해석하지 않습니다. 전체 곡선·말기 기울기·knee는 미래 관측 설명용으로 모델 입력에서 제외합니다.

### 3. ΔQ(V) 곡선 분석

`ΔQ(V) = Q100(V) − Q10(V)`를 공통 2.0–3.5V·1,000점 전압축에서 계산했습니다. 배치별 곡선의 중앙값과 IQR은 다음과 같습니다.

![배치별 Cycle 100−10 ΔQ 곡선과 설명용 열화 속도](DAY2/results/04_eda_curve_evidence.png)

![배치 내 수명 그룹별 ΔQ 곡선의 중앙값과 IQR](DAY2/results/08_dq_life_groups.png)

두 번째 그림은 배치마다 장·중간·단수명 곡선을 비교합니다. 실제로 없는 수명 그룹은 그리지 않습니다. 음영은 셀 간 IQR로 평균의 신뢰구간이 아닙니다.

{markdown_table(dt)}

같은 Batch 2에서 단수명 {int(b2_short.n_cells)}셀의 log 분산 중앙값은 {b2_short.log10_dq_variance_median:.3f}, 장수명 {int(b2_long.n_cells)}셀은 {b2_long.log10_dq_variance_median:.3f}입니다. 단수명에서 곡선 변화의 크기가 더 크지만 장수명 표본이 3셀뿐이므로 일반화에 주의합니다. Batch 1의 ΔQ log 분산–log 수명 Pearson은 −0.844입니다.

**핵심 발견:** 초기 총 용량에서 뚜렷하지 않은 변화가 ΔQ 곡선에 드러나므로 log 분산을 핵심 피처로 구현했습니다. 분산은 곡선의 상수 원점 이동에 불변이지만 평균·최솟값은 그렇지 않아, 배치 간 원점·절차 차이를 모두 교정했다고 해석하지 않습니다. 단수명·장수명 곡선 차이가 화학적 원인이나 예측 정확도를 직접 입증하지도 않습니다.

### 4. 충전 속도(C-rate)와 수명의 관계

![Batch 1 프로토콜별 평균 수명 및 배치별 등가 C-rate와 수명](DAY2/results/09_charging_policy_life.png)

첫 그림과 아래 표는 **학습 Batch 1의 모든 충전 프로토콜**을 셀 수와 함께 비교합니다. 두 단계 정책은 80% SOC까지의 이상적 충전 시간을 환산한 등가 C-rate로 요약했으며 실제 측정 충전 시간과는 다릅니다. 전체 배치의 프로토콜별 평균·중앙값은 [policy_summary.csv](DAY2/results/policy_summary.csv)에 있습니다.

{markdown_table(pt)}

{markdown_table(ct)}

표의 동률 보정은 부동소수점 계산의 미세한 차이로 같은 C-rate의 순위가 달라지는 것을 막기 위해 소수 8자리로 반올림한 민감도 비교입니다. 외부 평가 배치의 등가 C-rate는 거의 4.8C에 집중되어 있어 상관 부호만으로 속도 효과가 뒤집혔다고 해석하기 어렵습니다.

**핵심 발견:** Batch 1에서 등가 C-rate–수명 Spearman은 약 −0.44이지만 거의 같은 등가 속도에서도 프로토콜과 배치별 수명이 다릅니다. 속도 하나로 수명을 설명하거나 빠른 충전의 인과 효과를 확정할 수 없습니다. Batch 1 프로토콜별 1–2셀의 평균 차이는 통계적 유의성을 입증한 결과가 아니며, 같은 정책이 train·valid에 겹치지 않도록 정책 단위로 검증했습니다.

### 5. 추가 확인: 피처 중복·전이 안정성과 품질

![통합 탐색 상관행렬 및 배치별 피처와 수명의 Spearman 상관](DAY2/results/05_feature_stability.png)

ΔQ 요약값 사이의 중복이 크고 보조 피처는 배치별 상관 부호가 달라질 수 있습니다. Batch 1 전체 10개 피처 최대 VIF {full_vif:.2f}를 대표 5개로 줄이면 {core_vif:.2f}입니다. 통합 상관행렬은 탐색용이며 실제 후보 선택은 Batch 1 개발 CV로 제한합니다. 상관 변화 자체를 오차의 인과 원인으로 단정하지 않습니다.

**핵심 발견:** 중복 피처를 줄이고 포함·제외 비교와 정규화를 사용하되 실제 예측 개선은 CV로 검증해야 합니다. IR 0 이하·비유한 값과 초기 용량 범위 이상을 점검하고 제외·대치 기준과 유효 표본 수를 기록했습니다. 상세 계산은 `eda_distribution.csv`, `life_group_statistics.csv`, `knee_timing_summary.csv`, `eda_vif.csv`, `cell_audit_all.csv` 및 [DAY 2 노트북](DAY2/03_modeling.ipynb)에 있습니다. [기존 통합 EDA](30-ESSHealth-scratch.ipynb)는 DAY 1 탐색 기록입니다.

### EDA → 모델 전략 연결

| 관측 근거 | 모델 설계·구현 | 검증 근거 |
| --- | --- | --- |
| 연속 수명과 초기 ΔQ 신호 | 총수명 회귀·ΔQ log 분산 핵심 입력 | 단일 피처 기준선 및 후보 CV 비교 |
| 후반 열화와 knee는 미래 관측 | 100회 이내 피처만 사용 | 미래 사이클 값을 바꾸어도 피처가 같은지 검사 |
| ΔQ 요약값의 중복 | 대표 5개 피처의 16개 부분집합 비교 | `feature_ablation.csv`, 학습 배치 VIF |
| 보조 피처의 배치별 상관 변화 | 포함·제외 효과를 Batch 1에서 비교 | `paired_ablation.csv`, 피처별 배치 상관 |
| 작은 표본과 정책별 실험 | 정규화 선형 모델·정책별 분할 | Nested-CV 및 정책 Hold-out |
| Batch 1의 단수명 부재 | 외부 배치의 과대예측 편향 점검 | Batch 2 기준선·부호 오차·수명 구간 분석 |

"""


if __name__=='__main__':main()
