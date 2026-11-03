"""Independent Brier audit: scalar math.fsum, reference-derived labels/masks.

Does not import the reporting or simulation code and does not change RNG state.
Baseline probabilities are aligned to exactly the same record/origin population.
"""
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'brier_audit';OUT.mkdir(exist_ok=True)
HORIZONS=[1,2,3,5]
METHODS={
    'clock_gate':'纯峰时钟（0/1硬规则）',
    'clock_constant':'峰时钟＋常数风险',
    'age_survival':'峰间隔生存模型',
    'age_logistic_ungated':'仅峰龄概率模型',
    'age_logistic_gated':'峰龄概率＋时钟门控（补充）',
}


def read(path):
    with Path(path).open(newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))


def write(path,data):
    fields=list(dict.fromkeys(k for r in data for k in r))
    with Path(path).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fields);w.writeheader();w.writerows(data)


def score(method,horizon,items,population='horizon_followup'):
    # items: record, probability, independent target, independently rebuilt score flag
    use=[(rec,p,y) for rec,p,y,keep in items if keep]
    assert use and all(math.isfinite(p) and 0<=p<=1 and y in (0,1) for rec,p,y in use)
    error=[(p-y)**2 for rec,p,y in use]
    positive=[(p-y)**2 for rec,p,y in use if y==1]
    negative=[(p-y)**2 for rec,p,y in use if y==0]
    bs=math.fsum(error)/len(use)
    per=defaultdict(list)
    for rec,p,y in use:per[rec].append((p-y)**2)
    macro=math.fsum(math.fsum(v)/len(v) for v in per.values())/len(per)
    assert abs(bs-(math.fsum(positive)+math.fsum(negative))/len(use))<1e-14
    return dict(method=method,horizon_s=horizon,population=population,records=len(per),
                origins=len(use),positive_origins=len(positive),negative_origins=len(negative),
                squared_error_sum=math.fsum(error),brier=bs,
                positive_brier=math.fsum(positive)/len(positive),negative_brier=math.fsum(negative)/len(negative),
                macro_record_brier=macro,
                target_prevalence=len(positive)/len(use))


def main():
    origins=read(ROOT/'results/origins.csv')
    group=defaultdict(list)
    for r in origins:group[(r['record'],r['variant'])].append(r)
    references={};raw_ends={}
    for record,_ in group:
        if record in references:continue
        references[record]=[float(r['onset']) for r in read(ROOT/'source_snapshot/records'/record/'reference_events.csv')]
        with np.load(ROOT/'reference_snapshot/data/records'/(record+'.npz')) as z:
            raw_ends[record]=float(z['t'][-1])
    results=[];mismatches=[];population={};checks=[]
    for variant in ['zero_noise','learned_seed_2026']:
        for H in HORIZONS:
            all_items=[];common_items=[]
            for (record,v),data in sorted(group.items()):
                if v!=variant:continue
                path=ROOT/'results/records'/record/(variant+'_first_onsets.npz')
                with np.load(path) as z:
                    first=z['first_onset_offsets'];time_saved=z['origin_time']
                assert len(first)==len(data)
                for index,r in enumerate(data):
                    time=float(r['origin_time']);prob=float(r[f'risk_{H}s'])
                    assert abs(time-time_saved[index])<1e-9
                    target=int(any(time+1e-9<onset<=time+H+1e-9 for onset in references[record]))
                    complete=time+H<=raw_ends[record]+1e-9
                    common=time+5<=raw_ends[record]+1e-9
                    path_probability=float(np.count_nonzero(first[index]<=H+1e-9)/first.shape[1])
                    assert prob==path_probability,(record,variant,H,index)
                    if target!=int(r[f'label_{H}s']) or complete!=bool(int(r[f'complete_{H}s'])):
                        mismatches.append(dict(record=record,variant=variant,horizon=H,time=time))
                    assert math.isfinite(prob)
                    all_items.append((record,prob,target,complete));common_items.append((record,prob,target,common))
                    if variant=='learned_seed_2026' and H==1:
                        population[(record,round(time,9))]=(target,complete)
            results += [score(variant,H,all_items),score(variant,H,common_items,'common_5s_followup')]
            checks.append(dict(variant=variant,horizon=H,raw_reference_labels_checked=True,
                               raw_record_end_mask_checked=True,per_path_probability_checked=True))
    assert not mismatches,mismatches
    # Compare with published values, using independent scalar summation.
    published={(r['variant'],int(r['horizon_s'])):r for r in read(ROOT/'results/summary.csv')}
    for result in results:
        key='brier' if result['population']=='horizon_followup' else 'common_brier'
        saved=float(published[(result['method'],result['horizon_s'])][key])
        result['reported_brier']=saved;result['absolute_difference']=abs(result['brier']-saved)
        assert result['absolute_difference']<1e-14
    write(OUT/'model_brier_recomputed.csv',results)
    # Frozen baseline risks, not historical neural-model predictions or alarm flags.
    frozen=OUT/'baseline_predictions_source.csv'
    if not frozen.exists():
        source=ROOT.parent/'inverse_hopf_delivery_20261027/analysis/baseline_comparison/origins.csv'
        base=[{k:r[k] for k in ['record','participant','method','origin_time','risk','label','scored']}
              for r in read(source) if r['method'] in METHODS]
        write(frozen,base)
    baseline=read(frozen);baseline_results=[];alignment=[]
    for method,display in METHODS.items():
        data=[r for r in baseline if r['method']==method]
        keys=[(r['record'],round(float(r['origin_time']),9)) for r in data]
        assert len(keys)==len(set(keys))==len(population)
        assert set(keys)==set(population)
        items=[]
        for key,r in zip(keys,data):
            y,scored=population[key]
            assert y==int(r['label']) and scored==bool(int(r['scored'])),(method,key)
            p=float(r['risk']);assert math.isfinite(p) and 0<=p<=1
            items.append((r['record'],p,y,scored))
        row=score(method,1,items);row['display_name']=display
        row['risk_type']='binary hard rule interpreted as 0/1' if method=='clock_gate' else 'probability'
        baseline_results.append(row)
        alignment.append(dict(method=method,total_origins=len(data),scored_origins=row['origins'],
                              identical_origin_set=True,identical_labels=True,identical_score_mask=True))
    write(OUT/'baseline_brier_1s.csv',baseline_results)
    main_rows=[r for r in results if r['horizon_s']==1 and r['population']=='horizon_followup']
    comparison=main_rows+baseline_results
    write(OUT/'comparison_1s.csv',comparison)
    # Existing fit-only constant references are available for all four event targets.
    constants={(r['record'],int(r['horizon_s'])):float(r['probability'])
               for r in read(ROOT/'results/fit_prefix_constant_risks.csv')}
    constant_rows=[]
    for H in HORIZONS:
        items=[];common=[]
        for (record,v),data in group.items():
            if v!='learned_seed_2026':continue
            for r in data:
                t=float(r['origin_time']);p=constants[(record,H)]
                y=int(any(t+1e-9<s<=t+H+1e-9 for s in references[record]))
                items.append((record,p,y,t+H<=raw_ends[record]+1e-9))
                common.append((record,p,y,t+5<=raw_ends[record]+1e-9))
        constant_rows += [score('fit_prefix_constant_ungated',H,items),score('fit_prefix_constant_ungated',H,common,'common_5s_followup')]
    write(OUT/'fit_constant_brier_all_horizons.csv',constant_rows)
    audit=dict(formula='sum((raw_probability - binary_future_onset_label)^2) / number_of_scored_origins',
        aggregation='micro / pooled origins; not an unweighted average of record scores',
        threshold_in_brier=False,confirmation_rule_in_brier=False,
        label_mismatches=0,tail_mask_mismatches=0,path_probability_mismatches=0,
        primary_origins=len(population),scored_origins=sum(v[1] for v in population.values()),
        checks=checks,baseline_alignment=alignment,
        max_model_difference=max(r['absolute_difference'] for r in results),
        baseline_input_sha256=hashlib.sha256(frozen.read_bytes()).hexdigest(),
        baseline_scope='Frozen 1-s baseline probability predictors, independently rescored. No neural historical risks reused. Baselines have no Monte Carlo path draws.',
        long_horizon_scope='Only the fit-prefix constant reference has 2/3/5-s targets here. The four main 1-s baselines must not be relabelled as long-horizon baselines.')
    (OUT/'audit_checks.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    lines=['# Brier 独立核查与基线分数','',
        '结论：统一 seed2026 报告的 Brier 没有发现计算错误。独立代码没有调用原统计函数，使用 math.fsum 对逐起点平方误差重新求和；真实标签从参考事件起点重新生成，完整随访标志从原始记录终点重新生成，预测概率从保存的逐路径首次事件时间重新计数。','',
        'Brier = Σ(p−y)²/N。这里 p 是未经阈值化的风险；y 是未来 H 秒内是否有新参考异常。0.80 阈值、连续两次确认、报警数和 91 个事件都不作为 Brier 的分母。','',
        '## 1 秒：所有方法使用相同的 8,234 个评分起点','',
        '| 方法 | pooled Brier | 正例 Brier | 负例 Brier | 每记录等权 Brier（补充） |','|---|---:|---:|---:|---:|']
    for r in comparison:
        name=r.get('display_name') or ('DeepONet–Hopf，seed2026' if r['method']=='learned_seed_2026' else 'DeepONet–Hopf，零噪声')
        lines.append(f"| {name} | {r['brier']:.9f} | {r['positive_brier']:.9f} | {r['negative_brier']:.9f} | {r['macro_record_brier']:.9f} |")
    lines+=['','所有方法在完整 8,264 个正常起点上逐个对齐，排除无完整 1 秒随访的 30 个起点后，共 8,234 个；其中正例 1,775 个、负例 6,459 个。各基线的标签和评分掩码均完全相同。','',
        '纯峰时钟是硬规则。表中把其输出 0/1 代入 Brier，所以分数也等于该起点级硬分类错误率；它不是经过概率校准的模型。主文可把它标成“0/1规则评分”，也可以在概率模型表中留空并在附录报告。','',
        '本次生存模型的 Brier 低于 DeepONet–Hopf，不能声称最新模型在 Brier 上优于该基线。Brier 与事件覆盖、报警数及提前量衡量不同方面。','',
        '按目标拆开看：DeepONet 在正例起点的误差约 0.3061，低于生存模型的 0.4089；但负例误差约 0.0637，高于生存模型的 0.0326。负例占多数，因此 pooled 总分中生存模型略占优。这是不同风险分配的权衡，不能只选有利的一项。','',
        '如果先算每条记录 Brier 再让 21 条记录等权平均，会得到另一组分数：DeepONet 0.114674、生存模型 0.117628。这不是当前主表的 pooled 汇总。两种汇总的权重不同，排名也可能不同；本次保留既定 pooled 主指标，等权分数仅作补充，不据结果更换主指标。','',
        '基线分数来自已拟合、冻结的 1 秒基线概率，重新按当前主评价的相同起点与标签计分，不使用旧 block-mean 神经模型风险。它们不含 128 路随机采样，因此更改 DeepONet 推断随机流不会改变这些基线的概率分数。本次没有重新拟合基线或选择新阈值。','',
        '## 多窗口核查','',
        '| H | N | 正例 | 模型 Brier（独立重算） | 原报告 | 绝对差 |','|---:|---:|---:|---:|---:|---:|']
    for r in results:
        if r['method']=='learned_seed_2026' and r['population']=='horizon_followup':
            lines.append(f"| {r['horizon_s']} s | {r['origins']} | {r['positive_origins']} | {r['brier']:.12f} | {r['reported_brier']:.12f} | {r['absolute_difference']:.2e} |")
    lines+=['','不同窗口的标签含义、目标比例和可评分尾部不同，原始 Brier 增大不能单独用来判断模型变差。另有共同完整 5 秒随访的 7,876 个起点版本，均独立复算一致。','',
        '### 关于 2、3、5 秒基线','',
        '现有四种峰龄基线的训练/事件公式是针对 1 秒目标的，不能把这些 1 秒 Brier 填进 2、3、5 秒比较表。长窗口目前可直接公平对比的是按训练段各自 H 秒标签估计的常数风险参考，分数见 fit_constant_brier_all_horizons.csv。它不等于“峰时钟＋常数风险”：后者还有峰时钟门控。正式生存/峰龄长窗口基线需要针对后续周期和各 H 秒目标另行构建。','',
        '## 额外核查','',
        '- 使用未经阈值化的概率；没有把报警标志当概率。',
        '- 概率范围均为 [0,1]；预测风险由 k/128 或零噪声 0/1 重新计数一致。',
        '- 没有把二分类的两类平方误差相加而多乘 2。',
        '- 主结果按所有起点汇总；macro_record_brier 另行输出，不能与 pooled Brier 混用。',
        '- 原模型数值与独立求和差异均小于 1e-14。',
        '- 所有检查和原始对齐基线概率保存在本目录，可脱离报警逻辑再次核算。']
    (OUT/'REPORT_中文.md').write_text('\n'.join(lines),encoding='utf-8')
    html_rows=''.join('<tr><td>'+str(r.get('display_name',r['method']))+'</td><td>'+f"{r['brier']:.9f}"+'</td><td>'+str(r['origins'])+'</td><td>'+f"{r['positive_brier']:.6f}"+'</td><td>'+f"{r['negative_brier']:.6f}"+'</td></tr>' for r in comparison)
    page='''<!doctype html><html lang="zh"><head><meta charset="utf-8"><title>Brier 核查与基线</title><style>body{max-width:1050px;margin:35px auto;font:16px/1.65 system-ui;color:#263b4c;padding:0 20px}table{border-collapse:collapse;width:100%}td,th{padding:10px;border-bottom:1px solid #ddd;text-align:right}td:first-child,th:first-child{text-align:left}.note{background:#edf6f3;padding:16px;border-left:4px solid #259c83}</style></head><body><h1>Brier 独立核查与基线分数</h1><div class="note">已核查：Brier 公式、逐路径概率、原始参考标签、尾部评分掩码与 pooled 汇总一致。最新 DeepONet–Hopf 的 1 秒 Brier=0.115939756。生存基线为 0.113682185，略低。</div><p>所有方法均使用同一 8,234 个起点，正例 1,775、负例 6,459。阈值和两次确认不进入 Brier 计算。</p><table><thead><tr><th>方法</th><th>Brier</th><th>N</th><th>正例 Brier</th><th>负例 Brier</th></tr></thead><tbody>'''+html_rows+'''</tbody></table><p>纯峰时钟是 0/1 硬规则，所列分数相当于起点级错误率，不是校准概率模型。基线均为 1 秒目标；不能作为 2/3/5 秒目标的基线分数。</p><p><a href="REPORT_中文.md">完整核查说明</a> · <a href="comparison_1s.csv">1 秒对比 CSV</a> · <a href="model_brier_recomputed.csv">全部窗口独立重算</a> · <a href="audit_checks.json">一致性检查</a> · <a href="../index.html">回到主报告</a></p></body></html>'''
    (OUT/'index.html').write_text(page,encoding='utf-8')
    print('All model labels, masks, path probabilities and Brier scores independently verified.')
    for r in comparison:print(r['method'],r['origins'],format(r['brier'],'.12f'))
    print('Maximum difference:',audit['max_model_difference'])


if __name__=='__main__':main()
