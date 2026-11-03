# Appended to the shared metric code in analyze_unified.py.
import hashlib
import html

remote='<SERVER_RUN_DIR>'
learned=o[o.variant==PRIMARY].copy()
main=primary.loc[1].copy()
zero=summary[(summary.variant=='zero_noise')&(summary.horizon_s==1)].iloc[0]
primary.to_csv(OUT/'multihorizon_evaluation.csv')
primary.loc[[1]].to_csv(OUT/'primary_evaluation.csv')
summary.to_csv(OUT/'noise_by_horizon.csv',index=False)
primary_risk=learned[['record','origin_time','risk_1s','alarm_1s','label_1s','complete_1s']].rename(
    columns={'risk_1s':'transition_probability','alarm_1s':'experimental_trigger','label_1s':'future_onset_label','complete_1s':'complete_followup'})
primary_risk.to_csv(OUT/'primary_risk.csv',index=False)
learned.to_csv(OUT/'multihorizon_risks.csv',index=False)
one=pd.read_csv(OUT/'primary_risk.csv');multi=pd.read_csv(OUT/'multihorizon_risks.csv')
assert np.array_equal(one.transition_probability,multi.risk_1s)
assert np.array_equal(one.experimental_trigger,multi.alarm_1s)
assert np.array_equal(one.future_onset_label,multi.label_1s)
np.testing.assert_allclose(((one.loc[one.complete_followup==1,'transition_probability']-
                          one.loc[one.complete_followup==1,'future_onset_label'])**2).mean(),main.brier,rtol=0,atol=1e-15)
assert main.alarms==one.experimental_trigger.sum()
audit=json.loads((OUT/'audit.json').read_text())
replay=json.loads((OUT/'full_record_replay_check.json').read_text())
assert len(audit)==21 and all(r['checkpoint_unchanged'] for r in audit)
assert all(v['seed']==2026 and v['record_offset']==0 and v['saved_rng_replay_exact'] for r in audit for v in r['variants'])
manifest=[]
for record in audit:
    for v in record['variants']:
        state=OUT/'records'/record['record']/v['rng_file']
        assert hashlib.sha256(state.read_bytes()).hexdigest()==v['rng_sha256']
        manifest.append(dict(record=record['record'],variant=v['variant'],seed=2026,record_offset=0,
            origins=record['origins'],draws=1 if v['variant']=='zero_noise' else 128,
            force_paths_bytes=v['force_paths_bytes'],force_paths_sha256=v['force_paths_sha256'],
            full_paths_remote=remote+'/results/records/'+record['record']+'/'+v['force_paths_file'],
            rng_state_file='results/records/'+record['record']+'/'+v['rng_file'],rng_sha256=v['rng_sha256']))
pd.DataFrame(manifest).to_csv(OUT/'saved_stream_manifest.csv',index=False)
consistency=dict(seed=2026,record_offset=0,records=21,normal_origins=len(learned),
    primary_multihorizon_risks_exact=True,primary_multihorizon_alarms_exact=True,
    primary_multihorizon_labels_exact=True,all_checkpoint_hashes_unchanged=True,
    all_saved_rng_file_hashes_verified=True,all_42_variant_last_batch_replays_exact=True,
    independent_full_record_replay=replay,
    full_force_paths_bytes=sum(v['force_paths_bytes'] for v in manifest),
    full_paths_storage='Retained on SSH server; local package includes states, first-onset paths and hashes, not 4-GB force arrays')
(OUT/'consistency_checks.json').write_text(json.dumps(consistency,indent=2),encoding='utf-8')

events_primary=e[(e.variant==PRIMARY)&(e.horizon==1)]
events_zero=e[(e.variant=='zero_noise')&(e.horizon==1)]
paired=events_zero.set_index(['record','reference_onset'])[['covered_horizon','lead_horizon']].join(
    events_primary.set_index(['record','reference_onset'])[['covered_horizon','lead_horizon']],
    lsuffix='_zero',rsuffix='_learned')
paired['category']=np.select([(paired.covered_horizon_zero==1)&(paired.covered_horizon_learned==1),
    (paired.covered_horizon_zero==1)&(paired.covered_horizon_learned==0),
    (paired.covered_horizon_zero==0)&(paired.covered_horizon_learned==1)],
    ['both','zero_only','learned_only'],default='neither')
paired.reset_index().to_csv(OUT/'noise_paired_events_1s.csv',index=False)
contrast=dict(seed=2026,alarm_reduction_pct=float(100*(1-main.alarms/zero.alarms)),
    brier_reduction_pct=float(100*(1-main.brier/zero.brier)),
    covered_zero=int(zero.covered_within_horizon),covered_learned=int(main.covered_within_horizon),
    paired_events=paired.category.value_counts().to_dict())
(OUT/'noise_contrast.json').write_text(json.dumps(contrast,indent=2))

fig,axs=plt.subplots(2,2,figsize=(10,6.5),layout='constrained')
x=np.arange(4);colors=['#2776AE','#259C83','#BA8A26','#9859A3']
ax=axs[0,0];ax.bar(x,primary.covered_within_horizon,color=colors,width=.6)
for k,r in enumerate(primary.itertuples()):ax.text(k,r.covered_within_horizon+1,f'{r.covered_within_horizon}/91',ha='center')
ax.set(xticks=x,xticklabels=[f'{h} s' for h in HORIZONS],ylim=(0,100),ylabel='Events covered',title='a   Coverage in the respective event window')
ax=axs[0,1];ax.plot(HORIZONS,primary.median_lead_s,'o-',color='#2776AE')
ax.set(xticks=HORIZONS,xlabel='Event horizon (s)',ylabel='Median lead (s)',title='b   Warning lead; fixed seed 2026');ax.grid(alpha=.15)
ax=axs[1,0];ax.bar(x,primary.alarms,color=colors,width=.6)
for k,r in enumerate(primary.itertuples()):ax.text(k,r.alarms+2,str(r.alarms),ha='center')
ax.set(xticks=x,xticklabels=[f'{h} s' for h in HORIZONS],ylabel='Alarms',ylim=(0,max(primary.alarms)*1.15),title='c   Same origins and warning rule')
ax=axs[1,1];ax.plot(HORIZONS,primary.common_mean_risk,'o-',label='Mean predicted risk',color='#2776AE')
ax.plot(HORIZONS,primary.common_prevalence,'s--',label='Observed event fraction',color='#BA8A26')
ax.set(xticks=HORIZONS,xlabel='Event horizon (s)',ylabel='Probability',ylim=(0,1),title='d   Same complete-5-s-follow-up origins');ax.legend(fontsize=8)
save(fig,'unified_multihorizon_seed2026')

fig,axs=plt.subplots(1,3,figsize=(10,3.4),layout='constrained')
for ax,key,title in zip(axs,['alarms','brier','median_lead_s'],['Alarm count','Pooled Brier','Median lead (s)']):
    values=[zero[key],main[key]];ax.bar([0,1],values,color=['#D18432','#2776AE'],width=.55)
    for i,v in enumerate(values):ax.text(i,v+max(values)*.025,f'{v:.4f}' if key=='brier' else f'{v:.3f}' if key=='median_lead_s' else str(int(v)),ha='center')
    ax.set(xticks=[0,1],xticklabels=['Zero diffusion','Learned diffusion'],title=title,ylim=(0,max(values)*1.18))
save(fig,'unified_noise_seed2026')

all_onsets=events_primary[['record','reference_onset']]
links=[]
for record,d in learned.groupby('record'):
    times=d.origin_time.to_numpy();grid=np.arange(round((times.max()-times.min())/.05)+1)*.05+times.min()
    ix=np.rint((times-grid[0])/.05).astype(int)
    fig,axs=plt.subplots(2,1,figsize=(12,5.4),sharex=True,layout='constrained',height_ratios=[2,1])
    for H,color in zip(HORIZONS,colors):
        curve=np.full(len(grid),np.nan);curve[ix]=d[f'risk_{H}s']
        axs[0].plot(grid,curve,lw=1,color=color,label=f'{H} s')
        ar=a[(a.record==record)&(a.variant==PRIMARY)&(a.horizon==H)]
        axs[1].scatter(ar.warning_time,np.full(len(ar),HORIZONS.index(H)),marker='|',s=85,color=color)
    for ax in axs:
        for onset in all_onsets[all_onsets.record==record].reference_onset:ax.axvline(onset,color='#BE3D4D',ls=':',lw=.8)
        ax.grid(alpha=.12)
    axs[0].axhline(.8,color='#777777',ls='--',lw=.8);axs[0].legend(ncol=4,fontsize=8,loc='lower left')
    axs[0].set(title=record+' | seed 2026 | main evaluation = 1-s curve',ylabel='New-onset probability',ylim=(-.03,1.04))
    axs[1].set(yticks=np.arange(4),yticklabels=[f'{h}-s alarms' for h in HORIZONS],xlabel='Time (s)',ylim=(-.5,3.5))
    fig.savefig(FIG/(record+'.png'),bbox_inches='tight');plt.close(fig)
    links.append(f'<details><summary>{record}</summary><img src="figures/{record}.png"></details>')

main_dict={k:(v.item() if hasattr(v,'item') else v) for k,v in main.to_dict().items()}
(OUT/'primary_evaluation.json').write_text(json.dumps(dict(seed=2026,source='multihorizon_evaluation.csv: horizon_s=1',metrics=main_dict),indent=2))
noise_columns=['variant','horizon_s','covered_within_horizon','alarms','matched','unmatched','median_lead_s','brier']
cols=['covered_within_horizon','covered_common_1s','alarms','median_lead_s','brier']
table=['| 事件窗口 | 各自窗口内覆盖 | 共同 1 秒内覆盖 | 报警数 | 提前量中位数 | Brier |',
       '|---|---:|---:|---:|---:|---:|']
for H,r in primary.iterrows():table.append(f'| {H} s | {int(r.covered_within_horizon)}/91 | {int(r.covered_common_1s)}/91 | {int(r.alarms)} | {r.median_lead_s:.3f} s | {r.brier:.6f} |')
report=f'''# 主评价、噪声干预和多窗口分析：统一 seed=2026

本报告替代先前混用原主评价 RNG、20261027、20261028、20261029 等随机流的当前报告。旧文件保留为历史结果，不能与本次主表拼接使用。

## 当前唯一主评价

最新 5 秒 endpoint 检查点，128 条路径，1 秒新异常目标，风险严格大于 0.80，连续两个正常起点确认：

- 覆盖：**{int(main.covered_within_horizon)}/91（{main.coverage*100:.2f}%）**。
- 报警：**{int(main.alarms)} 次**；每正常分钟 **{main.alarms_per_normal_minute:.3f} 次**。
- 报警后 1 秒内匹配：{int(main.matched)} 次；未匹配：{int(main.unmatched)} 次；尾部删失：{int(main.tail_censored)} 次。
- 提前量中位数：**{main.median_lead_s:.3f} 秒**；均值：{main.mean_lead_s:.3f} 秒。
- Brier：**{main.brier:.6f}**；评分起点 {int(main.scored_origins)} 个。

主评价直接导出多窗口结果的 1 秒列，未再次独立抽样。primary_risk.csv 与 multihorizon_risks.csv 的 1 秒概率、标签、报警均已重新读盘并逐元素验证完全相同。主评价表 primary_evaluation.csv 是 multihorizon_evaluation.csv 的 horizon_s=1 行。

## 多窗口结果

{chr(10).join(table)}

四列风险使用同一组未来 5 秒路径，故逐起点满足 p1≤p2≤p3≤p5。1 秒使用原有事件判定；长窗口继续检查模拟中的后续周期。匹配窗口、目标比例和完整随访要求随 H 改变，不能按原始 Brier 或覆盖率跨窗口简单排名。共同 1 秒覆盖用于显示报警提前后是否仍在事件前 1 秒内产生新报警。

## 同一随机流下的噪声干预

| 指标 | 零噪声 | 学习噪声 seed2026 |
|---|---:|---:|
| 1 秒内覆盖 | {int(zero.covered_within_horizon)}/91 | {int(main.covered_within_horizon)}/91 |
| 报警数 | {int(zero.alarms)} | {int(main.alarms)} |
| Brier | {zero.brier:.6f} | {main.brier:.6f} |
| 提前量中位数 | {zero.median_lead_s:.3f} s | {main.median_lead_s:.3f} s |

恢复扩散减少报警 {contrast['alarm_reduction_pct']:.1f}%，降低 Brier {contrast['brier_reduction_pct']:.1f}%，但本次也少覆盖 {int(zero.covered_within_horizon-main.covered_within_horizon)} 个事件，预警更晚。因此旧随机流下“相同覆盖率”的表述已不适用于这份统一报告；不能为了保留旧结论而挑选另一个种子。

## 种子与随机流如何固定

1. 所有评估统一 Python、NumPy、PyTorch seed=2026。每条记录、每个扩散条件重新设置为 2026，**不再添加记录序号偏移**。
2. 使用固定输入顺序和 manifest 顺序，固定 batch size=32，固定 128 条随机路径；零噪声只保存 1 条唯一轨迹，因为其余路径完全重合。
3. 启用 PyTorch deterministic algorithms，关闭 TF32，CUBLAS_WORKSPACE_CONFIG=:4096:8。运行环境版本在 protocol.json 中保存。
4. 在**每个 batch 开始模拟前**保存 CPU 与 CUDA RNG 状态；同时保存所有起点的完整逐路径力轨迹、每条路径的首次新异常时间，以及文件 SHA256。
5. 主评价、零噪声对比、1/2/3/5 秒分析全部读取这一套输出；没有为不同表再次运行随机推断。

固定种子本身不等于能恢复任意时刻的随机流；因此还保存了每批的实际状态、批次边界、计算配置与完整输出。

## 校验与保存位置

- 21 个检查点 SHA256 全部不变，原始 153 个输入/代码/检查点文件的哈希已核对。
- 42 个记录×扩散条件各自恢复了已保存的最后一个 batch RNG 状态，完整力路径逐元素完全相同。
- 另在独立进程中恢复 Pt_10_FD10_1 的全部 17 个 batch：521 个起点，零噪声与学习噪声路径均逐元素完全一致。
- 主评价和多窗口 1 秒列的概率、标签、报警全部相同；详见 consistency_checks.json。
- 服务器实际保存的完整逐路径数组共 {consistency['full_force_paths_bytes']/1024**3:.3f} GiB。它们存于 `{remote}/results/records/*/*_force_paths.npy`。
- 本地报告包保存 RNG 状态、路径首次事件时间、概率、全部评分 CSV 和完整路径的 SHA256 清单；为避免重复传输约 4 GB，完整力路径数组保留在服务器。见 saved_stream_manifest.csv。

## 训练与评估的边界

本次统一的是**固定检查点之后的评估随机流**，没有重新训练模型。现有模型原训练主种子为 2026，训练内部的历史固定验证噪声种子 777 保留不变；本报告没有把历史训练过程改称为全程 seed2026。这样保持同一批权重，只消除这轮评价随机流不统一的问题。

旧的 82/91、146 次、0.490 秒，以及 81/91、149 次、0.505 秒等组合不再作为本次主评价。当前唯一数字以本报告和 primary_evaluation.json 为准。

所有结果仍是对周期衍生异常参考的回顾性正常起点评估；长期窗口没有重新训练或重新校准。
'''
(ROOT/'REPORT_中文.md').write_text(report,encoding='utf-8')

replacement=f'''% Unified current evaluation. Do not mix with previous random streams.
% Frozen-checkpoint inference only; seed 2026; no record offset; batch size 32.
Primary evaluation used the latest 5-s endpoint checkpoints with 128 paths per origin, a strict 0.80 threshold and confirmation at two consecutive origins. Python, NumPy and PyTorch inference streams were reset to seed 2026 for each record and diffusion condition, without record-specific seed offsets. The same saved paths produced the 1-, 2-, 3- and 5-s event risks. The primary evaluation was exported directly from the 1-s column rather than resimulated. It covered {int(main.covered_within_horizon)}/91 onsets within 1~s, issued {int(main.alarms)} alarms, and achieved a median lead of {main.median_lead_s:.3f}~s and pooled Brier score {main.brier:.6f}.

With the same fitted checkpoints, suppressing diffusion covered {int(zero.covered_within_horizon)}/91 onsets, issued {int(zero.alarms)} alarms and yielded Brier score {zero.brier:.6f}, with median lead {zero.median_lead_s:.3f}~s. Restoring diffusion reduced alarm burden and probability error, but covered {int(zero.covered_within_horizon-main.covered_within_horizon)} fewer events and shortened warning lead. Thus, unchanged event coverage from earlier random-stream analyses is not claimed for this unified run.

CPU and CUDA RNG states were archived immediately before each simulation batch. Full decoded force paths and per-path first-onset times were retained with SHA256 checksums. Deterministic algorithms were enabled, TF32 was disabled, and the batch size, manifest order and software configuration were fixed. Replaying the archived stream reproduced complete paths exactly in the recorded environment. Main and multihorizon one-second probabilities, labels and alarms were identical.
'''
(ROOT/'manuscript_replacement.tex').write_text(replacement,encoding='utf-8')
primary[cols].to_latex(ROOT/'unified_multihorizon_table.tex',float_format=lambda v:f'{v:.6f}',caption='Unified seed-2026 evaluation; coverage and lead use each event window.')
table_html=primary[cols].to_html(float_format=lambda v:f'{v:.6f}',border=0)
page=f'''<!doctype html><html lang="zh"><head><meta charset="utf-8"><title>统一 seed2026：主评价与多窗口</title><style>body{{max-width:1200px;margin:30px auto;padding:0 24px;font:16px/1.65 system-ui;color:#273b4e}}img{{max-width:100%}}table{{border-collapse:collapse;width:100%;font-size:13px}}td,th{{padding:7px;border-bottom:1px solid #ddd;text-align:right}}details{{border:1px solid #ddd;margin:12px 0;padding:12px}}.card{{padding:18px;background:#edf6f3;border-left:5px solid #269C83}}a{{color:#176995}}</style></head><body>
<h1>当前统一评价：seed = 2026</h1><p>21 条记录 · 最新 5 秒 endpoint 检查点 · 0.80 阈值 · 连续两起点确认 · 完整保存随机流</p>
<div class="card"><strong>主评价 = 多窗口分析的 1 秒列：</strong>{int(main.covered_within_horizon)}/91 事件，{int(main.alarms)} 次报警，提前量中位数 {main.median_lead_s:.3f} 秒，Brier {main.brier:.6f}。两处使用同一批保存路径，概率、标签、报警已验证逐元素完全相同。本版本替代旧随机流报告。</div>
<p>每条记录直接使用 seed2026，无记录偏移；固定 batch32 与 128 条路径。已保存每批 CPU/CUDA RNG 状态及服务器约 4 GB 完整逐路径数组；独立恢复随机状态已逐元素复现。</p>
<p><a href="REPORT_中文.md">完整中文报告</a> · <a href="results/primary_evaluation.csv">主评价 CSV</a> · <a href="results/multihorizon_evaluation.csv">多窗口 CSV</a> · <a href="results/saved_stream_manifest.csv">随机流与路径清单</a> · <a href="results/consistency_checks.json">一致性校验</a> · <a href="run_unified.py">实际运行代码</a></p>
<img src="figures/unified_multihorizon_seed2026.png"><h2>学习噪声：同一组轨迹的四个事件窗口</h2>{table_html}
<p>较长窗口的目标、匹配和尾部要求不同，不能简单按覆盖或 Brier 排名；common_1s 指统一前 1 秒的新报警覆盖。</p>
<h2>零噪声与学习噪声，均统一本次数据</h2>{summary[noise_columns].to_html(index=False,float_format=lambda v:f'{v:.6f}',border=0)}
<img src="figures/unified_noise_seed2026.png"><h2>保存与重放</h2><p>完整力轨迹在 SSH 服务器：<code>{remote}/results/records</code>。本地包包含全部 RNG 状态、逐路径首次事件、评分结果和完整路径哈希，不重复下载大数组。固定检查点未重新训练；训练内部历史验证流777未修改。</p>
<h2>21 条记录</h2><p>上方为四种事件窗口风险；下方为相应报警；红色竖线为参考异常起点。</p>{''.join(links)}</body></html>'''
(ROOT/'index.html').write_text(page,encoding='utf-8')
print(primary[cols].to_string());print(json.dumps(contrast,indent=2))

# Attach optional independent Brier audit without rerunning inference.
from attach_brier_audit import attach
attach(ROOT)
