"""Require all six complete result files and publish their paired counts."""
import csv
import json
import statistics
from run import HERE

lines=['# 实测结果','', '| 最大命令摆幅 rad | 均匀逆流到达 | 解析漩涡到达 | 均匀/漩涡物理失效 | 均匀/漩涡超时 |', '|---:|---:|---:|---:|---:|']
results=[]
for cap in [.38,.42,.75]:
    group=[]
    for mode in ['uniform','wake']:
        stem=f'{mode}_A{cap:.2f}'
        data=json.loads((HERE/f'{stem}_summary.json').read_text())
        rows=list(csv.DictReader((HERE/f'{stem}.csv').open(encoding='utf-8')))
        assert data['n']==len(rows)==100
        assert data['arrived']==sum(int(r['success']) for r in rows)
        assert len({r['case_id'] for r in rows})==100
        reached=[float(r['seconds']) for r in rows if int(r['success'])]
        data['mean_arrival_s']=statistics.mean(reached) if reached else None
        group.append(data);results.append(data)
    a,b=group
    lines.append(f"| {cap} | {a['arrived']}/100 | {b['arrived']}/100 | {a['physical_failure']}/{b['physical_failure']} | {a['timeout']}/{b['timeout']} |")
lines += ['', '最大档 0.75 rad：均匀流平均到达时间 18.90 s，解析漩涡平均 17.30 s。两种低摆幅是否存在严格物理不可达，需要额外的速度包络与可达性分析；本表只证明它们在本次固定 PD、时限与场景下未到达。', '', '![到达率](arrival_rates.png)', '', '![九条实际轨迹](wake_trajectories_9.png)', '', '100/100 是这 100 个规定解析流场场景的结果，不代表真实持续脱涡流场或任意多障碍场景的保证。']
(HERE/'RESULTS_zh.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
(HERE/'summary.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
print('\n'.join(lines[:8]))
