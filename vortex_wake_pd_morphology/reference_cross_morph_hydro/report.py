"""Report only results that were actually produced by this isolated run."""
from __future__ import annotations
import json
import numpy as np
from .config import RUN,TRAIN_N,SEEDS,PROTOCOL,write,sha


def main():
    ds=json.loads((RUN/'dataset_audit.json').read_text())
    assert ds['passed'] and ds['sha256']==sha(RUN/'dataset.h5')
    counts={str(n):json.loads((RUN/'data'/f'N{n}'/'complete.json').read_text()) for n in TRAIN_N}
    assert all(r['passed'] for r in counts.values())
    bc={str(s):{k:json.loads((RUN/'bc'/str(s)/k/'complete.json').read_text()) for k in 'ABC'} for s in SEEDS}
    wm={str(s):dict(complete=json.loads((RUN/'world'/str(s)/'complete.json').read_text()),
                    diagnostics=json.loads((RUN/'world'/str(s)/'diagnostics.json').read_text())) for s in SEEDS}
    evaluation={str(s):{} for s in SEEDS}
    for s in SEEDS:
        for suite in PROTOCOL['suites']:
            evaluation[str(s)][suite]={k:json.loads((RUN/'evaluation/N8'/str(s)/suite/k/'summary.json').read_text()) for k in 'ABC'}
    aggregate={}
    for k in 'ABC':
        vals=[evaluation[str(s)][suite][k] for s in SEEDS for suite in PROTOCOL['suites']]
        trials=sum(r['count'] for r in vals)
        aggregate[k]=dict(successes=sum(r['successes'] for r in vals),trials=trials,
            success_rate=sum(r['successes'] for r in vals)/trials,
            mean_capped_time_s=sum(r['count']*r['mean_capped_time_s'] for r in vals)/trials,
            mean_path_length_m=sum(r['count']*r['mean_path_length_m'] for r in vals)/trials,
            mean_path_deviation_rms_m=sum(r['count']*r['mean_path_deviation_rms_m'] for r in vals)/trials)
    all_results=dict(scope='oracle_force_success_subset',protocol=PROTOCOL,data=ds,
        counts={n:{'accepted':x['accepted'],'counts':x['counts'],'attempts':x['attempts']} for n,x in counts.items()},
        world=wm,bc=bc,evaluation=evaluation,aggregate=aggregate,
        baseline_old_results_not_copied=True,n8_in_fit=False)
    write(RUN/'all_results.json',all_results)
    lines=['# Robot Fish 新水动力交互历史：独立 A/B/C 实验', '',
        '> **仿真 Oracle Force 实验。**输入是 MuJoCo Warp 内部刚体流体载荷；尚非实体鱼可直接部署的传感器读数。',
        '> 8 个区域由 N+1 刚体合力/力矩沿轴向保总量分配，区域内部受力位置是近似；没有压力场或独立尾鳍载荷。', '',
        '## 输入和网络', '',
        '- A：不变的 17D 当前观测 → 原 MLP → 4D tanh 波形动作。',
        '- B：50×36D（水动力载荷 8×(三轴力+竖轴力矩)，加上前一时刻实际平滑波形 4D）→ 三层 128 通道 CNN → 16D；与 17D 拼接后 BC 联合训练。',
        '- C：同结构、同输入；先用未来 0.2/0.5/1.0 秒的 32D 载荷及 12D 运动影响监督预训练；BC 中冻结 Encoder。',
        '- WM Predictor 以当前 17D、16D latent 和实际未来 50×4D 控制序列为条件，GRU 后三头各预测 44D；闭环 Policy 只使用 Encoder，不调用 Predictor。',
        '- 50 帧为 50 Hz 预动作状态的过去约 1 秒；当前动作及未来信息不进入 Policy。真实水流及形态 ID 仅用于模拟和分层统计。', '',
        '## 实际数据覆盖', '',
        f"成功轨迹 {sum(x['accepted'] for x in counts.values())} 条；窗口 {ds['samples']} 个；训练 N={','.join(map(str,TRAIN_N))}；N8 仅用于冻结评估。",'']
    for n,x in counts.items():
        matrix=np.asarray(x['counts'])
        lines.append(f"- N{n}：{x['accepted']} 条，72 个尺寸×流型格中有 {int((matrix>0).sum())} 格，最少 {int(matrix.min())} 条/格。")
    lines+=['','## 独立 test 动作 MSE（四维均值）','',
        '| Seed | A | B | C |','|---|---:|---:|---:|']
    for s in SEEDS:
        lines.append('| '+str(s)+' | '+' | '.join(f"{bc[str(s)][k]['scores']['test']['action_mse']:.6f}" for k in 'ABC')+' |')
    lines+=['','## WM test 预测误差', '',
        '各 seed 的 `world/<seed>/diagnostics.json` 给出三个时域的未来水动力合力/力矩 RMSE、运动 RMSE，以及载荷保持和恒速度基线；量纲及维度见 JSON。', '']
    for s in SEEDS:
        x=wm[str(s)]['complete']['test_loss']; lines.append(f"- seed {s}：标准化总 loss 正常 {x['normal']:.5f}、latent 置零 {x['zero']:.5f}、打乱 {x['shuffle']:.5f}。")
    lines+=['','## N8 六类配对闭环结果', '',
        '每个 seed、suite 在完全相同初态、目标和真实水流上对照 A/B/C；失败到达时间按 60 秒计。', '',
        '| Seed | 条件 | 方法 | 成功/总数 | 成功率 | 失败封顶平均时间(s) | 路径长度(m) | 轨迹偏离(m) |',
        '|---|---|---|---:|---:|---:|---:|---:|']
    for s in SEEDS:
        for suite in PROTOCOL['suites']:
            for k in 'ABC':
                r=evaluation[str(s)][suite][k]
                lines.append(f"| {s} | {suite} | {k} | {r['successes']}/{r['count']} | {r['success_rate']:.1%} | {r['mean_capped_time_s']:.2f} | {r['mean_path_length_m']:.2f} | {r['mean_path_deviation_rms_m']:.2f} |")
    lines+=['','## 汇总（按 episode 数加权）','',
        '| 方法 | 成功/总数 | 成功率 | 封顶平均时间(s) | 路径长度(m) |',
        '|---|---:|---:|---:|---:|']
    for k,r in aggregate.items():
        lines.append(f"| {k} | {r['successes']}/{r['trials']} | {r['success_rate']:.1%} | {r['mean_capped_time_s']:.2f} | {r['mean_path_length_m']:.2f} |")
    lines+=['',
        '解释范围：这是新同批数据上的 A/B/C 因果对照。旧 34D 实验使用另一批轨迹，若需断言“新 16D 优于旧速度历史”，应在同一场景做旧模型冻结重放，且先核对动作和目标协议完全一致；旧结果不能当作这轮实验的对照样本。',
        'World Model 只做表征预训练，不在闭环进行候选动作预测、规划或搜索。C−B 反映预训练加冻结这一整体条件，不能单凭结果归因于 latent 容量或数据覆盖。', '']
    (RUN/'RESULTS_zh.md').write_text('\n'.join(lines),encoding='utf-8')

if __name__=='__main__':main()
