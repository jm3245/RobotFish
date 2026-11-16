# 前方半圆圆柱目标：最新版 6D PD 与解析漩涡

六组完整结果见 [`RESULTS_zh.md`](RESULTS_zh.md)。

这次修正了此前两个不同设定：目标覆盖整个前方半圆，并且到达定义为接近实体圆柱表面。`wake_fish_setup.png` 左侧给出正前方目标的流速与流线，黄色鱼位于 `(0,0)`、初始朝向 `+x`，圆柱位于 `(3,0)`，背景来流为 `-x` 方向 0.4 m/s。右侧的圆柱是所有独立场景的位置叠加；**每次仅有一个圆柱，不是 100 根圆柱形成的障碍墙**。

## 固定场景

- 鱼头初始位置 `(0,0,0.5)`，航向为零；背景来流 `(-0.4,0,0)`，与初始朝向完全相反。
- 前方 180° 分成 25 个等宽角度区间，取区间中心 `-86.4,-79.2,…,0,…,79.2,86.4` 度。区间中心避免目标恰好位于纯侧向的 ±90°；目标都具有严格正向的逆流分量。
- 每个角度取距离 `2,3,4,5 m`，共 100 个配对场景。圆柱是目标本身，半径 0.35 m、高 1 m、中心 z=0.5 m。
- 鱼体与目标圆柱碰撞启用，鱼体各段之间不新增自碰撞。
- 到达判据：鱼头所属几何体与圆柱的三维最短表面距离 ≤0.05 m，并且同一步没有物理失效；使用 MuJoCo `mj_geomDistance`。目标是圆柱表面附近，不再要求头部进入圆柱中心。
- 只用原封不动的 `flow_040_navigation/blind_6d.py` 中的最新版 `BlindAdaptive6DPD`，未调参、没有 RL、没有输入流速真值。输入严格为 `[head_x,head_y,yaw,yaw_rate,target_x,target_y]`，输出 `[B_raw,ΔA]`。
- 最大命令摆幅三档 `0.38,0.42,0.75 rad`，基础摆幅 `Amax−0.1`，实际命令随控制器在 `Amax−0.2` 与 `Amax` 间切换；关节目标仍截断 ±0.75 rad。频率固定 2.5 Hz、60 s 上限、控制周期 0.02 s、原生物理步长 0.001 s。
- 0.557 m/s 是以前测出的静水平均直游速度，不能填入摆幅的单位 rad，也不是本次的速度硬上限。

## 漩涡如何真正参与鱼的运动

复用 `wake_preview.py` 的解析流场：圆柱势流加 12 个正则化交替点涡，将其旋转 180°、平移到目标圆柱。背景流沿 −x，圆柱后方的涡列也向 −x 对流。预设直径 0.7 m、涡距参数 2.8 m、环量绝对值 0.84 m²/s、核半径 0.105 m、涡对流速度 0.3 m/s。这里的 0.4 m/s 是**远处背景流速**，局部流速会变大、变小且方向改变。

这是定性的规定流场：初始 12 个涡向下游移动，**没有持续生成新涡**，没有真实无滑移边界、CFD 或双向流固耦合，不能把结果当作论文 CFD 涡街的验证。

每 0.02 s 在每个有质量的鱼身体段质心采样局部水速度，按当前 MuJoCo 模型使用的惯性盒阻力计算：

`F(v_rel) = −3π μ d_eq v_rel − 0.5 ρ S |v_rel| v_rel`

其中速度转到身体惯性坐标，`S` 是惯性等效盒在各轴的迎流面积，`d_eq` 是三个边长的均值。添加到该段质心的修正力为 `F(v_body−u_local)−F(v_body−u_background)`，旋转回世界坐标。原生背景流阻力和角向阻力仍由 MuJoCo 计算；这样避免把背景水阻计算两次。修正力在 20 个原生物理子步内保持不变，是时间离散近似。

该力公式与 [MuJoCo 官方惯性盒流体模型实现](https://github.com/google-deepmind/mujoco/blob/3.3.7/src/engine/engine_passive.c) 对应。`check_coupling.py` 用本机 MuJoCo 3.2.3 的原生 `qfrc_fluid` 对比均匀流改变量：最大广义力差为 `4.44e-16`；局部水速等于背景时修正力为零。此检查验证力的转换，不验证解析涡场的真实水动力精度或所有时间步长的收敛性。

## 输出与复现

`scenes.json` 为 100 个场景；`uniform_A*.csv` 与 `wake_A*.csv` 为六组逐场结果；`*_summary.json` 为成功/超时/物理失效计数，并保存控制器 SHA-256；`*_traces.npz` 为每 0.1 s 的实际鱼头轨迹（列：时间、x、y、z、yaw、B_raw、A、表面间距）。`arrival_rates.png` 是配对统计图，`wake_trajectories_9.png` 是三个角度与三档摆幅的九条实际轨迹。底图是 t=1 s 的流场快照，黄线包含整个时间过程，不表示整个过程中流场静止。

```powershell
cd D:\RobotFish_VortexWake_PD_Morphology_20261116
python experiments/front180_wake/check_coupling.py
python experiments/front180_wake/run.py --mode wake --cap 0.75
python experiments/front180_wake/run.py --mode uniform --cap 0.75
python experiments/front180_wake/plot.py
```

对另外两档 `0.38`、`0.42` 分别执行对应命令即可复现全部六组。
