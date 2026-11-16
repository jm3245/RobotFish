# Robot Fish Hydrodynamic Interaction v1（独立仿真 Oracle Force 实验）

本目录复制并冻结了上一轮的多尺寸环境、Teacher 推理和成功轨迹质量门槛；不读取上一轮 Student 数据集进行拟合，不覆盖上一轮 checkpoint、XML 或评估。新输出仅在 `runs/hydro_interaction_v1`。

## 物理审计结论

- 场景选项：水密度 1000 kg/m³、黏度 0.001 Pa·s、重力 0、时间步 0.001 s；控制间隔 0.02 s。三维水流是每 lane 空间均匀、随时间可变的 `opt.wind`，不是 Navier–Stokes 场或局部湍流。
- MuJoCo Warp 3.11 的 `_fluid_force` 内核按每个刚体的惯量等效盒计算流体合力和质心力矩；`qfrc_fluid` 是由这些刚体载荷投影得到的广义流体力。`qfrc_actuator`、`actuator_force` 是执行器项，`qfrc_passive` 还含弹簧、阻尼等，不能充当纯水动力。`cfrc_ext` 不是逐区域压力场。
- `InteractionSensor` 复用**同一原生内核**读取刚体载荷，`preflight` 用独立投影核确认其与 `qfrc_fluid` 一致。8 个区域按刚体轴向长度交叠比例分配刚体载荷，合计守恒，但单刚体内部位置分配只是近似。尾鳍与最后一段共用刚体，无法读到独立尾鳍压力。
- 这是仿真 **Oracle Force** 输入。真实部署需要沿体段的应变/力矩/压差等传感器及标定的载荷估计器；当前项目没有这类硬件记录，不能宣称直接部署性能。真正的压力、涡量、局部流速均没有作为传感器读数，也没有传给 Student。

## 输入与时序

| 通道 | 维度 | 定义 | 用途 |
|---|---:|---|---|
| Current | 17 | 目标体坐标 XY 2，头部线/角速度 6，下方向 3，已平滑波形参数 4，相位正余弦 2 | A/B/C 当前观测，不变 |
| 旧 history | 50×34 | 头部线/角速度 6，8 个 site 的体坐标速度 24，已平滑波形 4 | 旧实验；本轮仅诊断保留 |
| 新 interaction | 50×36 | 8 个归一化轴向区间 ×（头部坐标 Fx,Fy,Fz,Mz）32，前一动作后实际已平滑波形 4 | B/C Encoder 输入 |
| latent | 16 | 三层 1D CNN，128 通道，时间池化，线性输出 | B 联合 BC；C 新 WM 预训练后冻结 |

载荷除以 `0.5×ρ×(0.45m/s)²×L²`，力矩再除以 L。L 是机器鱼自身的标定几何尺寸，仅用于物理无量纲化，不把形态 ID、质量、真实水流或 Teacher privileged Observation 直接加为 Student 通道。再由 N3–N6 **训练 split** 计算各通道均值标准差；N8 永不参与标准化或选模。历史的第 50 帧是决策前 `t`，已平滑波形对应上一控制步执行结果。窗口跨越约 1 秒，未来动作不进入 Policy。

## World Model 与对照

`50×36` → CNN(36→128→128→128, kernel5) → 时间均值 → Linear(128→16)。预测器输入当前 17D、16D latent、**实际未来** 50×4 已平滑控制；GRU(4→128) 后在 0.2、0.5、1.0 秒三头分别输出 44D：32D 区域流体载荷 + 12D 位移、姿态变化与末端速度。两组标准化 MSE 等权相加。真实水流与形态 ID 不作 WM 标签或输入；Predictor 不在 BC 或闭环决策时调用。

A 为原 17D→4D MLP。B 用随机新 Encoder 并以单一动作 BC Loss 同时更新 Encoder 和同结构 Policy。C 每 seed 先独立训练本轮 WM，然后将对应 Encoder 冻结；BC 只更新 Policy。三组均为 3 seed、每组每 seed 40×96=3840 BC 更新、batch 256，按整轨迹训练/验证/test 划分；WM 的额外预训练成本单列。

Teacher 使用上一轮真实完成训练的 N3/4/5/6 冻结 checkpoint 推理；新场景种子独立采集 20 次/尺寸×流型格候选，只接受完整 XYZ 到达、停留且运动质量过关的轨迹。不能把已有覆盖缺格说成合格完整覆盖。N8 只在所有模型冻结后用于六条件配对零样本评估，失败时间按 60 秒封顶。

入口：`python -m fish_rl.cross_morph_20261021_hydro_v1.pipeline`。完成证据：`runs/hydro_interaction_v1/final_complete.json` 的 `passed=true`，以及 `RESULTS_zh.md`。
