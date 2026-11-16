# 实验条件索引

## 新增：前方半圆目标与解析漩涡受力测试

[`front180_wake/README_zh.md`](front180_wake/README_zh.md) 包含 25 个前方角度 × 4 个距离的 100 个新场景，使用最新版 6D PD，配对测试均匀逆流和解析圆柱涡列。鱼体各段接收局部水速产生的阻力修正，目标是启用碰撞的圆柱表面。流场图标出鱼的位置，并保存三档摆幅的统计和轨迹。它是规定解析流场的测试，尚不是论文 CFD 或持续脱涡实验。

## 已完成：0.4 m/s 纯逆流圆柱目标

[`upstream_cylinder_040/README_zh.md`](upstream_cylinder_040/README_zh.md) 说明新的 100 个左右前方逆流场景、半径 0.35 m 的非碰撞圆柱目标、三档最大摆幅，以及现行 6D PD 与旧版航向优先 PD 的六组实测结果。场景、逐场 CSV、汇总 JSON 和两张统计图都在 `upstream_cylinder_040/`。该圆柱仅用于到达判定，**不是**下节的涡流发生器。

## 已完成：均匀流基线

[`uniform_flow/conditions.json`](uniform_flow/conditions.json) 给出测试入口、两套场景和三档摆幅结果的对应关系。实际场景在 [`../structured_100_040/scenes.json`](../structured_100_040/scenes.json) 与 [`../structured_100_050/scenes.json`](../structured_100_050/scenes.json)：分别是恒定 `0.4`、`0.5 m/s` 水流，各 100 个场景。每套场景按“10 个目标世界方向 × 10 个水流相对目标方向”生成。每个场景还写有起点、目标、鱼初始朝向、`flow_base` 世界坐标水流向量和初始关节状态。场景内**没有摆幅**，也**没有圆柱或涡流**。

摆幅在运行 [`../flow_040_navigation/blind_6d_benchmark.py`](../flow_040_navigation/blind_6d_benchmark.py) 时用 `--max-amplitude 0.38/0.42/0.75` 设定；频率用 `--frequency-hz 2.5`。因此，同一个场景可以在三档摆幅下严格配对复测。旧结果位于 [`../reference_results`](../reference_results)，每一档都有逐场景 `_results.csv` 与汇总 `_summary.json`。命令示例：

```powershell
cd D:\RobotFish_VortexWake_PD_Morphology_20261116
python flow_040_navigation/blind_6d_benchmark.py --scene-file structured_100_050/scenes.json --frequency-hz 2.5 --max-amplitude 0.42 --prefix new_050_a042
```

该命令将新结果写入 `flow_040_navigation/new_050_a042_results.csv` 和 `_summary.json`，不会覆盖 `reference_results`。

## 准备中：固定圆柱与涡流

[`vortex_wake/conditions.json`](vortex_wake/conditions.json) 记录目前已有的两种**独立准备条件**：

1. `models/fish_N4_cylinder_only.xml`：已放入实体圆柱，仍使用全域均匀 `opt.wind`；当前鱼体碰撞标志为零，圆柱本身不会生成尾流。
2. `wake_preview.py`：可查询空间变化的预览速度 `u(x,y,t)` 和梯度，仅画场图，未参与鱼的 MuJoCo 受力与 PD 控制。

这两项旧准备条件使用不同的圆柱坐标。新增的 `front180_wake/` 已另行统一目标位置、接入局部阻力修正，并保存 100 场景测试结果；旧准备模型和旧 preview 入口本身仍维持各自的功能。
