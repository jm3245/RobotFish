# 文件夹导览与测试文件位置

根目录：`D:\RobotFish_VortexWake_PD_Morphology_20261116`

```text
RobotFish_VortexWake_PD_Morphology_20261116/
├─ README_zh.md                    项目状态、物理边界、运行命令
├─ DIRECTORY_GUIDE_zh.md           本导览
├─ experiments/
│  ├─ README_zh.md                 条件索引和运行方式
│  ├─ uniform_flow/conditions.json 已完成的 0.4/0.5 m/s × 三摆幅测试矩阵
│  └─ vortex_wake/conditions.json  圆柱/涡流准备条件；尚无正式到达率
├─ flow_040_navigation/
│  ├─ blind_6d.py                  当前使用的 6D PD 控制器
│  ├─ blind_6d_benchmark.py        逐场景运行器；参数设置摆幅与频率
│  └─ screen.py                    指向本目录 MuJoCo 环境的适配器
├─ controllers/
│  ├─ flow_aware_8d.py             含真实水流两维输入的旧对照 PD
│  └─ heading_first_6d.py          较早的 6D 航向优先对照 PD
├─ legacy_stack/                   N=4 导航环境及原有波形/配置依赖
│  ├─ 01_Baseline_Controller/      原波形 PD 核心及实体 XML
│  ├─ 02... → 03... → 04...         环境包装链前段
│  ├─ 05_PD_FishVelocity_Feedback/ 旧速度反馈 PD
│  ├─ 06... → 07... → 12... → 13... 环境包装链后段；13 为当前测试环境
│  ├─ 1027_formal/                 原 N=4 物理环境/配置/实体 XML
│  └─ fish_rl/astra_direct_n4.../  原 ServoEnv 和控制参数
├─ structured_100_040/
│  ├─ scenes.json                  0.4 m/s，100 个实际场景
│  └─ protocol.json                方向、目标、初始状态和判据设计
├─ structured_100_050/
│  ├─ scenes.json                  0.5 m/s，和 0.4 场景逐一配对
│  └─ protocol.json                同一设计，仅流速放大 1.25 倍
├─ reference_results/              六组已完成的均匀流 CSV/JSON 结果
├─ fish_morphologies/
│  ├─ generate_fish.py             N=3/4/5/6/8 鱼 XML 生成器
│  ├─ test_swim.py                 静水直游频率/摆幅/相位扫描
│  ├─ fish_N{3,4,5,6,8}.xml        五种可加载的鱼体模型
│  ├─ best_gaits.json              原扫描选出的步态
│  └─ swim_results.csv             原扫描逐组合数据
├─ eel_swimming/assets/            生成鱼体需要的泳池场景和视觉 XML
├─ reference_cross_morph_hydro/    多形态、水流与受力项目的源码参考
├─ models/fish_N4_cylinder_only.xml 仅有固定圆柱几何的对照模型
├─ make_cylinder_control.py        重新生成仅有圆柱的模型
├─ wake_preview.py                 空间涡流速度/梯度预览函数
├─ wake_field_preview.png          预览图，未参与鱼的实际仿真
├─ references/paper_zh_translation.pdf 用户提供的论文译本
├─ source_manifest.json            正式文件来源、大小和 SHA-256
├─ refresh_manifest.py             更新清单
└─ verify_bundle.py                文件、鱼模型、场景和小样本自检
```

`__pycache__` 是 Python 运行时缓存；`flow_040_navigation/smoke_040_*` 是首次迁移时留下的单场景验证输出，不属于 `source_manifest.json` 中的正式基线结果。

## 四个最容易混淆的位置

**1. 当前 PD 在哪里？** [`flow_040_navigation/blind_6d.py`](flow_040_navigation/blind_6d.py) 是最近 0.4/0.5 m/s 对比使用的版本；它与 `D:\EEL_MOJOCO\flow_040_navigation\blind_6d.py` 的 SHA-256 一致：`dbc15fdebee5ab31a43439ef72d320dcd4514e672879de0f30ae54b0bf00dc01`。它输入 6D，输出偏置 `B` 和摆幅变化 `ΔA`。其他控制器在 `controllers/` 和 `legacy_stack/`，不是六组当前结果的默认控制器。

**2. 水流的大小和方向在哪里？** 两个 `structured_100_0xx/scenes.json` 的每个场景都含 `flow_base=[flow_x,flow_y,0]`，这是施加到 MuJoCo 中的**世界坐标水平水流向量**；`flow_relative_deg` 和 `flow_world_deg` 是解释方向的元数据。`protocol.json` 写整体生成规则。两套各 100 场景均覆盖相对目标 `0°,36°,...,324°` 的 10 档方向，`180°` 是正逆流。

**3. 摆幅在哪里？** 场景仅规定几何/水流，不存摆幅。测试器的 `--max-amplitude` 设置 `0.38/0.42/0.75 rad`；内部基准摆幅为 `A_base=A_max−0.10 rad`，控制器每步输出的 `ΔA` 在 `[-0.10,+0.10] rad`。频率固定 `--frequency-hz 2.5`。六组旧结果在 `reference_results/`，对应关系见 [`experiments/uniform_flow/conditions.json`](experiments/uniform_flow/conditions.json)。

**4. 涡流测试在哪里？** 目前只有 [`experiments/vortex_wake/conditions.json`](experiments/vortex_wake/conditions.json) 记录的预备条件：一个**仅加入圆柱实体**的 N4 XML，以及一个**尚未耦合鱼体受力**的空间水流预览。当前没有可与上述 100 场景相提并论的涡流导航场景、GIF 或成功率。圆柱 XML 的中心为 `(2,0)`，预览函数的圆柱中心为 `(0,0)`；还没有对齐成同一实验布局。
