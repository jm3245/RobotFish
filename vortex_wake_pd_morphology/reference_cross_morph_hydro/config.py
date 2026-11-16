from pathlib import Path
import hashlib, json, os, time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TRAIN_N = (3, 4, 5, 6)
SEEDS = (20261020, 20261021, 20261022)
DEVICE = 'cuda:0'
RUN = HERE / 'runs' / 'hydro_interaction_v1'
PARENT_RUN = HERE.parent / 'cross_morph_20261021_success_v2' / 'runs' / 'success_subset_v2'
PROTOCOL = dict(train_n=TRAIN_N, holdout_n=8, morphology_train_count=12,
    morphology_ood_count=4, control_dt=.02, physics_dt=.001,
    episode_seconds=60, arrival_radius_m=.5, arrival_dwell_steps=10,
    horizontal_flow_max_m_s=.45, vertical_constant_max_m_s=.025,
    dynamic_vertical_amplitude_max_m_s=.10,
    teacher_envs=384, teacher_parallel_workers=3, teacher_iterations=12000, teacher_eval_every=250,
    teacher_min_full_updates=1500, teacher_plateau_evaluations=8,
    teacher_plateau_min_gain=.008, teacher_learning_rate=2e-4, teacher_gamma=.999,
    teacher_critic_warmup_updates=50,
    teacher_finite_horizon=True,teacher_timeout_bootstrap=False,
    teacher_observation_version='physics_v2', teacher_legacy_prefix='30+2*N unchanged',
    curriculum_levels=[.15,.40,.70,1.0], curriculum_probe_success=[.92,.90,.86],
    curriculum_probe_worst=[.3333333,.3333333,.3333333], curriculum_min_updates=250,
    curriculum_max_stage_updates=1500,
    teacher_gate_overall=.85, teacher_gate_per_cell=.70,
    teacher_gate_trials_per_cell=12, collection_successes_per_cell=40,
    collection_max_attempts_per_cell=160, collection_min_steps=110,
    world_epochs=40, bc_epochs=40, updates_per_epoch=96, batch_size=256,
    eval_trials_per_cell=24, seeds=SEEDS,
    student_base_dim=17, history_shape=[50,36], latent_dim=16, action_dim=4,
    suites=['ID','size','direction','strength','dynamic3d','size_dynamic3d'],
    flow_model='MuJoCo native inertia-box fluid, spatially uniform prescribed wind',
    buoyancy='fully submerged exact neutral buoyancy: rho_body=rho_water=1000; weight and buoyancy cancel at each COM; net gravity zero',
    depth_actuation='yaw hinges only; no depth servo, constraints, target attraction or residual controller',
    target='random XY at fixed z=.5; success uses full world XYZ distance')

# User-authorized exploratory branch; original v2 qualification stays intact.
PROTOCOL.update(experiment='hydro_interaction_v1', original_teacher_gate_required=False,
    collection_rounds=10, collection_attempts_per_cell=20, collection_successes_per_cell=None,
    subset_selection='fixed 20 independent attempts per N-size-flow cell; keep only physical full XYZ successes',
    original_full_coverage_claim=False,teacher_rl_updates_this_run=0,
    teacher_assessment_seed_base=434110000,collection_seed_base=43400000,
    evaluation_seed_offset=109000000,selection_basis='latest actually trained 12000 checkpoints; no N8 selection',
    previous_dataset_reused=False,interaction_channels='8 body-length bins x (Fx,Fy,Fz,Mz) plus 4 actually executed wave controls',
    interaction_observation_class='oracle_force_exact_MuJoCo_Warp_fluid_kernel',
    interaction_force_frame='head body frame',interaction_hz=50,
    interaction_normalization='per-bin fluid force by water density, fixed 0.45 m/s speed and local body length squared; torque also by local length',
    world_horizons_steps=[10,25,50],world_targets='future 8x4 hydro load and body displacement/rotation/terminal velocity',
    world_true_flow_supervision=False,policy_true_flow_input=False,policy_morphology_id_input=False)

def write(path, value):
    path=Path(path);path.parent.mkdir(parents=True, exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    temp.replace(path)

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def status(stage, **kw):
    row=dict(stage=stage,pid=os.getpid(),unix_time=time.time(),**kw)
    write(RUN/'status.json',row);print(json.dumps(row,ensure_ascii=False),flush=True)

def check_stop():
    if (RUN/'STOP').exists():raise RuntimeError('STOP requested; checkpoints and partial data retained')

def source_manifest():
    paths=list((RUN/'parent_checkpoints').glob('N*.pt'))
    paths+=list(HERE.glob('*.py'))
    paths+=list((HERE.parent/'shared_model').glob('*.py'))
    paths+=[HERE.parent/p for p in ('mdp.py','rl_cfg.py','omni_multi_wave_action.py','wave_action.py','env_cfg.py','omni_env_cfg.py')]
    paths+=[ROOT/'fish_morphologies'/'generate_fish.py']
    return {str(p.relative_to(ROOT)):sha(p) for p in sorted(paths)}
