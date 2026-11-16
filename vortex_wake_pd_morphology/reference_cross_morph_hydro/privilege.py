"""Teacher-only physical descriptors. No student/history/WM reads this module."""
import numpy as np
import torch
import mujoco
from mjlab.utils.lab_api.math import quat_apply_inverse

def static_dim(n):return 51+26*n
def extra_dynamic_dim(n):return 42+6*n
def observation_dim(n):return 30+2*n+static_dim(n)+extra_dynamic_dim(n)

def static_physics(model,cfg,meta,wave):
    """All dimensions use SI references, never statistics fitted to N8."""
    n=cfg.n_joints
    bodies=np.array([i for i in range(model.nbody) if (mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,i) or '').startswith('fish/')])
    geoms=np.flatnonzero(np.isin(model.geom_bodyid,bodies))
    joints=np.array([model.joint(f'fish/joint_{i}').id for i in range(n)])
    actuators=np.arange(model.nu)
    assert len(bodies)==n+1 and len(geoms)==n+2 and len(actuators)==n
    mass=model.body_mass[bodies];total=mass.sum()
    d=mujoco.MjData(model);mujoco.mj_forward(model,d)
    com=(mass[:,None]*d.xipos[bodies]).sum(0)/total
    inertia=np.zeros((3,3))
    for j,b in enumerate(bodies):
        rot=d.ximat[b].reshape(3,3);r=d.xipos[b]-com
        inertia+=rot@np.diag(model.body_inertia[b])@rot.T+mass[j]*((r@r)*np.eye(3)-np.outer(r,r))
    values=[];names=[]
    def add(name,a,scale=1.):
        a=np.asarray(a,dtype=float).reshape(-1)/np.asarray(scale)
        values.extend(a.tolist());names.extend([name] if len(a)==1 else [f'{name}[{j}]' for j in range(len(a))])
    add('length_m',cfg.body_length)
    add('max_width_m',2*max(cfg.width_profile),.21)
    add('max_height_m',2*max(cfg.height_profile),.29)
    add('tail_area_m2',cfg.tail_fin_area,.018);add('tail_height_m',cfg.tail_fin_height,.18)
    add('proportion',meta['proportion']);add('joint_count',n,8)
    add('total_mass_kg',total,10);add('displaced_volume_m3',total/1000,.01)
    add('buoyancy_N',total*9.81,98.1);add('density_kg_m3',1000,1000);add('viscosity_Pa_s',.001,.001)
    add('rigid_reference_inertia_kg_m2',inertia[np.triu_indices(3)],10)
    add('body_mass_kg',model.body_mass[bodies],10)
    add('body_principal_inertia_kg_m2',model.body_inertia[bodies].ravel(),10)
    add('body_com_offset_m',model.body_ipos[bodies].ravel())
    add('body_origin_offset_m',model.body_pos[bodies].ravel())
    add('body_inertial_quaternion',model.body_iquat[bodies].ravel())
    add('joint_limits_rad',model.jnt_range[joints].ravel(),np.pi)
    add('actuator_kp',model.actuator_gainprm[actuators,0],35)
    add('actuator_kv',-model.actuator_biasprm[actuators,2],.6)
    add('actuator_torque_limits_Nm',model.actuator_forcerange[actuators].ravel(),12)
    dofs=model.jnt_dofadr[joints]
    add('joint_damping_Nms',model.dof_damping[dofs],.015)
    add('joint_armature_kg_m2',model.dof_armature[dofs],.0005)
    add('joint_stiffness_Nm',model.jnt_stiffness[joints],.01)
    add('wave_low',wave._low.cpu().numpy(),np.array([3,.7,1.5,.3]))
    add('wave_high',wave._high.cpu().numpy(),np.array([3,.7,1.5,.3]))
    add('action_keep_previous',wave.cfg.keep_previous);add('action_ramp_seconds',wave.cfg.ramp_seconds)
    add('geom_semiaxes_m',model.geom_size[geoms].ravel())
    add('net_gravity_m_s2',model.opt.gravity,9.81)
    out=np.asarray(values,np.float32)
    assert len(out)==static_dim(n),(n,len(out),static_dim(n))
    assert np.isfinite(out).all()
    return out,names

def flow_derivative_world(env):
    # Current derivative of the imposed flow law; no future states/trajectory.
    phase=env.flow_phase+2*torch.pi*env.flow_age/env.flow_period
    return env.flow_amp*(2*torch.pi/env.flow_period)[:,None]*phase.cos()[:,None]

def dynamic_physics(env):
    f=env.scene['fish'];q=f.data.root_link_quat_w;w=env.action_manager.get_term('wave')
    p=f.data.root_link_pos_w
    target=torch.cat((env.command_manager.get_term('goal').goal_xy-p[:,:2],.5-p[:,2:3]),-1)
    flow_b=quat_apply_inverse(q,env.flow)
    derivative_b=quat_apply_inverse(q,flow_derivative_world(env))-torch.cross(f.data.root_link_ang_vel_b,flow_b,dim=-1)
    body_flow=env.flow[:,None,:]-f.data.body_com_lin_vel_w
    body_flow=quat_apply_inverse(q[:,None,:].expand(-1,body_flow.shape[1],-1).reshape(-1,4),body_flow.reshape(-1,3)).reshape(env.num_envs,-1)
    torque=env.sim.model.actuator_forcerange[:,:,1].abs().clamp_min(1e-3)
    # True orientation represented by two world axes in body frame (6D).
    ex=torch.tensor([1.,0.,0.],device=env.device).expand(env.num_envs,3)
    ey=torch.tensor([0.,1.,0.],device=env.device).expand(env.num_envs,3)
    speed=env.flow.norm(dim=-1,keepdim=True)
    rows=[quat_apply_inverse(q,target)/6,target.norm(dim=-1,keepdim=True)/6,
        quat_apply_inverse(q,ex),quat_apply_inverse(q,ey),f.data.root_link_lin_vel_w/.75,
        f.data.root_link_ang_vel_w/6,(flow_b-f.data.root_link_lin_vel_b)/.75,
        env.flow/.45,env.flow/speed.clamp_min(1e-6),speed/.45,derivative_b/.6,flow_derivative_world(env)/.6,
        env.flow_amp/.20,env.flow_period[:,None]/6,
        (env.flow_phase+2*torch.pi*env.flow_age/env.flow_period).sin()[:,None],
        (env.flow_phase+2*torch.pi*env.flow_age/env.flow_period).cos()[:,None],
        env.episode_length_buf[:,None]/env.max_episode_length,
        f.data.actuator_force/torque,(w.q_target-f.data.joint_pos)/w._range,
        f.data.joint_vel*env.step_dt/w._range,body_flow/.75]
    out=torch.cat(rows,-1)
    assert out.shape==(env.num_envs,extra_dynamic_dim(env.n)),(out.shape,extra_dynamic_dim(env.n))
    return out

def append(env,legacy):
    if not hasattr(env,'physical_bank'):
        return torch.cat((legacy,torch.zeros(env.num_envs,static_dim(env.n)+extra_dynamic_dim(env.n),device=env.device)),-1)
    return torch.cat((legacy,env.physical_bank[env.morph_ids],dynamic_physics(env)),-1)
