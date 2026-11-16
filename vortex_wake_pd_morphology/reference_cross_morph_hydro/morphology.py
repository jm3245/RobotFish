"""Compile physical size variants without editing any legacy XML/checkpoint."""
from dataclasses import asdict, replace
import json, math
import numpy as np
import mujoco
from fish_morphologies.generate_fish import FishConfig, generate
from .config import RUN, write

# Fixed training bank allows genuine held-out sizes, plus random independent
# lengths/profiles/tails for each N. Index 0 deliberately preserves legacy size.
def parameters(n, ood=False):
    # N8 ID uses size parameters already seen on N3, so size shift is separable
    # from topology shift. No N8 states enter fitting or model selection.
    rng=np.random.default_rng(20261020+101*(3 if n==8 else n)+int(ood)*10000)
    lengths=[.50,.56,1.62,1.75] if ood else [1.,.65,.72,.80,.88,.96,1.05,1.13,1.21,1.29,1.37,1.45]
    rows=[]
    for i,L in enumerate(lengths):
        w,h,tail,prop=(1.,1.,1.,1.) if i==0 and not ood else (rng.uniform(.72,1.38),rng.uniform(.78,1.30),rng.uniform(.65,1.50),rng.uniform(.78,1.25))
        ratios=tuple((1.4-.7*j/n)**prop for j in range(n+1))
        scale=L**3*w*h
        cfg=replace(FishConfig(),n_joints=n,body_length=L,length_ratios=ratios,
            width_profile=tuple(x*L*w for x in FishConfig.width_profile),
            height_profile=tuple(x*L*h for x in FishConfig.height_profile),
            tail_fin_area=.018*L*L*tail,tail_fin_height=.18*L*math.sqrt(tail),
            joint_damping=.015*scale*L**2,joint_stiffness=.01*scale*L,
            joint_armature=.0005*scale*L**2,
            actuator_kp=35*scale*L,actuator_kv=.6*scale*L**2,force_limit=12*scale*L)
        rows.append((cfg,dict(id=i,n=n,ood=ood,length=L,width_scale=w,height_scale=h,tail_scale=tail,proportion=prop)))
    return rows

def entity_xml(n,index,ood=False):
    import xml.etree.ElementTree as ET
    cfg,meta=parameters(n,ood)[index]
    dest=RUN/'morphologies'/f'N{n}'/('ood' if ood else 'train')/f'{index:02d}.xml'
    if not dest.exists():
        generate(cfg,dest)
        root=ET.parse(dest).getroot();world=root.find('worldbody')
        for child in list(world):
            if child.tag!='body' or child.get('name')!='head':world.remove(child)
        # Scene owns the water/visual settings; entity options are not duplicated.
        for tag in ('option','visual'):
            child=root.find(tag)
            if child is not None:root.remove(child)
        ET.ElementTree(root).write(dest,encoding='utf-8',xml_declaration=True)
        m=mujoco.MjModel.from_xml_path(str(dest))
        assert np.all(m.body_mass[1:]>0) and np.all(m.body_inertia[1:]>0)
        volumes=4*np.pi/3*np.prod(m.geom_size,axis=1)
        assert abs(m.body_mass.sum()-1000*volumes.sum())<1e-5
        assert np.all(m.body_inertia[1:].sum(1)[:,None]-2*m.body_inertia[1:]>-1e-9)
        write(dest.with_suffix('.json'),dict(**meta,total_mass_kg=float(m.body_mass.sum()),
            displaced_volume_m3=float(volumes.sum()),buoyant_force_N=float(volumes.sum()*1000*9.81),
            weight_N=float(m.body_mass.sum()*9.81),config=asdict(cfg)))
    return dest

FIELDS=('body_pos','body_ipos','body_iquat','body_mass','body_inertia',
    'geom_pos','geom_quat','geom_size','geom_rbound','geom_aabb','geom_fluid',
    'site_pos','dof_damping','dof_armature','jnt_stiffness',
    'actuator_gainprm','actuator_biasprm','actuator_forcerange',
    'body_subtreemass','dof_invweight0','body_invweight0','actuator_acc0')
