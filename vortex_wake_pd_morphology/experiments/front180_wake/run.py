"""Latest unmodified 6D PD, front semicircle cylinder targets, paired flow tests."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'flow_040_navigation')]
from screen import EvaluationEnv
from blind_6d import BlindAdaptive6DPD
from wake_preview import velocity


def field(xy, t, center):
    # Rotate preview 180 degrees: background -x; wake behind upstream target.
    return -velocity(-(np.asarray(xy)-center), t, u=.4, diameter=.7)


class LocalFlow:
    """Native inertia-box translational drag difference, held over 20 ms.

    Native uniform-flow drag and rotational drag stay active. The added force
    replaces uniform-flow translation drag with drag under each body's local
    water velocity. Geometry uses inertia-box fluid, not ellipsoid fluid.
    """
    def __init__(self, env):
        self.e = env.physics
        m = self.e.model
        assert not np.any(m.geom_fluid[:, 0])
        self.ids = np.flatnonzero(m.body_mass > 1e-12)
        inertia = m.body_inertia[self.ids]
        self.box = np.sqrt(np.maximum(1e-15, inertia.sum(axis=1)[:, None]-2*inertia)
                           / m.body_mass[self.ids, None]*6)
        self.areas = np.stack([self.box[:,1]*self.box[:,2], self.box[:,0]*self.box[:,2], self.box[:,0]*self.box[:,1]], axis=-1)
        self.visc = 3*np.pi*self.box.mean(axis=1)[:,None]*m.opt.viscosity

    def correction(self, water):
        m, d = self.e.model, self.e.data
        rotations = d.ximat[self.ids].reshape(-1,3,3)
        speed = np.empty((len(self.ids),3))
        for j, bid in enumerate(self.ids):
            local = np.zeros(6)
            mujoco.mj_objectVelocity(m,d,mujoco.mjtObj.mjOBJ_BODY,int(bid),local,1)
            speed[j] = local[3:]
        bg = speed - np.einsum('bji,j->bi', rotations, m.opt.wind)
        rel = speed - np.einsum('bji,bj->bi', rotations, water)
        def force(v):
            return -self.visc*v - .5*m.opt.density*self.areas*np.abs(v)*v
        delta = np.einsum('bij,bj->bi',rotations,force(rel)-force(bg))
        return delta

    def update(self, center):
        d = self.e.data
        water = np.zeros((len(self.ids),3))
        water[:,:2] = field(d.xipos[self.ids,:2],d.time,center)
        d.xfrc_applied[:] = 0
        d.xfrc_applied[self.ids,:3] = self.correction(water)


def scenes():
    out = {}
    # 25 equal angular bins across front 180 degrees; use bin centers.
    for i, angle in enumerate(np.linspace(-86.4,86.4,25)):
        for j, distance in enumerate([2.,3.,4.,5.]):
            a = np.deg2rad(angle)
            key = f'F{i:02d}_{j:02d}'
            out[key] = dict(case_id=key, start=[0.,0.], target=[float(distance*np.cos(a)),float(distance*np.sin(a))],
                            yaw=0., qpos=[0.,0.,.5,1.,0.,0.,0.,0.,0.,0.,0.],qvel=[0.]*10,
                            flow='constant',flow_base=[-.4,0.,0.],flow_phase=0.,flow_freq=.25,
                            angle_deg=float(angle),distance_m=distance)
    return out


def trial(scene, cap, mode, trace=False):
    from core import servo_xml
    env = EvaluationEnv(flow='constant')
    try:
        root = ET.fromstring(servo_xml())
        # Fish collision is enabled only against the target (bit 2).
        for body in root.findall('.//body'):
            for g in body.findall('geom'):
                g.set('contype','1'); g.set('conaffinity','2')
        ET.SubElement(root.find('worldbody'),'geom',name='reach_cylinder',type='cylinder',
                      pos=f"{scene['target'][0]} {scene['target'][1]} .5",size='.35 .5',
                      contype='2',conaffinity='1',rgba='.8 .35 .15 1')
        m = mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'))
        env.physics.model=m; env.physics.data=mujoco.MjData(m); env.physics.head=m.body('head').id
        obs,_=env.reset(seed=0,options=scene)
        ctrl=BlindAdaptive6DPD(obs[:6])
        env.wave.p=dict(env.wave.p);env.wave.p['frequency']=2.5;env.wave.p['amplitude']=cap-.1
        env.wave.slow_near_target=False
        hydro=LocalFlow(env)
        e=env.physics; d=e.data
        gid=m.geom('reach_cylinder').id
        head_geoms=[g for g in range(m.ngeom) if m.geom_bodyid[g]==e.head]
        closest=1e9; records=[];success=False;outcome='timeout';peak_a=0.
        while True:
            if mode=='wake': hydro.update(np.asarray(scene['target']))
            action,_=ctrl.action(obs[:6]);peak_a=max(peak_a,float(cap-.1+action[1]))
            obs,_,done,trunc,info=env.step(action)
            gap=min(mujoco.mj_geomDistance(m,d,g,gid,100.,None) for g in head_geoms)
            closest=min(closest,gap)
            if trace or e.age%5==0:
                records.append([d.time,*d.xpos[e.head,:3],float(obs[2]),float(action[0]),float(cap-.1+action[1]),gap])
            if info['physical_failure']:outcome='physical_failure';break
            if gap<=.05:success=True;outcome='reached_surface';break
            if done or trunc:break
        assert peak_a<=cap+1e-6
        row=dict(case_id=scene['case_id'],mode=mode,cap_rad=cap,angle_deg=scene['angle_deg'],distance_m=scene['distance_m'],
                 success=int(success),outcome=outcome,seconds=e.age*e.dt,min_head_surface_gap_m=closest,max_A_rad=peak_a)
        return row,np.asarray(records)
    finally:env.close()


def main():
    p=argparse.ArgumentParser();p.add_argument('--cap',type=float,choices=[.38,.42,.75],required=True)
    p.add_argument('--mode',choices=['uniform','wake'],required=True);p.add_argument('--limit',type=int,default=100)
    a=p.parse_args();HERE.mkdir(exist_ok=True)
    cases=scenes();(HERE/'scenes.json').write_text(json.dumps(cases,indent=2),encoding='utf-8')
    rows=[];traces={};stem=f'{a.mode}_A{a.cap:.2f}'
    for i,sc in enumerate(list(cases.values())[:a.limit],1):
        row,trajectory=trial(sc,a.cap,a.mode);rows.append(row);traces[sc['case_id']]=trajectory
        if i%10==0 or i==a.limit:
            with (HERE/f'{stem}.csv').open('w',newline='',encoding='utf-8') as f:
                w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
            print(f'{stem}: {i}/{a.limit}, reached={sum(r["success"] for r in rows)}',flush=True)
    np.savez_compressed(HERE/f'{stem}_traces.npz',**traces)
    result=dict(n=len(rows),arrived=sum(r['success'] for r in rows),mode=a.mode,cap_rad=a.cap,
                physical_failure=sum(r['outcome']=='physical_failure' for r in rows),
                timeout=sum(r['outcome']=='timeout' for r in rows),
                controller_sha256=hashlib.sha256((ROOT/'flow_040_navigation/blind_6d.py').read_bytes()).hexdigest())
    (HERE/f'{stem}_summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(result)

if __name__=='__main__':main()
