"""Compare local-flow drag correction with MuJoCo's native fluid-force change."""
import json
import mujoco
import numpy as np
from run import EvaluationEnv, LocalFlow, HERE, scenes, field

env=EvaluationEnv(flow='constant')
env.reset(seed=0,options=next(iter(scenes().values())))
m,d=env.physics.model,env.physics.data
d.qvel[:]=np.random.default_rng(42).normal(0,.1,m.nv)
mujoco.mj_forward(m,d)
hydro=LocalFlow(env)
base=d.qfrc_fluid.copy()
water=np.tile([-.25,.12,0.],(len(hydro.ids),1))
delta=hydro.correction(water)
mapped=np.zeros(m.nv)
for bid,f in zip(hydro.ids,delta):
    mujoco.mj_applyFT(m,d,f,np.zeros(3),d.xipos[bid],int(bid),mapped)
m.opt.wind[:]=water[0]
mujoco.mj_passive(m,d)
error=float(np.max(np.abs(d.qfrc_fluid-base-mapped)))
assert error<1e-10,error
# Same local and background current adds zero force.
assert np.max(np.abs(hydro.correction(water)))<1e-12
# Rotation maps remote background to -x.
remote=field(np.array([[-1000.,1000.]]),0.,np.zeros(2))[0]
assert np.linalg.norm(remote-[-.4,0.])<1e-3
result=dict(native_uniform_force_difference_max_error=error,
            equal_flow_correction_zero=True,remote_background_m_s=remote.tolist(),
            temporal_scheme='local drag force sampled each 0.02 s and held for 20 native 0.001 s substeps')
(HERE/'coupling_check.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(result)
env.close()
