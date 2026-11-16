"""Validate exact Warp fluid wrench projection for every training topology."""
import json
import torch
from .config import RUN,TRAIN_N,DEVICE,write
from .environment import DiverseFishEnv
from .interaction import InteractionSensor,FEATURE_DIM

def main():
    rows={}
    for n in TRAIN_N:
        env=DiverseFishEnv(n,2,random_morph=False)
        try:
            sensor=InteractionSensor(env)
            env.set_morph(torch.arange(2,device=DEVICE),torch.tensor([0,11],device=DEVICE))
            env.flow[:]=torch.tensor([[.2,-.1,.015],[-.35,.12,-.025]],device=DEVICE)
            env.sim.forward()
            projection=sensor.check_projection()
            feature=sensor.read()
            assert projection['passed'] and projection['fluid_l2_N']>0
            assert tuple(feature.shape)==(2,FEATURE_DIM)
            assert torch.isfinite(feature).all()
            rows[str(n)]=dict(projection=projection,feature_min=float(feature.min()),
                feature_max=float(feature.max()),body_count=n+1)
        finally:env.close()
    write(RUN/'physics_preflight.json',dict(passed=True,rows=rows,
        physical_model='MuJoCo Warp 3.11 inertia-box; uniform time-varying wind, no pressure field',
        method='call same private fluid kernel, check Jacobian projection equals qfrc_fluid'))
    print(json.dumps(dict(passed=True,rows=rows)),flush=True)

if __name__=='__main__':main()
