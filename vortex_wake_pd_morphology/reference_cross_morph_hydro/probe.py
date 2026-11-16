"""Read-only physical interface probe for the separate hydro experiment."""
import json
import torch
from .environment import DiverseFishEnv
from .interaction import InteractionSensor

env = DiverseFishEnv(3, 1, random_morph=False)
try:
    m, d = env.sim.wp_model, env.sim.wp_data
    sensor = InteractionSensor(env)
    env.flow[:] = torch.tensor([[.25, -.1, .02]], device=env.device)
    env.sim.forward()
    projection = sensor.check_projection()
    feature = sensor.read()
    env.step(torch.ones((1, 4), device=env.device) * .3)
    post_without_forward = sensor.read().clone()
    env.sim.forward()
    post_with_forward = sensor.read().clone()
    print(json.dumps({
        'nbody': int(m.nbody), 'nv': int(m.nv),
        'model_fluid': bool(m.has_fluid),
        'model_body_rootid': m.body_rootid.numpy().tolist(),
        'model_body_mass_shape': list(m.body_mass.shape),
        'qfrc_fluid_shape': list(d.qfrc_fluid.shape),
        'qfrc_actuator_shape': list(d.qfrc_actuator.shape),
        'cfrc_ext_shape': list(d.cfrc_ext.shape),
        'data_xipos_shape': list(d.xipos.shape),
        'cpu_model_body_names': [env.sim.mj_model.body(i).name for i in range(env.sim.mj_model.nbody)] if hasattr(env.sim, 'mj_model') else None,
        'sim_keys': [x for x in dir(env.sim) if 'model' in x or 'data' in x],
        'option_density': m.opt.density.numpy().tolist(),
        'option_viscosity': m.opt.viscosity.numpy().tolist(),
        'projection': projection,
        'feature_shape': list(feature.shape),
        'feature': feature[0].tolist(),
        'post_forward_delta': float((post_without_forward-post_with_forward).abs().max()),
    }, default=str), flush=True)
finally:
    env.close()
