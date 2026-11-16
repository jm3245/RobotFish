"""11D SAC observation: original 8D plus frozen PD desired velocity and heading."""
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
from gymnasium import spaces

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "04_SAC_ProgressReward_FromScratch_20261029"
spec = importlib.util.spec_from_file_location("unchanged_8d_progress_sac_environment", SOURCE / "environment.py")
base = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(base)

CFG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
ROOT, BASELINE, astra, PARAMS = base.ROOT, base.BASELINE, base.astra, base.PARAMS
PD_PARAMETERS = PARAMS["constant"]
assert PD_PARAMETERS["speed"] == CFG["pd_desired_speed_m_s"] == 0.25
assert PD_PARAMETERS["flow_gain"] == CFG["pd_flow_gain"] == 1.2
assert len(CFG["observation_order"]) == 11


def pd_desired_features(e) -> np.ndarray:
    """Exact v_desired/heading expressions from Baseline Controller.action.

    These features are observational only: Controller.action is never invoked,
    so its yaw filter, phase and steering state are not changed.
    """
    p = PD_PARAMETERS
    pos = e.data.xpos[e.head]
    delta = e.target - pos[:2]
    distance = np.linalg.norm(delta)
    wind = e.model.opt.wind[:2]
    speed = p.get("speed", .25)
    desired = speed * delta / max(distance, 1e-8) - p.get("flow_gain", 0.) * wind
    heading = math.atan2(desired[1], desired[0])
    return np.asarray([desired[0], desired[1], heading], dtype=np.float32)


class PDDesiredObservation:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._base_reset_observation = False
        self.observation_space = spaces.Box(-np.inf, np.inf, (11,), np.float32)

    def reset(self, *, seed=None, options=None):
        # The frozen planar reset verifies its own two flow entries with
        # obs[6:]. Give that inherited check the unchanged 8D vector, then
        # return the 11D vector to SAC after reset has finished.
        self._base_reset_observation = True
        try:
            _, info = super().reset(seed=seed, options=options)
        finally:
            self._base_reset_observation = False
        return self.observation(), info

    def observation(self):
        original = super().observation()
        assert original.shape == (8,)
        if self._base_reset_observation:
            return original
        return np.concatenate((original, pd_desired_features(self.physics))).astype(np.float32)


class SteeringEnv(PDDesiredObservation, base.SteeringEnv):
    """Training environment: same reward, dynamics, action and flow sampler as 8D."""


class EvaluationEnv(PDDesiredObservation, base.EvaluationEnv):
    """Fixed evaluation conditions, with the same 11D observation."""
