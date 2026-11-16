"""13D SAC observation: unchanged 11D plus whole-fish world XY COM velocity."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
from gymnasium import spaces

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "06_SAC_PDDesired_11D_RuntimeMatched_20261029"
spec = importlib.util.spec_from_file_location("frozen_11d_sac_environment", SOURCE / "environment.py")
base = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(base)

PD_SOURCE = HERE.parent / "05_PD_FishVelocity_Feedback" / "controller.py"
pd_spec = importlib.util.spec_from_file_location("frozen_pd_fish_velocity", PD_SOURCE)
pd_velocity = importlib.util.module_from_spec(pd_spec)
assert pd_spec.loader is not None
pd_spec.loader.exec_module(pd_velocity)
fish_com_velocity_world = pd_velocity.fish_com_velocity_world

CFG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
ROOT, BASELINE, astra = base.ROOT, base.BASELINE, base.astra
assert len(CFG["observation_order"]) == 13


class FishVelocityObservation:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.observation_space = spaces.Box(-np.inf, np.inf, (13,), np.float32)

    def observation(self):
        original = super().observation()
        # The inherited planar reset must see its original 8D vector while it
        # validates flow entries. PDDesiredObservation.reset then returns 13D.
        if self._base_reset_observation:
            assert original.shape == (8,)
            return original
        assert original.shape == (11,)
        velocity = fish_com_velocity_world(self.physics)
        return np.concatenate((original, velocity.astype(np.float32))).astype(np.float32)


class SteeringEnv(FishVelocityObservation, base.SteeringEnv):
    """Only observation differs from the frozen 11D training environment."""


class EvaluationEnv(FishVelocityObservation, base.EvaluationEnv):
    """Only observation differs from the frozen 11D evaluation environment."""
