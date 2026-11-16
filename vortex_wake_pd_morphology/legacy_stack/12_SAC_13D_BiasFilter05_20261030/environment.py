"""13D SAC. The only change from 07 is the raw-bias filter: 0.5 previous + 0.5 current."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "07_SAC_PDDesired_FishVelocity_13D_20261029"
spec = importlib.util.spec_from_file_location("frozen_13d_sac_environment", SOURCE / "environment.py")
base = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(base)

CFG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
ROOT, BASELINE, astra = base.ROOT, base.BASELINE, base.astra
WaveBase = base.base.base.SteeringWave
PREVIOUS, CURRENT = CFG["bias_filter"]
assert (PREVIOUS, CURRENT) == (0.5, 0.5)
assert len(CFG["observation_order"]) == 13


class HalfBiasWave(WaveBase):
    """Same traveling wave as the frozen 13D environment. Only the bias filter changes."""

    def action_with_bias(self, e, bias):
        p = self.p
        distance = np.linalg.norm(e.target - e.data.xpos[e.head, :2])
        self.last_bias = PREVIOUS * self.last_bias + CURRENT * float(np.clip(bias, -.3, .3))
        slowdown = np.clip(distance / p.get("slow_radius", .8), p.get("min_frequency_ratio", .4), 1.)
        self.phase += 2 * np.pi * p.get("frequency", 1.5) * slowdown * e.dt
        amplitude = p.get("amplitude", .5)
        profile = p.get("head_ratio", .5) + (1 - p.get("head_ratio", .5)) * self.index / 3
        q = self.last_bias + amplitude * profile * np.sin(self.phase - self.index * p.get("lag", 1.0))
        ramp = min((e.age + 1) * e.dt / .4, 1.)
        return np.clip(q * ramp, -.75, .75)


class InstallHalfBias:
    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self.wave = HalfBiasWave(self.wave.p)
        assert isinstance(self.wave, HalfBiasWave)
        assert obs.shape == (13,)
        return obs, info


class SteeringEnv(InstallHalfBias, base.SteeringEnv):
    """Training environment: 13D observation, progress reward, 0.5/0.5 raw-bias filter."""


class EvaluationEnv(InstallHalfBias, base.EvaluationEnv):
    """Fixed-flow evaluation with the same filter."""
