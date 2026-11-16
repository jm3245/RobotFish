"""13D observation, 2D SAC action: raw steering bias and amplitude offset."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
from gymnasium import spaces

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "12_SAC_13D_BiasFilter05_20261030"
spec = importlib.util.spec_from_file_location("frozen_13d_half_bias_environment", SOURCE / "environment.py")
base = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(base)

CFG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
ROOT, BASELINE, astra = base.ROOT, base.BASELINE, base.astra
assert len(CFG["observation_order"]) == 13
assert CFG["bias_filter"] == [0.5, 0.5]


class BiasAmplitudeWave(base.HalfBiasWave):
    def __init__(self, params):
        super().__init__(params)
        self.amplitude_delta_rad = 0.0
        self.slow_near_target = bool(CFG["near_target_frequency_slowdown"])

    def action_with_bias(self, e, bias):
        p = self.p
        self.last_bias = 0.5 * self.last_bias + 0.5 * float(np.clip(bias, -.3, .3))
        if self.slow_near_target:
            distance = np.linalg.norm(e.target - e.data.xpos[e.head, :2])
            slowdown = np.clip(distance / p.get("slow_radius", .8), p.get("min_frequency_ratio", .4), 1.)
        else:
            slowdown = 1.0
        self.phase += 2 * np.pi * p.get("frequency", 1.5) * slowdown * e.dt
        amplitude = p.get("amplitude", .5) + self.amplitude_delta_rad
        profile = p.get("head_ratio", .5) + (1 - p.get("head_ratio", .5)) * self.index / 3
        q = self.last_bias + amplitude * profile * np.sin(self.phase - self.index * p.get("lag", 1.0))
        ramp = min((e.age + 1) * e.dt / .4, 1.)
        return np.clip(q * ramp, -.75, .75)


class TwoActionControl:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_space = spaces.Box(
            low=np.array([-.3, -.1], dtype=np.float32),
            high=np.array([.3, .1], dtype=np.float32), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self.wave = BiasAmplitudeWave(self.wave.p)
        if options is not None and "near_target_frequency_slowdown" in options:
            self.wave.slow_near_target = bool(options["near_target_frequency_slowdown"])
        assert obs.shape == (13,)
        return obs, info

    def step(self, action):
        a = np.asarray(action, dtype=float)
        if a.shape != (2,) or not np.isfinite(a).all():
            raise ValueError("Expected finite [raw steering bias rad, amplitude offset rad]")
        bias = float(np.clip(a[0], -.3, .3))
        delta = float(np.clip(a[1], -.1, .1))
        self.wave.amplitude_delta_rad = delta
        obs, reward, terminated, truncated, info = super().step(np.asarray([bias], dtype=np.float32))
        info["steering_bias_action_rad"] = bias
        info["amplitude_offset_action_rad"] = delta
        info["amplitude_target_rad"] = float(self.wave.p.get("amplitude", .5) + delta)
        return obs, reward, terminated, truncated, info


class SteeringEnv(TwoActionControl, base.SteeringEnv):
    """Training environment with bias and amplitude actions."""


class EvaluationEnv(TwoActionControl, base.EvaluationEnv):
    """Fixed-flow evaluation with identical actions."""
