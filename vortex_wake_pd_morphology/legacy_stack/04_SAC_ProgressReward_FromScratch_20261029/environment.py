"""Independent reward wrapper around the frozen 8D/1D steering environments."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "03_SAC_PlanarFlow_0p2_0p3_20261029"
spec = importlib.util.spec_from_file_location("frozen_planar_sac_environment", SOURCE / "environment.py")
planar = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(planar)

CFG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
ROOT, BASELINE, astra, PARAMS = planar.ROOT, planar.BASELINE, planar.astra, planar.PARAMS
SteeringWave = planar.SteeringWave


class ProgressReward:
    """Change only the outer SAC reward; parent dynamics and interfaces are untouched."""

    def step(self, action):
        e = self.physics
        previous_distance = float(np.linalg.norm(e.target - e.data.xpos[e.head, :2]))
        obs, legacy_reward, term, trunc, info = super().step(action)
        progress = previous_distance - float(info["distance"])
        success = bool(info["is_success"])
        physical_failure = bool(info["physical_failure"])
        parts = {
            "progress": 10.0 * progress,
            "step": -0.001,
            "success": 30.0 * success,
            "physical_failure": -10.0 * physical_failure,
        }
        reward = float(sum(parts.values()))
        self.reward_sum += reward - float(legacy_reward)
        info["steering_reward_parts"] = parts
        if term or trunc:
            info["result"]["episode_return"] = float(self.reward_sum)
        return obs, reward, term, trunc, info


class SteeringEnv(ProgressReward, planar.SteeringEnv):
    """Training: exact original 0.2–0.3 m/s planar flow sampler."""


class EvaluationEnv(ProgressReward, planar.original.SteeringEnv):
    """Evaluation: same physics and task, with fixed 0/0.1/0.2/0.3 m/s flow."""

