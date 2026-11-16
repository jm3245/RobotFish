"""Original 8D -> raw b_t SAC steering, with uniform 2D flow in [0.2, 0.3] m/s."""
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ORIGINAL = HERE.parent / "02_SAC_Based_on_Baseline"
spec = importlib.util.spec_from_file_location("original_sac_environment", ORIGINAL / "environment.py")
original = importlib.util.module_from_spec(spec)
spec.loader.exec_module(original)

ROOT = original.ROOT
BASELINE = original.BASELINE
astra = original.astra
PARAMS = original.PARAMS
SteeringWave = original.SteeringWave
CFG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
FLOWS = CFG["training_flows"]
assert CFG["reward"] == original.CFG["reward"]
assert CFG["success_xy_radius_m"] == original.CFG["success_xy_radius_m"]


class SteeringEnv(original.SteeringEnv):
    """Only override flow sampling; inherit SAC bias, waveform, reward, and success."""

    def __init__(self, flow=None):
        if flow not in (None, "constant"):
            raise ValueError("Only constant planar flow is allowed")
        super().__init__(flow="constant")
        self._flow_rng = np.random.default_rng()

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._flow_rng = np.random.default_rng(int(seed) ^ 0x5A17C0DE)
        options = dict(options or {})
        if options.get("flow", "constant") != "constant":
            raise ValueError("Only constant planar flow is allowed")
        options["flow"] = "constant"
        if "flow_base" not in options:
            speed = float(self._flow_rng.uniform(0.2, 0.3))
            angle = float(self._flow_rng.uniform(0.0, 2 * math.pi))
            options["flow_base"] = [speed * math.cos(angle), speed * math.sin(angle), 0.0]
        wind = np.asarray(options["flow_base"], dtype=float)
        assert wind.shape == (3,) and abs(wind[2]) < 1e-12
        assert 0.2 - 1e-12 <= np.linalg.norm(wind[:2]) <= 0.3 + 1e-12
        obs, info = super().reset(seed=seed, options=options)
        assert self.physics.flow == "constant"
        assert np.allclose(self.physics.model.opt.wind, wind, atol=1e-12)
        assert np.allclose(obs[6:], wind[:2], atol=1e-7)
        return obs, info
