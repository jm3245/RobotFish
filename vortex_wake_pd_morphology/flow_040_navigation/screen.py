"""Local adapter for the frozen 6D navigation benchmark."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "legacy_stack" / "13_SAC_13D_BiasAmplitudeFilter05_20261104"))
from environment import EvaluationEnv  # noqa: E402

SCENES = json.loads((ROOT / "structured_100_040" / "scenes.json").read_text(encoding="utf-8"))
