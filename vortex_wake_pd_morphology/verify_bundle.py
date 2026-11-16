"""Check portability, file integrity and two representative PD episodes."""
from __future__ import annotations

import hashlib
import csv
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "flow_040_navigation"))
from blind_6d_benchmark import run  # noqa: E402


def main() -> None:
    manifest = json.loads((ROOT / "source_manifest.json").read_text(encoding="utf-8"))
    for entry in manifest["files"]:
        path = ROOT / entry["path"]
        assert path.is_file(), entry["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"], entry["path"]
    print(f"Manifest: {len(manifest['files'])} files verified")

    for n in (3, 4, 5, 6, 8):
        model = mujoco.MjModel.from_xml_path(str(ROOT / "fish_morphologies" / f"fish_N{n}.xml"))
        assert model.nu == n and model.opt.density == 1000
    print("Morphologies: N3, N4, N5, N6, N8 compiled")

    old = json.loads((ROOT / "structured_100_040/scenes.json").read_text(encoding="utf-8"))
    new = json.loads((ROOT / "structured_100_050/scenes.json").read_text(encoding="utf-8"))
    assert old.keys() == new.keys() and len(old) == 100
    for case_id in old:
        assert old[case_id]["target"] == new[case_id]["target"]
        assert old[case_id]["yaw"] == new[case_id]["yaw"]
        assert np.allclose(np.asarray(old[case_id]["flow_base"])*1.25,
                           new[case_id]["flow_base"])
    print("Scene sets: 100 paired cases, flow vectors scaled 0.4 -> 0.5 m/s")

    current_controller_sha = hashlib.sha256(
        (ROOT / "flow_040_navigation/blind_6d.py").read_bytes()).hexdigest()
    prefixes = ("blind_6d_final", "blind_6d_cap_042", "blind_6d_cap_038",
                "blind_6d_050_a075", "blind_6d_050_a042", "blind_6d_050_a038")
    for prefix in prefixes:
        result = json.loads((ROOT / "reference_results" / f"{prefix}_summary.json").read_text())
        assert result["n"] == 100 and result["controller_sha256"] == current_controller_sha
        assert result["arrived"] + result["timeout"] + result["physical_failure"] == 100
    print("Reference results: six 100-scene summaries match the bundled 6D PD")

    for folder in ("structured_100_040", "structured_100_050"):
        scenes = json.loads((ROOT / folder / "scenes.json").read_text(encoding="utf-8"))
        row = run("S00_00", scenes=scenes, max_amplitude=.75, frequency_hz=2.5)
        assert row["success"] == 1 and row["physical_failure"] == 0
        assert row["max_commanded_amplitude_rad"] <= .75 + 1e-6
        print(f"{folder}: S00_00 arrived in {row['seconds']:.2f} s")

    from wake_preview import gradient, velocity
    probes = np.array([[1., .4], [1., -.4]])
    flow = velocity(probes, 1.)
    assert flow.shape == (2, 2) and np.isfinite(flow).all()
    assert gradient(probes, 1.).shape == (2, 2, 2)
    print("Wake preview: velocity and gradient sampling finite")

    wake_root = ROOT / "experiments/front180_wake"
    wake_scenes = json.loads((wake_root / "scenes.json").read_text())
    assert len(wake_scenes) == 100
    for mode in ("uniform", "wake"):
        for cap in (.38, .42, .75):
            stem = f"{mode}_A{cap:.2f}"
            summary = json.loads((wake_root / f"{stem}_summary.json").read_text())
            with (wake_root / f"{stem}.csv").open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            assert len(rows) == summary["n"] == 100
            assert summary["controller_sha256"] == current_controller_sha
            assert sum(int(row["success"]) for row in rows) == summary["arrived"]
            assert summary["arrived"] + summary["timeout"] + summary["physical_failure"] == 100
            assert {row["case_id"] for row in rows} == set(wake_scenes)
            with np.load(wake_root / f"{stem}_traces.npz") as trajectories:
                assert set(trajectories.files) == set(wake_scenes)
    print("Forward 180-degree suite: all 600 rows and 600 trajectories verified")


if __name__ == "__main__":
    main()
