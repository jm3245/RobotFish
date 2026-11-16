"""Run paired upstream-only MuJoCo trials with a visible, noncolliding target cylinder."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "flow_040_navigation"))
sys.path.insert(0, str(ROOT / "controllers"))
from screen import EvaluationEnv  # noqa: E402
from blind_6d import BlindAdaptive6DPD  # noqa: E402
from heading_first_6d import HeadingFirst6DController  # noqa: E402


def install_target_cylinder(env: EvaluationEnv) -> None:
    """Compile the same servo fish and add a passive visual goal marker."""
    from core import servo_xml
    root = ET.fromstring(servo_xml())
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise RuntimeError("No MuJoCo worldbody")
    ET.SubElement(worldbody, "geom", name="target_cylinder", type="cylinder",
                  pos="0 0 0.5", size="0.35 0.5", rgba="0.1 0.9 0.25 0.23",
                  contype="0", conaffinity="0")
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    env.physics.model = model
    env.physics.data = mujoco.MjData(model)
    env.physics.head = model.body("head").id
    env.physics.substeps = round(env.physics.dt / model.opt.timestep)
    assert abs(env.physics.substeps * model.opt.timestep - env.physics.dt) < 1e-10


def run_trial(scene: dict, controller_name: str, cap: float) -> dict:
    env = EvaluationEnv(flow="constant")
    try:
        install_target_cylinder(env)
        obs, _ = env.reset(seed=0, options=scene)
        geom_id = env.physics.model.geom("target_cylinder").id
        target = np.asarray(scene["target"], dtype=float)
        env.physics.model.geom_pos[geom_id] = [*target, .5]
        mujoco.mj_forward(env.physics.model, env.physics.data)
        if controller_name == "blind_6d":
            ctrl = BlindAdaptive6DPD(obs[:6])
            base_amplitude = cap - .1
        elif controller_name == "heading_first_6d":
            ctrl = HeadingFirst6DController(obs[:6])
            base_amplitude = cap - .05
        else:
            raise ValueError(controller_name)
        env.wave.p = dict(env.wave.p)
        env.wave.p["frequency"] = 2.5
        env.wave.p["amplitude"] = base_amplitude
        env.wave.slow_near_target = False
        cylinder_first_s = None
        point_first_s = None
        min_distance_m = math.inf
        peak_amplitude = 0.
        outcome = "timeout"
        while True:
            action, _ = ctrl.action(obs[:6])
            peak_amplitude = max(peak_amplitude, float(base_amplitude + action[1]))
            obs, _, done, trunc, info = env.step(action)
            head = env.physics.data.xpos[env.physics.head]
            distance = float(np.linalg.norm(head[:2] - target))
            min_distance_m = min(min_distance_m, distance)
            valid = not info["physical_failure"]
            t = env.physics.age * env.physics.dt
            if valid and distance <= .35 and abs(float(head[2]) - .5) <= .5 and cylinder_first_s is None:
                cylinder_first_s = t
            if valid and distance < .3 and point_first_s is None:
                point_first_s = t
            if done or trunc:
                outcome = "physical_failure" if info["physical_failure"] else "point_reached" if info["is_success"] else "timeout"
                break
        assert peak_amplitude <= cap + 1e-6
        return dict(case_id=scene["case_id"], controller=controller_name, amplitude_cap_rad=cap,
                    angle_deg=scene["goal_bearing_deg"], distance_m=scene["goal_distance_m"],
                    cylinder_success=int(cylinder_first_s is not None),
                    point_success=int(point_first_s is not None),
                    cylinder_first_s=cylinder_first_s, point_first_s=point_first_s,
                    outcome=outcome, episode_s=t, min_distance_m=min_distance_m,
                    max_commanded_amplitude_rad=peak_amplitude)
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller", choices=["blind_6d", "heading_first_6d"], required=True)
    parser.add_argument("--cap", type=float, choices=[.38, .42, .75], required=True)
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()
    scenes_path = HERE / "scenes.json"
    scenes = json.loads(scenes_path.read_text(encoding="utf-8"))
    rows = []
    stem = f"{args.controller}_A{str(args.cap).replace('.', 'p')}"
    output = HERE / f"{stem}.csv"
    for i, scene in enumerate(list(scenes.values())[:args.limit], 1):
        rows.append(run_trial(scene, args.controller, args.cap))
        if i % 10 == 0 or i == args.limit:
            with output.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            print(f"{stem} {i}/{args.limit}: cylinder {sum(r['cylinder_success'] for r in rows)}, point {sum(r['point_success'] for r in rows)}", flush=True)
    summary = dict(controller=args.controller, cap_rad=args.cap, n=len(rows),
                   cylinder_arrived=sum(r["cylinder_success"] for r in rows),
                   point_arrived=sum(r["point_success"] for r in rows),
                   physical_failure=sum(r["outcome"] == "physical_failure" for r in rows),
                   timeout=sum(r["outcome"] == "timeout" for r in rows),
                   scenes_sha256=hashlib.sha256(scenes_path.read_bytes()).hexdigest())
    (HERE / f"{stem}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
