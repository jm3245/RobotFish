"""Freeze 100 headwind-only paired MuJoCo navigation scenes."""
from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
ANGLES = (-54, -42, -30, -18, -6, 6, 18, 30, 42, 54)
DISTANCES = (1.5, 2., 2.5, 3., 3.5, 4., 4.5, 5., 5.5, 6.)
FLOW = (-.4, 0., 0.)


def main() -> None:
    template = json.loads((ROOT / "structured_100_040/scenes.json").read_text(encoding="utf-8"))["S00_00"]
    scenes = {}
    for i, angle_deg in enumerate(ANGLES):
        for j, distance in enumerate(DISTANCES):
            a = math.radians(angle_deg)
            goal = [distance * math.cos(a), distance * math.sin(a)]
            case_id = f"U{i:02d}_{j:02d}"
            scene = dict(template)
            scene.update(case_id=case_id, start=[0., 0.], target=goal, yaw=0.,
                         qpos=[0., 0., .5, 1., 0., 0., 0., 0., 0., 0., 0.],
                         qvel=[0.] * 10, flow="constant", flow_base=list(FLOW),
                         flow_phase=0., flow_freq=.25,
                         goal_bearing_deg=angle_deg,
                         goal_distance_m=distance,
                         flow_world_deg=180,
                         flow_relative_deg=(180-angle_deg) % 360,
                         initial_yaw_offset_deg=-angle_deg)
            assert goal[0] > 0 and FLOW[0] * goal[0] < 0
            scenes[case_id] = scene
    assert len(scenes) == 100
    (HERE / "scenes.json").write_text(json.dumps(scenes, ensure_ascii=False, indent=2), encoding="utf-8")
    protocol = {
        "name": "upstream_cylinder_040_v1", "n": 100,
        "flow_m_s": list(FLOW), "flow_description": "constant uniform world-frame current toward -x",
        "start_xy_m": [0., 0.], "fish_initial_heading_deg": 0,
        "goal_bearings_deg": list(ANGLES), "goal_distances_m": list(DISTANCES),
        "design": "10 goal bearings (five left, five right) x 10 distances",
        "goal_x_positive": True, "flow_dot_goal_negative": True,
        "signed_flow_relative_goal_range_deg": [126, 234],
        "note_on_angle": "Unsigned angle between two vectors is at most 180 deg; signed relative direction spans 126..234 deg and every case has an upstream component.",
        "cylinder_target_radius_m": .35, "cylinder_target_height_m": 1.,
        "cylinder_center_z_m": .5, "cylinder_target_collidable": False,
        "cylinder_success": "fish head first enters the vertical cylinder: XY distance <=0.35 m and |z-0.5|<=0.5 m, without physical failure on that step",
        "point_baseline_success": "original environment first head XY distance <0.30 m, without physical failure on that step",
        "trial_limit_s": 60., "control_dt_s": .02,
        "controllers": ["BlindAdaptive6DPD", "HeadingFirst6DController"],
        "frequency_hz": 2.5, "max_commanded_amplitudes_rad": [.38, .42, .75],
        "clarification": "0.557 m/s was a still-water average forward speed measured with Amax=0.75 rad at 2.5 Hz. It is not an amplitude or a hard speed cap.",
        "wake_enabled": False,
    }
    (HERE / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(scenes)} upstream-only 0.4 m/s scenes")


if __name__ == "__main__":
    main()
