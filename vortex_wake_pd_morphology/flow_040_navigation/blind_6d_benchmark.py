"""Evaluate the blind 6D controller on the frozen 0.4 m/s scene suite."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from blind_6d import BlindAdaptive6DPD
from screen import SCENES, EvaluationEnv

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def run(case_id, *, params=None, scenes=SCENES, max_amplitude=None,
        frequency_hz=2.5):
    scene = scenes[case_id]
    env = EvaluationEnv(flow="constant")
    try:
        obs, _ = env.reset(seed=0, options=scene)
        ctrl = BlindAdaptive6DPD(obs[:6], **(params or {}))
        env.wave.p = dict(env.wave.p)
        env.wave.p["frequency"] = frequency_hz
        if max_amplitude is not None:
            env.wave.p["amplitude"] = max_amplitude - .1
        flow = np.asarray(scene["flow_base"][:2], dtype=float)  # metrics only
        data, head = env.physics.data, env.physics.head
        last_com = data.subtree_com[head, :2].copy()
        peak_relative = peak_ground = peak_amplitude = 0.0
        relative_above_040_steps = 0
        minimum = float(np.linalg.norm(obs[:2] - obs[4:6]))
        while True:
            action, _ = ctrl.action(obs[:6])
            peak_amplitude = max(peak_amplitude, float(env.wave.p["amplitude"] + action[1]))
            obs, _, done, trunc, info = env.step(action)
            com = data.subtree_com[head, :2].copy()
            ground_velocity = (com - last_com) / env.physics.dt
            relative_speed = float(np.linalg.norm(ground_velocity - flow))
            peak_relative = max(peak_relative, relative_speed)
            peak_ground = max(peak_ground, float(np.linalg.norm(ground_velocity)))
            relative_above_040_steps += relative_speed > .4 + 1e-8
            last_com = com
            minimum = min(minimum, float(np.linalg.norm(obs[:2] - obs[4:6])))
            if done or trunc:
                break
        result = info["result"]
        final_head = data.xpos[head].copy()
        return dict(case_id=case_id, flow_relative_deg=scene["flow_relative_deg"],
                    success=int(result["success"]), timeout=int(trunc),
                    physical_failure=int(result["physical_failure"]),
                    seconds=float(result["seconds"]),
                    final_distance_m=float(result["final_distance_m"]),
                    min_distance_m=minimum,
                    max_depth_deviation_m=float(result["max_depth_deviation_m"]),
                    final_head_xy_radius_m=float(np.linalg.norm(final_head[:2])),
                    max_commanded_amplitude_rad=peak_amplitude,
                    peak_relative_water_com_speed_m_s=peak_relative,
                    peak_ground_com_speed_m_s=peak_ground,
                    steps_relative_speed_above_040=relative_above_040_steps,
                    inferred_flow_x_m_s=float(ctrl.flow_hat[0]),
                    inferred_flow_y_m_s=float(ctrl.flow_hat[1]))
    finally:
        env.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cases", nargs="*", default=None)
    p.add_argument("--prefix", default="blind_6d")
    p.add_argument("--scene-file", type=Path, default=ROOT / "structured_100_040/scenes.json")
    p.add_argument("--speed", type=float, default=.54)
    p.add_argument("--drift-gain", type=float, default=.5)
    p.add_argument("--estimate-tau", type=float, default=1.)
    p.add_argument("--cross-p", type=float, default=1.)
    p.add_argument("--cross-i", type=float, default=.3)
    p.add_argument("--max-amplitude", type=float, default=None)
    p.add_argument("--frequency-hz", type=float, default=2.5)
    args = p.parse_args()
    scenes = json.loads(args.scene_file.read_text(encoding="utf-8"))
    if args.cases is None:
        args.cases = list(scenes)
    params = dict(speed=args.speed, drift_gain=args.drift_gain,
                  estimate_tau=args.estimate_tau,
                  cross_p=args.cross_p, cross_i=args.cross_i)
    if args.max_amplitude is not None and not 0.1 < args.max_amplitude <= .75:
        raise ValueError("max amplitude must be in (0.1, 0.75]")
    if not 0 < args.frequency_hz <= 5:
        raise ValueError("frequency must be in (0, 5] Hz")
    rows = []
    output = HERE / f"{args.prefix}_results.csv"
    for i, case_id in enumerate(args.cases, 1):
        row = run(case_id, params=params, scenes=scenes,
                  max_amplitude=args.max_amplitude,
                  frequency_hz=args.frequency_hz)
        if args.max_amplitude is not None:
            assert row["max_commanded_amplitude_rad"] <= args.max_amplitude + 1e-6
        rows.append(row)
        with output.open("w", newline="", encoding="utf-8") as out:
            writer = csv.DictWriter(out, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        if i % 10 == 0 or i == len(args.cases):
            print(f"{i}/{len(args.cases)}: {sum(r['success'] for r in rows)} arrived", flush=True)
    summary = dict(n=len(rows), arrived=sum(r["success"] for r in rows),
                   timeout=sum(r["timeout"] and not r["physical_failure"] for r in rows),
                   physical_failure=sum(r["physical_failure"] for r in rows),
                   max_amplitude_rad=args.max_amplitude or .75,
                   frequency_hz=args.frequency_hz,
                   max_observed_relative_water_com_speed_m_s=max(r["peak_relative_water_com_speed_m_s"] for r in rows),
                   max_observed_ground_com_speed_m_s=max(r["peak_ground_com_speed_m_s"] for r in rows),
                   cases_with_relative_speed_above_040=sum(r["steps_relative_speed_above_040"] > 0 for r in rows),
                   params=params,
                   by_flow_relative_deg={str(deg): dict(n=sum(r["flow_relative_deg"] == deg for r in rows),
                                                        arrived=sum(r["success"] for r in rows if r["flow_relative_deg"] == deg))
                                         for deg in range(0, 360, 36)},
                   scenes_sha256=hashlib.sha256(args.scene_file.read_bytes()).hexdigest(),
                   controller_sha256=hashlib.sha256((HERE / "blind_6d.py").read_bytes()).hexdigest())
    (HERE / f"{args.prefix}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
