"""Scan 20-second sine gaits in the generated V4-pool MuJoCo models.

Run: python test_swim.py [--pilot] [--n 3 4 5 6 8]
All time integration uses the source scene's 1 ms Euler step. The V4 pool is
visual only; all model geoms use its default inertia-box fluid model.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import time

import mujoco
import numpy as np

from generate_fish import FishConfig, generate

HERE = Path(__file__).resolve().parent
NS = (3, 4, 5, 6, 8)
FREQUENCIES = (0.5, 1.0, 1.5, 2.0, 3.0)
AMPLITUDES = (0.2, 0.4, 0.6)
PHASES = (0.5, 1.0, 1.5)
DURATION = 20.0
CONTROL_DT = 0.02
RAMP = 1.0
MAX_LATERAL_BODY_LENGTHS = 0.25
MAX_NET_YAW_DEG = 45.0


def validate_model(model: mujoco.MjModel, n: int) -> dict:
    assert model.opt.density == 1000 and model.opt.viscosity == 0.001
    assert model.opt.timestep == 0.001
    assert np.all(model.opt.gravity == 0)
    assert model.opt.integrator == mujoco.mjtIntegrator.mjINT_EULER
    assert model.njnt == n + 1 and model.nu == n
    assert [model.actuator(i).name for i in range(n)] == [f"joint_{i}" for i in range(n)]
    assert sorted(model.site(i).name for i in range(model.nsite) if model.site(i).name.startswith("s")) == [f"s{i}" for i in range(8)]
    fish_ids = [model.body("head").id] + [model.body(f"segment_{i}").id for i in range(1, n+1)]
    mass = float(sum(model.body_mass[i] for i in fish_ids))
    inertias = model.body_inertia[fish_ids]
    assert mass > 0 and np.all(inertias > 0)
    return {"mass_kg": mass, "min_body_inertia": float(np.min(inertias)),
            "max_body_inertia": float(np.max(inertias))}


def run_one(model: mujoco.MjModel, n: int, frequency: float, amplitude: float,
            phase: float, duration: float = DURATION, record: bool = False,
            a0: float = 0.3) -> tuple[dict, dict | None]:
    if not 0 <= a0 <= 1:
        raise ValueError("a0 must lie in [0, 1]")
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    head_id = model.body("head").id
    jids = [model.joint(f"joint_{i}").id for i in range(n)]
    qids = [int(model.jnt_qposadr[j]) for j in jids]
    ranges = model.jnt_range[jids, 1]
    start_com = data.subtree_com[head_id, :2].copy()
    start_head = data.xpos[head_id, :2].copy()
    start_quat = data.sensor("head_quat").data.copy()
    yaw0 = yaw_from_quat(start_quat)
    forward = np.array([math.cos(yaw0), math.sin(yaw0)])
    lateral = np.array([-forward[1], forward[0]])
    steps_per_control = round(CONTROL_DT / model.opt.timestep)
    steps = round(duration / model.opt.timestep)
    if not math.isclose(steps_per_control * model.opt.timestep, CONTROL_DT):
        raise ValueError("control_dt must be a multiple of physics timestep")
    target_amp = amplitude * (a0 + (1 - a0) * np.arange(n) / (n - 1 if n > 1 else 1))
    phase_offsets = 2 * math.pi * phase * np.arange(n) / n
    max_abs_torque = 0.0
    torque_sum = 0.0
    count = 0
    near_limit = 0
    max_abs_q = 0.0
    max_abs_state = 0.0
    finite = True
    warning_total = 0
    headings = [yaw0]
    head_path = [start_head]
    qpos_frames = [data.qpos.copy()] if record else None
    frame_stride = round(0.1 / model.opt.timestep)
    for step in range(steps):
        if step % steps_per_control == 0:
            t = step * model.opt.timestep
            fade = min(1.0, t / RAMP)
            target = fade * target_amp * np.sin(2 * math.pi * frequency * t - phase_offsets)
            data.ctrl[:] = np.clip(target, -ranges, ranges)
        mujoco.mj_step(model, data)
        if step % steps_per_control == steps_per_control - 1:
            q = data.qpos[qids]
            tau = np.abs(data.actuator_force)
            max_abs_torque = max(max_abs_torque, float(tau.max()))
            torque_sum += float(tau.mean())
            count += 1
            near_limit += int(np.any(np.abs(q) >= 0.98 * ranges))
            max_abs_q = max(max_abs_q, float(np.abs(q).max()))
            max_abs_state = max(max_abs_state, float(np.max(np.abs(data.qvel))))
            finite = bool(np.all(np.isfinite(data.qpos)) and np.all(np.isfinite(data.qvel)) and
                          np.all(np.isfinite(data.actuator_force)) and np.all(np.isfinite(data.sensordata)))
            warning_total = sum(w.number for w in data.warning)
            if not finite or max_abs_state > 1e4 or warning_total:
                break
        if step % frame_stride == frame_stride - 1:
            headings.append(yaw_from_quat(data.sensor("head_quat").data))
            head_path.append(data.xpos[head_id, :2].copy())
            if record:
                qpos_frames.append(data.qpos.copy())
    elapsed = float(data.time)
    delta = data.subtree_com[head_id, :2] - start_com
    heading_unwrapped = np.unwrap(np.asarray(headings))
    yaw_net_deg = float(np.degrees(heading_unwrapped[-1] - heading_unwrapped[0]))
    yaw_abs_deg = float(np.degrees(np.abs(np.diff(heading_unwrapped)).sum()))
    forward_m = float(np.dot(delta, forward))
    lateral_m = float(np.dot(delta, lateral))
    long_limit = near_limit / max(count, 1) > 0.05
    stable = bool(finite and not warning_total and max_abs_state <= 1e4 and not long_limit and
                  math.isclose(elapsed, duration, abs_tol=model.opt.timestep))
    body_length = float(np.sum([2 * model.geom_size[model.geom(f"body_geom_{i}").id, 0] for i in range(n+1)]))
    accepted = bool(stable and forward_m / elapsed > 0.1 * body_length and
                    abs(yaw_net_deg) < MAX_NET_YAW_DEG and
                    abs(lateral_m) < MAX_LATERAL_BODY_LENGTHS * body_length)
    result = dict(n_joints=n, frequency_hz=frequency, amplitude_rad=amplitude, a0=a0,
                  phase_waves=phase, duration_s=elapsed, body_length_m=body_length,
                  forward_m=forward_m, mean_forward_m_s=forward_m/elapsed,
                  speed_body_lengths_s=forward_m/elapsed/body_length,
                  lateral_m=lateral_m, net_yaw_deg=yaw_net_deg,
                  absolute_yaw_motion_deg=yaw_abs_deg,
                  nan_or_diverged=not finite or max_abs_state > 1e4,
                  warning_count=warning_total, long_joint_limit=long_limit,
                  limit_sample_fraction=near_limit/max(count,1),
                  mean_abs_joint_torque_nm=torque_sum/max(count,1),
                  max_abs_joint_torque_nm=max_abs_torque,
                  max_abs_joint_angle_rad=max_abs_q, max_abs_qvel=max_abs_state,
                  stable=stable, accepted=accepted)
    trace = None
    if record:
        trace = dict(head_xy=np.asarray(head_path), qpos=np.asarray(qpos_frames),
                     time=np.arange(len(head_path)) * (frame_stride * model.opt.timestep))
    return result, trace


def yaw_from_quat(q: np.ndarray) -> float:
    w, x, y, z = q
    return math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))


def draw_artifacts(model: mujoco.MjModel, n: int, gait: dict, trace: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import imageio.v2 as imageio
    xy = trace["head_xy"]
    fig, ax = plt.subplots(figsize=(6, 5), dpi=140)
    ax.plot(xy[:,0], xy[:,1], color="#1666aa", lw=1.5)
    ax.scatter(*xy[0], c="green", s=35, label="start")
    ax.scatter(*xy[-1], c="red", s=35, label="finish")
    ax.set(xlabel="x (m)", ylabel="y (m)", title=f"N={n} head path, 20 s")
    ax.axis("equal"); ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(HERE / f"fish_N{n}_head_path.png"); plt.close(fig)
    # MuJoCo offscreen rendering of saved integrated states; no re-integration.
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=480, width=640)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.azimuth = 90
    camera.elevation = -89.9
    camera.distance = 2.0
    frames = []
    try:
        for k in range(0, min(len(trace["qpos"]), 61), 2):
            data.qpos[:] = trace["qpos"][k]
            mujoco.mj_forward(model, data)
            camera.lookat[:] = data.subtree_com[model.body("head").id]
            renderer.update_scene(data, camera=camera)
            frames.append(renderer.render().copy())
        imageio.mimsave(HERE / f"fish_N{n}_best.gif", frames, fps=5, loop=0)
    finally:
        renderer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, nargs="+", default=list(NS))
    parser.add_argument("--pilot", action="store_true", help="one gait per N for a quick mechanics check")
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--render-only", action="store_true", help="render previously scanned best gaits")
    parser.add_argument("--a0", type=float, default=0.3, help="head-to-tail amplitude ratio at joint 0")
    args = parser.parse_args()
    if args.render_only:
        saved = json.loads((HERE / "best_gaits.json").read_text(encoding="utf-8"))
        for n in args.n:
            model = mujoco.MjModel.from_xml_path(str(HERE / f"fish_N{n}.xml"))
            gait = saved[str(n)]
            _, trace = run_one(model, n, gait["frequency_hz"], gait["amplitude_rad"],
                               gait["phase_waves"], record=True, a0=gait["a0"])
            draw_artifacts(model, n, gait, trace)
            print(f"Rendered N={n}", flush=True)
        return
    rows = []
    best = {}
    model_meta = {}
    started = time.perf_counter()
    for n in args.n:
        path = HERE / f"fish_N{n}.xml"
        if not path.exists():
            generate(FishConfig(n_joints=n), path)
        model = mujoco.MjModel.from_xml_path(str(path))
        model_meta[n] = validate_model(model, n)
        grid = [(1.5, 0.4, 1.0)] if args.pilot else [(f, a, p) for f in FREQUENCIES for a in AMPLITUDES for p in PHASES]
        nrows = []
        for index, (f, a, p) in enumerate(grid, 1):
            result, _ = run_one(model, n, f, a, p, a0=args.a0)
            rows.append(result); nrows.append(result)
            if index % 9 == 0 or args.pilot:
                print(f"N={n}: {index}/{len(grid)} scanned, latest speed={result['speed_body_lengths_s']:.3f} L/s", flush=True)
        eligible = [r for r in nrows if r["stable"] and abs(r["net_yaw_deg"]) < MAX_NET_YAW_DEG and
                    abs(r["lateral_m"]) < MAX_LATERAL_BODY_LENGTHS * r["body_length_m"]]
        chosen = max(eligible or [r for r in nrows if r["stable"]] or nrows,
                     key=lambda r: r["speed_body_lengths_s"])
        best[n] = {**chosen, **model_meta[n]}
        print(f"BEST N={n}: {chosen['speed_body_lengths_s']:.4f} L/s, yaw={chosen['net_yaw_deg']:.2f} deg, accepted={chosen['accepted']}", flush=True)
        if not args.pilot and not args.no_render:
            _, trace = run_one(model, n, chosen["frequency_hz"], chosen["amplitude_rad"], chosen["phase_waves"], record=True, a0=args.a0)
            draw_artifacts(model, n, chosen, trace)
    if not args.pilot:
        with (HERE / "swim_results.csv").open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
        (HERE / "best_gaits.json").write_text(json.dumps(best, indent=2), encoding="utf-8")
    print(f"Completed in {time.perf_counter()-started:.1f} wall seconds", flush=True)


if __name__ == "__main__":
    main()
