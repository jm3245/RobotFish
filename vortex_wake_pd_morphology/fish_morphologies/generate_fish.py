"""Generate articulated fish in the existing eel swimming water and V4 pool.

Coordinates are metres, kilograms, seconds and radians. The nose points +X.
The pool and option values are read from the existing files at generation time.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass, replace
from pathlib import Path
import xml.etree.ElementTree as ET
from xml.dom import minidom


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "eel_swimming" / "assets" / "eel_pool_v4_scene.xml"
POOL = HERE.parent / "eel_swimming" / "assets" / "eel_pool_v4_visuals.xml"
PROFILE_S = (0.0, 0.2, 0.45, 0.7, 0.88, 1.0)


@dataclass(frozen=True)
class FishConfig:
    n_joints: int = 5
    body_length: float = 1.0
    length_ratios: tuple[float, ...] | None = None
    height_profile: tuple[float, ...] = (0.090, 0.125, 0.145, 0.105, 0.055, 0.030)
    width_profile: tuple[float, ...] = (0.075, 0.105, 0.105, 0.070, 0.032, 0.018)
    tail_fin: bool = True
    tail_fin_area: float = 0.018  # projected XZ area, square metres
    tail_fin_height: float = 0.18  # full height, metres
    joint_range: float = 0.75  # symmetric angle limit, radians
    joint_damping: float = 0.015  # N m s / rad
    joint_stiffness: float = 0.01  # N m / rad
    joint_armature: float = 0.0005  # kg m^2 reflected motor inertia
    actuator_kp: float = 35.0  # N m / rad
    actuator_kv: float = 0.6  # N m s / rad
    force_limit: float = 12.0  # N m
    density: float = 1000.0  # kg / m^3; gravity is zero in source environment
    root_height: float = 0.5


def _fmt(value: float) -> str:
    return f"{value:.9g}"


def _interp(s: float, values: tuple[float, ...]) -> float:
    if len(values) != len(PROFILE_S):
        raise ValueError(f"profile must have {len(PROFILE_S)} values")
    for i in range(len(PROFILE_S) - 1):
        if s <= PROFILE_S[i + 1]:
            u = (s - PROFILE_S[i]) / (PROFILE_S[i + 1] - PROFILE_S[i])
            return values[i] * (1 - u) + values[i + 1] * u
    return values[-1]


def segment_lengths(cfg: FishConfig) -> list[float]:
    if cfg.n_joints < 1 or cfg.body_length <= 0:
        raise ValueError("n_joints and body_length must be positive")
    ratios = cfg.length_ratios
    if ratios is None:
        ratios = tuple(1.4 - 0.7 * i / cfg.n_joints for i in range(cfg.n_joints + 1))
    if len(ratios) != cfg.n_joints + 1 or any(x <= 0 for x in ratios):
        raise ValueError("length_ratios needs N+1 positive entries")
    return [cfg.body_length * x / sum(ratios) for x in ratios]


def generate(cfg: FishConfig, output: Path | None = None) -> Path:
    if any(x <= 0 for x in cfg.height_profile + cfg.width_profile):
        raise ValueError("body profiles must be positive")
    if not (0 < cfg.joint_range < 3.14 and cfg.density > 0 and cfg.force_limit > 0):
        raise ValueError("invalid joint, density, or force settings")
    if cfg.tail_fin and (cfg.tail_fin_area <= 0 or cfg.tail_fin_height <= 0):
        raise ValueError("tail fin area and height must be positive")
    lengths = segment_lengths(cfg)
    original = ET.parse(SOURCE).getroot()
    root = ET.Element("mujoco", {"model": f"fish_N{cfg.n_joints}"})
    root.append(copy.deepcopy(original.find("compiler")))
    root.append(copy.deepcopy(original.find("option")))
    root.append(copy.deepcopy(original.find("visual")))
    world = ET.SubElement(root, "worldbody")
    pool = ET.parse(POOL).getroot().find("worldbody")
    for element in pool:
        world.append(copy.deepcopy(element))
    body = ET.SubElement(world, "body", {"name": "head", "pos": f"0 0 {_fmt(cfg.root_height)}"})
    ET.SubElement(body, "freejoint", {"name": "root_free"})
    ET.SubElement(body, "site", {"name": "head_imu", "pos": "0 0 0", "size": "0.005"})
    starts = [0.0]
    for length in lengths:
        starts.append(starts[-1] + length)
    for i, length in enumerate(lengths):
        if i:
            body = ET.SubElement(body, "body", {"name": f"segment_{i}", "pos": f"{_fmt(-lengths[i-1])} 0 0"})
            ET.SubElement(body, "joint", {
                "name": f"joint_{i-1}", "type": "hinge", "axis": "0 0 1",
                "limited": "true", "range": f"{_fmt(-cfg.joint_range)} {_fmt(cfg.joint_range)}",
                "stiffness": _fmt(cfg.joint_stiffness), "damping": _fmt(cfg.joint_damping),
                "armature": _fmt(cfg.joint_armature),
            })
        midpoint = (starts[i] + starts[i + 1]) / (2 * cfg.body_length)
        # Ellipsoid X semiaxis exactly fills its allotted axial segment.
        ET.SubElement(body, "geom", {
            "name": f"body_geom_{i}", "type": "ellipsoid",
            "pos": f"{_fmt(-length/2)} 0 0",
            "size": f"{_fmt(length/2)} {_fmt(_interp(midpoint,cfg.width_profile)/2)} {_fmt(_interp(midpoint,cfg.height_profile)/2)}",
            "density": _fmt(cfg.density), "rgba": "0.22 0.58 0.77 1" if i % 2 == 0 else "0.30 0.68 0.80 1",
            "contype": "0", "conaffinity": "0",
        })
        if i == cfg.n_joints and cfg.tail_fin:
            # Thin vertical ellipsoid supplies projected area through the same
            # default inertia-box fluid model as every other geom in this pool.
            fin_length = cfg.tail_fin_area / cfg.tail_fin_height
            ET.SubElement(body, "geom", {
                "name": "tail_fin", "type": "ellipsoid",
                "pos": f"{_fmt(-length + fin_length/2)} 0 0",
                "size": f"{_fmt(fin_length/2)} {_fmt(_interp(1,cfg.width_profile)/3)} {_fmt(cfg.tail_fin_height/2)}",
                "density": _fmt(cfg.density), "rgba": "0.12 0.42 0.68 0.9",
                "contype": "0", "conaffinity": "0",
            })
        for j in range(8):
            x = cfg.body_length * j / 7
            # s=0 is the nose; s=1 is the tip of the last body segment.
            if starts[i] - 1e-10 <= x <= starts[i+1] + 1e-10 and (i == 0 or x > starts[i] + 1e-10):
                ET.SubElement(body, "site", {
                    "name": f"s{j}", "pos": f"{_fmt(starts[i]-x)} 0 0",
                    "size": "0.004", "rgba": "1 0.7 0.1 1",
                })
    actuators = ET.SubElement(root, "actuator")
    for i in range(cfg.n_joints):
        ET.SubElement(actuators, "position", {
            "name": f"joint_{i}", "joint": f"joint_{i}",
            "kp": _fmt(cfg.actuator_kp), "kv": _fmt(cfg.actuator_kv),
            "ctrllimited": "true", "ctrlrange": f"{_fmt(-cfg.joint_range)} {_fmt(cfg.joint_range)}",
            "forcelimited": "true", "forcerange": f"{_fmt(-cfg.force_limit)} {_fmt(cfg.force_limit)}",
        })
    sensors = ET.SubElement(root, "sensor")
    ET.SubElement(sensors, "framequat", {"name": "head_quat", "objtype": "site", "objname": "head_imu"})
    ET.SubElement(sensors, "velocimeter", {"name": "head_velocity", "site": "head_imu"})
    output = output or HERE / f"fish_N{cfg.n_joints}.xml"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(minidom.parseString(ET.tostring(root, encoding="utf-8")).toprettyxml(indent="  ", encoding="utf-8"))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-joints", type=int, nargs="+", default=[3, 4, 5, 6, 8])
    parser.add_argument("--body-length", type=float, default=1.0)
    parser.add_argument("--tail-fin-area", type=float, default=FishConfig.tail_fin_area)
    parser.add_argument("--tail-fin-height", type=float, default=FishConfig.tail_fin_height)
    args = parser.parse_args()
    for n in args.n_joints:
        cfg = replace(FishConfig(), n_joints=n, body_length=args.body_length,
                      tail_fin_area=args.tail_fin_area, tail_fin_height=args.tail_fin_height)
        print(generate(cfg))


if __name__ == "__main__":
    main()
