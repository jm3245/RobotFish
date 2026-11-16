"""Create a solid-only N4 scene to demonstrate MuJoCo's uniform wind limit."""
from __future__ import annotations

import argparse
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco

ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--flow", type=float, default=.4)
    parser.add_argument("--diameter", type=float, default=.7)
    args = parser.parse_args()
    if args.flow < 0 or args.diameter <= 0:
        raise ValueError("Flow must be nonnegative and diameter positive")
    source = ROOT / "fish_morphologies/fish_N4.xml"
    root = ET.parse(source).getroot()
    option = root.find("option")
    assert option is not None
    option.set("wind", f"{args.flow:.9g} 0 0")
    world = root.find("worldbody")
    assert world is not None
    ET.SubElement(world, "geom", {
        "name": "fixed_cylinder", "type": "cylinder", "pos": "2 0 0.5",
        "size": f"{args.diameter/2:.9g} 0.5", "rgba": "0.35 0.38 0.41 1",
        "contype": "1", "conaffinity": "1",
    })
    output = ROOT / "models/fish_N4_cylinder_only.xml"
    output.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(output, encoding="utf-8", xml_declaration=True)
    model = mujoco.MjModel.from_xml_path(str(output))
    fish_geom_ids = [model.geom(f"body_geom_{i}").id for i in range(5)]
    assert all(model.geom_contype[i] == 0 and model.geom_conaffinity[i] == 0
               for i in fish_geom_ids)
    assert tuple(model.opt.wind) == (args.flow, 0.0, 0.0)
    print(output)
    print("Cylinder added; wind remains one global vector; fish body contact flags remain zero.")


if __name__ == "__main__":
    main()
