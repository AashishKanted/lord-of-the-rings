"""Optional primitive bottle/ring feasibility fixture; no robot or controller solution."""
from __future__ import annotations

import argparse
import json
from math import cos, pi, sin
from pathlib import Path


def fixture_xml():
    # A convex truncated cone for the bottle shoulder; a compound ring preserves its hole.
    vertices = []
    for z, radius in ((.12, .03), (.14, .016)):
        for i in range(48):
            angle = i * 2 * pi / 48
            vertices.extend([radius * cos(angle), radius * sin(angle), z])
    vertex_text = " ".join(f"{v:.9f}" for v in vertices)
    ring = []
    for i in range(48):
        angle = i * 2 * pi / 48
        ring.append(f'<geom type="box" pos="{.032*cos(angle):.9f} {.032*sin(angle):.9f} 0" '
                    f'euler="0 0 {angle:.9f}" size=".008 {.032*sin(pi/48):.9f} .004" '
                    f'mass="{.04/48:.9f}" rgba=".1 .55 .9 1"/>')
    return f'''<mujoco model="rigid_ring_parts_fixture">
  <compiler angle="radian"/>
  <option timestep=".001" gravity="0 0 -9.81" solver="Newton"/>
  <default><geom friction=".5 .005 .0001" solref=".01 1" solimp=".95 .99 .001"/></default>
  <asset><mesh name="shoulder" vertex="{vertex_text}"/></asset>
  <worldbody>
    <geom name="kit" type="box" pos="0 0 -.0125" size=".15 .12 .0125"/>
    <body name="bottle">
      <geom name="bottle_body" type="cylinder" pos="0 0 .06" size=".03 .06"/>
      <geom name="bottle_shoulder" type="mesh" mesh="shoulder"/>
      <geom name="bottle_neck" type="cylinder" pos="0 0 .16" size=".016 .02"/>
    </body>
    <body name="ring" pos="0 0 .23"><freejoint/>{''.join(ring)}</body>
  </worldbody>
</mujoco>'''


def validate_parts():
    import mujoco
    import numpy as np

    from .contract import inspection_passes

    model = mujoco.MjModel.from_xml_string(fixture_xml())
    data = mujoco.MjData(model)
    ring_id = model.body("ring").id
    bottle_id = model.body("bottle").id
    hold = 0.
    measurement = {}
    for _ in range(6000):
        mujoco.mj_step(model, data)
        rotation = data.xmat[ring_id].reshape(3, 3)
        contact_bottle, contact_kit = False, False
        for contact in data.contact:
            bodies = {int(model.geom_bodyid[contact.geom1]), int(model.geom_bodyid[contact.geom2])}
            contact_bottle |= ring_id in bodies and bottle_id in bodies
            contact_kit |= ring_id in bodies and 0 in bodies
        xyz = data.xpos[ring_id]
        tilt = float(np.arccos(np.clip(rotation[2, 2], -1, 1)))
        # Floating body speed, not a timer alone, determines settled hold.
        settled = np.linalg.norm(data.qvel) < .01
        raw = dict(dx_m=float(xyz[0]), dy_m=float(xyz[1]), z_m=float(xyz[2]),
                   tilt_rad=tilt, released=True, bottle_contact=bool(contact_bottle),
                   kit_contact=bool(contact_kit), hold_s=.5)
        hold = hold + model.opt.timestep if settled and inspection_passes(raw) else 0.
        measurement = {**raw, "hold_s": hold}
    return {"passed": inspection_passes(measurement), "measurement": measurement,
            "scope": "Static bottle and dropped rigid ring only; not a grasp, moving belt or robot validation."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-xml", type=Path)
    args = parser.parse_args()
    if args.write_xml:
        args.write_xml.write_text(fixture_xml())
    result = validate_parts()
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
