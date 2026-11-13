# Frodo: mobile ring assembly on a moving line

A mobile manipulator in MuJoCo picks rings off kits on a moving conveyor and seats them
on bottle necks: five accepted at station A, then five at station B. Acceptance is
decided by the customer's line controller and inspection, never by the robot.

## Running it

```sh
uv sync --locked                    # setup; CPU only, no GPU or paid API

scripts/evaluate.sh                 # all five acceptance runs + systems-check -> evidence/
scripts/run.sh mission --scenario nominal    --seed 0 --out evidence/nominal-seed-0
scripts/run.sh mission --scenario human_stop --seed 0 --out evidence/human-stop
scripts/run.sh mission --scenario plc_retry  --seed 0 --out evidence/plc-retry
scripts/run.sh play                 # interactive: drag a box into the robot's way
```

Each run directory holds:

- `events.jsonl`: the evidence log.
- `check.json`: the starter `systems-check` result.
- `evaluation.json`: our customer-side checks, computed from simulator state.
- `summary.json`: seed parameters, attempts, docking errors, wall time.
- `seat_samples.jsonl.gz`: raw seat measurements, so the 0.5 s hold can be recomputed.

The PLC run adds the socket trace and controller ledger, and the human-stop run adds
`demo.mp4`.

## Results

| Run | Belt A / B (mm/s) | Accepted | Rejects | Accepted cycle | Mission (sim) | systems-check | Evaluator | Notes |
|---|---|---|---|---|---|---|---|---|
| nominal seed 0 | 31.4 / 27.7 | 5 + 5 | 0 | 5.0–5.1 s | 228 s | pass | pass | |
| nominal seed 1 | 30.1 / 34.5 | 5 + 5 | 0 | 5.1 s | 213 s | pass | pass | |
| nominal seed 2 | 27.6 / 28.0 | 5 + 5 | 0 | 5.0–5.2 s | 240 s | pass | pass | |
| human stop | 31.4 / 27.7 | 5 + 5 | 0 | 5.0 s (interrupted kit 9.1 s) | 228 s | pass | pass | stop 10 ms after the person entered; 3.0 s in the zone |
| PLC retry | 31.4 / 27.7 | 5 + 5 | 0 | 5.0–5.1 s | 228 s | pass | pass | retried ASSEMBLY_DONE applied once, accepted count unchanged |

Across all runs: settled ring tilt ≤ 1.5° (limit 10°), radial error ≤ 0.6 mm (limit 4 mm),
docking error 3.1–3.6 mm (limit 25 mm). Limits: 12 s cycle, 270 s nominal and 360 s fault
missions. Wall time was 5–8 minutes per run on a laptop CPU, with all five running in parallel.

Every run is sensor-driven: lidars, odometry, IMU, the wrist RGB-D camera, the conveyor
encoder and the Modbus line controller. A test fails if robot code imports the
simulator or the plant.

## How it works

**Robot.** Stanford TidyBot (MuJoCo Menagerie): holonomic base, Kinova Gen3 arm and
Robotiq 2F-85 gripper. Two changes were needed to reach the belt from the specified
dock: a 0.40 m arm riser, and the gripper working tilted 35°. Every run docks within
4 mm of the spec pose and completes all assemblies from there. The evaluator checks
each run for robot contacts.

**Sensors.**

| Sensor | Rate | Latency | Noise |
|---|---|---|---|
| 2 × 2D lidar, 12 cm high, opposite base corners, 270° each | 10 Hz | 10 ms | 5 mm σ |
| Wheel odometry | 100 Hz | 0 | 1 % white + 0.5 % scale error |
| IMU gyro | 100 Hz | 0 | 0.002 rad/s σ, 0.0005 rad/s bias |
| Wrist RGB-D, 640 × 480 | 15 Hz | 30 ms | depth 1 mm σ at 1 m, ∝ depth² |
| Conveyor encoder | 50 Hz | 0 | 0.1 mm counts |

**Localization and navigation.** An EKF predicts the pose from odometry and the gyro.
Point-to-line ICP of both lidar scans against the known map corrects it, with lidar
latency compensated. The base follows an A* path, then a slow closed-loop docking
controller brings it in. Docking error is 3.1–3.6 mm (limit 25 mm, 2°).

**Perception.** The wrist camera looks straight down from the arm's waiting pose.

1. Pixels at the height of a ring top or a neck top are grouped into regions. No colour
   is used.
2. A circle of the spec radius is fitted to each region's rim.
3. A fit is rejected on residual, how much of the circumference was seen, height, and
   the region staying inside the ring or neck outline. A kit needs a bottle plus a ring
   at the right offset.
4. Kits are tracked between images with the conveyor encoder.

The camera pose comes from the robot's own kinematics of its measured joints. On 300
still frames, centres were found to 0.35 mm mean error with no false detections. A
learned segmenter (FastSAM) used as the region source gave the same accuracy at 14×
the cost; it remains an option. While carrying the ring, the robot measures where it
sits in the hand (it slides about 10 mm on closing) and places using that measurement.

**Manipulation.** A Cartesian tracker: a phase machine sets tool goals relative to a
frame riding on the belt (the ring, then the bottle), and differential IK (mink)
follows them. The ring is pinched across its 80 mm outer diameter, centred over the
neck within 1 mm, and released 2 mm above its seat. Settled tilt is at most 1.5°
(limit 10°) and radial error at most 0.6 mm (limit 4 mm).

**Grasp constraint.** A weld between gripper and ring, engaged only when all of these
hold:

1. the gripper is commanded closed,
2. both finger pads touch that ring,
3. the robot's own check passes: tool within 6 mm of the grasp point, fingers stopped
   at a ring-sized opening.

It freezes the existing relative pose, so nothing snaps into the hand. It is released
when the gripper opens, and collisions stay active throughout.

**Line integration.** The robot reads arrivals, inspection results and accepted counts
over Modbus/TCP. It sends ASSEMBLY_DONE when it lets go of a ring inside the work
window; that is a claim, not acceptance. A lost acknowledgement is retried with the same
sequence and payload, and an unacknowledged command is settled before any newer one.

## Safety and the human-stop scenario

The robot stops for any object; it does not detect people. A 2D lidar at 12 cm sees
legs, boxes and trolleys alike and cannot tell them apart. Person detection would need
3D lidars (or a camera) with a person detector on the point cloud. This is simulation
behaviour, not a certified safety function.

- **Driving:** anything in the swept path slows the base from 1.0 m and stops it at
  0.45 m. It resumes by itself once the way is clear. The dock's own conveyor is masked
  during docking, with a 0.15 m minimum clearance kept.
- **Working at a station:** anything within 1 m of the robot latches a stop of base and
  arm. It ends only when the zone is clear and an operator acknowledges.
- **Controlled stop:** the belt never pauses, so an arm frozen low over the kits would
  be hit by the next bottle. A low tool rises clear first, then holds; motion settles
  within 0.5 s.
- **Recovery:** a held ring is finished if there is time left in the work window.
  Otherwise the kit is given up: the ring is put back, and the kit is rejected and
  replaced.

In the scenario, a 1.7 m "person" body walks into the zone during a ring transfer and
stays 2–4 s (configurable in `[scenarios.human_stop]`). The scenario logs entry and exit
from the person's true position. The robot logs its own stop and resume.

## PLC retry scenario

A TCP relay between robot and controller forwards one ASSEMBLY_DONE, then drops the
acknowledgement and closes the connection. The robot resends the identical frame, and
the controller applies it once and acknowledges the duplicate. The run records both
request frames, the full socket trace and the controller ledger.

## Timing

| | Budget (starter) | Measured |
|---|---|---|
| Accepted cycle | 11.8 s with 25 % reserve | 5.0–5.2 s |
| Nominal mission | 270 s limit | 213–240 s |

The mission is dominated by belt spacing and downstream inspection, not robot motion.
Wall time is 4–6 min per mission, mostly from rendering the camera.

## Simplifications

- **Feeder:** kits come from a pool of four per station. A kit is placed on the belt
  0.15 m before the work zone and returned to the pool after the belt end. These are
  fixture resets outside the scored window.
- **Base:** three stiff position servos, no wheel slip. Localization error comes from
  the injected odometry noise only.
- **Ring-tray contacts:** only 12 of the ring's 48 segments collide with kit trays,
  for speed. Contacts with bottles, belts and fingers are full.
- **Arm planning:** not collision-aware. Motions are designed to clear the cell, tool
  targets are clamped to a workspace box, a watchdog aborts on a blocked arm, and the
  evaluator checks contacts.
- **Oracle mode:** it exists for development and fast tests, is labelled in every
  summary, and was not used for evidence.

## Failures that shaped the design

- **Ring dropped onto the neck:** the first grasp put a finger inside the ring, so it
  had to be dropped 54 mm and settled 5–10° tilted (one reject, one late seat). Grasping
  the outer diameter fixed it.
- **Wrong camera view:** the renderer's default near clip hid the gripper from the wrist
  camera. Fixed by a nearer clip and a forward camera bracket.
- **Frozen arm hit by a bottle:** a frozen arm was struck by the next bottle during a
  stop, which led to the controlled stop.
- **Fingertips on the tray:** the evaluator found them brushing the kit tray during
  grasps; the grasp is now 2 mm higher.
- **Grasp nudges the kit:** closing fingers push a kit up to 2 mm/s for a moment. It
  stays within 0.03 mm of its belt position.
- **PLC client could wedge:** it could reuse a sequence number after lost
  acknowledgements; fixed with the pending-command rule.

Remaining limits: an interruption before the grasp costs a kit (about 15 s). Obstacles
in the path are waited out, not avoided. Perception is untested on real depth sensor
artefacts.

## Suspected starter defects

1. `LineController.inspect` accepts a kit seated after the work window, while the packet
   and `systems-check` treat that as a failure.
2. Registers 16 and 26 always report 30 mm/s regardless of the run's belt speed.
3. The settle limits and the 0–1 s release phase are in the docs but not in
   `spec/customer_cell.json`.
4. CI installs only the dev and physics groups. Our dependencies are made a default
   group instead of editing the starter.

## Provenance

- **Robot model:** TidyBot from MuJoCo Menagerie; commit and licences in
  `third_party/PROVENANCE.md`. All modifications are made in code
  (`src/frodo/sim/scene.py`).
- **Libraries:** MuJoCo 3.8.1, mink, numpy, scipy, Pillow, imageio, all pinned in
  `uv.lock`. Ultralytics FastSAM (AGPL) is optional and used only for the perception
  comparison.
- **Authorship:** code written with an AI coding assistant (Claude), reviewed, run and
  measured by the author.
