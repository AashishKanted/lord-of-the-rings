# frodo architecture

Solution code for the mobile ring-on-bottle assembly cell. The starter
(`src/advaitic_systems/`, `spec/`, `docs/`) is unchanged; everything here lives in
`src/frodo/`, `config/`, `scripts/` and `third_party/`.

## The one rule

The robot software sees the world only through `frodo.interfaces` (a `SensorFrame`
in, a `RobotCommand` out) and Modbus/TCP. `frodo.robot` never imports `frodo.sim`
or `frodo.plant`; `tests/frodo/test_architecture.py` fails if it does. Everything
runs in one process and one deterministic loop, with the plant's PLC server on a thread.

```
            frodo.sim                          frodo.robot
  +---------------------------+   SensorFrame   +--------------------------------+
  | scene (MuJoCo, from spec) | --------------> | orchestrator (Mode machine)    |
  | sensors: 2 lidars, odom,  |                 |   localization  EKF + scan ICP |
  |   IMU, RGB-D, encoder     |                 |   perception    ring / bottle  |
  | actuation, grasp weld     | <-------------- |   navigation    plan/track/dock|
  +---------------------------+   RobotCommand  |   manipulation  IK, grasp      |
            ^ ground truth                      |   safety        slow / stop    |
  +---------------------------+   Modbus/TCP    |   plc           line client    |
  | frodo.plant            | <-------------> +--------------------------------+
  | line: feeder, PLC server, |
  | inspection, event log     |  --> results/<scenario>/events.jsonl --> systems-check
  +---------------------------+
```

## Layout

| Path | Role | State |
|---|---|---|
| `config/frodo.toml` | every tunable; customer numbers stay in `spec/` | in use |
| `scripts/run.sh` | launcher: `view`, `render`, `camera`, `demo`, `sensors`, `nav`, `line`, `mission` | in use |
| `frodo/config.py`, `log.py` | config loader (`with_overrides` for tests); logger with `file:line` | done |
| `frodo/cell.py` | read-only customer spec, station frames | done |
| `frodo/interfaces.py` | sensor/command dataclasses, `Mode` | done |
| `frodo/sim/scene.py`, `camera.py`, `grasp.py` | cell model, RGB-D camera, grasp constraint | done |
| `frodo/sim/sensors.py`, `lidar.py`, `actuation.py` | sensor suite, command bridge | done |
| `frodo/sim/oracle.py` | ORACLE kit poses for the oracle perception backend | done |
| `frodo/plant/line.py`, `feeder.py`, `plc.py`, `events.py` | belts, feeder, PLC host, evidence log | done (nominal only) |
| `frodo/plant/inspection.py`, `recorder.py` | independent seat measurement; measured motion samples | done |
| `frodo/robot/localization/` | EKF + scan-to-map ICP | done |
| `frodo/robot/navigation/` | A* planner, path tracker, docking, `Navigator` | done (no obstacle reaction) |
| `frodo/robot/frames.py` | station geometry in the robot base frame | done |
| `frodo/robot/perception/` | detector backends (`rgbd` default, `oracle`), segmenters, circle fits from depth, belt-encoder tracker, `Perception` facade | done; drives the nominal mission |
| `frodo/robot/manipulation/` | IK, motion profiles, `ArmController` (stow, pick-and-place cycle) | done |
| `frodo/robot/plc.py` | Modbus line client with sequence/retry rule | done |
| `frodo/robot/orchestrator/` | mission mode machine | done (nominal only) |
| `frodo/robot/safety/` | lidar slow/stop supervisor for the base, with a docking mask | rudimentary: no human-zone latch, arm not supervised |
| `frodo/apps/mission.py` | full mission runner | done (nominal scenario) |
| `frodo/apps/pick_place_demo.py`, `nav_demo.py`, `line_demo.py`, `sensors_demo.py` | single-subsystem demos | done |

Not built yet: sensor-driven perception, the human-zone stop (latched, base and arm),
human-stop and PLC-retry scenarios.

## Control tick

```
frame   = sensors.read(data)                 # + ORACLE kit poses when perception.method = "oracle"
command = orchestrator.step(frame)
    pose = localizer.update(frame)
    poll the line controller (10 Hz): arrivals, inspection results, accepted counts
    NAVIGATE / DOCK : navigator.step(pose)            arm stowed
    MANIPULATE      : perception.update(frame) -> arm.step(joints, tracked kit)
    STOW            : arm folds, then the next station is chosen
    safety.filter(frame, command, dock mask)          last: can only reduce the base command
bridge.apply(data, command); step physics; plant.line.step(data) every physics step
```

## Mission flow

```
IDLE -> NAVIGATE -> DOCK -> MANIPULATE -> STOW -> NAVIGATE -> DOCK -> MANIPULATE -> STOW -> DONE
```

- On docking the orchestrator sends READY, builds the `StationFrame` from the localizer's
  pose, unstows the arm and starts perception.
- A cycle starts when the line controller reports a new arrival id; the kit just inside
  the work window is bound to that id.
- `ASSEMBLY_DONE` is sent when the arm lets go of the ring inside the work window. It is
  the robot's claim. Acceptance is decided by the plant's own seat measurement at the
  downstream inspection, and the robot learns it only from the registers.
- `mission.leave_policy` chooses when to leave a station: `after_accepted` (default) waits
  for the full batch to be accepted; `after_done` leaves once the batch is reported done,
  waits at the next station's pre-dock pose until the batch is accepted, and turns back if
  a kit is rejected.

## Perception interface

`Perception.start(station_frame)` / `stop()` are called by the orchestrator. While active,
`update(frame)` runs a detector backend (`perception.method`) and a tracker that carries
each kit forward with the conveyor encoder between detections. Consumers get
`KitObservation`s (ring centre, bottle base, belt velocity, age) in the robot base frame.
A new detector implements `Detector.detect(frame, station) -> list[KitDetection]`; nothing
downstream changes.

The `rgbd` detector uses the wrist camera, pitched back so that it looks straight down
from the arm's waiting pose (about 0.41 m above the kits):

```
segmenter.masks(image)     proposals: "depth_bands" (height above the kit, no model)
                           or "fastsam" (class-agnostic network on the colour image)
geometry.top_surface       depth clean-up: keep each mask's upper surface only
geometry.fit_ring/bottle   centre from the rim points and the spec radii (48/80 mm ring,
                           32 mm neck top); checks: fit residual, circumference seen,
                           height above the kit, whole region inside the spec outline
rgbd_detector              a kit = a bottle plus the ring one ring-offset downstream
```

Colour is not used to decide what an object is. `scripts/run.sh perception` runs both
segmenters on the same frames and scores them against simulator truth
(`results/perception_bakeoff/summary.json`).

In the mission the camera detects kits from the waiting pose and the tracker carries them
with the belt encoder while the arm (and with it the camera) moves. A bottle seen without
its ring keeps updating its track. After the grasp the ring slides in the fingers, so
while carrying it the robot looks for the ring in its hand (`find_held_ring`) and averages
those looks into the tool-to-ring offset used for placing. It does not servo on the image.

## Arm controller

Cartesian IK tracker in the base frame: a phase machine sets a tool goal relative to a
frame riding on the belt (the ring's grasp point, then the bottle), a speed- and
acceleration-limited setpoint moves toward it, the belt velocity is added as a short lead,
and differential IK produces joint targets. Before release the observed ring-over-neck
error is integrated out. Stow and unstow are joint-space moves through the home posture.
The waiting pose is above bottle height so that bottles pass under the gripper.

The ring is pinched across its outer diameter (`manipulation.grasp = "outer"`), so both
fingers stay outside the bottle and the ring is lowered to 2 mm above its seat before
release. The older wall pinch (`"wall"`) is kept for comparison; it has to drop the ring
from above the neck.

The arm is not collision-aware: the IK knows joint limits only. Two guards exist: tool
targets are clamped to a workspace box over the belt, and a cycle is aborted when the
measured tool stays more than 30 mm from its commanded position (blocked arm).

## Safety monitor

`SafetyMonitor.filter` runs last in the tick on raw lidar returns (not on the localization
output). Returns in the corridor the footprint is about to sweep scale the base speed
from `slow_distance_m` down to zero at `stop_distance_m`; anything inside the corner
circle stops a turn; stale lidar data stops the base; a stop is held until the obstacle
has clearly gone. In `DOCK` mode the orchestrator passes the target conveyor: returns
from inside its outline are exempt from the slow/stop distances but still enforce
`dock_min_clearance_m`.

## How modules talk (no message bus)

There is no pub/sub middleware. Modules are plain classes with typed inputs and
outputs (`frodo.interfaces`); the orchestrator calls them in a fixed order each tick
and passes results along. Rules that keep this as decoupled as a bus would:

- a module never imports another module's internals, only `frodo.interfaces`,
  `frodo.cell`, `frodo.config`, `frodo.log`
- every message carries its own timestamp `t`; consumers check freshness themselves
  (safety treats stale data as an obstacle)
- slow sensors appear in `SensorFrame` only on the tick they sample (else `None`/empty)
- only apps (`frodo.apps`) wire sim, plant and robot together

Because modules do not know how their inputs arrive, a transport (ROS 2 topics, a
recorder for replay) can be wrapped around them later without changing them.

## Sensors

Two low 2D lidars on opposite base corners (360 degree cover; used for localization and
for safety), wheel odometry, IMU, conveyor encoder. Two camera mounts exist (wrist and a
fixed mast); `sensors.cameras.perception_source` selects the single RGB-D camera that
feeds perception, as the customer allows one.

## Honesty markers

- `apps/pick_place_demo.py` is an ORACLE run: true poses, robot pre-docked.
- With `perception.method = "oracle"` (not the default; used by the end-to-end test to keep
  it short) the mission runner gives the robot true kit poses. `summary.json` says so in its `perception` field. Localization
  and navigation in that run are sensor-driven; perception is not.
- `apps/nav_demo.py` and `apps/line_demo.py` exercise one subsystem each and produce no
  acceptance evidence.
- A passing `systems-check` on `events.jsonl` shows log consistency only.
