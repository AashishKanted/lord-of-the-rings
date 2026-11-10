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
| `scripts/run.sh` | launcher: `view`, `render`, `camera`, `demo`, `mission` | in use |
| `frodo/config.py`, `log.py` | config loader; logger with `file:line` | done |
| `frodo/cell.py` | read-only customer spec, station frames | done |
| `frodo/interfaces.py` | sensor/command dataclasses, `Mode` | done |
| `frodo/sim/scene.py`, `camera.py`, `grasp.py` | cell model, RGB-D camera, grasp constraint | done |
| `frodo/sim/sensors.py`, `actuation.py` | sensor suite, command bridge | placeholder |
| `frodo/plant/inspection.py` | independent seat measurement | done |
| `frodo/plant/line.py` | feeder, PLC, event log, scenarios | placeholder |
| `frodo/robot/manipulation/` | IK, grasp geometry, motion profile | done (oracle demo only) |
| `frodo/robot/localization/` | EKF + scan-to-map ICP | placeholder |
| `frodo/robot/perception/` | classical ring/bottle detector | placeholder |
| `frodo/robot/navigation/` | A* planner, path tracker, docking | placeholder |
| `frodo/robot/safety/` | independent slow/stop supervisor | placeholder |
| `frodo/robot/orchestrator/` | mission mode machine | placeholder |
| `frodo/robot/plc.py` | Modbus line client | placeholder |
| `frodo/apps/pick_place_demo.py` | ORACLE-pose manipulation demo | done |
| `frodo/apps/mission.py` | full mission runner | placeholder |

## Control tick

```
frame   = sensors.read(data)
pose    = localization.update(frame)
kit     = perception.detect(frame.camera)
wanted  = active controller for the current Mode (tracker / docking / manipulation)
command = safety.filter(frame, wanted)      # always last, uses raw lidar
bridge.apply(data, command); step physics; plant.line.step(data)
```

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

`apps/pick_place_demo.py` is an ORACLE run: true poses, robot pre-docked. It is not a
sensor-driven result and produces no acceptance evidence.
