# ringcell architecture

Solution code for the mobile ring-on-bottle assembly cell. The starter
(`src/advaitic_systems/`, `spec/`, `docs/`) is unchanged; everything here lives in
`src/ringcell/`, `config/`, `scripts/` and `third_party/`.

## The one rule

The robot software sees the world only through `ringcell.interfaces` (a `SensorFrame`
in, a `RobotCommand` out) and Modbus/TCP. `ringcell.robot` never imports `ringcell.sim`
or `ringcell.plant`; `tests/ringcell/test_architecture.py` fails if it does. Everything
runs in one process and one deterministic loop, with the plant's PLC server on a thread.

```
            ringcell.sim                          ringcell.robot
  +---------------------------+   SensorFrame   +--------------------------------+
  | scene (MuJoCo, from spec) | --------------> | orchestrator (Mode machine)    |
  | sensors: 2 lidars, odom,  |                 |   localization  EKF + scan ICP |
  |   IMU, RGB-D, encoder     |                 |   perception    ring / bottle  |
  | actuation, grasp weld     | <-------------- |   navigation    plan/track/dock|
  +---------------------------+   RobotCommand  |   manipulation  IK, grasp      |
            ^ ground truth                      |   safety        slow / stop    |
  +---------------------------+   Modbus/TCP    |   plc           line client    |
  | ringcell.plant            | <-------------> +--------------------------------+
  | line: feeder, PLC server, |
  | inspection, event log     |  --> results/<scenario>/events.jsonl --> systems-check
  +---------------------------+
```

## Layout

| Path | Role | State |
|---|---|---|
| `config/ringcell.toml` | every tunable; customer numbers stay in `spec/` | in use |
| `scripts/run.sh` | launcher: `view`, `render`, `camera`, `demo`, `mission` | in use |
| `ringcell/config.py`, `log.py` | config loader; logger with `file:line` | done |
| `ringcell/cell.py` | read-only customer spec, station frames | done |
| `ringcell/interfaces.py` | sensor/command dataclasses, `Mode` | done |
| `ringcell/sim/scene.py`, `camera.py`, `grasp.py` | cell model, RGB-D camera, grasp constraint | done |
| `ringcell/sim/sensors.py`, `actuation.py` | sensor suite, command bridge | placeholder |
| `ringcell/plant/inspection.py` | independent seat measurement | done |
| `ringcell/plant/line.py` | feeder, PLC, event log, scenarios | placeholder |
| `ringcell/robot/manipulation/` | IK, grasp geometry, motion profile | done (oracle demo only) |
| `ringcell/robot/localization/` | EKF + scan-to-map ICP | placeholder |
| `ringcell/robot/perception/` | classical ring/bottle detector | placeholder |
| `ringcell/robot/navigation/` | A* planner, path tracker, docking | placeholder |
| `ringcell/robot/safety/` | independent slow/stop supervisor | placeholder |
| `ringcell/robot/orchestrator/` | mission mode machine | placeholder |
| `ringcell/robot/plc.py` | Modbus line client | placeholder |
| `ringcell/apps/pick_place_demo.py` | ORACLE-pose manipulation demo | done |
| `ringcell/apps/mission.py` | full mission runner | placeholder |

## Control tick

```
frame   = sensors.read(data)
pose    = localization.update(frame)
kit     = perception.detect(frame.camera)
wanted  = active controller for the current Mode (tracker / docking / manipulation)
command = safety.filter(frame, wanted)      # always last, uses raw lidar
bridge.apply(data, command); step physics; plant.line.step(data)
```

## Sensors

Two low 2D lidars on opposite base corners (360 degree cover; used for localization and
for safety), wheel odometry, IMU, conveyor encoder. Two camera mounts exist (wrist and a
fixed mast); `sensors.cameras.perception_source` selects the single RGB-D camera that
feeds perception, as the customer allows one.

## Honesty markers

`apps/pick_place_demo.py` is an ORACLE run: true poses, robot pre-docked. It is not a
sensor-driven result and produces no acceptance evidence.
