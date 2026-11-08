# Customer integration contract

A PLC is the small industrial controller coordinating a production line. For this
assignment, our mock PLC is just a local program playing that role: the robot
reports readiness and completion, while customer inspection owns accept/reject.
You do not need to program a real PLC or buy industrial software.

## Roles

1. **Robot process:** reads line state, sends commands through Modbus/TCP.
2. **Plant / customer process:** moves conveyors and kits, reads independent
   simulator geometry/contact measurements, updates the line controller.
3. **Inspector:** records ground-truth acceptance downstream and exports evidence.

The robot may not write accepted counts or inspection results. Do not implement
inspection by copying the robot's own success flag. The plant process can use
ground truth; the robot's sensor inputs must remain a separate interface.

## Minimal Modbus/TCP map

Localhost TCP port 1502, unit ID 1, zero-based addresses, 16-bit unsigned registers,
big-endian wire values. Supported subset: FC03 read holding registers, FC16 write
multiple registers. No TLS/authentication or production deployment is provided.

| Address | Meaning | Robot access |
|---|---|---|
| 0 | Schema version = 1 | Read |
| 1 | Last applied command sequence | Read |
| 2 | Last command status: 1 applied, 2 duplicate acknowledged | Read |
| 3 | Total unique kit arrivals | Read |
| 10 / 20 | A / B ready | Read |
| 11 / 21 | Latest A / B arrival ID | Read |
| 12 / 22 | Latest inspected kit ID | Read |
| 13 / 23 | Last result: 0 none, 1 accepted, 2 rejected | Read |
| 14 / 24 | Accepted count | Read |
| 15 / 25 | Rejected count | Read |
| 16 / 26 | Nominal belt speed, mm/s = 30 | Read |
| 100–103 | `[sequence, station_id, kit_id, command]`, atomic FC16 write | Write |

Station IDs A=1, B=2. Commands: 1 READY, 2 NOT_READY, 3 ASSEMBLY_DONE,
4 ROBOT_FAULT. For commands without a kit, use kit_id=0. IDs/sequences are 1–65535;
no wrap is needed for this task. Start a fresh controller per mission. Retry an
ambiguous write using exactly the same sequence and payload. Advance by one only
when acknowledged. Reusing the sequence with different contents is an error.

The latest-arrival register is not a queue. Poll faster than arrival rate and
keep your own ID ledger; after a long outage, reconcile with the plant event log
and reject unknown/stale work rather than inventing success. You can propose an
extension for a production event queue. The speed register is explicitly nominal;
use the conveyor encoder for actual test speeds.

## Attach the simulator

Use this pattern inside your plant runner, not as a robot-side shortcut:

```python
import threading
from advaitic_systems.modbus import PLCServer

server = PLCServer(("127.0.0.1", 1502))
thread = threading.Thread(target=server.serve_forever, daemon=True)
thread.start()
try:
    # In your simulation loop, on an actual bottle entry crossing:
    with server.lock:
        server.model.arrive(station=1, kit_id=101)

    # Step your simulator and robot. At the downstream crossing, measure ring
    # pose relative to bottle, contacts, release and continuous settled hold.
    # with server.lock:
    #     server.model.inspect(1, 101, actual_sim_measurement)
finally:
    server.shutdown()
    server.server_close()
    thread.join()
```

Robot adapter example, in a separate process:

```python
from advaitic_systems.modbus import PLCClient
plc = PLCClient(port=1502)
plc.command(1, 1, 0, 1)       # ready at A
kit_id = plc.read(11, 1)[0]
# After real manipulation, with the same kit ID:
plc.command(2, 1, kit_id, 3)  # completion only; accepted remains unchanged
```

The test double stores state in memory. Restart recovery across a PLC process
crash is outside this assignment; a TCP disconnect/reconnect is in scope.
To test lost acknowledgement, use a local TCP fault proxy or deliberately close
the client connection after a write is applied but before reading its response.
Then reconnect and retry; include the controller's real ledger in evidence.

## Evidence format

Export UTF-8 JSONL, one event per line, timestamps `t` in simulated seconds,
monotonic from mission start. Station IDs are integers 1/2. Every run needs:

| `type` | Additional fields |
|---|---|
| `sample` at ≥10 Hz | `x,y,yaw,base_speed,ee_speed,arm_stowed` |
| `dock` | `station,x,y,yaw` |
| `kit_arrived` | `station,kit_id,belt_speed_mps` |
| `assembly_done` | `station,kit_id` |
| `seat_verified` | `station,kit_id,dx_m,dy_m,z_m,tilt_rad,hold_s,released,bottle_contact,kit_contact` |
| `inspection` | `station,kit_id,dx_m,dy_m,z_m,tilt_rad,hold_s,released,bottle_contact,kit_contact,belt_speed_mps` |
| `human_enter`, `stop_command`, `human_exit`, `resume` | timestamps; attach richer sensor/control logs separately |
| `plc_retry` | `first_request_hex,retry_request_hex,controller_applied_count,accepted_before,accepted_after` plus raw socket/controller trace separately |

Speeds are nonnegative magnitudes; positions/angles use the customer world frame.
The plant emits `seat_verified` when independent geometry/contact measurement first
completes a continuous 0.5 s valid hold. Its timestamp ends the assembly cycle;
`assembly_done` remains a robot command, not proof of seating. The checker expects
successive arrivals at least one 450 mm pitch apart at constant station belt speed
and inspection at the bottle's downstream crossing, allowing 0.11 s timestamp
error. The inspection speed must match that kit's arrival speed.
For `plc_retry`, include the complete Modbus/TCP FC16 request frames before and
after reconnect. The checker compares their command payload and checks the stated
single application; the separate socket trace and controller ledger still need a
live rerun to establish that the fault really occurred.

`hold_s` is a continuously accumulated valid, released, settled contact duration,
reset on any violation. Require ring relative linear speed ≤0.01 m/s and angular
speed ≤0.1 rad/s throughout the hold; retain raw samples so reviewers can recompute
it. Inspection uses real geometry at the downstream crossing. Keep richer sensor,
state-transition and command logs in separate files; checker rejects unknown event
types to catch schema mistakes. Export files are evidence, not trusted attestations.

The starter checker validates count/order, pose thresholds, basic motion continuity,
nominal cycle/mission time, and stop/retry log consistency. It does **not** implement
collision detection, visibility, contact truth, live sensor freshness, belt motion,
reachability, gripping, or tamper resistance. The reviewer rerun validates those
against your actual simulator and source. A generated passing trace is not a solution.
