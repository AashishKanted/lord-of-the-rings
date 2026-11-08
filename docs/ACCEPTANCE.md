# Acceptance runs

Provide a headless command for each scenario and seed, without manual clicking.
Record simulated time, wall time, simulator version and any
external dependencies. Local CPU execution must be possible; realtime speed is
desirable but not a pass condition. No paid API should be required to evaluate.

## Published scenarios

| Scenario | Runs | What must happen |
|---|---|---|
| Nominal | Seeds 0, 1, 2 | 5 accepted at A, then 5 at B; every accepted cycle ≤12 s; mission ≤270 s |
| Human interruption | Seed 0 | Person enters during ring transfer; stop, clear/acknowledge, reacquire or reject affected kit, then finish 5+5 within 360 s |
| PLC retry | Seed 0 | Drop one ASSEMBLY_DONE acknowledgement after apply; reconnect/retry without duplicate acceptance; finish 5+5 within 360 s |

For nominal seeds, vary static belt speed within 25–35 mm/s and kit release phase
within 0–1 s after readiness. Use the seed to select values reproducibly and report
them; additional seeds may be chosen within the same ranges. No object geometry
randomization is required. During assembly, belts never pause. Feeder gating is
per the customer packet. Failed kits are rejected downstream and replaced, not
deleted from the logs. Nominal runs may include rejects but must meet all final
counts, accepted-cycle deadlines and mission limit. Report attempt-level yields
and timing for failures too; do not hide them behind eventual completion.
The accepted-cycle clock ends at independently measured `seat_verified`, not the
robot's `ASSEMBLY_DONE` message.

In the human test, publish a parameter for interruption timing and duration
(2–4 s). The timing must be configurable within an active transfer. Stop
behaviour must be sensor-triggered, not a scheduled animation of a stationary
person. Log detection, stop request, measured velocity, clear and resume. A kit
that exits the working window unfinished is rejected and replaced. Grasp retention
during the stop is allowed; the robot must not keep transporting the ring.

For PLC retry, record actual bytes/requests and controller application ledger.
An injected fault log without a socket interruption is insufficient. Do not
advance accepted counts on retry or treat ASSEMBLY_DONE as customer acceptance.

## Simulator requirements

- Ring physics, bottle/kit frames, non-stationary belt and correct downstream
  measurement. No object pose overwrite, hidden weld after release or fabricated
  contact/hold measurement.
- Known-map navigation is acceptable, but odometry/IMU/range observations must
  feed the estimate and docking. Show collision checks and arm-stowed travel.
- Robot sensing and customer ground truth are separate. Label any oracle mode.
- The same controller handles both station frames and the variable belt speeds.
- Human presence prevents motion until explicit recovery; stale moving kits are
  not blindly pursued. Integration retries preserve identity and counts.
- Clean-checkout reproducibility and alignment of exported logs with live state.

The starter `systems-check` is a partial evidence consistency check. Your evaluator
should also assert collisions, belt speed,
seating hold, stale-work rejection and live recovery from actual simulator state.
Keep those assertions separate from robot decision logic.

## Out of scope

Full SLAM, photorealism, real hardware, deformable rings, force-controlled press
fits, real industrial networks, production PLC programming, cloud hosting, and
complete production safety design. You can discuss these as transfer risks.

## Candidate deliverables

1. Solution repository URL and pinned commit; setup instructions and runnable local commands.
2. Digital twin and declared simplifications; all asset source/licence records.
3. Architecture and state transitions; customer-interface and sensing boundaries.
4. Logs and metrics for all five runs, including failures and retries.
5. A short video showing assembly, station change, human stop and recovery.
6. Failure analysis, timing budget versus measurements, next engineering steps.

We may ask you to change a parameter and diagnose a failure live. You should own
the system even when you use AI tools. There is no separate rule requiring you
to implement standard components from scratch.
