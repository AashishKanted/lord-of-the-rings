"""PLANT-ONLY run: the customer line by itself, with no robot action.

The robot stays parked at its start pose and never moves. This script stands in for the
robot in exactly one way: it sends READY for station A over Modbus/TCP so the feeder
starts, and then polls the line registers with the reference ``PLCClient``. Kits flow,
arrivals show up in the registers, and every inspection is REJECTED because nobody moves
a ring. The events file it writes is plant evidence only; it is not a robot run and it
cannot pass ``systems-check``.

    uv run python -m frodo.apps.line_demo --seconds 60 --out results/plant-only
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mujoco

from advaitic_systems.modbus import PLCClient

from ..cell import STATIONS
from ..config import ROOT, get_config
from ..log import get_logger, setup_logging
from ..plant import feeder
from ..plant.line import Line
from ..sim.scene import build_spec, reset

log = get_logger(__name__)
READY = 1
POLL_PERIOD_S = 0.5  # simulated seconds between register polls (faster than any arrival rate)


def run(seconds: float, out_dir: Path, belt_speed_mps: float | None = None,
        seed: int | None = None, port: int | None = None, station: str = "A") -> dict:
    model = build_spec(feeder.pool_layout()).compile()
    data = mujoco.MjData(model)
    reset(model, data)
    params = (feeder.draw_run_parameters(seed) if seed is not None
              else feeder.run_parameters(belt_speed_mps))
    st = STATIONS[station]
    base = 10 * st.id  # register block of this station (INTERFACES.md)
    seen_arrivals, polls = [], 0
    wall = time.perf_counter()
    with Line(model, params=params, events_path=out_dir / "events.jsonl", port=port) as line:
        plc = PLCClient(port=line.port)
        plc.command(1, st.id, 0, READY)
        next_poll = 0.0
        while data.time < seconds:
            mujoco.mj_step(model, data)
            line.step(data)
            if data.time >= next_poll:
                next_poll += POLL_PERIOD_S
                polls += 1
                latest = plc.read(base + 1, 1)[0]
                if latest and latest not in seen_arrivals:
                    seen_arrivals.append(latest)
                    log.info("t=%.2f Modbus poll: new arrival id %d at %s", data.time, latest,
                             station)
        registers = plc.read(0, 27)
        counts, kits = line.counts(), [dict(k) for k in line.kits]
        events = list(line.events.events)
    return {
        "mode": "PLANT ONLY: no robot action, not a robot run, not acceptance evidence",
        "simulated_s": round(data.time, 3),
        "wall_s": round(time.perf_counter() - wall, 2),
        "mujoco": mujoco.__version__,
        "station_fed": station,
        "belt_speed_mps": params.belt_speed_mps,
        "release_phase_s": params.release_phase_s,
        "arrival_ids_seen_over_modbus": seen_arrivals,
        "modbus_polls": polls,
        "registers_0_26": registers,
        "counts": counts,
        "kits": kits,
        "event_counts": {kind: sum(e["type"] == kind for e in events)
                         for kind in sorted({e["type"] for e in events})},
        "events_file": str(out_dir / "events.jsonl"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seconds", type=float, default=get_config().plant.demo_duration_s)
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "plant-only")
    parser.add_argument("--belt-speed", type=float, help="m/s for both belts (default nominal)")
    parser.add_argument("--seed", type=int, help="draw belt speeds and release phase from a seed")
    parser.add_argument("--station", choices=list(STATIONS), default="A")
    parser.add_argument("--port", type=int, help="PLC port (default plant.plc_port)")
    args = parser.parse_args()
    setup_logging()
    result = run(args.seconds, args.out, args.belt_speed, args.seed, args.port, args.station)
    (args.out / "summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
