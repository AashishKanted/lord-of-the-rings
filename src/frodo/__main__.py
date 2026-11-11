"""Command-line entry point.

    python -m frodo view [--dock A]         interactive viewer
    python -m frodo render OUT.png          headless snapshot
    python -m frodo camera [--camera NAME]  save RGB-D frames from one camera
    python -m frodo demo [--video FILE]     ORACLE pick-and-place demo
    python -m frodo sensors [--out DIR]     drive a scripted twist, dump lidar/camera/odometry
    python -m frodo mission [--scenario S]  full mission (not implemented yet)
"""
from __future__ import annotations

import sys


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        return
    command, sys.argv = sys.argv[1], [f"frodo {sys.argv[1]}", *sys.argv[2:]]
    if command == "view":
        sys.argv.insert(1, "--view")
        from .sim.scene import main as run
    elif command == "render":
        sys.argv.insert(1, "--render")
        from .sim.scene import main as run
    elif command == "camera":
        from .sim.camera import main as run
    elif command == "demo":
        from .apps.pick_place_demo import main as run
    elif command == "sensors":
        from .apps.sensors_demo import main as run
    elif command == "mission":
        import argparse

        from .apps.mission import run as mission
        parser = argparse.ArgumentParser()
        parser.add_argument("--scenario", default="nominal")
        parser.add_argument("--out", default=None)
        args = parser.parse_args()

        def run():
            mission(args.scenario, args.out or f"results/{args.scenario}")
    else:
        sys.exit(f"unknown command '{command}'\n{__doc__}")
    run()


if __name__ == "__main__":
    main()
