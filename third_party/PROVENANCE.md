# Third-party assets

| Path | Source | Commit | Licence | Changes |
|---|---|---|---|---|
| `mujoco_menagerie/stanford_tidybot/` | https://github.com/google-deepmind/mujoco_menagerie (`stanford_tidybot/`) | `98a70c569c4c1b6da467a20012f766ee1f8dc521` | MIT (base); Kinova Gen3 BSD-3-Clause and Robotiq 2F-85 BSD-2-Clause retain their own licences, see the folder's `LICENSE`/`README.md` | Preview PNGs removed; model files unmodified. Scene-level changes (0.40 m arm riser, gripper tilt, lidar and camera mounts, wrist camera bracket, gravity compensation) are applied in code (`src/frodo/sim/scene.py`), not by editing these files. |
