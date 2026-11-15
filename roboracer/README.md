# RoboRacer stack

- `controller/adaptive_pure_pursuit.py`: ALPP racing controller (ROS 2 node). It subscribes to odometry, loads a raceline CSV, adapts the look-ahead to speed, and blends a curvature-limited speed with a deviation-limited speed.
- `localization/particle_filter/`: particle-filter localisation node (likelihood-field sensor model, odometry motion model) and its configs. It is based on the open-source JMU ROS 2 particle filter (MIT licence; see `LICENSE`).
- `mapping/cartographer/`: Cartographer configurations (`only_scan.lua`, `scan_imu.lua`, `localization*.lua`) and launch files, plus a small odometry publisher.
- `maps/competition_track.*`: the competition track map (1.2 cm per pixel).

The animations in the main README are drawn on this map by `docs/make_figures.py`.
