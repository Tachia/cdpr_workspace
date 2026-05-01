# CDPR Model 0/1 Gazebo Verification Workspace

This catkin workspace implements an 8-cable cable-driven parallel robot (CDPR) for reproducing the requested Model 0 and Model 1 comparisons from "Comparative Study of Classical and Novel Dynamic Models for Cable-Driven Parallel Robot Control".

The referenced PDF did not expose the requested Table I cable constants, so the remaining cable properties are centralized in `src/cdpr_gazebo/config/cdpr_params.yaml`. The supplied linear density is chosen so the catenary check gives a 14.5 mm sag for a 40 N, 8 m span.

## Packages

- `cdpr_gazebo`: Gazebo world, static 12 x 12 x 6 m frame, 50 kg 1 m cube platform, and C++ force plugin.
- `cdpr_control`: Python Model 0/Model 1 computed-torque controllers, QP tension allocation, trajectories, constant-tension experiment, and rosbag logger.
- `cdpr_postprocess`: Analytical QP workspace computation, rosbag-to-CSV extraction, and 300 dpi figure generation.

## Dependencies

Target stack:

- Ubuntu 20.04, ROS Noetic, Gazebo 11
- `ros-noetic-gazebo-ros-pkgs`
- `python3-numpy python3-scipy python3-matplotlib python3-yaml python3-cvxopt`

Install example:

```bash
sudo apt update
sudo apt install ros-noetic-desktop-full ros-noetic-gazebo-ros-pkgs \
  python3-numpy python3-scipy python3-matplotlib python3-yaml python3-cvxopt
```

## Build

From this workspace root:

```bash
catkin_make
source devel/setup.bash
```

If your archive tool drops executable bits:

```bash
chmod +x src/cdpr_control/scripts/*.py src/cdpr_postprocess/scripts/*.py
```

## Simulation Runs

Model 0 classical CTC on the 20 s circular trajectory:

```bash
roslaunch cdpr_control run_experiment.launch controller:=classical model_type:=0 trajectory:=circle bag_path:=/tmp/cdpr_circle_classical.bag
```

Model 1 hybrid CTC on the 20 s circular trajectory:

```bash
roslaunch cdpr_control run_experiment.launch controller:=hybrid model_type:=1 trajectory:=circle bag_path:=/tmp/cdpr_circle_hybrid.bag
```

Constant-tension sag/pose experiment:

```bash
roslaunch cdpr_control constant_tension.launch model_type:=1 cable_index:=0 tension:=40.0 bag_path:=/tmp/cdpr_constant_tension.bag
```

Zig-zag tracking comparison:

```bash
timeout 15s roslaunch cdpr_control run_experiment.launch controller:=classical model_type:=0 trajectory:=zigzag bag_path:=/tmp/cdpr_zigzag_classical.bag
timeout 15s roslaunch cdpr_control run_experiment.launch controller:=hybrid model_type:=1 trajectory:=zigzag bag_path:=/tmp/cdpr_zigzag_hybrid.bag
```

Long hybrid circle for spatial tracking error:

```bash
timeout 30s roslaunch cdpr_control run_experiment.launch controller:=hybrid model_type:=1 trajectory:=circle bag_path:=/tmp/cdpr_circle_hybrid_long.bag
```

## Analytical Workspace

The workspace maps are computed offline without Gazebo. The script uses the same straight-cable and catenary endpoint equations as the plugin, and `cvxopt` is the primary QP solver.

Full manuscript setting:

```bash
rosrun cdpr_postprocess compute_workspace.py --spacing 0.5 --z 2.5 --samples 200 --v-max 0.5 --a-max 1.0 --model 1
```

This writes:

- `data/workspace_maps.csv`
- `data/workspace_summary.json`

The included preview CSV/figures were generated with a smaller sample count on this local machine because `cvxopt` is not installed here. Re-run the command above on the ROS target to overwrite them with the full 200-sample DFW.

## Figures

Generate all seven requested 300 dpi PNGs:

```bash
rosrun cdpr_postprocess make_figures.py --figures figures --workspace-csv data/workspace_maps.csv --workspace-summary data/workspace_summary.json
```

Outputs:

- `figures/fig1_cable_sag_profile.png`
- `figures/fig2_tension_distribution_circle.png`
- `figures/fig3_binary_workspace_maps.png`
- `figures/fig4_feasibility_difference_contour.png`
- `figures/fig5_feasibility_difference_surface.png`
- `figures/fig6_tracking_error_zigzag.png`
- `figures/fig7_spatial_tracking_error_scatter.png`

Convert a rosbag to CSV:

```bash
rosrun cdpr_postprocess rosbag_to_csv.py /tmp/cdpr_circle_hybrid.bag --outdir data/circle_hybrid
```

## Additional Manuscript Figures

Three extra Gazebo/ROS-derived figures are available for the journal version:

```bash
rosrun cdpr_postprocess make_additional_figures.py --bag /tmp/cdpr_circle_hybrid.bag --figures figures
```

Equivalent launch wrapper:

```bash
roslaunch cdpr_postprocess additional_figures.launch bag:=/tmp/cdpr_circle_hybrid.bag figures:=figures
```

Outputs:

- `figures/fig8_catenary_direction_correction.png`
- `figures/fig9_tension_utilization_heatmap.png`
- `figures/fig10_controller_timing_and_residual.png`

The controller publishes `/controller_metrics` for Fig. 10. The logger records that topic automatically. See `MANUSCRIPT_ADDITIONAL_FIGURES.md` for where each figure should be used in the final manuscript.

## Model Notes

- `/cable_tensions` is a `std_msgs/Float32MultiArray` of eight commanded winch tensions.
- Model 0 applies `T_i u_i` along the straight segment from platform attachment to frame anchor.
- Model 1 computes the catenary tangent at the platform by solving horizontal tension with bracketing. The plugin applies `T_platform u_sag`, with `T_platform = T_cmd - 0.5 cable_weight`.
- The hybrid controller solves for platform tensions using the catenary-corrected structure matrix, then adds the half-cable-weight feed-forward compensation before publishing commands.
- The logger records `/gazebo/model_states`, `/cable_tensions`, `/actual_cable_tensions`, `/sag_directions`, `/desired_pose`, `/tracking_error`, and `/controller_metrics`.
