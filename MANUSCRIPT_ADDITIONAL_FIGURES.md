# Additional Gazebo/ROS Figures for the Journal Manuscript

These three figures are supplementary to the seven reproduction figures already generated. They should be produced from the same Gazebo/ROS runs by recording the standard logger topics, especially `/gazebo/model_states`, `/actual_cable_tensions`, `/sag_directions`, `/tracking_error`, and `/controller_metrics`.

Generate them from a recorded hybrid circular trajectory bag:

```bash
rosrun cdpr_postprocess make_additional_figures.py --bag /tmp/cdpr_circle_hybrid.bag --figures figures
```

If no rosbag is supplied, the script generates deterministic preview plots from the same analytical equations; use the rosbag version for the final journal figures.

## Fig. 8 - Catenary Direction Correction

File: `figures/fig8_catenary_direction_correction.png`

Best manuscript location: Section 2.2, immediately after the Hybrid Catenary-Elastic Dynamic Model derivation.

Purpose: This figure makes Model 1 visually interpretable. The main text currently states that the catenary solver updates the true tangent vector, but readers benefit from seeing how large the correction is during an actual trajectory. The upper panel reports the mean and maximum angular deviation between the rigid Model 0 direction and the Model 1 sag-corrected direction. The lower panel reports the corresponding cable sag envelope.

Suggested manuscript use: Refer to it after introducing the corrected structure matrix `W(x,t)`. The caption should emphasize that even sub-degree direction changes accumulate into measurable wrench and pose errors in a large CDPR.

## Fig. 9 - Cable Tension Utilization Heatmap

File: `figures/fig9_tension_utilization_heatmap.png`

Best manuscript location: Section 4.1, after the computed torque control comparison and before the tracking-error plot.

Purpose: The original tension line plot shows individual cable histories, but it is hard to see actuator duty sharing and bound proximity at a glance. This heatmap compresses all eight tensions into a single controller-effort view, with the lower panel showing distance from `Tmin` and `Tmax`.

Suggested manuscript use: Use it to support claims about unilateral tension constraints, saturation, and tension redistribution during circular motion. It is especially useful when explaining why QP-based allocation is required instead of a simple pseudoinverse.

## Fig. 10 - Controller Timing and Numerical Health

File: `figures/fig10_controller_timing_and_residual.png`

Best manuscript location: Section 4.1 or the Experimental Validation subsection, immediately after presenting the controller architecture.

Purpose: Journal reviewers will likely ask whether the hybrid catenary correction is still compatible with real-time control. This figure directly answers that question by plotting controller loop time, QP/model solve time, the 200 Hz timing budget, wrench reconstruction residual, tracking error, and the number of saturated cables.

Suggested manuscript use: Use it as a credibility figure for implementation feasibility. In the caption, state the CPU and ROS/Gazebo configuration used for the final run, then report median and 95th-percentile loop time in the text.

## Recommended Cross-References

- Use Fig. 8 when explaining why Model 0 loses accuracy despite using the same anchor and platform geometry.
- Use Fig. 9 when discussing the unilateral tension bounds `Tmin <= Ti <= Tmax`.
- Use Fig. 10 when defending the practical real-time viability of Model 1 in the controller loop.

These figures can go in the main paper if page budget permits. If the journal has strict figure limits, keep Fig. 10 in the main paper and move Figs. 8 and 9 to supplementary material.
