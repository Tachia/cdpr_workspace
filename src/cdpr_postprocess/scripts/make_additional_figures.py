#!/usr/bin/env python3
"""Generate three extra Gazebo/ROS-derived figures for the CDPR manuscript.

When --bag is supplied, the script reads the same topics logged by data_logger.py.
Without a bag it produces deterministic preview data from the shared analytical
model so the plotting pipeline can be checked on machines without ROS/Gazebo.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np


def _add_source_paths() -> Path:
    root = Path(__file__).resolve().parents[3]
    control_src = root / "src" / "cdpr_control" / "src"
    if str(control_src) not in sys.path:
        sys.path.insert(0, str(control_src))
    return root


ROOT = _add_source_paths()

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

try:  # noqa: E402
    from scipy.interpolate import interp1d
except Exception:  # pragma: no cover
    interp1d = None

from cdpr_control.cdpr_model import (  # noqa: E402
    anchors,
    cable_params,
    cable_weight_compensation,
    catenary_endpoint,
    desired_wrench,
    iterative_catenary_qp,
    load_params,
    structure_matrix,
    world_attachment_points,
)
from cdpr_control.trajectories import sample as trajectory_sample  # noqa: E402


def set_style() -> None:
    plt.rcParams.update({
        "font.size": 10,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.constrained_layout.use": True,
    })


def save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"saved {path}")


def quaternion_to_rpy(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (qw * qy - qz * qx)
    pitch = math.copysign(math.pi / 2.0, sinp) if abs(sinp) >= 1.0 else math.asin(sinp)

    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return np.array([roll, pitch, yaw], dtype=float)


def interp_matrix(source_t: np.ndarray, source_values: np.ndarray, target_t: np.ndarray) -> np.ndarray:
    if len(source_t) == 0:
        return np.empty((0, source_values.shape[1] if source_values.ndim == 2 else 0))
    if len(source_t) == 1:
        return np.repeat(source_values[:1], len(target_t), axis=0)
    if interp1d is not None:
        fn = interp1d(source_t, source_values, axis=0, fill_value="extrapolate", bounds_error=False)
        return np.asarray(fn(target_t), dtype=float)
    return np.vstack([np.interp(target_t, source_t, source_values[:, i]) for i in range(source_values.shape[1])]).T


def read_rosbag(bag_path: Path, platform_name: str) -> Dict[str, np.ndarray]:
    try:
        import rosbag
    except Exception as exc:  # pragma: no cover - only used on ROS target.
        raise RuntimeError("rosbag is required to read Gazebo/ROS logs") from exc

    pose_t = []
    positions = []
    rpy = []
    cmd_t = []
    cmd_tensions = []
    actual_t = []
    actual_tensions = []
    sag_t = []
    sag_dirs = []
    error_t = []
    errors_mm = []
    metrics_t = []
    metrics = []

    with rosbag.Bag(str(bag_path), "r") as bag:
        for topic, msg, stamp in bag.read_messages():
            t = float(stamp.to_sec())
            if topic == "/gazebo/model_states":
                try:
                    idx = msg.name.index(platform_name)
                except ValueError:
                    continue
                pose = msg.pose[idx]
                pose_t.append(t)
                positions.append([pose.position.x, pose.position.y, pose.position.z])
                rpy.append(quaternion_to_rpy(
                    pose.orientation.x,
                    pose.orientation.y,
                    pose.orientation.z,
                    pose.orientation.w,
                ))
            elif topic == "/cable_tensions":
                cmd_t.append(t)
                cmd_tensions.append(list(msg.data))
            elif topic == "/actual_cable_tensions":
                actual_t.append(t)
                actual_tensions.append(list(msg.data))
            elif topic == "/sag_directions":
                if len(msg.data) >= 24:
                    sag_t.append(t)
                    sag_dirs.append(np.asarray(msg.data[:24], dtype=float).reshape(8, 3))
            elif topic == "/tracking_error":
                error_t.append(t)
                errors_mm.append(1000.0 * math.sqrt(msg.vector.x ** 2 + msg.vector.y ** 2 + msg.vector.z ** 2))
            elif topic == "/controller_metrics" and len(msg.data) >= 9:
                metrics_t.append(t)
                metrics.append(list(msg.data[:9]))

    data = {
        "pose_t": np.asarray(pose_t, dtype=float),
        "positions": np.asarray(positions, dtype=float),
        "rpy": np.asarray(rpy, dtype=float),
        "cmd_t": np.asarray(cmd_t, dtype=float),
        "cmd_tensions": np.asarray(cmd_tensions, dtype=float),
        "actual_t": np.asarray(actual_t, dtype=float),
        "actual_tensions": np.asarray(actual_tensions, dtype=float),
        "sag_t": np.asarray(sag_t, dtype=float),
        "sag_dirs": np.asarray(sag_dirs, dtype=float),
        "error_t": np.asarray(error_t, dtype=float),
        "errors_mm": np.asarray(errors_mm, dtype=float),
        "metrics_t": np.asarray(metrics_t, dtype=float),
        "metrics": np.asarray(metrics, dtype=float),
    }
    nonempty_times = [value for key, value in data.items() if key.endswith("_t") and len(value) > 0]
    if nonempty_times:
        t0 = min(float(value[0]) for value in nonempty_times)
        for key in list(data.keys()):
            if key.endswith("_t") and len(data[key]) > 0:
                data[key] = data[key] - t0
    return data


def preview_data(params: Dict[str, object], duration: float = 20.0, rate: float = 100.0) -> Dict[str, np.ndarray]:
    times = np.arange(0.0, duration, 1.0 / rate)
    cable = cable_params(params)
    positions = []
    rpy_rows = []
    tensions = []
    applied_tensions = []
    sag_dirs = []
    errors_mm = []
    metrics = []
    rng = np.random.default_rng(41)

    for t in times:
        desired = trajectory_sample("circle", float(t), params)
        positions.append(desired.position)
        rpy_rows.append(desired.rpy)
        wrench = desired_wrench(desired.acceleration, params=params)
        platform_tensions, W, lengths, units = iterative_catenary_qp(desired.position, desired.rpy, wrench, params=params)
        command = platform_tensions + cable_weight_compensation(lengths, params)
        command = np.clip(command, 0.0, float(cable["max_tension"]))
        tensions.append(command)
        applied_tensions.append(platform_tensions)
        sag_dirs.append(units)
        residual = float(np.linalg.norm(W @ platform_tensions - wrench))
        min_margin = float(np.min(command - float(cable["min_tension"])))
        upper_margin = float(np.min(float(cable["max_tension"]) - command))
        solve_ms = 2.8 + 0.55 * math.sin(2.0 * math.pi * t / 6.0) + rng.normal(0.0, 0.07)
        loop_ms = solve_ms + 0.55 + rng.normal(0.0, 0.04)
        saturation_count = int(np.count_nonzero((command <= float(cable["min_tension"]) + 1e-3) |
                                                (command >= float(cable["max_tension"]) - 1e-3)))
        err = max(0.4, 4.8 + 1.6 * math.sin(2.0 * math.pi * t / 8.0) + rng.normal(0.0, 0.35))
        errors_mm.append(err)
        metrics.append([1.0, t, loop_ms, solve_ms, residual, min_margin, upper_margin, saturation_count, err])

    values = np.asarray(tensions, dtype=float)
    applied_values = np.asarray(applied_tensions, dtype=float)
    return {
        "pose_t": times,
        "positions": np.asarray(positions, dtype=float),
        "rpy": np.asarray(rpy_rows, dtype=float),
        "cmd_t": times,
        "cmd_tensions": values,
        "actual_t": times,
        "actual_tensions": applied_values,
        "sag_t": times,
        "sag_dirs": np.asarray(sag_dirs, dtype=float),
        "error_t": times,
        "errors_mm": np.asarray(errors_mm, dtype=float),
        "metrics_t": times,
        "metrics": np.asarray(metrics, dtype=float),
    }


def get_tensions(data: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    if len(data["actual_t"]) > 0 and data["actual_tensions"].size:
        return data["actual_t"], data["actual_tensions"]
    return data["cmd_t"], data["cmd_tensions"]


def get_command_tensions(data: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    if len(data["cmd_t"]) > 0 and data["cmd_tensions"].size:
        return data["cmd_t"], data["cmd_tensions"]
    return get_tensions(data)


def figure_catenary_correction(data: Dict[str, np.ndarray], params: Dict[str, object], outdir: Path) -> None:
    if len(data["sag_t"]) == 0 or len(data["pose_t"]) == 0:
        raise RuntimeError("Figure 8 requires /sag_directions and /gazebo/model_states, or preview data.")

    t = data["sag_t"]
    positions = interp_matrix(data["pose_t"], data["positions"], t)
    rpy = interp_matrix(data["pose_t"], data["rpy"], t)
    tension_t, tension_values = get_tensions(data)
    tensions = interp_matrix(tension_t, tension_values, t) if len(tension_t) > 0 else np.zeros((len(t), 8))

    anchor_pts = anchors(params)
    angle_deg = np.zeros((len(t), 8))
    sag_mm = np.zeros((len(t), 8))
    for k, tk in enumerate(t):
        _, _, straight_units = structure_matrix(positions[k], rpy[k], model=0, params=params)
        sag_units = data["sag_dirs"][k]
        world_pts, _ = world_attachment_points(positions[k], rpy[k], params=params)
        for i in range(8):
            dot = float(np.clip(np.dot(straight_units[i], sag_units[i]), -1.0, 1.0))
            angle_deg[k, i] = math.degrees(math.acos(dot))
            endpoint = catenary_endpoint(
                anchor_pts[i],
                world_pts[i],
                float(tensions[k, i]) if tensions.size else float(cable_params(params)["nominal_tension"]),
                params=params,
                command_is_winch_tension=False,
            )
            sag_mm[k, i] = 1000.0 * abs(endpoint.sag_mid)

    fig, axes = plt.subplots(2, 1, figsize=(7.0, 5.0), sharex=True)
    axes[0].plot(t, np.mean(angle_deg, axis=1), lw=2.0, color="tab:blue", label="mean")
    axes[0].plot(t, np.max(angle_deg, axis=1), lw=1.6, color="tab:orange", label="max")
    axes[0].set_ylabel("Direction change [deg]")
    axes[0].set_title("Model 1 catenary direction correction along the circular path")
    axes[0].legend(loc="upper right")

    axes[1].plot(t, np.mean(sag_mm, axis=1), lw=2.0, color="tab:green", label="mean sag")
    axes[1].fill_between(t, np.min(sag_mm, axis=1), np.max(sag_mm, axis=1), color="tab:green", alpha=0.18,
                         label="min-max cable sag")
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Sag estimate [mm]")
    axes[1].legend(loc="upper right")
    save(fig, outdir / "fig8_catenary_direction_correction.png")


def figure_tension_heatmap(data: Dict[str, np.ndarray], params: Dict[str, object], outdir: Path) -> None:
    tension_t, tensions = get_command_tensions(data)
    if len(tension_t) == 0:
        raise RuntimeError("Figure 9 requires /actual_cable_tensions or /cable_tensions, or preview data.")

    cable = cable_params(params)
    tmin = float(cable["min_tension"])
    tmax = float(cable["max_tension"])
    utilization = 100.0 * (tensions - tmin) / max(tmax - tmin, 1e-9)
    utilization = np.clip(utilization, 0.0, 100.0)

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.1), sharex=True, gridspec_kw={"height_ratios": [3.0, 1.0]})
    image = axes[0].imshow(
        utilization.T,
        aspect="auto",
        origin="lower",
        extent=[float(tension_t[0]), float(tension_t[-1]), 1, tensions.shape[1]],
        cmap="magma",
        vmin=0.0,
        vmax=100.0,
    )
    axes[0].set_title("Cable tension utilization heatmap")
    axes[0].set_ylabel("Cable index")
    axes[0].set_yticks(np.arange(1, tensions.shape[1] + 1))
    fig.colorbar(image, ax=axes[0], label="utilization of allowable range [%]")

    axes[1].plot(tension_t, np.min(tensions - tmin, axis=1), color="tab:blue", lw=1.8, label="lower margin")
    axes[1].plot(tension_t, np.min(tmax - tensions, axis=1), color="tab:red", lw=1.8, label="upper margin")
    axes[1].axhline(0.0, color="black", ls="--", lw=1.0)
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Margin [N]")
    axes[1].legend(loc="upper right", ncol=2)
    save(fig, outdir / "fig9_tension_utilization_heatmap.png")


def figure_controller_metrics(data: Dict[str, np.ndarray], outdir: Path) -> None:
    if len(data["metrics_t"]) == 0 or data["metrics"].size == 0:
        raise RuntimeError("Figure 10 requires /controller_metrics, or preview data.")

    t = data["metrics_t"]
    metrics = data["metrics"]
    loop_ms = metrics[:, 2]
    solve_ms = metrics[:, 3]
    residual = metrics[:, 4]
    saturation = metrics[:, 7]
    error_mm = metrics[:, 8]

    fig, axes = plt.subplots(3, 1, figsize=(7.1, 6.2), sharex=True)
    axes[0].plot(t, loop_ms, color="tab:blue", lw=1.6, label="controller loop")
    axes[0].plot(t, solve_ms, color="tab:orange", lw=1.6, label="QP/model solve")
    axes[0].axhline(5.0, color="black", ls="--", lw=1.0, label="200 Hz budget")
    axes[0].set_ylabel("Time [ms]")
    axes[0].set_title("Controller timing and numerical health")
    axes[0].legend(loc="upper right", ncol=3)

    axes[1].plot(t, residual, color="tab:green", lw=1.6)
    axes[1].set_ylabel("||WT - w||")
    axes[1].set_yscale("symlog", linthresh=1e-3)

    axes[2].plot(t, error_mm, color="tab:purple", lw=1.5, label="tracking error")
    axes[2].step(t, saturation, where="post", color="tab:red", lw=1.1, alpha=0.85, label="saturated cables")
    axes[2].set_xlabel("Time [s]")
    axes[2].set_ylabel("mm / count")
    axes[2].legend(loc="upper right", ncol=2)
    save(fig, outdir / "fig10_controller_timing_and_residual.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag", type=Path, default=None, help="Gazebo/ROS rosbag recorded with data_logger.py")
    parser.add_argument("--figures", type=Path, default=ROOT / "figures")
    parser.add_argument("--platform-name", default="cdpr_platform")
    parser.add_argument("--preview-duration", type=float, default=20.0)
    parser.add_argument("--preview-rate", type=float, default=100.0)
    args = parser.parse_args()

    set_style()
    params = load_params()
    if args.bag:
        data = read_rosbag(args.bag, args.platform_name)
    else:
        data = preview_data(params, duration=args.preview_duration, rate=args.preview_rate)

    figure_catenary_correction(data, params, args.figures)
    figure_tension_heatmap(data, params, args.figures)
    figure_controller_metrics(data, args.figures)


if __name__ == "__main__":
    main()
