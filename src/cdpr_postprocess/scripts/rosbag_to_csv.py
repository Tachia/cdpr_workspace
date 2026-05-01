#!/usr/bin/env python3
"""Extract platform pose, cable tensions, and tracking errors from a CDPR rosbag."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def write_rows(path: Path, fieldnames, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag", type=Path)
    parser.add_argument("--outdir", type=Path, default=Path("data/rosbag_csv"))
    parser.add_argument("--platform-name", default="cdpr_platform")
    args = parser.parse_args()

    import rosbag

    pose_rows = []
    command_rows = []
    actual_rows = []
    error_rows = []
    metrics_rows = []
    with rosbag.Bag(str(args.bag), "r") as bag:
        for topic, msg, stamp in bag.read_messages():
            time = stamp.to_sec()
            if topic == "/gazebo/model_states":
                try:
                    idx = msg.name.index(args.platform_name)
                except ValueError:
                    continue
                pose = msg.pose[idx]
                twist = msg.twist[idx]
                pose_rows.append({
                    "time": time,
                    "x": pose.position.x,
                    "y": pose.position.y,
                    "z": pose.position.z,
                    "qx": pose.orientation.x,
                    "qy": pose.orientation.y,
                    "qz": pose.orientation.z,
                    "qw": pose.orientation.w,
                    "vx": twist.linear.x,
                    "vy": twist.linear.y,
                    "vz": twist.linear.z,
                })
            elif topic == "/cable_tensions":
                row = {"time": time}
                row.update({f"cable_{i + 1}": value for i, value in enumerate(msg.data)})
                command_rows.append(row)
            elif topic == "/actual_cable_tensions":
                row = {"time": time}
                row.update({f"cable_{i + 1}": value for i, value in enumerate(msg.data)})
                actual_rows.append(row)
            elif topic == "/tracking_error":
                norm = (msg.vector.x ** 2 + msg.vector.y ** 2 + msg.vector.z ** 2) ** 0.5
                error_rows.append({
                    "time": time,
                    "ex": msg.vector.x,
                    "ey": msg.vector.y,
                    "ez": msg.vector.z,
                    "error_norm": norm,
                })
            elif topic == "/controller_metrics":
                labels = [
                    "model",
                    "ros_time",
                    "loop_ms",
                    "solve_ms",
                    "wrench_residual",
                    "min_margin",
                    "upper_margin",
                    "saturation_count",
                    "error_norm_mm",
                ]
                row = {"time": time}
                row.update({label: msg.data[i] for i, label in enumerate(labels) if i < len(msg.data)})
                metrics_rows.append(row)

    write_rows(args.outdir / "platform_pose.csv", ["time", "x", "y", "z", "qx", "qy", "qz", "qw", "vx", "vy", "vz"], pose_rows)
    if command_rows:
        write_rows(args.outdir / "commanded_tensions.csv", command_rows[0].keys(), command_rows)
    if actual_rows:
        write_rows(args.outdir / "actual_tensions.csv", actual_rows[0].keys(), actual_rows)
    write_rows(args.outdir / "tracking_errors.csv", ["time", "ex", "ey", "ez", "error_norm"], error_rows)
    if metrics_rows:
        write_rows(args.outdir / "controller_metrics.csv", metrics_rows[0].keys(), metrics_rows)
    print(f"CSV files written to {args.outdir}")


if __name__ == "__main__":
    main()
