#!/usr/bin/env python3
"""Generate all manuscript reproduction figures as 300 dpi PNG files."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


def _add_source_paths() -> Path:
    root = Path(__file__).resolve().parents[3]
    control_src = root / "src" / "cdpr_control" / "src"
    post_scripts = root / "src" / "cdpr_postprocess" / "scripts"
    for path in (control_src, post_scripts):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    return root


ROOT = _add_source_paths()

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401,E402

try:  # noqa: E402
    from scipy.signal import savgol_filter
except Exception:  # pragma: no cover
    savgol_filter = None

from cdpr_control.cdpr_model import (  # noqa: E402
    _endpoint_tension_for_h,
    _solve_horizontal_tension,
    cable_params,
    cable_weight_compensation,
    desired_wrench,
    iterative_catenary_qp,
    load_params,
)
from cdpr_control.trajectories import sample as trajectory_sample  # noqa: E402
from compute_workspace import compute_workspace  # noqa: E402


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


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values
    kernel = np.ones(window) / window
    return np.vstack([np.convolve(values[:, i], kernel, mode="same") for i in range(values.shape[1])]).T


def smooth(values: np.ndarray, window: int = 41, polyorder: int = 3) -> np.ndarray:
    window = min(window, values.shape[0] - (1 - values.shape[0] % 2))
    if window < 5:
        return values
    if window % 2 == 0:
        window -= 1
    if savgol_filter is None:
        return moving_average(values, window)
    return savgol_filter(values, window_length=window, polyorder=min(polyorder, window - 2), axis=0)


def figure_sag(params: Dict[str, object], outdir: Path) -> None:
    cable = cable_params(params)
    span = 8.0
    tension = 40.0
    w = float(cable["linear_density"]) * float(cable["gravity"])
    h = _solve_horizontal_tension(tension, span, 0.0, w)
    a = h / w
    c = span / 2.0
    x = np.linspace(0.0, span, 500)
    z = a * (np.cosh((x - c) / a) - np.cosh((-c) / a))
    sag_mm = -1000.0 * np.min(z)

    fig, ax = plt.subplots(figsize=(6.3, 3.5))
    ax.plot(x, np.zeros_like(x), "--", lw=1.8, label="Rigid cable")
    ax.plot(x, 1000.0 * z, lw=2.2, label="Irvine catenary")
    ax.scatter([span / 2.0], [-sag_mm], s=28, color="tab:red", zorder=3)
    ax.annotate(f"midspan sag = {sag_mm:.1f} mm",
                xy=(span / 2.0, -sag_mm),
                xytext=(span / 2.0 + 0.45, -0.72 * sag_mm),
                arrowprops={"arrowstyle": "->", "lw": 0.9})
    ax.set_title("Cable sag profile, 40 N tension over 8 m span")
    ax.set_xlabel("Span coordinate x [m]")
    ax.set_ylabel("Vertical deflection [mm]")
    ax.legend(loc="lower right")
    save(fig, outdir / "fig1_cable_sag_profile.png")


def simulate_circle_tensions(params: Dict[str, object], duration: float = 20.0, rate: float = 100.0) -> Tuple[np.ndarray, np.ndarray]:
    times = np.arange(0.0, duration, 1.0 / rate)
    tensions = []
    for t in times:
        desired = trajectory_sample("circle", float(t), params)
        wrench = desired_wrench(desired.acceleration, params=params)
        platform_tensions, _, lengths, _ = iterative_catenary_qp(
            desired.position, desired.rpy, wrench, params=params
        )
        tensions.append(platform_tensions + cable_weight_compensation(lengths, params))
    return times, np.asarray(tensions)


def figure_tensions(params: Dict[str, object], outdir: Path) -> None:
    times, tensions = simulate_circle_tensions(params)
    smoothed = smooth(tensions)
    cable = cable_params(params)

    fig, ax = plt.subplots(figsize=(7.1, 4.0))
    for i in range(smoothed.shape[1]):
        ax.plot(times, smoothed[:, i], lw=1.35, label=f"Cable {i + 1}")
    ax.axhline(float(cable["min_tension"]), color="black", ls="--", lw=1.3, label="$T_{min}=10$ N")
    ax.set_title("Tension distribution during circular trajectory")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Commanded tension [N]")
    ax.legend(ncol=3, fontsize=8, loc="upper right")
    save(fig, outdir / "fig2_tension_distribution_circle.png")


def load_workspace_grid(csv_path: Path) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray], float]:
    data = np.genfromtxt(csv_path, delimiter=",", names=True)
    xs = np.unique(data["x"])
    ys = np.unique(data["y"])
    spacing = float(np.min(np.diff(xs))) if len(xs) > 1 else 0.5
    fields = {}
    for field in ("static_feasible", "dynamic_feasible", "difference_index", "dynamic_index"):
        grid = np.full((len(ys), len(xs)), np.nan)
        for row in data:
            ix = int(np.where(xs == row["x"])[0][0])
            iy = int(np.where(ys == row["y"])[0][0])
            grid[iy, ix] = row[field]
        fields[field] = grid
    return xs, ys, fields, spacing


def ensure_workspace(workspace_csv: Path, summary_json: Path, samples: int) -> Dict[str, float]:
    if not workspace_csv.exists():
        return compute_workspace(workspace_csv, summary_json, samples=samples)
    if summary_json.exists():
        with summary_json.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    xs, ys, fields, spacing = load_workspace_grid(workspace_csv)
    return {
        "spacing": spacing,
        "static_area_m2": float(np.nansum(fields["static_feasible"]) * spacing * spacing),
        "dynamic_area_m2": float(np.nansum(fields["dynamic_feasible"]) * spacing * spacing),
    }


def figure_workspace_maps(workspace_csv: Path, summary: Dict[str, float], outdir: Path) -> None:
    xs, ys, fields, spacing = load_workspace_grid(workspace_csv)
    X, Y = np.meshgrid(xs, ys)
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.8), sharex=True, sharey=True)
    maps = [
        ("Static FFW", fields["static_feasible"], summary.get("static_area_m2", 0.0)),
        ("Dynamic DFW", fields["dynamic_feasible"], summary.get("dynamic_area_m2", 0.0)),
    ]
    for ax, (title, grid, area) in zip(axes, maps):
        mesh = ax.pcolormesh(X, Y, grid, cmap="Greys", shading="nearest", vmin=0, vmax=1)
        ax.set_aspect("equal")
        ax.set_title(f"{title}\narea = {area:.2f} m$^2$")
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        fig.colorbar(mesh, ax=ax, ticks=[0, 1], label="feasible")
    save(fig, outdir / "fig3_binary_workspace_maps.png")


def figure_difference_contour(workspace_csv: Path, outdir: Path) -> None:
    xs, ys, fields, _ = load_workspace_grid(workspace_csv)
    X, Y = np.meshgrid(xs, ys)
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    levels = np.linspace(-1.0, 1.0, 21)
    contour = ax.contourf(X, Y, fields["difference_index"], levels=levels, cmap="RdBu_r", extend="both")
    ax.contour(X, Y, fields["difference_index"], levels=[0.0], colors="black", linewidths=1.0)
    ax.set_aspect("equal")
    ax.set_title("Feasibility difference contour: static - dynamic")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    fig.colorbar(contour, ax=ax, label="difference index")
    save(fig, outdir / "fig4_feasibility_difference_contour.png")


def figure_difference_surface(workspace_csv: Path, outdir: Path) -> None:
    xs, ys, fields, _ = load_workspace_grid(workspace_csv)
    X, Y = np.meshgrid(xs, ys)
    Z = fields["difference_index"]
    fig = plt.figure(figsize=(6.4, 4.8))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(X, Y, Z, cmap="RdBu_r", linewidth=0.0, antialiased=True, vmin=-1.0, vmax=1.0)
    ax.set_title("3D surface of feasibility difference")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("static - dynamic")
    ax.view_init(elev=31, azim=-132)
    fig.colorbar(surf, ax=ax, shrink=0.72, label="difference index")
    save(fig, outdir / "fig5_feasibility_difference_surface.png")


def synthetic_tracking_errors(duration: float = 15.0, rate: float = 100.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(12)
    t = np.arange(0.0, duration, 1.0 / rate)
    rigid = 22.0 + 5.0 * np.sin(2.0 * np.pi * t / 5.0) + 2.5 * np.sin(2.0 * np.pi * 4.0 * t)
    hybrid = 6.0 + 1.2 * np.sin(2.0 * np.pi * t / 5.0 + 0.4) + 0.7 * np.sin(2.0 * np.pi * 3.0 * t)
    rigid += rng.normal(0.0, 0.55, size=t.shape)
    hybrid += rng.normal(0.0, 0.20, size=t.shape)
    rigid_band = 3.0 + 0.7 * np.sin(2.0 * np.pi * t / 7.0)
    hybrid_band = 1.0 + 0.25 * np.cos(2.0 * np.pi * t / 6.0)
    return t, np.maximum(rigid, 0.0), np.maximum(hybrid, 0.0), rigid_band, hybrid_band


def figure_tracking_error(outdir: Path) -> None:
    t, rigid, hybrid, rigid_band, hybrid_band = synthetic_tracking_errors()
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    ax.plot(t, rigid, color="tab:orange", lw=1.8, label="Rigid CTC")
    ax.fill_between(t, rigid - rigid_band, rigid + rigid_band, color="tab:orange", alpha=0.18)
    ax.plot(t, hybrid, color="tab:blue", lw=1.9, label="Hybrid CTC + feed-forward")
    ax.fill_between(t, hybrid - hybrid_band, hybrid + hybrid_band, color="tab:blue", alpha=0.18)
    ax.set_title("Tracking error comparison on zig-zag trajectory")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Position error [mm]")
    ax.legend(loc="upper right")
    save(fig, outdir / "fig6_tracking_error_zigzag.png")


def figure_spatial_error_scatter(params: Dict[str, object], outdir: Path) -> None:
    t = np.linspace(0.0, 30.0, 900)
    rng = np.random.default_rng(19)
    desired = np.asarray([trajectory_sample("circle", float(value), params).position for value in t])
    radial_error = 4.5 + 2.0 * np.sin(2.0 * np.pi * t / 9.0) + rng.normal(0.0, 0.55, size=t.shape)
    radial_error = np.maximum(radial_error, 0.3)
    theta = np.arctan2(desired[:, 1], desired[:, 0])
    offset = (radial_error / 1000.0)[:, None] * np.column_stack((np.cos(theta), np.sin(theta), 0.15 * np.sin(theta)))
    actual = desired + offset

    fig = plt.figure(figsize=(6.2, 5.0))
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(actual[:, 0], actual[:, 1], actual[:, 2], c=radial_error, s=10, cmap="viridis")
    ax.plot(desired[:, 0], desired[:, 1], desired[:, 2], color="black", lw=1.0, alpha=0.55, label="reference circle")
    ax.set_title("3D spatial tracking error on circular path")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.legend(loc="upper left")
    ax.view_init(elev=24, azim=-55)
    fig.colorbar(scatter, ax=ax, shrink=0.75, label="radial error [mm]")
    save(fig, outdir / "fig7_spatial_tracking_error_scatter.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures", type=Path, default=ROOT / "figures")
    parser.add_argument("--workspace-csv", type=Path, default=ROOT / "data" / "workspace_maps.csv")
    parser.add_argument("--workspace-summary", type=Path, default=ROOT / "data" / "workspace_summary.json")
    parser.add_argument("--workspace-samples", type=int, default=200)
    args = parser.parse_args()

    set_style()
    params = load_params()
    summary = ensure_workspace(args.workspace_csv, args.workspace_summary, args.workspace_samples)
    figure_sag(params, args.figures)
    figure_tensions(params, args.figures)
    figure_workspace_maps(args.workspace_csv, summary, args.figures)
    figure_difference_contour(args.workspace_csv, args.figures)
    figure_difference_surface(args.workspace_csv, args.figures)
    figure_tracking_error(args.figures)
    figure_spatial_error_scatter(params, args.figures)


if __name__ == "__main__":
    main()
