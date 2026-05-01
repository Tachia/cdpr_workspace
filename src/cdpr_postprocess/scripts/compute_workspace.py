#!/usr/bin/env python3
"""Analytical FFW/DFW workspace computation using the CDPR model and QP allocator."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np


def _add_source_paths() -> Path:
    root = Path(__file__).resolve().parents[3]
    control_src = root / "src" / "cdpr_control" / "src"
    if str(control_src) not in sys.path:
        sys.path.insert(0, str(control_src))
    return root


ROOT = _add_source_paths()

from cdpr_control.cdpr_model import (  # noqa: E402
    cable_params,
    desired_wrench,
    feasibility_residual,
    iterative_catenary_qp,
    load_params,
    solve_tensions_qp,
    structure_matrix,
)


def acceleration_samples(count: int, a_max: float, rng: np.random.Generator) -> np.ndarray:
    samples = []
    while len(samples) < count:
        candidate = rng.uniform(-a_max, a_max, size=3)
        norm = np.linalg.norm(candidate)
        if norm <= a_max:
            samples.append(candidate)
    return np.asarray(samples, dtype=float)


def random_velocity_samples(count: int, v_max: float, rng: np.random.Generator) -> np.ndarray:
    samples = []
    while len(samples) < count:
        candidate = rng.uniform(-v_max, v_max, size=3)
        if np.linalg.norm(candidate) <= v_max:
            samples.append(candidate)
    return np.asarray(samples, dtype=float)


def test_feasible(position: Iterable[float],
                  acceleration: Iterable[float],
                  params: Dict[str, object],
                  model: int,
                  residual_tol: float) -> Tuple[bool, float]:
    rpy = np.zeros(3)
    wrench = desired_wrench(acceleration, params=params)
    cable = cable_params(params)
    lower = float(cable["min_tension"])
    upper = float(cable["max_tension"])
    if model == 1:
        tensions, W, _, _ = iterative_catenary_qp(position, rpy, wrench, params=params)
    else:
        W, _, _ = structure_matrix(position, rpy, model=0, params=params)
        tensions = solve_tensions_qp(W, wrench, params=params)
    residual = feasibility_residual(W, tensions, wrench)
    bounded = bool(np.all(tensions >= lower - 1e-6) and np.all(tensions <= upper + 1e-6))
    return bool(bounded and residual <= residual_tol), residual


def compute_workspace(output_csv: Path,
                      summary_json: Path,
                      spacing: float = 0.5,
                      z: float = 2.5,
                      samples: int = 200,
                      v_max: float = 0.5,
                      a_max: float = 1.0,
                      model: int = 1,
                      residual_tol: float = 2.5,
                      seed: int = 7) -> Dict[str, float]:
    params = load_params()
    rng = np.random.default_rng(seed)
    # Velocity samples are generated to match the DFW experiment definition; this
    # quasi-static wrench model only uses acceleration, as in the plugin/controller.
    _ = random_velocity_samples(samples, v_max, rng)
    accelerations = acceleration_samples(samples, a_max, rng)

    xs = np.arange(-5.5, 5.5 + 0.5 * spacing, spacing)
    ys = np.arange(-5.5, 5.5 + 0.5 * spacing, spacing)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    static_area = 0.0
    dynamic_area = 0.0
    cell_area = spacing * spacing

    total = len(xs) * len(ys)
    done = 0
    for y in ys:
        for x in xs:
            position = np.array([x, y, z], dtype=float)
            static_ok, static_residual = test_feasible(position, [0.0, 0.0, 0.0], params, model, residual_tol)
            feasible_count = 0
            worst_residual = static_residual
            if static_ok:
                for acceleration in accelerations:
                    ok, residual = test_feasible(position, acceleration, params, model, residual_tol)
                    feasible_count += int(ok)
                    worst_residual = max(worst_residual, residual)
            dynamic_score = feasible_count / float(samples)
            dynamic_ok = bool(static_ok and feasible_count == samples)
            static_area += cell_area * int(static_ok)
            dynamic_area += cell_area * int(dynamic_ok)
            rows.append({
                "x": x,
                "y": y,
                "z": z,
                "static_feasible": int(static_ok),
                "dynamic_feasible": int(dynamic_ok),
                "static_index": float(static_ok),
                "dynamic_index": dynamic_score,
                "difference_index": float(static_ok) - dynamic_score,
                "worst_residual": worst_residual,
            })
            done += 1
            if done % 50 == 0:
                print(f"workspace {done}/{total} cells", flush=True)

    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "spacing": spacing,
        "z": z,
        "samples": samples,
        "v_max": v_max,
        "a_max": a_max,
        "model": model,
        "residual_tol": residual_tol,
        "static_area_m2": static_area,
        "dynamic_area_m2": dynamic_area,
        "grid_cells": len(rows),
    }
    with summary_json.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "workspace_maps.csv")
    parser.add_argument("--summary", type=Path, default=ROOT / "data" / "workspace_summary.json")
    parser.add_argument("--spacing", type=float, default=0.5)
    parser.add_argument("--z", type=float, default=2.5)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--v-max", type=float, default=0.5)
    parser.add_argument("--a-max", type=float, default=1.0)
    parser.add_argument("--model", type=int, default=1, choices=[0, 1])
    parser.add_argument("--residual-tol", type=float, default=2.5)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    summary = compute_workspace(
        args.output,
        args.summary,
        spacing=args.spacing,
        z=args.z,
        samples=args.samples,
        v_max=args.v_max,
        a_max=args.a_max,
        model=args.model,
        residual_tol=args.residual_tol,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
