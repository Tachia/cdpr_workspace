"""Analytical model utilities shared by controllers and offline post-processing.

The Gazebo plugin mirrors the catenary endpoint equations implemented here:
for Model 0 each cable applies force along the straight segment; for Model 1
the cable tangent at the platform is corrected by an Irvine-style catenary
solution with horizontal tension obtained by scalar bracketing.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np

try:
    import yaml
except Exception:  # pragma: no cover - ROS installations normally have PyYAML.
    yaml = None

_CVXOPT_SOLVERS = None
_CVXOPT_MATRIX = None
_CVXOPT_MISSING = False


DEFAULT_PARAMS: Dict[str, object] = {
    "cdpr": {
        "model_type": 1,
        "rate_hz": 200.0,
        "platform": {
            "name": "cdpr_platform",
            "link_name": "platform_link",
            "mass": 50.0,
            "size": [1.0, 1.0, 1.0],
            "initial_pose": [0.0, 0.0, 3.0, 0.0, 0.0, 0.0],
        },
        "cable": {
            "count": 8,
            "min_tension": 10.0,
            "max_tension": 500.0,
            "nominal_tension": 80.0,
            "linear_density": 0.00739,
            "diameter": 0.0030,
            "young_modulus": 1.0e10,
            "area": 7.06858347e-6,
            "axial_stiffness": 7.06858347e4,
            "gravity": 9.80665,
            "catenary_iterations": 4,
        },
        "anchors": [
            [-6.0, -6.0, 6.0],
            [6.0, -6.0, 6.0],
            [6.0, 6.0, 6.0],
            [-6.0, 6.0, 6.0],
            [-6.0, -6.0, 0.0],
            [6.0, -6.0, 0.0],
            [6.0, 6.0, 0.0],
            [-6.0, 6.0, 0.0],
        ],
        "platform_attachments": [
            [-0.5, -0.5, 0.5],
            [0.5, -0.5, 0.5],
            [0.5, 0.5, 0.5],
            [-0.5, 0.5, 0.5],
            [-0.5, -0.5, -0.5],
            [0.5, -0.5, -0.5],
            [0.5, 0.5, -0.5],
            [-0.5, 0.5, -0.5],
        ],
    },
    "controller": {
        "frequency": 200.0,
        "kp": 800.0,
        "kd": 40.0,
        "kp_rot": 80.0,
        "kd_rot": 8.0,
        "trajectory": "circle",
        "circle_radius": 2.0,
        "circle_speed": 0.3,
        "zigzag_period": 5.0,
        "zigzag_amplitude": 2.0,
    },
}


@dataclass
class CatenaryEndpoint:
    unit: np.ndarray
    platform_tension: float
    horizontal_tension: float
    sag_mid: float
    length: float


def _deep_update(base: Dict[str, object], update: Dict[str, object]) -> Dict[str, object]:
    merged = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_update(merged[key], value)  # type: ignore[arg-type]
        else:
            merged[key] = value
    return merged


def default_config_path() -> Optional[Path]:
    try:
        import rospkg

        candidate = Path(rospkg.RosPack().get_path("cdpr_gazebo")) / "config" / "cdpr_params.yaml"
        if candidate.exists():
            return candidate
    except Exception:
        pass

    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "cdpr_gazebo" / "config" / "cdpr_params.yaml"
        if candidate.exists():
            return candidate
        candidate = parent.parent / "cdpr_gazebo" / "config" / "cdpr_params.yaml"
        if candidate.exists():
            return candidate
    env_path = os.environ.get("CDPR_CONFIG")
    return Path(env_path) if env_path else None


def load_params(path: Optional[str] = None) -> Dict[str, object]:
    params = DEFAULT_PARAMS
    config_path = Path(path) if path else default_config_path()
    if config_path and config_path.exists() and yaml is not None:
        with config_path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        params = _deep_update(DEFAULT_PARAMS, loaded)
    return params


def cdpr_params(params: Optional[Dict[str, object]] = None) -> Dict[str, object]:
    return (params or load_params())["cdpr"]  # type: ignore[index]


def cable_params(params: Optional[Dict[str, object]] = None) -> Dict[str, float]:
    return cdpr_params(params)["cable"]  # type: ignore[index,return-value]


def anchors(params: Optional[Dict[str, object]] = None) -> np.ndarray:
    return np.asarray(cdpr_params(params)["anchors"], dtype=float)  # type: ignore[index]


def attachments(params: Optional[Dict[str, object]] = None) -> np.ndarray:
    return np.asarray(cdpr_params(params)["platform_attachments"], dtype=float)  # type: ignore[index]


def rotation_matrix_rpy(rpy: Iterable[float]) -> np.ndarray:
    roll, pitch, yaw = [float(v) for v in rpy]
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    return rz @ ry @ rx


def skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = vector
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def world_attachment_points(position: Iterable[float],
                            rpy: Iterable[float] = (0.0, 0.0, 0.0),
                            params: Optional[Dict[str, object]] = None) -> Tuple[np.ndarray, np.ndarray]:
    position_np = np.asarray(position, dtype=float)
    rot = rotation_matrix_rpy(rpy)
    local = attachments(params)
    return position_np + local @ rot.T, local @ rot.T


def straight_unit(anchor: np.ndarray, point: np.ndarray) -> Tuple[np.ndarray, float]:
    delta = np.asarray(anchor, dtype=float) - np.asarray(point, dtype=float)
    length = float(np.linalg.norm(delta))
    if length < 1e-12:
        return np.zeros(3), 0.0
    return delta / length, length


def _endpoint_tension_for_h(horizontal_tension: float,
                            horizontal_span: float,
                            vertical_delta: float,
                            linear_weight: float) -> Tuple[float, float, float]:
    a = max(horizontal_tension / max(linear_weight, 1e-12), 1e-12)
    half_arg = horizontal_span / (2.0 * a)
    if abs(half_arg) > 50.0:
        return float("inf"), math.copysign(float("inf"), -vertical_delta if vertical_delta else -1.0), float("inf")
    sinh_half = math.sinh(half_arg)
    denom = max(2.0 * a * sinh_half, 1e-12)
    c = horizontal_span / 2.0 - a * math.asinh(vertical_delta / denom)
    slope0 = math.sinh(-c / a)
    z_mid = a * (math.cosh((horizontal_span / 2.0 - c) / a) - math.cosh((-c) / a))
    sag_mid = vertical_delta * 0.5 - z_mid
    return horizontal_tension * math.sqrt(1.0 + slope0 * slope0), slope0, sag_mid


def _solve_horizontal_tension(platform_tension: float,
                              horizontal_span: float,
                              vertical_delta: float,
                              linear_weight: float) -> float:
    if horizontal_span < 1e-9:
        return platform_tension

    def residual(value: float) -> float:
        tension, _, _ = _endpoint_tension_for_h(value, horizontal_span, vertical_delta, linear_weight)
        return tension - platform_tension

    lo = max(1e-3, platform_tension * 0.02)
    hi = max(1.0, platform_tension * 1.5)
    flo = residual(lo)
    fhi = residual(hi)

    for _ in range(30):
        if flo <= 0.0:
            break
        lo *= 0.5
        flo = residual(lo)
    for _ in range(30):
        if fhi >= 0.0:
            break
        hi *= 2.0
        fhi = residual(hi)

    if flo * fhi > 0.0:
        line_slope = vertical_delta / max(horizontal_span, 1e-12)
        return platform_tension / math.sqrt(1.0 + line_slope * line_slope)

    for _ in range(80):
        mid = 0.5 * (lo + hi)
        fmid = residual(mid)
        if fmid > 0.0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def catenary_endpoint(anchor: np.ndarray,
                      point: np.ndarray,
                      command_tension: float,
                      params: Optional[Dict[str, object]] = None,
                      command_is_winch_tension: bool = True) -> CatenaryEndpoint:
    cable = cable_params(params)
    rho = float(cable["linear_density"])
    gravity = float(cable["gravity"])
    max_tension = float(cable["max_tension"])
    delta = np.asarray(anchor, dtype=float) - np.asarray(point, dtype=float)
    length = float(np.linalg.norm(delta))
    if length < 1e-12:
        return CatenaryEndpoint(np.zeros(3), 0.0, 0.0, 0.0, 0.0)

    cable_weight = rho * gravity * length
    platform_tension = float(command_tension)
    if command_is_winch_tension:
        platform_tension -= 0.5 * cable_weight
    platform_tension = float(np.clip(platform_tension, 0.0, max_tension))

    horizontal = np.array([delta[0], delta[1], 0.0])
    horizontal_span = float(np.linalg.norm(horizontal))
    if platform_tension <= 1e-9 or horizontal_span < 1e-9:
        unit, _ = straight_unit(anchor, point)
        return CatenaryEndpoint(unit, platform_tension, platform_tension, 0.0, length)

    eh = horizontal / horizontal_span
    vertical_delta = float(delta[2])
    linear_weight = rho * gravity
    horizontal_tension = _solve_horizontal_tension(
        platform_tension, horizontal_span, vertical_delta, linear_weight
    )
    _, slope0, sag_mid = _endpoint_tension_for_h(
        horizontal_tension, horizontal_span, vertical_delta, linear_weight
    )
    tangent = np.array([eh[0], eh[1], slope0], dtype=float)
    tangent /= max(np.linalg.norm(tangent), 1e-12)
    return CatenaryEndpoint(tangent, platform_tension, horizontal_tension, sag_mid, length)


def structure_matrix(position: Iterable[float],
                     rpy: Iterable[float] = (0.0, 0.0, 0.0),
                     model: int = 0,
                     tensions: Optional[np.ndarray] = None,
                     params: Optional[Dict[str, object]] = None,
                     command_is_winch_tension: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    anchor_pts = anchors(params)
    world_pts, r_vectors = world_attachment_points(position, rpy, params)
    cable = cable_params(params)
    nominal = float(cable["nominal_tension"])
    tension_guess = np.full(anchor_pts.shape[0], nominal) if tensions is None else np.asarray(tensions, dtype=float)

    columns = []
    lengths = []
    units = []
    for i, (anchor_pt, point, r_vec) in enumerate(zip(anchor_pts, world_pts, r_vectors)):
      if model == 1:
          endpoint = catenary_endpoint(
              anchor_pt, point, float(tension_guess[i]), params, command_is_winch_tension=command_is_winch_tension
          )
          unit = endpoint.unit
          length = endpoint.length
      else:
          unit, length = straight_unit(anchor_pt, point)
      columns.append(np.hstack((unit, np.cross(r_vec, unit))))
      lengths.append(length)
      units.append(unit)
    return np.asarray(columns, dtype=float).T, np.asarray(lengths), np.asarray(units)


def desired_wrench(acceleration: Iterable[float],
                   force_correction: Optional[Iterable[float]] = None,
                   torque_correction: Optional[Iterable[float]] = None,
                   params: Optional[Dict[str, object]] = None) -> np.ndarray:
    cdpr = cdpr_params(params)
    cable = cable_params(params)
    mass = float(cdpr["platform"]["mass"])  # type: ignore[index]
    gravity = float(cable["gravity"])
    acc = np.asarray(acceleration, dtype=float)
    force = mass * (acc + np.array([0.0, 0.0, gravity]))
    if force_correction is not None:
        force = force + np.asarray(force_correction, dtype=float)
    torque = np.zeros(3)
    if torque_correction is not None:
        torque = torque + np.asarray(torque_correction, dtype=float)
    return np.hstack((force, torque))


def _solve_qp_cvxopt(W: np.ndarray,
                     wrench: np.ndarray,
                     lower: np.ndarray,
                     upper: np.ndarray,
                     nominal: np.ndarray) -> np.ndarray:
    global _CVXOPT_MATRIX, _CVXOPT_MISSING, _CVXOPT_SOLVERS
    if _CVXOPT_MISSING:
        raise ImportError("cvxopt is not available")
    if _CVXOPT_SOLVERS is None or _CVXOPT_MATRIX is None:
        try:
            from cvxopt import matrix, solvers
        except Exception:
            _CVXOPT_MISSING = True
            raise
        _CVXOPT_MATRIX = matrix
        _CVXOPT_SOLVERS = solvers

    n = W.shape[1]
    _CVXOPT_SOLVERS.options["show_progress"] = False
    P = _CVXOPT_MATRIX(np.eye(n), tc="d")
    q = _CVXOPT_MATRIX(-nominal, tc="d")
    G = _CVXOPT_MATRIX(np.vstack((np.eye(n), -np.eye(n))), tc="d")
    h = _CVXOPT_MATRIX(np.hstack((upper, -lower)), tc="d")
    A = _CVXOPT_MATRIX(W, tc="d")
    b = _CVXOPT_MATRIX(wrench, tc="d")
    solution = _CVXOPT_SOLVERS.qp(P, q, G, h, A, b)
    if solution["status"] != "optimal":
        raise RuntimeError(f"cvxopt QP status: {solution['status']}")
    return np.asarray(solution["x"], dtype=float).reshape(n)


def _solve_qp_fallback(W: np.ndarray,
                       wrench: np.ndarray,
                       lower: np.ndarray,
                       upper: np.ndarray,
                       nominal: np.ndarray) -> np.ndarray:
    # Fast exact allocator for the 8-cable wrench problem. The equality solution
    # is T = T0 + N z; at symmetric poses the structure matrix can drop to rank
    # five, so the bounded nullspace search handles up to three free variables.
    try:
        u, s, vt = np.linalg.svd(W, full_matrices=True)
        rank = int(np.sum(s > 1e-9))
        null_dim = W.shape[1] - rank
        if rank > 0 and null_dim <= 3:
            t0 = vt[:rank].T @ ((u[:, :rank].T @ wrench) / s[:rank])
            if np.linalg.norm(W @ t0 - wrench) > 1e-5:
                raise RuntimeError("wrench is outside the structure-matrix range")
            nullspace = vt[rank:].T
            if nullspace.shape[1] == 0:
                return np.clip(t0, lower, upper)

            A = np.vstack((nullspace, -nullspace))
            b = np.hstack((upper - t0, -(lower - t0)))
            candidates = []
            gram = nullspace.T @ nullspace
            gradient = nullspace.T @ (t0 - nominal)
            z_pref = np.linalg.lstsq(nullspace, nominal - t0, rcond=None)[0]
            candidates.append(z_pref)

            rows = A.shape[0]
            if nullspace.shape[1] == 1:
                lo = -np.inf
                hi = np.inf
                for ai, bi in zip(A[:, 0], b):
                    if abs(ai) < 1e-12:
                        continue
                    bound = bi / ai
                    if ai > 0:
                        hi = min(hi, bound)
                    else:
                        lo = max(lo, bound)
                if lo <= hi:
                    candidates.extend([np.array([lo]), np.array([hi]), np.array([np.clip(z_pref[0], lo, hi)])])
            else:
                import itertools

                # Boundary projections for one and two active bounds keep the
                # fallback close to the cvxopt objective instead of jumping to
                # a distant polytope vertex.
                max_active_projection = min(nullspace.shape[1] - 1, 2)
                for active_count in range(1, max_active_projection + 1):
                    for combo in itertools.combinations(range(rows), active_count):
                        active = A[list(combo), :]
                        rhs = b[list(combo)]
                        kkt = np.block([
                            [gram, active.T],
                            [active, np.zeros((active_count, active_count))],
                        ])
                        target = np.hstack((-gradient, rhs))
                        try:
                            candidates.append(np.linalg.solve(kkt, target)[:nullspace.shape[1]])
                        except np.linalg.LinAlgError:
                            candidates.append(np.linalg.lstsq(kkt, target, rcond=None)[0][:nullspace.shape[1]])

                for combo in itertools.combinations(range(rows), nullspace.shape[1]):
                    mat = A[list(combo), :]
                    if abs(np.linalg.det(mat)) < 1e-10:
                        continue
                    candidates.append(np.linalg.solve(mat, b[list(combo)]))

            feasible = []
            for z in candidates:
                if np.all(A @ z <= b + 1e-7):
                    tension = t0 + nullspace @ z
                    residual = np.linalg.norm(W @ tension - wrench)
                    if residual <= 1e-5:
                        feasible.append(tension)
            if feasible:
                return min(feasible, key=lambda value: float(np.linalg.norm(value - nominal)))
    except Exception:
        pass

    try:
        from scipy.optimize import lsq_linear

        result = lsq_linear(W, wrench, bounds=(lower, upper), lsmr_tol="auto", max_iter=100)
        if result.success:
            return np.asarray(result.x, dtype=float)
    except Exception:
        pass

    # Last-resort bounded least squares for environments without cvxopt/scipy.
    regularized = np.vstack((W, 1e-3 * np.eye(W.shape[1])))
    target = np.hstack((wrench, 1e-3 * nominal))
    solution = np.linalg.lstsq(regularized, target, rcond=None)[0]
    return np.clip(solution, lower, upper)


def solve_tensions_qp(W: np.ndarray,
                      wrench: np.ndarray,
                      params: Optional[Dict[str, object]] = None,
                      nominal: Optional[np.ndarray] = None,
                      allow_fallback: bool = True) -> np.ndarray:
    cable = cable_params(params)
    n = W.shape[1]
    lower = np.full(n, float(cable["min_tension"]))
    upper = np.full(n, float(cable["max_tension"]))
    if nominal is None:
        nominal = np.full(n, float(cable["nominal_tension"]))
    try:
        return _solve_qp_cvxopt(W, np.asarray(wrench, dtype=float), lower, upper, nominal)
    except Exception:
        if not allow_fallback:
            raise
        return _solve_qp_fallback(W, np.asarray(wrench, dtype=float), lower, upper, nominal)


def cable_weight_compensation(lengths: np.ndarray,
                              params: Optional[Dict[str, object]] = None) -> np.ndarray:
    cable = cable_params(params)
    return 0.5 * float(cable["linear_density"]) * float(cable["gravity"]) * np.asarray(lengths, dtype=float)


def iterative_catenary_qp(position: Iterable[float],
                          rpy: Iterable[float],
                          wrench: np.ndarray,
                          params: Optional[Dict[str, object]] = None,
                          iterations: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cable = cable_params(params)
    n = int(cable["count"])
    platform_tensions = np.full(n, float(cable["nominal_tension"]))
    if iterations is None:
        iterations = int(cable.get("catenary_iterations", 4))
    W = np.zeros((6, n))
    lengths = np.zeros(n)
    units = np.zeros((n, 3))
    for _ in range(max(1, iterations)):
        W, lengths, units = structure_matrix(
            position,
            rpy,
            model=1,
            tensions=platform_tensions,
            params=params,
            command_is_winch_tension=False,
        )
        platform_tensions = solve_tensions_qp(W, wrench, params=params, nominal=platform_tensions)
    return platform_tensions, W, lengths, units


def feasibility_residual(W: np.ndarray, tensions: np.ndarray, wrench: np.ndarray) -> float:
    return float(np.linalg.norm(W @ tensions - wrench))
