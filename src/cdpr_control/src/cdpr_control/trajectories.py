"""Reference trajectories used by the launch files and figure scripts."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass
class TrajectorySample:
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    rpy: np.ndarray
    angular_velocity: np.ndarray
    angular_acceleration: np.ndarray


def _controller_params(params: Dict[str, object]) -> Dict[str, object]:
    return params.get("controller", {})  # type: ignore[return-value]


def circle(t: float, params: Dict[str, object]) -> TrajectorySample:
    ctrl = _controller_params(params)
    radius = float(ctrl.get("circle_radius", 2.0))
    speed = float(ctrl.get("circle_speed", 0.3))
    omega = speed / max(radius, 1e-9)
    center = np.array([0.0, 0.0, 3.0])
    phase = omega * t
    position = center + np.array([radius * math.cos(phase), radius * math.sin(phase), 0.0])
    velocity = np.array([-radius * omega * math.sin(phase), radius * omega * math.cos(phase), 0.0])
    acceleration = np.array([-radius * omega * omega * math.cos(phase), -radius * omega * omega * math.sin(phase), 0.0])
    zeros = np.zeros(3)
    return TrajectorySample(position, velocity, acceleration, zeros.copy(), zeros.copy(), zeros.copy())


def zigzag(t: float, params: Dict[str, object]) -> TrajectorySample:
    ctrl = _controller_params(params)
    amplitude = float(ctrl.get("zigzag_amplitude", 2.0))
    period = float(ctrl.get("zigzag_period", 5.0))
    omega = 2.0 * math.pi / max(period, 1e-9)
    x = amplitude * math.sin(omega * t)
    vx = amplitude * omega * math.cos(omega * t)
    ax = -amplitude * omega * omega * math.sin(omega * t)

    # A smooth triangular-looking lateral sweep; it keeps acceleration finite.
    y = 0.65 * amplitude * math.sin(2.0 * omega * t + math.pi / 6.0)
    vy = 1.30 * amplitude * omega * math.cos(2.0 * omega * t + math.pi / 6.0)
    ay = -2.60 * amplitude * omega * omega * math.sin(2.0 * omega * t + math.pi / 6.0)

    position = np.array([x, y, 3.0])
    velocity = np.array([vx, vy, 0.0])
    acceleration = np.array([ax, ay, 0.0])
    zeros = np.zeros(3)
    return TrajectorySample(position, velocity, acceleration, zeros.copy(), zeros.copy(), zeros.copy())


def hold(t: float, params: Dict[str, object]) -> TrajectorySample:
    del t
    zeros = np.zeros(3)
    return TrajectorySample(np.array([0.0, 0.0, 3.0]), zeros.copy(), zeros.copy(), zeros.copy(), zeros.copy(), zeros.copy())


def sample(name: str, t: float, params: Dict[str, object]) -> TrajectorySample:
    normalized = (name or "circle").lower()
    if normalized == "circle":
        return circle(t, params)
    if normalized in {"zigzag", "zig-zag", "zig_zag"}:
        return zigzag(t, params)
    return hold(t, params)
