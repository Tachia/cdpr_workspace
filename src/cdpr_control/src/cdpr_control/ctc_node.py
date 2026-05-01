"""ROS controller node implementation used by classical_ctc.py and hybrid_ctc.py."""

from __future__ import annotations

import math
import time
from typing import Dict, Optional

import numpy as np
import rospy
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import PoseStamped, Vector3Stamped
from std_msgs.msg import Float32MultiArray
from tf.transformations import euler_from_quaternion, quaternion_from_euler

from cdpr_control.cdpr_model import (
    cable_params,
    cable_weight_compensation,
    desired_wrench,
    iterative_catenary_qp,
    load_params,
    solve_tensions_qp,
    structure_matrix,
)
from cdpr_control.trajectories import sample


class ComputedTorqueController:
    def __init__(self, model: int):
        self.model = int(model)
        self.params = load_params()
        self.params["cdpr"]["model_type"] = self.model  # type: ignore[index]

        # Overlay ROS parameters loaded by launch.
        if rospy.has_param("/cdpr"):
            self.params["cdpr"] = rospy.get_param("/cdpr")
        if rospy.has_param("/controller"):
            self.params["controller"] = rospy.get_param("/controller")

        self.cdpr = self.params["cdpr"]  # type: ignore[index]
        self.ctrl = self.params.get("controller", {})  # type: ignore[assignment]
        self.cable = cable_params(self.params)

        self.frequency = float(rospy.get_param("~frequency", self.ctrl.get("frequency", 200.0)))
        self.trajectory = str(rospy.get_param("~trajectory", self.ctrl.get("trajectory", "circle")))
        self.platform_name = str(rospy.get_param("~platform_name", self.cdpr["platform"].get("name", "cdpr_platform")))  # type: ignore[index]
        self.kp = float(rospy.get_param("~kp", self.ctrl.get("kp", 800.0)))
        self.kd = float(rospy.get_param("~kd", self.ctrl.get("kd", 40.0)))
        self.kp_rot = float(rospy.get_param("~kp_rot", self.ctrl.get("kp_rot", 80.0)))
        self.kd_rot = float(rospy.get_param("~kd_rot", self.ctrl.get("kd_rot", 8.0)))

        self.position: Optional[np.ndarray] = None
        self.rpy: Optional[np.ndarray] = None
        self.velocity: Optional[np.ndarray] = None
        self.angular_velocity: Optional[np.ndarray] = None
        self.start_time: Optional[rospy.Time] = None

        self.tension_pub = rospy.Publisher("/cable_tensions", Float32MultiArray, queue_size=1)
        self.desired_pub = rospy.Publisher("/desired_pose", PoseStamped, queue_size=1)
        self.error_pub = rospy.Publisher("/tracking_error", Vector3Stamped, queue_size=1)
        self.metrics_pub = rospy.Publisher("/controller_metrics", Float32MultiArray, queue_size=1)
        self.model_state_sub = rospy.Subscriber("/gazebo/model_states", ModelStates, self._on_model_states, queue_size=1)

    def _on_model_states(self, msg: ModelStates) -> None:
        try:
            idx = msg.name.index(self.platform_name)
        except ValueError:
            return

        pose = msg.pose[idx]
        twist = msg.twist[idx]
        self.position = np.array([pose.position.x, pose.position.y, pose.position.z], dtype=float)
        quat = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
        self.rpy = np.asarray(euler_from_quaternion(quat), dtype=float)
        self.velocity = np.array([twist.linear.x, twist.linear.y, twist.linear.z], dtype=float)
        self.angular_velocity = np.array([twist.angular.x, twist.angular.y, twist.angular.z], dtype=float)

    def _publish_reference(self, stamp: rospy.Time, desired) -> None:
        pose = PoseStamped()
        pose.header.stamp = stamp
        pose.header.frame_id = "world"
        pose.pose.position.x = float(desired.position[0])
        pose.pose.position.y = float(desired.position[1])
        pose.pose.position.z = float(desired.position[2])
        q = quaternion_from_euler(float(desired.rpy[0]), float(desired.rpy[1]), float(desired.rpy[2]))
        pose.pose.orientation.x = q[0]
        pose.pose.orientation.y = q[1]
        pose.pose.orientation.z = q[2]
        pose.pose.orientation.w = q[3]
        self.desired_pub.publish(pose)

    def _publish_error(self, stamp: rospy.Time, error: np.ndarray) -> None:
        msg = Vector3Stamped()
        msg.header.stamp = stamp
        msg.header.frame_id = "world"
        msg.vector.x = float(error[0])
        msg.vector.y = float(error[1])
        msg.vector.z = float(error[2])
        self.error_pub.publish(msg)

    def _publish_tensions(self, tensions: np.ndarray) -> None:
        msg = Float32MultiArray()
        msg.data = [float(value) for value in tensions]
        self.tension_pub.publish(msg)

    def _publish_metrics(self,
                         stamp: rospy.Time,
                         loop_ms: float,
                         solve_ms: float,
                         residual: float,
                         min_margin: float,
                         upper_margin: float,
                         saturation_count: int,
                         error_norm_mm: float) -> None:
        msg = Float32MultiArray()
        msg.layout.dim.resize(1)
        msg.layout.dim[0].label = "model,time,loop_ms,solve_ms,wrench_residual,min_margin,upper_margin,saturation_count,error_norm_mm"
        msg.layout.dim[0].size = 9
        msg.layout.dim[0].stride = 9
        msg.data = [
            float(self.model),
            float(stamp.to_sec()),
            float(loop_ms),
            float(solve_ms),
            float(residual),
            float(min_margin),
            float(upper_margin),
            float(saturation_count),
            float(error_norm_mm),
        ]
        self.metrics_pub.publish(msg)

    def _nominal_tensions(self) -> np.ndarray:
        return np.full(int(self.cable["count"]), float(self.cable["nominal_tension"]))

    def step(self) -> None:
        loop_start = time.perf_counter()
        stamp = rospy.Time.now()
        if self.start_time is None:
            self.start_time = stamp
        t = (stamp - self.start_time).to_sec()
        desired = sample(self.trajectory, t, self.params)
        self._publish_reference(stamp, desired)

        if self.position is None or self.rpy is None or self.velocity is None or self.angular_velocity is None:
            self._publish_tensions(self._nominal_tensions())
            self._publish_metrics(stamp, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0)
            return

        pos_error = desired.position - self.position
        vel_error = desired.velocity - self.velocity
        rot_error = desired.rpy - self.rpy
        # Keep the orientation error on the principal branch.
        rot_error = (rot_error + math.pi) % (2.0 * math.pi) - math.pi
        omega_error = desired.angular_velocity - self.angular_velocity

        force_pd = self.kp * pos_error + self.kd * vel_error
        torque_pd = self.kp_rot * rot_error + self.kd_rot * omega_error
        wrench = desired_wrench(desired.acceleration, force_pd, torque_pd, self.params)

        solve_start = time.perf_counter()
        if self.model == 1:
            platform_tensions, W, lengths, _ = iterative_catenary_qp(
                self.position, self.rpy, wrench, self.params
            )
            command = platform_tensions + cable_weight_compensation(lengths, self.params)
            residual_tensions = platform_tensions
        else:
            W, _, _ = structure_matrix(self.position, self.rpy, model=0, params=self.params)
            command = solve_tensions_qp(W, wrench, params=self.params)
            residual_tensions = command
        solve_ms = 1000.0 * (time.perf_counter() - solve_start)

        command = np.clip(command, 0.0, float(self.cable["max_tension"]))
        residual = float(np.linalg.norm(W @ residual_tensions - wrench))
        min_tension = float(self.cable["min_tension"])
        max_tension = float(self.cable["max_tension"])
        min_margin = float(np.min(command - min_tension))
        upper_margin = float(np.min(max_tension - command))
        saturation_count = int(np.count_nonzero((command <= min_tension + 1e-3) | (command >= max_tension - 1e-3)))
        error_norm_mm = 1000.0 * float(np.linalg.norm(pos_error))
        self._publish_tensions(command)
        self._publish_error(stamp, pos_error)
        loop_ms = 1000.0 * (time.perf_counter() - loop_start)
        self._publish_metrics(
            stamp,
            loop_ms,
            solve_ms,
            residual,
            min_margin,
            upper_margin,
            saturation_count,
            error_norm_mm,
        )

    def spin(self) -> None:
        rate = rospy.Rate(self.frequency)
        while not rospy.is_shutdown():
            self.step()
            rate.sleep()
