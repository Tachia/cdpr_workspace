#!/usr/bin/env python3
"""Record CDPR simulation topics to a rosbag."""

from __future__ import annotations

import os
import threading

import rosbag
import rospy
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import PoseStamped, Vector3Stamped
from std_msgs.msg import Float32MultiArray


class DataLogger:
    def __init__(self):
        default_bag = os.path.expanduser("~/cdpr_experiment.bag")
        self.bag_path = rospy.get_param("~bag_path", default_bag)
        directory = os.path.dirname(self.bag_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        self.bag = rosbag.Bag(self.bag_path, "w")
        self.lock = threading.Lock()
        self.subscribers = [
            rospy.Subscriber("/gazebo/model_states", ModelStates, self._write("/gazebo/model_states"), queue_size=20),
            rospy.Subscriber("/cable_tensions", Float32MultiArray, self._write("/cable_tensions"), queue_size=100),
            rospy.Subscriber("/actual_cable_tensions", Float32MultiArray, self._write("/actual_cable_tensions"), queue_size=100),
            rospy.Subscriber("/sag_directions", Float32MultiArray, self._write("/sag_directions"), queue_size=100),
            rospy.Subscriber("/desired_pose", PoseStamped, self._write("/desired_pose"), queue_size=100),
            rospy.Subscriber("/tracking_error", Vector3Stamped, self._write("/tracking_error"), queue_size=100),
            rospy.Subscriber("/controller_metrics", Float32MultiArray, self._write("/controller_metrics"), queue_size=100),
        ]
        rospy.on_shutdown(self.close)
        rospy.loginfo("Writing CDPR rosbag to %s", self.bag_path)

    def _stamp_for(self, msg):
        header = getattr(msg, "header", None)
        if header is not None and getattr(header, "stamp", None) is not None and header.stamp.to_sec() > 0:
            return header.stamp
        return rospy.Time.now()

    def _write(self, topic):
        def callback(msg):
            with self.lock:
                self.bag.write(topic, msg, self._stamp_for(msg))
        return callback

    def close(self):
        with self.lock:
            if self.bag is not None:
                rospy.loginfo("Closing CDPR rosbag %s", self.bag_path)
                self.bag.close()
                self.bag = None


def main():
    rospy.init_node("cdpr_data_logger")
    DataLogger()
    rospy.spin()


if __name__ == "__main__":
    main()
