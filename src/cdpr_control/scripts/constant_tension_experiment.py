#!/usr/bin/env python3
"""Apply a constant tension vector for the static sag/pose experiment."""

import rospy
from std_msgs.msg import Float32MultiArray


def main():
    rospy.init_node("constant_tension_experiment")
    count = int(rospy.get_param("/cdpr/cable/count", 8))
    min_tension = float(rospy.get_param("/cdpr/cable/min_tension", 10.0))
    value = float(rospy.get_param("~tension", 40.0))
    cable_index = int(rospy.get_param("~cable_index", 0))
    rate_hz = float(rospy.get_param("~rate", 100.0))
    duration = float(rospy.get_param("~duration", 10.0))

    tensions = [min_tension] * count
    if 0 <= cable_index < count:
        tensions[cable_index] = value

    pub = rospy.Publisher("/cable_tensions", Float32MultiArray, queue_size=1)
    rate = rospy.Rate(rate_hz)
    end_time = rospy.Time.now() + rospy.Duration(duration)
    while not rospy.is_shutdown() and rospy.Time.now() < end_time:
        msg = Float32MultiArray()
        msg.data = tensions
        pub.publish(msg)
        rate.sleep()


if __name__ == "__main__":
    main()
