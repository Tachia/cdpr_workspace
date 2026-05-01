#!/usr/bin/env python3
import rospy

from cdpr_control.ctc_node import ComputedTorqueController


def main():
    rospy.init_node("classical_ctc")
    ComputedTorqueController(model=0).spin()


if __name__ == "__main__":
    main()
