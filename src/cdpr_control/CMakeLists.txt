cmake_minimum_required(VERSION 3.0.2)
project(cdpr_control)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  geometry_msgs
  gazebo_msgs
)

catkin_python_setup()
catkin_package()

catkin_install_python(PROGRAMS
  scripts/classical_ctc.py
  scripts/hybrid_ctc.py
  scripts/constant_tension_experiment.py
  scripts/data_logger.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
