cmake_minimum_required(VERSION 3.0.2)
project(cdpr_postprocess)

find_package(catkin REQUIRED)
catkin_package()

catkin_install_python(PROGRAMS
  scripts/compute_workspace.py
  scripts/make_additional_figures.py
  scripts/make_figures.py
  scripts/rosbag_to_csv.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
